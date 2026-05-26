"""Qwen3-TTS sidecar worker — persistent subprocess to avoid model reload.

This module is **not** imported by the tts-server FastAPI app. It runs as a
separate Python process managed by `tts_server.providers.qwen.QwenProvider`.
Keeping it isolated means restarting the API server does NOT reload the
~1.5 GB Qwen model (cold start ~30-120 s).

Run as a standalone subprocess:
    python -m tts_server.sidecars.qwen_worker --port 8890 --preload \\
        --ref-audio-dir data/refs-catalog

API:
    POST /synthesize
    {
        "text": "Hello, how are you?",
        "language": "en",
        "ref_audio": "/path/to/ref.wav"   (optional, for voice cloning)
        "ref_text":  "spoken text in the ref clip"
    }
    -> 200 audio/wav binary, header X-Sample-Rate: 24000
    -> 400 {"error": "..."} for bad input
    -> 500 {"error": "synthesis_failed"} for internal failures
       (full traceback logged server-side; not echoed to caller)

    GET /health -> {"status": "ok", "model_loaded": true|false}
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tts_server.providers.qwen_constants import (  # noqa: E402 — kept after stdlib imports
    LANG_MAP,
    REF_TEXTS,
    UNSUPPORTED_LANGS,
)

logger = logging.getLogger(__name__)

_model = None
_ref_audio_cache: dict[str, str] = {}  # primary lang tag -> ref audio path


def _load_model():
    """Load Qwen3-TTS model (0.6B Base, ~1.5GB GPU RAM)."""
    global _model
    if _model is not None:
        return

    import torch
    from qwen_tts import Qwen3TTSModel

    model_name = os.environ.get(
        "QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    )
    logger.info("Loading Qwen3-TTS model: %s ...", model_name)

    _model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    logger.info("Qwen3-TTS model loaded successfully")


def synthesize_text(
    text: str,
    language: str = "en",
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> tuple[bytes, int]:
    """Synthesize text using Qwen3-TTS. Returns (wav_bytes, sample_rate).

    Raises ValueError for client-facing errors (unsupported language, missing
    ref-audio/ref-text). Other exceptions are internal failures.
    """
    _load_model()

    import soundfile as sf

    primary = language.split("-")[0].lower() if language else "en"
    if primary in UNSUPPORTED_LANGS:
        raise ValueError(
            f"language {primary!r} is not supported by Qwen3-TTS; "
            f"route to an alternate provider (edge-tts handles most)"
        )

    lang_full = LANG_MAP.get(primary)
    if lang_full is None:
        raise ValueError(
            f"unknown language {language!r}; supported: {sorted(LANG_MAP)}"
        )

    # If caller didn't pass ref_audio, try the catalog by primary tag.
    if ref_audio is None:
        ref_audio = _ref_audio_cache.get(primary)
        # Reference text falls back to the catalog's curated default — but
        # ONLY if the language matches the catalog entry. Pairing a German
        # text intent with an English ref_text (or vice-versa) produced
        # broken output in earlier versions; refuse instead.
        if ref_audio is not None and ref_text is None:
            ref_text = REF_TEXTS.get(primary)

    if ref_audio is None:
        raise ValueError(
            f"no reference audio for language {primary!r}; "
            f"either upload one via POST /v1/refs or pass `ref_audio` directly"
        )
    if ref_text is None:
        raise ValueError(
            f"`ref_text` is required for voice cloning and was not provided "
            f"(no curated default for {primary!r})"
        )

    wavs, sr = _model.generate_voice_clone(
        text=text,
        language=lang_full,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)
    return buf.read(), sr


class TTSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/synthesize":
            self._json_error(404, "not_found", "unknown path")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._json_error(400, "invalid_content_length", "Content-Length must be an integer")
            return
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_error(400, "invalid_json", "request body is not valid JSON")
            return

        text = data.get("text", "")
        language = data.get("language", "en")
        ref_audio = data.get("ref_audio")
        ref_text = data.get("ref_text")

        if not text:
            self._json_error(400, "missing_text", "request requires non-empty 'text' field")
            return

        try:
            audio_bytes, sr = synthesize_text(text, language, ref_audio, ref_text)
        except ValueError as e:
            # Client-facing input errors — message is safe to echo.
            self._json_error(400, "invalid_input", str(e))
            return
        except Exception as e:
            # Internal failure — log full detail server-side, hide from caller.
            logger.exception("Synthesis failed: %s", e)
            self._json_error(500, "synthesis_failed", "internal synthesis error")
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio_bytes)))
        self.send_header("X-Sample-Rate", str(sr))
        self.end_headers()
        self.wfile.write(audio_bytes)

    def do_GET(self):
        if self.path == "/health":
            payload = json.dumps(
                {"status": "ok", "model_loaded": _model is not None}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self._json_error(404, "not_found", "unknown path")

    def _json_error(self, status: int, code: str, message: str) -> None:
        payload = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):  # noqa: A003
        logger.info(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS sidecar worker")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--preload", action="store_true",
        help="Load model on startup",
    )
    parser.add_argument(
        "--ref-audio-dir",
        default="data/refs-catalog",
        help="Directory with reference audio files (en.wav, en.mp3, ...)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ref_dir = Path(args.ref_audio_dir)
    if ref_dir.exists():
        for lang_code in LANG_MAP:
            for ext in (".wav", ".mp3"):
                ref_path = ref_dir / f"{lang_code}{ext}"
                if ref_path.exists():
                    _ref_audio_cache[lang_code] = str(ref_path)
                    logger.info("Loaded ref audio: %s", ref_path)

    if args.preload:
        logger.info("Preloading Qwen3-TTS model...")
        _load_model()

    # ThreadingHTTPServer allows /health to respond during a slow synthesize().
    # The actual concurrency cap is enforced by the proxy's per-provider
    # asyncio.Semaphore — this is just to avoid head-of-line blocking on
    # health probes.
    server = ThreadingHTTPServer((args.host, args.port), TTSHandler)
    logger.info(
        "Qwen3-TTS sidecar on %s:%d (ref voices: %s)",
        args.host, args.port,
        list(_ref_audio_cache.keys()) or "none",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
