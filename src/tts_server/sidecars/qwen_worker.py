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
        "ref_audio": "/path/to/ref.wav"  (optional, for voice cloning)
        "ref_text":  "spoken text in the ref clip"
    }
    -> returns audio/wav binary, header X-Sample-Rate: 24000

    GET /health -> {"status": "ok", "model_loaded": true|false}
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_ref_audio_cache: dict[str, str] = {}  # lang -> ref audio path


LANG_MAP = {
    "en": "English",
    # "uk" not supported by Qwen — use edge-tts (uk-UA-PolinaNeural)
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "pt": "Portuguese",
    "it": "Italian",
}

# Languages NOT supported by Qwen3-TTS (use edge-tts fallback)
UNSUPPORTED_LANGS = {"uk", "pl", "ar", "hi", "tr"}

# Reference texts for generating default voice samples
REF_TEXTS = {
    "en": (
        "Hello, my name is your English teacher. "
        "Today we will learn new vocabulary words together."
    ),
    "uk": (
        "Привіт, мене звати ваша вчителька. "
        "Сьогодні ми вивчимо нові слова разом."
    ),
}


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


def _get_ref_audio(language: str) -> tuple[str, str] | tuple[None, None]:
    """Get or create reference audio for voice cloning.

    Returns (ref_audio_path, ref_text) or (None, None).
    """
    if language in _ref_audio_cache:
        ref_text = REF_TEXTS.get(language, REF_TEXTS["en"])
        return _ref_audio_cache[language], ref_text

    # No cached ref audio — caller should use without cloning
    return None, None


def synthesize_text(
    text: str,
    language: str = "en",
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> tuple[bytes, int]:
    """Synthesize text using Qwen3-TTS.

    Returns (wav_bytes, sample_rate).
    """
    _load_model()

    import soundfile as sf

    lang_full = LANG_MAP.get(language, language)

    # Use provided ref_audio, or try cached
    if ref_audio is None:
        ref_audio, ref_text = _get_ref_audio(language)

    if ref_audio is None:
        # Fallback: try English ref audio for any language
        ref_audio, ref_text = _get_ref_audio("en")

    if ref_audio is None:
        raise ValueError(
            f"No reference audio for language '{language}'. "
            "Place en.mp3/uk.mp3 in --ref-audio-dir."
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
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        text = data.get("text", "")
        language = data.get("language", "en")
        ref_audio = data.get("ref_audio")
        ref_text = data.get("ref_text")

        if not text:
            self.send_error(400, "Missing 'text' field")
            return

        try:
            audio_bytes, sr = synthesize_text(
                text, language, ref_audio, ref_text,
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.send_header("X-Sample-Rate", str(sr))
            self.end_headers()
            self.wfile.write(audio_bytes)
        except Exception as e:
            logger.error("Synthesis error: %s", e)
            error_msg = str(e).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "ok",
                "model_loaded": _model is not None,
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_error(404, "Not Found")

    def log_message(self, fmt, *args):
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
        help="Directory with reference audio files (en.wav, uk.wav, ...)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load reference audio files if available
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

    server = HTTPServer((args.host, args.port), TTSHandler)
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
