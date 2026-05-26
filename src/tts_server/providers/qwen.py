"""QwenProvider — in-process proxy to the Qwen3-TTS sidecar subprocess.

Rationale: loading the Qwen3-TTS 0.6B model takes ~30-120 s (~1.5 GB GPU
RAM). We keep it in a separate process (`tts_server.sidecars.qwen_worker`)
so that restarting the API server does **not** reload the model.

This module:
    1. Manages the subprocess lifetime (Popen in `load()`, terminate in
       `teardown()` and during garbage collection).
    2. Forwards `synthesize()` calls to the sidecar over httpx.
    3. Scans the ref-audio catalog to populate `describe()` voices
       WITHOUT starting the worker (cheap inventory for /v1/models).

`describe()` must remain cheap and avoid spawning the subprocess.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from tts_server.core.errors import ProviderFailure, ProviderUnavailable
from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    SynthesisStream,
    VoiceInfo,
)

logger = logging.getLogger(__name__)


# Legacy fallback texts for plain-stem ref clips (e.g. en.mp3 → ref:en-default).
# New deployments use <lang>-<name>.{mp3,wav} + <lang>-<name>.json sidecars
# (see :meth:`QwenProvider._scan_voices`). Kept here so existing single-file
# catalogs without a sidecar still expose a usable ref_text.
REF_TEXTS: dict[str, str] = {
    "en": (
        "Hello, my name is your English teacher. "
        "Today we will learn new vocabulary words together."
    ),
}

# Qwen3-TTS supported BCP-47 primary tags (mirrored from worker LANG_MAP).
QWEN_LANGUAGES: tuple[str, ...] = (
    "en", "de", "fr", "it", "es", "ru", "ja", "ko", "zh", "pt",
)

_REF_EXTS = (".wav", ".mp3")

# Stem must be either a primary BCP-47 tag ("en", "uk") OR
# <lang>-<voice-name> ("en-owen", "ja-akari", "de-anna_v2").
# Anything else gets logged + skipped so accidental files in the
# catalog don't pollute /v1/voices.
_STEM_RE = re.compile(r"^([a-z]{2,3})(?:-([a-z0-9][a-z0-9_-]{0,30}))?$")


class QwenProvider:
    """Proxy to the Qwen3-TTS sidecar subprocess."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        opts = options or {}
        self._port: int = int(opts.get("port", 8890))
        self._host: str = str(opts.get("host", "127.0.0.1"))
        self._ref_audio_dir: Path = Path(
            opts.get("ref_audio_dir", "data/refs-catalog")
        )
        self._model_name: str = str(
            opts.get("model_name", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        )
        self._device: str = str(opts.get("device", "cuda:0"))
        self._startup_timeout: float = float(
            opts.get("startup_timeout_seconds", 180.0)
        )
        self._request_timeout: float = float(
            opts.get("request_timeout", 120.0)
        )

        # Path to capture sidecar stdout/stderr (CREATE_NEW_PROCESS_GROUP on
        # Windows detaches stdio, so we always log to a file too).
        log_dir = Path(opts.get("log_dir", "data"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path: Path = log_dir / "qwen_worker.log"

        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None
        # Single lock prevents two concurrent load() calls racing the subprocess.
        self._load_lock = asyncio.Lock()

    # ---- describe / probe (no subprocess) ----

    async def describe(self) -> ProviderCapabilities:
        voices = self._scan_voices()
        return ProviderCapabilities(
            id="qwen3-0.6b",
            provider_family="qwen",
            languages=QWEN_LANGUAGES,
            voices=voices,
            supports_voice_id=False,
            supports_voice_cloning=True,
            native_sample_rate=24000,
            native_format="wav",
            max_text_length=1000,
            accepts_speed=False,
            is_gpu=True,
            is_remote=True,
        )

    async def probe_voice(self, voice_id: str) -> bool:
        """Catalog-membership check; never talks to the worker."""
        return any(v.id == voice_id for v in self._scan_voices())

    def _scan_voices(self) -> tuple[VoiceInfo, ...]:
        """Inventory ``<ref_audio_dir>/*.{wav,mp3}`` restricted to Qwen languages.

        Two stem patterns are recognised:

        * ``<lang>`` (e.g. ``en.mp3``)            → id ``ref:<lang>-default``
        * ``<lang>-<name>`` (e.g. ``en-owen.mp3``) → id ``ref:<lang>-<name>``

        For each audio file we look for a sibling ``<stem>.json`` sidecar:

        .. code-block:: json

           {
             "ref_text": "Exact transcript of the audio.",
             "gender": "male",
             "description": "Primary host voice.",
             "role": "host"
           }

        Without a sidecar, the legacy :data:`REF_TEXTS` dict (keyed by primary
        language) supplies a fallback ``ref_text`` for plain-stem files. Stems
        whose primary tag is not in :data:`QWEN_LANGUAGES` are skipped because
        the worker would reject synthesis with HTTP 400 — silent-but-broken
        catalog entries are worse than missing ones.
        """
        ref_dir = self._ref_audio_dir
        if not ref_dir.exists() or not ref_dir.is_dir():
            return ()

        # Deduplicate by stem (en.mp3 + en.wav → only one).
        seen: dict[str, Path] = {}
        for ext in _REF_EXTS:
            for path in sorted(ref_dir.glob(f"*{ext}")):
                stem = path.stem
                if stem not in seen:
                    seen[stem] = path

        voices: list[VoiceInfo] = []
        for stem, path in sorted(seen.items()):
            stem_lc = stem.lower()
            match = _STEM_RE.match(stem_lc)
            if match is None:
                logger.warning(
                    "Skipping %s: stem %r doesn't match '<lang>' or '<lang>-<name>'",
                    path, stem,
                )
                continue

            primary, voice_name = match.group(1), match.group(2)
            if primary not in QWEN_LANGUAGES:
                continue  # Qwen can't synthesize this language.

            voice_id = f"ref:{stem_lc}" if voice_name else f"ref:{primary}-default"
            metadata, gender = _load_sidecar(path.with_suffix(".json"))

            # Legacy fallback: hardcoded REF_TEXTS for plain-stem files only.
            if voice_name is None and "ref_text" not in metadata:
                if ref_text := REF_TEXTS.get(primary):
                    metadata["ref_text"] = ref_text

            voices.append(
                VoiceInfo(
                    id=voice_id,
                    languages=(primary,),
                    gender=gender,
                    accepts_voice_id=False,
                    accepts_clone_ref=True,
                    metadata=metadata,
                )
            )
        return tuple(voices)

    # ---- lifecycle ----

    async def load(self) -> None:
        """Idempotent: start the sidecar and wait until /health reports model_loaded."""
        async with self._load_lock:
            if await self._is_healthy():
                # Either an existing process we manage or someone else's
                # sidecar already serving on this port; either way we're good.
                return

            if self._proc is not None and self._proc.poll() is not None:
                # Stale handle from a crashed earlier attempt.
                self._proc = None

            if self._proc is None:
                self._spawn_subprocess()

            await self._wait_until_ready()

    def _spawn_subprocess(self) -> None:
        ref_dir = self._ref_audio_dir
        cmd = [
            sys.executable,
            "-m",
            "tts_server.sidecars.qwen_worker",
            "--port", str(self._port),
            "--host", self._host,
            "--preload",
            "--ref-audio-dir", str(ref_dir),
        ]
        env = os.environ.copy()
        env["QWEN_MODEL"] = self._model_name
        # Qwen worker hard-codes device_map="cuda:0"; we steer it via
        # CUDA_VISIBLE_DEVICES so the operator can pick a different GPU.
        device = self._device
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        else:
            env["CUDA_VISIBLE_DEVICES"] = device

        logger.info(
            "Spawning Qwen sidecar: %s (model=%s, device=%s, log=%s)",
            " ".join(cmd), self._model_name, self._device, self._log_path,
        )
        # On Windows, CUDA init from a subprocess that inherits the uvicorn
        # parent's handles/job-object can hit access violations. CREATE_NEW_PROCESS_GROUP
        # detaches the child so it gets a clean CUDA context. Since that also
        # disconnects stdio, redirect both streams to a file.
        self._log_fh = self._log_path.open("ab")
        popen_kwargs: dict[str, Any] = {
            "env": env,
            "stdout": self._log_fh,
            "stderr": subprocess.STDOUT,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        try:
            self._proc = subprocess.Popen(cmd, **popen_kwargs)
        except OSError as exc:
            self._log_fh.close()
            self._log_fh = None
            raise ProviderUnavailable(
                f"Failed to launch Qwen sidecar: {exc}"
            ) from exc

    async def _wait_until_ready(self) -> None:
        """Poll /health every 2 s until model_loaded=True or timeout."""
        deadline = asyncio.get_event_loop().time() + self._startup_timeout
        last_error: str = "no response"

        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                # Sidecar process died before becoming healthy?
                if self._proc is not None:
                    rc = self._proc.poll()
                    if rc is not None:
                        self._proc = None
                        raise ProviderUnavailable(
                            f"Qwen sidecar exited during startup with code {rc}"
                        )

                try:
                    resp = await client.get(self._health_url)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if resp.status_code == 200:
                        try:
                            body = resp.json()
                        except ValueError:
                            body = {}
                        if body.get("model_loaded") is True:
                            logger.info(
                                "Qwen sidecar ready on %s:%d",
                                self._host, self._port,
                            )
                            return
                        last_error = f"health ok but model_loaded={body.get('model_loaded')!r}"
                    else:
                        last_error = f"HTTP {resp.status_code}"

                if asyncio.get_event_loop().time() >= deadline:
                    # Best-effort cleanup of the still-loading subprocess.
                    await self._terminate_proc()
                    raise ProviderUnavailable(
                        f"Qwen sidecar did not become healthy within "
                        f"{self._startup_timeout:.0f}s (last: {last_error})"
                    )

                await asyncio.sleep(2)

    async def _is_healthy(self) -> bool:
        """Single quick health probe — does not retry."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(self._health_url)
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        try:
            body = resp.json()
        except ValueError:
            return False
        return body.get("model_loaded") is True

    async def teardown(self) -> None:
        """Async cleanup hook invoked by the registry on shutdown."""
        await self._terminate_proc()

    # Alias to match constraint #7 wording ("Add a terminate() async method").
    terminate = teardown

    async def _terminate_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.poll() is not None:
            return  # already exited

        loop = asyncio.get_event_loop()
        try:
            proc.terminate()
        except OSError as exc:
            logger.warning("Failed to terminate Qwen sidecar: %s", exc)
            return

        try:
            await loop.run_in_executor(None, lambda: proc.wait(timeout=10.0))
        except subprocess.TimeoutExpired:
            logger.warning("Qwen sidecar did not exit in 10s; killing")
            try:
                proc.kill()
            except OSError as exc:
                logger.warning("Failed to kill Qwen sidecar: %s", exc)
                return
            try:
                await loop.run_in_executor(None, lambda: proc.wait(timeout=5.0))
            except subprocess.TimeoutExpired:
                logger.error("Qwen sidecar refused to die after SIGKILL")

    def __del__(self) -> None:  # best-effort GC cleanup
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            # __del__ must never raise.
            pass

    # ---- synthesize ----

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        # API layer hands us a fully resolved ref clip when voice_kind=="clone_ref".
        ref_audio: str | None = None
        if request.voice_kind == "clone_ref" and request.voice:
            ref_audio = request.voice

        payload: dict[str, Any] = {
            "text": request.text,
            "language": request.language or "auto",
        }
        if ref_audio is not None:
            payload["ref_audio"] = ref_audio
        if request.ref_text is not None:
            payload["ref_text"] = request.ref_text

        url = f"http://{self._host}:{self._port}/synthesize"
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"Qwen sidecar request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if resp.status_code != 200:
            body_preview = resp.text[:500] if resp.text else "<empty>"
            raise ProviderFailure(
                f"Qwen sidecar HTTP {resp.status_code}: {body_preview}"
            )

        audio_bytes = resp.content
        sample_rate = _parse_sample_rate(resp.headers, audio_bytes)
        duration_ms = _wav_duration_ms(audio_bytes)

        async def _one_chunk() -> AsyncIterator[bytes]:
            yield audio_bytes

        return SynthesisStream(
            sample_rate=sample_rate,
            format="wav",
            duration_ms=duration_ms,
            chunks=_one_chunk(),
        )

    # ---- helpers ----

    @property
    def _health_url(self) -> str:
        return f"http://{self._host}:{self._port}/health"


def _parse_sample_rate(headers: httpx.Headers, audio: bytes) -> int:
    """Prefer the worker's X-Sample-Rate header; fall back to parsing the WAV."""
    hdr = headers.get("x-sample-rate") if headers else None
    if hdr:
        try:
            return int(hdr)
        except (TypeError, ValueError):
            pass
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf:
            return wf.getframerate()
    except (wave.Error, EOFError):
        return 24000


def _wav_duration_ms(audio: bytes) -> int:
    """Compute duration from WAV header. Returns 0 if not parseable."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return int(round(frames * 1000 / rate))
    except (wave.Error, EOFError):
        return 0


_ALLOWED_SIDECAR_KEYS = ("description", "role")


def _load_sidecar(sidecar: Path) -> tuple[dict[str, str], str | None]:
    """Parse the JSON sidecar next to a ref-audio file.

    Returns ``(metadata, gender)``. ``metadata`` is populated from any of
    ``ref_text``, ``description``, ``role`` (when present and non-empty).
    Malformed JSON or unreadable file is logged and treated as missing —
    a bad sidecar must not crash provider startup.
    """
    metadata: dict[str, str] = {}
    if not sidecar.is_file():
        return metadata, None

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Ignoring malformed ref sidecar %s: %s", sidecar, exc)
        return metadata, None

    if not isinstance(data, dict):
        logger.warning("Sidecar %s is not a JSON object; ignoring", sidecar)
        return metadata, None

    if ref_text := data.get("ref_text"):
        if isinstance(ref_text, str) and ref_text.strip():
            metadata["ref_text"] = ref_text
    for key in _ALLOWED_SIDECAR_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            metadata[key] = value

    gender = data.get("gender")
    if not isinstance(gender, str) or not gender.strip():
        gender = None
    return metadata, gender
