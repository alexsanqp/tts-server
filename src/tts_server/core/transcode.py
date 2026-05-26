"""Audio format transcoding via ffmpeg subprocess.

Used by the API layer when the caller's ``response_format`` differs from
the provider's native format. ffmpeg must be on PATH; if it's missing,
:func:`transcode` raises :class:`TranscoderUnavailable` and the API layer
returns 422 with a clear error so clients know to either ask for the
native format or install ffmpeg.

Kept deliberately small — no streaming, no chunking, in-memory only.
The WAV header repair (after ffmpeg) lives in :mod:`tts_server.core.wav`.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass

from tts_server.core.wav import normalize_wav

logger = logging.getLogger(__name__)

# Native input formats we know how to feed into ffmpeg. Output formats we
# know how to emit are independent — both sides use ffmpeg muxer/demuxer
# names that match the file extension.
_KNOWN_FORMATS = frozenset({"wav", "mp3", "ogg", "opus", "flac"})


@dataclass(frozen=True)
class TranscodeResult:
    audio: bytes
    sample_rate: int


class TranscoderError(Exception):
    """Raised when ffmpeg returns non-zero or output is empty."""


class TranscoderUnavailable(Exception):
    """ffmpeg is not on PATH."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def transcode(
    *,
    audio: bytes,
    source_format: str,
    target_format: str,
    target_sample_rate: int | None = None,
    timeout: float = 30.0,
) -> TranscodeResult:
    """Convert audio bytes between formats. Pass-through when formats match.

    ``target_sample_rate`` resamples; pass None to preserve source rate.
    """
    src = source_format.lower()
    tgt = target_format.lower()

    if src not in _KNOWN_FORMATS:
        raise TranscoderError(f"unknown source format: {source_format!r}")
    if tgt not in _KNOWN_FORMATS:
        raise TranscoderError(f"unknown target format: {target_format!r}")

    # No-op pass-through when format AND rate match.
    if src == tgt and target_sample_rate is None:
        return TranscodeResult(audio=audio, sample_rate=0)

    if not ffmpeg_available():
        raise TranscoderUnavailable(
            "ffmpeg is required for response_format transcoding but is not on PATH"
        )

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", src,
        "-i", "pipe:0",
    ]
    if target_sample_rate:
        args += ["-ar", str(target_sample_rate)]
    # Force mono — TTS output is consistently mono and avoids surprises.
    args += ["-ac", "1"]
    # Output codec; ffmpeg picks sensible defaults per container.
    args += ["-f", tgt, "pipe:1"]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(audio), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TranscoderError(f"ffmpeg timed out after {timeout}s") from None

    if proc.returncode != 0:
        msg = (err or b"").decode("utf-8", errors="replace")[:300]
        raise TranscoderError(f"ffmpeg exit {proc.returncode}: {msg}")
    if not out:
        raise TranscoderError("ffmpeg produced empty output")

    # ffmpeg writing WAV to pipe:1 leaves 0xFFFFFFFF size placeholders and
    # extra metadata chunks; :func:`normalize_wav` walks the bytes and
    # rebuilds a clean header strict parsers (Python's `wave`) accept.
    if tgt == "wav":
        out = normalize_wav(out)

    return TranscodeResult(audio=out, sample_rate=target_sample_rate or 0)


# Back-compat alias: existing tests import ``_normalize_wav`` from this module.
_normalize_wav = normalize_wav
