"""Audio format transcoding via ffmpeg subprocess.

Used by the API layer when the caller's ``response_format`` differs from
the provider's native format. ffmpeg must be on PATH; if it's missing,
:func:`transcode` raises :class:`TranscoderUnavailable` and the API layer
returns 422 with a clear error so clients know to either ask for the
native format or install ffmpeg.

Kept deliberately small — no streaming, no chunking, in-memory only.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass

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

    # ffmpeg writing WAV to pipe:1 can't seek back to patch the RIFF/data
    # chunk sizes, so they're left as 0xFFFFFFFF placeholders, and an extra
    # LIST/INFO metadata chunk gets inserted between `fmt ` and `data`.
    # Strict parsers (e.g. Python's stdlib :mod:`wave`) then either reject
    # the file or read garbage. Repair the header in-place.
    if tgt == "wav":
        out = _normalize_wav(out)

    return TranscodeResult(audio=out, sample_rate=target_sample_rate or 0)


def _normalize_wav(data: bytes) -> bytes:
    """Repair a WAV produced by piped ffmpeg.

    Keeps only the ``fmt `` and ``data`` chunks (drops LIST, JUNK, INFO,
    etc.) and rewrites the RIFF + data chunk sizes from the actual byte
    counts. Returns ``data`` unchanged if it doesn't look like a WAV —
    silent no-op rather than corrupting unknown payloads.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data

    fmt_chunk = b""
    audio_payload = b""
    pos = 12
    n = len(data)
    while pos + 8 <= n:
        chunk_id = data[pos:pos + 4]
        chunk_size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body_start = pos + 8
        body_end = body_start + chunk_size
        if chunk_id == b"fmt ":
            # Keep the chunk header + its body verbatim.
            fmt_chunk = data[pos:body_end]
        elif chunk_id == b"data":
            # In pipe mode `chunk_size` may be the 0xFFFFFFFF placeholder,
            # so trust the actual file bytes instead.
            audio_payload = data[body_start:]
            break
        # else: skip non-essential chunks.
        pos = body_end
        # WAV chunks pad to 2-byte alignment.
        if pos & 1:
            pos += 1

    if not fmt_chunk or not audio_payload:
        return data  # malformed → don't touch

    data_chunk = b"data" + len(audio_payload).to_bytes(4, "little") + audio_payload
    body = b"WAVE" + fmt_chunk + data_chunk
    riff_size = len(body).to_bytes(4, "little")
    return b"RIFF" + riff_size + body
