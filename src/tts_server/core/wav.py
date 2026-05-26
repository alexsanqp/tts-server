"""WAV byte helpers — duration probe + RIFF chunk normaliser.

Lives outside the provider layer because both providers and the ffmpeg
transcoder need these primitives. ``_wav_duration_ms`` used to be copy-
pasted into every provider that emitted WAV (qwen, styletts2-uk, fake);
``normalize_wav`` was wedged into ``core/transcode.py`` next to the
subprocess code it has nothing to do with.

Both helpers are pure — no imports beyond the stdlib, no I/O beyond
operating on ``bytes`` objects in memory.
"""

from __future__ import annotations

import io
import wave


def wav_duration_ms(audio: bytes, *, fallback_sample_rate: int = 24000) -> int:
    """Return duration in milliseconds for a WAV blob.

    Returns 0 when the bytes don't parse as WAV (so a caller can fall
    back to length × bytes-per-second math or just leave the
    ``X-Duration-Ms`` header at 0 rather than lying).
    ``fallback_sample_rate`` is only used when the file's own framerate
    is 0 (corrupt header) — most callers can ignore it.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            rate = w.getframerate() or fallback_sample_rate
            return int(round(w.getnframes() * 1000 / rate))
    except (wave.Error, EOFError):
        return 0


def normalize_wav(data: bytes) -> bytes:
    """Repair a WAV produced by piped ffmpeg.

    When ffmpeg writes WAV to ``pipe:1`` it can't seek back to patch
    the RIFF and data chunk sizes — both ship as 0xFFFFFFFF
    placeholders. It also inserts a LIST/INFO metadata chunk between
    ``fmt `` and ``data``, pushing the audio off the offset where
    strict parsers (Python's stdlib :mod:`wave`) expect it.

    This function walks the chunks in-memory, keeps just ``fmt `` and
    ``data``, and rewrites both sizes from the actual byte counts.
    Returns ``data`` unchanged when the input doesn't look like a WAV
    — defensive no-op, never corrupts unknown payloads.
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
            fmt_chunk = data[pos:body_end]
        elif chunk_id == b"data":
            # In pipe mode `chunk_size` may be the 0xFFFFFFFF placeholder,
            # so trust the actual file bytes instead.
            audio_payload = data[body_start:]
            break
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
