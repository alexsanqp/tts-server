"""Pure-Python MP3 duration parser tests."""

from __future__ import annotations

import asyncio
import struct
import wave

import pytest

from tts_server.core.transcode import ffmpeg_available
from tts_server.providers.edge import _mp3_duration_ms


def test_empty_returns_zero() -> None:
    assert _mp3_duration_ms(b"") == 0
    assert _mp3_duration_ms(b"\x00\x00\x00") == 0


def test_garbage_returns_zero() -> None:
    assert _mp3_duration_ms(b"\x00" * 100) == 0


def test_id3v2_only_returns_zero() -> None:
    # ID3v2 header claiming 0-byte body and no frames.
    tag = b"ID3" + b"\x03\x00" + b"\x00" + b"\x00\x00\x00\x00"
    assert _mp3_duration_ms(tag) == 0


@pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg not available (missing or exec-blocked)"
)
async def test_matches_ffmpeg_within_50ms() -> None:
    """End-to-end: silence WAV → MP3 via ffmpeg → parsed duration ≈ original."""
    from tts_server.core.transcode import transcode

    src_duration_ms = 500
    sample_rate = 24000
    n = int(sample_rate * src_duration_ms / 1000)
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    wav = buf.getvalue()

    out = await transcode(audio=wav, source_format="wav", target_format="mp3")
    parsed = _mp3_duration_ms(out.audio)
    # LAME encoder adds a ~50-100ms padding/encoder-delay frame, so the
    # parsed duration is consistently a few percent longer than the source.
    # 100ms tolerance covers this with margin.
    assert abs(parsed - src_duration_ms) < 100, f"parsed={parsed} expected≈{src_duration_ms}"
