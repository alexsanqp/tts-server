"""Audio transcoding tests (require ffmpeg on PATH)."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from tts_server.core.transcode import (
    TranscoderError,
    TranscoderUnavailable,
    ffmpeg_available,
    transcode,
)


def _silence_wav(duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return buf.getvalue()


pytestmark = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg not on PATH"
)


async def test_passthrough_when_formats_match() -> None:
    src = _silence_wav()
    out = await transcode(audio=src, source_format="wav", target_format="wav")
    assert out.audio == src


async def test_wav_to_mp3_returns_mp3_bytes() -> None:
    src = _silence_wav(duration_ms=200)
    out = await transcode(audio=src, source_format="wav", target_format="mp3")
    assert out.audio
    # MP3 magic: ID3 tag, raw frame, or a sync word.
    assert out.audio[:3] == b"ID3" or (out.audio[0] == 0xFF and (out.audio[1] & 0xE0) == 0xE0)


async def test_resample_changes_byte_size() -> None:
    src = _silence_wav(sample_rate=24000, duration_ms=200)
    out = await transcode(
        audio=src, source_format="wav", target_format="wav", target_sample_rate=16000
    )
    assert out.audio[:4] == b"RIFF"
    # 16k mono 16-bit ≈ 6400 bytes (200 ms) + header; well under the 24k version.
    assert len(out.audio) < len(src)


async def test_invalid_format_raises() -> None:
    with pytest.raises(TranscoderError):
        await transcode(audio=b"x", source_format="nope", target_format="wav")
