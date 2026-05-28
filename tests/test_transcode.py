"""Audio transcoding tests (require ffmpeg on PATH)."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from tts_server.core.transcode import (
    TranscoderError,
    TranscoderUnavailable,
    _normalize_wav,
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


async def test_spawn_oserror_maps_to_unavailable(monkeypatch) -> None:
    """If the OS refuses to launch ffmpeg (e.g. Windows WDAC / WinError 4551),
    surface TranscoderUnavailable (→ 422) instead of a bare OSError/500."""
    async def _boom(*args, **kwargs):
        raise OSError(4551, "An Application Control policy has blocked this file")

    monkeypatch.setattr(
        "tts_server.core.transcode.asyncio.create_subprocess_exec", _boom
    )
    src = _silence_wav(duration_ms=50)
    with pytest.raises(TranscoderUnavailable):
        await transcode(audio=src, source_format="wav", target_format="mp3")


async def test_wav_output_has_clean_header_after_ffmpeg() -> None:
    """ffmpeg-via-pipe usually leaves bogus chunk sizes + LIST metadata.

    The transcoder normalises both so Python's stdlib wave module reads
    the file correctly: matching frame count and computed RIFF size.
    """
    src = _silence_wav(duration_ms=200, sample_rate=24000)
    # Re-encode through ffmpeg (mp3 → wav) so we get the broken path.
    mp3 = await transcode(audio=src, source_format="wav", target_format="mp3")
    out = await transcode(audio=mp3.audio, source_format="mp3", target_format="wav")

    # RIFF / data chunk sizes match the actual byte counts.
    assert out.audio[:4] == b"RIFF"
    riff_size = int.from_bytes(out.audio[4:8], "little")
    assert riff_size == len(out.audio) - 8, "RIFF size must equal file_size - 8"
    # The data chunk should appear immediately after the fmt chunk
    # (no LIST or INFO chunks in between).
    assert b"LIST" not in out.audio[:200]

    # And Python's strict parser must be happy with the result.
    with wave.open(io.BytesIO(out.audio), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        # Roughly 4800 frames @ 24 kHz for 200 ms; allow LAME tail padding.
        assert 3000 <= w.getnframes() <= 10000


def test_normalize_wav_strips_list_chunk_and_fixes_sizes() -> None:
    """Synthetic check: build a WAV with LIST in the middle + 0xFFFFFFFF
    sizes (exactly what ffmpeg pipe:1 produces) and verify the helper
    repairs it without subprocess help."""
    fmt_body = struct.pack(
        "<HHIIHH",
        1,        # PCM
        1,        # mono
        24000,    # sample rate
        48000,    # byte rate
        2,        # block align
        16,       # bits per sample
    )
    fmt_chunk = b"fmt " + (len(fmt_body)).to_bytes(4, "little") + fmt_body
    list_chunk = b"LIST" + (4).to_bytes(4, "little") + b"INFO"
    audio = b"\x00\x00" * 1200  # 1200 samples ≈ 50 ms @ 24 kHz
    data_chunk = b"data" + b"\xff\xff\xff\xff" + audio  # bogus size placeholder
    broken = b"RIFF" + b"\xff\xff\xff\xff" + b"WAVE" + fmt_chunk + list_chunk + data_chunk

    fixed = _normalize_wav(broken)

    assert fixed[:4] == b"RIFF"
    assert int.from_bytes(fixed[4:8], "little") == len(fixed) - 8
    assert b"LIST" not in fixed
    # Python wave module must parse it cleanly.
    with wave.open(io.BytesIO(fixed), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getnframes() == 1200


def test_normalize_wav_is_no_op_on_non_wav_bytes() -> None:
    """Defensive: don't corrupt unknown payloads (mp3, opus, …)."""
    blob = b"ID3\x04\x00\x00\x00\x00\x00\x00garbage"
    assert _normalize_wav(blob) == blob
