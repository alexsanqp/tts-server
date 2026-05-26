"""FakeProvider — minimal reference implementation used in tests.

Generates a tiny valid WAV (200ms of silence at the requested sample rate)
without needing torch/transformers. Useful for exercising the HTTP layer.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import AsyncIterator
from typing import Any

from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    SynthesisStream,
    VoiceInfo,
)


class FakeProvider:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self._options = options or {}
        self._default_voice = self._options.get("default_voice", "fake-en")

    async def describe(self) -> ProviderCapabilities:
        voices = (
            VoiceInfo(id="fake-en", languages=("en",), gender="neutral"),
            VoiceInfo(id="fake-uk", languages=("uk",), gender="neutral"),
        )
        return ProviderCapabilities(
            id="fake",
            provider_family="fake",
            languages=("en", "uk"),
            voices=voices,
            supports_voice_id=True,
            supports_voice_cloning=False,
            native_sample_rate=24000,
            native_format="wav",
            max_text_length=10_000,
            accepts_speed=True,
            is_gpu=False,
            is_remote=False,
        )

    async def load(self) -> None:
        return  # nothing to warm

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        sample_rate = request.target_sample_rate or 24000
        duration_ms = 200
        wav_bytes = _make_silence_wav(sample_rate=sample_rate, duration_ms=duration_ms)

        async def _one_chunk() -> AsyncIterator[bytes]:
            yield wav_bytes

        return SynthesisStream(
            sample_rate=sample_rate,
            format="wav",
            duration_ms=duration_ms,
            chunks=_one_chunk(),
        )

    async def probe_voice(self, voice_id: str) -> bool:
        return voice_id in {"fake-en", "fake-uk"}


def _make_silence_wav(sample_rate: int, duration_ms: int) -> bytes:
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
    return buf.getvalue()
