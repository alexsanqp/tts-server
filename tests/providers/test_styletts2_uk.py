"""Unit tests for the StyleTTS2 Ukrainian provider.

Heavy ML deps (torch, styletts2_inference, ipa_uk, ukrainian_word_stress) are
not imported at module top so these tests run on CPU-only machines without
the optional ``[styletts2]`` extra. The GPU-marked test below is gated on a
runtime CUDA probe (or the ``TTS_RUN_STYLETTS2_GPU`` env var) so CI runners
skip it cleanly.
"""

from __future__ import annotations

import os

import pytest

from tts_server.core.errors import UnknownVoice, UnsupportedLanguage
from tts_server.providers.base import SynthesisRequest
from tts_server.providers.styletts2_uk import (
    _DEFAULT_VOICE,
    _clamp_speed,
    _is_ukrainian,
    _voice_filename,
    StyleTTS2UkProvider,
)


# ----- pure-function tests (no model needed) -----


def test_is_ukrainian_accepts_uk_variants() -> None:
    assert _is_ukrainian("uk")
    assert _is_ukrainian("uk-UA")
    assert _is_ukrainian("uk_UA")
    assert _is_ukrainian("UK-ua")
    assert not _is_ukrainian("en")
    assert not _is_ukrainian("ru")
    assert not _is_ukrainian("")


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (1.0, 1.0),
        (0.5, 0.5),
        (2.0, 2.0),
        (0.1, 0.5),   # clamps up
        (3.5, 2.0),   # clamps down
        (-1.0, 1.0),  # non-positive -> default
        (0.0, 1.0),
    ],
)
def test_clamp_speed(speed: float, expected: float) -> None:
    assert _clamp_speed(speed) == pytest.approx(expected)


def test_voice_filename_appends_pt_suffix() -> None:
    assert _voice_filename("Марина Панас") == "Марина Панас.pt"
    assert _voice_filename("Марина Панас.pt") == "Марина Панас.pt"


# ----- provider unit tests (no model load) -----


async def test_describe_returns_ukrainian_voices_and_is_gpu() -> None:
    provider = StyleTTS2UkProvider()
    caps = await provider.describe()

    assert caps.id == "styletts2-uk"
    assert caps.provider_family == "styletts2"
    assert caps.languages == ("uk",)
    assert caps.is_gpu is True
    assert caps.is_remote is False
    assert caps.native_sample_rate == 24000
    assert caps.native_format == "wav"
    assert caps.accepts_speed is True
    assert caps.supports_voice_id is True
    assert caps.supports_voice_cloning is False

    assert len(caps.voices) >= 1
    for voice in caps.voices:
        assert voice.languages == ("uk",)
    assert _DEFAULT_VOICE in {v.id for v in caps.voices}


async def test_default_voice_overridden_via_options() -> None:
    custom = "Тестовий Голос"
    provider = StyleTTS2UkProvider(options={"default_voice": custom})

    caps = await provider.describe()
    ids = {v.id for v in caps.voices}
    assert custom in ids, "custom default_voice must appear in the catalog"
    # probe_voice should accept the custom default too.
    assert await provider.probe_voice(custom) is True


async def test_default_voice_falls_back_when_options_missing() -> None:
    provider = StyleTTS2UkProvider()
    assert provider._default_voice == _DEFAULT_VOICE


async def test_default_voice_falls_back_when_options_none() -> None:
    provider = StyleTTS2UkProvider(options=None)
    assert provider._default_voice == _DEFAULT_VOICE


async def test_default_voice_falls_back_when_options_empty_string() -> None:
    provider = StyleTTS2UkProvider(options={"default_voice": ""})
    assert provider._default_voice == _DEFAULT_VOICE


async def test_synthesize_rejects_non_ukrainian_language() -> None:
    provider = StyleTTS2UkProvider()
    request = SynthesisRequest(
        text="hello",
        language="en-US",
        voice=_DEFAULT_VOICE,
        voice_kind="id",
        ref_text=None,
        speed=1.0,
        target_sample_rate=None,
        target_format="wav",
    )
    with pytest.raises(UnsupportedLanguage):
        await provider.synthesize(request)


async def test_synthesize_rejects_unknown_voice() -> None:
    provider = StyleTTS2UkProvider()
    request = SynthesisRequest(
        text="Привіт",
        language="uk",
        voice="No-Such-Voice-12345",
        voice_kind="id",
        ref_text=None,
        speed=1.0,
        target_sample_rate=None,
        target_format="wav",
    )
    with pytest.raises(UnknownVoice):
        await provider.synthesize(request)


async def test_probe_voice_checks_catalog_only() -> None:
    provider = StyleTTS2UkProvider()
    assert await provider.probe_voice(_DEFAULT_VOICE) is True
    assert await provider.probe_voice("Definitely-Not-Real") is False


# ----- GPU-gated integration test (real model load + synth) -----


def _gpu_available() -> bool:
    if os.environ.get("TTS_RUN_STYLETTS2_GPU"):
        return True
    try:
        import torch  # noqa: WPS433
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(
    not _gpu_available(),
    reason="StyleTTS2 model load is too expensive for CPU-only CI; "
    "set TTS_RUN_STYLETTS2_GPU=1 to run.",
)
async def test_synthesize_returns_wav_bytes_on_gpu() -> None:
    provider = StyleTTS2UkProvider()
    await provider.load()
    request = SynthesisRequest(
        text="Привіт",
        language="uk",
        voice=_DEFAULT_VOICE,
        voice_kind="id",
        ref_text=None,
        speed=1.0,
        target_sample_rate=None,
        target_format="wav",
    )
    stream = await provider.synthesize(request)
    assert stream.format == "wav"
    assert stream.sample_rate == 24000

    chunks: list[bytes] = []
    async for chunk in stream.chunks:
        chunks.append(chunk)
    body = b"".join(chunks)
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    assert len(body) > 1024  # at least a real frame of audio
