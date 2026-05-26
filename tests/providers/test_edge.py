"""Unit + (gated) integration tests for the Edge-TTS provider.

Unit tests mock `edge_tts.Communicate` so they run offline. The single
network test is gated behind RUN_NETWORK_TESTS=1 and actually talks to
Microsoft — skip it in CI unless you've budgeted for it.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    VoiceInfo,
)
from tts_server.providers.edge import EdgeProvider, _speed_to_rate


# ---------------------------------------------------------------------------
# Helpers — fake Communicate replacements
# ---------------------------------------------------------------------------


class _FakeCommunicate:
    """Records construction args and yields a fixed audio payload."""

    instances: list["_FakeCommunicate"] = []

    def __init__(self, text: str, voice: str | None = None, **kwargs: Any) -> None:
        self.text = text
        self.voice = voice
        self.kwargs = kwargs
        type(self).instances.append(self)

    async def stream(self):
        # Mimic edge-tts's chunk dict shape with one audio chunk + a boundary.
        yield {"type": "WordBoundary", "offset": 0.0, "duration": 0.1, "text": "x"}
        yield {"type": "audio", "data": b"\xff\xfbFAKE-MP3-BYTES"}


class _FlakyCommunicate:
    """Raises NoAudioReceived on the first N constructions, then succeeds."""

    instances: list["_FlakyCommunicate"] = []
    fail_until_attempt: int = 3  # succeed on the 3rd attempt

    def __init__(self, text: str, voice: str | None = None, **kwargs: Any) -> None:
        self.text = text
        self.voice = voice
        self.kwargs = kwargs
        self.attempt = len(type(self).instances) + 1
        type(self).instances.append(self)

    async def stream(self):
        if self.attempt < type(self).fail_until_attempt:
            from edge_tts.exceptions import NoAudioReceived

            raise NoAudioReceived("throttled")
        yield {"type": "audio", "data": b"\xff\xfbOK"}


@pytest.fixture(autouse=True)
def _reset_fake_state():
    """Each test gets a clean instance log + zeroed sleep delays."""
    _FakeCommunicate.instances.clear()
    _FlakyCommunicate.instances.clear()
    _FlakyCommunicate.fail_until_attempt = 3
    # Make exponential backoff free during unit tests.
    async def _no_sleep(_: float) -> None:
        return None

    with patch("tts_server.providers.edge.asyncio.sleep", new=_no_sleep):
        yield


def _make_request(
    text: str = "hello",
    language: str = "en-US",
    voice: str | None = None,
    speed: float = 1.0,
) -> SynthesisRequest:
    return SynthesisRequest(
        text=text,
        language=language,
        voice=voice,
        voice_kind="id" if voice else "none",
        ref_text=None,
        speed=speed,
        target_sample_rate=None,
        target_format="mp3",
    )


async def _drain(stream) -> bytes:
    out = b""
    async for chunk in stream.chunks:
        out += chunk
    return out


# ---------------------------------------------------------------------------
# Pure-function tests (no mocks needed)
# ---------------------------------------------------------------------------


def test_speed_to_rate_native_is_zero() -> None:
    assert _speed_to_rate(1.0) == "+0%"


def test_speed_to_rate_fast() -> None:
    assert _speed_to_rate(1.5) == "+50%"


def test_speed_to_rate_slow() -> None:
    assert _speed_to_rate(0.75) == "-25%"


def test_speed_to_rate_clamps_high() -> None:
    # 5.0 should clamp to 2.0 → +100%
    assert _speed_to_rate(5.0) == "+100%"


def test_speed_to_rate_clamps_low() -> None:
    # 0.1 should clamp to 0.5 → -50%
    assert _speed_to_rate(0.1) == "-50%"


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


async def test_describe_returns_expected_catalog() -> None:
    provider = EdgeProvider()
    caps = await provider.describe()

    assert isinstance(caps, ProviderCapabilities)
    assert caps.id == "edge"
    assert caps.provider_family == "edge"
    assert caps.languages == ("*",)
    assert caps.supports_voice_id is True
    assert caps.supports_voice_cloning is False
    assert caps.native_sample_rate == 24000
    assert caps.native_format == "mp3"
    assert caps.max_text_length == 4500
    assert caps.accepts_speed is True
    assert caps.is_gpu is False
    assert caps.is_remote is True

    voice_ids = {v.id for v in caps.voices}
    assert "en-US-AriaNeural" in voice_ids
    assert "uk-UA-PolinaNeural" in voice_ids


async def test_describe_voices_are_voiceinfo_with_languages() -> None:
    provider = EdgeProvider()
    caps = await provider.describe()
    for voice in caps.voices:
        assert isinstance(voice, VoiceInfo)
        assert voice.languages, f"{voice.id} has no languages"
        assert voice.accepts_voice_id is True
        assert voice.accepts_clone_ref is False


async def test_describe_is_idempotent_and_cheap() -> None:
    # Calling describe() should not hit the network — instantiate without
    # patching anything and confirm it returns immediately.
    provider = EdgeProvider()
    caps_a = await provider.describe()
    caps_b = await provider.describe()
    assert caps_a is caps_b  # cached


async def test_describe_options_override_default_voice() -> None:
    provider = EdgeProvider(options={"default_voice": "en-US-GuyNeural"})
    # default_voice doesn't change the advertised catalog, it changes the
    # fallback — verify by synthesizing without a voice/language.
    assert provider._fallback_voice == "en-US-GuyNeural"


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


async def test_load_is_noop_and_idempotent() -> None:
    provider = EdgeProvider()
    await provider.load()
    await provider.load()  # no-op, no exceptions


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------


async def test_synthesize_returns_mp3_bytes() -> None:
    provider = EdgeProvider()

    with patch("edge_tts.Communicate", _FakeCommunicate):
        stream = await provider.synthesize(_make_request())

    assert stream.format == "mp3"
    assert stream.sample_rate == 24000

    body = await _drain(stream)
    assert body == b"\xff\xfbFAKE-MP3-BYTES"


async def test_synthesize_passes_speed_as_rate_string() -> None:
    provider = EdgeProvider()

    with patch("edge_tts.Communicate", _FakeCommunicate):
        await provider.synthesize(_make_request(speed=1.5))

    assert len(_FakeCommunicate.instances) == 1
    assert _FakeCommunicate.instances[0].kwargs.get("rate") == "+50%"


async def test_synthesize_uses_explicit_voice() -> None:
    provider = EdgeProvider()

    with patch("edge_tts.Communicate", _FakeCommunicate):
        await provider.synthesize(_make_request(voice="en-GB-RyanNeural"))

    assert _FakeCommunicate.instances[0].voice == "en-GB-RyanNeural"


async def test_synthesize_resolves_voice_from_language() -> None:
    provider = EdgeProvider()

    with patch("edge_tts.Communicate", _FakeCommunicate):
        await provider.synthesize(_make_request(language="uk-UA"))

    assert _FakeCommunicate.instances[0].voice == "uk-UA-PolinaNeural"


async def test_synthesize_falls_back_to_default_voice_without_language() -> None:
    provider = EdgeProvider(options={"default_voice": "en-US-AriaNeural"})

    with patch("edge_tts.Communicate", _FakeCommunicate):
        await provider.synthesize(_make_request(language=""))

    assert _FakeCommunicate.instances[0].voice == "en-US-AriaNeural"


async def test_synthesize_uses_language_fuzzy_match() -> None:
    # "en" without region should fuzzy-match to en-US-AriaNeural.
    provider = EdgeProvider()

    with patch("edge_tts.Communicate", _FakeCommunicate):
        await provider.synthesize(_make_request(language="en"))

    assert _FakeCommunicate.instances[0].voice == "en-US-AriaNeural"


async def test_synthesize_retries_on_no_audio_received() -> None:
    """First two attempts raise NoAudioReceived, third succeeds."""
    provider = EdgeProvider(options={"max_attempts": 3})

    with patch("edge_tts.Communicate", _FlakyCommunicate):
        stream = await provider.synthesize(_make_request())

    body = await _drain(stream)
    assert body == b"\xff\xfbOK"
    assert len(_FlakyCommunicate.instances) == 3  # 2 failures + 1 success


async def test_synthesize_raises_after_exhausting_retries() -> None:
    """All attempts raise NoAudioReceived → final exception propagates."""
    from edge_tts.exceptions import NoAudioReceived

    provider = EdgeProvider(options={"max_attempts": 2})
    # Make the flaky communicate fail on all attempts within budget.
    _FlakyCommunicate.fail_until_attempt = 99

    with patch("edge_tts.Communicate", _FlakyCommunicate):
        with pytest.raises(NoAudioReceived):
            await provider.synthesize(_make_request())

    assert len(_FlakyCommunicate.instances) == 2


# ---------------------------------------------------------------------------
# probe_voice()
# ---------------------------------------------------------------------------


async def test_probe_voice_returns_false_on_exception() -> None:
    """Unknown voice → edge raises, probe should swallow and return False."""

    class _FailingCommunicate:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def stream(self):
            from edge_tts.exceptions import NoAudioReceived

            raise NoAudioReceived("no such voice")
            yield  # pragma: no cover — make this an async generator

    provider = EdgeProvider()
    with patch("edge_tts.Communicate", _FailingCommunicate):
        assert await provider.probe_voice("nonsense-voice") is False


async def test_probe_voice_returns_true_when_audio_received() -> None:
    provider = EdgeProvider()
    with patch("edge_tts.Communicate", _FakeCommunicate):
        assert await provider.probe_voice("en-US-AriaNeural") is True


# ---------------------------------------------------------------------------
# Network integration test — gated. Run with RUN_NETWORK_TESTS=1.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="Set RUN_NETWORK_TESTS=1 to run real edge-tts network calls",
)
async def test_synthesize_real_call_produces_valid_mp3() -> None:
    provider = EdgeProvider()
    stream = await provider.synthesize(
        _make_request(text="hello", language="en-US", voice="en-US-AriaNeural")
    )
    assert stream.format == "mp3"
    body = await _drain(stream)
    assert len(body) > 0
    # MP3 magic: either an MPEG frame sync (0xFFFx) or an ID3 tag.
    assert body[:3] == b"ID3" or (body[0] == 0xFF and (body[1] & 0xE0) == 0xE0)


@pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="Set RUN_NETWORK_TESTS=1 to run real edge-tts network calls",
)
async def test_probe_voice_real_call() -> None:
    provider = EdgeProvider()
    assert await provider.probe_voice("en-US-AriaNeural") is True
