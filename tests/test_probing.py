"""Voice probing tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tts_server.core.probing import probe_provider, voice_available
from tts_server.providers.base import ProviderCapabilities, VoiceInfo


@dataclass
class _FakeProviderInstance:
    available_ids: set[str]

    async def probe_voice(self, voice_id: str) -> bool:
        return voice_id in self.available_ids


@dataclass
class _FakeEntry:
    id: str
    instance: _FakeProviderInstance
    capabilities: ProviderCapabilities
    voice_status: dict[str, bool] = field(default_factory=dict)


def _caps_with(voices: list[VoiceInfo]) -> ProviderCapabilities:
    return ProviderCapabilities(
        id="t", provider_family="t", languages=("en",), voices=tuple(voices)
    )


async def test_probe_records_status() -> None:
    voices = [VoiceInfo(id="alive", languages=("en",)), VoiceInfo(id="dead", languages=("en",))]
    entry = _FakeEntry(id="t", instance=_FakeProviderInstance({"alive"}), capabilities=_caps_with(voices))

    await probe_provider(entry, per_voice_timeout=0.5)

    assert entry.voice_status == {"alive": True, "dead": False}


async def test_probe_no_voices_is_noop() -> None:
    entry = _FakeEntry(id="t", instance=_FakeProviderInstance(set()), capabilities=_caps_with([]))
    await probe_provider(entry)
    # voice_status remains default empty dict; no crash
    assert entry.voice_status == {}


def test_voice_available_true_when_not_yet_probed() -> None:
    entry = _FakeEntry(id="t", instance=_FakeProviderInstance(set()), capabilities=_caps_with([]))
    # Simulate "never probed" by removing the attribute entirely
    del entry.voice_status
    assert voice_available(entry, "any") is True


def test_voice_available_reads_probe_result() -> None:
    entry = _FakeEntry(id="t", instance=_FakeProviderInstance(set()), capabilities=_caps_with([]))
    entry.voice_status = {"alive": True, "dead": False}
    assert voice_available(entry, "alive") is True
    assert voice_available(entry, "dead") is False
    # Unknown voice — default to True (don't hide things we didn't probe)
    assert voice_available(entry, "unknown") is True


async def test_probe_timeout_marks_unavailable() -> None:
    import asyncio

    class _SlowInstance:
        async def probe_voice(self, voice_id: str) -> bool:
            await asyncio.sleep(10)
            return True

    voices = [VoiceInfo(id="slow", languages=("en",))]
    entry = _FakeEntry(id="t", instance=_SlowInstance(), capabilities=_caps_with(voices))

    await probe_provider(entry, per_voice_timeout=0.1)
    assert entry.voice_status == {"slow": False}
