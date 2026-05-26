"""Provider Protocol + data shapes.

A provider is a class that implements `TTSProvider`. It declares its
capabilities (languages, voices, what kinds of `voice` strings it accepts)
through `describe()`, lazily loads heavy resources via `load()`, and produces
an async-streamable audio response through `synthesize()`.

Even when a provider doesn't actually stream, the response is an
`AsyncIterator[bytes]` that yields once — this preserves the door for v2
streaming without rewriting providers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VoiceInfo:
    """A single voice exposed by a provider.

    `accepts_clone_ref` flags voices that double as cloning references
    (e.g. an XTTS-v2 speaker whose audio can also be used to clone a new
    voice). Most voices have `accepts_voice_id=True, accepts_clone_ref=False`.
    """

    id: str
    languages: tuple[str, ...]
    gender: str | None = None
    accepts_voice_id: bool = True
    accepts_clone_ref: bool = False
    # Free-form provider-specific bits. For Qwen, this carries ref_text.
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static (or async-discovered) description of what a provider can do."""

    id: str
    provider_family: str  # "edge" | "qwen" | "styletts2" | ...
    languages: tuple[str, ...]  # BCP-47 primary tags, or ("*",) for catch-all
    voices: tuple[VoiceInfo, ...] = ()
    supports_voice_id: bool = True
    supports_voice_cloning: bool = False
    native_sample_rate: int = 24000
    native_format: str = "wav"  # "wav" | "mp3"
    max_text_length: int = 5000
    accepts_speed: bool = False
    is_gpu: bool = False  # hints default concurrency to 1
    is_remote: bool = False  # provider talks over network (edge-tts, qwen sidecar)


@dataclass(frozen=True)
class SynthesisRequest:
    """What the HTTP layer hands to a provider after validation + routing.

    `voice` is the resolved voice id OR (for cloning) the absolute local
    path to a ref audio file. `voice_kind` disambiguates. `ref_text` is
    only populated for `voice_kind == "clone_ref"` providers that need it
    (Qwen does).
    """

    text: str
    language: str  # BCP-47, e.g. "en-US"
    voice: str | None
    voice_kind: str  # "id" | "clone_ref" | "none"
    ref_text: str | None
    speed: float  # 1.0 = native; provider clamps/approximates
    target_sample_rate: int | None
    target_format: str  # "wav" | "mp3"


@dataclass
class SynthesisStream:
    """Response from a provider — headers known up front, body streams."""

    sample_rate: int
    format: str  # "wav" | "mp3"
    duration_ms: int  # 0 if not known until done
    chunks: AsyncIterator[bytes]


@runtime_checkable
class TTSProvider(Protocol):
    """Provider interface. Implement these four methods."""

    async def describe(self) -> ProviderCapabilities:
        """Return capabilities. Cheap; called for /v1/models. No heavy imports."""
        ...

    async def load(self) -> None:
        """Warm up. Idempotent. Heavy ML imports go here, not at module top."""
        ...

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        """Produce audio. Stream-shaped even for non-streaming providers."""
        ...

    async def probe_voice(self, voice_id: str) -> bool:
        """Verify a voice is actually usable (e.g. edge-tts retires voices silently).

        Default-implementable as `return True`. Used by /v1/voices to mark
        advertised-but-broken voices unavailable.
        """
        ...
