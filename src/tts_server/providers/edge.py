"""Edge-TTS provider — free, high-quality Microsoft Neural TTS.

Edge-TTS speaks over a WebSocket to the public Microsoft Translator endpoint.
The provider here is "remote" (no local model), produces MP3 natively at
24 kHz, and is wrapped in a defensive retry loop because the public endpoint
silently throttles bursts with `NoAudioReceived` and occasionally drops the
underlying connection.

The provider buffers the entire MP3 in memory and returns it as a single-chunk
`SynthesisStream`. Transcoding to wav is deliberately left to the API layer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    SynthesisStream,
    VoiceInfo,
)

logger = logging.getLogger(__name__)


# Default voices per BCP-47 language tag. These are the voices the provider
# actively advertises through `describe()`. Microsoft hosts ~400 voices total
# but they retire/rename without warning, so we curate a stable subset and
# expose `probe_voice` for honest availability checks.
DEFAULT_VOICES: dict[str, str] = {
    "en-US": "en-US-AriaNeural",
    "en-GB": "en-GB-SoniaNeural",
    "uk-UA": "uk-UA-PolinaNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "ru-RU": "ru-RU-SvetlanaNeural",
    "it-IT": "it-IT-ElsaNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "ko-KR": "ko-KR-SunHiNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "pl-PL": "pl-PL-AgnieszkaNeural",
}

# Gender hints for the curated catalog. Edge exposes more metadata over its
# `list_voices` endpoint, but that's a network call — keep `describe()`
# offline by hardcoding what we ship.
_VOICE_GENDERS: dict[str, str] = {
    "en-US-AriaNeural": "female",
    "en-GB-SoniaNeural": "female",
    "uk-UA-PolinaNeural": "female",
    "fr-FR-DeniseNeural": "female",
    "de-DE-KatjaNeural": "female",
    "es-ES-ElviraNeural": "female",
    "ru-RU-SvetlanaNeural": "female",
    "it-IT-ElsaNeural": "female",
    "zh-CN-XiaoxiaoNeural": "female",
    "ja-JP-NanamiNeural": "female",
    "ko-KR-SunHiNeural": "female",
    "pt-BR-FranciscaNeural": "female",
    "pl-PL-AgnieszkaNeural": "female",
}

FALLBACK_VOICE = "en-US-AriaNeural"

# Retry config — Microsoft throttles bursts with a silent NoAudioReceived.
# A2 mnemonic batch 2026-05-09 showed 10/24 lessons failed at the first
# English word; with these settings we observe <1% terminal failures.
_MAX_ATTEMPTS = 3
_PER_ATTEMPT_TIMEOUT_S = 60.0


def _speed_to_rate(speed: float) -> str:
    """Convert linear speed multiplier (1.0 = native) to Edge SSML rate.

    Edge accepts strings like "+10%", "-25%". A speed of 1.5 → "+50%".
    Clamps to [0.5, 2.0] to keep within Edge's practical sweet spot.
    """
    clamped = max(0.5, min(2.0, speed))
    pct = int(round((clamped - 1.0) * 100))
    sign = "+" if pct >= 0 else "-"
    return f"{sign}{abs(pct)}%"


def _resolve_voice(
    explicit: str | None,
    language: str | None,
    default_voices: dict[str, str],
    fallback: str,
) -> str:
    """Pick a voice in order: explicit → langcode match → fallback."""
    if explicit:
        return explicit
    if not language:
        return fallback
    # Exact tag wins.
    if language in default_voices:
        return default_voices[language]
    # Fuzzy match via CLDR distance.
    try:
        from langcodes import closest_match
    except ImportError:
        # Naive prefix fallback if langcodes isn't installed for some reason.
        prefix = language.split("-", 1)[0]
        for tag, vid in default_voices.items():
            if tag.lower().startswith(prefix.lower() + "-") or tag.lower() == prefix.lower():
                return vid
        return fallback
    try:
        tag, distance = closest_match(language, list(default_voices.keys()))
    except Exception:
        return fallback
    if tag == "und" or distance > 25:
        return fallback
    return default_voices[tag]


def _build_retry_exceptions() -> tuple[type[BaseException], ...]:
    """Assemble the retry-eligible exception tuple defensively.

    edge-tts depends on aiohttp; aiohttp.ClientError covers transient HTTP
    blips. We tolerate the import failing so a slimmed-down test env still
    gets the core retry surface (NoAudioReceived + timeout + OSError).
    """
    from edge_tts.exceptions import EdgeTTSException, NoAudioReceived

    excs: tuple[type[BaseException], ...] = (
        NoAudioReceived,
        EdgeTTSException,
        asyncio.TimeoutError,
        ConnectionError,
        OSError,
    )
    try:
        import aiohttp

        excs = (*excs, aiohttp.ClientError)
    except ImportError:
        pass
    return excs


async def _stream_to_bytes(communicate: Any) -> bytes:
    """Drain Communicate.stream() into a single bytes buffer."""
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and "data" in chunk:
            chunks.append(chunk["data"])
    return b"".join(chunks)


class EdgeProvider:
    """Microsoft Edge TTS provider.

    Options (from `[providers.edge]` in TOML, passed via constructor):
      - default_voice: str — fallback Microsoft voice id (default "en-US-AriaNeural")
      - voices: dict[str, str] — override the BCP-47 → voice-id map (optional)
      - max_attempts: int — retry budget per request (default 3)
      - per_attempt_timeout_s: float — hard timeout per synthesis call (default 60s)
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        opts = options or {}
        # Build voice map: merge defaults with caller override.
        overrides = opts.get("voices") or {}
        self._voices: dict[str, str] = {**DEFAULT_VOICES, **overrides}
        self._fallback_voice: str = opts.get("default_voice") or FALLBACK_VOICE
        self._max_attempts: int = int(opts.get("max_attempts", _MAX_ATTEMPTS))
        self._per_attempt_timeout_s: float = float(
            opts.get("per_attempt_timeout_s", _PER_ATTEMPT_TIMEOUT_S)
        )
        # Cached static catalog — built once.
        self._capabilities: ProviderCapabilities | None = None

    async def describe(self) -> ProviderCapabilities:
        """Return static capabilities. No I/O."""
        if self._capabilities is None:
            voices = tuple(
                VoiceInfo(
                    id=voice_id,
                    languages=(lang,),
                    gender=_VOICE_GENDERS.get(voice_id),
                    accepts_voice_id=True,
                    accepts_clone_ref=False,
                )
                for lang, voice_id in self._voices.items()
            )
            self._capabilities = ProviderCapabilities(
                id="edge",
                provider_family="edge",
                # Edge covers ~100 languages — advertise catch-all so the
                # router can dispatch any BCP-47 tag here.
                languages=("*",),
                voices=voices,
                supports_voice_id=True,
                supports_voice_cloning=False,
                native_sample_rate=24000,
                native_format="mp3",
                max_text_length=4500,
                accepts_speed=True,
                is_gpu=False,
                is_remote=True,
            )
        return self._capabilities

    async def load(self) -> None:
        """No-op. Edge has no local model to warm. Safe to call repeatedly."""
        return

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        """Synthesize text → MP3 bytes via Edge's WebSocket endpoint.

        Returns a single-chunk SynthesisStream. The API layer is responsible
        for any transcoding (e.g. mp3 → wav) the caller asked for.
        """
        import edge_tts

        voice = _resolve_voice(
            request.voice, request.language, self._voices, self._fallback_voice
        )
        rate = _speed_to_rate(request.speed)
        retry_excs = _build_retry_exceptions()

        last_exc: BaseException | None = None
        audio: bytes = b""
        for attempt in range(1, self._max_attempts + 1):
            communicate = edge_tts.Communicate(request.text, voice=voice, rate=rate)
            try:
                audio = await asyncio.wait_for(
                    _stream_to_bytes(communicate),
                    timeout=self._per_attempt_timeout_s,
                )
                if audio:
                    break
                # Empty body but no exception — treat as a soft NoAudioReceived.
                last_exc = RuntimeError("edge-tts returned empty audio")
            except retry_excs as exc:
                last_exc = exc
                if attempt >= self._max_attempts:
                    raise
                backoff = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Edge-TTS failure (%s: %s) for %r (attempt %d/%d); retrying in %ds",
                    type(exc).__name__,
                    str(exc)[:80] or "<no message>",
                    request.text[:40],
                    attempt,
                    self._max_attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            # Empty-audio path: don't sleep on the final attempt.
            if attempt >= self._max_attempts:
                break
            backoff = min(2 ** (attempt - 1), 8)
            logger.warning(
                "Edge-TTS returned empty audio for %r (attempt %d/%d); retrying in %ds",
                request.text[:40],
                attempt,
                self._max_attempts,
                backoff,
            )
            await asyncio.sleep(backoff)

        if not audio:
            # Exhausted retries with no audio — surface the last cause.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("edge-tts produced no audio after retries")

        logger.info(
            "edge: synthesized %d chars -> %d bytes mp3 (voice=%s, rate=%s)",
            len(request.text),
            len(audio),
            voice,
            rate,
        )

        async def _one_chunk() -> AsyncIterator[bytes]:
            yield audio

        return SynthesisStream(
            sample_rate=24000,
            format="mp3",
            duration_ms=_mp3_duration_ms(audio),
            chunks=_one_chunk(),
        )

    async def probe_voice(self, voice_id: str) -> bool:
        """Synthesize a single character to verify the voice still exists.

        Microsoft retires voices silently — `describe()` advertises what we
        ship, this is the honest check. Returns False on any failure
        (timeout, NoAudioReceived, network error, unknown voice).
        """
        import edge_tts

        try:

            async def _try() -> bool:
                communicate = edge_tts.Communicate("a", voice=voice_id)
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        return True
                return False

            return await asyncio.wait_for(_try(), timeout=5.0)
        except Exception as exc:
            logger.debug("probe_voice(%r) failed: %s", voice_id, exc)
            return False


# --- MP3 duration parser (pure Python; no extra deps) -----------------------
#
# Walks the MP3 frame headers and sums each frame's duration. Handles
# MPEG-1/2/2.5 layer III at any bitrate / sample rate. Skips ID3v2 tag at
# the start and ID3v1 tag at the end. Returns 0 on malformed input rather
# than raising — duration is metadata, not safety-critical.

_MP3_BITRATES = {
    # (version, layer) -> tuple keyed by 4-bit bitrate index. Layer III only.
    # MPEG-1 Layer III: ISO 11172-3 table.
    (3, 1): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),
    # MPEG-2 / MPEG-2.5 Layer III: ISO 13818-3 table (lower rates).
    (2, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    (0, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
}
_MP3_SAMPLE_RATES = {
    3: (44100, 48000, 32000, 0),  # MPEG-1
    2: (22050, 24000, 16000, 0),  # MPEG-2
    0: (11025, 12000, 8000, 0),   # MPEG-2.5
}


def _mp3_duration_ms(data: bytes) -> int:
    """Sum per-frame durations across an MP3 bytestream."""
    if not data or len(data) < 4:
        return 0

    offset = 0
    # Skip ID3v2 tag at start.
    if data[:3] == b"ID3" and len(data) > 10:
        size = (
            (data[6] & 0x7F) << 21
            | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7
            | (data[9] & 0x7F)
        )
        offset = 10 + size

    # Strip ID3v1 trailer.
    end = len(data) - 128 if data[-128:-125] == b"TAG" else len(data)

    duration_ms = 0.0
    failures = 0
    while offset + 4 <= end:
        b0, b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
            offset += 1
            failures += 1
            if failures > 1024:
                return 0
            continue

        version = (b1 >> 3) & 0x03  # 0=2.5, 2=2, 3=1
        layer = (b1 >> 1) & 0x03    # 1=LayerIII
        if layer != 1 or version == 1:
            return 0  # not Layer III, or reserved

        bitrate_idx = (b2 >> 4) & 0x0F
        rate_idx = (b2 >> 2) & 0x03
        padding = (b2 >> 1) & 0x01

        try:
            bitrate_kbps = _MP3_BITRATES[(version, layer)][bitrate_idx]
            sample_rate = _MP3_SAMPLE_RATES[version][rate_idx]
        except (KeyError, IndexError):
            return 0
        if bitrate_kbps == 0 or sample_rate == 0:
            return 0

        samples_per_frame = 1152 if version == 3 else 576  # Layer III
        frame_bytes = (samples_per_frame * bitrate_kbps * 1000) // (8 * sample_rate) + padding
        if frame_bytes <= 0:
            return 0
        duration_ms += samples_per_frame * 1000 / sample_rate
        offset += frame_bytes
        failures = 0

    return int(round(duration_ms))
