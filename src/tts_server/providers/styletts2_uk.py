"""StyleTTS2 Ukrainian provider — ``patriotyk/styletts2_ukrainian_multispeaker``.

License: MIT (model + inference code). Ukrainian-only — any other language
raises ``UnsupportedLanguage`` so the router can fail loudly rather than
mis-routing to the wrong engine.

The model weights (~770 MB) auto-download into the HuggingFace cache on first
:meth:`load`; voice style ``.pt`` tensors (~2 KB each) are pulled from the
companion HF Space ``patriotyk/styletts2-ukrainian`` and cached under
``~/.cache/tts-server/styletts2_voices/`` (honouring ``XDG_CACHE_HOME`` on
Linux). Heavy ML deps (``torch``, ``styletts2_inference``, ``ipa_uk``,
``ukrainian_word_stress``) are imported lazily inside :meth:`load` /
:meth:`synthesize` so importing this module — or :meth:`describe`-ing it for
``/v1/models`` — does not require the optional ``[styletts2]`` extra.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unicodedata import normalize as _unicode_normalize

from tts_server.core.errors import UnknownVoice, UnsupportedLanguage
from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    SynthesisStream,
    VoiceInfo,
)

logger = logging.getLogger(__name__)


_VOICE_DOWNLOAD_TEMPLATE = (
    "https://huggingface.co/spaces/patriotyk/styletts2-ukrainian/"
    "resolve/main/voices/{filename}"
)
_DEFAULT_VOICE = "Марина Панас"
_NATIVE_SAMPLE_RATE = 24000
_MAX_TEXT_LENGTH = 2000

# Catalog of voices advertised through /v1/voices. The HF Space exposes
# more, but only "Марина Панас" is verified in the lingua-pairs source.
# probe_voice() uses set membership; synthesis itself accepts arbitrary
# voice names (downloaded on demand) so users can still target unlisted
# voices via the API if they know the exact filename on the HF Space.
_BUILTIN_VOICES: tuple[tuple[str, str | None], ...] = (
    ("Марина Панас", "female"),
)

_DASH_CHARS_RE = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_SPACED_HYPHEN_RE = re.compile(r" - ")
_SUPPORTED_LANG_PRIMARY = "uk"

# Match a numeric token: optional leading minus (with optional space),
# integer digits, optional decimal part using `.` or `,` as separator.
# The leading-minus alternation requires either start-of-string or a
# non-word boundary so we don't swallow the hyphen in compound words
# like "кафе-бар". Decimal uses (?:\.|,) so we don't match trailing
# punctuation that happens to follow a bare integer.
_NUMBER_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s(\[]))-?\d+(?:[.,]\d+)?"
)


def _is_ukrainian(lang: str) -> bool:
    if not lang:
        return False
    return lang.replace("_", "-").split("-", 1)[0].lower() == _SUPPORTED_LANG_PRIMARY


def _clamp_speed(speed: float) -> float:
    """Clamp speed into the supported [0.5, 2.0] range.

    StyleTTS2 accepts arbitrary positive speeds but quality degrades sharply
    outside this band. Clamp rather than reject so the router can pass the
    request through unmodified.
    """
    if speed <= 0:
        return 1.0
    return max(0.5, min(2.0, float(speed)))


def _voice_cache_dir() -> Path:
    """Cache directory honouring ``XDG_CACHE_HOME`` on Linux.

    On Windows / macOS we fall back to ``~/.cache/tts-server`` for
    consistency — apps cross-mounting the cache between WSL and Windows
    benefit from a stable path.
    """
    xdg = os.environ.get("XDG_CACHE_HOME") if sys.platform.startswith("linux") else None
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "tts-server" / "styletts2_voices"


def normalize_digits_uk(text: str) -> str:
    """Replace numeric runs with Ukrainian word equivalents (cardinal form).

    StyleTTS2 -> ipa_uk -> phoneme tokens has no digit handling; bare digits
    silently drop out of the audio. Pre-expanding via num2words keeps the
    fix pure-Python and offline.

    Handles cardinals, negatives ("-3" -> "минус три"), and decimals using
    `.` or `,` as separator ("3.5" / "3,5" -> "три кома пять"). Tokens
    embedded inside word-like sequences (e.g. "MP3", "COVID19") are NOT
    matched — the regex requires a leading whitespace / `(` / `[` / SOL
    boundary so identifier-like compounds stay intact for the IPA layer
    to deal with.
    """
    from num2words import num2words

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        for sep in (".", ","):
            if sep in token:
                negative = token.startswith("-")
                body = token[1:] if negative else token
                int_part, frac_part = body.split(sep, 1)
                int_words = num2words(int(int_part or "0"), lang="uk")
                frac_words = num2words(int(frac_part), lang="uk")
                spoken = f"{int_words} кома {frac_words}"
                return f"мінус {spoken}" if negative else spoken
        return num2words(int(token), lang="uk")

    return _NUMBER_RE.sub(_replace, text)


def _normalize_uk(text: str) -> str:
    """Reproduce the HF Space's text-normalization chain for StyleTTS2.

    NFKC collapses presentation forms (full-width punctuation, ligatures) into
    canonical code points so the IPA tokenizer sees consistent input. The
    ``+`` -> combining-acute remap is the model's stress-marking convention.
    Adding a trailing terminator avoids a known prosody glitch where the
    final phoneme gets clipped by the vocoder.
    """
    from ukrainian_word_stress import StressSymbol

    text = text.replace("+", StressSymbol.CombiningAcuteAccent)
    text = _unicode_normalize("NFKC", text)
    text = normalize_digits_uk(text)
    text = _DASH_CHARS_RE.sub("-", text)
    if not text:
        return text
    if text[-1] not in ".?!:-":
        text = text + "."
    text = _SPACED_HYPHEN_RE.sub(": ", text)
    return text


def _voice_filename(voice: str) -> str:
    return voice if voice.endswith(".pt") else f"{voice}.pt"


def _ensure_voice_file(voice: str, cache_dir: Path) -> Path:
    """Resolve voice name -> local ``.pt`` path; download from HF Space if needed."""
    target = cache_dir / _voice_filename(voice)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = _VOICE_DOWNLOAD_TEMPLATE.format(
        filename=urllib.parse.quote(_voice_filename(voice))
    )
    logger.info("StyleTTS2: downloading voice %r from %s", voice, url)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _build_capabilities(voice_ids: tuple[str, ...]) -> ProviderCapabilities:
    """Static capabilities, parameterized by voice catalog (test hook)."""
    gender_by_id = {vid: gender for vid, gender in _BUILTIN_VOICES}
    voices = tuple(
        VoiceInfo(
            id=vid,
            languages=("uk",),
            gender=gender_by_id.get(vid),
        )
        for vid in voice_ids
    )
    return ProviderCapabilities(
        id="styletts2-uk",
        provider_family="styletts2",
        languages=("uk",),
        voices=voices,
        supports_voice_id=True,
        supports_voice_cloning=False,
        native_sample_rate=_NATIVE_SAMPLE_RATE,
        native_format="wav",
        max_text_length=_MAX_TEXT_LENGTH,
        accepts_speed=True,
        is_gpu=True,
        is_remote=False,
    )


class StyleTTS2UkProvider:
    """Local StyleTTS2 Ukrainian multispeaker provider (MIT-licensed).

    Loads ``patriotyk/styletts2_ukrainian_multispeaker`` (~770 MB) on the
    first :meth:`load` call. CUDA is used when available; otherwise it
    falls back to CPU (slower, but functional). Output is mono 24 kHz WAV
    bytes wrapped in a single-chunk ``SynthesisStream``.

    The registry serializes :meth:`synthesize` calls via the concurrency
    controller (``is_gpu=True`` defaults to 1) so the underlying model is
    safe to call without an internal lock.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self._options = options or {}
        self._default_voice = str(
            self._options.get("default_voice") or _DEFAULT_VOICE
        )
        self._cache_dir = _voice_cache_dir()
        self._voice_ids: tuple[str, ...] = tuple(
            vid for vid, _ in _BUILTIN_VOICES
        )
        # Ensure default voice is always advertised so /v1/voices reflects it.
        if self._default_voice not in self._voice_ids:
            self._voice_ids = (self._default_voice, *self._voice_ids)
        self._capabilities = _build_capabilities(self._voice_ids)

        # Lazy-loaded heavy state.
        self._model: Any = None
        self._device: str | None = None
        self._stressify: Any = None
        self._voice_cache: dict[str, Any] = {}
        self._load_lock = asyncio.Lock()

    async def describe(self) -> ProviderCapabilities:
        return self._capabilities

    async def load(self) -> None:
        """Idempotent warm-up: imports torch + styletts2_inference, builds the model."""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        # huggingface_hub uses httpx, which honours certifi instead of the
        # OS trust store. On corp networks with TLS-intercepting proxies,
        # the corp root CA lives in the OS store but not certifi —
        # downloads then fail with self-signed-cert errors. truststore
        # routes httpx through the OS store. Best-effort: missing import
        # falls back to certifi (fine on non-intercepted networks).
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass

        import torch
        from styletts2_inference.models import StyleTTS2
        from ukrainian_word_stress import Stressifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "StyleTTS2: loading patriotyk/styletts2_ukrainian_multispeaker on %s",
            device,
        )
        self._model = StyleTTS2(
            hf_path="patriotyk/styletts2_ukrainian_multispeaker",
            device=device,
        )
        self._device = device
        self._stressify = Stressifier()

    async def probe_voice(self, voice_id: str) -> bool:
        return voice_id in self._voice_ids

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        if not _is_ukrainian(request.language):
            raise UnsupportedLanguage(
                f"StyleTTS2UkProvider only supports Ukrainian (uk/uk-UA); "
                f"got language={request.language!r}"
            )

        voice = request.voice or self._default_voice
        # probe_voice gates the advertised catalog; reject anything unknown
        # so callers don't silently trigger a network download for typos.
        if voice not in self._voice_ids:
            raise UnknownVoice(
                f"Unknown StyleTTS2 voice: {voice!r}"
            )

        speed = _clamp_speed(request.speed)

        await self.load()
        wav_bytes = await asyncio.to_thread(
            self._synthesize_blocking, request.text, voice, speed
        )

        async def _one_chunk() -> AsyncIterator[bytes]:
            yield wav_bytes

        return SynthesisStream(
            sample_rate=_NATIVE_SAMPLE_RATE,
            format="wav",
            duration_ms=0,  # streamed back as a single WAV; client can probe.
            chunks=_one_chunk(),
        )

    def _synthesize_blocking(self, text: str, voice: str, speed: float) -> bytes:
        import soundfile
        from ipa_uk import ipa

        assert self._model is not None
        assert self._stressify is not None

        normalized = _normalize_uk(text)
        stressed = self._stressify(normalized)
        phonemes = ipa(stressed)
        tokens = self._model.tokenizer.encode(phonemes)

        style = self._load_voice_style(voice)
        wav = self._model(tokens, speed=speed, s_prev=style)
        wav_np = wav.cpu().numpy() if hasattr(wav, "cpu") else wav

        buf = io.BytesIO()
        soundfile.write(buf, wav_np, _NATIVE_SAMPLE_RATE, format="WAV")
        return buf.getvalue()

    def _load_voice_style(self, voice: str) -> Any:
        if voice in self._voice_cache:
            return self._voice_cache[voice]
        import torch

        path = _ensure_voice_file(voice, self._cache_dir)
        style = torch.load(str(path), map_location=self._device or "cpu")
        self._voice_cache[voice] = style
        return style
