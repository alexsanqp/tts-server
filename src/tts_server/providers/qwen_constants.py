"""Shared constants for the Qwen3-TTS proxy and the sidecar subprocess.

The proxy (`tts_server.providers.qwen.QwenProvider`) lives in the FastAPI
process; the worker (`tts_server.sidecars.qwen_worker`) is a child Python
process spawned by it. They MUST agree on language tags and reference-text
fallbacks. Keeping that shared vocabulary in one module makes it impossible
for them to drift silently.
"""

from __future__ import annotations

# Qwen3-TTS supported BCP-47 primary tags.
QWEN_LANGUAGES: tuple[str, ...] = (
    "en", "de", "fr", "it", "es", "ru", "ja", "ko", "zh", "pt",
)

# Lang code → human-readable name expected by the Qwen model API.
LANG_MAP: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "pt": "Portuguese",
    "it": "Italian",
}

# Languages explicitly NOT supported by Qwen3-TTS. The worker returns 400
# for these so the caller can route elsewhere (e.g. lingua-pairs falls
# back to edge-tts for `uk`).
UNSUPPORTED_LANGS: frozenset[str] = frozenset({
    "uk", "pl", "ar", "hi", "tr", "nl", "sv", "no", "fi", "da",
})

# Legacy fallback texts for plain-stem ref clips (e.g. en.mp3 → ref:en-default).
# New deployments use <lang>-<name>.{mp3,wav} + <lang>-<name>.json sidecars;
# this dict only matters for catalogues that pre-date the sidecar layout.
REF_TEXTS: dict[str, str] = {
    "en": (
        "Hello, my name is your English teacher. "
        "Today we will learn new vocabulary words together."
    ),
}
