"""Error hierarchy mapped to HTTP responses by the API layer."""

from __future__ import annotations


class TTSError(Exception):
    """Base TTS server error. `code` becomes the JSON error.code in responses."""

    code: str = "tts_error"
    http_status: int = 500

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class UnknownModel(TTSError):
    code = "unknown_model"
    http_status = 422


class UnsupportedLanguage(TTSError):
    code = "unsupported_language"
    http_status = 422


class UnknownVoice(TTSError):
    code = "unknown_voice"
    http_status = 422


class InputTooLong(TTSError):
    code = "input_too_long"
    http_status = 413


class ProviderUnavailable(TTSError):
    code = "provider_unavailable"
    http_status = 503


class ProviderFailure(TTSError):
    code = "provider_failure"
    http_status = 502


class CapacityExceeded(TTSError):
    code = "capacity_exceeded"
    http_status = 503
