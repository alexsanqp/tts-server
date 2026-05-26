"""Honest voice availability probing.

Microsoft retires edge-tts voices silently — a voice that synthesized fine
yesterday may 404 today. Calling `provider.probe_voice(voice_id)` on each
advertised voice at startup catches this proactively. Results are stored
on the ProviderEntry and surfaced via /v1/voices.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def probe_provider(entry, *, per_voice_timeout: float = 6.0) -> None:
    """Probe every voice on a provider and stash availability."""
    voices = entry.capabilities.voices
    if not voices:
        return

    status: dict[str, bool] = {}

    async def _probe(voice_id: str) -> tuple[str, bool]:
        try:
            ok = await asyncio.wait_for(
                entry.instance.probe_voice(voice_id),
                timeout=per_voice_timeout,
            )
            return voice_id, bool(ok)
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.debug("Voice probe failed: %s/%s: %s", entry.id, voice_id, exc)
            return voice_id, False

    results = await asyncio.gather(*(_probe(v.id) for v in voices), return_exceptions=False)
    for vid, ok in results:
        status[vid] = ok

    unavailable = [vid for vid, ok in status.items() if not ok]
    if unavailable:
        logger.warning(
            "Provider %r: %d/%d voices unavailable: %s",
            entry.id, len(unavailable), len(status), unavailable[:5],
        )
    else:
        logger.info("Provider %r: all %d voices probed OK", entry.id, len(status))

    setattr(entry, "voice_status", status)


def voice_available(entry, voice_id: str) -> bool:
    """True if the voice has been probed and is alive (or not probed yet)."""
    status = getattr(entry, "voice_status", None)
    if status is None:
        return True  # not yet probed — assume available
    return status.get(voice_id, True)
