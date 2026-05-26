"""Reference-audio storage for voice cloning.

Two tiers:
* **catalog_dir**: read-only baked-in refs committed with the deployment
  (e.g. `data/refs-catalog/en.mp3`). These have stable ids like
  ``ref:en-default`` derived from the filename stem.
* **upload_dir**: client-uploaded refs via POST /v1/refs. Filename is the
  sha256 of the content; ref id is ``ref:<first 12 hex chars>``. Subject
  to TTL eviction.

`resolve(ref_id)` looks in both tiers and returns an absolute Path or None.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Slug whitelist for ref ids. Blocks path traversal and OS-specific
# weirdness (NTFS alternate data streams, NUL bytes, leading dots, etc.).
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/vorbis": ".ogg",
}


@dataclass
class RefStored:
    ref_id: str
    path: Path
    size_bytes: int


class RefStoreError(Exception):
    pass


class RefStore:
    def __init__(
        self,
        *,
        catalog_dir: Path,
        upload_dir: Path,
        upload_ttl_hours: int = 24,
        max_upload_mb: int = 10,
    ) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.upload_dir = Path(upload_dir)
        self.upload_ttl_seconds = upload_ttl_hours * 3600
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self._sweep_task: asyncio.Task | None = None

    # ---- catalog ids ----

    def catalog_ids(self) -> list[str]:
        if not self.catalog_dir.is_dir():
            return []
        ids = []
        seen: set[str] = set()
        for path in sorted(self.catalog_dir.iterdir()):
            if path.suffix.lower() not in _ALLOWED_EXTS:
                continue
            stem = path.stem.lower()
            if stem in seen:
                continue
            seen.add(stem)
            ids.append(f"ref:{stem}-default")
        return ids

    # ---- resolve ----

    def resolve(self, ref_id: str) -> Path | None:
        """Return absolute Path for a ref_id or None if not found."""
        if not ref_id.startswith("ref:"):
            return None
        slug = ref_id[4:]

        # Catalog: "ref:<stem>-default"
        if slug.endswith("-default"):
            stem = slug[: -len("-default")]
            if not _SAFE_SLUG_RE.match(stem):
                return None
            catalog_root = self.catalog_dir.resolve()
            for ext in _ALLOWED_EXTS:
                p = (self.catalog_dir / f"{stem}{ext}").resolve()
                # Defense in depth: even with the regex above, refuse anything
                # that resolves outside catalog_dir.
                if not _is_within(p, catalog_root):
                    continue
                if p.is_file():
                    return p
            return None

        # Upload: "ref:<first12hex>" — match files starting with that prefix
        if not _looks_like_hex(slug):
            return None
        if not self.upload_dir.is_dir():
            return None
        upload_root = self.upload_dir.resolve()
        for path in self.upload_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_EXTS:
                continue
            if not path.stem.startswith(slug):
                continue
            resolved = path.resolve()
            if _is_within(resolved, upload_root):
                return resolved
        return None

    # ---- upload ----

    async def store(
        self,
        *,
        content: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> RefStored:
        if len(content) == 0:
            raise RefStoreError("empty upload")
        if len(content) > self.max_upload_bytes:
            raise RefStoreError(
                f"upload exceeds {self.max_upload_bytes // 1024 // 1024} MB limit"
            )

        ext = _guess_extension(filename, content_type)
        if ext is None:
            raise RefStoreError(
                f"unsupported audio type (filename={filename!r}, content_type={content_type!r})"
            )

        digest = hashlib.sha256(content).hexdigest()
        ref_id = f"ref:{digest[:12]}"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.upload_dir / f"{digest}{ext}"
        if not target.exists():
            # atomic write
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(content)
            tmp.replace(target)

        return RefStored(ref_id=ref_id, path=target.resolve(), size_bytes=len(content))

    # ---- TTL sweep ----

    async def start_background_sweep(self, *, interval_seconds: float = 600.0) -> None:
        if self._sweep_task is not None:
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop(interval_seconds))

    async def stop(self) -> None:
        if self._sweep_task is None:
            return
        self._sweep_task.cancel()
        try:
            await self._sweep_task
        except asyncio.CancelledError:
            pass
        self._sweep_task = None

    async def _sweep_loop(self, interval: float) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("RefStore sweep failed: %s", exc)

    def sweep_once(self) -> int:
        """Delete uploads older than TTL. Returns count removed."""
        if not self.upload_dir.is_dir():
            return 0
        cutoff = time.time() - self.upload_ttl_seconds
        removed = 0
        for path in self.upload_dir.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.debug("Could not sweep %s: %s", path, exc)
        if removed:
            logger.info("RefStore sweep removed %d expired uploads", removed)
        return removed


def _guess_extension(filename: str | None, content_type: str | None) -> str | None:
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        if base in _MIME_TO_EXT:
            return _MIME_TO_EXT[base]
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in _ALLOWED_EXTS:
            return suffix
    return None


def _looks_like_hex(s: str) -> bool:
    return 8 <= len(s) <= 64 and all(c in "0123456789abcdef" for c in s)


def _is_within(path: Path, root: Path) -> bool:
    """True iff `path` is the same as or a descendant of `root` (both resolved)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
