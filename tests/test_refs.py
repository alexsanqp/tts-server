"""RefStore tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tts_server.core.refs import RefStore, RefStoreError


def _wav_bytes() -> bytes:
    # Minimal valid RIFF/WAVE header + 1 frame of silence (mono, 8kHz, 8-bit PCM)
    return (
        b"RIFF" + (44).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")  # PCM
        + (1).to_bytes(2, "little")  # mono
        + (8000).to_bytes(4, "little")
        + (8000).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (8).to_bytes(2, "little")
        + b"data" + (1).to_bytes(4, "little") + b"\x80"
    )


@pytest.fixture
def store(tmp_path: Path) -> RefStore:
    catalog = tmp_path / "catalog"
    uploads = tmp_path / "uploads"
    catalog.mkdir()
    return RefStore(
        catalog_dir=catalog,
        upload_dir=uploads,
        upload_ttl_hours=1,
        max_upload_mb=1,
    )


async def test_store_upload_returns_ref_id(store: RefStore) -> None:
    res = await store.store(content=_wav_bytes(), filename="x.wav", content_type="audio/wav")
    assert res.ref_id.startswith("ref:")
    assert res.path.is_file()
    assert res.size_bytes > 0


async def test_store_rejects_empty(store: RefStore) -> None:
    with pytest.raises(RefStoreError):
        await store.store(content=b"", filename="x.wav", content_type="audio/wav")


async def test_store_rejects_oversize(store: RefStore) -> None:
    too_big = b"\x00" * (2 * 1024 * 1024)  # 2 MiB > 1 MiB limit
    with pytest.raises(RefStoreError):
        await store.store(content=too_big, filename="x.wav", content_type="audio/wav")


async def test_store_rejects_unknown_type(store: RefStore) -> None:
    with pytest.raises(RefStoreError):
        await store.store(content=b"abc", filename="x.txt", content_type="text/plain")


async def test_store_dedupes_same_bytes(store: RefStore) -> None:
    a = await store.store(content=_wav_bytes(), filename="a.wav", content_type="audio/wav")
    b = await store.store(content=_wav_bytes(), filename="b.wav", content_type="audio/wav")
    assert a.ref_id == b.ref_id  # sha256 collision = same id
    assert a.path == b.path


def test_resolve_returns_none_for_missing(store: RefStore) -> None:
    assert store.resolve("ref:nope") is None
    assert store.resolve("ref:abcdef123456") is None
    assert store.resolve("not-a-ref") is None


@pytest.mark.parametrize(
    "ref_id",
    [
        "ref:../etc-default",
        "ref:..\\windows-default",
        "ref:/etc/passwd-default",
        "ref:foo/bar-default",
        "ref:.hidden-default",
        "ref:-leading-dash-default",
        "ref:UPPERCASE-default",  # whitelist is lowercase only
        "ref:" + "a" * 200 + "-default",  # too long
        "ref:..-default",
    ],
)
def test_resolve_blocks_traversal_and_unsafe_slugs(store: RefStore, ref_id: str) -> None:
    assert store.resolve(ref_id) is None


def test_resolve_finds_catalog_entry(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "en.mp3").write_bytes(b"x")
    store = RefStore(catalog_dir=catalog, upload_dir=tmp_path / "u")

    p = store.resolve("ref:en-default")
    assert p is not None
    assert p.name == "en.mp3"


async def test_resolve_finds_upload(store: RefStore) -> None:
    res = await store.store(content=_wav_bytes(), filename="x.wav", content_type="audio/wav")
    found = store.resolve(res.ref_id)
    assert found == res.path


def test_catalog_ids_lists_distinct_stems(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "en.mp3").write_bytes(b"x")
    (catalog / "uk.wav").write_bytes(b"x")
    (catalog / "en-owen.mp3").write_bytes(b"x")
    (catalog / "ignored.txt").write_text("nope")
    store = RefStore(catalog_dir=catalog, upload_dir=tmp_path / "u")

    ids = store.catalog_ids()
    assert "ref:en-default" in ids
    assert "ref:uk-default" in ids
    assert "ref:en-owen" in ids
    assert all(i.startswith("ref:") for i in ids)


def test_resolve_finds_named_catalog_entry(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "en-owen.mp3").write_bytes(b"x")
    store = RefStore(catalog_dir=catalog, upload_dir=tmp_path / "u")

    p = store.resolve("ref:en-owen")
    assert p is not None
    assert p.name == "en-owen.mp3"


def test_resolve_named_rejects_dotted_or_traversal(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    store = RefStore(catalog_dir=catalog, upload_dir=tmp_path / "u")
    for bad in [
        "ref:en-..",
        "ref:en-../etc",
        "ref:EN-Owen",       # uppercase rejected
        "ref:en-OWEN",       # uppercase rejected
        "ref:en-",           # missing name
        "ref:-owen",         # missing lang
    ]:
        assert store.resolve(bad) is None, f"resolved unsafe slug: {bad!r}"


async def test_sweep_removes_expired(tmp_path: Path) -> None:
    store = RefStore(
        catalog_dir=tmp_path / "c",
        upload_dir=tmp_path / "u",
        upload_ttl_hours=0,  # everything immediately expired
        max_upload_mb=1,
    )
    res = await store.store(content=_wav_bytes(), filename="x.wav", content_type="audio/wav")
    # Backdate the file just to be safe across filesystems with low mtime resolution
    import os

    past = time.time() - 1
    os.utime(res.path, (past, past))

    removed = store.sweep_once()
    assert removed == 1
    assert not res.path.exists()
