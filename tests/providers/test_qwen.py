"""Unit tests for the Qwen3-TTS proxy provider.

All tests run **without** a real subprocess and **without** loading the
~1.5 GB Qwen model: subprocess.Popen, httpx, and asyncio.sleep are mocked.
"""

from __future__ import annotations

import io
import struct
import wave
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tts_server.core.errors import ProviderFailure, ProviderUnavailable
from tts_server.providers.base import (
    ProviderCapabilities,
    SynthesisRequest,
    VoiceInfo,
)
from tts_server.providers.qwen import (
    REF_TEXTS,
    QwenProvider,
    _parse_sample_rate,
    _wav_duration_ms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(sample_rate: int = 24000, n_samples: int = 4800) -> bytes:
    """Build a tiny silent WAV (default 200 ms @ 24 kHz)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
    return buf.getvalue()


def _make_request(
    *,
    text: str = "hello",
    language: str = "en",
    voice: str | None = None,
    voice_kind: str = "none",
    ref_text: str | None = None,
) -> SynthesisRequest:
    return SynthesisRequest(
        text=text,
        language=language,
        voice=voice,
        voice_kind=voice_kind,
        ref_text=ref_text,
        speed=1.0,
        target_sample_rate=None,
        target_format="wav",
    )


async def _drain(stream) -> bytes:
    out = b""
    async for chunk in stream.chunks:
        out += chunk
    return out


class _FakeResponse:
    """Lightweight stand-in for httpx.Response."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: dict[str, Any] | None = None,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
        raise_invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.content = content
        self.text = text
        self.headers = httpx.Headers(headers or {})
        self._raise_invalid_json = raise_invalid_json

    def json(self) -> Any:
        if self._raise_invalid_json:
            raise ValueError("not json")
        return self._json


class _FakeClient:
    """Minimal AsyncClient stub: AsyncMock get/post against a queue of responses."""

    def __init__(
        self,
        *,
        get_responses: list[Any] | None = None,
        post_responses: list[Any] | None = None,
    ) -> None:
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self._get_q = list(get_responses or [])
        self._post_q = list(post_responses or [])

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def get(self, url: str) -> Any:
        self.get_calls.append(url)
        if not self._get_q:
            raise AssertionError(f"unexpected GET {url}")
        item = self._get_q.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def post(self, url: str, json: dict[str, Any]) -> Any:
        self.post_calls.append((url, json))
        if not self._post_q:
            raise AssertionError(f"unexpected POST {url}")
        item = self._post_q.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_async_client(client_instances: list[_FakeClient]):
    """Return a context manager that swaps httpx.AsyncClient in qwen module.

    Each `httpx.AsyncClient(...)` call returns the next prepared _FakeClient.
    """
    queue = list(client_instances)

    def factory(*_args: Any, **_kwargs: Any) -> _FakeClient:
        if not queue:
            raise AssertionError("no fake AsyncClient prepared")
        return queue.pop(0)

    return patch("tts_server.providers.qwen.httpx.AsyncClient", side_effect=factory)


@pytest.fixture(autouse=True)
def _zero_asyncio_sleep():
    """Disable the 2 s polling backoff so load() tests run fast."""
    async def _no_sleep(_: float) -> None:
        return None

    with patch("tts_server.providers.qwen.asyncio.sleep", new=_no_sleep):
        yield


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_wav_duration_ms_parses_valid_wav() -> None:
    audio = _make_wav(sample_rate=24000, n_samples=4800)  # 200 ms
    assert _wav_duration_ms(audio) == 200


def test_wav_duration_ms_returns_zero_on_garbage() -> None:
    assert _wav_duration_ms(b"not a wav") == 0


def test_parse_sample_rate_prefers_header() -> None:
    headers = httpx.Headers({"x-sample-rate": "16000"})
    assert _parse_sample_rate(headers, b"") == 16000


def test_parse_sample_rate_falls_back_to_wav_header() -> None:
    audio = _make_wav(sample_rate=22050)
    headers = httpx.Headers()
    assert _parse_sample_rate(headers, audio) == 22050


def test_parse_sample_rate_final_default_for_garbage() -> None:
    headers = httpx.Headers()
    assert _parse_sample_rate(headers, b"junk") == 24000


# ---------------------------------------------------------------------------
# describe() / probe_voice() — must NOT spawn subprocess
# ---------------------------------------------------------------------------


async def test_describe_returns_static_capabilities(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})

    # Belt-and-suspenders: blow up if anything tries to spawn the worker.
    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        caps = await provider.describe()

    assert isinstance(caps, ProviderCapabilities)
    assert caps.id == "qwen3-0.6b"
    assert caps.provider_family == "qwen"
    assert caps.languages == ("en", "de", "fr", "it", "es", "ru", "ja", "ko", "zh", "pt")
    assert caps.supports_voice_id is False
    assert caps.supports_voice_cloning is True
    assert caps.native_sample_rate == 24000
    assert caps.native_format == "wav"
    assert caps.max_text_length == 1000
    assert caps.accepts_speed is False
    assert caps.is_gpu is True
    assert caps.is_remote is True
    assert caps.voices == ()  # empty dir


async def test_provider_id_drives_caps_and_defaults(tmp_path) -> None:
    """provider_id from registry picks the HF checkpoint and sidecar port.

    Both Qwen variants share QwenProvider; the registry injects provider_id
    into ``opts`` so multiple QwenProvider instances can coexist with
    distinct models/ports/logs.
    """
    p06 = QwenProvider(options={
        "provider_id": "qwen3-0.6b",
        "ref_audio_dir": str(tmp_path),
        "log_dir": str(tmp_path),
    })
    p17 = QwenProvider(options={
        "provider_id": "qwen3-1.7b",
        "ref_audio_dir": str(tmp_path),
        "log_dir": str(tmp_path),
    })

    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        caps_06 = await p06.describe()
        caps_17 = await p17.describe()

    # caps.id reflects the registry id, not a hardcoded string.
    assert caps_06.id == "qwen3-0.6b"
    assert caps_17.id == "qwen3-1.7b"

    # Defaults per-variant: model checkpoint + sidecar port + log file.
    assert p06._model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert p17._model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert p06._port == 8890
    assert p17._port == 8891
    assert p06._log_path != p17._log_path
    assert "qwen3-0.6b" in p06._log_path.name
    assert "qwen3-1.7b" in p17._log_path.name


async def test_explicit_options_override_variant_defaults(tmp_path) -> None:
    """Explicit port / model_name in options beats the variant default."""
    provider = QwenProvider(options={
        "provider_id": "qwen3-1.7b",
        "ref_audio_dir": str(tmp_path),
        "log_dir": str(tmp_path),
        "port": 9999,
        "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    })
    assert provider._port == 9999
    assert provider._model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


async def test_unknown_provider_id_falls_back_to_0_6b_defaults(tmp_path) -> None:
    """Unknown provider_id still works: keeps caps.id but uses 0.6B defaults."""
    provider = QwenProvider(options={
        "provider_id": "qwen3-experimental",
        "ref_audio_dir": str(tmp_path),
        "log_dir": str(tmp_path),
    })
    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        caps = await provider.describe()
    assert caps.id == "qwen3-experimental"  # preserved
    assert provider._model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # 0.6B fallback
    assert provider._port == 8890


async def test_describe_scans_plain_stems_legacy_fallback(tmp_path) -> None:
    # Plain stems still work: en.mp3 → ref:en-default with REF_TEXTS fallback,
    # de.wav → ref:de-default with empty metadata. Junk files are ignored.
    (tmp_path / "en.mp3").write_bytes(b"\x00")
    (tmp_path / "uk.wav").write_bytes(b"\x00")  # Ukrainian: not in QWEN_LANGUAGES.
    (tmp_path / "de.wav").write_bytes(b"\x00")
    (tmp_path / "notes.txt").write_text("ignored")

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})

    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        caps = await provider.describe()

    voice_ids = {v.id for v in caps.voices}
    assert voice_ids == {"ref:en-default", "ref:de-default"}

    by_id = {v.id: v for v in caps.voices}
    en = by_id["ref:en-default"]
    assert isinstance(en, VoiceInfo)
    assert en.languages == ("en",)
    assert en.accepts_voice_id is False
    assert en.accepts_clone_ref is True
    assert en.metadata.get("ref_text") == REF_TEXTS["en"]

    # German plain stem: no sidecar, no REF_TEXTS entry → empty metadata.
    de = by_id["ref:de-default"]
    assert de.metadata == {}


async def test_describe_reads_json_sidecar_for_named_voices(tmp_path) -> None:
    """`<lang>-<name>.mp3` + `<lang>-<name>.json` becomes `ref:<lang>-<name>`."""
    import json

    (tmp_path / "en-owen.mp3").write_bytes(b"\x00")
    (tmp_path / "en-owen.json").write_text(
        json.dumps(
            {
                "ref_text": "Sample transcript for cloning.",
                "gender": "male",
                "description": "Primary host voice.",
                "role": "host",
                "language": "en",  # ignored; primary tag comes from the stem
            }
        ),
        encoding="utf-8",
    )

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    caps = await provider.describe()

    assert len(caps.voices) == 1
    v = caps.voices[0]
    assert v.id == "ref:en-owen"
    assert v.languages == ("en",)
    assert v.gender == "male"
    assert v.metadata["ref_text"] == "Sample transcript for cloning."
    assert v.metadata["description"] == "Primary host voice."
    assert v.metadata["role"] == "host"


async def test_describe_named_voice_without_sidecar_has_empty_metadata(tmp_path) -> None:
    """No sidecar → voice still appears, but ref_text must come from caller."""
    (tmp_path / "en-anon.mp3").write_bytes(b"\x00")

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    caps = await provider.describe()

    assert len(caps.voices) == 1
    v = caps.voices[0]
    assert v.id == "ref:en-anon"
    assert v.languages == ("en",)
    assert v.metadata == {}  # no fallback for named voices — sidecar is required
    assert v.gender is None


async def test_describe_ignores_malformed_sidecar(tmp_path) -> None:
    (tmp_path / "en-x.mp3").write_bytes(b"\x00")
    (tmp_path / "en-x.json").write_text("not valid json {{{", encoding="utf-8")

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    caps = await provider.describe()

    assert len(caps.voices) == 1
    assert caps.voices[0].id == "ref:en-x"
    assert caps.voices[0].metadata == {}


async def test_describe_skips_unsafe_stems(tmp_path) -> None:
    """Stems with disallowed chars (paths, dots, uppercase) get logged + skipped."""
    (tmp_path / "Foo.mp3").write_bytes(b"\x00")        # uppercase rejected
    (tmp_path / "en.bad.stem.mp3").write_bytes(b"\x00")  # extra dots rejected
    (tmp_path / "en-OK_name.mp3").write_bytes(b"\x00")  # lowercase + underscore OK

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    caps = await provider.describe()

    ids = {v.id for v in caps.voices}
    # Only en-OK_name passes after lowercasing (regex permits underscore).
    assert ids == {"ref:en-ok_name"}


async def test_describe_dedupes_when_both_wav_and_mp3_exist(tmp_path) -> None:
    (tmp_path / "en.wav").write_bytes(b"\x00")
    (tmp_path / "en.mp3").write_bytes(b"\x00")

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    caps = await provider.describe()
    assert [v.id for v in caps.voices] == ["ref:en-default"]


async def test_describe_handles_missing_ref_dir(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    provider = QwenProvider(options={"ref_audio_dir": str(missing)})
    caps = await provider.describe()
    assert caps.voices == ()


async def test_probe_voice_membership_only_no_worker_contact(tmp_path) -> None:
    (tmp_path / "en.mp3").write_bytes(b"\x00")
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})

    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        # Even patch httpx to scream if touched.
        with patch(
            "tts_server.providers.qwen.httpx.AsyncClient",
            side_effect=AssertionError("must not contact worker"),
        ):
            assert await provider.probe_voice("ref:en-default") is True
            assert await provider.probe_voice("ref:zz-default") is False


# ---------------------------------------------------------------------------
# load() — Popen args, polling backoff, success and failure modes
# ---------------------------------------------------------------------------


async def test_load_spawns_subprocess_with_expected_args(tmp_path) -> None:
    provider = QwenProvider(
        options={
            "port": 9999,
            "host": "127.0.0.1",
            "ref_audio_dir": str(tmp_path),
            "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "device": "cuda:1",
        }
    )

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # still alive

    # Pre-check: connect error → load() proceeds to spawn.
    precheck = _FakeClient(get_responses=[httpx.ConnectError("nope")])
    # Poll loop: two not-ready responses, then ready.
    poll = _FakeClient(
        get_responses=[
            _FakeResponse(status_code=503, json_body={"model_loaded": False}),
            _FakeResponse(status_code=200, json_body={"model_loaded": False}),
            _FakeResponse(status_code=200, json_body={"model_loaded": True}),
        ]
    )

    with patch("subprocess.Popen", return_value=fake_proc) as popen, \
            _patch_async_client([precheck, poll]):
        await provider.load()

    # Subprocess invocation
    popen.assert_called_once()
    call_args, call_kwargs = popen.call_args
    cmd = call_args[0]
    assert cmd[1:] == [
        "-m", "tts_server.sidecars.qwen_worker",
        "--port", "9999",
        "--host", "127.0.0.1",
        "--preload",
        "--ref-audio-dir", str(tmp_path),
    ]
    # cmd[0] is sys.executable — just check it's a non-empty string
    assert isinstance(cmd[0], str) and cmd[0]

    env = call_kwargs.get("env", {})
    assert env["QWEN_MODEL"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"

    # Polling actually polled three times against the right URL
    assert len(poll.get_calls) == 3
    assert poll.get_calls[0] == "http://127.0.0.1:9999/health"


async def test_load_polls_until_model_loaded_then_returns(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path), "port": 8890})

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    # Pre-check fails (so spawn happens), then poll loop sees network error, then
    # http-but-not-ready, then ready.
    precheck = _FakeClient(get_responses=[httpx.ConnectError("nope")])
    poll = _FakeClient(
        get_responses=[
            httpx.ConnectError("connection refused"),
            _FakeResponse(status_code=200, json_body={"model_loaded": False}),
            _FakeResponse(status_code=200, json_body={"model_loaded": True}),
        ]
    )

    with patch("subprocess.Popen", return_value=fake_proc), \
            _patch_async_client([precheck, poll]):
        await provider.load()

    assert len(poll.get_calls) == 3


async def test_load_raises_provider_unavailable_when_subprocess_exits(tmp_path) -> None:
    provider = QwenProvider(
        options={"ref_audio_dir": str(tmp_path), "startup_timeout_seconds": 30.0},
    )

    fake_proc = MagicMock()
    # Pre-check returns alive=True (poll returns None there is no second poll() call
    # before the wait loop reads it), but inside the wait loop poll() returns 1
    # the very first time → ProviderUnavailable.
    fake_proc.poll.return_value = 1

    precheck = _FakeClient(get_responses=[httpx.ConnectError("nope")])
    poll = _FakeClient(get_responses=[])  # poll loop never reaches an HTTP call

    with patch("subprocess.Popen", return_value=fake_proc), \
            _patch_async_client([precheck, poll]):
        with pytest.raises(ProviderUnavailable, match="exited during startup"):
            await provider.load()


async def test_load_raises_provider_unavailable_on_timeout(tmp_path) -> None:
    provider = QwenProvider(
        options={
            "ref_audio_dir": str(tmp_path),
            "startup_timeout_seconds": 0.0,  # deadline already passed → first iter fails
        }
    )

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # subprocess alive but not ready

    precheck = _FakeClient(get_responses=[httpx.ConnectError("nope")])
    poll = _FakeClient(
        get_responses=[
            _FakeResponse(status_code=200, json_body={"model_loaded": False}),
        ]
    )

    # _terminate_proc uses run_in_executor → also needs to "succeed" without blocking.
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc), \
            _patch_async_client([precheck, poll]):
        with pytest.raises(ProviderUnavailable, match="did not become healthy"):
            await provider.load()


async def test_load_is_idempotent_when_already_healthy(tmp_path) -> None:
    """Second load() call should skip spawn + skip polling when worker is healthy."""
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    # 1st load: pre-check fails (connect error) → spawn → ready on first poll.
    # 2nd load: pre-check succeeds → return immediately, no further Popen calls.
    first_precheck = _FakeClient(get_responses=[httpx.ConnectError("nope")])
    first_poll = _FakeClient(
        get_responses=[_FakeResponse(status_code=200, json_body={"model_loaded": True})]
    )
    second_precheck = _FakeClient(
        get_responses=[_FakeResponse(status_code=200, json_body={"model_loaded": True})]
    )

    with patch("subprocess.Popen", return_value=fake_proc) as popen, \
            _patch_async_client([first_precheck, first_poll, second_precheck]):
        await provider.load()
        await provider.load()

    assert popen.call_count == 1, "second load() should not spawn another subprocess"


async def test_load_raises_provider_unavailable_when_popen_fails(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})

    # First call is the pre-check (must fail so load() proceeds to spawn).
    fake_client = _FakeClient(get_responses=[httpx.ConnectError("nope")])

    with patch(
        "subprocess.Popen", side_effect=OSError("python missing")
    ), _patch_async_client([fake_client]):
        with pytest.raises(ProviderUnavailable, match="Failed to launch"):
            await provider.load()


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------


async def test_synthesize_returns_wav_stream_with_parsed_metadata(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path), "port": 8890})
    wav_bytes = _make_wav(sample_rate=24000, n_samples=24000)  # 1 second

    response = _FakeResponse(
        status_code=200,
        content=wav_bytes,
        headers={"x-sample-rate": "24000", "content-type": "audio/wav"},
    )
    fake_client = _FakeClient(post_responses=[response])

    with _patch_async_client([fake_client]):
        stream = await provider.synthesize(_make_request(text="hi", language="en"))

    assert stream.format == "wav"
    assert stream.sample_rate == 24000
    assert stream.duration_ms == 1000
    body = await _drain(stream)
    assert body == wav_bytes

    # Body shape
    assert fake_client.post_calls[0][0] == "http://127.0.0.1:8890/synthesize"
    sent = fake_client.post_calls[0][1]
    assert sent == {"text": "hi", "language": "en"}


async def test_synthesize_forwards_clone_ref_and_ref_text(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    wav_bytes = _make_wav()

    response = _FakeResponse(status_code=200, content=wav_bytes, headers={"x-sample-rate": "24000"})
    fake_client = _FakeClient(post_responses=[response])

    req = _make_request(
        voice="/abs/path/to/ref.wav",
        voice_kind="clone_ref",
        ref_text="This is the reference text.",
        language="de",
    )

    with _patch_async_client([fake_client]):
        await provider.synthesize(req)

    sent = fake_client.post_calls[0][1]
    assert sent == {
        "text": "hello",
        "language": "de",
        "ref_audio": "/abs/path/to/ref.wav",
        "ref_text": "This is the reference text.",
    }


async def test_synthesize_uses_auto_language_when_empty(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    response = _FakeResponse(
        status_code=200, content=_make_wav(), headers={"x-sample-rate": "24000"}
    )
    fake_client = _FakeClient(post_responses=[response])

    with _patch_async_client([fake_client]):
        await provider.synthesize(_make_request(language=""))

    assert fake_client.post_calls[0][1]["language"] == "auto"


async def test_synthesize_does_not_send_ref_for_voice_kind_id(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    response = _FakeResponse(
        status_code=200, content=_make_wav(), headers={"x-sample-rate": "24000"}
    )
    fake_client = _FakeClient(post_responses=[response])

    req = _make_request(voice="something", voice_kind="id")  # not clone_ref
    with _patch_async_client([fake_client]):
        await provider.synthesize(req)

    body = fake_client.post_calls[0][1]
    assert "ref_audio" not in body


async def test_synthesize_raises_provider_failure_on_sidecar_5xx(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    response = _FakeResponse(
        status_code=500,
        content=b"boom",
        text="model OOM on GPU",
    )
    fake_client = _FakeClient(post_responses=[response])

    with _patch_async_client([fake_client]):
        with pytest.raises(ProviderFailure, match="HTTP 500"):
            await provider.synthesize(_make_request())


async def test_synthesize_raises_provider_unavailable_on_network_error(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    fake_client = _FakeClient(post_responses=[httpx.ConnectError("refused")])

    with _patch_async_client([fake_client]):
        with pytest.raises(ProviderUnavailable, match="request failed"):
            await provider.synthesize(_make_request())


# ---------------------------------------------------------------------------
# teardown() / terminate()
# ---------------------------------------------------------------------------


async def test_teardown_is_noop_when_no_subprocess(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    # Should not raise — there's nothing to terminate.
    await provider.teardown()


async def test_teardown_terminates_running_subprocess(tmp_path) -> None:
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    fake_proc = MagicMock()
    # Alive when teardown starts, exits cleanly on terminate.
    fake_proc.poll.return_value = None
    fake_proc.wait.return_value = 0
    provider._proc = fake_proc

    await provider.teardown()

    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_called_once()
    fake_proc.kill.assert_not_called()
    assert provider._proc is None


async def test_teardown_kills_subprocess_after_grace_timeout(tmp_path) -> None:
    import subprocess as _sp

    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    # First wait (after terminate) times out → kill → second wait succeeds.
    fake_proc.wait.side_effect = [_sp.TimeoutExpired(cmd="qwen", timeout=10), 0]
    provider._proc = fake_proc

    await provider.teardown()

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()
    assert fake_proc.wait.call_count == 2


async def test_terminate_alias_calls_teardown(tmp_path) -> None:
    """The registry calls `terminate()` on shutdown; verify it's wired to teardown."""
    provider = QwenProvider(options={"ref_audio_dir": str(tmp_path)})
    # Both must resolve to the same underlying function — bound methods compare
    # equal when they wrap the same descriptor, even if not `is`-identical.
    assert provider.terminate == provider.teardown
    # And both must be callable — calling teardown() without a subprocess is a no-op.
    await provider.terminate()
