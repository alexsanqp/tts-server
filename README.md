# tts-server

A self-hosted, pluggable Text-to-Speech HTTP service. OpenAI-compatible
request shape, Wyoming-style capability introspection, drop-in provider
plugins. One HTTP endpoint for every TTS engine you wire in.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-125%20passing-success.svg)](#development)

---

## Why

You probably already use several TTS engines: a cloud one for the long tail of
languages, a small fast model on GPU for your main locale, and maybe a
voice-cloning model for a hero voice. Each one ships its own SDK, its own
voice catalog, its own quirks. **tts-server** is one HTTP service that hides
all of that behind a single OpenAI-shaped endpoint.

- One **`POST /v1/audio/speech`** — works for any registered provider.
- One **`GET /v1/models`** — every provider's capabilities, advertised honestly.
- **Per-request model selection** (`"model": "edge"`) or **automatic routing by
  language** (`"model": "auto"` + a config table).
- Voice cloning via **uploaded reference audio** (`ref:<id>`), shared across
  providers that support it.
- **Pluggable**: add a new engine by implementing a 4-method Protocol and
  registering it.

## Features

- **OpenAI-compatible** synthesis endpoint (`/v1/audio/speech`). Drop-in for
  clients that already speak OpenAI TTS.
- **Built-in providers**: [edge-tts](https://github.com/rany2/edge-tts)
  (Microsoft cloud, free), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
  (local GPU voice cloning — both `0.6B` and `1.7B` checkpoints supported
  side-by-side, runs as a sidecar subprocess so reloads don't cost a model
  reload), [StyleTTS2-UK](https://github.com/patriotyk/styletts2_inference)
  (Ukrainian local GPU).
- **Honest voice catalog**: voices are probed at startup; retired or
  unreachable voices are flagged unavailable instead of failing at synthesis.
- **Content-addressable cache**: re-synthesizing the same text twice doesn't
  re-spend GPU minutes. SHA-256 of inputs by default; pass `idempotency_key`
  for explicit dedup.
- **Reference-audio uploads** (`POST /v1/refs`): bring your own voice, get
  back a stable `ref:<id>`, use it in synthesis. Auth-gated, TTL-swept.
- **Production-aware**: per-provider concurrency semaphores, request
  timeouts, queue-depth backpressure with `Retry-After`, graceful shutdown.
- **Lean v1**: no streaming, no opus, no pitch/volume knobs — these are
  forward-compatible additions, not v1 surface area.

## Installation

```bash
git clone https://github.com/alexsanqp/tts-server.git
cd tts-server

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux / macOS

# Core install + the provider extras you want:
pip install -e ".[edge]"                       # edge-tts only (lightweight, no GPU)
pip install -e ".[edge,qwen]"                  # + Qwen3-TTS (needs CUDA)
pip install -e ".[all]"                        # everything
pip install -e ".[all,dev]"                    # + test tooling
```

Provider extras:

| Extra        | Pulls in                                         | Needs |
|--------------|--------------------------------------------------|-------|
| `edge`       | `edge-tts`, `langcodes`                          | Internet |
| `qwen`       | `qwen-tts`, `soundfile`                          | CUDA GPU — 0.6B: ~4 GB VRAM, 1.7B: ~6-8 GB VRAM |
| `styletts2`  | `styletts2-inference`, `ipa-uk`, `truststore`, … | CUDA GPU |

## Quick start

```bash
python -m tts_server                  # http://0.0.0.0:8880
# or:  tts-server --port 8881 --log-level debug
```

Synthesize. Defaults are tuned for maximum fidelity — lossless WAV at
24 kHz (the native rate of every shipped provider, so no resampling
happens on the hot path):

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "input": "Hello, world.",
        "language": "en",
        "voice": "en-US-AriaNeural",
        "model": "auto"
      }' \
  --output hello.wav
```

The response is raw audio with metadata in headers:

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-TTS-Provider: edge
X-TTS-Model: edge
X-Audio-Format: wav
X-Sample-Rate: 24000
X-Duration-Ms: 1820
X-Cache: miss
X-Request-Id: 1f3a…
```

Want a smaller payload (mp3) instead? Set ``response_format``:

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "input": "Hello, world.",
        "voice": "en-US-AriaNeural",
        "response_format": "mp3"
      }' \
  --output hello.mp3
```

Need JSON-with-base64 instead of raw bytes (browser clients, easier
metadata access)? Add `?envelope=json`:

```bash
curl -X POST 'http://localhost:8880/v1/audio/speech?envelope=json' \
  -H "Content-Type: application/json" \
  -d '{ "input": "Hi", "voice": "en-US-AriaNeural" }'
```

```json
{
  "audio_base64": "UklGRn...",
  "format": "wav",
  "sample_rate": 24000,
  "duration_ms": 1820,
  "provider": "edge",
  "model": "edge",
  "request_id": "1f3a..."
}
```

## API reference

| Method | Path                  | Purpose                                                    |
|--------|-----------------------|------------------------------------------------------------|
| `POST` | `/v1/audio/speech`    | Synthesize text → audio. Add `?envelope=json` for JSON.    |
| `GET`  | `/v1/models`          | List enabled providers, languages, voice-listing URLs.     |
| `GET`  | `/v1/voices`          | Voice catalog. Filter with `?model=` and `?language=`.     |
| `POST` | `/v1/refs`            | Upload reference audio for voice cloning. **Auth required.** |
| `GET`  | `/v1/refs/catalog`    | List baked-in reference voices.                            |
| `GET`  | `/v1/route`           | Preview routing for `?language=&model=auto`.               |
| `GET`  | `/healthz`            | Liveness.                                                  |
| `GET`  | `/readyz`             | Readiness — 503 if a required provider failed to load.     |

### `POST /v1/audio/speech` — request body

```jsonc
{
  "input": "Text to speak (required)",
  "model": "auto",            // or "edge" / "qwen3-0.6b" / "qwen3-1.7b" / "styletts2-uk" / ...
  "language": "en-US",        // BCP-47
  "voice": "en-US-AriaNeural",// or "ref:<id>" for cloning, or null for default
  "speed": 1.0,               // 0.25–4.0, 1.0 = native
  "response_format": "wav",   // "wav" (default, lossless) | "mp3"
  "sample_rate": 24000,       // 8000–48000, default 24000 = providers' native rate
  "idempotency_key": null     // optional; when set, bypasses the content hash
}
```

Defaults (omit a field to take its default):

| Field | Default | Why |
|---|---|---|
| `model` | `"auto"` | Route by `language` (see `[routing.by_language]`) |
| `speed` | `1.0` | Native cadence |
| `response_format` | `"wav"` | Lossless — no codec noise |
| `sample_rate` | `24000` | Matches every shipped provider's native rate; no resampling on the hot path |

### Audio quality and sample rate

Every provider in this repo (`fake`, `edge`, `qwen3-0.6b`, `qwen3-1.7b`,
`styletts2-uk`) emits audio at **24 000 Hz mono**. That's the ceiling of
information the model actually produces — anything higher in the API is
just upsampling.

| `sample_rate` | What happens | When to use |
|---|---|---|
| `8000` – `16000` | Downsample via ffmpeg. Audible loss of high frequencies. | Telephone-grade, very small files |
| `22050` | Slight downsample. Almost transparent on speech. | Compromise size vs quality |
| **`24000`** (default) | No resampling. Whatever the model produced, you get. | Max fidelity, fastest |
| `32000` – `48000` | Upsample via ffmpeg. **Pure interpolation — no extra fidelity from the model**. Bigger files, identical perceptual quality. | Pipeline that demands 48 kHz |

> **Note on Qwen model naming.** `Qwen3-TTS-12Hz-1.7B-Base` mentions
> `12Hz` — that's the **codec frame rate** (audio tokens per second
> before the vocoder), **not** the audio sample rate. The decoded
> waveform is 24 kHz.

What actually moves the needle on perceptual quality:

1. **Model size** — `qwen3-1.7b` consistently beats `qwen3-0.6b` on WER (~7-15 % lower) at ~3x VRAM. Use `1.7b` when you have the budget.
2. **`response_format: "wav"`** — keeps the model's full output. `mp3` adds codec noise; OK for delivery but not for downstream processing.
3. **Reference audio quality** (voice cloning only) — a clean 5-15 s mono clip of the target speaker, sample-rate ≥ 16 kHz, no music/reverb. The closer the ref-audio matches your desired output style, the better.
4. **`ref_text` exactly matches the reference audio**, word-for-word. Qwen conditions on the (audio, text) pair; mismatched text causes hallucinated phonemes.

### Voice cloning

For providers that support cloning (today: Qwen3-TTS):

1. `POST /v1/refs` with a small audio file → get `{"id": "ref:abc123…"}`.
2. Use that id as `"voice": "ref:abc123…"` in subsequent synthesis requests.
3. Uploads are kept for `refs.upload_ttl_hours` (default 24h), then swept.

#### Local catalog (operator-supplied voices)

`data/refs-catalog/` is where each deployment drops its own reference
clips. **Nothing audio-shaped ships in this repo** — voices and their
metadata are operator-specific (and may carry licensing constraints), so
the directory is gitignored apart from a single `example.json` schema
template.

Drop your files in with this layout:

```
data/refs-catalog/
├── example.json                ← schema reference, committed
├── <lang>-<name>.mp3           ← 5-15 s clean speech, one speaker
├── <lang>-<name>.json          ← sidecar metadata (same stem)
├── <lang>.mp3                  ← optional default-per-language form
├── <lang>.json
└── ...
```

Filename schema:

| Pattern | Resulting voice id | Use for |
|---|---|---|
| `<lang>-<name>.{mp3,wav}` | `ref:<lang>-<name>` | Named voices |
| `<lang>.{mp3,wav}` | `ref:<lang>-default` | One default voice per language |

`<lang>` must be a Qwen-supported BCP-47 primary tag (`en, de, fr, it, es,
ru, ja, ko, zh, pt`). Other tags get scanned but skipped — the file stays on
disk for future providers (XTTS, Coqui) that may use it.

#### Sidecar `.json` schema

Each audio file pairs with `<stem>.json` — see
[`data/refs-catalog/example.json`](data/refs-catalog/example.json) for a
working template:

```json
{
  "ref_text": "Exact transcript of the reference audio, word for word.",
  "gender": "male",
  "description": "Free-text label, surfaced via /v1/voices.",
  "role": "host"
}
```

| Field | Required | Purpose |
|---|---|---|
| `ref_text` | recommended | Sent to Qwen alongside the audio; mismatch → broken output |
| `gender` | optional | Surfaced in `/v1/voices` for client-side filtering |
| `description` | optional | Free-text in `/v1/voices` metadata |
| `role` | optional | Free-text (`host`, `sidekick`, `narrator`, …) |

If the sidecar is missing or `ref_text` is empty, the caller must pass
`ref_text` in the synthesis request body. Plain-stem files (e.g. `en.mp3`)
fall back to a built-in default text — handy for quick demos, not for
production.

> **Sourcing voices:** any clean recording of a single speaker works. Public
> domain catalogues (LibriVox), permissively licensed datasets (Common
> Voice), or your own recordings are all fine. Make sure the licence of
> your source permits use as a TTS conditioning signal before you ship.

## Configuration

`config/tts-server.toml` — TOML, overridable by env (`TTS_*`, double
underscore for nested keys, e.g. `TTS_SERVER__PORT=9000`). Source priority:
**env > TOML > defaults**.

### Secrets

**Never commit `auth_token` (or any secret) to the tracked
`config/tts-server.toml`.** The repo file ships with `auth_token = ""` as a
placeholder. Real tokens go in one of:

- **An env var** (recommended for systemd / Docker):
  ```bash
  export TTS_SERVER__AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
  python -m tts_server
  ```
- **A local TOML override** that's gitignored (`config/*.local.toml`, also
  `config/secrets.toml`):
  ```bash
  # config/tts-server.local.toml  (NOT in git)
  [server]
  auth_token = "..."
  ```
  ```bash
  python -m tts_server --config config/tts-server.local.toml
  ```
- **An `.env` file** (gitignored). See [`.env.example`](.env.example).

Rotate the token by generating a new one and pushing it via the same
channel (env var / local file) — no commit required.

```toml
[server]
host = "0.0.0.0"
port = 8880
auth_token = ""                  # empty disables auth on /v1/audio/speech etc.
                                 # /v1/refs ALWAYS requires it.
request_timeout_seconds = 120
max_queue_depth = 32

[refs]
catalog_dir = "data/refs-catalog"
upload_dir  = "data/refs"
upload_ttl_hours = 24
max_upload_mb = 10

[cache]
enabled = true
max_entries = 256

[providers]
enabled  = ["fake", "edge"]      # only these will be instantiated
required = []                    # failure of a required provider -> /readyz 503

[providers.edge]
default_voice = "en-US-AriaNeural"

[providers.qwen3-0.6b]
port = 8890                      # internal port for the Qwen sidecar
ref_audio_dir = "data/refs-catalog"
model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
device = "cuda:0"
startup_timeout_seconds = 180

[providers.qwen3-1.7b]           # opt-in: higher-quality 1.7B variant
port = 8891                      # distinct sidecar port from 0.6B
ref_audio_dir = "data/refs-catalog"
model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
device = "cuda:0"
startup_timeout_seconds = 300

[routing]
default = "fake"

[routing.by_language]            # used when model="auto"
en = "edge"
uk = "edge"
de = "edge"
# ...
```

## Adding a new provider

Three files, no core changes.

1. **Implement the Protocol** — `src/tts_server/providers/my_provider.py`:

```python
from tts_server.providers.base import (
    ProviderCapabilities, SynthesisRequest, SynthesisStream,
    TTSProvider, VoiceInfo,
)

class MyProvider:
    def __init__(self, options: dict | None = None) -> None:
        self._opts = options or {}

    async def describe(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            id="my-provider",
            provider_family="myfamily",
            languages=("en", "de"),
            voices=(VoiceInfo(id="amy", languages=("en",), gender="female"),),
            native_sample_rate=24000,
            native_format="wav",
            accepts_speed=True,
            is_gpu=True,
        )

    async def load(self) -> None:
        # Heavy imports + model warm-up go here (lazy).
        ...

    async def synthesize(self, request: SynthesisRequest) -> SynthesisStream:
        audio_bytes = self._do_synthesis(request.text, request.voice, request.speed)
        async def chunks():
            yield audio_bytes
        return SynthesisStream(
            sample_rate=24000, format="wav",
            duration_ms=len(audio_bytes) // 48,  # rough
            chunks=chunks(),
        )

    async def probe_voice(self, voice_id: str) -> bool:
        return voice_id == "amy"
```

2. **Register** — `src/tts_server/core/registry.py`, add one row:

```python
BUILTIN_PROVIDERS = {
    ...
    "my-provider": lambda opts: _lazy_import(
        "tts_server.providers.my_provider", "MyProvider"
    )(opts),
}
```

3. **Enable in config** — `config/tts-server.toml`:

```toml
[providers]
enabled = ["fake", "edge", "my-provider"]

[providers.my-provider]
# any options your provider reads
```

That's it. `/v1/models` now lists your provider; `/v1/audio/speech` with
`"model": "my-provider"` routes to it.

## Development

```bash
pip install -e ".[all,dev]"
pytest                                  # 125 tests (~1s; one StyleTTS2 GPU test ~30s when CUDA present)
pytest -m "not network and not gpu"     # default — skips env-gated tests
RUN_NETWORK_TESTS=1 pytest              # include real edge-tts calls
ruff check src tests
```

Test layout: `tests/test_smoke.py` covers the FastAPI app end-to-end with a
`FakeProvider`; `tests/test_*.py` cover each core module (`cache`, `refs`,
`auth`, `probing`); `tests/providers/test_*.py` cover each provider with
mocks plus optional real-network/real-GPU integration tests.

## Architecture at a glance

```
                          ┌─────────────────────────────┐
  client ── HTTP ───────► │  FastAPI app (uvicorn)      │
                          │                             │
                          │  ┌── api/speech.py          │
                          │  │   route → cache → semaphore → provider
                          │  │                          │
                          │  ├── api/models.py          │── /v1/models
                          │  ├── api/voices.py          │── /v1/voices
                          │  ├── api/refs.py            │── /v1/refs (auth-gated)
                          │  └── api/routing.py         │── /v1/route
                          │                             │
                          │  ProviderRegistry           │
                          │   ├ FakeProvider            │
                          │   ├ EdgeProvider ───────────┼─── Microsoft Edge TTS
                          │   ├ StyleTTS2UkProvider     │   (in-process GPU)
                          │   └ QwenProvider ─── HTTP ──┼──► qwen_worker.py
                          │                             │    (subprocess, GPU)
                          └─────────────────────────────┘
```

Single FastAPI worker; GPU providers get a `Semaphore(1)` so they serialize;
network providers default to `Semaphore(16)`. Qwen runs as a managed
subprocess so model reloads survive code edits.

## Roadmap (v2 candidates)

- **Streaming responses** (`?stream_format=audio|sse`) — `synthesize()` is
  already shaped as `AsyncIterator[bytes]` to make this non-breaking.
- **Server-side format transcoding** (`response_format` is currently a hint —
  v2 will honor it via ffmpeg).
- **Setuptools entry-point provider discovery** for out-of-tree providers.
- **Server-side loudness normalization** as an opt-in postprocess.
- More providers: Piper, XTTS-v2, Bark, OpenAI-passthrough.

## Contributing

Issues and PRs welcome. For new providers, follow the
[Adding a new provider](#adding-a-new-provider) walkthrough — the test
fixtures in `tests/providers/test_edge.py` and `tests/providers/test_qwen.py`
are good templates.

## License

[MIT](LICENSE).
