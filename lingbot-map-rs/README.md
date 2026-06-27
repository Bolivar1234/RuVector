# lingbot-map-rs

A standalone **Rust port of [LingBot-Map](https://huggingface.co/robbyant/lingbot-map)** —
a feed-forward, streaming 3D reconstruction foundation model (arXiv:2604.14141,
Apache-2.0). This port replaces the Python/PyTorch + FlashInfer stack with a
pure-Rust [candle](https://github.com/huggingface/candle) tensor backend and an
HNSW trajectory-memory layer backed by
[`ruvector-core`](../crates/ruvector-core).

> **Status:** active port (see [`PROGRESS.md`](./PROGRESS.md)). The streaming
> memory layer, model config, and safetensors loading are implemented and
> tested; the model, IO, pipeline, and demo crates are landing iteratively.

## Why a Rust port?

| Concern | LingBot-Map (Python) | lingbot-map-rs |
|---|---|---|
| Tensor backend | PyTorch 2.9 | candle 0.9 (pure Rust, WASM-capable) |
| Long-range memory | FlashInfer **paged KV cache** in VRAM, forgets past a sliding window | **ruvector-core HNSW**, `O(log N)` retrieval, no VRAM ceiling |
| Deployment | Python + CUDA | native (wgpu) **and** WebGPU/WASM in the browser |

The headline change is the memory layer: instead of a linearly-growing KV cache
bounded by GPU VRAM, keyframe features stream into a lock-free HNSW index and the
transformer retrieves the top-K structurally similar past frames each step —
solving long-range drift without a forgetful sliding window. See
[ADR-0002](./docs/adr/ADR-0002-ruvector-streaming-memory.md).

## Crates

| Crate | Purpose |
|---|---|
| [`lingbot-memory`](./crates/lingbot-memory) | Streaming trajectory memory over `ruvector-core` (KV-cache replacement) |
| [`lingbot-tensor`](./crates/lingbot-tensor) | `ModelConfig` + safetensors header indexing + candle weight loading |
| `lingbot-model` *(landing)* | Geometric Context Transformer (candle) |
| `lingbot-io` *(landing)* | Frame sources, PNG export, streaming MP4 |
| `lingbot-pipeline` *(landing)* | Streaming inference orchestration |
| `lingbot-cli` *(landing)* | Native wgpu demo + headless render |
| `lingbot-wasm` *(landing)* | WebGPU/WASM browser demo |

## Build & test

```bash
cd lingbot-map-rs
cargo build
cargo test
```

The build uses the crates.io **sparse** protocol (configured in
`.cargo/config.toml`) for sandboxed/offline-friendly environments.

## The model checkpoint

The ~4.63 GB checkpoint is **borrowed from the upstream project at runtime**
(local path or HF download) and never committed. When weights are absent, a
deterministic synthetic fallback drives the full pipeline so the demo and
video/image export run end-to-end. See
[ADR-0003](./docs/adr/ADR-0003-candle-tensor-backend-and-weights.md).

## Architecture Decision Records

See [`docs/adr/`](./docs/adr). ADR-0001 (topology) · 0002 (memory) · 0003
(tensors/weights) · 0004 (rendering/deploy) · 0005 (MP4/PNG output).

## License

Apache-2.0, matching the upstream LingBot-Map model. See [`LICENSE`](./LICENSE).
This is an independent reimplementation; model weights remain property of the
original authors under their license.
