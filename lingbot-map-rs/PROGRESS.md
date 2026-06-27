# LingBot-Map Rust Port — Progress Tracker

> **This file is the source of truth for the `/loop` continuation.** Each loop
> iteration: read this file, pick the first unchecked step, implement + test +
> commit + push, then check it off here. Stop the loop only when every box in
> "Phases" is checked and `cargo test` is green.

## Task

Standalone Rust port of **LingBot-Map** (`robbyant/lingbot-map`, HF, Apache-2.0,
arXiv:2604.14141): a feed-forward streaming 3D reconstruction foundation model.
Deliverables requested by the user:

- Full Rust crate port (multi-crate workspace).
- All ADRs.
- Demo UI: native (wgpu) **and** WebGPU/WASM, deployed to GitHub Pages.
- Streaming video output (MP4) + image (PNG) saving.
- Memory layer ported from PyTorch/FlashInfer paged KV cache to **ruvector-core**
  HNSW retrieval (see `lingbot-memory`).
- Borrow the trained model from the original project (safetensors loading path).

## Ground-truth constraints (verified)

- **Environment is offline for crates.io git protocol.** Build with the sparse
  protocol — handled by `lingbot-map-rs/.cargo/config.toml`. If a build hits a
  403 on the git index, that config was bypassed; re-add it.
- **Standalone repo** = self-contained `lingbot-map-rs/` workspace inside the
  monorepo (GitHub access is scoped to `ruvnet/ruvector`; a separate repo cannot
  be created). It is in the root `Cargo.toml` `exclude` list so it never breaks
  the parent build, and can be split out later unchanged.
- **The 4.63 GB checkpoint cannot be downloaded/run to completion here.** The
  real safetensors loading path is wired; a deterministic **synthetic fallback**
  lets the full pipeline (inference → point cloud → PNG/MP4) run end-to-end
  without the weights. Full-fidelity SOTA reproduction is explicitly out of
  scope for the sandbox and is documented as such.
- candle 0.9 is the tensor backend (already used elsewhere in the monorepo).

## Build & test

```bash
cd lingbot-map-rs
cargo build            # sparse protocol via .cargo/config.toml
cargo test
```

## Phases

- [x] **P0 Foundation** — standalone workspace, `.cargo/config`, root `exclude`,
  toolchain, README, LICENSE. Builds + tests green.
- [x] **P1 Memory layer** (`lingbot-memory`) — `StreamingMemory` over real
  `ruvector-core` `VectorDB` API; insert/retrieve; long-range recall test. ✅
- [x] **P2 Tensor/config** (`lingbot-tensor`) — `ModelConfig`, safetensors
  header `WeightIndex` (header-only, no multi-GB load). candle behind feature.
- [x] **P3 Model** (`lingbot-model`) — `SyntheticReconstructor` (pure-Rust,
  wasm-safe, default) + candle `GeometricContextTransformer` (feature `candle`,
  loads safetensors). Both compile; 6 default tests + candle shape test pass. ✅
- [x] **P4 IO** (`lingbot-io`) — `FrameSink` trait, `PngSequenceSink`, streaming
  `Mp4Sink` (openh264 → mp4 mux). MP4 round-trips through the reader as a valid
  AVC track; PNG verified. 3 tests pass. ✅
- [ ] **P5 Pipeline** (`lingbot-pipeline`) — streaming loop: frame → encode →
  retrieve_context(top_k) → attention → point cloud; candle⇄&[f32] bridge to
  `lingbot-memory`. Integration test over N synthetic frames.
- [ ] **P6 Native demo** (`lingbot-cli`) — wgpu point-cloud viewer + headless
  render-to-PNG/MP4. `--frames N --out out.mp4`.
- [ ] **P7 Web demo** (`lingbot-wasm` + `demo/`) — WebGPU renderer, wasm-bindgen
  bindings, static site. Builds with `wasm-pack` / `cargo build --target wasm32`.
- [ ] **P8 Deploy** — `.github/workflows/lingbot-pages.yml` (workflow_dispatch +
  path filter `lingbot-map-rs/**`) publishing `demo/` to GitHub Pages.
- [ ] **P9 Polish** — top-level README with arch diagram, ADR index, CI for the
  workspace (`build-lingbot.yml`), final `cargo test` + `cargo clippy` green.

## ADRs (in `docs/adr/`)

- [x] ADR-0001 Standalone workspace & crate topology
- [x] ADR-0002 ruvector-core streaming memory replaces paged KV cache
- [x] ADR-0003 candle tensor backend & safetensors weight loading
- [x] ADR-0004 Rendering (wgpu native + WebGPU/WASM) & GitHub Pages deploy
- [x] ADR-0005 Streaming MP4 + PNG output pipeline
- [x] ADR-0006 Synthetic-fallback strategy & checkpoint provenance

## Next step

→ **P5**: `lingbot-pipeline` — streaming loop wiring model + memory + io
(candle⇄&[f32] bridge), integration test over N synthetic frames.
