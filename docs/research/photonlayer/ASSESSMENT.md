# PhotonLayer — Assessment & Research Roadmap

> Strongest claim: **PhotonLayer is a deterministic optical AI front end where a learned phase mask
> performs task-specific analog preprocessing before a tiny digital decoder sees the compressed
> measurement.** This matters because the field is moving toward *meta-optic front ends + electronic
> back ends* for low-latency, low-power, privacy-preserving, compact sensing.

Companion to ADR-260 (optical computing simulator) and ADR-261 (mask exchange & determinism).
Measured numbers in this doc come from `photonlayer-bench` (`more_data_bench`, see the README table).

## Why it's differentiated

The unique angle is **not** "optical neural network" — it's **auditable optical compression for
task-useful sensing**. Most optical-AI narratives overclaim; PhotonLayer's wedge is:

1. **Task-first** — mask trained for the downstream objective, not generic reconstruction.
2. **Compression-first** — flagship 16×16 → 4 sensor pixels (64× pixel reduction; measured ~99% vs ~74% random).
3. **Privacy by physics** — verify/classify from a measurement that need not look like the scene.
4. **Deterministic receipts** — reproducible, BLAKE3-bound; suitable for regulated experiments and audit trails.
5. **Rust-native** — embedded, WASM, deterministic benchmarking, eventual hardware control.

## Best use cases (positioned by risk)

| Use case | Why it fits | Risk |
|---|---|---|
| Industrial inspection | Detect defects without full-frame processing | Low |
| Barcode / symbol / package verification | Strong demo path, easy ground truth | Low |
| Drone perception preprocessing | Lower bandwidth, smaller backend model | Medium |
| Scientific imaging | Task-useful measurement vs full capture | Medium |
| Medical imaging *research* | Compression, morphology classification, uncertainty | High |
| Consented identity verification | Strong privacy story if tightly bounded | High |
| Autonomous-vehicle sensing | Valuable but needs hardware + safety validation | Very high |

First commercial wedge: **industrial & scientific sensing**, not healthcare or AV. For medical/AV,
position as **research infrastructure and preprocessing**, not decision automation.

## What to prove next

### 1. Energy model
A measured/simulated energy comparison. Target: **equal-or-better accuracy with ≥10× lower digital
compute and ≥16× lower sensor bandwidth** vs a direct-image-plus-CNN pipeline (compare sensor pixels,
decoder params, MACs, latency, estimated energy).

### 2. Harder datasets
Move beyond synthetic: MNIST / Fashion-MNIST optical compression, CIFAR-10 binary subsets, MVTec-AD
industrial anomaly detection, a public microscopy cell-morphology set, and face *verification* on
consented pairs only (no identification gallery).

### 3. Reconstruction-attack suite
Quantify the privacy claim by publishing attacks: linear reconstruction, learned-decoder
reconstruction, diffusion-prior reconstruction, nearest-neighbour leakage, membership inference, and
attribute leakage (as *risk metrics only*). **"No readable image is stored" is a safer claim than
"privacy-preserving" until leakage is quantified.**

### 4. Hardware bridge
Software phase mask → printed static diffractive mask → SLM lab prototype → lensless camera module →
CMOS sensor integration → tunable metasurface. The credibility unlock is a physical path.

## Demos to build (for the Pages UI)

- **Optical privacy gate** — original face → noise-like measurement → verification result → failed
  reconstruction → receipt hash. Headline: *"The face was verified. The face was never stored."*
  (consented verification, **not** mass identification).
- **Microscope compressor** — cell image → learned compression → morphology class / anomaly score →
  uncertainty → reconstruction failure (no diagnostic claim). Headline: *"The microscope learned what
  not to measure."*
- **Drone vision front end** — full-frame baseline vs 4/8/16/32-pixel optical sensors → decision +
  latency/bandwidth comparison. Headline: *"The drone doesn't need the image. It needs the decision surface."*

## Products

| Product | Buyer | Value |
|---|---|---|
| PhotonLayer Studio | researchers, startups, labs | design & test optical AI masks |
| PhotonLayer Edge | industrial sensor companies | smaller models, lower bandwidth |
| PhotonLayer Verify | privacy-sensitive identity workflows | verification without storing readable images |

Near-term wedge: software + simulation + benchmark receipts. Long-term value: hardware co-design.

## Scoring

| Criterion | Score | Note |
|---|---:|---|
| Novelty | 9 | optical compression + Rust determinism + receipts + memory |
| Technical defensibility | 8 | good bounded claims; needs harder datasets |
| Viral potential | 9 | privacy gate + microscope compressor are highly visual |
| Commercial path | 7 | industrial sensing first, medical later |
| Safety posture | 8 | strong non-goal on surveillance; needs leakage testing |
| Hardware readiness | 5 | strong simulator; physical validation still required |

**Overall: 8.0 platform · 9.0 research demo · 7.0 near-term product.**

## Acceptance test (becomes hard to dismiss when)

> On **three public datasets**, a learned optical mask achieves within **2 pp** of full-image baseline
> accuracy while reducing sensor pixels by **≥16×**, digital MACs by **≥10×**, and reconstruction
> similarity below a documented privacy threshold.

## References

- Optical neural networks: progress and challenges — *Light: Science & Applications* (Nature, 2024).
- Metaoptics merging computational optics and electronics — PMC/NIH.
- Privacy-Aware Meta-Optics for Person Detection — *ACS Photonics* (2026).
- Target-depth sensing with metasurface-encoder integrated optoelectronic neural network — arXiv:2604.25160.
