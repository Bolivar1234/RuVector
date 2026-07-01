# ADR-272: Association-Native Memory Layer (Calyx-inspired) for RuVector

**Status**: Accepted
**Date**: 2026-06-30
**Authors**: Claude Code MetaHarness Architect
**Supersedes**: None
**Extends**: ADR-269 (MRAgent Graph Memory over RuVector), ADR-270
(Self-Reconstructing Graph Memory)
**Related**: ADR-266 (MetaHarness Darwin ANN Integration), ADR-271 (Darwin/SONA
self-improvement), ADR-264 (Matryoshka coarse-fine search), ADR-268
(capability-gated ANN, SPANN partition spill)

---

## Context

RuVector today is best described as *fast vector memory plus graph-aware
association*: HNSW/IVF indexes, a self-learning GNN, min-cut association edges,
and graph memory (ADR-269/270). The dominant retrieval pattern it serves is
still the industry default:

```
input → one embedding → nearest neighbours → answer
```

Chris Royse's **Calyx** white paper and reference engine
(<https://github.com/ChrisRoyse/Calyx>, a pre-1.0 Rust project under the
Business Source License 1.1) argue that this single-embedding pattern is *too
lossy*. Its central claim: one input should **not** collapse into one flattened
vector. It should become a **constellation** — the same object measured through
many *frozen lenses* (a semantic embedder, a lexical index, a code model, a
domain/structural encoder, a temporal model, a sensor model …), with **each
lens kept as a distinct typed slot, never flattened**. Relationships are then
*derived between* the slots, grounded against real-world anchors, scored for how
much signal each lens adds, and gated so the system fails closed rather than
answering from "semantic fog".

Calyx organises this around **four verbs** — *measure, count, differentiate,
compose* — and eleven subsystems (`Aster` storage, `Forge` math, `Registry`
content-addressed lenses, `Sextant` fusion search, `Loom` cross-lens
associations, `Assay` signal-in-bits, `Lodestar` grounding kernels, `Ward`
fail-closed guard, `Ledger` hash-chained provenance, `Anneal` reversible
self-optimization, `Oracle` grounded prediction). Its three trust principles
are **grounding is mandatory**, **no flattening**, and **fail closed**.

This is not a competitor to RuVector — it is a **design pattern RuVector should
absorb**. It directly validates the project's "memory is the moat" thesis
(`CLAUDE.md`, MetaHarness): the embedding *model* is replaceable; the *measured
association substrate plus governance* is the product. It also composes with
Darwin Mode (ADR-266/271): MetaHarness already evolves planners, routers, and
memory policy — Calyx says it should also **route lenses**, not just models.

**Decision needed**: Add a first-class, RuVector-native **association memory
layer** — multi-slot constellations, lens manifests, cross-lens agreement, RRF
fusion, signal-density scoring, grounding anchors, a fail-closed guard, a
hash-chained provenance ledger, and a reversible weight optimizer — with a
deterministic Rust benchmark proving it beats single-embedding RAG on grounded
accuracy and unsupported-claim rate, and with a clear path to back each per-lens
ranking with RuVector's existing HNSW/IVF/GNN indexes.

### Licensing note (clean-room)

Calyx's reference engine is **BSL-1.1** licensed. The crate this ADR introduces,
`crates/ruvector-calyx`, is an **independent, clean-room implementation of the
published *architecture pattern*** (constellations, the four verbs, the trust
principles) — it does **not** copy or derive from Calyx's source. It is licensed
`MIT OR Apache-2.0` like the rest of RuVector and is dependency-free (an inline
SplitMix64 PRNG keeps benchmarks reproducible with no external crates).

---

## Decision

Ship `crates/ruvector-calyx`: an association-native memory layer that maps the
Calyx pattern onto RuVector primitives. The mapping the crate implements:

| Calyx concept       | RuVector translation (this crate)                              |
|---------------------|----------------------------------------------------------------|
| Constellation       | `Constellation` — one object, many typed slots, never flattened |
| Lens                | `LensManifest` — content-addressed (name, version, kind, dims)  |
| Slot                | A typed `Vec<f32>` keyed by lens name in `Constellation::slots`  |
| Cross-term (`Loom`) | `loom::{min,mean}_agreement`, `top_dissenter` — disagreement is signal |
| Signal (`Assay`)    | `assay::signal_density` — mutual-information bits per µs of cost |
| Fusion (`Sextant`)  | `fusion::weighted_rrf` — Reciprocal Rank Fusion across lenses    |
| Grounding (`Lodestar`) | `Anchor` (accepted-answer / passed-test / sensor-label / citation / reward) |
| Guard (`Ward`)      | `ward::adjudicate` → `GuardDecision::{Answer, Refuse(reason)}` (fail closed) |
| Provenance (`Ledger`) | `Ledger` — FNV-chained, replayable answer paths               |
| Self-opt (`Anneal`) | `anneal::anneal_weights` — reversible SA over fusion weights     |
| Panel routing       | per-lens fusion weights tuned by `Anneal` (lens routing)        |

### 1. No flattening — the constellation is the record

`Constellation` stores slots in a `BTreeMap<String, Vec<f32>>` keyed by lens.
The deliberately lossy `Constellation::flatten()` exists *only* to construct the
single-embedding baseline the layer is meant to beat. The `ConstellationStore`
validates every slot against a lens registry on insert and **fails closed** on
an unknown lens or a shape mismatch (`StoreError::{UnknownLens, ShapeMismatch}`).

### 2. The four verbs, wired in `CalyxEngine`

`measure → count → differentiate → compose`:

- **measure** — the query arrives already measured through the registry's frozen
  lenses (slots in, never flattened).
- **count** — one ranked list per lens (`ConstellationStore::search_lens`, exact
  here; HNSW/IVF in production) plus cross-lens agreement (`loom`).
- **differentiate** — per-lens fusion weights (tuned by `anneal`) bias the fusion
  toward high-signal-density lenses (`assay`).
- **compose** — `fusion::weighted_rrf` fuses the lists, `ledger::grounded_path`
  resolves grounding, `ward::adjudicate` returns a guarded decision, and the
  whole path is recorded as a replayable `Ledger` entry.

### 3. Fail closed, with structured refusals

`Ward` refuses — never silently guesses — on any shortfall, each with a typed
reason: `NoCandidate`, `LowFusionScore`, `CrossLensDisagreement`,
`NoGroundedPath`, `UncalibratedGuard`. The cross-lens guard uses the *minimum*
agreement across lenses, so a single dissenting lens (the high-value
disagreement signal) blocks the answer.

### 4. Grounding is mandatory and replayable

Every answer requires a grounded path (`GuardProfile::require_grounding`) above a
calibrated anchor weight. Every query appends a hash-chained `LedgerEntry`
(input · lenses · retrieval · output); `Ledger::verify()` replays the entire
chain, giving 100% lineage reproducibility and tamper-evidence.

### 5. Reversible lens routing (Anneal)

`anneal_weights` runs simulated annealing over the fusion weights to maximize
*accepted answers per unit cost* — "signal per dollar". Every rejected proposal
restores the prior weights exactly (reversibility), so the optimizer never
corrupts the deployed configuration. This is the retrieval analogue of
MetaHarness Darwin Mode (ADR-266): instead of "which model answers", it learns
"which lenses to consult and how to weight them".

---

## Benchmark (`calyx-bench`)

A deterministic, dependency-free benchmark (`cargo run --release -p
ruvector-calyx`) constructs a synthetic corpus engineered to expose the
flattening failure: 240 documents across 12 topics × 20 docs. Within a topic
every document shares the same **semantic** region ("semantic fog"); the
discriminating signal lives in two *minority* lenses — a per-doc **lexical**
fingerprint and a **structural** class. A single-embedding system sees only the
semantic vector and cannot tell the right document from its 19 topic twins; the
association-native store keeps the minority lenses as full-weight ranking lists
and fuses them. The query set adds **unanswerable** queries (right topic,
specifics matching no document) to test fail-closed abstention.

### Measured results (seed 42, x86_64)

| Metric                      | Single-embedding | Calyx multi-lens |
|-----------------------------|------------------|------------------|
| Grounded answer accuracy    | 6.7%             | **99.2%**        |
| Recall@10 (answerable)      | 51.7%            | **100.0%**       |
| Unsupported claims (count)  | 172              | **0**            |
| Abstentions (correct)       | 0                | 61               |
| Replayable provenance       | n/a              | **180/180 (100%)** |

`Assay` correctly ranks the cheap lexical lens highest by signal density
(0.11 bits/µs vs the expensive semantic lens at 0.01 bits/µs); `Anneal`
reversibly converges the fusion weights onto the high-density lens, *raising*
grounded accuracy to 100% while *dropping* the expensive semantic lens —
demonstrating the cost-shifting thesis (value moves from expensive model
intelligence to a measured substrate + cheap sufficient reasoning).

### Acceptance targets (all PASS)

| Target                                   | Result   |
|------------------------------------------|----------|
| Grounded answer accuracy ≥ +15 pp        | +92.5 pp |
| Unsupported claims ≥ −50%                | −100%    |
| Recall@10 ≥ +10 pp                       | +48.3 pp |
| Replayability 100%                       | 180/180  |
| Anneal improves utility & is reversible  | Δu=+46.2 (73 reverted moves) |

The large margins are a property of the *adversarial* corpus (within-topic
single-embedding accuracy is near the 1/20 chance floor by construction); they
demonstrate the *direction and mechanism*, not a claim about any real dataset.
The acceptance thresholds are the ADR's contract; a production validation
(ADR-267 protocol) should re-measure on real multi-lens corpora.

---

## Consequences

### Positive

- RuVector gains a first-class **association layer**: multi-slot records, lens
  manifests, cross-lens graphs, grounding anchors, signal-density routing,
  fail-closed guards, and a provenance ledger — "less like Pinecone, more like
  an AI memory operating system".
- Directly reinforces the **memory-is-the-moat** thesis and gives enterprise
  governance language (grounding mandatory, no flattening, fail closed) for
  Cognitum One deployments (manufacturing, telecom, elder-care, security).
- Composes with **Darwin/MetaHarness**: lens routing becomes another evolvable
  gene alongside model routing and memory policy.
- Dependency-free and deterministic → builds anywhere, reproducible benchmarks.

### Negative / risks

- The reference per-lens search is exact brute force. Production must back each
  lens ranking with RuVector HNSW/IVF (follow-up below); the API is shaped for
  that swap (`search_lens` → indexed retrieval).
- The provenance hash (FNV-1a) is non-cryptographic; a production ledger should
  use BLAKE3/SHA-256 with signed checkpoints (as Calyx's `Ledger` does).
- Mutual-information signal-density is a binned lower-bound proxy, adequate for
  ranking lenses but not a calibrated bits measurement.
- Multi-lens retrieval is ~3× the single-lens latency (3 lens searches + fusion
  + guard); `Anneal` mitigates by pruning low-density lenses.

### Follow-up work

1. **[done — see Update 1]** Adaptive lens routing + calibration.
2. Back `search_lens` with `ruvector-core` HNSW and `ruvector-spann` partitions.
3. Map the cross-lens `Loom` graph onto `ruvector-graph` / `ruvector-mincut`
   association edges so agreement/disagreement is itself searchable.
4. Swap FNV for BLAKE3 + signed ledger checkpoints.
5. Wire lens routing into MetaHarness Darwin as an evolvable gene (ADR-266).
6. Add a sensor/RF lens (RuView: Wi-Fi CSI, mmWave) to demonstrate non-text
   constellations and cross-modal disagreement detection.

---

## Update 1 — Adaptive routing + calibration (2026-06-30)

Sequencing note: **route → calibrate → stress-test → graph → HNSW.** Calibration
before speed, because speed optimizations (HNSW) can *hide* bad retrieval, and
calibration tells you *where* HNSW/graph even matter. This update lands the first
two.

### What shipped

- `calibrate.rs` — per-query confidence from four signals (score **margin**,
  **cross-lens agreement**, **grounding/source density**, **contradiction** =
  top dissenter gap; freshness is N/A in this corpus), a histogram
  **`Calibrator`** (reliability map: raw confidence → empirical accuracy),
  **Expected Calibration Error**, and **abstention precision/recall**.
- `routing.rs` — `route()` consults lenses **cheapest-first**, stops early once
  calibrated confidence clears `stop_threshold`, escalates to expensive lenses
  only when unsure, and abstains if the full panel is still not confident. A
  `min_lenses_to_answer` gate enforces **"don't trust a lone lens"** — never
  commit on a single lens's say-so. Every decision emits a `Witness` (lenses
  consulted, per-lens tops, signals, confidence, action, escalation reason,
  cost, latency).
- `calyx-routing-bench` — compares **brute-force / static / adaptive** on the
  multi-lens corpus (160 answerable + 80 unanswerable, 50/50 train/test; the
  calibrator is fit on train only).

### Measured results (test split, seed 2026)

| Mode | Grounded acc | Accepted acc | Cost µs | Latency µs | ECE | Abstain-P |
|------|-------------|--------------|---------|-----------|-----|-----------|
| brute-force (all lenses) | 93.8% | 100.0% | 15.5 | 15.5 | 0.000 | 88.9% |
| static (2 cheap lenses)  | 100.0% | 100.0% | 3.5 | 3.5 | 0.000 | 100.0% |
| **adaptive (calibrated)** | **100.0%** | **100.0%** | **7.5** | **7.5** | **0.000** | **100.0%** |

Adaptive **strictly dominates brute-force** — higher accuracy at less than half
the cost — because the expensive semantic lens is the low-signal "fog" lens here,
so always consulting it both costs more and can flip a correct answer. Adaptive
skips it when two cheap lenses corroborate, and escalates to it only for hard or
unanswerable queries (then abstains). A well-chosen *static* policy is even
cheaper here — an honest result, and precisely the kind of thing calibration and
signal-density analysis surface (you can only justify hard-coding "skip semantic"
*after* measuring that it is low-signal).

### Acceptance (adaptive vs brute-force — all PASS)

| Target | Result |
|--------|--------|
| Accuracy loss ≤ 1 pp | −6.2 pp (adaptive *higher*) ✓ |
| Query cost reduction ≥ 30% | −51.6% ✓ |
| Latency reduction ≥ 25% | −51.6% ✓ |
| ECE ≤ 0.08 | 0.000 ✓ |
| Abstention precision ≥ 0.80 | 100% ✓ |

Product claim this unlocks: *ruvector-calyx doesn't just retrieve — it knows
which memory lens to trust, when to escalate, and when to abstain, with a
calibrated confidence and a witness log for every decision.*

---

## References

- Royse 2026, "Calyx: An Association-Native Database and Its Path to
  Planetary-Scale Grounded Intelligence" — ResearchGate preprint (pub.
  408248277); reference engine <https://github.com/ChrisRoyse/Calyx> (Rust,
  edition 2024, BSL-1.1, pre-1.0).
- Royse 2026, "The Calculus of Association: A Formula for Artificial General
  Intelligence — Meaning Compression, Frozen Embedders as Designable Measurement
  Instruments, Derived Data Abundance, and Teleological Constellations" —
  ResearchGate preprint (pub. 405933676).
- Cormack, Clarke & Buettcher 2009, "Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods", SIGIR.
- ADR-269, ADR-270 (graph memory over RuVector); ADR-266, ADR-271 (Darwin/SONA
  self-improvement).
