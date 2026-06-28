# Earth Pulse Observatory — `earth-pulse-harness`

A **MetaHarness Darwin-Mode** research pod for Earth's stable **~26-second microseism**
(the "26-second pulse") originating from the Gulf of Guinea, and the associated **gliding
tremors** documented by Bruland & Hadziioannou (2023).

> **The core idea: freeze the physics, evolve the harness.** Darwin Mode does **not**
> evolve the scientific truth. It evolves the *investigation workflow* around it — feature
> extraction, source-localization checks, ruVector embedding schemas, hypothesis scoring,
> anomaly tests, report generation, and failure detection — and keeps only the changes
> that *measurably* beat a baseline through a strict promotion gate.

This is a self-contained metaharness bundle, built in the same style as the sibling
`harnesses/timesfm-harness`.

## Why this is tractable

We do **not** try to "solve Earth's heartbeat." We turn it into a **bounded causal-discovery
benchmark**: move from *"Earth has a heartbeat"* to *"this mechanism predicts the pulse better
than all alternatives."* The first target is deliberately narrow:

> **Can the system predict the amplitude of the 26-second pulse from ocean-state variables,
> beating a seasonal baseline by ≥ 10 % on held-out months?**

A positive result is real but modest: *the pulse is predictably coupled to measurable planetary
state variables.* See `docs/research/benchmark-design.md`.

## Quick start

```bash
npm install
npm run doctor          # kernel + host adapter health check
npm test                # 17 tests: offline science pipeline + install smoke test
npm run build           # compile src/ -> dist/
npm run pipeline        # run detect->extract->embed->score over the bundled fixtures
```

## The pipeline (`src/`)

| File | Role |
|---|---|
| `detect-26s.ts` | DFT scan of the 24–28 s band → `PulseEvent` (period, amplitude, coherence, glide, confidence). |
| `extract-features.ts` | Spectral sub-band shape, amplitude envelope, glide slope, station geometry, environment context. |
| `embed-events.ts` | **Separate** L2-normalized waveform / environment / source embeddings (+combined), cosine NN search. |
| `score-hypotheses.ts` | Weighted discovery score + ranking of the candidate mechanisms, with killer contradictions. |
| `validate.ts` | The promotion gate: F1, false-positive, held-out error, citation grounding, leakage. |
| `pipeline.ts` | Wires the five stages together. |

All pipeline code is **deterministic and offline** — no network, no fabricated observations.

## Darwin Mode (`/evolve`)

```bash
npm run evolve          # real sandbox, deterministic mutator (no API key, no network)
npm run evolve:dry      # mock sandbox, fully offline dry run
# or pass flags straight through:
npx metaharness-darwin evolve . --generations 20 --children 8 --concurrency 4 --seed 26
```

**Evolvable surfaces** (`.metaharness/safety-policy.json`): detector band, feature schema,
embedding schema, retrieval strategy, scoring weights, validator/holdout strategy.

**Forbidden mutations:** fabricating observations, inventing citations, leaking test windows
into training, promoting a hypothesis without beating a baseline, or adding any new
import / network / filesystem / shell / env access.

**Promotion gate** (`.metaharness/objective.json`):

```
pulse_detection_f1 improves by >= 3%
AND false_positive_rate does not increase
AND held_out_prediction_error improves by >= 5%
AND every cited claim maps to a source document
AND no leakage from test windows into training windows
```

## Discovery score

```
score = 0.25 * source_stability
      + 0.20 * environmental_correlation
      + 0.20 * out_of_sample_prediction
      + 0.15 * contradiction_survival
      + 0.10 * mechanistic_plausibility
      + 0.10 * citation_grounding
```

## Candidate mechanisms (priors, not results)

| Hypothesis | Prior | Killer contradiction |
|---|---|---|
| Ocean shelf resonance | 0.72 | Strong pulses during calm-ocean windows |
| Coupled ocean + geology | 0.68 | Either factor alone explains everything |
| Water-column / bathymetric mode | 0.55 | Same geometry elsewhere lacks the signal |
| Volcanic / hydrothermal tremor | 0.46 | No thermal/gas/seismic volcanic proxy |
| Instrument artifact | 0.12 | Appears across independent global stations |

The bet to watch is **coupled ocean + geology**: ocean shelf resonance likely explains the
*carrier* frequency, while the gliding tremors may require a second mechanism.
See `docs/research/hypothesis-catalog.md`.

## Data spine (`data/`)

`seismic/`, `ocean/`, `tides/`, `bathymetry/`, `papers/` — real observations and the literature
corpus live here (each has a README describing the expected format). The harness **never
fabricates** observations, and `data/` is write-denied to agents by default.

## Documentation

- **ADRs** — `docs/adr/ADR-001…005` (see `docs/adr/README.md`).
- **Research** — `docs/research/` (literature review, benchmark design, hypothesis catalog).
- **Provenance** — `PROVENANCE.md`.

## License

MIT.
