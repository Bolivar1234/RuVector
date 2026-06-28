# earth-pulse-harness — provenance

This directory is a **metaharness bundle** ("metaharness as program synthesis"): a
self-contained agent harness whose *workflow* can be evolved by Darwin Mode while the
model and the underlying physics stay frozen. It was authored in the same structure as
the sibling `harnesses/timesfm-harness` bundle in this repository.

## What it governs

A causal-discovery research pod for Earth's ~26-second microseism (Gulf of Guinea) and
the associated gliding tremors. The investigation is framed as a bounded benchmark
(predict pulse amplitude from ocean state; beat a seasonal baseline out-of-sample), not
as an attempt to "solve" the phenomenon. See `README.md` and `docs/research/`.

## Structure (synthesized, then hand-verified)

```
earth-pulse-harness/
  bin/cli.js                 init | doctor | pipeline | --version
  src/                       detect-26s, extract-features, embed-events,
                             score-hypotheses, validate, pipeline, types, agents/
  __tests__/                 pipeline.test.ts (offline science), smoke.test.ts (install)
  tests/fixtures/            synthetic 26s seismic window + Gulf of Guinea env
  .metaharness/              objective.json, safety-policy.json, genome.json
  .claude/                   settings.json (default-deny MCP policy), commands, skills
  .claude-plugin/plugin.json
  docs/adr/                  ADR-001 … ADR-005 + index
  docs/research/             literature review, benchmark design, hypothesis catalog
  data/                      seismic|ocean|tides|bathymetry|papers (READMEs; real data goes here)
  .harness/                  manifest.json + manifest.sha256 (witness)
```

## Scientific honesty contract

- The harness **never fabricates** seismic, ocean, tide, or bathymetry observations.
- No citation is invented. The only firm modern reference is **Bruland & Hadziioannou
  (2023)** on gliding tremors associated with the 26 s microseism; older work is framed
  generically rather than with fabricated DOIs/author lists (see
  `docs/research/26-second-pulse-literature.md`).
- All numeric figures in the docs and configs are **targets/priors/illustrations**, not
  measured findings, until a real run produces them.
- Every *promoted* claim must map to a document in `data/papers/`.

## Verification at authoring time

Run locally and confirmed green:

```
npm install          # resolves @metaharness/{kernel,host-claude-code,darwin}
npm test             # 17 passing (13 offline pipeline + 4 install smoke)
npm run build        # tsc, exit 0
npm run doctor       # all checks pass (kernel wasm backend, host claude-code)
npm run pipeline     # detects dominant period ~26.06s; ranks coupled-ocean-geology top
```

## Witness / provenance

`.harness/manifest.json` records `schema:1`, the template/vars/hosts, and a SHA-256 for
every emitted source/doc/config file. `.harness/manifest.sha256` is the witness over the
manifest. Verify integrity:

```bash
sha256sum .harness/manifest.json   # must equal the contents of .harness/manifest.sha256
```

## Optimizing the harness — Darwin evolve

The harness ships a Darwin-Mode self-improvement loop (the `/evolve` skill). Its default
mutator is **deterministic** (air-gapped, no key). The optional **LLM mutator**
(`OpenRouterMutator`) is library-only — not exposed by the `metaharness-darwin` CLI — so
`scripts/evolve-openrouter.{sh,mjs}` wire it into the `evolve()` engine. The OpenRouter
API key is **sourced from a secret manager at runtime** and exported only into the run's
process — never stored in the repo, a dotfile, or the logs. Every mutation passes the
`validateGeneratedCode` safety gate and only promotes on measured improvement.
