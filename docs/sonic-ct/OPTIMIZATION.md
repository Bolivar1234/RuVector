# Harness optimization with Darwin Mode

MetaBioHacker optimizes its reconstruction pipeline with
[`@metaharness/darwin`](https://www.npmjs.com/package/@metaharness/darwin) using
the **"freeze the model, evolve the harness"** principle.

- **Frozen model** — the Rust acoustic engine (`sonic_ct`, compiled to WASM).
  We never mutate the physics.
- **Evolved harness** — the reconstruction configuration genome
  `{ elements, fan, iters }` (ring density, receiver fan, SART sweeps). These
  trade phantom fidelity (mean Dice) against compute.

## How it works (`examples/sonic-ct/optimize.mjs`)

1. Load the frozen WASM engine once.
2. **Cheap → frontier tiering** (Darwin compute arbitrage): every candidate is
   first scored on a cheap volume (n=40, nz=6, 1 seed); only the top survivors
   are re-scored on a frontier volume (n=56, nz=14, 3 seeds).
3. **Pareto selection** via Darwin's `paretoFront`, maximising Dice while
   minimising wall-clock cost; bounded-concurrency evaluation via `mapLimit`.
4. Evolve elites + mutations over several generations.

## Result (representative)

```
baseline 180-90-6 : dice 0.527, 2228 ms
evolved  185-86-12: dice 0.587, 3878 ms
Δ dice +0.060 (freeze model, evolve harness)
```

The search reliably discovers that additional SART sweeps lift fidelity, and the
Pareto front exposes the quality/compute trade-off so a deployment can pick the
operating point it can afford.

```bash
cd examples/sonic-ct && npm run optimize   # writes optimize.report.json
```

This is the same cost-aware evolutionary loop Darwin Mode applies to LLM agent
harnesses (cheap→frontier tiering, Pareto efficiency), applied here to a
deterministic physics engine instead of an LLM.
