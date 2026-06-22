// MetaBioHacker harness optimizer — Darwin Mode for acoustic reconstruction.
//
// Philosophy (from @metaharness/darwin): "freeze the model, evolve the harness."
// Here the FROZEN MODEL is the Rust acoustic engine (sonic_ct, compiled to
// WASM). The EVOLVED HARNESS is the reconstruction configuration genome
// {elements, fan, iters} — the knobs that trade reconstruction quality against
// compute. We use Darwin's cheap->frontier tiering (compute arbitrage) and its
// Pareto primitive to find configs that maximise phantom fidelity per unit cost.
//
// Run: npm run optimize

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mapLimit, paretoFront } from "@metaharness/darwin";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM = path.join(__dirname, "public", "sonic_ct.wasm");

// ---- frozen model: load the WASM acoustic engine once ----------------------
const bytes = fs.readFileSync(WASM);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// Evaluate one harness config by reconstructing a small volume and reading the
// mean phantom fidelity (Dice) the engine reports. Returns dice + a cost proxy.
function evaluate({ elements, fan, iters }, { n, nz, seeds }) {
  let diceSum = 0;
  let paths = 0;
  const t0 = performance.now();
  for (const seed of seeds) {
    e.sct_vol_begin(nz, n, elements, Math.min(fan, elements - 1), iters, seed);
    while (e.sct_vol_step() < nz) {}
    diceSum += e.sct_vol_mean_dice();
    paths += e.sct_vol_measurements();
  }
  const ms = performance.now() - t0;
  return { dice: diceSum / seeds.length, ms, paths: Math.round(paths / seeds.length) };
}

// Two evaluation tiers (Darwin compute arbitrage): a cheap filter and an
// accurate "frontier" re-evaluation for the best survivors.
const CHEAP = { n: 40, nz: 6, seeds: [1] };
const FRONTIER = { n: 56, nz: 14, seeds: [1, 2, 3] };

// ---- genome ----------------------------------------------------------------
const BOUNDS = { elements: [48, 280], fan: [24, 200], iters: [2, 12] };
const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, Math.round(v)));
const rnd = (lo, hi) => lo + Math.random() * (hi - lo);

function randomGenome() {
  return {
    elements: clamp(rnd(...BOUNDS.elements), BOUNDS.elements),
    fan: clamp(rnd(...BOUNDS.fan), BOUNDS.fan),
    iters: clamp(rnd(...BOUNDS.iters), BOUNDS.iters),
  };
}
function mutate(g) {
  return {
    elements: clamp(g.elements + rnd(-40, 40), BOUNDS.elements),
    fan: clamp(g.fan + rnd(-30, 30), BOUNDS.fan),
    iters: clamp(g.iters + rnd(-2, 2), BOUNDS.iters),
  };
}
const key = (g) => `${g.elements}-${g.fan}-${g.iters}`;

// ---- Darwin Mode evolution -------------------------------------------------
const POP = 10;
const GENERATIONS = 5;
const ELITE = 4;

const baseline = { elements: 180, fan: 90, iters: 6 };
let population = [baseline, ...Array.from({ length: POP - 1 }, randomGenome)];
let best = null;
const history = [];

console.log("== MetaBioHacker · Darwin harness optimizer ==");
console.log("frozen model: sonic_ct WASM engine | evolving {elements, fan, iters}\n");

for (let gen = 0; gen < GENERATIONS; gen++) {
  // Tier 1 — cheap filter over the whole population (bounded concurrency).
  const cheap = await mapLimit(population, 1, async (g) => ({ g, r: evaluate(g, CHEAP) }));
  cheap.sort((a, b) => b.r.dice - a.r.dice);

  // Tier 2 — frontier re-evaluation of the cheap survivors.
  const survivors = cheap.slice(0, ELITE);
  const scored = await mapLimit(survivors, 1, async ({ g }) => ({ g, r: evaluate(g, FRONTIER) }));

  // Pareto front: maximise dice, minimise cost (paretoFront maximises, so negate ms).
  const front = paretoFront(scored, (s) => [s.r.dice, -s.r.ms]);
  const winner = scored.reduce((a, b) => (b.r.dice > a.r.dice ? b : a));
  if (!best || winner.r.dice > best.r.dice) best = winner;

  history.push({ gen, winner: { ...winner.g, ...winner.r }, frontSize: front.length });
  console.log(
    `gen ${gen}: best dice ${winner.r.dice.toFixed(3)} @ ${key(winner.g)} ` +
      `(${winner.r.paths.toLocaleString()} paths, ${winner.r.ms.toFixed(0)} ms) · pareto ${front.length}`
  );

  // Next generation: elites + their mutations.
  const elites = scored.map((s) => s.g);
  const seen = new Set(elites.map(key));
  const next = [...elites];
  while (next.length < POP) {
    const child = mutate(elites[Math.floor(Math.random() * elites.length)]);
    if (!seen.has(key(child))) {
      seen.add(key(child));
      next.push(child);
    }
  }
  population = next;
}

// ---- report ----------------------------------------------------------------
const baseR = evaluate(baseline, FRONTIER);
const improvement = best.r.dice - baseR.dice;
console.log("\n-- result --");
console.log(`baseline ${key(baseline)}: dice ${baseR.dice.toFixed(3)}, ${baseR.ms.toFixed(0)} ms`);
console.log(`evolved  ${key(best.g)}: dice ${best.r.dice.toFixed(3)}, ${best.r.ms.toFixed(0)} ms`);
console.log(`Δ dice ${improvement >= 0 ? "+" : ""}${improvement.toFixed(3)} (freeze model, evolve harness)`);

const report = {
  tool: "metaharness/darwin",
  philosophy: "freeze the model, evolve the harness",
  frozenModel: "sonic_ct WASM acoustic engine",
  genome: ["elements", "fan", "iters"],
  tiers: { cheap: CHEAP, frontier: FRONTIER },
  baseline: { config: baseline, ...baseR },
  evolved: { config: best.g, ...best.r },
  improvement,
  history,
};
const out = path.join(__dirname, "optimize.report.json");
fs.writeFileSync(out, JSON.stringify(report, null, 2));
console.log(`\nreport -> ${out}`);
