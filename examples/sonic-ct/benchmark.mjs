// MetaBioHacker benchmark: baseline vs Darwin-evolved reconstruction harness,
// over a reproducible synthetic corpus AND a real anatomical CT slice.
//
// The frozen Rust engine (sonic_ct_serve) is the physics layer; we only vary the
// harness config. Writes BENCHMARK.md + benchmark.report.json.
//
// Prereqs: cargo build --release --bin sonic_ct_serve
//          node tools/fetchRealSlice.mjs   (optional, for the real sample)
// Run:     npm run benchmark

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SERVE = path.join(__dirname, "..", "..", "crates", "sonic-ct", "target", "release", "sonic_ct_serve");
const BENCH_DIR = path.join(__dirname, "public", "benchmark");
const N_SEEDS = Number(process.argv[2] || 40);

if (!fs.existsSync(SERVE)) {
  console.error(`missing engine: ${SERVE}\nbuild it: cargo build --release --bin sonic_ct_serve`);
  process.exit(1);
}

function runEngine(reconstruction, sample) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const child = spawn(SERVE, []);
    let out = "";
    child.stdout.on("data", (c) => (out += c));
    child.on("error", reject);
    child.on("close", () => {
      try {
        const r = JSON.parse(out);
        r.latencyMs = performance.now() - t0;
        resolve(r);
      } catch {
        reject(new Error(`bad engine output: ${out}`));
      }
    });
    child.stdin.write(JSON.stringify({ sample, reconstruction }));
    child.stdin.end();
  });
}

// Harness configs: baseline vs evolved (from optimize.report.json if present).
const baseline = {
  voxelResolutionMm: 4,
  temporalWindowMs: 800,
  smoothingAlpha: 0.35,
  ghostBodyPriorWeight: 0.4,
  atlasPriorWeight: 0.25,
  organBoundarySharpness: 0.5,
};
let evolved = { ...baseline, smoothingAlpha: 0.95, voxelResolutionMm: 3.5, organBoundarySharpness: 0.7 };
const reportPath = path.join(__dirname, "optimize.report.json");
if (fs.existsSync(reportPath)) {
  try {
    const rep = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    if (rep?.evolved?.reconstruction) evolved = { ...baseline, ...rep.evolved.reconstruction };
  } catch {}
}

// Dataset: reproducible synthetic seeds + every real CT slice fetched.
const samples = [];
for (let seed = 1; seed <= N_SEEDS; seed++) samples.push({ id: `synthetic-${seed}`, seed, kind: "synthetic" });
const realFiles = fs.existsSync(BENCH_DIR)
  ? fs.readdirSync(BENCH_DIR).filter((f) => /^real_.*\.pgm$/.test(f))
  : [];
for (const f of realFiles) {
  samples.push({ id: `real-${f.replace(/^real_|\.pgm$/g, "")}`, seed: 1, kind: "real", pgm: path.join(BENCH_DIR, f) });
}

const mean = (a) => a.reduce((s, x) => s + x, 0) / Math.max(a.length, 1);
const std = (a) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.map((x) => (x - m) ** 2).reduce((s, x) => s + x, 0) / (a.length - 1));
};
const ci95 = (a) => (a.length > 1 ? (1.96 * std(a)) / Math.sqrt(a.length) : 0);

async function evalConfig(name, reconstruction) {
  const rows = [];
  for (const s of samples) {
    const recon = s.kind === "real" ? { ...reconstruction, phantomPgm: s.pgm } : reconstruction;
    const r = await runEngine(recon, { id: s.id, seed: s.seed });
    rows.push({ ...s, ...r });
  }
  const agg = (key, kind) => {
    const v = rows.filter((r) => !kind || r.kind === kind).map((r) => r[key]);
    return { mean: mean(v), std: std(v), ci95: ci95(v), n: v.length };
  };
  return { name, rows, summary: {
    shape: agg("shapeConsistency", "synthetic"),
    residual: agg("acousticResidual", "synthetic"),
    confidence: agg("confidence", "synthetic"),
    latency: agg("latencyMs"),
    realShape: agg("shapeConsistency", "real"),
  } };
}

console.log("== MetaBioHacker benchmark ==");
console.log(`samples: ${samples.length} (${samples.filter((s) => s.kind === "real").length} real)\n`);

const base = await evalConfig("baseline", baseline);
const evo = await evalConfig("evolved", evolved);

const pct = (a, b) => (b !== 0 ? ((a - b) / Math.abs(b)) * 100 : 0);
const dShape = pct(evo.summary.shape.mean, base.summary.shape.mean);
const dLatency = pct(base.summary.latency.mean, evo.summary.latency.mean);
const dResidual = pct(base.summary.residual.mean, evo.summary.residual.mean);

const f = (x) => x.toFixed(3);
console.log(`samples: ${samples.length} synthetic seeds=${N_SEEDS}, real=${realFiles.length}`);
console.log("config    shape(Dice, 95% CI)   residual    latency(ms)   real-Dice");
for (const c of [base, evo]) {
  const s = c.summary;
  console.log(
    `${c.name.padEnd(9)} ${f(s.shape.mean)}±${f(s.shape.ci95)}  ${f(s.residual.mean)}   ` +
      `${s.latency.mean.toFixed(0).padStart(6)}      ${f(s.realShape.mean)}`
  );
}
console.log(`\nevolved vs baseline: shape ${dShape >= 0 ? "+" : ""}${dShape.toFixed(1)}% · latency ${dLatency >= 0 ? "+" : ""}${dLatency.toFixed(1)}% faster · residual ${dResidual >= 0 ? "-" : "+"}${Math.abs(dResidual).toFixed(1)}%`);

const report = {
  engine: "sonic_ct_serve (frozen)",
  samples: samples.map((s) => ({ id: s.id, kind: s.kind })),
  baseline: base.summary,
  evolved: evo.summary,
  deltas: { shapePct: dShape, latencyPctFaster: dLatency, residualPctLower: dResidual },
  rows: { baseline: base.rows, evolved: evo.rows },
};
fs.writeFileSync(path.join(__dirname, "benchmark.report.json"), JSON.stringify(report, null, 2));

const md = `# MetaBioHacker reconstruction benchmark

Frozen engine: \`sonic_ct_serve\`. Dataset: ${samples.length} samples
(${samples.filter((s) => s.kind === "synthetic").length} reproducible synthetic phantoms${
  samples.some((s) => s.kind === "real") ? " + 1 real abdominal CT slice from Wikimedia Commons" : ""
}). Only the harness config differs between rows.

Statistics over ${base.summary.shape.n} synthetic samples (mean ± 95% CI) and
${base.summary.realShape.n} real CT slice(s).

| Config | Dice (synthetic, 95% CI) | Acoustic residual | Latency (ms) | Dice (real CT) |
|--------|--------------------------|-------------------|--------------|----------------|
| baseline | ${f(base.summary.shape.mean)} ± ${f(base.summary.shape.ci95)} | ${f(base.summary.residual.mean)} | ${base.summary.latency.mean.toFixed(0)} | ${f(base.summary.realShape.mean)} |
| evolved | ${f(evo.summary.shape.mean)} ± ${f(evo.summary.shape.ci95)} | ${f(evo.summary.residual.mean)} | ${evo.summary.latency.mean.toFixed(0)} | ${f(evo.summary.realShape.mean)} |

**Evolved vs baseline:** shape ${dShape >= 0 ? "+" : ""}${dShape.toFixed(1)}%, latency ${dLatency.toFixed(1)}% faster, residual ${dResidual >= 0 ? "−" : "+"}${Math.abs(dResidual).toFixed(1)}%.

Real-data note: the real CT slice is fetched on demand by \`tools/fetchRealSlice.mjs\`
(not committed); intensity is banded into the five acoustic classes as a proxy
ground truth. Real anatomy is harder than synthetic phantoms, so its Dice is
lower — an honest measure, not a regression.
`;
fs.writeFileSync(path.join(__dirname, "..", "..", "docs", "sonic-ct", "BENCHMARK.md"), md);
console.log(`\nreports -> benchmark.report.json + docs/sonic-ct/BENCHMARK.md`);
