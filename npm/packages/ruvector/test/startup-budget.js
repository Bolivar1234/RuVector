#!/usr/bin/env node

/**
 * Startup-budget CI guard (ADR-256 step 4).
 *
 * The ruvector CLI lazy-loads every heavy dependency to keep cold start fast.
 * ADR-256 borrows harness features (the `harness` router surface, MCP policy)
 * into the CLI — this guard fails CI if any of them regress startup.
 *
 * Strategy (robust across machines / loaded CI boxes):
 *   1. ABSOLUTE ceiling — `--help` cold start must stay under a generous cap.
 *   2. RELATIVE delta   — `harness status --json` must not add more than a small
 *      delta over the `--help` baseline (the real regression signal: eager-loading
 *      a heavy module into the harness path would blow this out).
 *
 * Most of the floor (~100ms) is the Node process spawn itself, which is why the
 * relative delta is the meaningful check. Budgets are overridable via env.
 */
'use strict';

const { execSync } = require('child_process');
const path = require('path');
const assert = require('assert');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const CWD = path.join(__dirname, '..');

const ABS_BUDGET_MS = Number(process.env.RUVECTOR_STARTUP_BUDGET_MS || 500);
const DELTA_BUDGET_MS = Number(process.env.RUVECTOR_STARTUP_DELTA_MS || 120);
// Hooks fire on every Claude Code tool use, so they get their own delta guard
// (ADR-274). `hooks stats` loads the intelligence engine today, so its budget
// is looser than the harness delta; tighten as ADR-274 phases land.
const HOOKS_DELTA_BUDGET_MS = Number(process.env.RUVECTOR_HOOKS_DELTA_MS || 1000);
// ADR-274 phase 1: the commander-free micro-entry (bin/hooks.js) must stay
// near bare Node boot cost. Measured median ~50ms; the default budget is ~2x
// that for CI headroom — a regression to full-CLI cost (commander/chalk/445KB
// parse leaking into the module path) blows well past it on CI boxes.
const HOOKS_ENTRY = path.join(__dirname, '..', 'bin', 'hooks.js');
const HOOKS_ENTRY_BUDGET_MS = Number(process.env.RUVECTOR_HOOKS_ENTRY_BUDGET_MS || 100);
const SAMPLES = Number(process.env.RUVECTOR_STARTUP_SAMPLES || 5);

function timeMs(args, entry = CLI) {
  const start = process.hrtime.bigint();
  try {
    execSync(`node ${entry} ${args}`, { stdio: ['pipe', 'pipe', 'pipe'], timeout: 20000, cwd: CWD });
  } catch {
    // non-zero exit still yields a valid timing
  }
  return Number(process.hrtime.bigint() - start) / 1e6;
}

function median(args, entry = CLI) {
  const t = [];
  for (let i = 0; i < SAMPLES; i++) t.push(timeMs(args, entry));
  t.sort((a, b) => a - b);
  return t[Math.floor(t.length / 2)];
}

let passed = 0;
let failed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed++; console.log(`  PASS  ${name}`); }
  catch (err) { failed++; failures.push({ name, error: err.message || String(err) }); console.log(`  FAIL  ${name}`); console.log(`        ${err.message || err}`); }
}

console.log('\n--- Startup-budget guard (ADR-256) ---\n');
console.log(`  samples=${SAMPLES}  abs_budget=${ABS_BUDGET_MS}ms  delta_budget=${DELTA_BUDGET_MS}ms\n`);

// Warm up (filesystem cache, AV scan of the file, etc.)
timeMs('--version');
timeMs('stats', HOOKS_ENTRY);

const helpMs = median('--help');
const harnessMs = median('harness status --json');
const hooksMs = median('hooks stats');
const hooksEntryMs = median('stats', HOOKS_ENTRY);
const delta = harnessMs - helpMs;
const hooksDelta = hooksMs - helpMs;

console.log(`  --help (cold):            ${helpMs.toFixed(0)}ms`);
console.log(`  harness status --json:    ${harnessMs.toFixed(0)}ms  (Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(0)}ms vs --help)`);
console.log(`  hooks stats:              ${hooksMs.toFixed(0)}ms  (Δ ${hooksDelta >= 0 ? '+' : ''}${hooksDelta.toFixed(0)}ms vs --help)`);
console.log(`  hooks.js stats (micro):   ${hooksEntryMs.toFixed(0)}ms\n`);

test(`--help cold start under ${ABS_BUDGET_MS}ms absolute budget`, () => {
  assert(helpMs < ABS_BUDGET_MS,
    `--help startup ${helpMs.toFixed(0)}ms exceeds ${ABS_BUDGET_MS}ms (set RUVECTOR_STARTUP_BUDGET_MS to override)`);
});

test(`harness status adds < ${DELTA_BUDGET_MS}ms over --help baseline (no lazy-load regression)`, () => {
  assert(delta < DELTA_BUDGET_MS,
    `harness status added ${delta.toFixed(0)}ms over --help — a heavy module may have leaked into the startup path`);
});

test(`hooks stats adds < ${HOOKS_DELTA_BUDGET_MS}ms over --help baseline (hooks hot path, ADR-274)`, () => {
  assert(hooksDelta < HOOKS_DELTA_BUDGET_MS,
    `hooks stats added ${hooksDelta.toFixed(0)}ms over --help — the hooks hot path regressed (fires on every Claude Code tool use)`);
});

test(`hooks micro-entry (bin/hooks.js stats) under ${HOOKS_ENTRY_BUDGET_MS}ms (ADR-274 phase 1)`, () => {
  assert(hooksEntryMs < HOOKS_ENTRY_BUDGET_MS,
    `bin/hooks.js stats took ${hooksEntryMs.toFixed(0)}ms, exceeds ${HOOKS_ENTRY_BUDGET_MS}ms — commander/chalk or an eager dist module ` +
    `may have leaked into the micro-entry (set RUVECTOR_HOOKS_ENTRY_BUDGET_MS to override)`);
});

console.log('\n' + '='.repeat(60));
console.log(`\nResults: ${passed} passed, ${failed} failed\n`);
if (failures.length > 0) {
  console.log('Failures:');
  for (const f of failures) console.log(`  - ${f.name}: ${f.error}`);
  console.log('');
}
if (failed > 0) { console.log('STARTUP BUDGET EXCEEDED\n'); process.exit(1); }
else { console.log('STARTUP WITHIN BUDGET\n'); }
