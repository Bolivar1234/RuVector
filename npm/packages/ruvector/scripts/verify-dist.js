#!/usr/bin/env node
/**
 * verify-dist.js — pre-publish gate for the packaged artifact (ADR-272).
 *
 * Checks:
 *   1. Every relative `require('../dist/...')` / `require('../src/...')` in
 *      bin/cli.js AND bin/mcp-server.js resolves to a file on disk.
 *      Why: 0.2.23 was published without a `dist/` directory at all
 *      (issue #399), which silently broke `ruvector doctor`, the `embed`
 *      subsystem, and `rvf` commands. mcp-server.js requires
 *      `../src/decompiler/...` paths that the original gate never scanned.
 *   2. dist/ contains no forbidden artifacts (*.db, *.sqlite, *.log, *.tgz).
 *      Why: the 0.2.26 tarball shipped a 1.6MB runtime `ruvector.db` that a
 *      dirty working tree leaked through the `src/core/onnx` -> dist copy.
 *      `.npmignore` cannot exclude files pulled in via the `files` array.
 */

const fs = require('fs');
const path = require('path');

const pkgRoot = path.resolve(__dirname, '..');
const failures = [];

// --- Check 1: relative requires in both entry points resolve ---
const REQUIRE_RE = /require\(['"]\.\.\/((?:dist|src)\/[^'"]+\.js)['"]\)/g;
const entryPoints = ['bin/cli.js', 'bin/mcp-server.js'];
let requireCount = 0;

for (const entry of entryPoints) {
  const entryPath = path.join(pkgRoot, entry);
  if (!fs.existsSync(entryPath)) {
    failures.push(`${entry} not found — package layout is broken.`);
    continue;
  }
  const source = fs.readFileSync(entryPath, 'utf8');
  const rels = Array.from(new Set(
    Array.from(source.matchAll(REQUIRE_RE), (m) => m[1]),
  )).sort();
  requireCount += rels.length;
  for (const rel of rels) {
    if (!fs.existsSync(path.join(pkgRoot, rel))) {
      failures.push(`${entry} requires missing file: ${rel}`);
    }
  }
}

// --- Check 2: forbidden artifacts under dist/ ---
const FORBIDDEN_RE = /\.(db|sqlite|sqlite3|log|tgz)$/i;
const distRoot = path.join(pkgRoot, 'dist');
let scanned = 0;

function scanForbidden(dir) {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      scanForbidden(full);
    } else {
      scanned++;
      if (FORBIDDEN_RE.test(name)) {
        failures.push(`forbidden artifact in publish payload: ${path.relative(pkgRoot, full)}`);
      }
    }
  }
}

if (fs.existsSync(distRoot)) {
  scanForbidden(distRoot);
} else {
  failures.push('dist/ does not exist — run `npm run build` before packing.');
}

if (failures.length > 0) {
  console.error(`verify-dist: ${failures.length} problem(s) found:`);
  for (const f of failures) {
    console.error(`  - ${f}`);
  }
  console.error(
    "\nRun `npm run build` and confirm tsc emitted under dist/. If a path was renamed,",
  );
  console.error('update the entry point to match. Forbidden files must be deleted before packing.');
  process.exit(1);
}

console.log(`verify-dist: ${requireCount} required path(s) present, ${scanned} dist file(s) clean.`);
