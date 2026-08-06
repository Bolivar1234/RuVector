import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const agentdbEntry = require.resolve('agentdb');
const wasmEntry = require.resolve('@ruvector/rvf-wasm');
function packageRoot(entry) {
  let dir = path.dirname(entry);
  while (dir !== path.dirname(dir)) {
    const candidate = path.join(dir, 'package.json');
    if (fs.existsSync(candidate)) return dir;
    dir = path.dirname(dir);
  }
  throw new Error(`package root not found for ${entry}`);
}
const agentdbPackage = JSON.parse(fs.readFileSync(path.join(packageRoot(agentdbEntry), 'package.json'), 'utf8'));
const wasmPackage = JSON.parse(fs.readFileSync(path.join(packageRoot(wasmEntry), 'package.json'), 'utf8'));
const agentdb = await import('agentdb');
const backends = await import('agentdb/backends');
const sqljsModule = await import(pathToFileURL(path.join(packageRoot(agentdbEntry), 'dist', 'src', 'backends', 'rvf', 'SqlJsRvfBackend.js')).href);

const checks = [];
function check(condition, label, detail = '') {
  if (!condition) throw new Error(`${label}${detail ? `: ${detail}` : ''}`);
  checks.push(label);
  console.log(`PASS ${label}`);
}

check(process.platform === 'win32', 'Windows runner', process.platform);
check(process.arch === 'x64', 'Windows x64 runner', process.arch);
check(agentdbPackage.version === '3.0.0-alpha.20', 'AgentDB package version', agentdbPackage.version);
check(wasmPackage.version === '0.1.10', 'Corrected RVF-WASM package version', wasmPackage.version);
check(typeof agentdb.AgentDB === 'function', 'AgentDB import');
check(typeof backends.createBackend === 'function', 'Backend factory import');

const detection = await backends.detectBackends();
check(detection.sqljsRvf === true, 'sql.js RVF fallback detected');
check(detection.available !== 'none', 'At least one backend available');

const backend = new sqljsModule.SqlJsRvfBackend({ dimensions: 4, metric: 'cosine' });
await backend.initialize();
await backend.insertAsync('win-alpha', new Float32Array([1, 0, 0, 0]), { platform: 'win32' });
check(backend.getStats().count === 1, 'Insert on Windows');
const first = await backend.searchAsync(new Float32Array([1, 0, 0, 0]), 1);
check(first[0]?.id === 'win-alpha', 'Query on Windows');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-win-clean-'));
const file = path.join(root, 'roundtrip.rvf');
await backend.save(file);
check(fs.existsSync(file), 'RVF export on Windows');
const reopened = new sqljsModule.SqlJsRvfBackend({ dimensions: 4, metric: 'cosine' });
await reopened.initialize();
await reopened.load(file);
check(reopened.getStats().count === 1, 'RVF reopen on Windows');
const after = await reopened.searchAsync(new Float32Array([1, 0, 0, 0]), 1);
check(after[0]?.id === 'win-alpha', 'Query preserved after Windows reopen');
reopened.close();
backend.close();

console.log(JSON.stringify({
  result: 'PASS',
  checks: checks.length,
  platform: process.platform,
  arch: process.arch,
  agentdb: agentdbPackage.version,
  rvfWasm: wasmPackage.version,
  detection,
  roundTrip: { count: 1, id: after[0]?.id },
}));
