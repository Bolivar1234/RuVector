// End-to-end durability test for COW branches through the public TypeScript SDK.
// Requires the native @ruvector/rvf-node package to be resolvable.
//
// Run: node scripts/smoke-cow-durability.mjs   (after `tsc` emits dist/)
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { RvfDatabase } = await import(
  pathToFileURL(path.resolve('dist/index.js')).href
);

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-cow-sdk-'));
const original = path.join(root, 'original');
const relocated = path.join(root, 'relocated');
fs.mkdirSync(original);

const parentPath = path.join(original, 'parent.rvf');
const childPath = path.join(original, 'child.rvf');
const a = new Float32Array([1, 0, 0, 0]);
const b = new Float32Array([0, 1, 0, 0]);
const c = new Float32Array([0, 0, 1, 0]);
const d = new Float32Array([0, 0, 0, 1]);

let parent = await RvfDatabase.create(parentPath, { dimensions: 4 }, 'node');
await parent.ingestBatch([
  { id: 'parent-a', vector: a },
  { id: 'deleted-b', vector: b },
  { id: 'parent-c', vector: c },
]);
await parent.freeze();

let child = await parent.branch(childPath);
await child.ingestBatch([{ id: 'child-d', vector: d }]);
assert.strictEqual((await child.delete(['deleted-b'])).deleted, 1);
await child.close();
await parent.close();

// Parent references are relative to the bundle, so moving both files together
// must preserve the lineage.
fs.renameSync(original, relocated);
child = await RvfDatabase.open(path.join(relocated, 'child.rvf'), 'node');

assert.strictEqual((await child.query(a, 1))[0]?.id, 'parent-a');
assert.strictEqual((await child.query(d, 1))[0]?.id, 'child-d');
assert.ok(
  !(await child.query(b, 4)).some((result) =>
    result.id === 'deleted-b' || result.id === '2'
  ),
  'the child tombstone must still hide the inherited vector after reopen',
);

await child.close();
fs.rmSync(root, { recursive: true, force: true });
console.log('ok - COW branch preserves inherited strings, child edits, tombstones, and relative parent references');
