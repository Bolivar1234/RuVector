#!/usr/bin/env node
/**
 * Fixture generator for `scripts/rvforge-parity-check.sh`.
 *
 * Writes the two inputs a publish needs — a synthetic `.rvf` and an
 * `rvforge.json` that packs clean — into a directory given on the command line.
 *
 * Both are built from the CLI's own compiled modules rather than from literals:
 * the RVF layout comes from `dist/rvf.js` (the same constants and CRC the
 * parser reads with) and the project scaffold from `dist/project.js`. A fixture
 * assembled from hand-copied magic numbers would drift from the parser it is
 * meant to feed, and the parity run would start failing for a reason that has
 * nothing to do with the registry.
 *
 * The shapes mirror `npm/packages/rvforge/tests/fixtures/{rvf,project}-fixture.ts`,
 * which are TypeScript and therefore not loadable from a plain node process.
 *
 * Usage: node scripts/rvforge-parity-fixture.cjs <cli-dist-dir> <out-dir>
 */

'use strict';

const { writeFileSync } = require('node:fs');
const { join, resolve } = require('node:path');

const [, , distDirArg, outDirArg] = process.argv;
if (!distDirArg || !outDirArg) {
  process.stderr.write('usage: rvforge-parity-fixture.cjs <cli-dist-dir> <out-dir>\n');
  process.exit(2);
}

const distDir = resolve(distDirArg);
const outDir = resolve(outDirArg);

const {
  ROOT_MANIFEST_MAGIC_BYTES,
  ROOT_MANIFEST_SIZE,
  SEGMENT_HEADER_SIZE,
  SEGMENT_MAGIC_BYTES,
  alignUp,
  crc32c,
} = require(join(distDir, 'rvf.js'));
const { defaultDenials, defaultProject } = require(join(distDir, 'project.js'));

/** The same minimal artifact the CLI's own tests pack: three segments, signed. */
const FIXTURE = {
  segments: [
    { type: 0x01, payloadLength: 128 },
    { type: 0x02, payloadLength: 64 },
    { type: 0x05, payloadLength: 32 },
  ],
  signed: true,
  formatVersion: 1,
  dimension: 8,
  vectorCount: 4,
  fileId: '0123456789abcdef0123456789abcdef',
};

function buildSegments(segments) {
  const chunks = [];
  segments.forEach((segment, index) => {
    const header = Buffer.alloc(SEGMENT_HEADER_SIZE);
    SEGMENT_MAGIC_BYTES.copy(header, 0);
    header[0x04] = 1; // version
    header[0x05] = segment.type;
    header.writeUInt16LE(0, 0x06); // flags
    header.writeBigUInt64LE(BigInt(index), 0x08); // segment_id
    header.writeBigUInt64LE(BigInt(segment.payloadLength), 0x10);
    header.writeBigUInt64LE(0n, 0x18); // timestamp_ns — zero keeps the bytes stable
    header[0x20] = 0; // checksum_algo: CRC32C
    header[0x21] = 0; // compression: none

    const unpadded = SEGMENT_HEADER_SIZE + segment.payloadLength;
    chunks.push(header, Buffer.alloc(alignUp(unpadded) - SEGMENT_HEADER_SIZE));
  });
  return Buffer.concat(chunks);
}

function buildRootManifest(spec, bodyLength) {
  const page = Buffer.alloc(ROOT_MANIFEST_SIZE);
  ROOT_MANIFEST_MAGIC_BYTES.copy(page, 0x000);
  page.writeUInt16LE(spec.formatVersion, 0x004);
  page.writeUInt16LE(0, 0x006); // flags
  page.writeBigUInt64LE(0n, 0x008); // l1_manifest_offset
  page.writeBigUInt64LE(BigInt(Math.min(SEGMENT_HEADER_SIZE, bodyLength)), 0x010);
  page.writeBigUInt64LE(BigInt(spec.vectorCount), 0x018);
  page.writeUInt16LE(spec.dimension, 0x020);
  page[0x022] = 1; // base_dtype: f32
  page[0x023] = 0; // profile_id
  page.writeUInt32LE(1, 0x024); // epoch
  page.writeBigUInt64LE(0n, 0x028); // created_ns
  page.writeBigUInt64LE(0n, 0x030); // modified_ns

  if (spec.signed) {
    page.writeUInt16LE(1, 0x098); // sig_algo: Ed25519
    page.writeUInt16LE(64, 0x09a); // sig_length
    Buffer.alloc(64, 0xab).copy(page, 0x09c); // placeholder signature bytes
  }

  Buffer.from(spec.fileId, 'hex').copy(page, 0xf00);
  page.writeUInt32LE(crc32c(page.subarray(0, 0xffc)), 0xffc);
  return page;
}

const body = buildSegments(FIXTURE.segments);
const rvf = Buffer.concat([body, buildRootManifest(FIXTURE, body.length)]);

/** Narrow, concrete scopes: nothing here trips a manual-review trigger. */
const requests = [
  { class: 'filesystem', scope: 'user-selected', rationale: 'reads only documents you choose' },
  { class: 'memory', scope: '512MiB', rationale: 'model working set' },
  { class: 'persistent-state', scope: 'encrypted-local', rationale: 'agent memory' },
];

const project = defaultProject({
  version: '1.0.0',
  license: 'Apache-2.0',
  listing: {
    name: 'analyst',
    displayName: 'Cognitum Analyst',
    description: 'Reads the documents you select and answers questions about them.',
    category: 'developer-tools',
    icon: null,
    screenshots: [],
    priceModel: 'free',
  },
  publisher: {
    displayName: 'RVForge Parity Check',
    identityEvidence: { method: 'manual-review', reference: 'scripts/rvforge-parity-check.sh' },
    contact: { support: 'support@example.invalid', privacyPolicy: 'https://example.invalid/privacy' },
    keyFile: null,
  },
  runtimeRequirements: {
    profiles: ['wasm'],
    systems: ['linux-x64', 'macos-universal', 'windows-x64'],
    memoryMiB: 512,
    rvmVersionMin: '0.1.0',
    stateSchemaVersion: 1,
    witnessSchemaVersion: 1,
  },
  capabilities: { defaultPolicy: 'deny', requests, denials: defaultDenials(requests) },
  modelManifest: { location: 'embedded', digests: [`sha256:${'a'.repeat(64)}`] },
  externalServices: [],
  state: { checkpointing: true, recovery: true, rollbackSafeUntilStateSchema: 2 },
});

const rvfPath = join(outDir, 'agent.rvf');
const projectPath = join(outDir, 'rvforge.json');
writeFileSync(rvfPath, rvf);
writeFileSync(projectPath, `${JSON.stringify(project, null, 2)}\n`, 'utf8');

process.stdout.write(`${rvfPath}\n${projectPath}\n`);
