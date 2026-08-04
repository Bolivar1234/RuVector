/**
 * Local, file-backed registry (registry data model v0.1, "Storage layout").
 *
 * ```text
 * registry/
 *   objects/sha256/<2-hex>/<digest>.json   all objects, content-addressed
 *   packages/<publisher>/publisher.json    publisher record pointer
 *   packages/<publisher>/<name>/releases.jsonl   append-only release index
 *   log/entries.jsonl                      transparency log
 *   log/tree-head.json                     current tree head
 *   receipts.jsonl                         witness chain over publish events
 * ```
 *
 * Identity follows the model's rules: an object's id is `sha256:<hex>` over
 * its canonical JSON, excluding `signatures` — and excluding the id field
 * itself, since a field cannot contain its own hash. That exclusion is the
 * convention `src/witness.ts` already established for `receiptId`.
 *
 * **Deviations from registry-model.md**, to be reconciled against the Rust
 * `crates/rvforge-registry` by the parity test:
 *
 * - The tree head is a **hash chain**, not a Merkle tree: `treeHead(n) =
 *   sha256(treeHead(n-1) || entryHash(n))`. That is append-only and tamper-
 *   evident, but it yields no inclusion proof for an individual entry, so a
 *   Reader cannot verify inclusion without the whole log. The Rust crate owns
 *   the Merkle implementation.
 * - `packages/<publisher>/publisher.json` and `receipts.jsonl` are additions;
 *   the documented layout names neither, and the model gives no way to resolve
 *   a `publisherId` back to its record.
 * - The tree head is unsigned. The model calls it "the current signed tree
 *   head", but signing it is the registry's act, and this registry has no
 *   registry key — only the publisher's.
 */

import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJson, canonicalJsonFile, sha256String } from './hash';

export const DEFAULT_REGISTRY_DIR = join(homedir(), '.rvforge', 'registry');
export const LOG_ENTRIES_FILE = join('log', 'entries.jsonl');
export const TREE_HEAD_FILE = join('log', 'tree-head.json');
export const RELEASES_FILENAME = 'releases.jsonl';
export const PUBLISHER_FILENAME = 'publisher.json';

export interface ObjectSignature {
  keyId: string;
  role: 'publisher' | 'registry';
  sig: string;
}

export interface PublisherRecord {
  schemaVersion: 1;
  type: 'publisher';
  publisherId: string;
  displayName: string;
  publicKeys: Array<{
    keyId: string;
    alg: 'ed25519';
    publicKey: string;
    validFrom: string;
    revokedAt: string | null;
  }>;
  identityEvidence: { method: string; reference: string };
  contact: { support: string; privacyPolicy: string };
  signatures: ObjectSignature[];
}

export interface CapabilityManifest {
  schemaVersion: 1;
  type: 'capability-manifest';
  manifestId: string;
  defaultPolicy: 'deny';
  requests: Array<{ class: string; scope: string; rationale: string }>;
  denials: string[];
  manualReviewTriggers: string[];
  signatures: ObjectSignature[];
}

export interface Release {
  schemaVersion: 1;
  type: 'release';
  releaseId: string;
  package: { name: string; publisherId: string };
  version: string;
  predecessor: string | null;
  rvfIdentity: string;
  rvfSize: number;
  capabilityManifest: string;
  runtimeProfiles: string[];
  compatibility: { rvmVersionMin: string; stateSchemaVersion: number; witnessSchemaVersion: number };
  modelManifest: { location: string; digests: string[] };
  softwareInventory: string;
  evaluationReport: string | null;
  securityReport: string | null;
  provenance: string;
  trustLevel: 'published' | 'tested' | 'reviewed' | 'enterprise-approved';
  rollbackSafeUntilStateSchema: number;
  listing: { category: string; description: string; priceModel: string };
  publishedAt: string;
  signatures: ObjectSignature[];
}

export interface TransparencyLogEntry {
  schemaVersion: 1;
  type: 'log-entry';
  index: number;
  entryHash: string;
  treeHead: string;
  object: string;
  objectType: 'release' | 'revocation' | 'publisher';
  timestamp: string;
}

export interface TreeHead {
  schemaVersion: 1;
  type: 'tree-head';
  /** Number of entries the head commits to. */
  size: number;
  treeHead: string;
  /** Named so a Reader is not misled into expecting an inclusion proof. */
  algorithm: 'sha256-hash-chain';
  updatedAt: string;
}

/** Any object the registry content-addresses. */
export type RegistryObject = PublisherRecord | CapabilityManifest | Release;

/** Fields excluded from an object's content id: its own id, and signatures. */
const ID_FIELDS = new Set(['publisherId', 'manifestId', 'releaseId', 'receiptId']);

/**
 * `sha256:<hex>` over an object's canonical JSON, excluding `signatures` and
 * the object's own id field.
 */
export function objectId(value: Record<string, unknown>): string {
  const content: Record<string, unknown> = {};
  for (const key of Object.keys(value)) {
    if (key === 'signatures' || ID_FIELDS.has(key)) continue;
    content[key] = value[key];
  }
  return `sha256:${sha256String(canonicalJson(content))}`;
}

/** Strip the `sha256:` prefix; ids are stored and pathed by their hex digest. */
export const digestOf = (id: string): string => (id.startsWith('sha256:') ? id.slice(7) : id);

/** Path of a content-addressed object within a registry directory. */
export const objectPath = (registryDir: string, id: string): string => {
  const digest = digestOf(id);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new ForgeError(ForgeErrorCode.REGISTRY, `"${id}" is not a sha256 object id.`, { id });
  }
  return join(registryDir, 'objects', 'sha256', digest.slice(0, 2), `${digest}.json`);
};

export const packagePath = (registryDir: string, publisherId: string, name: string): string =>
  join(registryDir, 'packages', digestOf(publisherId), name);

export const publisherPath = (registryDir: string, publisherId: string): string =>
  join(registryDir, 'packages', digestOf(publisherId), PUBLISHER_FILENAME);

/** Create the directory skeleton. Idempotent. */
export async function initRegistry(registryDir: string): Promise<void> {
  for (const dir of [join(registryDir, 'objects', 'sha256'), join(registryDir, 'log')]) {
    await mkdir(dir, { recursive: true });
  }
}

/**
 * Write an object at its content address.
 *
 * Content addressing makes the write idempotent: identical bytes land on the
 * same path, so re-publishing an unchanged manifest is a no-op rather than a
 * conflict. `existed` reports which happened.
 */
export async function putObject(
  registryDir: string,
  id: string,
  object: unknown,
): Promise<{ path: string; existed: boolean }> {
  const path = objectPath(registryDir, id);
  await mkdir(dirname(path), { recursive: true });
  const body = canonicalJsonFile(object);
  try {
    await writeFile(path, body, { encoding: 'utf8', flag: 'wx' });
    return { path, existed: false };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
      throw new ForgeError(ForgeErrorCode.IO, `Cannot write registry object ${id}.`, { id, path });
    }
    // A signature added later changes the stored bytes but not the id, so an
    // existing object is refreshed rather than left at its unsigned version.
    await writeFile(path, body, 'utf8');
    return { path, existed: true };
  }
}

/** Read a content-addressed object, or null when it is absent. */
export async function getObject<T>(registryDir: string, id: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(objectPath(registryDir, id), 'utf8')) as T;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
    if (err instanceof ForgeError) throw err;
    throw new ForgeError(ForgeErrorCode.REGISTRY, `Registry object ${id} is unreadable or malformed.`, { id });
  }
}

/** One line of `releases.jsonl`: the index, not the release itself. */
export interface ReleaseIndexEntry {
  releaseId: string;
  version: string;
  predecessor: string | null;
  rvfIdentity: string;
  publishedAt: string;
}

/** Read a package's release index; an absent file means no releases yet. */
export async function readReleaseIndex(
  registryDir: string,
  publisherId: string,
  name: string,
): Promise<ReleaseIndexEntry[]> {
  let raw: string;
  try {
    raw = await readFile(join(packagePath(registryDir, publisherId, name), RELEASES_FILENAME), 'utf8');
  } catch {
    return [];
  }
  return raw
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as ReleaseIndexEntry);
}

/**
 * Append a release to its package index after checking the lineage.
 *
 * A release must name the current head as its predecessor: the first release
 * names null, and every later one names the release before it. Rejecting a
 * mismatch here is what keeps `releases.jsonl` a chain rather than a set.
 *
 * @throws {ForgeError} `FORGE_E_LINEAGE` when the predecessor is not the head.
 */
export async function appendRelease(
  registryDir: string,
  release: Release,
): Promise<{ path: string; index: number }> {
  const dir = packagePath(registryDir, release.package.publisherId, release.package.name);
  const existing = await readReleaseIndex(registryDir, release.package.publisherId, release.package.name);
  const head = existing.length === 0 ? null : existing[existing.length - 1].releaseId;

  if (release.predecessor !== head) {
    throw new ForgeError(
      ForgeErrorCode.LINEAGE,
      `Release names predecessor ${String(release.predecessor)}, but the head of ` +
        `${release.package.name} is ${String(head)}.`,
      { predecessor: release.predecessor, head, package: release.package.name },
    );
  }
  if (existing.some((entry) => entry.version === release.version)) {
    throw new ForgeError(
      ForgeErrorCode.LINEAGE,
      `Version ${release.version} of ${release.package.name} is already published; releases are immutable.`,
      { version: release.version, package: release.package.name },
    );
  }

  const entry: ReleaseIndexEntry = {
    releaseId: release.releaseId,
    version: release.version,
    predecessor: release.predecessor,
    rvfIdentity: release.rvfIdentity,
    publishedAt: release.publishedAt,
  };
  await mkdir(dir, { recursive: true });
  await appendFile(join(dir, RELEASES_FILENAME), `${canonicalJson(entry)}\n`, 'utf8');
  return { path: join(dir, RELEASES_FILENAME), index: existing.length };
}

/** Read the transparency log. An absent file means an empty log. */
export async function readLog(registryDir: string): Promise<TransparencyLogEntry[]> {
  let raw: string;
  try {
    raw = await readFile(join(registryDir, LOG_ENTRIES_FILE), 'utf8');
  } catch {
    return [];
  }
  return raw
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as TransparencyLogEntry);
}

/** Read the current tree head, or null when the log is empty. */
export async function readTreeHead(registryDir: string): Promise<TreeHead | null> {
  try {
    return JSON.parse(await readFile(join(registryDir, TREE_HEAD_FILE), 'utf8')) as TreeHead;
  } catch {
    return null;
  }
}

/**
 * Append one entry to the transparency log and advance the tree head.
 *
 * `entryHash` commits to the entry's position and payload; the head folds it
 * into the previous head, so removing or reordering any entry changes every
 * head after it.
 */
export async function appendLogEntry(
  registryDir: string,
  input: { object: string; objectType: TransparencyLogEntry['objectType']; timestamp: string },
): Promise<{ entry: TransparencyLogEntry; head: TreeHead }> {
  const existing = await readLog(registryDir);
  const index = existing.length;
  const previousHead = index === 0 ? '' : existing[index - 1].treeHead;

  const entryHash = `sha256:${sha256String(
    canonicalJson({ index, object: input.object, objectType: input.objectType, timestamp: input.timestamp }),
  )}`;
  const treeHead = `sha256:${sha256String(`${previousHead}${entryHash}`)}`;

  const entry: TransparencyLogEntry = {
    schemaVersion: 1,
    type: 'log-entry',
    index,
    entryHash,
    treeHead,
    object: input.object,
    objectType: input.objectType,
    timestamp: input.timestamp,
  };
  const head: TreeHead = {
    schemaVersion: 1,
    type: 'tree-head',
    size: index + 1,
    treeHead,
    algorithm: 'sha256-hash-chain',
    updatedAt: input.timestamp,
  };

  await mkdir(join(registryDir, 'log'), { recursive: true });
  await appendFile(join(registryDir, LOG_ENTRIES_FILE), `${canonicalJson(entry)}\n`, 'utf8');
  await writeFile(join(registryDir, TREE_HEAD_FILE), canonicalJsonFile(head), 'utf8');
  return { entry, head };
}

/** Recompute the chain and report the first entry that does not match. */
export function verifyLog(entries: readonly TransparencyLogEntry[]): {
  ok: boolean;
  size: number;
  treeHead: string | null;
  problems: Array<{ index: number; reason: 'entry-hash' | 'tree-head' | 'index' }>;
} {
  const problems: Array<{ index: number; reason: 'entry-hash' | 'tree-head' | 'index' }> = [];
  let previousHead = '';

  entries.forEach((entry, position) => {
    if (entry.index !== position) problems.push({ index: position, reason: 'index' });
    const entryHash = `sha256:${sha256String(
      canonicalJson({
        index: entry.index,
        object: entry.object,
        objectType: entry.objectType,
        timestamp: entry.timestamp,
      }),
    )}`;
    if (entryHash !== entry.entryHash) problems.push({ index: position, reason: 'entry-hash' });
    if (`sha256:${sha256String(`${previousHead}${entryHash}`)}` !== entry.treeHead) {
      problems.push({ index: position, reason: 'tree-head' });
    }
    previousHead = entry.treeHead;
  });

  return {
    ok: problems.length === 0,
    size: entries.length,
    treeHead: entries.length === 0 ? null : entries[entries.length - 1].treeHead,
    problems,
  };
}
