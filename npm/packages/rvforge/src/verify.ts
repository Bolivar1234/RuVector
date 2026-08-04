/**
 * Artifact verification against a build provenance record.
 *
 * Recomputes the SHA256 of every file the record claims, re-derives the
 * manifest hash from the manifest's own bytes, and reports any divergence.
 * Verification reads files; it never runs them.
 */

import { readFile, stat } from 'node:fs/promises';
import type { Stats } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { sha256File } from './hash';
import { manifestHash } from './manifest';
import type { BuildManifest, ProvenanceRecord } from './types';

export const PROVENANCE_FILENAME = 'provenance.json';
export const MANIFEST_FILENAME = 'build-manifest.json';

/** How far above an artifact forge will look for its provenance record. */
const MAX_PROVENANCE_SEARCH_DEPTH = 4;

export interface FileCheck {
  path: string;
  expectedSha256: string;
  actualSha256: string | null;
  ok: boolean;
  reason?: 'missing' | 'digest-mismatch' | 'size-mismatch';
}

export interface VerifyResult {
  root: string;
  provenancePath: string;
  verified: boolean;
  rvfIdentity: string;
  manifestSha256: { expected: string; actual: string | null; ok: boolean };
  files: FileCheck[];
  failures: FileCheck[];
  /** Subset of `files` matching the artifact the caller named, if any. */
  target?: FileCheck;
}

export interface VerifyOptions {
  /** Explicit provenance record path; otherwise discovered next to the artifact. */
  provenancePath?: string;
}

/**
 * Verify `target` — a forge output directory, or a single artifact inside one.
 *
 * @throws {ForgeError} `FORGE_E_NOT_FOUND` when no provenance record can be
 * located, `FORGE_E_VERIFY_FAILED` when a recomputed digest diverges.
 */
export async function runVerify(target: string, options: VerifyOptions = {}): Promise<VerifyResult> {
  const targetPath = resolve(target);
  const info = await statOrThrow(targetPath);

  const provenancePath = options.provenancePath
    ? resolve(options.provenancePath)
    : await findProvenance(targetPath, info.isDirectory());
  const root = dirname(provenancePath);

  const provenance = await readProvenance(provenancePath);
  const files = await checkFiles(root, provenance);
  const manifestCheck = await checkManifest(root, provenance);

  const failures = files.filter((f) => !f.ok);
  const targetCheck = info.isDirectory() ? undefined : matchTarget(root, targetPath, files);

  if (!info.isDirectory() && !targetCheck) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `${targetPath} is not listed in ${provenancePath}; it did not come from this build.`,
      { artifact: relative(root, targetPath), provenance: PROVENANCE_FILENAME },
    );
  }

  const verified = failures.length === 0 && manifestCheck.ok;
  const result: VerifyResult = {
    root,
    provenancePath,
    verified,
    rvfIdentity: provenance.rvfIdentity,
    manifestSha256: manifestCheck,
    files,
    failures,
    ...(targetCheck ? { target: targetCheck } : {}),
  };

  if (!verified) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      failureMessage(failures, manifestCheck),
      {
        provenance: PROVENANCE_FILENAME,
        failedFiles: failures.map((f) => ({ path: f.path, reason: f.reason })),
        manifestOk: manifestCheck.ok,
      },
    );
  }
  return result;
}

function failureMessage(failures: FileCheck[], manifest: VerifyResult['manifestSha256']): string {
  const parts: string[] = [];
  if (failures.length > 0) {
    parts.push(`${failures.length} file(s) failed verification: ${failures.map((f) => f.path).join(', ')}`);
  }
  if (!manifest.ok) {
    parts.push('the build manifest hash does not match the provenance record');
  }
  return `Verification failed — ${parts.join('; ')}.`;
}

async function statOrThrow(path: string): Promise<Stats> {
  try {
    return await stat(path);
  } catch {
    throw new ForgeError(ForgeErrorCode.NOT_FOUND, `${path} does not exist.`, { path });
  }
}

/**
 * Locate the provenance record for `targetPath` by walking up from it.
 *
 * A staged bundle keeps its record at the build root while artifacts sit in
 * subdirectories, so a sibling-only lookup would miss `rvf/agent.rvf`. The
 * walk is depth-bounded so a stray record elsewhere on the filesystem cannot
 * be picked up by accident.
 */
async function findProvenance(targetPath: string, isDirectory: boolean): Promise<string> {
  let dir = isDirectory ? targetPath : dirname(targetPath);

  for (let depth = 0; depth <= MAX_PROVENANCE_SEARCH_DEPTH; depth++) {
    const candidate = join(dir, PROVENANCE_FILENAME);
    try {
      await stat(candidate);
      return candidate;
    } catch {
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }

  throw new ForgeError(
    ForgeErrorCode.NOT_FOUND,
    `No ${PROVENANCE_FILENAME} at or above ${targetPath}. Pass --provenance <path> to point at one.`,
    { path: targetPath },
  );
}

async function readProvenance(path: string): Promise<ProvenanceRecord> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(path, 'utf8'));
  } catch (err) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `${path} is not a readable provenance record: ${(err as Error).message}`,
      { path },
    );
  }
  const record = parsed as Partial<ProvenanceRecord>;
  if (!Array.isArray(record.files) || typeof record.manifestSha256 !== 'string') {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `${path} is missing required provenance fields (files, manifestSha256).`,
      { path },
    );
  }
  return record as ProvenanceRecord;
}

async function checkFiles(root: string, provenance: ProvenanceRecord): Promise<FileCheck[]> {
  const checks: FileCheck[] = [];
  for (const entry of provenance.files) {
    const abs = join(root, entry.path);
    let size: number;
    try {
      size = (await stat(abs)).size;
    } catch {
      checks.push({ path: entry.path, expectedSha256: entry.sha256, actualSha256: null, ok: false, reason: 'missing' });
      continue;
    }
    const actual = await sha256File(abs);
    if (actual !== entry.sha256) {
      checks.push({
        path: entry.path,
        expectedSha256: entry.sha256,
        actualSha256: actual,
        ok: false,
        reason: 'digest-mismatch',
      });
      continue;
    }
    if (size !== entry.size) {
      checks.push({
        path: entry.path,
        expectedSha256: entry.sha256,
        actualSha256: actual,
        ok: false,
        reason: 'size-mismatch',
      });
      continue;
    }
    checks.push({ path: entry.path, expectedSha256: entry.sha256, actualSha256: actual, ok: true });
  }
  return checks;
}

/**
 * Re-derive the manifest hash from the manifest file's own contents. A tamper
 * that rewrites both the manifest and its provenance digest still fails here,
 * because the recomputed canonical hash will not match the recorded one.
 */
async function checkManifest(
  root: string,
  provenance: ProvenanceRecord,
): Promise<VerifyResult['manifestSha256']> {
  const path = join(root, MANIFEST_FILENAME);
  try {
    const manifest = JSON.parse(await readFile(path, 'utf8')) as BuildManifest;
    const actual = manifestHash(manifest);
    return { expected: provenance.manifestSha256, actual, ok: actual === provenance.manifestSha256 };
  } catch {
    return { expected: provenance.manifestSha256, actual: null, ok: false };
  }
}

function matchTarget(root: string, targetPath: string, files: FileCheck[]): FileCheck | undefined {
  const rel = relative(root, targetPath).split(sep).join('/');
  if (rel.startsWith('..') || isAbsolute(rel)) return undefined;
  return files.find((f) => f.path === rel);
}
