/**
 * Local build.
 *
 * Produces everything that does not require a native packaging toolchain: the
 * canonical build manifest, a staged bundle directory, a software inventory,
 * SHA256 checksums, and a provenance record. When Tauri is not installed the
 * result is explicitly labelled `staged` and no installer is claimed.
 *
 * Two invariants shape the implementation:
 *
 * - **Nothing executes the RVF** (ADR-283 invariant 2). Embedded mode copies
 *   bytes; thin and shared-reader modes write a locator. The file is read, not
 *   interpreted.
 * - **A failed build leaves no partially trusted artifact** (invariant 7).
 *   Everything is assembled in a temporary directory and moved into place in
 *   one step; a failure removes the temporary directory instead.
 */

import { execFile } from 'node:child_process';
import { copyFile, mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJsonFile, sha256File, sha256String } from './hash';
import { manifestFileContents, manifestHash, buildManifest } from './manifest';
import { validateRvf, type ValidateResult } from './validate';
import { FORGE_PACKAGE_NAME, forgeVersion } from './version';
import type {
  BuildManifest,
  CapabilityPolicy,
  ForgeConfig,
  ProvenanceFile,
  ProvenanceRecord,
} from './types';

const execFileAsync = promisify(execFile);

export interface BuildOptions {
  config: ForgeConfig;
  /** Overrides `config.rvf`. */
  rvfPath?: string;
  /** Overrides `config.targets`. */
  targets?: readonly string[];
  outDir: string;
  /** Replace an existing output directory. */
  force?: boolean;
  /** Permit unsigned executable segments (development builds). */
  allowUnsigned?: boolean;
}

export interface BuildResult {
  outDir: string;
  manifest: BuildManifest;
  manifestSha256: string;
  provenance: ProvenanceRecord;
  validation: ValidateResult;
  /** True when a packaging toolchain produced installers. */
  packaged: boolean;
  toolchain: { tauri: string | null };
  warnings: string[];
}

export async function runBuild(options: BuildOptions): Promise<BuildResult> {
  const { config } = options;
  const rvfPath = resolve(options.rvfPath ?? config.rvf);
  const outDir = resolve(options.outDir);

  const validation = await validateRvf(rvfPath, { deep: true, allowUnsigned: options.allowUnsigned });

  const manifest = buildManifest({
    app: config.app,
    identity: validation.identity,
    targets: options.targets ?? config.targets,
    packaging: config.packaging,
    runtime: config.runtime,
    capabilityPolicy: config.capabilityPolicy,
    signing: config.signing,
  });

  await prepareOutDir(outDir, options.force);
  const stagingDir = `${outDir}.staging-${process.pid}`;

  try {
    await mkdir(join(stagingDir, 'rvf'), { recursive: true });

    const files: string[] = [];
    files.push(await writeText(stagingDir, 'build-manifest.json', manifestFileContents(manifest)));
    files.push(
      await writeText(
        stagingDir,
        'inventory.json',
        canonicalJsonFile(softwareInventory(manifest, validation)),
      ),
    );
    files.push(...(await stagePayload(stagingDir, rvfPath, manifest, config.capabilityPolicy)));

    const tauri = await detectTauri();
    const packaged = false; // Installer generation lands with the Tauri packaging layer.
    const warnings: string[] = [...validation.warnings];
    if (!tauri) {
      warnings.push(
        'Tauri was not found on PATH; forge staged the bundle and produced no installers. ' +
          'Install the Tauri CLI, or use "forge submit" to build on hosted workers.',
      );
    } else {
      warnings.push(
        `Tauri ${tauri} is available but local installer generation is not wired up yet; the bundle is staged only.`,
      );
    }

    const provenanceFiles = await hashFiles(stagingDir, files);
    const provenance = await buildProvenance({
      manifest,
      files: provenanceFiles,
      packaged,
      tauri,
    });
    await writeText(stagingDir, 'provenance.json', canonicalJsonFile(provenance));
    await writeText(stagingDir, 'checksums.txt', checksumsFile(provenanceFiles));

    await rename(stagingDir, outDir);

    return {
      outDir,
      manifest,
      manifestSha256: provenance.manifestSha256,
      provenance,
      validation,
      packaged,
      toolchain: { tauri },
      warnings,
    };
  } catch (err) {
    await rm(stagingDir, { recursive: true, force: true });
    throw err;
  }
}

async function prepareOutDir(outDir: string, force?: boolean): Promise<void> {
  try {
    await stat(outDir);
  } catch {
    await mkdir(dirname(outDir), { recursive: true });
    return;
  }
  if (!force) {
    throw new ForgeError(
      ForgeErrorCode.IO,
      `Output directory ${outDir} already exists. Pass --force to replace it.`,
      { outDir },
    );
  }
  await rm(outDir, { recursive: true, force: true });
}

/**
 * Stage the RVF itself. Embedded mode copies the file byte for byte so the
 * embedded hash matches the input (invariant 1); the other modes write a
 * locator the reader resolves and verifies at install time.
 */
async function stagePayload(
  stagingDir: string,
  rvfPath: string,
  manifest: BuildManifest,
  policy: CapabilityPolicy,
): Promise<string[]> {
  if (manifest.packaging.mode === 'embedded') {
    const target = join('rvf', basename(rvfPath));
    await copyFile(rvfPath, join(stagingDir, target));
    return [target];
  }

  const locator = {
    schemaVersion: 1,
    mode: manifest.packaging.mode,
    rvfSha256: manifest.rvf.sha256,
    rvfSizeBytes: manifest.rvf.size,
    fileId: manifest.rvf.fileId,
    distributionUrl: manifest.packaging.distributionUrl ?? null,
    updateChannel: manifest.packaging.updateChannel ?? null,
    capabilityPolicyHash: manifest.capabilityPolicyHash,
    defaultDeny: policy.defaultDeny,
  };
  const target = join('rvf', 'locator.json');
  await writeText(stagingDir, target, canonicalJsonFile(locator));
  return [target];
}

interface SoftwareInventory {
  schemaVersion: 1;
  components: Array<Record<string, string | number>>;
  rvfSegments: { count: number; executableCount: number; byType: Record<string, number> };
  capabilityPolicyHash: string;
}

/** Software inventory: what is in the package, from inspection only. */
function softwareInventory(manifest: BuildManifest, validation: ValidateResult): SoftwareInventory {
  return {
    schemaVersion: 1,
    components: [
      {
        name: FORGE_PACKAGE_NAME,
        version: manifest.generator.version,
        role: 'builder',
      },
      {
        name: 'rvf',
        version: `format-v${validation.root.formatVersion}`,
        role: 'payload',
        sha256: manifest.rvf.sha256,
        sizeBytes: manifest.rvf.size,
      },
      {
        name: 'rvm',
        version: manifest.runtime.rvmVersion,
        role: 'runtime',
        commit: manifest.runtime.rvmCommit,
        profile: manifest.runtime.runtimeProfile,
      },
    ],
    rvfSegments: {
      count: validation.segments.count,
      executableCount: validation.segments.executableCount,
      byType: validation.segments.byType,
    },
    capabilityPolicyHash: manifest.capabilityPolicyHash,
  };
}

async function buildProvenance(input: {
  manifest: BuildManifest;
  files: ProvenanceFile[];
  packaged: boolean;
  tauri: string | null;
}): Promise<ProvenanceRecord> {
  const { manifest, files, packaged } = input;
  const signed = Object.keys(manifest.signing).length > 0;
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    rvfIdentity: manifest.rvf.sha256,
    manifestSha256: manifestHash(manifest),
    builder: {
      tool: FORGE_PACKAGE_NAME,
      version: forgeVersion(),
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
      sourceRevision: await gitRevision(),
    },
    status: packaged ? 'built' : 'staged',
    packaging: {
      mode: manifest.packaging.mode,
      status: packaged ? 'packaged' : 'staged',
      ...(packaged ? {} : { reason: input.tauri ? 'installer-generation-not-implemented' : 'tauri-unavailable' }),
    },
    // Local builds are development builds: forge holds signing references, and
    // the signing operation itself happens on a worker with HSM/KMS access.
    signing: { mode: signed ? 'signed' : 'unsigned-development', refs: manifest.signing },
    targets: manifest.targets,
    files,
  };
}

async function hashFiles(root: string, relativePaths: readonly string[]): Promise<ProvenanceFile[]> {
  const entries: ProvenanceFile[] = [];
  for (const rel of [...relativePaths].sort()) {
    const abs = join(root, rel);
    const info = await stat(abs);
    entries.push({ path: rel.split('\\').join('/'), sha256: await sha256File(abs), size: info.size });
  }
  return entries;
}

/** `sha256sum`-compatible output so standard tooling can check it. */
function checksumsFile(files: readonly ProvenanceFile[]): string {
  return `${files.map((f) => `${f.sha256}  ${f.path}`).join('\n')}\n`;
}

async function writeText(root: string, relative: string, contents: string): Promise<string> {
  const abs = join(root, relative);
  await mkdir(dirname(abs), { recursive: true });
  await writeFile(abs, contents, 'utf8');
  return relative;
}

/** Tauri CLI version, or null when it is not on PATH. */
async function detectTauri(): Promise<string | null> {
  for (const [cmd, args] of [
    ['cargo-tauri', ['--version']],
    ['tauri', ['--version']],
  ] as const) {
    try {
      const { stdout } = await execFileAsync(cmd, [...args], { timeout: 5_000 });
      return stdout.trim();
    } catch {
      continue;
    }
  }
  return null;
}

async function gitRevision(): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], { timeout: 5_000 });
    const rev = stdout.trim();
    return /^[0-9a-f]{40}$/.test(rev) ? rev : null;
  } catch {
    return null;
  }
}

/** SHA256 of a manifest file's exact bytes, for cross-checking a record. */
export const manifestFileHash = (contents: string): string => sha256String(contents);
