/**
 * Embedding provenance — ADR-210 D0 cross-cutting invariant.
 *
 * Every persisted vector store written through the embedding path records
 * `{ embedderKind, modelId, dimension, normalize, prefixPolicy }`. Inserts
 * whose provenance does not match the store's recorded provenance are
 * REFUSED (clear error naming both sides), never coerced. Stores that
 * predate provenance metadata are treated as legacy hash stores and open
 * read-only for vector writes until re-embedded (`ruvector hooks reembed`).
 *
 * This module is the single source of truth for:
 *   - the provenance record type + compare/refuse logic (D0),
 *   - legacy-default derivation for pre-ADR-210 stores (D0),
 *   - per-model query/passage prefix policies (D4),
 *   - rollout flag resolution: RUVECTOR_EMBEDDER / RUVECTOR_ONNX /
 *     RUVECTOR_REEMBED (D5),
 *   - the once-per-process loud hash-fallback warning (D1).
 */

// ============================================================================
// Types
// ============================================================================

export type PrefixPolicy = 'none' | 'required' | 'query-recommended';
export type EmbedTextKind = 'query' | 'passage';

/** Embedder identity classes. `modelId` carries the exact model. */
export type EmbedderKind = 'onnx-minilm' | 'onnx' | 'hash';

export interface EmbeddingProvenance {
  /** Embedder family that produced the vectors. */
  embedderKind: EmbedderKind | string;
  /** Exact model id (e.g. 'all-MiniLM-L6-v2'); null for the hash embedder. */
  modelId: string | null;
  /** Vector dimension. */
  dimension: number;
  /** Whether vectors were L2-normalized at embed time. */
  normalize: boolean;
  /** Prefix convention the texts were embedded under (D4). */
  prefixPolicy: PrefixPolicy;
}

export type EmbedderSelection = 'auto' | 'minilm' | 'hash';
export type ReembedPolicy = 'refuse' | 'warn' | 'auto';

// ============================================================================
// D4 — per-model prefix policies (facts from the model cards cited in ADR-210)
// ============================================================================

export interface ModelPrefixSpec {
  prefixPolicy: PrefixPolicy;
  queryPrefix: string;
  passagePrefix: string;
}

const NO_PREFIX: ModelPrefixSpec = { prefixPolicy: 'none', queryPrefix: '', passagePrefix: '' };

/** BGE en v1.5 documented query instruction (short query → long passage). */
export const BGE_QUERY_INSTRUCTION =
  'Represent this sentence for searching relevant passages: ';

/**
 * Prefix conventions per model card:
 * - all-MiniLM-L6-v2 / L12: general semantic search, NO prefixes.
 * - e5-small-v2: REQUIRES 'query: ' / 'passage: ' (quality degrades without).
 * - bge-small/base-en-v1.5: query instruction recommended for retrieval;
 *   passages need no instruction.
 * - gte-small: no prefixes documented.
 */
export const MODEL_PREFIXES: Record<string, ModelPrefixSpec> = {
  'all-MiniLM-L6-v2': { ...NO_PREFIX },
  'all-MiniLM-L12-v2': { ...NO_PREFIX },
  'e5-small-v2': { prefixPolicy: 'required', queryPrefix: 'query: ', passagePrefix: 'passage: ' },
  'bge-small-en-v1.5': { prefixPolicy: 'query-recommended', queryPrefix: BGE_QUERY_INSTRUCTION, passagePrefix: '' },
  'bge-base-en-v1.5': { prefixPolicy: 'query-recommended', queryPrefix: BGE_QUERY_INSTRUCTION, passagePrefix: '' },
  'gte-small': { ...NO_PREFIX },
};

/** Prefix spec for a model; unknown models get the no-prefix policy. */
export function getModelPrefixSpec(modelId: string | null | undefined): ModelPrefixSpec {
  return (modelId && MODEL_PREFIXES[modelId]) || NO_PREFIX;
}

/**
 * Pure prefix application (D4): the exact text handed to the tokenizer for a
 * query/passage embed of `text` under `modelId`'s registered policy.
 * MiniLM applies NO prefix on either entry point (acceptance gates 6–7).
 */
export function prefixText(modelId: string | null | undefined, kind: EmbedTextKind, text: string): string {
  const spec = getModelPrefixSpec(modelId);
  const prefix = kind === 'query' ? spec.queryPrefix : spec.passagePrefix;
  return prefix ? prefix + text : text;
}

/** Embedder family for an ONNX model id. */
export function embedderKindForModel(modelId: string | null | undefined): EmbedderKind {
  return modelId && modelId.startsWith('all-MiniLM') ? 'onnx-minilm' : 'onnx';
}

// ============================================================================
// D0 — provenance compare / refuse / legacy derivation
// ============================================================================

/**
 * Legacy default for stores that predate provenance metadata: hash-embedded,
 * un-normalized as far as we can prove, no prefixes. Such stores open
 * READ-ONLY for vector writes until re-embedded.
 */
export function legacyHashProvenance(dimension: number = 256): EmbeddingProvenance {
  return { embedderKind: 'hash', modelId: null, dimension, normalize: false, prefixPolicy: 'none' };
}

/** Human-readable one-liner for error messages. */
export function describeProvenance(p: EmbeddingProvenance): string {
  const model = p.modelId ? `, model=${p.modelId}` : '';
  return `{ embedder=${p.embedderKind}${model}, dim=${p.dimension}, normalize=${p.normalize}, prefixPolicy=${p.prefixPolicy} }`;
}

/** Field names on which two provenance records disagree (empty = match). */
export function compareProvenance(a: EmbeddingProvenance, b: EmbeddingProvenance): string[] {
  const mismatches: string[] = [];
  if (a.embedderKind !== b.embedderKind) mismatches.push('embedderKind');
  if ((a.modelId ?? null) !== (b.modelId ?? null)) mismatches.push('modelId');
  if (a.dimension !== b.dimension) mismatches.push('dimension');
  if (!!a.normalize !== !!b.normalize) mismatches.push('normalize');
  if (a.prefixPolicy !== b.prefixPolicy) mismatches.push('prefixPolicy');
  return mismatches;
}

/** Thrown when an insert's provenance does not match the store's (D0). */
export class ProvenanceMismatchError extends Error {
  code = 'ERR_EMBEDDING_PROVENANCE';
  store: EmbeddingProvenance;
  active: EmbeddingProvenance;
  mismatches: string[];

  constructor(store: EmbeddingProvenance, active: EmbeddingProvenance, mismatches: string[], storeName: string) {
    super(
      `Embedding-provenance mismatch (ADR-210): refusing vector write to ${storeName}. ` +
      `Store records ${describeProvenance(store)} but the active embedder is ` +
      `${describeProvenance(active)} (differs on: ${mismatches.join(', ')}). ` +
      `Mixed stores are never created — re-embed the store ('ruvector hooks reembed') ` +
      `or switch the active embedder (RUVECTOR_EMBEDDER=auto|minilm|hash).`
    );
    this.name = 'ProvenanceMismatchError';
    this.store = store;
    this.active = active;
    this.mismatches = mismatches;
  }
}

/** Refuse mismatched inserts with an error naming both sides (D0). */
export function assertProvenanceMatch(
  store: EmbeddingProvenance,
  active: EmbeddingProvenance,
  storeName: string = 'vector store'
): void {
  const mismatches = compareProvenance(store, active);
  if (mismatches.length > 0) {
    throw new ProvenanceMismatchError(store, active, mismatches, storeName);
  }
}

// ============================================================================
// D5 — rollout flags (env overrides config)
// ============================================================================

/**
 * Resolve RUVECTOR_EMBEDDER / RUVECTOR_ONNX.
 * Precedence: RUVECTOR_EMBEDDER wins when both are set; RUVECTOR_ONNX=0 is
 * shorthand for `hash`, =1 for `minilm`. Unrecognized values fall back to
 * 'auto' (MiniLM when loadable, loud hash fallback otherwise).
 */
export function resolveEmbedderSelection(env: NodeJS.ProcessEnv = process.env): EmbedderSelection {
  const embedder = (env.RUVECTOR_EMBEDDER || '').trim().toLowerCase();
  if (embedder === 'auto' || embedder === 'minilm' || embedder === 'hash') return embedder;
  const onnx = (env.RUVECTOR_ONNX || '').trim();
  if (onnx === '0') return 'hash';
  if (onnx === '1') return 'minilm';
  return 'auto';
}

/**
 * Resolve RUVECTOR_REEMBED: what happens when opening a store whose
 * provenance mismatches the active embedder.
 *   refuse (default) — error;
 *   warn             — open read-only with a single warning;
 *   auto             — re-embed in place when source text exists, refuse otherwise.
 */
export function resolveReembedPolicy(env: NodeJS.ProcessEnv = process.env): ReembedPolicy {
  const v = (env.RUVECTOR_REEMBED || '').trim().toLowerCase();
  if (v === 'refuse' || v === 'warn' || v === 'auto') return v;
  return 'refuse';
}

// ============================================================================
// D1 — loud (but once-per-process) hash-fallback warning
// ============================================================================

let fallbackWarned = false;

/**
 * Emit exactly ONE stderr warning per process the first time the hash
 * fallback serves an embed that the ONNX embedder was supposed to handle
 * (acceptance gate 2). Returns true when the warning was emitted by this call.
 */
export function warnHashFallbackOnce(reason?: string): boolean {
  if (fallbackWarned) return false;
  fallbackWarned = true;
  const detail = reason ? ` Reason: ${reason}.` : '';
  process.stderr.write(
    `ruvector: ONNX semantic embedder unavailable — using deterministic hash-fallback embeddings ` +
    `(no semantic signal, reduced search quality).${detail} ` +
    `Set RUVECTOR_EMBEDDER=hash to silence this or RUVECTOR_EMBEDDER=minilm to hard-require the model. ` +
    `(warned once per process)\n`
  );
  return true;
}

/** Whether the once-per-process fallback warning has fired. */
export function hashFallbackWarned(): boolean {
  return fallbackWarned;
}

/** Test hook: reset the once-per-process warning latch. */
export function resetHashFallbackWarningForTests(): void {
  fallbackWarned = false;
}

export default {
  MODEL_PREFIXES,
  BGE_QUERY_INSTRUCTION,
  getModelPrefixSpec,
  prefixText,
  embedderKindForModel,
  legacyHashProvenance,
  describeProvenance,
  compareProvenance,
  ProvenanceMismatchError,
  assertProvenanceMatch,
  resolveEmbedderSelection,
  resolveReembedPolicy,
  warnHashFallbackOnce,
  hashFallbackWarned,
  resetHashFallbackWarningForTests,
};
