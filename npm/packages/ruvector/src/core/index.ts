/**
 * Core module exports — lazy barrel (ADR-274 phase 2).
 *
 * These wrappers provide safe, type-flexible interfaces to the underlying
 * native packages, handling array type conversions automatically.
 *
 * Value exports are installed as memoized lazy getters so that requiring this
 * barrel does not eagerly evaluate the ~25 heavy core modules; each module is
 * loaded on first property access. Type exports are re-exported normally
 * (erased at compile time, zero runtime cost).
 *
 * NOTE: this file is structured so that its emitted JS contains NO
 * `__exportStar(require(...))` calls for value modules — that helper would
 * reintroduce the eager loading this barrel exists to avoid. When adding a new
 * module, add its type exports to the `export type` section, its value names
 * to a `defineLazyExports` call, and the value names to the
 * `0 && (module.exports = ...)` annotation (which keeps ESM named imports
 * working via cjs-module-lexer).
 */

// ─────────────────────────────────────────────────────────────────────────────
// Runtime: memoized lazy module loading (mirrors router-wrapper.ts pattern)
// ─────────────────────────────────────────────────────────────────────────────

const lazyModuleCache: Record<string, any> = Object.create(null);

function loadModule(id: string): any {
  if (!(id in lazyModuleCache)) {
    lazyModuleCache[id] = require(id);
  }
  return lazyModuleCache[id];
}

/** Install enumerable lazy getters on this module's exports. */
function defineLazyExports(id: string, names: readonly string[]): void {
  for (const name of names) {
    Object.defineProperty(module.exports, name, {
      enumerable: true,
      configurable: true,
      get: () => loadModule(id)[name],
    });
  }
}

/** Install a single lazy getter that renames its source export. */
function defineLazyAlias(name: string, id: string, sourceName: string): void {
  Object.defineProperty(module.exports, name, {
    enumerable: true,
    configurable: true,
    get: () => loadModule(id)[sourceName],
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy value exports (one memoized require per module, getters per name)
// ─────────────────────────────────────────────────────────────────────────────

defineLazyExports('./gnn-wrapper', [
  'toFloat32Array', 'toFloat32ArrayBatch', 'differentiableSearch', 'hierarchicalForward',
  'getCompressionLevel', 'isGnnAvailable', 'RuvectorLayer',
]);
defineLazyExports('./attention-fallbacks', [
  'projectToPoincareBall', 'poincareDistance', 'mobiusAddition', 'expMap', 'logMap',
  'isAttentionAvailable', 'getAttentionVersion', 'parallelAttentionCompute',
  'batchAttentionCompute', 'computeFlashAttentionAsync', 'computeHyperbolicAttentionAsync',
  'infoNceLoss', 'mineHardNegatives', 'benchmarkAttention', 'MultiHeadAttention', 'FlashAttention',
  'HyperbolicAttention', 'LinearAttention', 'LocalGlobalAttention', 'MoEAttention',
  'GraphRoPeAttention', 'EdgeFeaturedAttention', 'DualSpaceAttention', 'DotProductAttention',
  'AdamOptimizer',
]);
defineLazyExports('./agentdb-fast', [
  'createFastAgentDB', 'getDefaultAgentDB', 'FastAgentDB',
]);
defineLazyExports('./sona-wrapper', [
  'isSonaAvailable', 'SonaEngine',
]);
defineLazyExports('./intelligence-engine', [
  'createIntelligenceEngine', 'createHighPerformanceEngine', 'createLightweightEngine',
]);
defineLazyExports('./embedding-provenance', [
  'getModelPrefixSpec', 'prefixText', 'embedderKindForModel', 'legacyHashProvenance',
  'describeProvenance', 'compareProvenance', 'sanitizeProvenance', 'assertProvenanceMatch',
  'resolveEmbedderSelection', 'resolveReembedPolicy', 'warnHashFallbackOnce', 'hashFallbackWarned',
  'resetHashFallbackWarningForTests', 'BGE_QUERY_INSTRUCTION', 'MODEL_PREFIXES',
  'MAX_PROVENANCE_DIMENSION', 'ProvenanceMismatchError',
]);
defineLazyExports('./onnx-embedder', [
  'isOnnxAvailable', 'initOnnxEmbedder', 'embed', 'embedQuery', 'embedPassage', 'embedBatch',
  'similarity', 'cosineSimilarity', 'getDimension', 'isReady', 'isOnnxInitialized',
  'getActiveModelId', 'getEmbedderProvenance', 'getStats', 'shutdown', 'initParallelEmbedder',
  'embedBatchParallel', 'getParallelWorkerCount', 'embedBulk', 'shutdownParallelEmbedder',
  'BULK_EMBED_THRESHOLD',
]);
defineLazyExports('./onnx-optimized', [
  'getOptimizedOnnxEmbedder', 'initOptimizedOnnx',
]);
defineLazyExports('./parallel-intelligence', [
  'getParallelIntelligence', 'initParallelIntelligence',
]);
defineLazyExports('./parallel-workers', [
  'getExtendedWorkerPool', 'initExtendedWorkerPool',
]);
defineLazyExports('./router-wrapper', [
  'isRouterAvailable', 'createAgentRouter',
]);
defineLazyExports('./graph-wrapper', [
  'isGraphAvailable', 'createCodeDependencyGraph',
]);
defineLazyExports('./cluster-wrapper', [
  'isClusterAvailable', 'createCluster',
]);
defineLazyExports('./ast-parser', [
  'isTreeSitterAvailable', 'getCodeParser', 'initCodeParser',
]);
defineLazyExports('./diff-embeddings', [
  'parseDiff', 'classifyChange', 'calculateRiskScore', 'analyzeFileDiff', 'getCommitDiff',
  'getStagedDiff', 'getUnstagedDiff', 'analyzeCommit', 'findSimilarCommits',
]);
defineLazyExports('./coverage-router', [
  'parseIstanbulCoverage', 'findCoverageReport', 'getFileCoverage', 'suggestTests',
  'shouldRouteToTester', 'getCoverageRoutingWeight',
]);
defineLazyExports('./graph-algorithms', [
  'buildGraph', 'minCut', 'spectralClustering', 'louvainCommunities', 'calculateModularity',
  'findBridges', 'findArticulationPoints',
]);
defineLazyExports('./adaptive-embedder', [
  'getAdaptiveEmbedder', 'initAdaptiveEmbedder',
]);
defineLazyExports('./neural-embeddings', [
  'NEURAL_CONSTANTS', 'defaultLogger', 'silentLogger', 'SemanticDriftDetector', 'MemoryPhysics',
  'EmbeddingStateMachine', 'SwarmCoordinator', 'CoherenceMonitor',
]);
defineLazyExports('./neural-perf', [
  'PERF_CONSTANTS', 'LRUCache', 'Float32BufferPool', 'TensorBufferManager', 'VectorOps',
  'ParallelBatchProcessor', 'OptimizedMemoryStore',
]);
defineLazyExports('./rvf-wrapper', [
  'isRvfAvailable', 'createRvfStore', 'openRvfStore', 'rvfIngest', 'rvfQuery', 'rvfDelete',
  'rvfStatus', 'rvfCompact', 'rvfDerive', 'rvfClose',
]);
defineLazyExports('./diskann-wrapper', [
  'isDiskAnnAvailable',
]);
defineLazyExports('../analysis', [
  'security', 'complexity', 'patterns', 'scanFile', 'scanFiles', 'getSeverityScore',
  'sortBySeverity', 'SECURITY_PATTERNS', 'analyzeFile', 'analyzeFiles', 'exceedsThresholds',
  'getComplexityRating', 'filterComplex', 'DEFAULT_THRESHOLDS', 'detectLanguage',
  'extractFunctions', 'extractClasses', 'extractImports', 'extractExports', 'extractTodos',
  'extractAllPatterns', 'extractFromFiles', 'toPatternMatches',
]);

// Re-exported default objects / renamed exports (backward compatible)
defineLazyAlias('gnnWrapper', './gnn-wrapper', 'default');
defineLazyAlias('attentionFallbacks', './attention-fallbacks', 'default');
defineLazyAlias('agentdbFast', './agentdb-fast', 'default');
defineLazyAlias('Sona', './sona-wrapper', 'default');
defineLazyAlias('IntelligenceEngine', './intelligence-engine', 'default');
defineLazyAlias('OnnxEmbedder', './onnx-embedder', 'default');
defineLazyAlias('OptimizedOnnxEmbedder', './onnx-optimized', 'default');
defineLazyAlias('ParallelIntelligence', './parallel-intelligence', 'default');
defineLazyAlias('ExtendedWorkerPool', './parallel-workers', 'default');
defineLazyAlias('SemanticRouter', './router-wrapper', 'default');
defineLazyAlias('CodeGraph', './graph-wrapper', 'default');
defineLazyAlias('RuvectorCluster', './cluster-wrapper', 'default');
defineLazyAlias('CodeParser', './ast-parser', 'default');
defineLazyAlias('ASTParser', './ast-parser', 'CodeParser');
defineLazyAlias('TensorCompress', './tensor-compress', 'default');
defineLazyAlias('LearningEngine', './learning-engine', 'default');
defineLazyAlias('AdaptiveEmbedder', './adaptive-embedder', 'default');
defineLazyAlias('NeuralSubstrate', './neural-embeddings', 'default');
defineLazyAlias('DiskAnnIndex', './diskann-wrapper', 'default');

// ─────────────────────────────────────────────────────────────────────────────
// Type-only re-exports (erased at runtime)
// ─────────────────────────────────────────────────────────────────────────────

export type {
  DifferentiableSearchResult,
} from './gnn-wrapper';
export type {
  AttentionOutput, MoEConfig,
} from './attention-fallbacks';
export type {
  Episode, Trajectory, EpisodeSearchResult,
} from './agentdb-fast';
export type {
  ArrayInput, SonaConfig, LearnedPattern, SonaStats,
} from './sona-wrapper';
export type {
  MemoryEntry, AgentRoute, LearningStats, IntelligenceConfig,
} from './intelligence-engine';
export type {
  PrefixPolicy, EmbedTextKind, EmbedderKind, EmbeddingProvenance, EmbedderSelection,
  ReembedPolicy, ModelPrefixSpec,
} from './embedding-provenance';
export type {
  OnnxEmbedderConfig, EmbeddingResult, SimilarityResult,
} from './onnx-embedder';
export type {
  OptimizedOnnxConfig,
} from './onnx-optimized';
export type {
  ParallelConfig, BatchEpisode, PatternMatchResult, CoEditAnalysis,
} from './parallel-intelligence';
export type {
  WorkerPoolConfig, SpeculativeEmbedding, ASTAnalysis, SecurityFinding, ContextChunk, GitBlame,
  CodeChurn,
} from './parallel-workers';
export type {
  Route, RouteMatch,
} from './router-wrapper';
export type {
  Node, Edge, Hyperedge, CypherResult, PathResult,
} from './graph-wrapper';
export type {
  ClusterNode, ShardInfo, ClusterConfig,
} from './cluster-wrapper';
export type {
  ASTNode, FunctionInfo, ClassInfo, ImportInfo, ExportInfo, FileAnalysis,
} from './ast-parser';
export type {
  DiffHunk, DiffAnalysis, CommitAnalysis,
} from './diff-embeddings';
export type {
  CoverageData, CoverageSummary, TestSuggestion,
} from './coverage-router';
export type {
  Graph, Partition, SpectralResult,
} from './graph-algorithms';
export type {
  CompressionLevel, CompressedTensor, CompressionStats, CompressionConfig,
} from './tensor-compress';
export type {
  LearningAlgorithm, TaskType, LearningConfig, Experience, LearningTrajectory, AlgorithmStats,
} from './learning-engine';
export type {
  AdaptiveConfig, LoRAWeights, DomainPrototype, AdaptiveStats,
} from './adaptive-embedder';
export type {
  LogLevel, NeuralLogger, DriftEvent, NeuralMemoryEntry, AgentState, CoherenceReport,
  NeuralConfig,
} from './neural-embeddings';
export type {
  BatchResult, CachedMemoryEntry,
} from './neural-perf';
export type {
  RvfStoreOptions, RvfEntry, RvfResult, RvfStoreStatus, RvfQueryOpts, RvfStore,
} from './rvf-wrapper';
export type {
  DiskAnnConfig, DiskAnnSearchResult,
} from './diskann-wrapper';
export type {
  SecurityPattern, ComplexityResult, ComplexityThresholds, PatternMatch, FilePatterns,
} from '../analysis';

// ─────────────────────────────────────────────────────────────────────────────
// Typed declarations for the lazy value exports above (no runtime emit)
// ─────────────────────────────────────────────────────────────────────────────

export declare const toFloat32Array: typeof import('./gnn-wrapper').toFloat32Array;
export declare const toFloat32ArrayBatch: typeof import('./gnn-wrapper').toFloat32ArrayBatch;
export declare const differentiableSearch: typeof import('./gnn-wrapper').differentiableSearch;
export declare const hierarchicalForward: typeof import('./gnn-wrapper').hierarchicalForward;
export declare const getCompressionLevel: typeof import('./gnn-wrapper').getCompressionLevel;
export declare const isGnnAvailable: typeof import('./gnn-wrapper').isGnnAvailable;
export declare const RuvectorLayer: typeof import('./gnn-wrapper').RuvectorLayer;
export type RuvectorLayer = import('./gnn-wrapper').RuvectorLayer;
export declare const TensorCompress: typeof import('./tensor-compress').default;
export type TensorCompress = import('./tensor-compress').default;
export declare const projectToPoincareBall: typeof import('./attention-fallbacks').projectToPoincareBall;
export declare const poincareDistance: typeof import('./attention-fallbacks').poincareDistance;
export declare const mobiusAddition: typeof import('./attention-fallbacks').mobiusAddition;
export declare const expMap: typeof import('./attention-fallbacks').expMap;
export declare const logMap: typeof import('./attention-fallbacks').logMap;
export declare const isAttentionAvailable: typeof import('./attention-fallbacks').isAttentionAvailable;
export declare const getAttentionVersion: typeof import('./attention-fallbacks').getAttentionVersion;
export declare const parallelAttentionCompute: typeof import('./attention-fallbacks').parallelAttentionCompute;
export declare const batchAttentionCompute: typeof import('./attention-fallbacks').batchAttentionCompute;
export declare const computeFlashAttentionAsync: typeof import('./attention-fallbacks').computeFlashAttentionAsync;
export declare const computeHyperbolicAttentionAsync: typeof import('./attention-fallbacks').computeHyperbolicAttentionAsync;
export declare const infoNceLoss: typeof import('./attention-fallbacks').infoNceLoss;
export declare const mineHardNegatives: typeof import('./attention-fallbacks').mineHardNegatives;
export declare const benchmarkAttention: typeof import('./attention-fallbacks').benchmarkAttention;
export declare const MultiHeadAttention: typeof import('./attention-fallbacks').MultiHeadAttention;
export type MultiHeadAttention = import('./attention-fallbacks').MultiHeadAttention;
export declare const FlashAttention: typeof import('./attention-fallbacks').FlashAttention;
export type FlashAttention = import('./attention-fallbacks').FlashAttention;
export declare const HyperbolicAttention: typeof import('./attention-fallbacks').HyperbolicAttention;
export type HyperbolicAttention = import('./attention-fallbacks').HyperbolicAttention;
export declare const LinearAttention: typeof import('./attention-fallbacks').LinearAttention;
export type LinearAttention = import('./attention-fallbacks').LinearAttention;
export declare const LocalGlobalAttention: typeof import('./attention-fallbacks').LocalGlobalAttention;
export type LocalGlobalAttention = import('./attention-fallbacks').LocalGlobalAttention;
export declare const MoEAttention: typeof import('./attention-fallbacks').MoEAttention;
export type MoEAttention = import('./attention-fallbacks').MoEAttention;
export declare const GraphRoPeAttention: typeof import('./attention-fallbacks').GraphRoPeAttention;
export type GraphRoPeAttention = import('./attention-fallbacks').GraphRoPeAttention;
export declare const EdgeFeaturedAttention: typeof import('./attention-fallbacks').EdgeFeaturedAttention;
export type EdgeFeaturedAttention = import('./attention-fallbacks').EdgeFeaturedAttention;
export declare const DualSpaceAttention: typeof import('./attention-fallbacks').DualSpaceAttention;
export type DualSpaceAttention = import('./attention-fallbacks').DualSpaceAttention;
export declare const DotProductAttention: typeof import('./attention-fallbacks').DotProductAttention;
export type DotProductAttention = import('./attention-fallbacks').DotProductAttention;
export declare const AdamOptimizer: typeof import('./attention-fallbacks').AdamOptimizer;
export type AdamOptimizer = import('./attention-fallbacks').AdamOptimizer;
export declare const createFastAgentDB: typeof import('./agentdb-fast').createFastAgentDB;
export declare const getDefaultAgentDB: typeof import('./agentdb-fast').getDefaultAgentDB;
export declare const FastAgentDB: typeof import('./agentdb-fast').FastAgentDB;
export type FastAgentDB = import('./agentdb-fast').FastAgentDB;
export declare const isSonaAvailable: typeof import('./sona-wrapper').isSonaAvailable;
export declare const SonaEngine: typeof import('./sona-wrapper').SonaEngine;
export type SonaEngine = import('./sona-wrapper').SonaEngine;
export declare const Sona: typeof import('./sona-wrapper').default;
export declare const createIntelligenceEngine: typeof import('./intelligence-engine').createIntelligenceEngine;
export declare const createHighPerformanceEngine: typeof import('./intelligence-engine').createHighPerformanceEngine;
export declare const createLightweightEngine: typeof import('./intelligence-engine').createLightweightEngine;
export declare const IntelligenceEngine: typeof import('./intelligence-engine').default;
export type IntelligenceEngine = import('./intelligence-engine').default;
export declare const getModelPrefixSpec: typeof import('./embedding-provenance').getModelPrefixSpec;
export declare const prefixText: typeof import('./embedding-provenance').prefixText;
export declare const embedderKindForModel: typeof import('./embedding-provenance').embedderKindForModel;
export declare const legacyHashProvenance: typeof import('./embedding-provenance').legacyHashProvenance;
export declare const describeProvenance: typeof import('./embedding-provenance').describeProvenance;
export declare const compareProvenance: typeof import('./embedding-provenance').compareProvenance;
export declare const sanitizeProvenance: typeof import('./embedding-provenance').sanitizeProvenance;
export declare const assertProvenanceMatch: typeof import('./embedding-provenance').assertProvenanceMatch;
export declare const resolveEmbedderSelection: typeof import('./embedding-provenance').resolveEmbedderSelection;
export declare const resolveReembedPolicy: typeof import('./embedding-provenance').resolveReembedPolicy;
export declare const warnHashFallbackOnce: typeof import('./embedding-provenance').warnHashFallbackOnce;
export declare const hashFallbackWarned: typeof import('./embedding-provenance').hashFallbackWarned;
export declare const resetHashFallbackWarningForTests: typeof import('./embedding-provenance').resetHashFallbackWarningForTests;
export declare const BGE_QUERY_INSTRUCTION: typeof import('./embedding-provenance').BGE_QUERY_INSTRUCTION;
export declare const MODEL_PREFIXES: typeof import('./embedding-provenance').MODEL_PREFIXES;
export declare const MAX_PROVENANCE_DIMENSION: typeof import('./embedding-provenance').MAX_PROVENANCE_DIMENSION;
export declare const ProvenanceMismatchError: typeof import('./embedding-provenance').ProvenanceMismatchError;
export type ProvenanceMismatchError = import('./embedding-provenance').ProvenanceMismatchError;
export declare const isOnnxAvailable: typeof import('./onnx-embedder').isOnnxAvailable;
export declare const initOnnxEmbedder: typeof import('./onnx-embedder').initOnnxEmbedder;
export declare const embed: typeof import('./onnx-embedder').embed;
export declare const embedQuery: typeof import('./onnx-embedder').embedQuery;
export declare const embedPassage: typeof import('./onnx-embedder').embedPassage;
export declare const embedBatch: typeof import('./onnx-embedder').embedBatch;
export declare const similarity: typeof import('./onnx-embedder').similarity;
export declare const cosineSimilarity: typeof import('./onnx-embedder').cosineSimilarity;
export declare const getDimension: typeof import('./onnx-embedder').getDimension;
export declare const isReady: typeof import('./onnx-embedder').isReady;
export declare const isOnnxInitialized: typeof import('./onnx-embedder').isOnnxInitialized;
export declare const getActiveModelId: typeof import('./onnx-embedder').getActiveModelId;
export declare const getEmbedderProvenance: typeof import('./onnx-embedder').getEmbedderProvenance;
export declare const getStats: typeof import('./onnx-embedder').getStats;
export declare const shutdown: typeof import('./onnx-embedder').shutdown;
export declare const initParallelEmbedder: typeof import('./onnx-embedder').initParallelEmbedder;
export declare const embedBatchParallel: typeof import('./onnx-embedder').embedBatchParallel;
export declare const getParallelWorkerCount: typeof import('./onnx-embedder').getParallelWorkerCount;
export declare const embedBulk: typeof import('./onnx-embedder').embedBulk;
export declare const shutdownParallelEmbedder: typeof import('./onnx-embedder').shutdownParallelEmbedder;
export declare const BULK_EMBED_THRESHOLD: typeof import('./onnx-embedder').BULK_EMBED_THRESHOLD;
export declare const OnnxEmbedder: typeof import('./onnx-embedder').default;
export type OnnxEmbedder = import('./onnx-embedder').default;
export declare const getOptimizedOnnxEmbedder: typeof import('./onnx-optimized').getOptimizedOnnxEmbedder;
export declare const initOptimizedOnnx: typeof import('./onnx-optimized').initOptimizedOnnx;
export declare const OptimizedOnnxEmbedder: typeof import('./onnx-optimized').default;
export type OptimizedOnnxEmbedder = import('./onnx-optimized').default;
export declare const getParallelIntelligence: typeof import('./parallel-intelligence').getParallelIntelligence;
export declare const initParallelIntelligence: typeof import('./parallel-intelligence').initParallelIntelligence;
export declare const ParallelIntelligence: typeof import('./parallel-intelligence').default;
export type ParallelIntelligence = import('./parallel-intelligence').default;
export declare const getExtendedWorkerPool: typeof import('./parallel-workers').getExtendedWorkerPool;
export declare const initExtendedWorkerPool: typeof import('./parallel-workers').initExtendedWorkerPool;
export declare const ExtendedWorkerPool: typeof import('./parallel-workers').default;
export type ExtendedWorkerPool = import('./parallel-workers').default;
export declare const isRouterAvailable: typeof import('./router-wrapper').isRouterAvailable;
export declare const createAgentRouter: typeof import('./router-wrapper').createAgentRouter;
export declare const SemanticRouter: typeof import('./router-wrapper').default;
export type SemanticRouter = import('./router-wrapper').default;
export declare const isGraphAvailable: typeof import('./graph-wrapper').isGraphAvailable;
export declare const createCodeDependencyGraph: typeof import('./graph-wrapper').createCodeDependencyGraph;
export declare const CodeGraph: typeof import('./graph-wrapper').default;
export type CodeGraph = import('./graph-wrapper').default;
export declare const isClusterAvailable: typeof import('./cluster-wrapper').isClusterAvailable;
export declare const createCluster: typeof import('./cluster-wrapper').createCluster;
export declare const RuvectorCluster: typeof import('./cluster-wrapper').default;
export type RuvectorCluster = import('./cluster-wrapper').default;
export declare const isTreeSitterAvailable: typeof import('./ast-parser').isTreeSitterAvailable;
export declare const getCodeParser: typeof import('./ast-parser').getCodeParser;
export declare const initCodeParser: typeof import('./ast-parser').initCodeParser;
export declare const CodeParser: typeof import('./ast-parser').default;
export type CodeParser = import('./ast-parser').default;
export declare const parseDiff: typeof import('./diff-embeddings').parseDiff;
export declare const classifyChange: typeof import('./diff-embeddings').classifyChange;
export declare const calculateRiskScore: typeof import('./diff-embeddings').calculateRiskScore;
export declare const analyzeFileDiff: typeof import('./diff-embeddings').analyzeFileDiff;
export declare const getCommitDiff: typeof import('./diff-embeddings').getCommitDiff;
export declare const getStagedDiff: typeof import('./diff-embeddings').getStagedDiff;
export declare const getUnstagedDiff: typeof import('./diff-embeddings').getUnstagedDiff;
export declare const analyzeCommit: typeof import('./diff-embeddings').analyzeCommit;
export declare const findSimilarCommits: typeof import('./diff-embeddings').findSimilarCommits;
export declare const parseIstanbulCoverage: typeof import('./coverage-router').parseIstanbulCoverage;
export declare const findCoverageReport: typeof import('./coverage-router').findCoverageReport;
export declare const getFileCoverage: typeof import('./coverage-router').getFileCoverage;
export declare const suggestTests: typeof import('./coverage-router').suggestTests;
export declare const shouldRouteToTester: typeof import('./coverage-router').shouldRouteToTester;
export declare const getCoverageRoutingWeight: typeof import('./coverage-router').getCoverageRoutingWeight;
export declare const buildGraph: typeof import('./graph-algorithms').buildGraph;
export declare const minCut: typeof import('./graph-algorithms').minCut;
export declare const spectralClustering: typeof import('./graph-algorithms').spectralClustering;
export declare const louvainCommunities: typeof import('./graph-algorithms').louvainCommunities;
export declare const calculateModularity: typeof import('./graph-algorithms').calculateModularity;
export declare const findBridges: typeof import('./graph-algorithms').findBridges;
export declare const findArticulationPoints: typeof import('./graph-algorithms').findArticulationPoints;
export declare const LearningEngine: typeof import('./learning-engine').default;
export type LearningEngine = import('./learning-engine').default;
export declare const getAdaptiveEmbedder: typeof import('./adaptive-embedder').getAdaptiveEmbedder;
export declare const initAdaptiveEmbedder: typeof import('./adaptive-embedder').initAdaptiveEmbedder;
export declare const AdaptiveEmbedder: typeof import('./adaptive-embedder').default;
export type AdaptiveEmbedder = import('./adaptive-embedder').default;
export declare const NEURAL_CONSTANTS: typeof import('./neural-embeddings').NEURAL_CONSTANTS;
export declare const defaultLogger: typeof import('./neural-embeddings').defaultLogger;
export declare const silentLogger: typeof import('./neural-embeddings').silentLogger;
export declare const SemanticDriftDetector: typeof import('./neural-embeddings').SemanticDriftDetector;
export type SemanticDriftDetector = import('./neural-embeddings').SemanticDriftDetector;
export declare const MemoryPhysics: typeof import('./neural-embeddings').MemoryPhysics;
export type MemoryPhysics = import('./neural-embeddings').MemoryPhysics;
export declare const EmbeddingStateMachine: typeof import('./neural-embeddings').EmbeddingStateMachine;
export type EmbeddingStateMachine = import('./neural-embeddings').EmbeddingStateMachine;
export declare const SwarmCoordinator: typeof import('./neural-embeddings').SwarmCoordinator;
export type SwarmCoordinator = import('./neural-embeddings').SwarmCoordinator;
export declare const CoherenceMonitor: typeof import('./neural-embeddings').CoherenceMonitor;
export type CoherenceMonitor = import('./neural-embeddings').CoherenceMonitor;
export declare const NeuralSubstrate: typeof import('./neural-embeddings').default;
export type NeuralSubstrate = import('./neural-embeddings').default;
export declare const PERF_CONSTANTS: typeof import('./neural-perf').PERF_CONSTANTS;
export declare const LRUCache: typeof import('./neural-perf').LRUCache;
export type LRUCache<K, V> = import('./neural-perf').LRUCache<K, V>;
export declare const Float32BufferPool: typeof import('./neural-perf').Float32BufferPool;
export type Float32BufferPool = import('./neural-perf').Float32BufferPool;
export declare const TensorBufferManager: typeof import('./neural-perf').TensorBufferManager;
export type TensorBufferManager = import('./neural-perf').TensorBufferManager;
export declare const VectorOps: typeof import('./neural-perf').VectorOps;
export declare const ParallelBatchProcessor: typeof import('./neural-perf').ParallelBatchProcessor;
export type ParallelBatchProcessor = import('./neural-perf').ParallelBatchProcessor;
export declare const OptimizedMemoryStore: typeof import('./neural-perf').OptimizedMemoryStore;
export type OptimizedMemoryStore = import('./neural-perf').OptimizedMemoryStore;
export declare const isRvfAvailable: typeof import('./rvf-wrapper').isRvfAvailable;
export declare const createRvfStore: typeof import('./rvf-wrapper').createRvfStore;
export declare const openRvfStore: typeof import('./rvf-wrapper').openRvfStore;
export declare const rvfIngest: typeof import('./rvf-wrapper').rvfIngest;
export declare const rvfQuery: typeof import('./rvf-wrapper').rvfQuery;
export declare const rvfDelete: typeof import('./rvf-wrapper').rvfDelete;
export declare const rvfStatus: typeof import('./rvf-wrapper').rvfStatus;
export declare const rvfCompact: typeof import('./rvf-wrapper').rvfCompact;
export declare const rvfDerive: typeof import('./rvf-wrapper').rvfDerive;
export declare const rvfClose: typeof import('./rvf-wrapper').rvfClose;
export declare const isDiskAnnAvailable: typeof import('./diskann-wrapper').isDiskAnnAvailable;
export declare const DiskAnnIndex: typeof import('./diskann-wrapper').default;
export type DiskAnnIndex = import('./diskann-wrapper').default;
export declare const security: typeof import('../analysis').security;
export declare const complexity: typeof import('../analysis').complexity;
export declare const patterns: typeof import('../analysis').patterns;
export declare const scanFile: typeof import('../analysis').scanFile;
export declare const scanFiles: typeof import('../analysis').scanFiles;
export declare const getSeverityScore: typeof import('../analysis').getSeverityScore;
export declare const sortBySeverity: typeof import('../analysis').sortBySeverity;
export declare const SECURITY_PATTERNS: typeof import('../analysis').SECURITY_PATTERNS;
export declare const analyzeFile: typeof import('../analysis').analyzeFile;
export declare const analyzeFiles: typeof import('../analysis').analyzeFiles;
export declare const exceedsThresholds: typeof import('../analysis').exceedsThresholds;
export declare const getComplexityRating: typeof import('../analysis').getComplexityRating;
export declare const filterComplex: typeof import('../analysis').filterComplex;
export declare const DEFAULT_THRESHOLDS: typeof import('../analysis').DEFAULT_THRESHOLDS;
export declare const detectLanguage: typeof import('../analysis').detectLanguage;
export declare const extractFunctions: typeof import('../analysis').extractFunctions;
export declare const extractClasses: typeof import('../analysis').extractClasses;
export declare const extractImports: typeof import('../analysis').extractImports;
export declare const extractExports: typeof import('../analysis').extractExports;
export declare const extractTodos: typeof import('../analysis').extractTodos;
export declare const extractAllPatterns: typeof import('../analysis').extractAllPatterns;
export declare const extractFromFiles: typeof import('../analysis').extractFromFiles;
export declare const toPatternMatches: typeof import('../analysis').toPatternMatches;
export declare const gnnWrapper: typeof import('./gnn-wrapper').default;
export declare const attentionFallbacks: typeof import('./attention-fallbacks').default;
export declare const agentdbFast: typeof import('./agentdb-fast').default;
export declare const ASTParser: typeof import('./ast-parser').CodeParser;
export type ASTParser = import('./ast-parser').CodeParser;

// Annotate the CommonJS export names for ESM import in node (cjs-module-lexer).
// The values must be plain local identifiers (NOT the exported names — tsc would
// rewrite those to `exports.X`, which the lexer does not recognise); the `0 &&`
// guard means this is never evaluated at runtime.
declare const _: unknown;
// eslint-disable-next-line no-constant-binary-expression, @typescript-eslint/no-unused-expressions
0 && (module.exports = {
  toFloat32Array: _, toFloat32ArrayBatch: _, differentiableSearch: _, hierarchicalForward: _,
  getCompressionLevel: _, isGnnAvailable: _, RuvectorLayer: _, TensorCompress: _,
  projectToPoincareBall: _, poincareDistance: _, mobiusAddition: _, expMap: _, logMap: _,
  isAttentionAvailable: _, getAttentionVersion: _, parallelAttentionCompute: _,
  batchAttentionCompute: _, computeFlashAttentionAsync: _, computeHyperbolicAttentionAsync: _,
  infoNceLoss: _, mineHardNegatives: _, benchmarkAttention: _, MultiHeadAttention: _,
  FlashAttention: _, HyperbolicAttention: _, LinearAttention: _, LocalGlobalAttention: _,
  MoEAttention: _, GraphRoPeAttention: _, EdgeFeaturedAttention: _, DualSpaceAttention: _,
  DotProductAttention: _, AdamOptimizer: _, createFastAgentDB: _, getDefaultAgentDB: _,
  FastAgentDB: _, isSonaAvailable: _, SonaEngine: _, Sona: _, createIntelligenceEngine: _,
  createHighPerformanceEngine: _, createLightweightEngine: _, IntelligenceEngine: _,
  getModelPrefixSpec: _, prefixText: _, embedderKindForModel: _, legacyHashProvenance: _,
  describeProvenance: _, compareProvenance: _, sanitizeProvenance: _, assertProvenanceMatch: _,
  resolveEmbedderSelection: _, resolveReembedPolicy: _, warnHashFallbackOnce: _,
  hashFallbackWarned: _, resetHashFallbackWarningForTests: _, BGE_QUERY_INSTRUCTION: _,
  MODEL_PREFIXES: _, MAX_PROVENANCE_DIMENSION: _, ProvenanceMismatchError: _, isOnnxAvailable: _,
  initOnnxEmbedder: _, embed: _, embedQuery: _, embedPassage: _, embedBatch: _, similarity: _,
  cosineSimilarity: _, getDimension: _, isReady: _, isOnnxInitialized: _, getActiveModelId: _,
  getEmbedderProvenance: _, getStats: _, shutdown: _, initParallelEmbedder: _,
  embedBatchParallel: _, getParallelWorkerCount: _, embedBulk: _, shutdownParallelEmbedder: _,
  BULK_EMBED_THRESHOLD: _, OnnxEmbedder: _, getOptimizedOnnxEmbedder: _, initOptimizedOnnx: _,
  OptimizedOnnxEmbedder: _, getParallelIntelligence: _, initParallelIntelligence: _,
  ParallelIntelligence: _, getExtendedWorkerPool: _, initExtendedWorkerPool: _,
  ExtendedWorkerPool: _, isRouterAvailable: _, createAgentRouter: _, SemanticRouter: _,
  isGraphAvailable: _, createCodeDependencyGraph: _, CodeGraph: _, isClusterAvailable: _,
  createCluster: _, RuvectorCluster: _, isTreeSitterAvailable: _, getCodeParser: _,
  initCodeParser: _, CodeParser: _, parseDiff: _, classifyChange: _, calculateRiskScore: _,
  analyzeFileDiff: _, getCommitDiff: _, getStagedDiff: _, getUnstagedDiff: _, analyzeCommit: _,
  findSimilarCommits: _, parseIstanbulCoverage: _, findCoverageReport: _, getFileCoverage: _,
  suggestTests: _, shouldRouteToTester: _, getCoverageRoutingWeight: _, buildGraph: _, minCut: _,
  spectralClustering: _, louvainCommunities: _, calculateModularity: _, findBridges: _,
  findArticulationPoints: _, LearningEngine: _, getAdaptiveEmbedder: _, initAdaptiveEmbedder: _,
  AdaptiveEmbedder: _, NEURAL_CONSTANTS: _, defaultLogger: _, silentLogger: _,
  SemanticDriftDetector: _, MemoryPhysics: _, EmbeddingStateMachine: _, SwarmCoordinator: _,
  CoherenceMonitor: _, NeuralSubstrate: _, PERF_CONSTANTS: _, LRUCache: _, Float32BufferPool: _,
  TensorBufferManager: _, VectorOps: _, ParallelBatchProcessor: _, OptimizedMemoryStore: _,
  isRvfAvailable: _, createRvfStore: _, openRvfStore: _, rvfIngest: _, rvfQuery: _, rvfDelete: _,
  rvfStatus: _, rvfCompact: _, rvfDerive: _, rvfClose: _, isDiskAnnAvailable: _, DiskAnnIndex: _,
  security: _, complexity: _, patterns: _, scanFile: _, scanFiles: _, getSeverityScore: _,
  sortBySeverity: _, SECURITY_PATTERNS: _, analyzeFile: _, analyzeFiles: _, exceedsThresholds: _,
  getComplexityRating: _, filterComplex: _, DEFAULT_THRESHOLDS: _, detectLanguage: _,
  extractFunctions: _, extractClasses: _, extractImports: _, extractExports: _, extractTodos: _,
  extractAllPatterns: _, extractFromFiles: _, toPatternMatches: _, gnnWrapper: _,
  attentionFallbacks: _, agentdbFast: _, ASTParser: _,
});
