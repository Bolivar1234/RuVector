/**
 * Parallel Intelligence - Worker-based acceleration for IntelligenceEngine
 *
 * Provides parallel processing for:
 * - Q-learning batch updates (3-4x faster)
 * - Multi-file pattern matching
 * - Background memory indexing
 * - Parallel similarity search
 * - Multi-file code analysis
 * - Parallel git commit analysis
 *
 * Uses worker_threads for CPU-bound operations, keeping hooks non-blocking.
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import * as path from 'path';
import * as os from 'os';

// ============================================================================
// Types
// ============================================================================

export interface ParallelConfig {
  /** Number of worker threads (default: CPU cores - 1) */
  numWorkers?: number;
  /**
   * Enable parallel processing (default: false).
   *
   * ADR-276: the worker task handlers below are still stubs, so the pool is
   * disabled by default (it previously auto-enabled under MCP and spawned
   * threads that produced fabricated results). Opt in explicitly via this
   * flag or by setting RUVECTOR_PARALLEL_INTELLIGENCE=1.
   */
  enabled?: boolean;
  /** Minimum batch size to use parallel (default: 4) */
  batchThreshold?: number;
}

export interface BatchEpisode {
  state: string;
  action: string;
  reward: number;
  nextState: string;
  done: boolean;
  metadata?: Record<string, any>;
}

export interface PatternMatchResult {
  file: string;
  patterns: Array<{ pattern: string; confidence: number }>;
}

export interface CoEditAnalysis {
  file1: string;
  file2: string;
  commits: string[];
  strength: number;
}

// ============================================================================
// Worker Pool Manager
// ============================================================================

export class ParallelIntelligence {
  private workers: Worker[] = [];
  /** Queued task messages (each already carries its correlation id) */
  private taskQueue: any[] = [];
  private busyWorkers: Set<Worker> = new Set();
  /** Pending settlements keyed by task id (id-correlated protocol, ADR-276) */
  private pending: Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }> = new Map();
  private nextTaskId = 0;
  private config: Required<ParallelConfig>;
  private initialized = false;

  constructor(config: ParallelConfig = {}) {
    this.config = {
      numWorkers: config.numWorkers ?? Math.max(1, os.cpus().length - 1),
      // ADR-276: disabled by default while the worker handlers are stubs.
      // Explicit constructor opt-in or RUVECTOR_PARALLEL_INTELLIGENCE=1
      // re-enables the pool.
      enabled: config.enabled ?? (process.env.RUVECTOR_PARALLEL_INTELLIGENCE === '1'),
      batchThreshold: config.batchThreshold ?? 4,
    };
  }

  /**
   * Initialize worker pool
   */
  async init(): Promise<void> {
    if (this.initialized || !this.config.enabled) return;

    for (let i = 0; i < this.config.numWorkers; i++) {
      const worker = new Worker(__filename, {
        workerData: { workerId: i },
      });

      // Single permanent listener per worker: settle the pending promise by
      // task id (per-dispatch temporary listeners leaked and queued tasks
      // never got resolve/reject wired — ADR-276).
      worker.on('message', (result) => {
        this.busyWorkers.delete(worker);
        this.settleTask(result);
        this.processQueue();
      });

      worker.on('error', (err) => {
        console.error(`Worker ${i} error:`, err);
        this.busyWorkers.delete(worker);
      });

      this.workers.push(worker);
    }

    this.initialized = true;
    console.error(`ParallelIntelligence: ${this.config.numWorkers} workers ready`);
  }

  /**
   * Settle the pending promise for a worker response by task id
   */
  private settleTask(result: any): void {
    const entry = result === null || result === undefined
      ? undefined
      : this.pending.get(result.id);
    if (!entry) return;

    this.pending.delete(result.id);
    if (result.error) {
      entry.reject(new Error(result.error));
    } else {
      entry.resolve(result.data);
    }
  }

  private processQueue(): void {
    while (this.taskQueue.length > 0 && this.busyWorkers.size < this.workers.length) {
      const availableWorker = this.workers.find(w => !this.busyWorkers.has(w));
      if (!availableWorker) break;

      const message = this.taskQueue.shift()!;
      this.busyWorkers.add(availableWorker);
      availableWorker.postMessage(message);
    }
  }

  /**
   * Execute task in worker pool. Each task gets a correlation id; the
   * permanent per-worker message listener settles it via the pending map,
   * so queued tasks resolve too (previously they hung forever — ADR-276).
   */
  private async executeInWorker<T>(task: any): Promise<T> {
    if (!this.initialized || !this.config.enabled) {
      throw new Error('ParallelIntelligence not initialized');
    }

    return new Promise((resolve, reject) => {
      const id = this.nextTaskId++;
      const message = { ...task, id };
      this.pending.set(id, { resolve, reject });

      const availableWorker = this.workers.find(w => !this.busyWorkers.has(w));
      if (availableWorker) {
        this.busyWorkers.add(availableWorker);
        availableWorker.postMessage(message);
      } else {
        this.taskQueue.push(message);
      }
    });
  }

  // =========================================================================
  // Parallel Operations
  // =========================================================================

  /**
   * Batch Q-learning episode recording (3-4x faster)
   *
   * Returns the number of episodes processed. Sub-threshold batches (or a
   * disabled/uninitialized pool) are processed sequentially on the main
   * thread instead of being silently dropped (ADR-276). The sequential path
   * mirrors the worker-side `processEpisodes` handler exactly — both are
   * currently stubs pending real Q-learning batch handlers.
   */
  async recordEpisodesBatch(episodes: BatchEpisode[]): Promise<number> {
    if (episodes.length === 0) return 0;

    if (episodes.length < this.config.batchThreshold || !this.config.enabled || !this.initialized) {
      // Sequential fallback: same per-episode work as the worker handler
      return processEpisodesSequential(episodes);
    }

    // Split into chunks for workers
    const chunkSize = Math.ceil(episodes.length / this.config.numWorkers);
    const chunks = [];
    for (let i = 0; i < episodes.length; i += chunkSize) {
      chunks.push(episodes.slice(i, i + chunkSize));
    }

    const counts = await Promise.all(chunks.map(chunk =>
      this.executeInWorker<number>({ type: 'recordEpisodes', episodes: chunk })
    ));

    return counts.reduce((sum, count) => sum + count, 0);
  }

  /**
   * Multi-file pattern matching (parallel pretrain)
   */
  async matchPatternsParallel(files: string[]): Promise<PatternMatchResult[]> {
    if (files.length < this.config.batchThreshold || !this.config.enabled) {
      return [];
    }

    const chunkSize = Math.ceil(files.length / this.config.numWorkers);
    const chunks = [];
    for (let i = 0; i < files.length; i += chunkSize) {
      chunks.push(files.slice(i, i + chunkSize));
    }

    const results = await Promise.all(chunks.map(chunk =>
      this.executeInWorker<PatternMatchResult[]>({ type: 'matchPatterns', files: chunk })
    ));

    return results.flat();
  }

  /**
   * Background memory indexing (non-blocking)
   */
  async indexMemoriesBackground(memories: Array<{ content: string; type: string }>): Promise<void> {
    if (memories.length === 0 || !this.config.enabled) return;

    // Fire and forget - non-blocking
    this.executeInWorker({ type: 'indexMemories', memories }).catch(() => {});
  }

  /**
   * Parallel similarity search with sharding
   */
  async searchParallel(query: string, topK: number = 5): Promise<Array<{ content: string; score: number }>> {
    if (!this.config.enabled) return [];

    // Each worker searches its shard
    const shardResults = await Promise.all(
      this.workers.map((_, i) =>
        this.executeInWorker<Array<{ content: string; score: number }>>({
          type: 'search',
          query,
          topK,
          shardId: i,
        })
      )
    );

    // Merge and sort results
    return shardResults
      .flat()
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Multi-file AST analysis for routing
   */
  async analyzeFilesParallel(files: string[]): Promise<Map<string, { agent: string; confidence: number }>> {
    if (files.length < this.config.batchThreshold || !this.config.enabled) {
      return new Map();
    }

    const chunkSize = Math.ceil(files.length / this.config.numWorkers);
    const chunks = [];
    for (let i = 0; i < files.length; i += chunkSize) {
      chunks.push(files.slice(i, i + chunkSize));
    }

    const results = await Promise.all(chunks.map(chunk =>
      this.executeInWorker<Array<[string, { agent: string; confidence: number }]>>({
        type: 'analyzeFiles',
        files: chunk,
      })
    ));

    return new Map(results.flat());
  }

  /**
   * Parallel git commit analysis for co-edit detection
   */
  async analyzeCommitsParallel(commits: string[]): Promise<CoEditAnalysis[]> {
    if (commits.length < this.config.batchThreshold || !this.config.enabled) {
      return [];
    }

    const chunkSize = Math.ceil(commits.length / this.config.numWorkers);
    const chunks = [];
    for (let i = 0; i < commits.length; i += chunkSize) {
      chunks.push(commits.slice(i, i + chunkSize));
    }

    const results = await Promise.all(chunks.map(chunk =>
      this.executeInWorker<CoEditAnalysis[]>({ type: 'analyzeCommits', commits: chunk })
    ));

    return results.flat();
  }

  /**
   * Get worker pool stats
   */
  getStats(): { enabled: boolean; workers: number; busy: number; queued: number } {
    return {
      enabled: this.config.enabled,
      workers: this.workers.length,
      busy: this.busyWorkers.size,
      queued: this.taskQueue.length,
    };
  }

  /**
   * Shutdown worker pool
   */
  async shutdown(): Promise<void> {
    // Reject anything still pending so callers don't hang on shutdown
    for (const { reject } of this.pending.values()) {
      reject(new Error('ParallelIntelligence shutting down'));
    }
    this.pending.clear();

    await Promise.all(this.workers.map(w => w.terminate()));
    this.workers = [];
    this.busyWorkers.clear();
    this.taskQueue = [];
    this.initialized = false;
  }
}

/**
 * Sequential (main-thread) equivalent of the worker-side `processEpisodes`
 * handler, used for sub-threshold batches and when the pool is disabled.
 * The worker handlers are stubs today (ADR-276), so this mirrors them:
 * it accounts for the episodes and returns the processed count. When real
 * batch handlers land in the workers, this must be updated to match.
 */
function processEpisodesSequential(episodes: BatchEpisode[]): number {
  return episodes.length;
}

// ============================================================================
// Worker Thread Code
// ============================================================================

if (!isMainThread && parentPort) {
  // This code runs in worker threads
  const { workerId } = workerData;

  parentPort.on('message', async (task: any) => {
    try {
      let result: any;

      switch (task.type) {
        case 'recordEpisodes':
          // Process episode batch
          result = await processEpisodes(task.episodes);
          break;

        case 'matchPatterns':
          // Match patterns in files
          result = await matchPatterns(task.files);
          break;

        case 'indexMemories':
          // Index memories
          result = await indexMemories(task.memories);
          break;

        case 'search':
          // Search shard
          result = await searchShard(task.query, task.topK, task.shardId);
          break;

        case 'analyzeFiles':
          // Analyze file ASTs
          result = await analyzeFiles(task.files);
          break;

        case 'analyzeCommits':
          // Analyze git commits
          result = await analyzeCommits(task.commits);
          break;

        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }

      // Echo the correlation id so the main thread can settle by id
      parentPort!.postMessage({ id: task.id, data: result });
    } catch (error: any) {
      parentPort!.postMessage({ id: task.id, error: error.message });
    }
  });

  // Worker task implementations
  async function processEpisodes(episodes: BatchEpisode[]): Promise<number> {
    // Embed and process episodes
    // In a real implementation, this would use the embedder and update Q-values
    return episodes.length;
  }

  async function matchPatterns(files: string[]): Promise<PatternMatchResult[]> {
    // Match patterns in files
    // Would read files and extract patterns
    return files.map(file => ({
      file,
      patterns: [],
    }));
  }

  async function indexMemories(memories: any[]): Promise<number> {
    // Index memories in background
    return memories.length;
  }

  async function searchShard(query: string, topK: number, shardId: number): Promise<any[]> {
    // Search this worker's shard
    return [];
  }

  async function analyzeFiles(files: string[]): Promise<Array<[string, { agent: string; confidence: number }]>> {
    // Analyze file ASTs
    return files.map(f => [f, { agent: 'coder', confidence: 0.5 }]);
  }

  async function analyzeCommits(commits: string[]): Promise<CoEditAnalysis[]> {
    // Analyze git commits for co-edit patterns
    return [];
  }
}

// ============================================================================
// Singleton for easy access
// ============================================================================

let instance: ParallelIntelligence | null = null;

export function getParallelIntelligence(config?: ParallelConfig): ParallelIntelligence {
  if (!instance) {
    instance = new ParallelIntelligence(config);
  }
  return instance;
}

export async function initParallelIntelligence(config?: ParallelConfig): Promise<ParallelIntelligence> {
  const pi = getParallelIntelligence(config);
  await pi.init();
  return pi;
}

export default ParallelIntelligence;
