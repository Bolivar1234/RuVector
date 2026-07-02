/**
 * AgentDB Fast - High-performance in-process alternative to AgentDB CLI
 *
 * The AgentDB CLI has ~2.3s startup overhead due to npx initialization.
 * This module provides 50-200x faster operations by using in-process calls.
 *
 * Features:
 * - In-memory episode storage with LRU eviction
 * - Vector similarity search using @ruvector/core
 * - Compatible API with AgentDB's episode/trajectory interfaces
 */

import type {
  VectorEntry,
  SearchResult,
  SearchQuery,
} from '../types';

// Lazy load ruvector core
let coreModule: any = null;

function getCoreModule() {
  if (coreModule) return coreModule;
  try {
    coreModule = require('@ruvector/core');
    return coreModule;
  } catch {
    // Fallback to ruvector if core not available
    try {
      coreModule = require('ruvector');
      return coreModule;
    } catch (e: any) {
      throw new Error(
        `Neither @ruvector/core nor ruvector is available: ${e.message}`
      );
    }
  }
}

/**
 * Episode entry for trajectory storage
 */
export interface Episode {
  id: string;
  state: number[];
  action: string | number;
  reward: number;
  nextState: number[];
  done: boolean;
  metadata?: Record<string, any>;
  timestamp?: number;
}

/**
 * Trajectory (sequence of episodes)
 */
export interface Trajectory {
  id: string;
  episodes: Episode[];
  totalReward: number;
  metadata?: Record<string, any>;
}

/**
 * Search result for episode queries
 */
export interface EpisodeSearchResult {
  episode: Episode;
  similarity: number;
  trajectoryId?: string;
}

/**
 * Fast in-memory AgentDB implementation
 */
export class FastAgentDB {
  // LRU via Map insertion order: delete+set refreshes recency, the first key
  // is always the least recently used (O(1) instead of an order array).
  private episodes: Map<string, Episode> = new Map();
  private trajectories: Map<string, Trajectory> = new Map();
  private vectorDb: any = null;
  private dimensions: number;
  private maxEpisodes: number;

  /**
   * Create a new FastAgentDB instance
   *
   * @param dimensions - Vector dimensions for state embeddings
   * @param maxEpisodes - Maximum episodes to store (LRU eviction)
   */
  constructor(dimensions: number = 128, maxEpisodes: number = 100000) {
    this.dimensions = dimensions;
    this.maxEpisodes = maxEpisodes;
  }

  /**
   * Initialize the vector database
   */
  private async initVectorDb(): Promise<void> {
    if (this.vectorDb) return;

    try {
      const core = getCoreModule();
      this.vectorDb = new core.VectorDB({
        dimensions: this.dimensions,
        distanceMetric: 'Cosine',
      });
    } catch (e: any) {
      // Vector DB not available, use fallback similarity
      console.warn(`VectorDB not available, using fallback similarity: ${e.message}`);
    }
  }

  /**
   * Store an episode
   *
   * @param episode - Episode to store
   * @returns Episode ID
   */
  async storeEpisode(episode: Omit<Episode, 'id'> & { id?: string }): Promise<string> {
    await this.initVectorDb();

    const id = episode.id ?? this.generateId();
    const fullEpisode: Episode = {
      ...episode,
      id,
      timestamp: episode.timestamp ?? Date.now(),
    };

    this.insertEpisode(fullEpisode);

    // Index in vector DB if available
    if (this.vectorDb && fullEpisode.state.length === this.dimensions) {
      try {
        await this.vectorDb.insert({
          id,
          vector: new Float32Array(fullEpisode.state),
        });
      } catch {
        // Ignore indexing errors
      }
    }

    return id;
  }

  /**
   * Insert an episode into the in-memory store with O(1) LRU eviction
   */
  private insertEpisode(episode: Episode): void {
    if (this.episodes.has(episode.id)) {
      // Delete + set refreshes recency for re-stored episodes
      this.episodes.delete(episode.id);
    } else if (this.episodes.size >= this.maxEpisodes) {
      // Evict the least recently used episode (oldest in insertion order)
      const oldestId = this.episodes.keys().next().value;
      if (oldestId !== undefined) {
        this.episodes.delete(oldestId);
      }
    }
    this.episodes.set(episode.id, episode);
  }

  /**
   * Store multiple episodes in batch
   *
   * Uses the native `insertBatch` binding (one round-trip) when available,
   * falling back to per-item `storeEpisode` otherwise.
   */
  async storeEpisodes(episodes: (Omit<Episode, 'id'> & { id?: string })[]): Promise<string[]> {
    await this.initVectorDb();

    if (this.vectorDb && typeof this.vectorDb.insertBatch === 'function') {
      const ids: string[] = [];
      const batch: Array<{ id: string; vector: Float32Array }> = [];

      for (const episode of episodes) {
        const id = episode.id ?? this.generateId();
        const fullEpisode: Episode = {
          ...episode,
          id,
          timestamp: episode.timestamp ?? Date.now(),
        };
        this.insertEpisode(fullEpisode);
        if (fullEpisode.state.length === this.dimensions) {
          batch.push({ id, vector: new Float32Array(fullEpisode.state) });
        }
        ids.push(id);
      }

      if (batch.length > 0) {
        try {
          await this.vectorDb.insertBatch(batch);
        } catch {
          // Ignore indexing errors (same as storeEpisode)
        }
      }

      return ids;
    }

    // Fallback: per-item inserts
    const ids: string[] = [];
    for (const episode of episodes) {
      const id = await this.storeEpisode(episode);
      ids.push(id);
    }
    return ids;
  }

  /**
   * Retrieve an episode by ID
   */
  async getEpisode(id: string): Promise<Episode | null> {
    const episode = this.episodes.get(id);
    if (episode) {
      // Update LRU recency: delete + set moves the entry to the end (O(1))
      this.episodes.delete(id);
      this.episodes.set(id, episode);
    }
    return episode ?? null;
  }

  /**
   * Search for similar episodes by state
   *
   * @param queryState - State vector to search for
   * @param k - Number of results to return
   * @returns Similar episodes sorted by similarity
   */
  async searchByState(
    queryState: number[] | Float32Array,
    k: number = 10
  ): Promise<EpisodeSearchResult[]> {
    await this.initVectorDb();

    const query = Array.isArray(queryState) ? queryState : Array.from(queryState);

    // Use vector DB if available
    if (this.vectorDb && query.length === this.dimensions) {
      try {
        const results: SearchResult[] = await this.vectorDb.search({
          vector: new Float32Array(query),
          k,
        });

        return results
          .map((r) => {
            const episode = this.episodes.get(r.id);
            if (!episode) return null;
            return {
              episode,
              similarity: 1 - r.score, // Convert distance to similarity
            };
          })
          .filter((r): r is EpisodeSearchResult => r !== null);
      } catch {
        // Fall through to fallback
      }
    }

    // Fallback: brute-force cosine similarity
    return this.fallbackSearch(query, k);
  }

  /**
   * Fallback similarity search using brute-force cosine similarity
   */
  private fallbackSearch(query: number[], k: number): EpisodeSearchResult[] {
    const results: EpisodeSearchResult[] = [];

    for (const episode of this.episodes.values()) {
      if (episode.state.length !== query.length) continue;

      const similarity = this.cosineSimilarity(query, episode.state);
      results.push({ episode, similarity });
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  }

  /**
   * Compute cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dotProduct / denom;
  }

  /**
   * Store a trajectory (sequence of episodes)
   */
  async storeTrajectory(
    episodes: (Omit<Episode, 'id'> & { id?: string })[],
    metadata?: Record<string, any>
  ): Promise<string> {
    const trajectoryId = this.generateId();
    const storedEpisodes: Episode[] = [];
    let totalReward = 0;

    for (const episode of episodes) {
      const id = await this.storeEpisode(episode);
      const stored = await this.getEpisode(id);
      if (stored) {
        storedEpisodes.push(stored);
        totalReward += stored.reward;
      }
    }

    const trajectory: Trajectory = {
      id: trajectoryId,
      episodes: storedEpisodes,
      totalReward,
      metadata,
    };

    this.trajectories.set(trajectoryId, trajectory);
    return trajectoryId;
  }

  /**
   * Get a trajectory by ID
   */
  async getTrajectory(id: string): Promise<Trajectory | null> {
    return this.trajectories.get(id) ?? null;
  }

  /**
   * Get top trajectories by total reward
   */
  async getTopTrajectories(k: number = 10): Promise<Trajectory[]> {
    return Array.from(this.trajectories.values())
      .sort((a, b) => b.totalReward - a.totalReward)
      .slice(0, k);
  }

  /**
   * Sample random episodes (for experience replay)
   */
  async sampleEpisodes(n: number): Promise<Episode[]> {
    const allEpisodes = Array.from(this.episodes.values());
    const sampled: Episode[] = [];

    for (let i = 0; i < Math.min(n, allEpisodes.length); i++) {
      const idx = Math.floor(Math.random() * allEpisodes.length);
      sampled.push(allEpisodes[idx]);
    }

    return sampled;
  }

  /**
   * Get database statistics
   */
  getStats(): {
    episodeCount: number;
    trajectoryCount: number;
    dimensions: number;
    maxEpisodes: number;
    vectorDbAvailable: boolean;
  } {
    return {
      episodeCount: this.episodes.size,
      trajectoryCount: this.trajectories.size,
      dimensions: this.dimensions,
      maxEpisodes: this.maxEpisodes,
      vectorDbAvailable: this.vectorDb !== null,
    };
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.episodes.clear();
    this.trajectories.clear();
  }

  /**
   * Generate a unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Create a fast AgentDB instance
 */
export function createFastAgentDB(
  dimensions: number = 128,
  maxEpisodes: number = 100000
): FastAgentDB {
  return new FastAgentDB(dimensions, maxEpisodes);
}

// Singleton instance for convenience
let defaultInstance: FastAgentDB | null = null;

/**
 * Get the default FastAgentDB instance
 */
export function getDefaultAgentDB(): FastAgentDB {
  if (!defaultInstance) {
    defaultInstance = new FastAgentDB();
  }
  return defaultInstance;
}

export default {
  FastAgentDB,
  createFastAgentDB,
  getDefaultAgentDB,
};
