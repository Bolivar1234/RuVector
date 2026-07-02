#!/usr/bin/env node

/**
 * ADR-274 phase 1: commander-free, chalk-free micro-entry for the Claude Code
 * hooks hot path.
 *
 * The lifecycle hooks (pre-edit / post-edit / pre-command / post-command /
 * stats) fire on EVERY Claude Code tool use. Going through bin/cli.js pays
 * full CLI startup per fire: 445 KB parse + ~186 commander registrations +
 * chalk. This entry hand-rolls argv dispatch and loads only what the matched
 * command needs, so a hook fire costs little more than bare Node boot.
 *
 * Dual role:
 *   - executable:  `ruvector-hooks <cmd> ...` (registered in package.json bin)
 *   - module:      bin/cli.js's `hooks <cmd>` .action bodies call the exported
 *                  handler functions below, so the lifecycle logic lives in
 *                  exactly ONE place (this file). When required as a module
 *                  this file MUST NOT print, exit, or mutate the environment —
 *                  only the `require.main === module` path at the bottom does.
 *
 * Output parity: handlers take a style object (`c`). The main CLI passes
 * chalk-backed styles (colored in a TTY); this entry uses the identity style,
 * which is byte-identical to the CLI's output when piped (chalk disables
 * color for non-TTY stdout).
 *
 * The Intelligence class below mirrors the subset of bin/cli.js's
 * Intelligence that the five lifecycle handlers can reach. If you change one,
 * change the other (bin/cli.js keeps the full class for the remaining hooks
 * subcommands: route, remember, recall, session-*, reembed, ...).
 */
'use strict';

const fs = require('fs');
const path = require('path');

// ============================================================
// Lazy dist modules (mirrors bin/cli.js loadIntelligenceEngine /
// loadProvenance): required on first use only, tolerate a missing build.
// ============================================================

let IntelligenceEngine = null;
let engineLoadAttempted = false;

function loadIntelligenceEngine() {
  if (engineLoadAttempted) return IntelligenceEngine;
  engineLoadAttempted = true;
  try {
    const core = require('../dist/core/intelligence-engine.js');
    IntelligenceEngine = core.IntelligenceEngine || core.default;
  } catch (e) {
    // IntelligenceEngine not available, use fallback
  }
  return IntelligenceEngine;
}

let provenanceMod = null;
let provenanceLoadAttempted = false;

function loadProvenance() {
  if (provenanceLoadAttempted) return provenanceMod;
  provenanceLoadAttempted = true;
  try {
    provenanceMod = require('../dist/core/embedding-provenance.js');
  } catch (e) {
    provenanceMod = null;
  }
  return provenanceMod;
}

/** Sanitize a provenance record read from disk (untrusted JSON, ADR-210). */
function sanitizeProvenanceSafe(value) {
  const prov = loadProvenance();
  if (prov && typeof prov.sanitizeProvenance === 'function') {
    return prov.sanitizeProvenance(value);
  }
  return (
    value && typeof value === 'object' && !Array.isArray(value) &&
    typeof value.embedderKind === 'string' &&
    Number.isInteger(value.dimension) && value.dimension > 0 && value.dimension <= 65536
  ) ? value : null;
}

// ============================================================
// Intelligence (lifecycle subset — see header note on parity with cli.js)
// ============================================================

class Intelligence {
  constructor(options = {}) {
    this.intelPath = this.getIntelPath();
    this.data = this.load();
    this.alpha = 0.1;
    this.lastEditedFile = null;
    this._engine = null;
    this._engineInitialized = false;
    this._skipEngine = options.skipEngine || false;
  }

  // Lazy getter for engine - only initializes when first accessed.
  // NOTE: none of the five lifecycle handlers trigger this in a fresh
  // process (they only use getEngineIfReady()), so the hot path never
  // loads dist/core/intelligence-engine.js.
  getEngine() {
    if (this._skipEngine) return null;
    if (this._engineInitialized) return this._engine;
    this._engineInitialized = true;

    const EngineClass = loadIntelligenceEngine();
    if (EngineClass) {
      try {
        this._engine = new EngineClass({
          maxMemories: 100000,
          maxEpisodes: 50000,
          enableSona: true,
          enableAttention: true,
          enableOnnx: true,
          learningRate: this.alpha,
        });
        if (this.data) {
          this._engine.import(this.convertLegacyData(this.data), true);
        }
      } catch (e) {
        this._engine = null;
      }
    }
    return this._engine;
  }

  get engine() { return this.getEngine(); }

  getEngineIfReady() {
    return this._engineInitialized ? this._engine : null;
  }

  convertLegacyData(data) {
    const converted = {
      memories: [],
      routingPatterns: {},
      errorPatterns: data.errors || {},
      coEditPatterns: {},
      agentMappings: {},
    };
    if (data.memories) {
      converted.memories = data.memories.map(m => ({
        id: m.id,
        content: m.content,
        type: m.memory_type || 'general',
        embedding: m.embedding || this.embed(m.content),
        created: m.timestamp ? new Date(m.timestamp * 1000).toISOString() : new Date().toISOString(),
        accessed: 0,
      }));
    }
    if (data.patterns) {
      for (const [key, value] of Object.entries(data.patterns)) {
        const [state, action] = key.split('|');
        if (state && action) {
          if (!converted.routingPatterns[state]) converted.routingPatterns[state] = {};
          converted.routingPatterns[state][action] = value.q_value || 0.5;
        }
      }
    }
    if (data.file_sequences) {
      for (const seq of data.file_sequences) {
        if (!converted.coEditPatterns[seq.from_file]) converted.coEditPatterns[seq.from_file] = {};
        converted.coEditPatterns[seq.from_file][seq.to_file] = seq.count;
      }
    }
    return converted;
  }

  // Prefer project-local storage, fall back to home directory
  getIntelPath() {
    const projectPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    const homePath = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');

    if (fs.existsSync(path.dirname(projectPath))) return projectPath;
    if (fs.existsSync(path.join(process.cwd(), '.claude'))) return projectPath;
    if (fs.existsSync(homePath)) return homePath;
    return projectPath;
  }

  load() {
    const defaults = {
      patterns: {},
      memories: [],
      trajectories: [],
      errors: {},
      file_sequences: [],
      agents: {},
      edges: [],
      stats: { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 }
    };
    try {
      if (fs.existsSync(this.intelPath)) {
        const data = JSON.parse(fs.readFileSync(this.intelPath, 'utf-8'));
        const asArray = (v, dflt) => (Array.isArray(v) ? v : dflt);
        const asObject = (v, dflt) => (v && typeof v === 'object' && !Array.isArray(v) ? v : dflt);
        return {
          patterns: asObject(data.patterns, defaults.patterns),
          memories: asArray(data.memories, defaults.memories),
          trajectories: asArray(data.trajectories, defaults.trajectories),
          errors: asObject(data.errors, defaults.errors),
          file_sequences: asArray(data.file_sequences, defaults.file_sequences),
          agents: asObject(data.agents, defaults.agents),
          edges: asArray(data.edges, defaults.edges),
          stats: { ...defaults.stats, ...asObject(data.stats, {}) },
          embeddingProvenance: sanitizeProvenanceSafe(data.embeddingProvenance),
          activeTrajectories: data.activeTrajectories || {},
          coEditPatterns: data.coEditPatterns || undefined,
          sequences: data.sequences || undefined,
          learning: data.learning || undefined
        };
      }
    } catch {}
    return defaults;
  }

  save() {
    const dir = path.dirname(this.intelPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const engineData = eng.export();
        this.data.patterns = {};
        for (const [state, actions] of Object.entries(engineData.routingPatterns || {})) {
          for (const [action, value] of Object.entries(actions)) {
            this.data.patterns[`${state}|${action}`] = { state, action, q_value: value, visits: 1, last_update: this.now() };
          }
        }
        this.data.stats.total_patterns = Object.keys(this.data.patterns).length;
        this.data.stats.total_memories = engineData.stats?.totalMemories || this.data.memories.length;
        this.data.engineStats = engineData.stats;
      } catch (e) {
        // Ignore engine export errors
      }
    }

    fs.writeFileSync(this.intelPath, JSON.stringify(this.data, null, 2));
  }

  now() { return Math.floor(Date.now() / 1000); }

  // Engine embedding if already initialized, otherwise 64-dim hash fallback
  embed(text) {
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        return eng.embed(text);
      } catch {}
    }
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
    return embedding;
  }

  // ---- ADR-210 D0 embedding-provenance write gate ----

  storedProvenance() { return this.data.embeddingProvenance || null; }

  vectorMemoryCount() {
    return (this.data.memories || []).filter(m => Array.isArray(m.embedding) && m.embedding.length > 0).length;
  }

  isLegacyVectorStore() {
    return !this.storedProvenance() && this.vectorMemoryCount() > 0;
  }

  inferredLegacyProvenance() {
    const prov = loadProvenance();
    const first = (this.data.memories || []).find(m => Array.isArray(m.embedding) && m.embedding.length > 0);
    const dim = first ? first.embedding.length : 256;
    if (prov) return prov.legacyHashProvenance(dim);
    return { embedderKind: 'hash', modelId: null, dimension: dim, normalize: false, prefixPolicy: 'none' };
  }

  syncWriteProvenance(embedding) {
    return { embedderKind: 'hash', modelId: null, dimension: embedding.length, normalize: true, prefixPolicy: 'none' };
  }

  checkVectorWrite(active) {
    const prov = loadProvenance();
    if (!prov || !active) return; // enforcement needs the dist module
    if (this.isLegacyVectorStore()) {
      const legacy = this.inferredLegacyProvenance();
      const err = new Error(
        `Vector store ${this.intelPath} predates embedding provenance (ADR-210) and is read-only for vector writes. ` +
        `Stored vectors are treated as ${prov.describeProvenance(legacy)}; the active embedder is ` +
        `${prov.describeProvenance(active)}. Run 'ruvector hooks reembed' to re-embed and unlock it.`
      );
      err.code = 'ERR_LEGACY_STORE_READONLY';
      throw err;
    }
    const stored = this.storedProvenance();
    if (!stored) {
      this.data.embeddingProvenance = active;
      return;
    }
    prov.assertProvenanceMatch(stored, active, this.intelPath);
  }

  guardVectorWrite(active) {
    try {
      this.checkVectorWrite(active);
      return { ok: true };
    } catch (e) {
      const prov = loadProvenance();
      const policy = prov ? prov.resolveReembedPolicy() : 'refuse';
      if (policy === 'warn') {
        if (!Intelligence._reembedWarned) {
          Intelligence._reembedWarned = true;
          console.error(`ruvector: ${e.message} (RUVECTOR_REEMBED=warn: store stays read-only, write skipped)`);
        }
        return { ok: false, skipped: true, error: e.message };
      }
      if (policy === 'auto') e.message += ` (RUVECTOR_REEMBED=auto: run 'ruvector hooks reembed' — in-place re-embedding needs the async path)`;
      throw e;
    }
  }

  // ---- memory / Q-learning subset used by the lifecycle hooks ----

  remember(memoryType, content, metadata = {}) {
    const id = `mem_${this.now()}`;
    const embedding = this.embed(content);
    const guard = this.guardVectorWrite(this.syncWriteProvenance(embedding));
    if (!guard.ok) return null;
    this.data.memories.push({ id, memory_type: memoryType, content, embedding, metadata, timestamp: this.now() });
    if (this.data.memories.length > 5000) this.data.memories.splice(0, 1000);
    this.data.stats.total_memories = this.data.memories.length;

    const eng = this.getEngineIfReady();
    if (eng) {
      eng.remember(content, memoryType).catch(() => {});
    }
    return id;
  }

  /** Best-effort remember for ambient learning hooks: never fails the hook. */
  tryRemember(memoryType, content, metadata = {}) {
    try {
      return this.remember(memoryType, content, metadata);
    } catch (e) {
      if (!Intelligence._rememberSkipNoted) {
        Intelligence._rememberSkipNoted = true;
        console.error(`   (memory write skipped: ${e.message})`);
      }
      return null;
    }
  }

  getQ(state, action) {
    const key = `${state}|${action}`;
    if (!this.data.patterns) this.data.patterns = {};
    return this.data.patterns[key]?.q_value ?? 0;
  }

  updateQ(state, action, reward) {
    const key = `${state}|${action}`;
    if (!this.data.patterns) this.data.patterns = {};
    if (!this.data.stats) this.data.stats = { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 };
    if (!this.data.patterns[key]) {
      this.data.patterns[key] = { state, action, q_value: 0, visits: 0, last_update: 0 };
    }
    const p = this.data.patterns[key];
    p.q_value = p.q_value + this.alpha * (reward - p.q_value);
    p.visits++;
    p.last_update = this.now();
    this.data.stats.total_patterns = Object.keys(this.data.patterns).length;

    const eng = this.getEngineIfReady();
    if (eng) {
      eng.recordEpisode(state, action, reward, state, false).catch(() => {});
    }
  }

  learn(state, action, outcome, reward) {
    const id = `traj_${this.now()}`;
    this.updateQ(state, action, reward);
    if (!this.data.trajectories) this.data.trajectories = [];
    if (!this.data.stats) this.data.stats = { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 };
    this.data.trajectories.push({ id, state, action, outcome, reward, timestamp: this.now() });
    if (this.data.trajectories.length > 1000) this.data.trajectories.splice(0, 200);
    this.data.stats.total_trajectories = this.data.trajectories.length;

    const eng = this.getEngineIfReady();
    if (eng) {
      eng.endTrajectory(reward > 0.5, reward);
    }
    return id;
  }

  suggest(state, actions) {
    let bestAction = actions[0] ?? '';
    let bestQ = -Infinity;
    for (const action of actions) {
      const q = this.getQ(state, action);
      if (q > bestQ) { bestQ = q; bestAction = action; }
    }
    return { action: bestAction, confidence: bestQ > 0 ? Math.min(bestQ, 1) : 0 };
  }

  // Canonical routing state key — MUST mirror IntelligenceEngine.getState()
  routeState(task, file) {
    const t = task || '';
    const taskType = t.includes('fix') ? 'fix' :
                     t.includes('test') ? 'test' :
                     t.includes('refactor') ? 'refactor' :
                     t.includes('document') ? 'docs' : 'edit';
    let ext = '';
    if (file) {
      const idx = file.lastIndexOf('.');
      ext = idx >= 0 ? file.slice(idx).toLowerCase() : '';
    }
    return `${taskType}:${ext || 'unknown'}`;
  }

  route(task, file, crateName, operation = 'edit') {
    const fileType = file ? path.extname(file).slice(1) : 'unknown';
    const state = this.routeState(task || operation, file);
    const agentMap = {
      rs: ['rust-developer', 'coder', 'reviewer', 'tester'],
      ts: ['typescript-developer', 'coder', 'frontend-dev'],
      tsx: ['react-developer', 'typescript-developer', 'coder'],
      js: ['javascript-developer', 'coder', 'frontend-dev'],
      jsx: ['react-developer', 'coder'],
      py: ['python-developer', 'coder', 'ml-developer'],
      go: ['go-developer', 'coder'],
      sql: ['database-specialist', 'coder'],
      md: ['documentation-specialist', 'coder'],
      yml: ['devops-engineer', 'coder'],
      yaml: ['devops-engineer', 'coder']
    };
    const agents = (agentMap[fileType] ?? ['coder', 'reviewer']).slice();
    const prefix = `${state}|`;
    for (const key of Object.keys(this.data.patterns || {})) {
      if (key.startsWith(prefix)) {
        const learned = key.slice(prefix.length);
        if (learned && !agents.includes(learned)) agents.push(learned);
      }
    }
    const { action, confidence } = this.suggest(state, agents);
    const reason = confidence > 0.5 ? 'learned from past success' : confidence > 0 ? 'based on patterns' : `default for ${fileType} files`;

    const eng = this.getEngineIfReady();
    if (eng) {
      eng.beginTrajectory(task || operation, file);
    }
    return { agent: action, confidence, reason };
  }

  shouldTest(file) {
    const ext = path.extname(file).slice(1);
    switch (ext) {
      case 'rs': {
        const crateMatch = file.match(/crates\/([^/]+)/);
        return crateMatch ? { suggest: true, command: `cargo test -p ${crateMatch[1]}` } : { suggest: true, command: 'cargo test' };
      }
      case 'ts': case 'tsx': case 'js': case 'jsx': return { suggest: true, command: 'npm test' };
      case 'py': return { suggest: true, command: 'pytest' };
      case 'go': return { suggest: true, command: 'go test ./...' };
      default: return { suggest: false, command: '' };
    }
  }

  recordFileSequence(fromFile, toFile) {
    const existing = this.data.file_sequences.find(s => s.from_file === fromFile && s.to_file === toFile);
    if (existing) existing.count++;
    else this.data.file_sequences.push({ from_file: fromFile, to_file: toFile, count: 1 });
    this.lastEditedFile = toFile;

    const eng = this.getEngineIfReady();
    if (eng) {
      eng.recordCoEdit(fromFile, toFile);
    }
  }

  suggestNext(file, limit = 3) {
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const results = eng.getLikelyNextFiles(file, limit);
        if (results.length > 0) {
          return results.map(r => ({ file: r.file, score: r.count }));
        }
      } catch {}
    }
    return this.data.file_sequences
      .filter(s => s.from_file === file)
      .sort((a, b) => b.count - a.count)
      .slice(0, limit)
      .map(s => ({ file: s.to_file, score: s.count }));
  }

  classifyCommand(command) {
    const cmd = command.toLowerCase();
    if (cmd.includes('cargo') || cmd.includes('rustc')) return { category: 'rust', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('npm') || cmd.includes('node') || cmd.includes('yarn') || cmd.includes('pnpm')) return { category: 'javascript', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('python') || cmd.includes('pip') || cmd.includes('pytest')) return { category: 'python', subcategory: cmd.includes('test') ? 'test' : 'run', risk: 'low' };
    if (cmd.includes('go ')) return { category: 'go', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('git')) return { category: 'git', subcategory: 'vcs', risk: cmd.includes('push') || cmd.includes('force') ? 'medium' : 'low' };
    if (cmd.includes('rm ') || cmd.includes('delete') || cmd.includes('rmdir')) return { category: 'filesystem', subcategory: 'destructive', risk: 'high' };
    if (cmd.includes('sudo') || cmd.includes('chmod') || cmd.includes('chown')) return { category: 'system', subcategory: 'privileged', risk: 'high' };
    if (cmd.includes('docker') || cmd.includes('kubectl')) return { category: 'container', subcategory: 'orchestration', risk: 'medium' };
    return { category: 'shell', subcategory: 'general', risk: 'low' };
  }

  swarmStats() {
    const agents = Object.keys(this.data.agents).length;
    const edges = this.data.edges.length;
    return { agents, edges };
  }

  stats() {
    const baseStats = this.data.stats;
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const engineStats = eng.getStats();
        return {
          ...baseStats,
          engineEnabled: true,
          sonaEnabled: engineStats.sonaEnabled,
          attentionEnabled: engineStats.attentionEnabled,
          embeddingDim: engineStats.memoryDimensions,
          embedderKind: engineStats.embedderKind,
          totalMemories: engineStats.totalMemories,
          totalEpisodes: engineStats.totalEpisodes,
          trajectoriesRecorded: engineStats.trajectoriesRecorded,
          patternsLearned: engineStats.patternsLearned,
          microLoraUpdates: engineStats.microLoraUpdates,
          baseLoraUpdates: engineStats.baseLoraUpdates,
          ewcConsolidations: engineStats.ewcConsolidations,
        };
      } catch {}
    }
    return { ...baseStats, engineEnabled: false };
  }

  getLastEditedFile() { return this.lastEditedFile; }
}

// ============================================================
// Shared lifecycle handlers — the ONE implementation used by both this
// micro-entry (identity style) and bin/cli.js's hooks .action bodies
// (chalk-backed style). Text MUST NOT diverge between the two callers.
// ============================================================

const ident = (s) => s;
const PLAIN_STYLE = {
  bold: ident,
  boldCyan: ident,
  cyan: ident,
  green: ident,
  greenBold: ident,
  dim: ident,
  red: ident,
  yellow: ident,
};

function preEdit(intel, file, c = PLAIN_STYLE) {
  const fileName = path.basename(file);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1];
  const { agent, confidence, reason } = intel.route(`edit ${fileName}`, file, crate, 'edit');
  console.log(c.bold('🧠 Intelligence Analysis:'));
  console.log(`   📁 ${c.cyan(crate ?? 'project')}/${fileName}`);
  console.log(`   🤖 Recommended: ${c.greenBold(agent)} (${(confidence * 100).toFixed(0)}% confidence)`);
  if (reason) console.log(`      → ${c.dim(reason)}`);
  const nextFiles = intel.suggestNext(file, 3);
  if (nextFiles.length > 0) {
    console.log('   📎 Likely next files:');
    nextFiles.forEach(n => console.log(`      - ${n.file} (${n.score} edits)`));
  }
}

function postEdit(intel, file, opts = {}, c = PLAIN_STYLE) {
  const success = opts.error ? false : (opts.success ?? true);
  const ext = path.extname(file).slice(1);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1] ?? 'project';
  const state = `edit_${ext}_in_${crate}`;
  const lastFile = intel.getLastEditedFile();
  if (lastFile && lastFile !== file) intel.recordFileSequence(lastFile, file);
  intel.learn(state, success ? 'successful-edit' : 'failed-edit', success ? 'completed' : 'failed', success ? 1.0 : -0.5);
  // Best-effort: a provenance-locked store (ADR-210) must not fail the hook
  intel.tryRemember('edit', `${success ? 'successful' : 'failed'} edit of ${ext} in ${crate}`);
  intel.save();
  console.log(`📊 Learning recorded: ${success ? '✅' : '❌'} ${path.basename(file)}`);
  const test = intel.shouldTest(file);
  if (test.suggest) console.log(`   🧪 Consider: ${c.cyan(test.command)}`);
}

function preCommand(intel, cmd, c = PLAIN_STYLE) {
  const classification = intel.classifyCommand(cmd);
  console.log(c.bold('🧠 Command Analysis:'));
  console.log(`   📦 Category: ${c.cyan(classification.category)}`);
  console.log(`   🏷️  Type: ${classification.subcategory}`);
  if (classification.risk === 'high') console.log(`   ⚠️  Risk: ${c.red('HIGH')} - Review carefully`);
  else if (classification.risk === 'medium') console.log(`   ⚡ Risk: ${c.yellow('MEDIUM')}`);
  else console.log(`   ✅ Risk: ${c.green('LOW')}`);
}

function postCommand(intel, cmd, opts = {}) {
  const success = opts.error ? false : (opts.success ?? true);
  const classification = intel.classifyCommand(cmd);
  intel.learn(`cmd_${classification.category}_${classification.subcategory}`, success ? 'success' : 'failure', success ? 'completed' : 'failed', success ? 0.8 : -0.3);
  // Best-effort: a provenance-locked store (ADR-210) must not fail the hook
  intel.tryRemember('command', `${cmd} ${success ? 'succeeded' : 'failed'}`);
  intel.save();
  console.log(`📊 Command ${success ? '✅' : '❌'} recorded`);
}

function stats(intel, c = PLAIN_STYLE) {
  const s = intel.stats();
  const swarm = intel.swarmStats();
  console.log(c.boldCyan('\n🧠 RuVector Intelligence Stats\n'));
  console.log(`  ${c.green(s.total_patterns)} Q-learning patterns`);
  console.log(`  ${c.green(s.total_memories)} vector memories`);
  console.log(`  ${c.green(s.total_trajectories)} learning trajectories`);
  console.log(`  ${c.green(s.total_errors)} error patterns\n`);
  console.log(c.bold('Swarm Status:'));
  console.log(`  ${c.cyan(swarm.agents)} agents registered`);
  console.log(`  ${c.cyan(swarm.edges)} coordination edges`);
}

// ============================================================
// Standalone dispatch (only under `require.main === module`)
// ============================================================

const USAGE = `Usage: ruvector-hooks <command> [arguments] [options]

Fast micro-entry for the Claude Code hooks hot path (ADR-274).

Commands:
  pre-edit <file>                                  Pre-edit intelligence
  post-edit <file> [--success] [--error <msg>]     Post-edit learning
  pre-command <command...>                         Pre-command intelligence
  post-command <command...> [--success] [--error <msg>]
                                                   Post-command learning
  stats                                            Show intelligence statistics

The full CLI exposes these plus setup/session/memory tooling: ruvector hooks --help`;

function fail(message) {
  process.stderr.write(message + '\n');
  process.exit(1);
}

/**
 * Minimal argv parser matching commander's behavior for these commands:
 * `--` ends option parsing; `--error` consumes the next token greedily
 * (or `--error=msg`); any other dash-leading token is an unknown option;
 * excess positionals are tolerated (commander v11 default).
 */
function parseTail(tokens, { allowStatusFlags = false } = {}) {
  const positionals = [];
  const opts = {};
  let endOfOpts = false;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (!endOfOpts && t === '--') { endOfOpts = true; continue; }
    if (!endOfOpts && t.length > 1 && t[0] === '-') {
      if (allowStatusFlags && t === '--success') { opts.success = true; continue; }
      if (allowStatusFlags && t === '--error') {
        if (i + 1 >= tokens.length) fail("error: option '--error <msg>' argument missing");
        opts.error = tokens[++i];
        continue;
      }
      if (allowStatusFlags && t.startsWith('--error=')) { opts.error = t.slice('--error='.length); continue; }
      fail(`error: unknown option '${t}'`);
    }
    positionals.push(t);
  }
  return { positionals, opts };
}

function main(argv) {
  const cmd = argv[0];
  const rest = argv.slice(1);
  switch (cmd) {
    case 'pre-edit': {
      const { positionals } = parseTail(rest);
      if (positionals.length < 1) fail("error: missing required argument 'file'");
      preEdit(new Intelligence(), positionals[0]);
      break;
    }
    case 'post-edit': {
      const { positionals, opts } = parseTail(rest, { allowStatusFlags: true });
      if (positionals.length < 1) fail("error: missing required argument 'file'");
      postEdit(new Intelligence(), positionals[0], opts);
      break;
    }
    case 'pre-command': {
      const { positionals } = parseTail(rest);
      if (positionals.length < 1) fail("error: missing required argument 'command'");
      preCommand(new Intelligence(), positionals.join(' '));
      break;
    }
    case 'post-command': {
      const { positionals, opts } = parseTail(rest, { allowStatusFlags: true });
      if (positionals.length < 1) fail("error: missing required argument 'command'");
      postCommand(new Intelligence(), positionals.join(' '), opts);
      break;
    }
    case 'stats': {
      parseTail(rest);
      stats(new Intelligence());
      break;
    }
    case '-h':
    case '--help':
    case 'help':
      process.stdout.write(USAGE + '\n');
      break;
    case undefined:
      fail(USAGE);
      break;
    default:
      fail(`error: unknown command '${cmd}'\n\n` + USAGE);
  }
}

module.exports = { preEdit, postEdit, preCommand, postCommand, stats, Intelligence, main };

if (require.main === module) {
  // Signal CLI context (disables parallel workers - hooks are short-lived)
  process.env.RUVECTOR_CLI = '1';
  // On-disk V8 compile cache (Node >=22, no-op elsewhere) — ADR-274.
  try { require('node:module').enableCompileCache?.(); } catch { /* best-effort */ }
  main(process.argv.slice(2));
}
