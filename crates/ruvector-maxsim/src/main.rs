//! maxsim-bench — benchmark for ruvector-maxsim three-variant comparison.
//!
//! Generates a synthetic multi-cluster Gaussian corpus and measures:
//!   1. FlatMaxSim    — brute-force O(N·M·K·D) exact baseline
//!   2. PrunedMaxSim  — centroid scan + MaxSim reranking
//!   3. GraphMaxSim   — kNN centroid graph + beam search + MaxSim reranking
//!
//! Run: cargo run --release -p ruvector-maxsim

use ruvector_maxsim::graph::GraphMaxSim;
use ruvector_maxsim::pruned::PrunedMaxSim;
use ruvector_maxsim::{FlatMaxSim, MaxSimResult, MultiVecDoc, MultiVecIndex, Token};

use std::collections::HashSet;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ── Dataset parameters ────────────────────────────────────────────────────────

const N_DOCS: usize = 2_000;
const N_CLUSTERS: usize = 20;
const K_TOKENS: usize = 8; // tokens per document
const M_TOKENS: usize = 4; // query tokens
const DIM: usize = 64;
const N_QUERIES: usize = 200;
const TOP_K: usize = 10;
const NOISE_STD: f32 = 0.3;
const SEED: u64 = 42;

// ── Data generation ───────────────────────────────────────────────────────────

fn rand_unit(rng: &mut StdRng, dim: usize) -> Token {
    let v: Token = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn noisy_unit(rng: &mut StdRng, center: &[f32], std: f32) -> Token {
    let v: Token = center
        .iter()
        .map(|&c| c + (rng.gen::<f32>() * 2.0 - 1.0) * std)
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

struct Dataset {
    docs: Vec<MultiVecDoc>,
    queries: Vec<Vec<Token>>,
}

fn generate(seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Token> = (0..N_CLUSTERS).map(|_| rand_unit(&mut rng, DIM)).collect();

    let docs: Vec<MultiVecDoc> = (0..N_DOCS)
        .map(|i| {
            let c = i % N_CLUSTERS;
            let tokens: Vec<Token> = (0..K_TOKENS)
                .map(|_| noisy_unit(&mut rng, &centers[c], NOISE_STD))
                .collect();
            MultiVecDoc::new(i, tokens)
        })
        .collect();

    let queries: Vec<Vec<Token>> = (0..N_QUERIES)
        .map(|i| {
            let c = i % N_CLUSTERS;
            (0..M_TOKENS)
                .map(|_| noisy_unit(&mut rng, &centers[c], NOISE_STD * 0.5))
                .collect()
        })
        .collect();

    Dataset { docs, queries }
}

// ── Measurement helpers ───────────────────────────────────────────────────────

fn latency_stats(times: &[Duration]) -> (f64, f64, f64) {
    let mut ns: Vec<u64> = times.iter().map(|d| d.as_nanos() as u64).collect();
    ns.sort_unstable();
    let mean = ns.iter().sum::<u64>() as f64 / ns.len() as f64;
    let p50 = ns[ns.len() / 2] as f64;
    let p95 = ns[(ns.len() as f64 * 0.95) as usize] as f64;
    (mean, p50, p95)
}

fn recall_at_k(results: &[MaxSimResult], gt: &HashSet<usize>) -> f32 {
    if gt.is_empty() {
        return 1.0;
    }
    let hits = results.iter().filter(|r| gt.contains(&r.doc_id)).count();
    hits as f32 / gt.len() as f32
}

fn run_queries(
    index: &dyn MultiVecIndex,
    queries: &[Vec<Token>],
    ground_truths: &[HashSet<usize>],
) -> (Vec<Duration>, f64) {
    let mut times = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0f64;
    for (qi, qt) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let res = index.search(qt, TOP_K);
        times.push(t0.elapsed());
        recall_sum += recall_at_k(&res, &ground_truths[qi]) as f64;
    }
    let mean_recall = recall_sum / queries.len() as f64;
    (times, mean_recall)
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  ruvector-maxsim: Multi-Vector Late Interaction Benchmark");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("\nEnvironment:");
    println!("  OS:   {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);
    println!("\nDataset:");
    println!(
        "  Docs:     {} ({} clusters, {} tokens/doc, dim={})",
        N_DOCS, N_CLUSTERS, K_TOKENS, DIM
    );
    println!("  Queries:  {} ({} tokens/query)", N_QUERIES, M_TOKENS);
    println!("  Noise σ:  {}", NOISE_STD);
    println!("  Top-K:    {}", TOP_K);

    println!("\nGenerating dataset (seed={})…", SEED);
    let ds = generate(SEED);

    // ── Ground truth from FlatMaxSim ──────────────────────────────────────────
    println!("Computing ground truth (FlatMaxSim)…");
    let mut flat_gt = FlatMaxSim::new();
    for doc in &ds.docs {
        flat_gt.add(doc.clone());
    }
    let ground_truths: Vec<HashSet<usize>> = ds
        .queries
        .iter()
        .map(|q| flat_gt.search(q, TOP_K).iter().map(|r| r.doc_id).collect())
        .collect();

    // Memory estimate (flat allocation, no overhead)
    let doc_mem_kb = (N_DOCS * K_TOKENS * DIM * 4) / 1024;
    let centroid_mem_kb = (N_DOCS * DIM * 4) / 1024;
    let m_graph = 12usize;
    let graph_edge_kb = (N_DOCS * m_graph * 8) / 1024;

    // ── Variant 1: FlatMaxSim ─────────────────────────────────────────────────
    println!("\n[1/3] FlatMaxSim…");
    let mut flat = FlatMaxSim::new();
    for doc in &ds.docs {
        flat.add(doc.clone());
    }
    let (flat_times, flat_recall) = run_queries(&flat, &ds.queries, &ground_truths);
    let (flat_mean, flat_p50, flat_p95) = latency_stats(&flat_times);
    let flat_qps = 1e9 / flat_mean;

    // ── Variant 2: PrunedMaxSim ───────────────────────────────────────────────
    let n_cand = (N_DOCS / 10).max(50); // 10% of corpus
    println!("[2/3] PrunedMaxSim (n_candidates={})…", n_cand);
    let mut pruned = PrunedMaxSim::new(n_cand);
    for doc in &ds.docs {
        pruned.add(doc.clone());
    }
    let (pruned_times, pruned_recall) = run_queries(&pruned, &ds.queries, &ground_truths);
    let (pruned_mean, pruned_p50, pruned_p95) = latency_stats(&pruned_times);
    let pruned_qps = 1e9 / pruned_mean;

    // ── Variant 3: GraphMaxSim ────────────────────────────────────────────────
    let ef = 64usize;
    println!(
        "[3/3] GraphMaxSim (m={}, ef={}, candidates={})…",
        m_graph, ef, n_cand
    );
    let t_build = Instant::now();
    let mut graph = GraphMaxSim::new(m_graph, ef, n_cand);
    for doc in &ds.docs {
        graph.add(doc.clone());
    }
    graph.build();
    let build_ms = t_build.elapsed().as_millis();
    println!("      Graph built in {}ms", build_ms);

    let (graph_times, graph_recall) = run_queries(&graph, &ds.queries, &ground_truths);
    let (graph_mean, graph_p50, graph_p95) = latency_stats(&graph_times);
    let graph_qps = 1e9 / graph_mean;

    // ── Results table ─────────────────────────────────────────────────────────
    println!("\n┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Variant         │ Mean (µs)    │ p50 (µs)     │ p95 (µs)     │ QPS          │ Recall@10    │ Mem (KB)     │");
    println!("├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    println!(
        "│ FlatMaxSim      │{:>13.1} │{:>13.1} │{:>13.1} │{:>13.1} │{:>11.1}%  │{:>13} │",
        flat_mean / 1e3,
        flat_p50 / 1e3,
        flat_p95 / 1e3,
        flat_qps,
        flat_recall * 100.0,
        doc_mem_kb
    );
    println!(
        "│ PrunedMaxSim    │{:>13.1} │{:>13.1} │{:>13.1} │{:>13.1} │{:>11.1}%  │{:>13} │",
        pruned_mean / 1e3,
        pruned_p50 / 1e3,
        pruned_p95 / 1e3,
        pruned_qps,
        pruned_recall * 100.0,
        doc_mem_kb + centroid_mem_kb
    );
    println!(
        "│ GraphMaxSim     │{:>13.1} │{:>13.1} │{:>13.1} │{:>13.1} │{:>11.1}%  │{:>13} │",
        graph_mean / 1e3,
        graph_p50 / 1e3,
        graph_p95 / 1e3,
        graph_qps,
        graph_recall * 100.0,
        doc_mem_kb + centroid_mem_kb + graph_edge_kb
    );
    println!("└─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");

    println!("\nSpeedup vs FlatMaxSim:");
    println!(
        "  PrunedMaxSim: {:.2}x  GraphMaxSim: {:.2}x",
        flat_mean / pruned_mean,
        flat_mean / graph_mean
    );
    println!("  GraphMaxSim build time: {}ms", build_ms);

    // ── Acceptance tests ──────────────────────────────────────────────────────
    println!("\nAcceptance tests:");
    let p_pass = pruned_recall >= 0.75;
    let g_pass = graph_recall >= 0.55;
    println!(
        "  PrunedMaxSim recall@10 ≥ 75%: {}  (actual {:.1}%)",
        if p_pass { "PASS ✓" } else { "FAIL ✗" },
        pruned_recall * 100.0
    );
    println!(
        "  GraphMaxSim  recall@10 ≥ 55%: {}  (actual {:.1}%)",
        if g_pass { "PASS ✓" } else { "FAIL ✗" },
        graph_recall * 100.0
    );
    println!("\nNote: recall is vs FlatMaxSim ground truth on synthetic Gaussian clusters.");
    println!(
        "      Dimensions={DIM}, K_tokens={K_TOKENS}, N_docs={N_DOCS}, N_queries={N_QUERIES}."
    );

    if p_pass && g_pass {
        println!("\nResult: ALL ACCEPTANCE TESTS PASSED ✓");
        std::process::exit(0);
    } else {
        eprintln!("\nResult: ACCEPTANCE TEST(S) FAILED ✗");
        std::process::exit(1);
    }
}
