//! Real SIFT1M benchmark: fvecs → recall@10 vs QPS Pareto.
//!
//! Loads local .fvecs/.ivecs files (no download required).
//! Tests ruvector-core HNSW and ruvector-rabitq against
//! published ANN-Benchmarks SOTA numbers.
//!
//! Usage (from workspace root):
//!   cargo run --release -p ruvector-sota-bench --bin sota-sift1m-fvecs -- \
//!     --base     bench_data/sift/sift_base.fvecs \
//!     --queries  bench_data/sift/sift_query.fvecs \
//!     --gt       bench_data/sift/sift_groundtruth.ivecs
//!
//! Add --max-n 100000 to use only the first 100K base vectors.

use anyhow::{Context, Result};
use clap::Parser;
use ruvector_core::{
    index::{hnsw::HnswIndex, VectorIndex},
    types::HnswConfig,
    DistanceMetric,
};
use ruvector_rabitq::index::{AnnIndex, FlatF32Index, RabitqIndex, RabitqPlusIndex};
use ruvector_rabitq::rotation::RandomRotationKind;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "sota-sift1m-fvecs")]
#[command(about = "Benchmark ruvector HNSW + RaBitQ on real SIFT1M data (fvecs format)")]
struct Args {
    #[arg(long, default_value = "bench_data/sift/sift_base.fvecs")]
    base: PathBuf,

    #[arg(long, default_value = "bench_data/sift/sift_query.fvecs")]
    queries: PathBuf,

    #[arg(long, default_value = "bench_data/sift/sift_groundtruth.ivecs")]
    gt: PathBuf,

    /// Cap corpus to this many vectors (0 = all).
    #[arg(long, default_value = "0")]
    max_n: usize,

    /// HNSW M (connectivity).
    #[arg(long, default_value = "16")]
    m: usize,

    /// HNSW efConstruction.
    #[arg(long, default_value = "200")]
    ef_construction: usize,

    /// Comma-separated ef_search sweep values.
    #[arg(long, default_value = "10,20,50,100,200,400,800")]
    ef_search: String,

    /// k for recall@k.
    #[arg(long, default_value = "10")]
    k: usize,

    /// Skip RaBitQ benchmarks (faster for a quick HNSW-only run).
    #[arg(long)]
    no_rabitq: bool,

    /// Output JSON path (optional).
    #[arg(long)]
    json_out: Option<PathBuf>,
}

// ─────────────────────────────────────────────────────────────────────────────
// fvecs / ivecs loaders
// ─────────────────────────────────────────────────────────────────────────────

/// Load an .fvecs file.  Format: [d: u32LE] [f32 × d] repeated.
fn load_fvecs(path: &PathBuf, max_n: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(64 * 1024 * 1024, file);
    let mut out: Vec<Vec<f32>> = Vec::new();
    let mut dim_buf = [0u8; 4];
    loop {
        match r.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let d = u32::from_le_bytes(dim_buf) as usize;
        let mut bytes = vec![0u8; d * 4];
        r.read_exact(&mut bytes)?;
        let vec: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.push(vec);
        if max_n > 0 && out.len() >= max_n {
            break;
        }
    }
    Ok(out)
}

/// Load an .ivecs file and return IDs as u64.
fn load_ivecs(path: &PathBuf) -> Result<Vec<Vec<u64>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut out: Vec<Vec<u64>> = Vec::new();
    let mut dim_buf = [0u8; 4];
    loop {
        match r.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let d = u32::from_le_bytes(dim_buf) as usize;
        let mut bytes = vec![0u8; d * 4];
        r.read_exact(&mut bytes)?;
        let ids: Vec<u64> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u64)
            .collect();
        out.push(ids);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Recall metric
// ─────────────────────────────────────────────────────────────────────────────

fn recall_at_k(result_ids: &[u64], gt: &[u64], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<u64> = gt.iter().take(k).cloned().collect();
    let res_set: std::collections::HashSet<u64> = result_ids.iter().take(k).cloned().collect();
    let hits = gt_set.intersection(&res_set).count();
    hits as f64 / k.min(gt_set.len()) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Result row
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
struct Row {
    system: String,
    dataset: String,
    n_base: usize,
    n_queries: usize,
    dims: usize,
    params: String,
    recall_at_10: f64,
    qps: f64,
    p50_us: f64,
    p99_us: f64,
    build_secs: f64,
    index_mb: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// HNSW benchmark (build once, sweep ef_search)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_hnsw(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt: &[Vec<u64>],
    m: usize,
    ef_construction: usize,
    ef_values: &[usize],
    k: usize,
) -> Result<Vec<Row>> {
    let dims = corpus[0].len();
    let n_base = corpus.len();
    let dataset_name = format!("sift-{}-euclidean", dims);

    eprintln!(
        "[HNSW] Building index: n={}, dims={}, M={}, efC={}",
        n_base, dims, m, ef_construction
    );

    let cfg = HnswConfig {
        m,
        ef_construction,
        ef_search: 50, // default; overridden per-query via search_with_ef
        max_elements: n_base + 1024,
    };

    let t_build = Instant::now();
    let mut idx = HnswIndex::new(dims, DistanceMetric::Euclidean, cfg)
        .map_err(|e| anyhow::anyhow!("HnswIndex::new: {e}"))?;

    for (i, v) in corpus.iter().enumerate() {
        idx.add(i.to_string(), v.clone())
            .map_err(|e| anyhow::anyhow!("HnswIndex::add {i}: {e}"))?;
        if i % 100_000 == 0 && i > 0 {
            eprintln!("  inserted {}/{}", i, n_base);
        }
    }
    let build_secs = t_build.elapsed().as_secs_f64();
    eprintln!("[HNSW] Build done in {:.1}s", build_secs);

    // Rough index size: raw floats + HNSW graph (≈1.5× overhead for edges)
    let index_mb = (n_base * dims * 4) as f64 / (1024.0 * 1024.0) * 1.5;

    let mut rows = Vec::new();

    for &ef in ef_values {
        eprint!("[HNSW] ef_search={} querying {} queries ... ", ef, queries.len());
        let mut latencies: Vec<u128> = Vec::with_capacity(queries.len());
        let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let results = idx
                .search_with_ef(q, k, ef)
                .map_err(|e| anyhow::anyhow!("search_with_ef: {e}"))?;
            latencies.push(t.elapsed().as_nanos());

            let ids: Vec<u64> = results
                .iter()
                .filter_map(|r| r.id.parse::<u64>().ok())
                .collect();
            recalls.push(recall_at_k(&ids, &gt[qi], k));
        }

        let n_q = queries.len() as f64;
        let mean_recall = recalls.iter().sum::<f64>() / n_q;
        let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
        let qps = n_q / total_s;

        let mut sorted_lat = latencies.clone();
        sorted_lat.sort_unstable();
        let p50_us = sorted_lat[(n_q * 0.50) as usize] as f64 / 1_000.0;
        let p99_us = sorted_lat[(n_q * 0.99) as usize] as f64 / 1_000.0;

        eprintln!("recall={:.4}  QPS={:.1}", mean_recall, qps);

        rows.push(Row {
            system: format!("ruvector-hnsw(M={},efC={},efS={})", m, ef_construction, ef),
            dataset: dataset_name.clone(),
            n_base,
            n_queries: queries.len(),
            dims,
            params: format!("M={},efC={},efS={}", m, ef_construction, ef),
            recall_at_10: mean_recall,
            qps,
            p50_us,
            p99_us,
            build_secs,
            index_mb,
        });
    }

    Ok(rows)
}

// ─────────────────────────────────────────────────────────────────────────────
// RaBitQ benchmark
// ─────────────────────────────────────────────────────────────────────────────

fn bench_rabitq_variant<I: AnnIndex>(
    label: &str,
    mut idx: I,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt: &[Vec<u64>],
    k: usize,
) -> Result<Row> {
    let dims = corpus[0].len();
    let n_base = corpus.len();
    let dataset_name = format!("sift-{}-euclidean", dims);

    eprint!("[RaBitQ] Building {} n={} ... ", label, n_base);
    let t_build = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        idx.add(i, v.clone())
            .map_err(|e| anyhow::anyhow!("rabitq add {i}: {e}"))?;
    }
    let build_secs = t_build.elapsed().as_secs_f64();
    let index_mb = idx.memory_bytes() as f64 / (1024.0 * 1024.0);
    eprintln!("done in {:.1}s, {:.1} MB", build_secs, index_mb);

    eprint!("[RaBitQ] Querying {} queries ... ", queries.len());
    let mut latencies: Vec<u128> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let results = idx
            .search(q, k)
            .map_err(|e| anyhow::anyhow!("rabitq search: {e}"))?;
        latencies.push(t.elapsed().as_nanos());

        let ids: Vec<u64> = results.iter().map(|r| r.id as u64).collect();
        recalls.push(recall_at_k(&ids, &gt[qi], k));
    }

    let n_q = queries.len() as f64;
    let mean_recall = recalls.iter().sum::<f64>() / n_q;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1e9;
    let qps = n_q / total_s;

    let mut sorted_lat = latencies.clone();
    sorted_lat.sort_unstable();
    let p50_us = sorted_lat[(n_q * 0.50) as usize] as f64 / 1_000.0;
    let p99_us = sorted_lat[(n_q * 0.99) as usize] as f64 / 1_000.0;

    eprintln!("recall={:.4}  QPS={:.1}", mean_recall, qps);

    Ok(Row {
        system: format!("ruvector-{}", label),
        dataset: dataset_name,
        n_base,
        n_queries: queries.len(),
        dims,
        params: label.to_string(),
        recall_at_10: mean_recall,
        qps,
        p50_us,
        p99_us,
        build_secs,
        index_mb,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let ef_values: Vec<usize> = args
        .ef_search
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // ── Load data ────────────────────────────────────────────────────────────
    eprintln!("[load] Reading base vectors from {}", args.base.display());
    let t0 = Instant::now();
    let corpus = load_fvecs(&args.base, args.max_n)?;
    eprintln!("  {} vectors in {:.2}s", corpus.len(), t0.elapsed().as_secs_f64());

    eprintln!("[load] Reading query vectors from {}", args.queries.display());
    let queries = load_fvecs(&args.queries, 0)?;
    eprintln!("  {} queries", queries.len());

    eprintln!("[load] Reading ground truth from {}", args.gt.display());
    let gt = load_ivecs(&args.gt)?;
    eprintln!("  {} GT rows, top-{} each", gt.len(), gt[0].len());

    let dims = corpus[0].len();
    let n_base = corpus.len();
    let n_queries = queries.len();

    println!();
    println!("=== ruvector SIFT1M Benchmark ===");
    println!("Dataset : sift-{}-euclidean", dims);
    println!("N base  : {}", n_base);
    println!("N query : {}", n_queries);
    println!();

    // ── HNSW sweep ───────────────────────────────────────────────────────────
    let mut all_rows: Vec<Row> = Vec::new();

    let hnsw_rows = bench_hnsw(
        &corpus,
        &queries,
        &gt,
        args.m,
        args.ef_construction,
        &ef_values,
        args.k,
    )?;
    all_rows.extend(hnsw_rows);

    // ── RaBitQ suite ─────────────────────────────────────────────────────────
    if !args.no_rabitq {
        let seed = 42u64;
        let rerank = 10usize;

        // Flat exact baseline
        let flat = FlatF32Index::new(dims);
        if let Ok(row) = bench_rabitq_variant("rabitq-flat-exact", flat, &corpus, &queries, &gt, args.k) {
            all_rows.push(row);
        }

        // 1-bit RaBitQ (HadamardSigned rotation — fastest)
        let rabitq = RabitqIndex::new_with_rotation(dims, seed, RandomRotationKind::HadamardSigned);
        if let Ok(row) = bench_rabitq_variant("rabitq-1bit", rabitq, &corpus, &queries, &gt, args.k) {
            all_rows.push(row);
        }

        // RaBitQ+ (with refinement re-ranking — highest recall)
        let rabitq_plus = RabitqPlusIndex::new(dims, seed, rerank);
        if let Ok(row) = bench_rabitq_variant("rabitq-plus", rabitq_plus, &corpus, &queries, &gt, args.k) {
            all_rows.push(row);
        }
    }

    // ── Print CSV ────────────────────────────────────────────────────────────
    println!();
    println!("system,dataset,n_base,n_queries,dims,params,recall@10,qps,p50_us,p99_us,build_secs,index_mb");
    for r in &all_rows {
        println!(
            "{},{},{},{},{},{},{:.5},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.system,
            r.dataset,
            r.n_base,
            r.n_queries,
            r.dims,
            r.params,
            r.recall_at_10,
            r.qps,
            r.p50_us,
            r.p99_us,
            r.build_secs,
            r.index_mb,
        );
    }

    // ── JSON output ──────────────────────────────────────────────────────────
    if let Some(json_path) = &args.json_out {
        let json = serde_json::to_string_pretty(&all_rows)?;
        std::fs::write(json_path, json)?;
        eprintln!("[out] JSON written to {}", json_path.display());
    }

    Ok(())
}
