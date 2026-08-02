//! ADR-281 acceptance tests for role-aware embedding APIs.
//!
//! These exercise the retrieval-boundary contract (ingestion => passage,
//! text-search => query), asymmetric providers that lack query support, the
//! "prefix applied exactly once" rule, and role preservation across
//! `Arc<dyn EmbeddingProvider>`. They use an in-process spy provider so no
//! network access or model download is required.
//!
//! Criteria that require pinned model artifacts (ADR-281 criteria 4, 5, 8) are
//! intentionally NOT covered here: `schemas/fixtures/` carries only
//! `embedding-space-identity-v1.json`, so faithful vector fixtures for MiniLM /
//! BGE / E5 / Qwen and the cross-language conformance corpus are not available.

use ruvector_core::embeddings::{
    legacy_embed_call_count, EmbeddingDistanceMetric, EmbeddingProvider, EmbeddingRole,
    EmbeddingRolePolicy, EmbeddingSpaceIdentity, OutputDtype, PoolingStrategy, PoolingStrategyName,
    PrefixPolicy,
};
use ruvector_core::error::{Result, RuvectorError};
use ruvector_core::{types::DbOptions, AgenticDB};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

/// A deterministic, normalized embedding whose value depends on the exact
/// (prefixed) text it is handed, so query and passage sides of an asymmetric
/// model produce different vectors.
fn embed_text(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dims];
    for (i, b) in text.as_bytes().iter().enumerate() {
        v[i % dims] += (*b as f32) / 255.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn spy_identity(
    model: &str,
    dims: u32,
    role_policy: EmbeddingRolePolicy,
    prefix_policy: PrefixPolicy,
) -> EmbeddingSpaceIdentity {
    EmbeddingSpaceIdentity {
        schema_version: 1,
        provider: "spy".into(),
        model_id: model.into(),
        model_artifact_sha256: "aa".repeat(32),
        model_graph_sha256: "bb".repeat(32),
        tokenizer_sha256: "cc".repeat(32),
        prompt_template_sha256: "dd".repeat(32),
        pooling_strategy: PoolingStrategy::Named(PoolingStrategyName::Mean),
        normalize: true,
        truncation_tokens: 512,
        output_dimension: dims,
        output_dtype: OutputDtype::F32,
        runtime_revision: "spy@1".into(),
        distance_metric: EmbeddingDistanceMetric::Cosine,
        role_policy,
        prefix_policy,
        prefix_policy_version: 1,
    }
}

/// Records which role each call used and the exact prepared (prefixed) text the
/// provider itself produced, so tests can assert both call routing and that the
/// model-owned prefix is applied exactly once.
struct SpyProvider {
    identity: EmbeddingSpaceIdentity,
    dims: usize,
    name: String,
    support_query: bool,
    support_passage: bool,
    query_prefix: String,
    passage_prefix: String,
    query_texts: Mutex<Vec<String>>,
    passage_texts: Mutex<Vec<String>>,
}

impl SpyProvider {
    fn query_count(&self) -> usize {
        self.query_texts.lock().unwrap().len()
    }

    fn passage_count(&self) -> usize {
        self.passage_texts.lock().unwrap().len()
    }

    fn last_query_text(&self) -> Option<String> {
        self.query_texts.lock().unwrap().last().cloned()
    }

    fn query_texts(&self) -> Vec<String> {
        self.query_texts.lock().unwrap().clone()
    }
}

impl EmbeddingProvider for SpyProvider {
    fn embed_for(&self, role: EmbeddingRole, text: &str) -> Result<Vec<f32>> {
        match role {
            EmbeddingRole::Query if !self.support_query => {
                return Err(RuvectorError::EmbeddingRoleUnsupported(
                    "spy provider has no query support".into(),
                ));
            }
            EmbeddingRole::Passage if !self.support_passage => {
                return Err(RuvectorError::EmbeddingRoleUnsupported(
                    "spy provider has no passage support".into(),
                ));
            }
            _ => {}
        }

        let prepared = match role {
            EmbeddingRole::Query => format!("{}{}", self.query_prefix, text),
            EmbeddingRole::Passage => format!("{}{}", self.passage_prefix, text),
        };
        match role {
            EmbeddingRole::Query => self.query_texts.lock().unwrap().push(prepared.clone()),
            EmbeddingRole::Passage => self.passage_texts.lock().unwrap().push(prepared.clone()),
        }
        Ok(embed_text(&prepared, self.dims))
    }

    fn embedding_space(&self) -> &EmbeddingSpaceIdentity {
        &self.identity
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A spy that supports both roles and applies a distinct prefix per role.
fn asymmetric_spy(dims: usize) -> SpyProvider {
    SpyProvider {
        identity: spy_identity(
            "spy/asymmetric",
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        ),
        dims,
        name: "SpyProvider(asymmetric)".into(),
        support_query: true,
        support_passage: true,
        query_prefix: "query: ".into(),
        passage_prefix: "passage: ".into(),
        query_texts: Mutex::new(Vec::new()),
        passage_texts: Mutex::new(Vec::new()),
    }
}

/// A spy that mimics an asymmetric encoder shipped without a query transform.
fn query_unsupported_spy(dims: usize) -> SpyProvider {
    SpyProvider {
        identity: spy_identity(
            "spy/passage-only",
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        ),
        dims,
        name: "SpyProvider(passage-only)".into(),
        support_query: false,
        support_passage: true,
        query_prefix: "query: ".into(),
        passage_prefix: "passage: ".into(),
        query_texts: Mutex::new(Vec::new()),
        passage_texts: Mutex::new(Vec::new()),
    }
}

fn agentic_db_with(spy: Arc<SpyProvider>, dims: usize) -> (AgenticDB, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = dims;
    let db = AgenticDB::with_embedding_provider(options, spy).unwrap();
    (db, dir)
}

// ADR-281 criterion 2: the same spy proves ingestion and re-index paths invoke
// embed_passage. AgenticDB has no separate re-index API; ingestion is the
// index-building path (each ingest writes the passage vector into the index),
// so covering every ingestion entry point covers the persisted-corpus side.
#[test]
fn ingestion_paths_use_passage_role_only() {
    let dims = 32;
    let spy = Arc::new(asymmetric_spy(dims));
    let (db, _dir) = agentic_db_with(spy.clone(), dims);

    db.store_episode(
        "task".into(),
        vec!["a".into()],
        vec!["o".into()],
        "critique text".into(),
    )
    .unwrap();
    db.create_skill(
        "Parse".into(),
        "parse json".into(),
        std::collections::HashMap::new(),
        vec!["json.parse()".into()],
    )
    .unwrap();
    db.add_causal_edge(vec!["rain".into()], vec!["wet".into()], 0.9, "weather".into())
        .unwrap();
    db.session_index("s1", 3600).add_turn(0, "user", "hello there").unwrap();
    db.witness_log().append("agent-1", "edit", "changed a file").unwrap();

    assert_eq!(
        spy.passage_count(),
        5,
        "each ingestion entry point must embed exactly one passage vector"
    );
    assert_eq!(
        spy.query_count(),
        0,
        "ingestion must never invoke the query role"
    );
}

// ADR-281 criterion 3 (positive half): every text-search path invokes
// embed_query. Each search is isolated on its own DB so the query count is
// attributable to exactly one search entry point.
#[test]
fn search_paths_use_query_role() {
    let dims = 32;

    // retrieve_similar_episodes
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.store_episode("t".into(), vec![], vec![], "c".into()).unwrap();
        let before = spy.query_count();
        db.retrieve_similar_episodes("find me", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "retrieve_similar_episodes");
    }

    // search_skills
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.create_skill("n".into(), "d".into(), std::collections::HashMap::new(), vec![])
            .unwrap();
        let before = spy.query_count();
        db.search_skills("find skill", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "search_skills");
    }

    // query_with_utility
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.add_causal_edge(vec!["a".into()], vec!["b".into()], 0.5, "ctx".into())
            .unwrap();
        let before = spy.query_count();
        db.query_with_utility("utility query", 3, 0.7, 0.2, 0.1).unwrap();
        assert_eq!(spy.query_count(), before + 1, "query_with_utility");
    }

    // find_relevant_turns
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        let session = db.session_index("s1", 3600);
        session.add_turn(0, "user", "hi").unwrap();
        let before = spy.query_count();
        session.find_relevant_turns("earlier context", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "find_relevant_turns");
    }

    // witness_log search
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        let log = db.witness_log();
        log.append("a", "act", "details").unwrap();
        let before = spy.query_count();
        log.search("audit query", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "witness_log::search");
    }
}

// ADR-281 criterion 3 (negative half): an asymmetric provider that lacks query
// support returns EmbeddingRoleUnsupported and never silently falls back to the
// passage side (or the deprecated `embed`).
#[test]
fn query_unsupported_provider_errors_without_falling_back() {
    let dims = 32;
    let spy = Arc::new(query_unsupported_spy(dims));

    // Direct provider surface: embed_query must error, not fall back.
    let err = spy.embed_query("hello").unwrap_err();
    assert!(
        matches!(err, RuvectorError::EmbeddingRoleUnsupported(_)),
        "expected EmbeddingRoleUnsupported, got {err:?}"
    );

    // Through a high-level search path.
    let (db, _dir) = agentic_db_with(spy.clone(), dims);
    db.store_episode("t".into(), vec![], vec![], "c".into()).unwrap();
    let passage_before = spy.passage_count();

    let result = db.retrieve_similar_episodes("cannot embed this", 3);
    let err = result.expect_err("search on a query-unsupported provider must fail");
    assert!(
        matches!(err, RuvectorError::EmbeddingRoleUnsupported(_)),
        "expected EmbeddingRoleUnsupported from search, got {err:?}"
    );
    assert_eq!(
        spy.passage_count(),
        passage_before,
        "a failed query must not fall back to the passage role"
    );
}

// ADR-281 criterion 6: prefixes are applied exactly once in single and batch
// APIs. The spy owns prefix application; high-level callers pass raw text.
#[test]
fn prefix_applied_exactly_once_single_and_batch() {
    let dims = 32;
    let spy = asymmetric_spy(dims);

    // Single API.
    spy.embed_query("cat").unwrap();
    spy.embed_passage("dog").unwrap();
    assert_eq!(spy.last_query_text().as_deref(), Some("query: cat"));
    assert_eq!(
        spy.query_texts()[0].matches("query: ").count(),
        1,
        "single embed_query must apply the query prefix exactly once"
    );

    // Batch API must not strip the role nor double-apply the prefix.
    let batch = spy.embed_query_batch(&["one", "two"]).unwrap();
    assert_eq!(batch.len(), 2);
    for prepared in spy.query_texts().iter().skip(1) {
        assert_eq!(
            prepared.matches("query: ").count(),
            1,
            "batch embed_query must apply the query prefix exactly once per item, got {prepared:?}"
        );
    }

    // The high-level DB path must also apply the prefix exactly once (i.e. the
    // DB passes raw text and never pre-prefixes).
    let spy = Arc::new(asymmetric_spy(dims));
    let (db, _dir) = agentic_db_with(spy.clone(), dims);
    db.create_skill("n".into(), "d".into(), std::collections::HashMap::new(), vec![])
        .unwrap();
    db.search_skills("kittens", 3).unwrap();
    assert_eq!(
        spy.last_query_text().as_deref(),
        Some("query: kittens"),
        "AgenticDB must forward raw query text so the provider prefixes exactly once"
    );
}

// ADR-281 criterion 7: trait-object use preserves both roles. An asymmetric
// provider behind Arc<dyn EmbeddingProvider> keeps query and passage distinct,
// and the trait-object results match the concrete-provider results, in both the
// single and batch APIs.
#[test]
fn trait_object_preserves_both_roles_for_asymmetric_provider() {
    let dims = 32;
    let concrete = asymmetric_spy(dims);

    // Reference results computed directly (no trait object).
    let ref_query = embed_text("query: same text", dims);
    let ref_passage = embed_text("passage: same text", dims);
    assert_ne!(
        ref_query, ref_passage,
        "sanity: asymmetric fixture must produce distinct role vectors"
    );

    let provider: Arc<dyn EmbeddingProvider> = Arc::new(concrete);

    let q = provider.embed_query("same text").unwrap();
    let p = provider.embed_passage("same text").unwrap();
    assert_ne!(
        q, p,
        "trait object collapsed query and passage into the same vector"
    );
    assert_eq!(q, ref_query, "trait-object query role diverged from concrete");
    assert_eq!(
        p, ref_passage,
        "trait-object passage role diverged from concrete"
    );

    // Batch surface preserves the role through the trait object as well.
    let legacy_before = legacy_embed_call_count();
    let qb = provider.embed_query_batch(&["same text"]).unwrap();
    let pb = provider.embed_passage_batch(&["same text"]).unwrap();
    assert_eq!(qb[0], ref_query);
    assert_eq!(pb[0], ref_passage);
    assert_ne!(qb[0], pb[0]);
    assert_eq!(
        legacy_embed_call_count(),
        legacy_before,
        "role-explicit trait-object calls must not route through deprecated embed"
    );
}
