//! Golden corpus recall@k on the sqlite-vec + `FastEmbed` path.
//!
//! Default CI skips the embedding download via `#[ignore]`.
//! Required local gate for accuracy changes (see `docs/WAVE_C_ACCEPTANCE.md`):
//!
//! ```bash
//! cargo test --features sqlite-vec --test golden_recall -- --ignored
//! ```

#![cfg(feature = "sqlite-vec")]

use memory_gate_rs::{
    eval::{mean_recall, recall_at_k},
    storage::SqliteVecStore,
    AgentDomain, KnowledgeStore, LearningContext, SupportedEmbeddingModel,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;

const CORPUS_ID_META: &str = "golden_corpus_id";

#[derive(Debug, Deserialize)]
struct GoldenCorpus {
    version: u32,
    model_id: String,
    k: usize,
    /// Authoritative mean recall@k from a pinned green run (same model + fixture).
    baseline_mean_recall_at_k: f64,
    /// Floor for pass: typically `baseline * 0.98` (≤2% relative drop).
    min_mean_recall_at_k: f64,
    documents: Vec<GoldenDocument>,
    queries: Vec<GoldenQuery>,
}

#[derive(Debug, Deserialize)]
struct GoldenDocument {
    id: String,
    content: String,
    domain: String,
}

#[derive(Debug, Deserialize)]
struct GoldenQuery {
    id: String,
    text: String,
    relevant_ids: Vec<String>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_corpus.json")
}

fn load_corpus() -> GoldenCorpus {
    let raw = std::fs::read_to_string(fixture_path()).expect("read golden_corpus.json");
    serde_json::from_str(&raw).expect("parse golden_corpus.json")
}

fn parse_domain(domain: &str) -> AgentDomain {
    domain
        .parse()
        .unwrap_or_else(|_| panic!("unknown domain in fixture: {domain}"))
}

fn corpus_id_from_context(ctx: &LearningContext) -> Option<String> {
    ctx.metadata
        .as_ref()
        .and_then(|m| m.get(CORPUS_ID_META))
        .cloned()
}

/// Fail closed on fixture hygiene before any embedding work.
fn validate_corpus(corpus: &GoldenCorpus) {
    assert_eq!(corpus.version, 1, "unsupported fixture version");
    assert!(corpus.k > 0, "fixture k must be > 0");
    assert!(
        corpus.baseline_mean_recall_at_k > 0.0 && corpus.baseline_mean_recall_at_k <= 1.0,
        "baseline_mean_recall_at_k out of range"
    );
    assert!(
        corpus.min_mean_recall_at_k > 0.0 && corpus.min_mean_recall_at_k <= 1.0,
        "min_mean_recall_at_k out of range"
    );
    // ≤2% relative drop: min >= baseline * 0.98 − tiny float slack
    let floor = corpus.baseline_mean_recall_at_k * 0.98;
    assert!(
        corpus.min_mean_recall_at_k + 1e-9 >= floor,
        "min_mean_recall_at_k ({}) must be >= baseline*0.98 ({floor})",
        corpus.min_mean_recall_at_k
    );

    let model = SupportedEmbeddingModel::parse(&corpus.model_id)
        .unwrap_or_else(|e| panic!("fixture model_id must be catalog id: {e}"));
    assert_eq!(
        model,
        SupportedEmbeddingModel::AllMiniLmL6V2,
        "golden corpus is pinned to all-minilm-l6-v2 for Wave C"
    );

    let mut seen = HashSet::new();
    for doc in &corpus.documents {
        assert!(
            seen.insert(doc.id.as_str()),
            "duplicate document id: {}",
            doc.id
        );
        let _ = parse_domain(&doc.domain);
    }
    assert!(!corpus.documents.is_empty(), "fixture needs documents");
    assert!(!corpus.queries.is_empty(), "fixture needs queries");

    for query in &corpus.queries {
        assert!(
            !query.relevant_ids.is_empty(),
            "query {} has empty relevant_ids (vacuous recall forbidden in golden)",
            query.id
        );
        for rid in &query.relevant_ids {
            assert!(
                seen.contains(rid.as_str()),
                "query {} relevant id {rid} not in documents",
                query.id
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires embedding model download"]
async fn golden_corpus_mean_recall_at_k_sqlite_vec() {
    let corpus = load_corpus();
    validate_corpus(&corpus);

    let model = SupportedEmbeddingModel::parse(&corpus.model_id).expect("validated model_id");
    let store = SqliteVecStore::open_in_memory_with_model(model)
        .await
        .expect("open in-memory sqlite-vec store");

    for doc in &corpus.documents {
        let ctx = LearningContext::new(&doc.content, parse_domain(&doc.domain))
            .with_meta(CORPUS_ID_META, &doc.id);
        store
            .store_experience(&doc.id, ctx)
            .await
            .unwrap_or_else(|e| panic!("store {}: {e}", doc.id));
    }

    let mut per_query = Vec::with_capacity(corpus.queries.len());
    for query in &corpus.queries {
        let results = store
            .retrieve_context(&query.text, corpus.k, None)
            .await
            .unwrap_or_else(|e| panic!("retrieve {}: {e}", query.id));

        let retrieved_ids: Vec<String> =
            results.iter().filter_map(corpus_id_from_context).collect();
        assert_eq!(
            retrieved_ids.len(),
            results.len(),
            "query {}: missing {CORPUS_ID_META} on {} of {} hits (metadata strip regression?)",
            query.id,
            results.len() - retrieved_ids.len(),
            results.len()
        );

        let r = recall_at_k(&retrieved_ids, &query.relevant_ids, corpus.k);
        per_query.push(r);
    }

    let mean = mean_recall(&per_query);
    assert!(
        mean >= corpus.min_mean_recall_at_k,
        "mean recall@{} = {mean:.4} below min_mean_recall_at_k {:.4} \
         (baseline {:.4}, allow ≤2% relative drop); per-query: {per_query:?}",
        corpus.k,
        corpus.min_mean_recall_at_k,
        corpus.baseline_mean_recall_at_k
    );
}
