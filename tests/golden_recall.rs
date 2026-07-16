//! Golden corpus recall@k on the sqlite-vec + FastEmbed path.
//!
//! Default CI skips the embedding download via `#[ignore]`.
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
use std::path::PathBuf;

const CORPUS_ID_META: &str = "golden_corpus_id";

#[derive(Debug, Deserialize)]
struct GoldenCorpus {
    version: u32,
    model_id: String,
    k: usize,
    min_mean_recall_at_k: Option<f64>,
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

#[tokio::test]
#[ignore = "requires embedding model download"]
async fn golden_corpus_mean_recall_at_k_sqlite_vec() {
    let corpus = load_corpus();
    assert_eq!(corpus.version, 1, "unsupported fixture version");

    let model = SupportedEmbeddingModel::parse(&corpus.model_id)
        .unwrap_or_else(|e| panic!("fixture model_id: {e}"));
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

        let r = recall_at_k(&retrieved_ids, &query.relevant_ids, corpus.k);
        per_query.push(r);
    }

    let mean = mean_recall(&per_query);
    let threshold = corpus.min_mean_recall_at_k.unwrap_or(0.75);
    assert!(
        mean >= threshold,
        "mean recall@{} = {mean:.4} below threshold {threshold:.4}; per-query: {per_query:?}",
        corpus.k
    );
}
