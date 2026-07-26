//! Criterion baselines for vector storage backends (`sqlite-vec`).
//!
//! Wave B perf work uses these as before/after measurement paths alongside
//! `benches/storage_benchmarks.rs` (in-memory gateway).
//!
//! # Running
//!
//! ```bash
//! # Compile only (weights download happens at run time, not --no-run):
//! cargo bench --bench vector_storage_benchmarks --no-run --features sqlite-vec
//!
//! # Full run (downloads FastEmbed weights on first use; slow, not for CI):
//! cargo bench --bench vector_storage_benchmarks --features sqlite-vec
//! ```
//!
//! Set `MEMORY_GATE_SKIP_HEAVY_BENCH=1` to skip the cold-open benchmark (re-open
//! loads the embedder each iteration).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use memory_gate_rs::{
    storage::SqliteVecStore, AgentDomain, KnowledgeStore, LearningContext, SupportedEmbeddingModel,
};
use std::env;
use std::hint::black_box;
use tokio::runtime::Runtime;

const MODEL: SupportedEmbeddingModel = SupportedEmbeddingModel::BgeSmallEnV15;

fn skip_heavy() -> bool {
    env::var("MEMORY_GATE_SKIP_HEAVY_BENCH")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn bench_sqlite_vec_open_in_memory(c: &mut Criterion) {
    if skip_heavy() {
        return;
    }

    let rt = Runtime::new().expect("tokio runtime");

    c.bench_function("sqlite_vec_open_in_memory_with_model", |b| {
        b.iter(|| {
            rt.block_on(async {
                SqliteVecStore::open_in_memory_with_model(MODEL)
                    .await
                    .expect("open in-memory sqlite-vec store")
            });
        });
    });
}

fn bench_sqlite_vec_store(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let store = rt.block_on(async {
        SqliteVecStore::open_in_memory_with_model(MODEL)
            .await
            .expect("open store for store bench")
    });

    let mut seq = 0u64;

    c.bench_function("sqlite_vec_store_experience", |b| {
        b.iter(|| {
            let key = format!("bench-store-{seq}");
            seq = seq.wrapping_add(1);
            rt.block_on(async {
                let ctx = LearningContext::new(
                    black_box(format!("Vector bench store content {key}")),
                    AgentDomain::General,
                );
                store
                    .store_experience(black_box(&key), ctx)
                    .await
                    .expect("store experience");
            });
        });
    });
}

fn bench_sqlite_vec_retrieve(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("sqlite_vec_retrieve_context");

    for store_size in [10_usize, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(store_size),
            &store_size,
            |b, &size| {
                let store = rt.block_on(async {
                    let store = SqliteVecStore::open_in_memory_with_model(MODEL)
                        .await
                        .expect("open store for retrieve bench");
                    for i in 0..size {
                        let ctx = LearningContext::new(
                            format!("Searchable vector bench item {i} about operations"),
                            AgentDomain::Infrastructure,
                        );
                        store
                            .store_experience(&format!("retrieve-prep-{i}"), ctx)
                            .await
                            .expect("prepopulate");
                    }
                    store
                });

                b.iter(|| {
                    rt.block_on(async {
                        store
                            .retrieve_context(black_box("operations CPU"), 10, None)
                            .await
                            .expect("retrieve")
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    vector_benches,
    bench_sqlite_vec_open_in_memory,
    bench_sqlite_vec_store,
    bench_sqlite_vec_retrieve,
);

criterion_main!(vector_benches);
