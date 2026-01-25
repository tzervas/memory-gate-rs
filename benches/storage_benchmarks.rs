//! Benchmarks for storage operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use memory_gate_rs::{
    adapters::PassthroughAdapter,
    storage::InMemoryStore,
    AgentDomain, GatewayConfig, KnowledgeStore, LearningContext, MemoryGateway,
};
use tokio::runtime::Runtime;

fn create_gateway() -> MemoryGateway<PassthroughAdapter, InMemoryStore> {
    MemoryGateway::new(
        PassthroughAdapter,
        InMemoryStore::new(),
        GatewayConfig::default().with_consolidation_enabled(false),
    )
}

fn bench_store_operation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("store_single_context", |b| {
        b.iter(|| {
            let gateway = create_gateway();
            rt.block_on(async {
                let ctx = LearningContext::new(
                    black_box("Test content for benchmarking"),
                    AgentDomain::Infrastructure,
                );
                gateway.learn_from_interaction(ctx, None).await.unwrap();
            });
        });
    });
}

fn bench_retrieve_operation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("retrieve_context");

    for store_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(store_size),
            store_size,
            |b, &size| {
                let gateway = create_gateway();

                // Pre-populate the store
                rt.block_on(async {
                    for i in 0..size {
                        let ctx = LearningContext::new(
                            format!("Benchmark memory content {i}"),
                            AgentDomain::Infrastructure,
                        );
                        gateway.learn_from_interaction(ctx, None).await.unwrap();
                    }
                });

                b.iter(|| {
                    rt.block_on(async {
                        gateway
                            .retrieve_context(black_box("memory"), Some(10), None)
                            .await
                            .unwrap()
                    });
                });
            },
        );
    }

    group.finish();
}

fn bench_bulk_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("store_100_contexts", |b| {
        b.iter(|| {
            let gateway = create_gateway();
            rt.block_on(async {
                for i in 0..100 {
                    let ctx = LearningContext::new(
                        format!("Bulk content {i}"),
                        AgentDomain::General,
                    );
                    gateway.learn_from_interaction(ctx, None).await.unwrap();
                }
            });
        });
    });
}

fn bench_in_memory_store(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("in_memory_store_direct", |b| {
        let store = InMemoryStore::new();
        let mut i = 0;

        b.iter(|| {
            rt.block_on(async {
                let ctx = LearningContext::new(
                    black_box(format!("Direct store content {i}")),
                    AgentDomain::General,
                );
                store.store_experience(&format!("key{i}"), ctx).await.unwrap();
            });
            i += 1;
        });
    });

    c.bench_function("in_memory_retrieve_direct", |b| {
        let store = InMemoryStore::new();

        // Pre-populate
        rt.block_on(async {
            for i in 0..1000 {
                let ctx = LearningContext::new(
                    format!("Searchable content item {i}"),
                    AgentDomain::General,
                );
                store.store_experience(&format!("key{i}"), ctx).await.unwrap();
            }
        });

        b.iter(|| {
            rt.block_on(async {
                store
                    .retrieve_context(black_box("content"), 10, None)
                    .await
                    .unwrap()
            });
        });
    });
}

criterion_group!(
    benches,
    bench_store_operation,
    bench_retrieve_operation,
    bench_bulk_operations,
    bench_in_memory_store,
);

criterion_main!(benches);
