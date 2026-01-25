//! VSA/HDC operation benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use memory_gate_rs::vsa::{HolographicVector, VsaCodebook, VsaOps, BindingMode, BundlingMode};

fn bench_vector_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_vector_creation");
    
    for dim in [1000, 4096, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("random", dim), dim, |b, &dim| {
            b.iter(|| HolographicVector::random(black_box(dim)));
        });
    }
    
    group.finish();
}

fn bench_bind_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_bind");
    
    for dim in [1000, 4096, 10000].iter() {
        let v1 = HolographicVector::random(*dim);
        let v2 = HolographicVector::random(*dim);
        
        group.bench_with_input(BenchmarkId::new("xor_bind", dim), dim, |b, _| {
            b.iter(|| v1.bind(black_box(&v2)));
        });
        
        group.bench_with_input(BenchmarkId::new("inverse_bind", dim), dim, |b, _| {
            let bound = v1.bind(&v2);
            b.iter(|| bound.inverse_bind(black_box(&v2)));
        });
    }
    
    group.finish();
}

fn bench_bundle_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_bundle");
    
    for dim in [1000, 4096, 10000].iter() {
        let vectors: Vec<_> = (0..10).map(|_| HolographicVector::random(*dim)).collect();
        
        group.bench_with_input(BenchmarkId::new("bundle_10", dim), dim, |b, _| {
            b.iter(|| {
                let refs: Vec<_> = vectors.iter().collect();
                HolographicVector::bundle(black_box(&refs))
            });
        });
    }
    
    group.finish();
}

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_similarity");
    
    for dim in [1000, 4096, 10000].iter() {
        let v1 = HolographicVector::random(*dim);
        let v2 = HolographicVector::random(*dim);
        
        group.bench_with_input(BenchmarkId::new("cosine", dim), dim, |b, _| {
            b.iter(|| v1.cosine_similarity(black_box(&v2)));
        });
    }
    
    group.finish();
}

fn bench_codebook_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_codebook");
    
    // Pre-populate codebook with symbols
    let mut codebook = VsaCodebook::new(10000);
    for i in 0..1000 {
        codebook.get_or_create(&format!("symbol_{}", i));
    }
    
    group.bench_function("get_existing", |b| {
        b.iter(|| codebook.get(black_box("symbol_500")));
    });
    
    group.bench_function("find_nearest_5", |b| {
        let query = HolographicVector::random(10000);
        b.iter(|| codebook.find_nearest(black_box(&query), 5));
    });
    
    group.bench_function("find_nearest_10", |b| {
        let query = HolographicVector::random(10000);
        b.iter(|| codebook.find_nearest(black_box(&query), 10));
    });
    
    group.finish();
}

fn bench_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_factorization");
    
    let mut codebook = VsaCodebook::new(4096);
    
    // Create a bundled structure
    let role1 = codebook.get_or_create("role1");
    let role2 = codebook.get_or_create("role2");
    let filler1 = codebook.get_or_create("filler1");
    let filler2 = codebook.get_or_create("filler2");
    
    let record = role1.bind(&filler1).bundle(&role2.bind(&filler2));
    
    group.bench_function("unbind_query", |b| {
        b.iter(|| record.bind(black_box(&role1)));
    });
    
    group.bench_function("full_factorization", |b| {
        b.iter(|| {
            let unbound1 = record.bind(&role1);
            let unbound2 = record.bind(&role2);
            (
                codebook.find_nearest(&unbound1, 1),
                codebook.find_nearest(&unbound2, 1),
            )
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_vector_creation,
    bench_bind_operations,
    bench_bundle_operations,
    bench_similarity,
    bench_codebook_lookup,
    bench_factorization,
);
criterion_main!(benches);
