//! VSA Holographic Memory Example
//!
//! Demonstrates the Vector Symbolic Architecture (VSA) based holographic
//! memory system - "RAG on crack" with compositional memory operations.
//!
//! Run with: `cargo run --example holographic_memory`

use memory_gate_rs::{
    vsa::{HolographicConfig, HolographicStore, HolographicVector, VsaCodebook},
    AgentDomain, KnowledgeStore, LearningContext,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VSA Holographic Memory Demo ===\n");

    // Part 1: Basic VSA Operations
    demo_vsa_operations();

    // Part 2: Holographic Memory Store
    demo_holographic_store().await?;

    // Part 3: Analogical Reasoning
    demo_analogical_reasoning().await?;

    Ok(())
}

/// Demonstrate core VSA operations: bind, bundle, permute
fn demo_vsa_operations() {
    println!("--- Part 1: Core VSA Operations ---\n");

    let dim = 10000; // High dimensionality for near-orthogonality

    // Create a codebook for symbols
    let mut codebook = VsaCodebook::with_seed(dim, 42);

    // Get vectors for concepts
    let country = codebook.get_or_create("country");
    let capital = codebook.get_or_create("capital");
    let france = codebook.get_or_create("france");
    let paris = codebook.get_or_create("paris");
    let germany = codebook.get_or_create("germany");
    let berlin = codebook.get_or_create("berlin");

    println!("Created vectors for: country, capital, france, paris, germany, berlin");

    // Random vectors are nearly orthogonal
    let sim = france.cosine_similarity(&germany);
    println!("Similarity(france, germany) = {sim:.4} (should be ~0, random vectors are orthogonal)");

    // BINDING: Create associations
    // France record: country->france + capital->paris
    let france_country = country.bind(&france);
    let france_capital = capital.bind(&paris);
    let france_record = france_country.bundle(&france_capital);

    // Germany record: country->germany + capital->berlin
    let germany_country = country.bind(&germany);
    let germany_capital = capital.bind(&berlin);
    let germany_record = germany_country.bundle(&germany_capital);

    println!("\nCreated country records using bind + bundle");

    // QUERY: What is the capital of France?
    // Unbind the 'capital' role to get the filler
    let query = france_record.unbind(&capital);
    let paris_sim = query.cosine_similarity(&paris);
    let berlin_sim = query.cosine_similarity(&berlin);

    println!("\nQuery: What is the capital of France?");
    println!("  Similarity to 'paris': {paris_sim:.4}");
    println!("  Similarity to 'berlin': {berlin_sim:.4}");
    println!("  Answer: {} (paris wins!)", if paris_sim > berlin_sim { "paris" } else { "berlin" });

    // BUNDLING: Superposition of multiple items
    let european_capitals = paris.bundle(&berlin);
    println!("\nBundled 'paris' and 'berlin' into european_capitals");
    println!("  Similarity to paris: {:.4}", european_capitals.cosine_similarity(&paris));
    println!("  Similarity to berlin: {:.4}", european_capitals.cosine_similarity(&berlin));
    println!("  (Bundle is similar to both constituents!)");

    // PERMUTATION: Sequence encoding
    let items = vec![
        codebook.get_or_create("first"),
        codebook.get_or_create("second"),
        codebook.get_or_create("third"),
    ];
    let position_base = codebook.get_or_create_role("position");
    let sequence = HolographicVector::encode_sequence(&items, &position_base).unwrap();

    // Decode position 1 (should get "second")
    let decoded = sequence.decode_sequence_position(1, &position_base);
    println!("\nSequence encoding: [first, second, third]");
    println!("  Decoded position 1:");
    println!("    Sim to 'first': {:.4}", decoded.cosine_similarity(&items[0]));
    println!("    Sim to 'second': {:.4}", decoded.cosine_similarity(&items[1]));
    println!("    Sim to 'third': {:.4}", decoded.cosine_similarity(&items[2]));

    println!();
}

/// Demonstrate the holographic memory store
async fn demo_holographic_store() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Part 2: Holographic Memory Store ---\n");

    // Create store with configuration
    let store = HolographicStore::with_config(
        HolographicConfig::with_dimensions(10000)
            .with_threshold(0.05)
            .with_max_traces(10000)
            .with_seed(42),
    );

    // Store operational knowledge
    let experiences = vec![
        ("nginx_restart", "nginx service restart fixed high CPU usage on production servers", AgentDomain::Infrastructure, 0.9),
        ("docker_networking", "docker container networking issues resolved by recreating bridge network", AgentDomain::Infrastructure, 0.8),
        ("k8s_pod_crash", "kubernetes pod crashloopbackoff caused by memory limit too low", AgentDomain::Infrastructure, 0.85),
        ("redis_memory", "redis out of memory error fixed by increasing maxmemory config", AgentDomain::Infrastructure, 0.75),
        ("api_timeout", "API timeout errors traced to database connection pool exhaustion", AgentDomain::Infrastructure, 0.7),
        ("ssl_renewal", "SSL certificate renewal automated with certbot cron job", AgentDomain::Infrastructure, 0.65),
    ];

    println!("Storing {} operational memories...", experiences.len());

    for (key, content, domain, importance) in experiences {
        let ctx = LearningContext::new(content, domain).with_importance(importance);
        store.store_experience(key, ctx).await?;
    }

    // Query the holographic memory
    println!("\n--- Semantic Search ---");

    let queries = vec![
        "CPU performance issues",
        "container problems",
        "memory errors",
        "certificate management",
    ];

    for query in queries {
        println!("\nQuery: '{query}'");
        let results = store.retrieve_context(query, 3, None).await?;
        for (i, ctx) in results.iter().enumerate() {
            println!("  {}. [importance={:.2}] {}", i + 1, ctx.importance, ctx.content);
        }
    }

    // Show statistics
    let stats = store.stats().await;
    println!("\n--- Store Statistics ---");
    println!("  Traces: {}", stats.trace_count);
    println!("  Codebook symbols: {}", stats.codebook_size);
    println!("  Vector dimensions: {}", stats.dimensions);
    println!("  Avg importance: {:.3}", stats.avg_importance);

    Ok(())
}

/// Demonstrate analogical reasoning with VSA
async fn demo_analogical_reasoning() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Part 3: Analogical Reasoning ---\n");

    let store = HolographicStore::with_config(
        HolographicConfig::with_dimensions(10000)
            .with_threshold(0.1)
            .with_seed(123),
    );

    // Store some knowledge for the codebook to learn
    store.store_experience(
        "k1",
        LearningContext::new("france paris capital city", AgentDomain::General),
    ).await?;
    store.store_experience(
        "k2", 
        LearningContext::new("germany berlin capital city", AgentDomain::General),
    ).await?;
    store.store_experience(
        "k3",
        LearningContext::new("spain madrid capital city", AgentDomain::General),
    ).await?;

    // Analogical query: france:paris :: germany:?
    // This uses the relational structure to find analogies
    println!("Analogical Query: france:paris :: germany:?");
    let results = store.analogical_search("france", "paris", "germany", 5).await;
    println!("Top results:");
    for (symbol, similarity) in results.iter().take(3) {
        println!("  {symbol}: {similarity:.4}");
    }

    println!("\n=== Demo Complete ===");
    println!("\nThe holographic memory provides:");
    println!("  ✓ Compositional representations (bind/bundle)");
    println!("  ✓ Content-addressable retrieval");
    println!("  ✓ Analogical reasoning support");
    println!("  ✓ Graceful degradation under noise");
    println!("  ✓ Efficient similarity search");

    Ok(())
}
