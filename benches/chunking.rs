use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use langextract_rust::chunking::{TextChunker, ChunkingConfig, ChunkingStrategy};

/// Generate a realistic document with paragraphs, headings, and varied sentence lengths.
fn generate_document(target_bytes: usize) -> String {
    let paragraphs = [
        "The architecture employs a microservices pattern with each service owning its data store. \
         Services communicate through an event bus using CloudEvents specification. \
         This ensures loose coupling while maintaining eventual consistency across boundaries.",

        "Performance requirements dictate that the system must handle 10,000 concurrent WebSocket \
         connections per node. Each connection maintains a heartbeat interval of 30 seconds. \
         Load balancing uses consistent hashing to minimize connection migration during scaling events.",

        "Security considerations include mandatory mTLS for all east-west traffic within the cluster. \
         JWT tokens are validated at the API gateway level with JWKS rotation every 24 hours. \
         Rate limiting is enforced per tenant with configurable burst allowances.",

        "The data pipeline processes approximately 2TB of raw event data daily. Events are first \
         landed in a staging area, then validated against the schema registry before being \
         transformed and loaded into the analytical data warehouse. CDC streams provide near \
         real-time updates to downstream consumers.",

        "Monitoring and observability are built on OpenTelemetry with traces, metrics, and logs \
         correlated by trace ID. Custom dashboards track the four golden signals: latency, traffic, \
         errors, and saturation. Alerting rules use multi-window burn rate calculations.",

        "The deployment model uses blue-green deployments for stateless services and canary \
         releases for stateful components. Rollback is automated when error rates exceed the \
         baseline by more than two standard deviations within the first 10 minutes.",

        "Database operations use connection pooling with a maximum of 50 connections per service \
         instance. Read replicas are used for reporting queries to avoid impacting transactional \
         workloads. Schema migrations are applied using a forward-only versioned approach.",

        "The search subsystem indexes approximately 500 million documents using an inverted index \
         with BM25 scoring. Faceted navigation supports hierarchical category filtering with \
         sub-second response times for 95th percentile queries.",
    ];

    let mut text = String::with_capacity(target_bytes + 500);
    let mut i = 0;
    while text.len() < target_bytes {
        if i > 0 && i % 3 == 0 {
            text.push_str(&format!("\n\n## Section {}\n\n", i / 3));
        }
        text.push_str(paragraphs[i % paragraphs.len()]);
        text.push_str("\n\n");
        i += 1;
    }
    text
}

#[allow(deprecated)]
fn bench_chunking_semantic(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_semantic");
    group.sample_size(20); // Semantic chunking involves BPE tokenization

    for &doc_size in &[10_000, 50_000, 100_000] {
        let doc = generate_document(doc_size);
        let chunker = TextChunker::with_config(ChunkingConfig {
            max_chunk_size: 2000,
            strategy: ChunkingStrategy::Semantic,
            ..Default::default()
        });

        group.bench_with_input(
            BenchmarkId::new("doc_size", format!("{}kb", doc_size / 1000)),
            &doc_size,
            |b, _| {
                b.iter(|| {
                    chunker.chunk_text(black_box(&doc), None).unwrap()
                });
            },
        );
    }
    group.finish();
}

#[allow(deprecated)]
fn bench_chunking_fixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_fixed");

    for &doc_size in &[10_000, 50_000, 100_000] {
        let doc = generate_document(doc_size);
        let chunker = TextChunker::with_config(ChunkingConfig {
            max_chunk_size: 2000,
            strategy: ChunkingStrategy::FixedSize,
            ..Default::default()
        });

        group.bench_with_input(
            BenchmarkId::new("doc_size", format!("{}kb", doc_size / 1000)),
            &doc_size,
            |b, _| {
                b.iter(|| {
                    chunker.chunk_text(black_box(&doc), None).unwrap()
                });
            },
        );
    }
    group.finish();
}

#[allow(deprecated)]
fn bench_chunking_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_chunk_sizes");
    group.sample_size(20);

    let doc = generate_document(50_000);

    for &chunk_size in &[500, 1000, 2000, 4000, 8000] {
        let chunker = TextChunker::with_config(ChunkingConfig {
            max_chunk_size: chunk_size,
            strategy: ChunkingStrategy::Semantic,
            ..Default::default()
        });

        group.bench_with_input(
            BenchmarkId::new("max_chunk", chunk_size),
            &chunk_size,
            |b, _| {
                b.iter(|| {
                    chunker.chunk_text(black_box(&doc), None).unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chunking_semantic, bench_chunking_fixed, bench_chunking_chunk_sizes);
criterion_main!(benches);
