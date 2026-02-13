use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use langextract_rust::alignment::{TextAligner, AlignmentConfig};
use langextract_rust::data::Extraction;

/// Generate a realistic source document of approximately `target_bytes` size.
fn generate_source_text(target_bytes: usize) -> String {
    let sentences = [
        "The system shall process 100 transactions per second under normal load conditions.",
        "All user data must be encrypted at rest using AES-256 encryption.",
        "The application shall support concurrent access by up to 10,000 users.",
        "Response times for API calls shall not exceed 200 milliseconds at the 99th percentile.",
        "The platform must maintain 99.99% uptime measured on a monthly basis.",
        "Database backups shall be performed every 6 hours with point-in-time recovery.",
        "The system shall comply with GDPR requirements for data processing and storage.",
        "Authentication tokens shall expire after 30 minutes of inactivity.",
        "The service mesh shall route traffic across at least 3 availability zones.",
        "Log retention policy requires all audit logs to be stored for 7 years.",
        "Memory usage shall not exceed 4GB per service instance under peak load.",
        "The CI/CD pipeline shall complete full deployment in under 15 minutes.",
        "File uploads are limited to 50MB per request with chunked transfer encoding.",
        "The recommendation engine shall return results within 500 milliseconds.",
        "All inter-service communication shall use mutual TLS authentication.",
    ];

    let mut text = String::with_capacity(target_bytes + 200);
    let mut i = 0;
    while text.len() < target_bytes {
        text.push_str(sentences[i % sentences.len()]);
        text.push(' ');
        i += 1;
    }
    text
}

/// Create extractions that exist verbatim in the source (exact match).
fn generate_exact_extractions(source: &str, count: usize) -> Vec<Extraction> {
    let words: Vec<&str> = source.split_whitespace().collect();
    let mut extractions = Vec::with_capacity(count);

    for i in 0..count {
        // Pick a 3-6 word span from the source
        let start = (i * 7) % words.len().saturating_sub(6);
        let end = (start + 3 + (i % 4)).min(words.len());
        let text = words[start..end].join(" ");
        extractions.push(Extraction::new(format!("field_{}", i), text));
    }
    extractions
}

/// Create extractions that require fuzzy matching (slightly modified text).
fn generate_fuzzy_extractions(source: &str, count: usize) -> Vec<Extraction> {
    let words: Vec<&str> = source.split_whitespace().collect();
    let mut extractions = Vec::with_capacity(count);

    for i in 0..count {
        let start = (i * 11) % words.len().saturating_sub(8);
        let end = (start + 4 + (i % 3)).min(words.len());
        let mut text = words[start..end].join(" ");
        // Introduce minor mutations: swap a word, add a typo
        if i % 3 == 0 {
            text = text.replacen("the", "teh", 1);
        } else if i % 3 == 1 {
            text.push_str(" extra");
        }
        extractions.push(Extraction::new(format!("fuzzy_{}", i), text));
    }
    extractions
}

fn bench_alignment_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_exact");

    for &(source_size, extraction_count) in &[
        (5_000, 20),
        (5_000, 100),
        (50_000, 100),
        (50_000, 500),
    ] {
        let source = generate_source_text(source_size);
        let extractions = generate_exact_extractions(&source, extraction_count);
        let aligner = TextAligner::new();

        group.bench_with_input(
            BenchmarkId::new(
                format!("src_{}kb", source_size / 1000),
                extraction_count,
            ),
            &extraction_count,
            |b, _| {
                b.iter(|| {
                    let mut exts = extractions.clone();
                    aligner.align_extractions(
                        black_box(&mut exts),
                        black_box(&source),
                        0,
                    ).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_alignment_fuzzy(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_fuzzy");
    group.sample_size(30); // Fuzzy is slower, fewer samples

    for &(source_size, extraction_count) in &[
        (5_000, 20),
        (5_000, 50),
        (50_000, 50),
        (50_000, 100),
    ] {
        let source = generate_source_text(source_size);
        let extractions = generate_fuzzy_extractions(&source, extraction_count);
        let aligner = TextAligner::with_config(AlignmentConfig {
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: 0.4,
            accept_match_lesser: true,
            case_sensitive: false,
            max_search_window: 100,
        });

        group.bench_with_input(
            BenchmarkId::new(
                format!("src_{}kb", source_size / 1000),
                extraction_count,
            ),
            &extraction_count,
            |b, _| {
                b.iter(|| {
                    let mut exts = extractions.clone();
                    aligner.align_extractions(
                        black_box(&mut exts),
                        black_box(&source),
                        0,
                    ).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_alignment_mixed(c: &mut Criterion) {
    let source = generate_source_text(20_000);
    let mut extractions = generate_exact_extractions(&source, 50);
    extractions.extend(generate_fuzzy_extractions(&source, 50));
    let aligner = TextAligner::new();

    c.bench_function("alignment_mixed_100_on_20kb", |b| {
        b.iter(|| {
            let mut exts = extractions.clone();
            aligner.align_extractions(
                black_box(&mut exts),
                black_box(&source),
                0,
            ).unwrap()
        });
    });
}

criterion_group!(benches, bench_alignment_exact, bench_alignment_fuzzy, bench_alignment_mixed);
criterion_main!(benches);
