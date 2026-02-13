use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use langextract_rust::resolver::{Resolver, ValidationConfig};
use langextract_rust::ExtractConfig;

fn make_resolver(save_raw: bool) -> Resolver {
    let config = ExtractConfig {
        debug: false,
        ..Default::default()
    };
    let validation_config = ValidationConfig {
        save_raw_outputs: save_raw,
        ..Default::default()
    };
    Resolver::with_validation_config(&config, false, validation_config).unwrap()
}

/// Generate a well-formed JSON response with N extractions.
fn generate_json_response(count: usize) -> String {
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(format!(
            r#"{{"name": "Entity {i}", "value": "{val}", "category": "type_{cat}"}}"#,
            i = i,
            val = i * 17 + 3,
            cat = i % 5
        ));
    }
    format!("[{}]", items.join(","))
}

/// Generate a JSON response wrapped in code fences (common LLM output).
fn generate_fenced_response(count: usize) -> String {
    format!("```json\n{}\n```", generate_json_response(count))
}

/// Generate a malformed JSON response that needs repair.
fn generate_malformed_response(count: usize) -> String {
    // Missing closing bracket, trailing comma — common LLM failures
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(format!(
            r#"{{"name": "Entity {i}", "value": "{val}",}}"#,
            i = i,
            val = i * 17 + 3,
        ));
    }
    format!("[{}]", items.join(","))
}

fn expected_fields(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("name_{}", i)).collect()
}

fn bench_parse_clean_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolver_parse_clean");
    let resolver = make_resolver(false);
    let fields = expected_fields(3);

    for &count in &[1, 5, 20, 100] {
        let response = generate_json_response(count);

        group.bench_with_input(
            BenchmarkId::new("extractions", count),
            &count,
            |b, _| {
                b.iter(|| {
                    resolver.validate_and_parse(
                        black_box(&response),
                        black_box(&fields),
                    ).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_parse_fenced_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolver_parse_fenced");
    let resolver = make_resolver(false);
    let fields = expected_fields(3);

    for &count in &[1, 5, 20, 100] {
        let response = generate_fenced_response(count);

        group.bench_with_input(
            BenchmarkId::new("extractions", count),
            &count,
            |b, _| {
                b.iter(|| {
                    resolver.validate_and_parse(
                        black_box(&response),
                        black_box(&fields),
                    ).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_parse_malformed_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolver_parse_malformed");
    let resolver = make_resolver(false);
    let fields = expected_fields(3);

    for &count in &[1, 5, 20] {
        let response = generate_malformed_response(count);

        group.bench_with_input(
            BenchmarkId::new("extractions", count),
            &count,
            |b, _| {
                b.iter(|| {
                    // Malformed may fail — that's fine, we're benchmarking the repair attempt
                    let _ = resolver.validate_and_parse(
                        black_box(&response),
                        black_box(&fields),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_type_coercion(c: &mut Criterion) {
    let resolver = make_resolver(false);
    let fields = vec!["name".to_string(), "count".to_string(), "active".to_string()];

    // Response with values that trigger type coercion
    let response = r#"[
        {"name": "Widget A", "count": "42", "active": "true", "price": "$19.99", "ratio": "3.14"},
        {"name": "Widget B", "count": "100", "active": "false", "price": "$29.50", "ratio": "2.71"},
        {"name": "Widget C", "count": "7", "active": "yes", "price": "$9.00", "ratio": "1.41"}
    ]"#;

    c.bench_function("resolver_type_coercion", |b| {
        b.iter(|| {
            resolver.validate_and_parse(
                black_box(response),
                black_box(&fields),
            ).unwrap()
        });
    });
}

criterion_group!(benches, bench_parse_clean_json, bench_parse_fenced_json, bench_parse_malformed_json, bench_type_coercion);
criterion_main!(benches);
