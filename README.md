# LangExtract (Rust)

A Rust library for extracting structured, source-grounded information from unstructured text using LLMs. Every extraction is mapped back to exact character offsets in the original document, enabling verification and interactive highlighting.

The core workflow: provide a few examples of what to extract, the library builds a few-shot prompt, chunks large documents, sends chunks to an LLM in parallel, parses structured JSON responses, aligns extracted values back to character positions in the source text, then deduplicates and aggregates results.

## Key Features

- **High-performance async processing** with configurable concurrency via `buffer_unordered`
- **Multiple provider support** — OpenAI, Ollama, and custom HTTP APIs
- **Character-level alignment** — exact match then fuzzy word-overlap fallback
- **Validation and type coercion** — schema validation, raw data preservation, automatic type detection
- **Visualization** — export to interactive HTML, Markdown, JSON, and CSV
- **Multi-pass extraction** — improved recall through targeted reprocessing of low-yield chunks
- **Semantic chunking** — intelligent text splitting via `semchunk-rs` with sentence boundary awareness
- **Memory efficient** — zero-copy document sharing via `Arc`, pre-computed tokenization

## Quick Start

### CLI Installation

**From source (requires Rust):**
```bash
cargo install langextract-rust --features cli
```

**From repository:**
```bash
git clone https://github.com/modularflow/langextract-rust
cd langextract-rust
cargo install --path . --features cli
```

### CLI Usage

```bash
# Initialize configuration
lx-rs init --provider ollama

# Extract from text
lx-rs extract "John Doe is 30 years old" --prompt "Extract names and ages" --provider ollama

# Process files with HTML export
lx-rs extract document.txt --examples examples.json --export html --provider ollama

# Test provider connectivity
lx-rs test --provider ollama

# List available providers
lx-rs providers
```

### Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
langextract-rust = "0.4"
```

Basic example:

```rust
use langextract_rust::{
    extract, ExtractConfig, FormatType,
    ExampleData, Extraction,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ExampleData::new(
            "John Doe is 30 years old and works as a doctor".to_string(),
            vec![
                Extraction::new("person".to_string(), "John Doe".to_string()),
                Extraction::new("age".to_string(), "30".to_string()),
                Extraction::new("profession".to_string(), "doctor".to_string()),
            ],
        )
    ];

    let config = ExtractConfig {
        model_id: "mistral".to_string(),
        model_url: Some("http://localhost:11434".to_string()),
        temperature: 0.3,
        max_char_buffer: 8000,
        max_workers: 6,
        ..Default::default()
    };

    let result = extract(
        "Alice Smith is 25 years old and works as a doctor. Bob Johnson is 35 and is an engineer.",
        Some("Extract person names, ages, and professions from the text"),
        &examples,
        config,
    ).await?;

    println!("Extracted {} items", result.extraction_count());

    if let Some(extractions) = &result.extractions {
        for e in extractions {
            println!("  [{}] '{}' at {:?}",
                e.extraction_class,
                e.extraction_text,
                e.char_interval,
            );
        }
    }

    Ok(())
}
```

## CLI Reference

### Extract Command

```bash
# From file with options
lx-rs extract document.txt \
  --examples patterns.json \
  --provider openai \
  --model gpt-4o \
  --max-chars 12000 \
  --workers 10 \
  --batch-size 6 \
  --temperature 0.1 \
  --multipass \
  --passes 3 \
  --export html \
  --show-intervals \
  --verbose

# From URL
lx-rs extract "https://example.com/article.html" \
  --prompt "Extract key facts" \
  --provider openai
```

### Configuration Commands

```bash
lx-rs init --provider ollama          # Initialize config
lx-rs init --provider openai --force  # Overwrite existing
lx-rs test --provider ollama          # Test connectivity
lx-rs providers                       # List providers
```

### Configuration Files

**examples.json**
```json
[
  {
    "text": "Dr. Sarah Johnson works at Mayo Clinic in Rochester, MN",
    "extractions": [
      {"extraction_class": "person", "extraction_text": "Dr. Sarah Johnson"},
      {"extraction_class": "organization", "extraction_text": "Mayo Clinic"},
      {"extraction_class": "location", "extraction_text": "Rochester, MN"}
    ]
  }
]
```

**.env**
```bash
OPENAI_API_KEY=your_openai_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

## Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo | Via `async-openai`, feature-gated (`--features openai`) |
| **Ollama** | mistral, llama2, codellama, qwen | Local inference via HTTP to `/api/generate` |
| **Custom** | Any OpenAI-compatible API | For vLLM, LiteLLM, and other compatible endpoints |

### Provider Setup

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Ollama (local)
ollama serve
ollama pull mistral
```

## Configuration

The `ExtractConfig` struct controls extraction behavior:

```rust
let config = ExtractConfig {
    model_id: "mistral".to_string(),
    temperature: 0.3,              // Lower = more consistent output
    max_char_buffer: 8000,         // Characters per chunk
    batch_length: 6,               // Chunks per batch
    max_workers: 8,                // Concurrent workers
    extraction_passes: 1,          // Passes (use with enable_multipass)
    enable_multipass: false,       // Multi-pass extraction
    debug: false,                  // Debug output and raw file saving
    ..Default::default()
};
```

### Tuning Guidelines

- **max_workers**: 6-12 for parallel throughput
- **batch_length**: 4-8 for optimal batching
- **max_char_buffer**: 6000-12000 characters per chunk
- **temperature**: 0.1-0.3 for consistent extraction

## Advanced Features

### Validation and Type Coercion

```rust
use langextract_rust::{ValidationConfig};

let validation_config = ValidationConfig {
    enable_schema_validation: true,
    enable_type_coercion: true,    // "$1,234" -> 1234.0, "95%" -> 0.95
    require_all_fields: false,
    save_raw_outputs: false,
    ..Default::default()
};
```

Supported coercion types: integers, floats, booleans, currencies, percentages, emails, phone numbers, dates, URLs.

### Visualization

```rust
use langextract_rust::visualization::{export_document, ExportConfig, ExportFormat};

let config = ExportConfig {
    format: ExportFormat::Html,
    title: Some("Analysis".to_string()),
    highlight_extractions: true,
    show_char_intervals: true,
    ..Default::default()
};

let html = export_document(&annotated_doc, &config)?;
std::fs::write("analysis.html", html)?;
```

### Provider Configuration

```rust
use langextract_rust::providers::ProviderConfig;

let openai = ProviderConfig::openai("gpt-4o-mini", Some(api_key));
let ollama = ProviderConfig::ollama("mistral", Some("http://localhost:11434".to_string()));
```

## Error Handling

```rust
use langextract_rust::LangExtractError;

match extract(/* ... */).await {
    Ok(result) => println!("{} extractions", result.extraction_count()),
    Err(LangExtractError::ConfigurationError(msg)) => {
        eprintln!("Configuration: {}", msg);
    }
    Err(LangExtractError::InferenceError { message, provider, .. }) => {
        eprintln!("Inference ({}): {}", provider.unwrap_or("unknown".into()), message);
    }
    Err(LangExtractError::NetworkError(e)) => {
        eprintln!("Network: {}", e);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Architecture

```
Text Input -> extract() -> Prompting -> Chunking -> LLM Inference -> Parsing -> Alignment -> Aggregation -> Result
```

Key modules:
- `annotation.rs` — orchestrates the chunk-infer-parse-align loop
- `chunking.rs` — semantic and token-based text splitting
- `alignment.rs` — exact + fuzzy character offset mapping
- `resolver.rs` — JSON parsing, repair, and type coercion
- `multipass.rs` — multi-pass extraction with quality scoring
- `pipeline.rs` — multi-step extraction with dependency resolution
- `visualization.rs` — HTML, Markdown, CSV, JSON export

See [SPEC.md](SPEC.md) for the complete technical specification.

## Testing

```bash
cargo test --lib
```

## Documentation

- [SPEC.md](SPEC.md) — Technical specification, architecture, known issues, and fix priorities

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

This is a Rust port of Google's [langextract](https://github.com/google/langextract) Python library.

```bibtex
@misc{langextract,
  title={langextract},
  author={Google Research Team},
  year={2024},
  publisher={GitHub},
  url={https://github.com/google/langextract}
}
```
