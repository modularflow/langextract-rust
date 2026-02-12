# langextract-rust — Project Specification

> **Version:** 0.4.3  
> **Last updated:** 2026-02-12  
> **Status:** Functional with critical bugs in the extraction pipeline

---

## 1. Purpose

langextract-rust is a Rust port of Google's [langextract](https://github.com/google/langextract) Python library. It extracts structured, source-grounded information from unstructured text using LLMs. "Source-grounded" means every extraction is mapped back to exact character offsets in the original document, enabling verification and interactive highlighting.

The core workflow is: user provides a few examples of what to extract → the library builds a few-shot prompt → chunks large documents → sends chunks to an LLM in parallel → parses structured JSON responses → aligns extracted values back to character positions in the source text → deduplicates and aggregates results.

---

## 2. Architecture

```
User Input (text / file / URL)
        │
        ▼
   ┌──────────┐
   │ extract() │  ← lib.rs entry point
   └────┬─────┘
        │
        ├─► Prompting (prompting.rs)       — builds few-shot prompt from examples
        ├─► Factory (factory.rs)           — creates LLM provider instance
        ├─► Chunking (chunking.rs)         — splits text into processable chunks
        │       ├─ Semantic (semchunk-rs)
        │       ├─ Token-based (tokenizer.rs + ChunkIterator)
        │       └─ Fixed / Sentence / Paragraph (deprecated)
        │
        ├─► Annotation (annotation.rs)     — orchestrates chunk → LLM → parse → align
        │       ├─ LLM inference (providers/universal.rs)
        │       ├─ Resolver (resolver.rs)   — JSON parsing, validation, type coercion
        │       └─ Alignment (alignment.rs) — exact + fuzzy text alignment
        │
        ├─► Multi-pass (multipass.rs)      — re-processes low-yield chunks
        ├─► Pipeline (pipeline.rs)         — chains multiple extraction steps
        └─► Visualization (visualization.rs) — HTML / Markdown / CSV / JSON export
```

### 2.1 Module Inventory

| Module              | Lines | Role                                                  |
|---------------------|------:|-------------------------------------------------------|
| `visualization.rs`  | 1852  | Export formats (HTML with highlighting, Markdown, CSV, JSON) |
| `chunking.rs`       | 1607  | Chunking strategies, `ChunkIterator`, `ResultAggregator`, tests |
| `resolver.rs`       | 1517  | JSON parsing, response cleaning/repair, type coercion, validation |
| `main.rs`           | 969   | CLI binary (`lx-rs`)                                  |
| `pipeline.rs`       | 808   | Multi-step extraction pipeline with dependency resolution |
| `multipass.rs`      | 660   | Multi-pass extraction for improved recall              |
| `config.rs`         | 635   | Unified configuration system                           |
| `annotation.rs`     | 559   | Core annotate → infer → parse → align loop            |
| `data.rs`           | 512   | Core data types (`Extraction`, `Document`, `AnnotatedDocument`) |
| `tokenizer.rs`      | 451   | Regex-based word/punctuation tokenizer                 |
| `http_client.rs`    | 454   | HTTP client with retry logic (partially unused)        |
| `templates.rs`      | 447   | Template rendering utilities                           |
| `providers/universal.rs` | 432 | Unified OpenAI + Ollama provider                 |
| `alignment.rs`      | 419   | Exact match then word-overlap fuzzy alignment          |
| `prompting.rs`      | 479   | Prompt construction and template rendering             |
| `schema.rs`         | 201   | Schema definitions for structured output               |
| `io.rs`             | 176   | URL detection, text download                           |
| `factory.rs`        | 100   | Model/provider factory                                 |

### 2.2 Provider Support

| Provider   | Status       | Notes                                                    |
|------------|--------------|----------------------------------------------------------|
| OpenAI     | ✅ Working   | Via `async-openai` crate, feature-gated (`--features openai`) |
| Ollama     | ✅ Working   | Via raw HTTP to `/api/generate`                          |
| Gemini     | ❌ Missing   | Default `model_id` is `gemini-2.5-flash` but no provider exists |
| Anthropic  | ❌ Missing   |                                                          |
| Custom     | 🔲 Stub     | Returns error "not yet implemented"                      |

### 2.3 Key Dependencies

| Crate            | Purpose                        | Actually used?                        |
|------------------|--------------------------------|---------------------------------------|
| `tokio`          | Async runtime                  | ✅ Yes                                |
| `reqwest`        | HTTP client for LLM APIs       | ✅ Yes                                |
| `serde` / `serde_json` | Serialization / JSON parsing | ✅ Yes                            |
| `semchunk-rs`    | Semantic text chunking         | ✅ Yes (but with word-count proxy, not BPE tokens) |
| `regex`          | Tokenization, text processing  | ✅ Yes                                |
| `async-openai`   | OpenAI API (feature-gated)     | ✅ Yes                                |
| `rayon`          | CPU-parallel data processing   | ❌ Declared but never imported or used |
| `tiktoken-rs`    | BPE token counting (via semchunk-rs feature) | ❌ Feature enabled but word-count used instead |

---

## 3. Current State — What Works

- **Core single-pass extraction** pipeline is functional for OpenAI and Ollama providers: text → chunk → infer → parse → align → aggregate.
- **Semantic chunking** via `semchunk-rs` splits documents at natural boundaries.
- **Token-based chunking** via `ChunkIterator` respects sentence boundaries and newlines.
- **JSON response parsing** with fallback repair logic (code-fence stripping, malformed JSON detection, wrapped-JSON extraction).
- **Type coercion** for extracted values (integers, floats, booleans, currency, percentages, emails, phones, dates, URLs).
- **Text alignment** maps extracted values to character offsets using exact match then word-overlap fuzzy match.
- **Deduplication** via Jaccard word similarity in `ResultAggregator` (0.8 threshold, enabled by default).
- **Multi-pass extraction** in `multipass.rs` with quality scoring, targeted reprocessing of low-yield chunks, and temperature decay.
- **Pipeline system** for multi-step extraction (e.g., extract sections → extract entities per section) with dependency resolution and parallel step execution.
- **Visualization** produces interactive HTML with in-context highlighting, plus Markdown, CSV, and JSON exports.
- **CLI** (`lx-rs`) for init, extract, test, and provider listing.

---

## 4. What's Broken — Bugs That Must Be Fixed

These are correctness issues. They cause silent data loss, ignored configuration, or advertised features that don't work.

### 4.1 🔴 Batch processing silently drops chunks

**Location:** `annotation.rs:338-339`

```rust
let batch_futures: Vec<_> = chunk_batch.iter()
    .take(effective_workers)   // ← only processes first N chunks per batch
    .map(|chunk| self.process_chunk(...))
    .collect();
let batch_results = join_all(batch_futures).await;
```

When `batch_length > max_workers` (which is common — defaults are `batch_length=10`, `max_workers=10`, but users often set fewer workers), `.take(effective_workers)` truncates each batch and the remaining chunks are never processed. There is no warning, no error, and no retry. Data is silently lost.

**Fix:** Replace the batch-loop-with-take pattern with bounded streaming concurrency:
```rust
use futures::stream::{self, StreamExt};

let results: Vec<_> = stream::iter(chunks.iter())
    .map(|chunk| self.process_chunk(chunk, resolver, additional_context, debug))
    .buffer_unordered(max_workers)
    .collect()
    .await;
```

This processes ALL chunks, keeps exactly `max_workers` in flight at all times, and handles stragglers gracefully.

### 4.2 🔴 Hardcoded inference parameters ignore user configuration

**Location:** `annotation.rs:118-120`

```rust
let mut kwargs = HashMap::new();
kwargs.insert("temperature".to_string(), serde_json::json!(1));
kwargs.insert("max_completion_tokens".to_string(), serde_json::json!(8000));
```

The `Annotator` hardcodes `temperature=1` and `max_completion_tokens=8000`, completely ignoring the user's `ExtractConfig` which has `temperature: 0.5` as default. The `Annotator` struct doesn't receive or store the config — it only gets the language model, prompt template, format type, and fence_output flag.

**Impact:**
- `temperature=1` produces maximally variable LLM output, directly increasing JSON parse failures and reducing extraction consistency. The user's configured temperature (default 0.5) is ignored.
- `max_completion_tokens=8000` is 10-40x more than most extraction schemas need (typical: 200-500 tokens), wasting latency and API cost on every single LLM call.
- Users have no way to control these values even by setting the config.

**Fix:** Thread the `ExtractConfig` (or at minimum `temperature` and a `max_output_tokens` field) into the `Annotator`, and use those values when building `kwargs`. Default `max_completion_tokens` to something proportional to the expected output size (e.g., `num_extraction_classes × 200`).

### 4.3 🔴 Multi-pass extraction in `annotation.rs` is a no-op

**Location:** `annotation.rs:360-371`

```rust
if extraction_passes > 1 {
    if debug {
        report_progress(ProgressEvent::Debug {
            operation: "multipass".to_string(),
            details: format!("Running {} additional extraction passes", extraction_passes - 1),
        });
    }
    // TODO: Implement multi-pass extraction
    // For now, we just use the single pass results
}
```

The `process_text_chunks_in_batches` method accepts an `extraction_passes` parameter, logs that it's "running additional passes," but does nothing. A fully implemented `MultiPassProcessor` exists in `multipass.rs` and is used when `config.enable_multipass && config.extraction_passes > 1` in `lib.rs:318`. But the two code paths are separate — a user setting `extraction_passes: 3` without `enable_multipass: true` hits this dead branch and silently gets 1 pass.

**Fix:** Either remove the dead code path and always route multi-pass through `MultiPassProcessor`, or wire the actual multipass logic into this branch. At minimum, log a warning telling users to enable `enable_multipass`.

### 4.4 🟠 No Gemini provider despite being the default model

**Location:** `lib.rs:144`, `providers/config.rs`

The default `model_id` is `"gemini-2.5-flash"`, matching the Python library. But there's no Gemini provider. The original Python library's primary value proposition is Gemini's controlled generation (`response_mime_type: "application/json"` + `response_schema`), which guarantees structurally valid JSON output. Without this, extraction relies entirely on prompt-based JSON compliance, which is less reliable.

**Fix:** Add a native Gemini provider using the `generativelanguage.googleapis.com/v1beta` API. The key feature to implement is `generationConfig.response_mime_type` and `response_schema` for schema-enforced structured output.

---

## 5. What's Slow — Performance Bottlenecks

These don't cause incorrect results but significantly degrade throughput and memory usage on real workloads.

### 5.1 Document cloned per chunk in `ChunkIterator`

**Location:** `chunking.rs:621, 626, 666`

`ChunkIterator` stores `document: Option<&'a Document>` and calls `self.document.cloned()` for every chunk it yields. `Document` owns `text: String`, so each clone deep-copies the entire document text. For a 1MB document producing 100 chunks, that's ~100MB of unnecessary allocation.

**Fix (ordered by impact):**
1. Change `Document { text: Arc<str> }` (or `Arc<String>`) so clones share the underlying allocation.
2. Change `TokenChunk` to hold `Arc<Document>` instead of `Option<Document>`.
3. If lifetimes are manageable, hold `&'a Document` references instead of cloning.

### 5.2 `TokenChunk` re-tokenizes the entire document on every access

**Location:** `chunking.rs:165-191`, `chunking.rs:204-234`

Both `chunk_text()` and `char_interval()` call `tokenizer.tokenize(&document.text)` from scratch each time. For 50 chunks from one document, that's 100+ full re-tokenizations. `TokenChunk` has cache fields (`chunk_text: Option<String>`, `char_interval: Option<CharInterval>`) but they're never populated because the methods take `&self` (no mutation).

This compounds with 5.1 — each chunk first clones the document, then re-tokenizes the cloned copy.

**Fix:** Pre-compute `chunk_text` and `char_interval` during chunk creation in `ChunkIterator` (the tokenized text is already available there), storing the results in the `TokenChunk`. Alternatively, use `OnceCell` for lazy interior-mutable caching.

### 5.3 Alignment re-lowercases the source text per extraction

**Location:** `alignment.rs:83-93`

```rust
let search_text = source_text.to_lowercase();  // called for EVERY extraction
```

For a chunk with 20 extractions from a 10KB chunk, this creates 20 × 10KB = 200KB of throwaway lowercase copies. On large documents with hundreds of extractions, this becomes the dominant allocation cost in the alignment phase.

The fuzzy matching (`alignment.rs:153-234`) additionally lowercases all words again inside `calculate_word_similarity`, and uses `Vec::contains` (O(n)) instead of `HashSet` (O(1)) for word lookup.

**Fix:**
- Pre-lowercase the source text once per chunk, pass the lowercase view into `align_extractions`.
- Build the source word list and word `HashSet` once, reuse across all extractions.
- Reuse the tokenized text from the chunking pipeline (`process_token_chunked_text` already tokenizes) instead of re-deriving word boundaries.

### 5.4 `sanitize_text` recompiles regex on every call

**Location:** `chunking.rs:238-249`

```rust
fn sanitize_text(text: &str) -> LangExtractResult<String> {
    let sanitized = regex::Regex::new(r"\s+")   // compiled every call
        .replace_all(text.trim(), " ")
        .to_string();
```

Called once per chunk. Regex compilation is ~microseconds per call but adds up across hundreds of chunks.

**Fix:**
```rust
use once_cell::sync::Lazy;
static WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
```

### 5.5 Semantic chunk position tracking uses substring search

**Location:** `chunking.rs:440-445`

```rust
let start_pos = if let Some(found_pos) = text[current_pos..].find(&chunk_text) {
    current_pos + found_pos
} else {
    current_pos   // silent fallback — wrong position
};
```

`String::find` is O(n·m) per chunk. If the same sentence appears twice in a document, the second occurrence gets the first occurrence's character offset. The fallback uses an incorrect position without warning.

**Fix:** Track cumulative byte offset directly from `semchunk-rs` output — chunks are returned in order and are contiguous, so offset is just `previous_offset + previous_chunk.len()`.

### 5.6 Deduplication is O(n²)

**Location:** `chunking.rs:822-843`

`deduplicate_extractions` compares each new extraction against all previously accepted extractions using Jaccard word similarity, creating new `HashSet`s per comparison. For 500+ extractions this becomes a bottleneck.

**Fix:** Use a `HashMap` keyed on `(extraction_class, normalized_text)` for O(1) exact/near-exact dedup, falling back to similarity comparison only for fuzzy cases.

### 5.7 OpenAI and Ollama providers process prompts sequentially

**Location:** `providers/universal.rs:124-198` (OpenAI), `providers/universal.rs:202-304` (Ollama)

Both provider `infer()` methods loop over `batch_prompts` with a serial `for` loop, sending and awaiting each prompt one at a time. This defeats the purpose of accepting `batch_prompts: &[String]`.

**Fix:** Use `join_all` or `FuturesUnordered` to send prompts concurrently within a batch (with optional rate-limit semaphore for cloud APIs).

---

## 6. What's Wrong — Code Quality & Correctness Issues

### 6.1 `println!()` throughout `resolver.rs` bypasses logging system

**Location:** `resolver.rs:547-569` and throughout

The resolver uses raw `println!()` with emoji (`🔍`, `✅`, `❌`, `🔧`, `💾`) instead of the `report_progress(ProgressEvent::...)` system used everywhere else. These can't be silenced by library consumers, aren't captured by custom progress handlers, and pollute output.

**Fix:** Replace all `println!()` with `report_progress()` or `log::info!()`/`log::debug!()`.

### 6.2 `debug: true` is the default

**Location:** `lib.rs:157`

By default, every extraction call creates a `./raw_outputs/` directory and writes raw LLM responses to disk. This is surprising for library consumers and creates filesystem side effects without user consent.

**Fix:** Default to `debug: false`.

### 6.3 Duplicate `CharInterval` types

**Location:** `data.rs:25-57` vs `tokenizer.rs:27-45`

Two distinct `CharInterval` structs:
- `data::CharInterval` — `start_pos: Option<usize>`, `end_pos: Option<usize>`
- `tokenizer::CharInterval` — `start_pos: usize`, `end_pos: usize`

This forces manual conversion in `chunking.rs:225-228` and is a source of confusion.

**Fix:** Unify into one type, or rename the tokenizer version to `TokenCharSpan`.

### 6.4 Duplicate retry logic

**Location:** `providers/universal.rs:27-71` vs `http_client.rs:63-79`

`UniversalProvider` has its own `retry_with_backoff`. `HttpClient` has a nearly identical one. `UniversalProvider` creates its own `reqwest::Client` and never uses `HttpClient`.

**Fix:** Have `UniversalProvider` use `HttpClient` internally.

### 6.5 `rayon` is a dead dependency

**Location:** `Cargo.toml:44`

`rayon = "1.0"` is declared but never imported or used anywhere. Adds compile time and binary size.

**Fix:** Remove from `Cargo.toml`.

### 6.6 Type coercion order produces incorrect results for common values

**Location:** `resolver.rs:170-226`

Coercion tries boolean before integer, so `"1"` → `true`, `"0"` → `false` instead of integers. Anything matching the date regex (including product codes like `"2024-01-15"`) becomes a date object.

**Fix:** Make coercion field-name-aware or opt-in per field.

### 6.7 Semantic chunk merging corrupts character offsets

**Location:** `chunking.rs:471-474`

When chunks exceed `semantic_max_chunks`, excess chunks are merged by joining text with `" "`. This introduces spaces not present in the original text, so the merged chunk no longer matches the document at its stated `char_offset`.

**Fix:** Concatenate using actual inter-chunk text from the original document, or store sub-ranges.

### 6.8 Expected-fields recomputed on every chunk

**Location:** `annotation.rs:154-160`

Expected fields are collected from examples → `HashSet` → `Vec<String>` on every `process_single_text` call. The examples never change during a request.

**Fix:** Cache in `Annotator` or `PromptTemplateStructured` at construction time.

---

## 7. Improvement Suggestions

### 7.1 Near-Term (Low Effort, High Value)

| Suggestion | Effort | Impact |
|------------|--------|--------|
| Use `semchunk-rs` with actual tiktoken tokenizer instead of word-count | Small | Prevents chunks from exceeding LLM context windows |
| Add semantic chunk overlap (configurable boundary padding) | Small | Prevents entity loss at chunk boundaries |
| Implement `Custom` provider type (currently returns error) | Medium | Enables any OpenAI-compatible API (vLLM, LiteLLM, etc.) |
| Add rate limiting for cloud API providers | Small | Prevents 429 errors when concurrency is fixed |
| Use `include_str!` for HTML visualization template | Small | Moves 1000+ lines of inline HTML/CSS/JS to a template file |

### 7.2 Medium-Term (Moderate Effort, High Value)

| Suggestion | Effort | Impact |
|------------|--------|--------|
| Add native Gemini provider with controlled generation | Large | Enables schema-enforced JSON output (core library value prop) |
| Add Anthropic provider | Medium | Broadens provider support |
| Streaming LLM responses | Medium | Reduces perceived latency for large documents |
| Consensus/voting for multi-pass extractions | Medium | Boosts confidence when same entity found across passes |
| Zero-copy chunking with `Arc<str>` for source text | Medium | Eliminates all document cloning throughout the pipeline |

### 7.3 Long-Term (Architecture)

| Suggestion | Effort | Impact |
|------------|--------|--------|
| Unify the two multi-pass code paths (annotation.rs vs multipass.rs) | Medium | Eliminates the no-op bug class entirely |
| Make `Annotator` config-aware (not just model/prompt/format) | Medium | Enables proper parameter threading |
| Replace `HashMap<String, serde_json::Value>` kwargs with typed inference params | Medium | Type safety, discoverability |
| Add proper JSON Schema generation from examples | Large | Enables schema validation and controlled generation across all providers |
| Benchmark suite with representative documents | Medium | Prevents performance regressions, validates fixes |

---

## 8. Prioritized Fix Order

Combining correctness, quality, and performance considerations:

| # | Issue | Category | Why This Order |
|---|-------|----------|----------------|
| 1 | Fix batch `.take()` — drops chunks silently | 🔴 Data loss | Users are losing extraction results right now |
| 2 | Wire config temperature/max_tokens into Annotator | 🔴 Quality | Every LLM call uses wrong params; biggest quality win |
| 3 | Fix multi-pass no-op in `annotation.rs` | 🔴 Dead feature | Advertised feature silently does nothing |
| 4 | Stop cloning Document per chunk → `Arc` | 🟠 Memory | Biggest memory win; prerequisite for large doc support |
| 5 | Cache tokenization in TokenChunk | 🟠 CPU | Eliminates O(n²) re-tokenization per document |
| 6 | Pre-lowercase source text once in alignment | 🟠 CPU/Memory | Eliminates dominant allocation in alignment phase |
| 7 | Use `buffer_unordered` for streaming concurrency | 🟠 Throughput | Smooth scheduling, better tail latency |
| 8 | Replace `println!` with logging in resolver | 🟡 Library hygiene | Consumers can't silence output |
| 9 | Static regex compilation in `sanitize_text` | 🟡 Easy win | One-line fix, no risk |
| 10 | Default `debug: false` | 🟡 Surprising behavior | Creates files without user consent |
| 11 | Add Gemini provider | 🟡 Feature | Core value prop, but existing providers work |
| 12 | Remove dead `rayon` dependency | 🔵 Cleanup | Reduces compile time |
