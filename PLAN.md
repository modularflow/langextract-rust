# langextract-rust: Cross-Referenced Assessment & Remaining Work Plan

**Date:** 2026-02-12  
**Based on:** Full source audit + two independent LLM reviews + project SPEC.md

---

## Part 1: Cross-Reference — Where the Two Reviews Agree, Disagree, and What Both Miss

### Agreement Points (both reviews + my audit confirm)

| Issue | Review A | Review B | Actual Code State |
|-------|----------|----------|-------------------|
| tiktoken not wired into semantic chunking | "not yet fully utilized" | "your own tokenizer—not model-specific token counting" | **Confirmed.** `chunking.rs:468` uses `split_whitespace().count()` despite `tiktoken-rs` feature in Cargo.toml |
| println! bypasses logging | Mentioned under observability | "library code doing println! debugging in resolver/multipass paths" | **Confirmed.** 13 in multipass.rs, 17 in pipeline.rs, all hardcoded |
| Multi-pass API confusion | "move beyond first-pass-wins" | "TODO gap / unify multi-pass story" | **Confirmed.** TODO replaced with log::warn but two paths still exist; dedup is exact-text-match only |
| Consensus merging absent | Suggests majority vote | Suggests vote counts + tie-breaking | **Confirmed.** `filter_and_deduplicate_extractions` uses HashSet of normalized text—first-seen wins, no voting |
| Streaming API absent | "architecturally complex, low priority" | "straightforward if chunking is refactored to indices" | **Confirmed.** `aggregate_chunk_results` collects everything before returning |

### Disagreement Points

| Issue | Review A (first) | Review B (uploaded) | My Assessment |
|-------|-------------------|---------------------|---------------|
| **Alignment lowercasing** | "Fixed — pre-lowercases once per chunk" | "Lowercases entire source text and each extraction text, allocates per extraction" | **Review B is working from stale information.** `align_extractions()` pre-lowercases once per chunk (line 70). Per-extraction lowercasing only happens in the `align_single_extraction` public API (line 138) which is used in tests and single-text paths, not the hot chunk path. |
| **rayon / spawn_blocking** | "Unwarranted — alignment is microseconds vs seconds of API" | "Minimum fix is spawn_blocking around alignment" | **Review A is correct for the normal case.** Alignment on a typical chunk (<20 extractions, few KB) takes microseconds. spawn_blocking adds overhead (~5μs per dispatch). Only becomes worth it for the degenerate case of local models + massive documents. Not a priority. |
| **aho-corasick** | "Unwarranted — str::find already SIMD-optimized" | "Add exact-match prefilter as Phase 2" | **Review A is more accurate.** Each extraction is searched individually, so aho-corasick's multi-pattern advantage doesn't apply. However, Review B's phasing is sensible: if you batch all extraction strings into one AC automaton per chunk, you could do one pass instead of N `str::find` calls. Low priority since exact match is already fast. |
| **RawValue** | "Completely unwarranted" | "plausible but not top bottleneck" | **Agree with both partially.** Not a priority, but Review B's framing is more measured. |
| **Zero-copy chunking** | "Largely addressed via Arc<Document>" | "TextChunk holds text: String, refactor to (start,end) indices" | **Review B is more precise here.** `Arc<Document>` prevents full document clones, but each `TextChunk` still owns a `text: String` copy of its slice. The token-based path pre-computes and stores `chunk_text: Some(String)`. True zero-copy would replace `text: String` with `(start, end)` indices into a shared source. |

### What Both Reviews Miss

These are real issues I found in the codebase that neither review flags:

1. **Fuzzy alignment char-position reconstruction is broken** (`alignment.rs:230,236`): `source_words[..idx].join(" ")` assumes single-space separation. If the original text has tabs, newlines, or multiple spaces, the returned character positions are wrong. Also allocates throwaway strings on every fuzzy match.

2. **Semantic chunk position tracking uses String::find with silent fallback** (`chunking.rs:482-486`): If the same text appears twice, the second chunk gets the first's offset. The fallback (`current_pos`) is wrong but produces no warning.

3. **Semantic chunk merging corrupts offsets** (`chunking.rs:513-516`): `.join(" ")` introduces spaces not in the original text, so merged chunk text no longer maps to stated character offsets.

4. **Duplicate `CharInterval` types** (`data.rs:26` vs `tokenizer.rs:28`): One with `Option<usize>`, one with plain `usize`. Forces manual conversion in chunking.rs.

5. **HttpClient is dead code** — `UniversalProvider` creates its own `reqwest::Client` and has its own retry logic. `HttpClient` in `http_client.rs` is never used by anything.

6. **multipass hardcodes `max_char_buffer=2000`, `batch_length=1`, `max_workers=1`** (`multipass.rs:228-232`): Completely ignores user's ExtractConfig values. Multi-pass always runs single-threaded regardless of config.

7. **Type coercion: `"1"` → `true`, `"0"` → `false`** (`resolver.rs:281`): Boolean matching includes `"1"` and `"0"`. Since boolean is checked before integer (line 203 vs 208), the value `"1"` becomes `true` instead of integer `1`.

---

## Part 2: What Remains To Be Coded — Prioritized Plan

### Tier 1: Correctness Bugs (These produce wrong results now)

#### 1.1 Wire tiktoken into semantic chunking
**File:** `chunking.rs:465-468`  
**Problem:** Word-count proxy means chunks can exceed LLM context windows.  
**Change:** Replace the closure with actual tiktoken BPE counting. The `semchunk-rs` crate already has `tiktoken-rs` as a feature (enabled in Cargo.toml). Need to instantiate the correct tokenizer for the model being used (cl100k_base for GPT-4, etc.) and pass it as the counter.

```rust
// Current (broken):
let token_counter = Box::new(|s: &str| s.split_whitespace().count());

// Target:
let bpe = tiktoken_rs::cl100k_base().unwrap(); // or model-specific
let token_counter = Box::new(move |s: &str| bpe.encode_with_special_tokens(s).len());
```

**Scope:** ~20 lines changed in `chunk_semantic()`, plus adding a `model_tokenizer` field to `TextChunker` config so the correct BPE is selected based on provider.

#### 1.2 Fix semantic chunk position tracking
**File:** `chunking.rs:480-499`  
**Problem:** Uses `String::find()` to locate chunks in source text. Duplicate text gets wrong offsets. Fallback is silently wrong.  
**Change:** Track cumulative byte offset. `semchunk-rs` returns chunks in order and contiguously.

```rust
// Current (broken):
let start_pos = if let Some(found_pos) = text[current_pos..].find(&chunk_text) {
    current_pos + found_pos
} else {
    current_pos  // ← silent wrong position
};

// Target:
let start_pos = current_pos;
// Verify alignment (log warning if mismatch):
debug_assert!(text[start_pos..].starts_with(&chunk_text),
    "Semantic chunk text doesn't match source at offset {}", start_pos);
```

**Scope:** ~10 lines.

#### 1.3 Fix semantic chunk merging offset corruption
**File:** `chunking.rs:508-524`  
**Problem:** `.join(" ")` inserts spaces not in the original text.  
**Change:** Slice the original text from first-remaining-chunk start to last-remaining-chunk end.

```rust
// Current (broken):
let merged_text = remaining_chunks.iter()
    .map(|c| c.text.as_str())
    .collect::<Vec<_>>()
    .join(" ");

// Target:
let merged_start = remaining_chunks[0].char_offset;
let last = remaining_chunks.last().unwrap();
let merged_end = last.char_offset + last.char_length;
let merged_text = text[merged_start..merged_end].to_string();
```

**Scope:** ~8 lines.

#### 1.4 Fix fuzzy alignment char-position reconstruction
**File:** `alignment.rs:226-237`  
**Problem:** `source_words[..idx].join(" ")` assumes single-space separation. Returns wrong positions on multi-space/tab/newline text. Allocates throwaway strings.  
**Change:** Pre-compute word byte offsets during the `split_whitespace` step and look them up directly.

```rust
// In align_extractions, alongside source_words:
let word_byte_offsets: Vec<(usize, usize)> = search_text
    .split_whitespace()
    .map(|word| {
        let start = word.as_ptr() as usize - search_text.as_ptr() as usize;
        (start, start + word.len())
    })
    .collect();

// Then in find_fuzzy_match_with_words, replace join():
let char_start = word_byte_offsets[start_word_idx].0;
let char_end = word_byte_offsets[end_word_idx - 1].1;
```

**Scope:** ~15 lines changed across two methods.

#### 1.5 Fix type coercion order
**File:** `resolver.rs:155-221`  
**Problem:** Boolean coercion matches `"1"/"0"` before integer coercion.  
**Change:** Move integer/float coercion before boolean, or make boolean only match keyword strings ("true"/"false"/"yes"/"no"), not numeric values.

```rust
// Current order (broken):  ...date → currency → boolean → integer → float
// Target order:             ...date → currency → integer → float → boolean
// AND: remove "1"/"0" from boolean matches
```

**Scope:** ~10 lines reordered + boolean match arms trimmed.

#### 1.6 Multipass ignores user config for chunking params
**File:** `multipass.rs:224-233`  
**Problem:** Hardcoded `max_char_buffer=2000`, `batch_length=1`, `max_workers=1`.  
**Change:** Thread relevant `ExtractConfig` fields into `MultiPassProcessor` or `MultiPassConfig`.

**Scope:** ~20 lines: add fields to `MultiPassConfig`, wire from `lib.rs:324-331`.

---

### Tier 2: Code Quality & API Hygiene (Users hit these as friction)

#### 2.1 Route all println! through logging system
**Files:** `multipass.rs` (13 calls), `pipeline.rs` (17 calls)  
**Problem:** Library consumers cannot silence output.  
**Change:** Replace `println!("[multipass] ...")` with `report_progress(ProgressEvent::Debug { ... })` and `println!("[pipeline] ...")` similarly.

**Scope:** ~60 line changes (mechanical find/replace with formatting adjustments).

#### 2.2 Unify multi-pass API
**Files:** `annotation.rs:408-425`, `lib.rs:322-366`  
**Problem:** `extraction_passes > 1` without `enable_multipass = true` logs a warning but does nothing. Two config knobs for one feature.  
**Change:** Option A (recommended): Remove `extraction_passes` from `ExtractConfig`. If `enable_multipass` is true, use `MultiPassConfig.max_passes`. If false, single pass. One knob, one behavior.  
Option B: If `extraction_passes > 1`, automatically enable multi-pass routing (treat the two flags as an OR).

**Scope:** ~30 lines across lib.rs, annotation.rs, config.

#### 2.3 Unify CharInterval types
**Files:** `data.rs:26`, `tokenizer.rs:28`  
**Problem:** Two structs with the same name, different field types.  
**Change:** Rename `tokenizer::CharInterval` to `TokenCharSpan` (or `ByteSpan`), make it hold plain `usize` values. Keep `data::CharInterval` as the public API type with `Option<usize>`. Add a `From` impl.

**Scope:** ~20 lines + grep/replace through chunking.rs.

#### 2.4 Remove dead code
- **`http_client.rs`**: Entire module unused. `UniversalProvider` has its own client + retry. Either delete or refactor `UniversalProvider` to use it.  
- **`alignment.rs:179-183`**: `find_fuzzy_match` (non-cached version) is `#[allow(dead_code)]`.  
- **`annotation.rs:518-612`**: `parse_response`, `parse_json_response`, `parse_single_item` are all `#[allow(dead_code)]` — the resolver handles parsing now.

**Scope:** Delete ~300 lines or consolidate retry logic into HttpClient.

#### 2.5 Default model_id should work out of box
**File:** `lib.rs:144`  
**Problem:** Default is `"gemini-2.5-flash"` but no Gemini provider exists.  
**Change:** Either add a Gemini provider (Tier 3) or change default to an existing provider. Pragmatic option: default to `"gpt-4o-mini"` with `ProviderType::OpenAI` since that provider works.

**Scope:** 1 line (temporary) or a full provider implementation (Tier 3).

---

### Tier 3: Performance & Feature Improvements (Real but lower priority than correctness)

#### 3.1 Zero-copy TextChunk representation
**File:** `chunking.rs` (TextChunk struct + chunk_semantic + process_token_chunked_text)  
**Problem:** Every TextChunk owns a `text: String` copy of its slice.  
**Change:** Replace `text: String` with `text_range: (usize, usize)` + `source: Arc<str>`. Materialize text only when needed (prompt assembly, serialization).

```rust
pub struct TextChunk {
    pub id: usize,
    source: Arc<str>,          // shared ref to original document
    pub text_range: (usize, usize),  // (start, end) into source
    pub char_offset: usize,    // = text_range.0
    pub char_length: usize,    // = text_range.1 - text_range.0
    // ...
}

impl TextChunk {
    pub fn text(&self) -> &str {
        &self.source[self.text_range.0..self.text_range.1]
    }
}
```

**Scope:** ~100 lines across chunking.rs + annotation.rs (updating all `.text` accesses to `.text()`).  
**Impact:** Eliminates chunk-count × chunk-size bytes of allocation. For 100 chunks averaging 4KB each, saves ~400KB.

#### 3.2 Consensus merging for multi-pass
**File:** `multipass.rs:525-583`  
**Problem:** Dedup is exact-normalized-text match (HashSet). No voting or confidence weighting.  
**Change:** Replace with a merge map keyed on `(extraction_class, normalized_text)`:

```rust
struct MergeCandidate {
    extraction: Extraction,
    vote_count: usize,
    best_quality_score: f32,
    best_alignment_status: Option<AlignmentStatus>,
    pass_numbers: Vec<usize>,
}

// Build map: (class, normalized_text) → MergeCandidate
// For fuzzy matches: group by class, then cluster by Jaccard similarity > 0.8
// Final selection: highest vote_count, tie-break by quality_score
```

**Scope:** ~80 lines replacing current dedup function.

#### 3.3 Add Gemini provider
**Files:** New `providers/gemini.rs` + modifications to `factory.rs`, `providers/mod.rs`  
**Problem:** Default model has no provider. The Python library's core value prop is Gemini's controlled generation.  
**Key feature:** `response_mime_type: "application/json"` + `response_schema` in `generationConfig`. This gives structurally guaranteed JSON output, reducing parse failures.

**Scope:** ~250 lines for provider + ~30 lines for factory wiring.

#### 3.4 Add tracing instrumentation
**Files:** Cargo.toml + all modules with `report_progress` calls  
**Problem:** No way to measure API latency vs alignment time vs chunking overhead.  
**Change:** Add `tracing` crate, instrument key functions with `#[tracing::instrument]`, emit spans for each pipeline phase. The existing `ProgressHandler` can become a tracing subscriber adapter.

**Scope:** ~50 lines of instrumentation + Cargo.toml dep.

#### 3.5 Benchmark suite (criterion)
**File:** New `benches/` directory  
**Problem:** No performance baselines exist.  
**Key benchmarks:**
- `bench_alignment_exact`: 1000 extractions against a 50KB document
- `bench_alignment_fuzzy`: 100 extractions requiring fuzzy match
- `bench_chunking_semantic`: 100KB document → chunks
- `bench_chunking_token`: Same document, token-based
- `bench_dedup`: 500 extractions through deduplication
- `bench_resolver_parse`: Various malformed JSON responses

**Scope:** ~200 lines across 3-4 bench files.

---

### Tier 4: Strategic / Roadmap (After core is solid)

| Item | What | When |
|------|------|------|
| Anthropic provider | Native Claude API support | After Gemini provider pattern established |
| Custom provider | Wire the stub that currently returns error | Small lift, enables vLLM/LiteLLM |
| Streaming API | `Stream<Item = ChunkResult>` | After zero-copy chunking refactor |
| PyO3 bindings | Python wrapper around core | After all Tier 1-2 bugs fixed |
| N-API bindings | Node.js wrapper | After PyO3 validates the FFI boundary design |
| Rate limiting | Semaphore per provider for cloud APIs | When concurrent chunk processing is actually used at scale |
| Aho-Corasick prefilter | Batch exact-match before fuzzy | Only if benchmarks show alignment is a bottleneck |

---

## Part 3: Recommended Execution Order

```
Phase 1 — Correctness (est. 1-2 days)
├── 1.1  Wire tiktoken into semantic chunking
├── 1.2  Fix semantic chunk position tracking  
├── 1.3  Fix semantic chunk merging offsets
├── 1.4  Fix fuzzy alignment char-position reconstruction
├── 1.5  Fix type coercion order
└── 1.6  Thread config into multipass chunking params

Phase 2 — Hygiene (est. 1 day)
├── 2.1  Route println! through logging (multipass + pipeline)
├── 2.2  Unify multi-pass API (remove extraction_passes flag)
├── 2.3  Unify CharInterval types  
├── 2.4  Remove dead code (http_client, dead_code fns)
└── 2.5  Fix default model_id

Phase 3 — Performance foundations (est. 2-3 days)
├── 3.4  Add tracing instrumentation (do this FIRST to measure)
├── 3.5  Benchmark suite (establish baselines)
├── 3.1  Zero-copy TextChunk (measure improvement with benchmarks)
└── 3.2  Consensus merging for multi-pass

Phase 4 — Provider expansion (est. 3-4 days)
├── 3.3  Gemini provider (with controlled generation)
├── 4.x  Anthropic provider
└── 4.x  Custom provider stub → working

Phase 5 — Ecosystem (est. 1-2 weeks)
├── Streaming API
├── PyO3 bindings  
└── N-API bindings
```

### Key Principle

Phases 1 and 2 should be completed before any performance work. Every Tier 3 optimization is meaningless if the library returns wrong character offsets, silently loses extractions from merged chunks, or coerces "1" to `true` instead of an integer. Fix correctness, then measure, then optimize.