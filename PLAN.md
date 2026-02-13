# langextract-rust: Cross-Referenced Assessment & Remaining Work Plan

**Date:** 2026-02-12 (updated 2026-02-13)  
**Based on:** Full source audit + two independent LLM reviews + project SPEC.md

---

## Implementation Status

### Phase 1 — Correctness: COMPLETE

All six correctness bugs have been fixed:

| Item | Description | Status |
|------|-------------|--------|
| 1.1 | Wire tiktoken into semantic chunking | **Done.** `chunking.rs` uses `tiktoken_rs::cl100k_base()` with BPE `encode_with_special_tokens` |
| 1.2 | Fix semantic chunk position tracking | **Done.** Uses cumulative `current_pos` with `starts_with` check and `log::warn!` fallback |
| 1.3 | Fix semantic chunk merging offsets | **Done.** Builds merged text via `text[merged_start..merged_end]` from original source |
| 1.4 | Fix fuzzy alignment char-position reconstruction | **Done.** Pre-computes `word_byte_offsets` from pointer arithmetic; fuzzy path uses them directly |
| 1.5 | Fix type coercion order | **Done.** Integer (step 7) before boolean (step 9); boolean only matches keyword strings |
| 1.6 | Multipass uses user config | **Done.** `ExtractConfig` fields wired through to `MultiPassConfig` |

### Phase 2 — Code Quality & API Hygiene: COMPLETE

All five hygiene items have been addressed:

| Item | Description | Status |
|------|-------------|--------|
| 2.1 | Route println! through logging | **Done.** No `println!` in `multipass.rs` or `pipeline.rs`; remaining in `logging.rs` (ConsoleProgressHandler, intentional) and `chunking.rs` (tests only) |
| 2.2 | Unify multi-pass API | **Done.** Removed `extraction_passes` field. Single switch: `enable_multipass: bool` + `multipass_max_passes: usize` (default 2). Routing simplified to `if config.enable_multipass`. `extraction_passes` param removed from `annotate_text` and all downstream methods. CLI `--passes N` auto-enables multipass when N > 1. |
| 2.3 | Unify CharInterval types | **Done.** `tokenizer.rs` has `TokenCharSpan` (plain `usize`) with `From<TokenCharSpan> for CharInterval` |
| 2.4 | Remove dead code | **Done.** Deleted `http_client.rs` (~340 lines), removed `find_fuzzy_match` wrapper from `alignment.rs`, removed unused `format_type`/`fence_output` fields and dead `parse_response`/`parse_json_response`/`parse_single_item` from `annotation.rs`. Zero `#[allow(dead_code)]` remaining. ~456 lines removed total. |
| 2.5 | Default model_id | **Done.** Defaults to `"gpt-4o-mini"` with working OpenAI provider |

---

## Part 1: Cross-Reference — Where the Two Reviews Agree, Disagree, and What Both Miss

### Agreement Points (both reviews + my audit confirm)

| Issue | Review A | Review B | Actual Code State |
|-------|----------|----------|-------------------|
| tiktoken not wired into semantic chunking | "not yet fully utilized" | "your own tokenizer—not model-specific token counting" | **Fixed.** `chunking.rs` now uses `tiktoken_rs::cl100k_base()` BPE tokenizer |
| println! bypasses logging | Mentioned under observability | "library code doing println! debugging in resolver/multipass paths" | **Fixed.** `println!` removed from library code paths |
| Multi-pass API confusion | "move beyond first-pass-wins" | "TODO gap / unify multi-pass story" | **Fixed.** Unified to single `enable_multipass` switch; `extraction_passes` removed |
| Consensus merging absent | Suggests majority vote | Suggests vote counts + tie-breaking | **Still open.** `filter_and_deduplicate_extractions` uses HashSet of normalized text—first-seen wins, no voting |
| Streaming API absent | "architecturally complex, low priority" | "straightforward if chunking is refactored to indices" | **Still open.** `aggregate_chunk_results` collects everything before returning |

### Disagreement Points

| Issue | Review A (first) | Review B (uploaded) | My Assessment |
|-------|-------------------|---------------------|---------------|
| **Alignment lowercasing** | "Fixed — pre-lowercases once per chunk" | "Lowercases entire source text and each extraction text, allocates per extraction" | **Review B is working from stale information.** `align_extractions()` pre-lowercases once per chunk (line 70). Per-extraction lowercasing only happens in the `align_single_extraction` public API (line 138) which is used in tests and single-text paths, not the hot chunk path. |
| **rayon / spawn_blocking** | "Unwarranted — alignment is microseconds vs seconds of API" | "Minimum fix is spawn_blocking around alignment" | **Review A is correct for the normal case.** Alignment on a typical chunk (<20 extractions, few KB) takes microseconds. spawn_blocking adds overhead (~5μs per dispatch). Only becomes worth it for the degenerate case of local models + massive documents. Not a priority. |
| **aho-corasick** | "Unwarranted — str::find already SIMD-optimized" | "Add exact-match prefilter as Phase 2" | **Review A is more accurate.** Each extraction is searched individually, so aho-corasick's multi-pattern advantage doesn't apply. However, Review B's phasing is sensible: if you batch all extraction strings into one AC automaton per chunk, you could do one pass instead of N `str::find` calls. Low priority since exact match is already fast. |
| **RawValue** | "Completely unwarranted" | "plausible but not top bottleneck" | **Agree with both partially.** Not a priority, but Review B's framing is more measured. |
| **Zero-copy chunking** | "Largely addressed via Arc<Document>" | "TextChunk holds text: String, refactor to (start,end) indices" | **Review B is more precise here.** `Arc<Document>` prevents full document clones, but each `TextChunk` still owns a `text: String` copy of its slice. The token-based path pre-computes and stores `chunk_text: Some(String)`. True zero-copy would replace `text: String` with `(start, end)` indices into a shared source. |

### What Both Reviews Miss

Issues found in the codebase that neither review flags (status updated):

1. ~~**Fuzzy alignment char-position reconstruction is broken**~~ → **Fixed (1.4)**

2. ~~**Semantic chunk position tracking uses String::find with silent fallback**~~ → **Fixed (1.2)**

3. ~~**Semantic chunk merging corrupts offsets**~~ → **Fixed (1.3)**

4. ~~**Duplicate `CharInterval` types**~~ → **Fixed (2.3).** Renamed to `TokenCharSpan` with `From` impl.

5. ~~**HttpClient is dead code**~~ → **Fixed (2.4).** Deleted entirely.

6. ~~**multipass hardcodes chunking params**~~ → **Fixed (1.6)**

7. ~~**Type coercion: `"1"` → `true`, `"0"` → `false`**~~ → **Fixed (1.5)**

---

## Part 2: What Remains To Be Coded — Prioritized Plan

### ~~Tier 1: Correctness Bugs~~ — ALL COMPLETE

All six correctness bugs (1.1–1.6) have been fixed. See status table above.

### ~~Tier 2: Code Quality & API Hygiene~~ — ALL COMPLETE

All five hygiene items (2.1–2.5) have been addressed. See status table above.

---

### Tier 3: Performance & Feature Improvements (Next up)

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
**Problem:** The Python library's core value prop is Gemini's controlled generation.  
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
| PyO3 bindings | Python wrapper around core | After all Tier 1-2 bugs fixed ✅ |
| N-API bindings | Node.js wrapper | After PyO3 validates the FFI boundary design |
| Rate limiting | Semaphore per provider for cloud APIs | When concurrent chunk processing is actually used at scale |
| Aho-Corasick prefilter | Batch exact-match before fuzzy | Only if benchmarks show alignment is a bottleneck |

---

## Part 3: Recommended Execution Order

```
Phase 1 — Correctness ✅ COMPLETE
├── 1.1  Wire tiktoken into semantic chunking ✅
├── 1.2  Fix semantic chunk position tracking ✅
├── 1.3  Fix semantic chunk merging offsets ✅
├── 1.4  Fix fuzzy alignment char-position reconstruction ✅
├── 1.5  Fix type coercion order ✅
└── 1.6  Thread config into multipass chunking params ✅

Phase 2 — Hygiene ✅ COMPLETE
├── 2.1  Route println! through logging (multipass + pipeline) ✅
├── 2.2  Unify multi-pass API (remove extraction_passes flag) ✅
├── 2.3  Unify CharInterval types ✅
├── 2.4  Remove dead code (http_client, dead_code fns) ✅
└── 2.5  Fix default model_id ✅

Phase 3 — Performance foundations (est. 2-3 days)       ← YOU ARE HERE
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

Phases 1 and 2 are complete. The library now returns correct character offsets, properly merges chunks, coerces types in the right order, and has a clean single-knob multipass API with no dead code. Phase 3 should start with tracing and benchmarks to establish baselines before optimizing.
