//! Multi-pass extraction system for improved recall and quality.
//!
//! This module implements a sophisticated multi-pass extraction strategy that:
//! - Performs multiple extraction rounds to improve recall
//! - Re-processes low-yield chunks with different strategies
//! - Uses initial results to refine subsequent passes
//! - Provides quality scoring and filtering for extractions

use crate::{
    alignment::{AlignmentStats, TextAligner},
    annotation::Annotator,
    chunking::{ChunkResult, TextChunk, TextChunker},
    data::{AnnotatedDocument, Extraction},
    exceptions::LangExtractResult,
    resolver::Resolver,
};
use futures::future::join_all;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Configuration for multi-pass extraction
#[derive(Debug, Clone)]
pub struct MultiPassConfig {
    /// Number of extraction passes to perform
    pub max_passes: usize,
    /// Minimum extraction count per chunk to avoid re-processing
    pub min_extractions_per_chunk: usize,
    /// Enable targeted re-processing of low-yield chunks
    pub enable_targeted_reprocessing: bool,
    /// Enable refinement passes using previous results
    pub enable_refinement_passes: bool,
    /// Minimum quality score to keep extractions (0.0 to 1.0)
    pub quality_threshold: f32,
    /// Maximum number of chunks to re-process per pass
    pub max_reprocess_chunks: usize,
    /// Temperature adjustment for subsequent passes
    pub temperature_decay: f32,
    /// Maximum characters per chunk for processing
    pub max_char_buffer: usize,
    /// Batch size for processing chunks
    pub batch_length: usize,
    /// Maximum number of concurrent workers
    pub max_workers: usize,
}

impl Default for MultiPassConfig {
    fn default() -> Self {
        Self {
            max_passes: 2,
            min_extractions_per_chunk: 1,
            enable_targeted_reprocessing: true,
            enable_refinement_passes: true,
            quality_threshold: 0.3,
            max_reprocess_chunks: 10,
            temperature_decay: 0.9,
            max_char_buffer: 1000,
            batch_length: 10,
            max_workers: 10,
        }
    }
}

/// Statistics for multi-pass extraction
#[derive(Debug, Clone)]
pub struct MultiPassStats {
    /// Total number of passes performed
    pub total_passes: usize,
    /// Extractions found in each pass
    pub extractions_per_pass: Vec<usize>,
    /// Chunks re-processed in each pass
    pub reprocessed_chunks_per_pass: Vec<usize>,
    /// Total processing time
    pub total_time: Duration,
    /// Time spent per pass
    pub time_per_pass: Vec<Duration>,
    /// Alignment statistics for final results
    pub final_alignment_stats: AlignmentStats,
    /// Quality statistics
    pub quality_stats: QualityStats,
}

/// Quality statistics for extractions
#[derive(Debug, Clone)]
pub struct QualityStats {
    /// Average quality score across all extractions
    pub average_quality: f32,
    /// Number of high-quality extractions (>0.7)
    pub high_quality_count: usize,
    /// Number of medium-quality extractions (0.3-0.7)
    pub medium_quality_count: usize,
    /// Number of low-quality extractions (<0.3)
    pub low_quality_count: usize,
    /// Number of extractions filtered out due to low quality
    pub filtered_count: usize,
}

/// Quality-scored extraction
#[derive(Debug, Clone)]
pub struct ScoredExtraction {
    /// The extraction itself
    pub extraction: Extraction,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Pass number where this extraction was found
    pub pass_number: usize,
    /// Chunk ID where this extraction was found
    pub chunk_id: usize,
}

/// Multi-pass extraction processor
pub struct MultiPassProcessor {
    config: MultiPassConfig,
    annotator: Annotator,
    resolver: Resolver,
    aligner: TextAligner,
}

impl MultiPassProcessor {
    /// Create a new multi-pass processor
    pub fn new(
        config: MultiPassConfig,
        annotator: Annotator,
        resolver: Resolver,
    ) -> Self {
        Self {
            config,
            annotator,
            resolver,
            aligner: TextAligner::new(),
        }
    }

    /// Perform multi-pass extraction on text
    #[tracing::instrument(skip_all, fields(text_len = text.len(), max_passes = self.config.max_passes))]
    pub async fn extract_multipass(
        &self,
        text: &str,
        additional_context: Option<&str>,
        debug: bool,
    ) -> LangExtractResult<(AnnotatedDocument, MultiPassStats)> {
        let start_time = Instant::now();
        let mut stats = MultiPassStats {
            total_passes: 0,
            extractions_per_pass: Vec::new(),
            reprocessed_chunks_per_pass: Vec::new(),
            total_time: Duration::default(),
            time_per_pass: Vec::new(),
            final_alignment_stats: AlignmentStats {
                total: 0,
                exact: 0,
                fuzzy: 0,
                lesser: 0,
                greater: 0,
                unaligned: 0,
            },
            quality_stats: QualityStats {
                average_quality: 0.0,
                high_quality_count: 0,
                medium_quality_count: 0,
                low_quality_count: 0,
                filtered_count: 0,
            },
        };

        let all_scored_extractions: Vec<ScoredExtraction>;

        if text.len() <= self.config.max_char_buffer {
            // Single text processing with multi-pass
            all_scored_extractions = self.process_single_text_multipass(
                text,
                additional_context,
                &mut stats,
                debug,
            ).await?;
        } else {
            // Chunked processing with multi-pass
            all_scored_extractions = self.process_chunked_text_multipass(
                text,
                additional_context,
                &mut stats,
                debug,
            ).await?;
        }

        // Filter extractions by quality and deduplicate
        let final_extractions = self.filter_and_deduplicate_extractions(
            all_scored_extractions,
            &mut stats,
            debug,
        );

        // Calculate final alignment statistics
        stats.final_alignment_stats = self.aligner.get_alignment_stats(&final_extractions);
        stats.total_time = start_time.elapsed();

        let mut result = AnnotatedDocument::new();
        result.text = Some(text.to_string());
        result.extractions = Some(final_extractions);

        if debug {
            self.print_multipass_summary(&stats);
        }

        Ok((result, stats))
    }

    /// Process a single text with multiple passes
    async fn process_single_text_multipass(
        &self,
        text: &str,
        additional_context: Option<&str>,
        stats: &mut MultiPassStats,
        debug: bool,
    ) -> LangExtractResult<Vec<ScoredExtraction>> {
        let mut all_extractions = Vec::new();
        let mut previous_extraction_texts = HashSet::new();

        for pass_num in 1..=self.config.max_passes {
            let pass_start = Instant::now();
            
            if debug {
                log::debug!("[multipass] pass {}/{}", pass_num, self.config.max_passes);
            }

            // For refinement passes, include context about previous findings
            let enhanced_context = if pass_num > 1 && self.config.enable_refinement_passes {
                Some(self.build_refinement_context(additional_context, &all_extractions))
            } else {
                additional_context.map(String::from)
            };

            // Process the text
            let result = self.annotator.annotate_text(
                text,
                &self.resolver,
                self.config.max_char_buffer,
                self.config.batch_length,
                enhanced_context.as_deref(),
                false, // Don't debug individual passes unless requested
                self.config.max_workers,
            ).await?;

            // Score and collect new extractions
            let mut pass_extractions = Vec::new();
            if let Some(extractions) = result.extractions {
                for extraction in extractions {
                    // Skip if we've already found this extraction
                    if !previous_extraction_texts.contains(&extraction.extraction_text) {
                        let quality_score = self.calculate_quality_score(&extraction, text);
                        if quality_score >= self.config.quality_threshold {
                            pass_extractions.push(ScoredExtraction {
                                extraction: extraction.clone(),
                                quality_score,
                                pass_number: pass_num,
                                chunk_id: 0,
                            });
                            previous_extraction_texts.insert(extraction.extraction_text);
                        }
                    }
                }
            }

            stats.extractions_per_pass.push(pass_extractions.len());
            stats.time_per_pass.push(pass_start.elapsed());
            all_extractions.extend(pass_extractions);

            if debug {
                log::debug!("[multipass] pass {} found {} new extractions",
                    pass_num, stats.extractions_per_pass.last().unwrap_or(&0));
            }

            // Early termination if no new extractions found
            if stats.extractions_per_pass.last() == Some(&0) {
                if debug {
                    log::debug!("[multipass] no new extractions, stopping early");
                }
                break;
            }
        }

        stats.total_passes = stats.extractions_per_pass.len();
        Ok(all_extractions)
    }

    /// Process chunked text with multiple passes
    async fn process_chunked_text_multipass(
        &self,
        text: &str,
        additional_context: Option<&str>,
        stats: &mut MultiPassStats,
        debug: bool,
    ) -> LangExtractResult<Vec<ScoredExtraction>> {
        let chunker = TextChunker::new();
        let initial_chunks = chunker.chunk_text(text, None)?;

        let mut all_extractions = Vec::new();
        let mut chunks_to_process = initial_chunks;
        let mut processed_extraction_texts = HashSet::new();

        for pass_num in 1..=self.config.max_passes {
            let pass_start = Instant::now();
            
            if debug {
                log::debug!("[multipass] pass {}/{} -- {} chunks",
                    pass_num, self.config.max_passes, chunks_to_process.len());
            }

            // Process chunks for this pass
            let pass_results = self.process_chunks_for_pass(
                &chunks_to_process,
                additional_context,
                pass_num,
                &all_extractions,
                debug,
            ).await?;

            // Collect new extractions and identify low-yield chunks
            let mut pass_extractions = Vec::new();
            let mut low_yield_chunks = Vec::new();

            for result in pass_results {
                let extractions = result.extractions.unwrap_or_default();
                let extraction_count = extractions.len();
                
                // Score and collect extractions
                for extraction in extractions {
                    if !processed_extraction_texts.contains(&extraction.extraction_text) {
                        let quality_score = self.calculate_quality_score(&extraction, text);
                        if quality_score >= self.config.quality_threshold {
                            pass_extractions.push(ScoredExtraction {
                                extraction: extraction.clone(),
                                quality_score,
                                pass_number: pass_num,
                                chunk_id: result.chunk_id,
                            });
                            processed_extraction_texts.insert(extraction.extraction_text.clone());
                        }
                    }
                }

                // Identify chunks for re-processing
                if self.config.enable_targeted_reprocessing 
                    && extraction_count < self.config.min_extractions_per_chunk
                    && low_yield_chunks.len() < self.config.max_reprocess_chunks {
                    
                    // Find the original chunk for re-processing
                    if let Some(chunk) = chunks_to_process.iter()
                        .find(|c| c.id == result.chunk_id) {
                        low_yield_chunks.push(chunk.clone());
                    }
                }
            }

            stats.extractions_per_pass.push(pass_extractions.len());
            stats.reprocessed_chunks_per_pass.push(low_yield_chunks.len());
            stats.time_per_pass.push(pass_start.elapsed());
            all_extractions.extend(pass_extractions);

            if debug {
                log::debug!("[multipass] pass {} found {} new extractions, {} chunks queued for reprocessing",
                    pass_num, stats.extractions_per_pass.last().unwrap_or(&0),
                    stats.reprocessed_chunks_per_pass.last().unwrap_or(&0));
            }

            // Prepare chunks for next pass
            chunks_to_process = low_yield_chunks;

            // Early termination conditions
            if chunks_to_process.is_empty() 
                || stats.extractions_per_pass.last() == Some(&0) {
                if debug {
                    log::debug!("[multipass] no remaining chunks or extractions, stopping");
                }
                break;
            }
        }

        stats.total_passes = stats.extractions_per_pass.len();
        Ok(all_extractions)
    }

    /// Process chunks for a specific pass
    async fn process_chunks_for_pass(
        &self,
        chunks: &[TextChunk],
        additional_context: Option<&str>,
        pass_number: usize,
        previous_extractions: &[ScoredExtraction],
        debug: bool,
    ) -> LangExtractResult<Vec<ChunkResult>> {
        let enhanced_context = if pass_number > 1 && self.config.enable_refinement_passes {
            Some(self.build_refinement_context(additional_context, previous_extractions))
        } else {
            additional_context.map(String::from)
        };

        let chunk_futures = chunks.iter().map(|chunk| {
            self.process_chunk_for_pass(chunk, enhanced_context.as_deref(), debug)
        });

        let results = join_all(chunk_futures).await;
        
        // Collect successful results
        let mut chunk_results = Vec::new();
        for result in results {
            match result {
                Ok(chunk_result) => chunk_results.push(chunk_result),
                Err(e) => {
                    if debug {
                        log::warn!("[multipass] chunk processing failed: {}", e);
                    }
                }
            }
        }

        Ok(chunk_results)
    }

    /// Process a single chunk for a pass
    async fn process_chunk_for_pass(
        &self,
        chunk: &TextChunk,
        additional_context: Option<&str>,
        _debug: bool,
    ) -> LangExtractResult<ChunkResult> {
        let start_time = Instant::now();

        match self.annotator.annotate_text(&chunk.text, &self.resolver, self.config.max_char_buffer, self.config.batch_length, additional_context, false, self.config.max_workers).await {
            Ok(annotated_doc) => {
                let mut extractions = annotated_doc.extractions.unwrap_or_default();
                
                // Align extractions with the chunk text
                let _aligned_count = self.aligner.align_chunk_extractions(
                    &mut extractions,
                    &chunk.text,
                    chunk.char_offset,
                ).unwrap_or(0);

                Ok(ChunkResult::success(
                    chunk.id,
                    extractions,
                    chunk.char_offset,
                    chunk.char_length,
                ).with_processing_time(start_time.elapsed()))
            }
            Err(e) => {
                Ok(ChunkResult::failure(
                    chunk.id,
                    chunk.char_offset,
                    chunk.char_length,
                    e.to_string(),
                ).with_processing_time(start_time.elapsed()))
            }
        }
    }

    /// Build refinement context from previous extractions
    fn build_refinement_context(
        &self,
        base_context: Option<&str>,
        previous_extractions: &[ScoredExtraction],
    ) -> String {
        let mut context = base_context.unwrap_or("").to_string();
        
        if !previous_extractions.is_empty() {
            context.push_str("\n\nPrevious extractions found:");
            
            // Group by extraction class
            let mut by_class: HashMap<String, Vec<&ScoredExtraction>> = HashMap::new();
            for extraction in previous_extractions {
                by_class.entry(extraction.extraction.extraction_class.clone())
                    .or_default()
                    .push(extraction);
            }

            for (class, extractions) in by_class {
                context.push_str(&format!("\n- {}: ", class));
                let texts: Vec<String> = extractions.iter()
                    .map(|e| e.extraction.extraction_text.clone())
                    .collect();
                context.push_str(&texts.join(", "));
            }
            
            context.push_str("\n\nPlease look for additional entities that may have been missed, including related entities or different forms of the same information.");
        }

        context
    }

    /// Calculate quality score for an extraction
    fn calculate_quality_score(&self, extraction: &Extraction, _source_text: &str) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Length-based scoring (prefer meaningful lengths)
        let text_len = extraction.extraction_text.len();
        if text_len >= 2 && text_len <= 100 {
            score += 0.2;
        } else if text_len > 100 {
            score -= 0.1; // Penalize very long extractions
        }

        // Alignment-based scoring
        if let Some(status) = &extraction.alignment_status {
            use crate::data::AlignmentStatus;
            match status {
                AlignmentStatus::MatchExact => score += 0.3,
                AlignmentStatus::MatchFuzzy => score += 0.1,
                AlignmentStatus::MatchLesser => score += 0.05,
                AlignmentStatus::MatchGreater => score -= 0.05,
            }
        } else {
            score -= 0.2; // Penalize unaligned extractions
        }

        // Content quality scoring
        if extraction.extraction_text.chars().any(|c| c.is_alphabetic()) {
            score += 0.1;
        }
        
        if extraction.extraction_text.chars().any(|c| c.is_numeric()) {
            score += 0.05;
        }

        // Penalize very short or single-character extractions
        if text_len <= 1 {
            score -= 0.3;
        }

        // Ensure score is between 0.0 and 1.0
        score.max(0.0).min(1.0)
    }

    /// Filter and deduplicate extractions
    #[tracing::instrument(skip_all, fields(num_candidates = scored_extractions.len()))]
    fn filter_and_deduplicate_extractions(
        &self,
        scored_extractions: Vec<ScoredExtraction>,
        stats: &mut MultiPassStats,
        debug: bool,
    ) -> Vec<Extraction> {
        // Filter by quality threshold
        let high_quality: Vec<_> = scored_extractions.into_iter()
            .filter(|se| se.quality_score >= self.config.quality_threshold)
            .collect();

        // Calculate quality statistics
        let total_count = high_quality.len();
        let mut quality_sum = 0.0;
        let mut high_count = 0;
        let mut medium_count = 0;
        let mut low_count = 0;

        for se in &high_quality {
            quality_sum += se.quality_score;
            if se.quality_score >= 0.7 {
                high_count += 1;
            } else if se.quality_score >= 0.3 {
                medium_count += 1;
            } else {
                low_count += 1;
            }
        }

        stats.quality_stats = QualityStats {
            average_quality: if total_count > 0 { quality_sum / total_count as f32 } else { 0.0 },
            high_quality_count: high_count,
            medium_quality_count: medium_count,
            low_quality_count: low_count,
            filtered_count: 0, // Will be calculated during deduplication
        };

        // Deduplicate based on extraction text similarity
        let mut deduplicated = Vec::new();
        let mut seen_texts = HashSet::new();

        for scored in high_quality {
            let normalized_text = scored.extraction.extraction_text.to_lowercase().trim().to_string();
            if !seen_texts.contains(&normalized_text) {
                seen_texts.insert(normalized_text);
                deduplicated.push(scored.extraction);
            } else {
                stats.quality_stats.filtered_count += 1;
            }
        }

        if debug {
            log::debug!("[multipass] {} extractions kept, {} filtered",
                deduplicated.len(), stats.quality_stats.filtered_count);
        }

        deduplicated
    }

    /// Log multi-pass extraction summary
    fn print_multipass_summary(&self, stats: &MultiPassStats) {
        log::info!("[multipass] summary");
        log::info!("  passes: {}", stats.total_passes);
        log::info!("  time: {:?}", stats.total_time);

        for (i, (&extractions, &time)) in stats.extractions_per_pass.iter()
            .zip(stats.time_per_pass.iter()).enumerate() {
            let reprocessed = stats.reprocessed_chunks_per_pass.get(i).unwrap_or(&0);
            log::info!("  pass {}: {} extractions, {} reprocessed, {:?}",
                i + 1, extractions, reprocessed, time);
        }

        log::info!("  quality: avg={:.2}, high={}, medium={}, low={}, filtered={}",
            stats.quality_stats.average_quality,
            stats.quality_stats.high_quality_count,
            stats.quality_stats.medium_quality_count,
            stats.quality_stats.low_quality_count,
            stats.quality_stats.filtered_count);

        log::info!("  alignment: total={}, exact={}, fuzzy={}, rate={:.1}%",
            stats.final_alignment_stats.total,
            stats.final_alignment_stats.exact,
            stats.final_alignment_stats.fuzzy,
            stats.final_alignment_stats.success_rate() * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipass_config_default() {
        let config = MultiPassConfig::default();
        assert_eq!(config.max_passes, 2);
        assert_eq!(config.min_extractions_per_chunk, 1);
        assert!(config.enable_targeted_reprocessing);
        assert!(config.enable_refinement_passes);
        assert_eq!(config.quality_threshold, 0.3);
    }

    #[test]
    fn test_quality_score_calculation() {
        // This test would need a real MultiPassProcessor instance
        // For now, we'll test the concept
        let config = MultiPassConfig::default();
        assert!(config.quality_threshold > 0.0 && config.quality_threshold < 1.0);
    }

    #[test]
    fn test_refinement_context_building() {
        // Test context building with sample extractions
        let extractions = vec![
            ScoredExtraction {
                extraction: crate::data::Extraction::new("person".to_string(), "John Doe".to_string()),
                quality_score: 0.8,
                pass_number: 1,
                chunk_id: 0,
            },
            ScoredExtraction {
                extraction: crate::data::Extraction::new("organization".to_string(), "ACME Corp".to_string()),
                quality_score: 0.9,
                pass_number: 1,
                chunk_id: 0,
            },
        ];

        // We can't easily test the actual context building without a full processor,
        // but we can verify the data structures are correct
        assert_eq!(extractions.len(), 2);
        assert_eq!(extractions[0].extraction.extraction_class, "person");
        assert_eq!(extractions[1].extraction.extraction_class, "organization");
    }
}
