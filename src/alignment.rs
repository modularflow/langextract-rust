//! Text alignment functionality for mapping extractions to source text positions.
//!
//! This module provides algorithms to align extracted text with their positions
//! in the original source text, supporting both exact and fuzzy matching.

use crate::{
    data::{AlignmentStatus, CharInterval, Extraction},
    exceptions::LangExtractResult,
};
use std::cmp::min;

/// Configuration for text alignment
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Enable fuzzy alignment when exact matching fails
    pub enable_fuzzy_alignment: bool,
    /// Minimum overlap ratio for fuzzy alignment (0.0 to 1.0)
    pub fuzzy_alignment_threshold: f32,
    /// Accept partial exact matches (MATCH_LESSER status)
    pub accept_match_lesser: bool,
    /// Case-sensitive matching
    pub case_sensitive: bool,
    /// Maximum search window size for fuzzy matching
    pub max_search_window: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: 0.4, // Lower threshold for better fuzzy matching
            accept_match_lesser: true,
            case_sensitive: false,
            max_search_window: 100,
        }
    }
}

/// Text aligner for mapping extractions to source text positions
pub struct TextAligner {
    config: AlignmentConfig,
}

impl TextAligner {
    /// Create a new text aligner with default configuration
    pub fn new() -> Self {
        Self {
            config: AlignmentConfig::default(),
        }
    }

    /// Create a new text aligner with custom configuration
    pub fn with_config(config: AlignmentConfig) -> Self {
        Self { config }
    }

    /// Align extractions with the source text.
    /// Pre-lowercases the source text once and reuses it across all extractions
    /// to avoid repeated O(n) allocations.
    pub fn align_extractions(
        &self,
        extractions: &mut [Extraction],
        source_text: &str,
        char_offset: usize,
    ) -> LangExtractResult<usize> {
        // Pre-lowercase the source text once for all extractions (issue 5.3 fix)
        let search_text = if self.config.case_sensitive {
            source_text.to_string()
        } else {
            source_text.to_lowercase()
        };

        // Pre-compute source words and word set for fuzzy matching reuse
        let source_words: Vec<&str> = search_text.split_whitespace().collect();

        let mut aligned_count = 0;
        for extraction in extractions.iter_mut() {
            if let Some(interval) = self.align_single_extraction_with_cache(
                extraction, &search_text, &source_words, char_offset,
            )? {
                extraction.char_interval = Some(interval);
                aligned_count += 1;
            }
        }

        Ok(aligned_count)
    }

    /// Align a single extraction using pre-computed lowercase source text and word list.
    fn align_single_extraction_with_cache(
        &self,
        extraction: &mut Extraction,
        search_text: &str,
        source_words: &[&str],
        char_offset: usize,
    ) -> LangExtractResult<Option<CharInterval>> {
        let extraction_text = if self.config.case_sensitive {
            extraction.extraction_text.clone()
        } else {
            extraction.extraction_text.to_lowercase()
        };

        // Try exact matching first
        if let Some((start, end, status)) = self.find_exact_match(&extraction_text, search_text) {
            extraction.alignment_status = Some(status);
            return Ok(Some(CharInterval::new(
                Some(start + char_offset),
                Some(end + char_offset),
            )));
        }

        // Try fuzzy matching if enabled (reuse pre-computed source_words)
        if self.config.enable_fuzzy_alignment {
            if let Some((start, end, status)) = self.find_fuzzy_match_with_words(&extraction_text, search_text, source_words) {
                extraction.alignment_status = Some(status);
                return Ok(Some(CharInterval::new(
                    Some(start + char_offset),
                    Some(end + char_offset),
                )));
            }
        }

        // No alignment found
        extraction.alignment_status = None;
        Ok(None)
    }

    /// Align a single extraction with the source text (public API, lowercases on each call)
    pub fn align_single_extraction(
        &self,
        extraction: &mut Extraction,
        source_text: &str,
        char_offset: usize,
    ) -> LangExtractResult<Option<CharInterval>> {
        let search_text = if self.config.case_sensitive {
            source_text.to_string()
        } else {
            source_text.to_lowercase()
        };
        let source_words: Vec<&str> = search_text.split_whitespace().collect();

        self.align_single_extraction_with_cache(extraction, &search_text, &source_words, char_offset)
    }

    /// Find exact matches in the source text
    fn find_exact_match(&self, extraction_text: &str, source_text: &str) -> Option<(usize, usize, AlignmentStatus)> {
        // Try to find the exact extraction text
        if let Some(start) = source_text.find(extraction_text) {
            let end = start + extraction_text.len();
            return Some((start, end, AlignmentStatus::MatchExact));
        }

        // Try to find the extraction text as a substring (MATCH_LESSER)
        if self.config.accept_match_lesser {
            // Look for words from the extraction text
            let extraction_words: Vec<&str> = extraction_text.split_whitespace().collect();
            if extraction_words.len() > 1 {
                // Try to find the first and last words
                if let (Some(first_word), Some(last_word)) = (extraction_words.first(), extraction_words.last()) {
                    if let Some(first_start) = source_text.find(first_word) {
                        if let Some(last_start) = source_text[first_start..].find(last_word) {
                            let last_absolute_start = first_start + last_start;
                            let last_end = last_absolute_start + last_word.len();
                            
                            // Check if this span is reasonable (not too long)
                            if last_end - first_start < extraction_text.len() * 2 {
                                return Some((first_start, last_end, AlignmentStatus::MatchLesser));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Find fuzzy matches using sliding window approach (allocates source words internally)
    #[allow(dead_code)]
    fn find_fuzzy_match(&self, extraction_text: &str, source_text: &str) -> Option<(usize, usize, AlignmentStatus)> {
        let source_words: Vec<&str> = source_text.split_whitespace().collect();
        self.find_fuzzy_match_with_words(extraction_text, source_text, &source_words)
    }

    /// Find fuzzy matches using pre-computed source word list.
    /// Avoids re-splitting and re-lowercasing the source text per extraction.
    fn find_fuzzy_match_with_words(&self, extraction_text: &str, source_text: &str, source_words: &[&str]) -> Option<(usize, usize, AlignmentStatus)> {
        let extraction_words: Vec<&str> = extraction_text.split_whitespace().collect();

        if extraction_words.is_empty() || source_words.is_empty() {
            return None;
        }

        let mut best_match: Option<(usize, usize, f32)> = None;
        
        // Try different window sizes, starting with a reasonable size
        let max_window = min(source_words.len(), self.config.max_search_window);
        let min_window = extraction_words.len();
        
        for window_size in min_window..=max_window {
            for start_idx in 0..=source_words.len().saturating_sub(window_size) {
                let end_idx = start_idx + window_size;
                let window = &source_words[start_idx..end_idx];

                // Words are already lowercased (from pre-computation), use direct comparison
                let similarity = self.calculate_word_similarity_direct(&extraction_words, window);
                
                if similarity >= self.config.fuzzy_alignment_threshold {
                    if let Some((_, _, current_best)) = best_match {
                        if similarity > current_best {
                            best_match = Some((start_idx, end_idx, similarity));
                        }
                    } else {
                        best_match = Some((start_idx, end_idx, similarity));
                    }
                }
            }
            
            // If we found a good match with a smaller window, prefer it
            if best_match.is_some() {
                break;
            }
        }

        // Convert word positions back to character positions
        if let Some((start_word_idx, end_word_idx, _)) = best_match {
            let char_start = if start_word_idx == 0 {
                0
            } else {
                source_words[..start_word_idx].join(" ").len() + 1
            };

            let char_end = if end_word_idx >= source_words.len() {
                source_text.len()
            } else {
                source_words[..end_word_idx].join(" ").len()
            };

            return Some((char_start, char_end, AlignmentStatus::MatchFuzzy));
        }

        None
    }

    /// Calculate similarity between two pre-lowercased word sequences.
    /// Uses direct comparison (no re-lowercasing) since input words are
    /// already normalized.
    fn calculate_word_similarity_direct(&self, words1: &[&str], words2: &[&str]) -> f32 {
        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Build HashSet for O(1) lookup instead of Vec::contains O(n)
        let word_set2: std::collections::HashSet<&str> = words2.iter().copied().collect();

        // Count how many words from extraction are found in the source window
        let found_count = words1.iter().filter(|w| word_set2.contains(**w)).count();

        // Calculate coverage: what percentage of extraction words are found
        found_count as f32 / words1.len() as f32
    }

    /// Calculate similarity between two word sequences using coverage-based similarity
    /// (legacy method that lowercases internally, kept for backward compatibility)
    fn calculate_word_similarity(&self, words1: &[&str], words2: &[&str]) -> f32 {
        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Convert to lowercase for case-insensitive comparison
        let normalized_words1: Vec<String> = words1.iter().map(|w| w.to_lowercase()).collect();
        let normalized_words2: Vec<String> = words2.iter().map(|w| w.to_lowercase()).collect();

        // Build HashSet for O(1) lookup instead of Vec::contains O(n)
        let word_set2: std::collections::HashSet<&str> = normalized_words2.iter().map(|s| s.as_str()).collect();

        // Count how many words from extraction are found in the source window
        let found_count = normalized_words1.iter().filter(|w| word_set2.contains(w.as_str())).count();

        // Calculate coverage: what percentage of extraction words are found
        found_count as f32 / normalized_words1.len() as f32
    }

    /// Align extractions for chunked text processing
    pub fn align_chunk_extractions(
        &self,
        extractions: &mut [Extraction],
        chunk_text: &str,
        chunk_char_offset: usize,
    ) -> LangExtractResult<usize> {
        self.align_extractions(extractions, chunk_text, chunk_char_offset)
    }

    /// Get alignment statistics
    pub fn get_alignment_stats(&self, extractions: &[Extraction]) -> AlignmentStats {
        let total = extractions.len();
        let mut exact = 0;
        let mut fuzzy = 0;
        let mut lesser = 0;
        let mut greater = 0;
        let mut unaligned = 0;

        for extraction in extractions {
            match extraction.alignment_status {
                Some(AlignmentStatus::MatchExact) => exact += 1,
                Some(AlignmentStatus::MatchFuzzy) => fuzzy += 1,
                Some(AlignmentStatus::MatchLesser) => lesser += 1,
                Some(AlignmentStatus::MatchGreater) => greater += 1,
                None => unaligned += 1,
            }
        }

        AlignmentStats {
            total,
            exact,
            fuzzy,
            lesser,
            greater,
            unaligned,
        }
    }
}

impl Default for TextAligner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about alignment results
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    pub total: usize,
    pub exact: usize,
    pub fuzzy: usize,
    pub lesser: usize,
    pub greater: usize,
    pub unaligned: usize,
}

impl AlignmentStats {
    /// Get the alignment success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            (self.total - self.unaligned) as f32 / self.total as f32
        }
    }

    /// Get the exact match rate (0.0 to 1.0)
    pub fn exact_match_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.exact as f32 / self.total as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_alignment() {
        let aligner = TextAligner::new();
        let mut extraction = Extraction::new("person".to_string(), "John Doe".to_string());
        let source_text = "Hello, John Doe is a software engineer.";

        let result = aligner.align_single_extraction(&mut extraction, source_text, 0).unwrap();

        assert!(result.is_some());
        let interval = result.unwrap();
        assert_eq!(interval.start_pos, Some(7));
        assert_eq!(interval.end_pos, Some(15));
        assert_eq!(extraction.alignment_status, Some(AlignmentStatus::MatchExact));
    }

    #[test]
    fn test_case_insensitive_alignment() {
        let aligner = TextAligner::new();
        let mut extraction = Extraction::new("person".to_string(), "JOHN DOE".to_string());
        let source_text = "Hello, john doe is a software engineer.";

        let result = aligner.align_single_extraction(&mut extraction, source_text, 0).unwrap();

        assert!(result.is_some());
        let interval = result.unwrap();
        assert_eq!(interval.start_pos, Some(7));
        assert_eq!(interval.end_pos, Some(15));
        assert_eq!(extraction.alignment_status, Some(AlignmentStatus::MatchExact));
    }

    #[test]
    fn test_fuzzy_alignment() {
        let aligner = TextAligner::new();
        let mut extraction = Extraction::new("person".to_string(), "John Smith".to_string());
        let source_text = "Hello, John is a software engineer named Smith.";

        let result = aligner.align_single_extraction(&mut extraction, source_text, 0).unwrap();

        assert!(result.is_some());
        assert_eq!(extraction.alignment_status, Some(AlignmentStatus::MatchFuzzy));
    }

    #[test]
    fn test_no_alignment() {
        let aligner = TextAligner::new();
        let mut extraction = Extraction::new("person".to_string(), "Jane Doe".to_string());
        let source_text = "Hello, John Smith is a software engineer.";

        let result = aligner.align_single_extraction(&mut extraction, source_text, 0).unwrap();

        assert!(result.is_none());
        assert_eq!(extraction.alignment_status, None);
    }

    #[test]
    fn test_chunk_offset() {
        let aligner = TextAligner::new();
        let mut extraction = Extraction::new("person".to_string(), "John Doe".to_string());
        let chunk_text = "John Doe is here.";
        let chunk_offset = 100;

        let result = aligner.align_single_extraction(&mut extraction, chunk_text, chunk_offset).unwrap();

        assert!(result.is_some());
        let interval = result.unwrap();
        assert_eq!(interval.start_pos, Some(100)); // 0 + 100
        assert_eq!(interval.end_pos, Some(108));   // 8 + 100
    }

    #[test]
    fn test_alignment_stats() {
        let aligner = TextAligner::new();
        let extractions = vec![
            Extraction {
                extraction_class: "test".to_string(),
                extraction_text: "test".to_string(),
                alignment_status: Some(AlignmentStatus::MatchExact),
                ..Default::default()
            },
            Extraction {
                extraction_class: "test".to_string(),
                extraction_text: "test".to_string(),
                alignment_status: Some(AlignmentStatus::MatchFuzzy),
                ..Default::default()
            },
            Extraction {
                extraction_class: "test".to_string(),
                extraction_text: "test".to_string(),
                alignment_status: None,
                ..Default::default()
            },
        ];

        let stats = aligner.get_alignment_stats(&extractions);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.exact, 1);
        assert_eq!(stats.fuzzy, 1);
        assert_eq!(stats.unaligned, 1);
        assert_eq!(stats.success_rate(), 2.0 / 3.0);
        assert_eq!(stats.exact_match_rate(), 1.0 / 3.0);
    }
}
