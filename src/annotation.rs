//! Text annotation functionality.

use crate::{
    alignment::TextAligner,
    chunking::{ChunkResult, ResultAggregator, TextChunk, TokenChunk, ChunkIterator},
    data::{AnnotatedDocument, Extraction, Document},
    exceptions::LangExtractResult,
    inference::BaseLanguageModel,
    logging::{report_progress, ProgressEvent},
    prompting::PromptTemplateStructured,
    resolver::Resolver,
    tokenizer::Tokenizer,
};
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::time::Instant;

/// Main annotator for processing text through language models
pub struct Annotator {
    language_model: Box<dyn BaseLanguageModel>,
    prompt_template: PromptTemplateStructured,
    /// Sampling temperature for LLM inference (from user config)
    temperature: f32,
    /// Maximum output tokens for LLM inference (from user config or estimated)
    max_output_tokens: usize,
    /// Cached expected fields derived from prompt_template examples
    expected_fields: Vec<String>,
}

impl Annotator {
    /// Create a new annotator
    pub fn new(
        language_model: Box<dyn BaseLanguageModel>,
        prompt_template: PromptTemplateStructured,
    ) -> Self {
        // Pre-compute expected fields from examples (fixes issue 6.8)
        let expected_fields: Vec<String> = prompt_template.examples
            .iter()
            .flat_map(|example| example.extractions.iter())
            .map(|extraction| extraction.extraction_class.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Estimate max_output_tokens from number of extraction classes
        let estimated_max_tokens = std::cmp::max(expected_fields.len() * 200, 500);

        Self {
            language_model,
            prompt_template,
            temperature: 0.5,
            max_output_tokens: estimated_max_tokens,
            expected_fields,
        }
    }

    /// Create a new annotator with explicit inference parameters from user config
    pub fn with_config(
        language_model: Box<dyn BaseLanguageModel>,
        prompt_template: PromptTemplateStructured,
        temperature: f32,
        max_output_tokens: Option<usize>,
    ) -> Self {
        // Pre-compute expected fields from examples (fixes issue 6.8)
        let expected_fields: Vec<String> = prompt_template.examples
            .iter()
            .flat_map(|example| example.extractions.iter())
            .map(|extraction| extraction.extraction_class.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Use provided max_output_tokens or estimate from extraction classes
        let computed_max_tokens = max_output_tokens
            .unwrap_or_else(|| std::cmp::max(expected_fields.len() * 200, 500));

        Self {
            language_model,
            prompt_template,
            temperature,
            max_output_tokens: computed_max_tokens,
            expected_fields,
        }
    }

    /// Annotate text and return annotated document
    #[tracing::instrument(skip_all, fields(text_len = text.len(), max_char_buffer, max_workers))]
    pub async fn annotate_text(
        &self,
        text: &str,
        resolver: &Resolver,
        max_char_buffer: usize,
        batch_length: usize,
        additional_context: Option<&str>,
        debug: bool,
        max_workers: usize,
    ) -> LangExtractResult<AnnotatedDocument> {
        // Check if we need to chunk the text
        if text.len() <= max_char_buffer {
            // Text is small enough, process directly
            return self.process_single_text(text, resolver, additional_context, debug).await;
        }

        // Text is too large, use token-based chunking
        if debug {
            report_progress(ProgressEvent::Debug {
                operation: "chunking".to_string(),
                details: format!("Text length ({} chars) exceeds buffer limit ({} chars), using token-based chunking", 
                    text.len(), max_char_buffer),
            });
        }

        self.process_token_chunked_text(
            text,
            resolver,
            max_char_buffer,
            batch_length,
            additional_context,
            debug,
            max_workers,
        ).await
    }

    /// Process text that fits within the buffer limit
    #[tracing::instrument(skip_all, fields(text_len = text.len()))]
    async fn process_single_text(
        &self,
        text: &str,
        resolver: &Resolver,
        additional_context: Option<&str>,
        debug: bool,
    ) -> LangExtractResult<AnnotatedDocument> {
        // Build the prompt
        let prompt = self.build_prompt(text, additional_context)?;
        
        // Report processing started
        report_progress(ProgressEvent::ProcessingStarted {
            text_length: text.len(),
            model: self.language_model.model_id().to_string(),
            provider: self.language_model.provider_name().to_string(),
        });
        
        if debug {
            let prompt_preview = if prompt.len() > 200 {
                format!("{}... (truncated, total length: {} chars)", 
                    &prompt.chars().take(200).collect::<String>(), prompt.len())
            } else {
                prompt.clone()
            };
            report_progress(ProgressEvent::Debug {
                operation: "model_call".to_string(),
                details: format!("Model: {}, Provider: {}, Prompt: {}", 
                    self.language_model.model_id(),
                    self.language_model.provider_name(),
                    prompt_preview),
            });
        }

        // Create inference parameters from config (not hardcoded)
        let mut kwargs = HashMap::new();
        kwargs.insert("temperature".to_string(), serde_json::json!(self.temperature));
        kwargs.insert("max_completion_tokens".to_string(), serde_json::json!(self.max_output_tokens));

        // Call the language model
        let results = self.language_model.infer(&[prompt], &kwargs).await?;
        
        report_progress(ProgressEvent::ModelResponse {
            success: true,
            output_length: results.first()
                .and_then(|batch| batch.first())
                .map(|output| output.text().len()),
        });
        
        if debug {
            report_progress(ProgressEvent::Debug {
                operation: "model_response".to_string(),
                details: format!("Received {} batches from language model", results.len()),
            });
        }

        // Extract the response
        let mut annotated_doc = AnnotatedDocument::with_extractions(Vec::new(), text.to_string());
        
        if let Some(batch) = results.first() {
            if let Some(output) = batch.first() {
                let response_text = output.text();
                
                if debug {
                    report_progress(ProgressEvent::Debug {
                        operation: "model_response".to_string(),
                        details: format!("Raw response from model: {}", response_text),
                    });
                }

                // Use cached expected fields (computed once at Annotator creation)
                let expected_fields = &self.expected_fields;

                // Use new validation system with raw data preservation
                report_progress(ProgressEvent::ValidationStarted {
                    raw_output_length: response_text.len(),
                });

                match resolver.validate_and_parse(response_text, &expected_fields) {
                    Ok((mut extractions, validation_result)) => {
                        // Report validation results
                        report_progress(ProgressEvent::ValidationCompleted {
                            extractions_found: extractions.len(),
                            aligned_count: 0, // Will be updated after alignment
                            errors: validation_result.errors.len(),
                            warnings: validation_result.warnings.len(),
                        });

                        if debug {
                            if let Some(raw_file) = &validation_result.raw_output_file {
                                report_progress(ProgressEvent::Debug {
                                    operation: "validation".to_string(),
                                    details: format!("Raw output saved to: {}", raw_file),
                                });
                            }

                            for error in &validation_result.errors {
                                report_progress(ProgressEvent::Debug {
                                    operation: "validation".to_string(),
                                    details: format!("Validation error: {}", error.message),
                                });
                            }
                            for warning in &validation_result.warnings {
                                report_progress(ProgressEvent::Debug {
                                    operation: "validation".to_string(),
                                    details: format!("Validation warning: {}", warning.message),
                                });
                            }
                        }

                        // Align extractions with the source text
                        let aligner = TextAligner::new();
                        let aligned_count = aligner.align_extractions(&mut extractions, text, 0)
                            .unwrap_or(0);
                        
                        annotated_doc.extractions = Some(extractions);
                        
                        // Update validation result with actual aligned count
                        report_progress(ProgressEvent::ValidationCompleted {
                            extractions_found: annotated_doc.extraction_count(),
                            aligned_count,
                            errors: validation_result.errors.len(),
                            warnings: validation_result.warnings.len(),
                        });
                    }
                    Err(e) => {
                        if debug {
                            report_progress(ProgressEvent::Debug {
                                operation: "validation".to_string(),
                                details: format!("Failed to parse response as structured data: {}. Treating as unstructured response", e),
                            });
                        }
                        // If parsing fails, create a single extraction with the raw response
                        let extraction = Extraction::new("raw_response".to_string(), response_text.to_string());
                        annotated_doc.extractions = Some(vec![extraction]);
                    }
                }
            }
        }

        Ok(annotated_doc)
    }

    /// Process large text using chunking
    /// Process text with chunking using token-based strategy
    #[tracing::instrument(skip_all, fields(text_len = text.len(), max_char_buffer, max_workers))]
    async fn process_token_chunked_text(
        &self,
        text: &str,
        resolver: &Resolver,
        max_char_buffer: usize,
        batch_length: usize,
        additional_context: Option<&str>,
        debug: bool,
        max_workers: usize,
    ) -> LangExtractResult<AnnotatedDocument> {
        // Create tokenizer and tokenize the text
        let tokenizer = Tokenizer::new()?;
        let tokenized_text = tokenizer.tokenize(text)?;
        
        // Create document for chunking
        let document = Document {
            document_id: None,
            text: text.to_string(),
            additional_context: None,
        };
        
        // Create token-based chunk iterator
        let chunk_iter = ChunkIterator::new(&tokenized_text, &tokenizer, max_char_buffer, Some(&document))?;
        
        // Collect chunks from iterator
        let token_chunks: Result<Vec<TokenChunk>, _> = chunk_iter.collect();
        let token_chunks = token_chunks?;
        
        // Convert TokenChunks to TextChunks for compatibility with existing pipeline
        let mut text_chunks = Vec::new();
        for (i, token_chunk) in token_chunks.iter().enumerate() {
            let chunk_text = token_chunk.chunk_text(&tokenizer)?;
            let char_interval = token_chunk.char_interval(&tokenizer)?;
            let chunk_len = chunk_text.len();
            
            let text_chunk = TextChunk {
                id: i,
                text: chunk_text,
                char_offset: char_interval.start_pos.unwrap_or(0),
                char_length: chunk_len,
                document_id: None,
                has_overlap: false,
                overlap_info: None,
            };
            text_chunks.push(text_chunk);
        }
        
        // Report chunking started
        report_progress(ProgressEvent::ChunkingStarted {
            total_chars: text.len(),
            chunk_count: text_chunks.len(),
            strategy: "token-based".to_string(),
        });
        
        if debug {
            for (i, chunk) in text_chunks.iter().enumerate() {
                report_progress(ProgressEvent::Debug {
                    operation: "chunking".to_string(),
                    details: format!("Token Chunk {}: {} chars (offset: {})", i, chunk.char_length, chunk.char_offset),
                });
            }
        }

        // Process chunks in parallel batches
        self.process_text_chunks_in_batches(
            text_chunks,
            text,
            resolver,
            batch_length,
            additional_context,
            debug,
            max_workers,
        ).await
    }

    /// Common method to process text chunks with bounded streaming concurrency
    #[tracing::instrument(skip_all, fields(num_chunks = chunks.len(), max_workers))]
    async fn process_text_chunks_in_batches(
        &self,
        chunks: Vec<TextChunk>,
        original_text: &str,
        resolver: &Resolver,
        _batch_length: usize,
        additional_context: Option<&str>,
        debug: bool,
        max_workers: usize,
    ) -> LangExtractResult<AnnotatedDocument> {
        let total_chunks = chunks.len();

        report_progress(ProgressEvent::BatchProgress {
            batch_number: 1,
            total_batches: 1,
            chunks_processed: 0,
            total_chunks,
        });

        // Use buffer_unordered to process ALL chunks with bounded concurrency.
        // This replaces the previous batch-loop-with-take pattern that silently
        // dropped chunks when batch_length > max_workers.
        let chunk_results: Vec<LangExtractResult<ChunkResult>> = stream::iter(chunks.iter())
            .map(|chunk| self.process_chunk(chunk, resolver, additional_context, debug))
            .buffer_unordered(max_workers)
            .collect()
            .await;

        // Collect results, propagating any errors
        let mut collected_results = Vec::with_capacity(chunk_results.len());
        for (i, result) in chunk_results.into_iter().enumerate() {
            collected_results.push(result?);
            if debug && (i + 1) % max_workers == 0 {
                report_progress(ProgressEvent::Debug {
                    operation: "batch_processing".to_string(),
                    details: format!("Progress: {}/{} chunks processed", i + 1, total_chunks),
                });
            }
        }

        if debug {
            report_progress(ProgressEvent::Debug {
                operation: "batch_processing".to_string(),
                details: format!("All {}/{} chunks processed", collected_results.len(), total_chunks),
            });
        }

        // Aggregate results
        report_progress(ProgressEvent::AggregationStarted {
            chunk_count: chunks.len(),
        });
        let aggregator = ResultAggregator::new();
        let final_result = aggregator.aggregate_chunk_results(
            collected_results,
            original_text.to_string(),
            None,
        )?;

        report_progress(ProgressEvent::ProcessingCompleted {
            total_extractions: final_result.extraction_count(),
            processing_time_ms: 0, // We don't track time here, but it's required
        });
        
        if debug {
            report_progress(ProgressEvent::Debug {
                operation: "aggregation".to_string(),
                details: format!("Aggregated {} total extractions from {} chunks", 
                    final_result.extraction_count(), chunks.len()),
            });
        }

        Ok(final_result)
    }



    /// Process a single chunk
    #[tracing::instrument(skip_all, fields(chunk_id = chunk.id, chunk_len = chunk.text.len()))]
    async fn process_chunk(
        &self,
        chunk: &TextChunk,
        resolver: &Resolver,
        additional_context: Option<&str>,
        debug: bool,
    ) -> LangExtractResult<ChunkResult> {
        let start_time = Instant::now();

        match self.process_single_text(&chunk.text, resolver, additional_context, false).await {
            Ok(annotated_doc) => {
                let mut extractions = annotated_doc.extractions.unwrap_or_default();
                
                // Align extractions with the chunk text
                let aligner = TextAligner::new();
                let aligned_count = aligner.align_chunk_extractions(
                    &mut extractions,
                    &chunk.text,
                    chunk.char_offset,
                ).unwrap_or(0);
                
                if debug {
                    report_progress(ProgressEvent::Debug {
                        operation: "chunk_processing".to_string(),
                        details: format!("Chunk {} produced {} extractions ({} aligned)", 
                            chunk.id, extractions.len(), aligned_count),
                    });
                }

                Ok(ChunkResult::success(
                    chunk.id,
                    extractions,
                    chunk.char_offset,
                    chunk.char_length,
                ).with_processing_time(start_time.elapsed()))
            }
            Err(e) => {
                if debug {
                    report_progress(ProgressEvent::Debug {
                        operation: "chunk_processing".to_string(),
                        details: format!("Chunk {} failed: {}", chunk.id, e),
                    });
                }

                Ok(ChunkResult::failure(
                    chunk.id,
                    chunk.char_offset,
                    chunk.char_length,
                    e.to_string(),
                ).with_processing_time(start_time.elapsed()))
            }
        }
    }

    /// Build the prompt using the new template system
    fn build_prompt(&self, text: &str, additional_context: Option<&str>) -> LangExtractResult<String> {
        // Use the new template system for better prompt generation
        self.prompt_template.render(text, additional_context)
    }

}
