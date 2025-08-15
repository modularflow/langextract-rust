//! # LangExtract
//!
//! A Rust library for extracting structured and grounded information from text using LLMs.
//!
//! This library provides a clean, async API for working with various language model providers
//! to extract structured data from unstructured text.
//!
//! ## Features
//!
//! - Support for multiple LLM providers (Gemini, OpenAI, Ollama)
//! - Async/await API for concurrent processing
//! - Schema-driven extraction with validation
//! - Text chunking and tokenization
//! - Flexible output formats (JSON, YAML)
//! - Built-in visualization and progress tracking
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use langextract::{extract, ExampleData, Extraction, FormatType};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let examples = vec![
//!         ExampleData {
//!             text: "John Doe is 30 years old".to_string(),
//!             extractions: vec![
//!                 Extraction::new("person".to_string(), "John Doe".to_string()),
//!                 Extraction::new("age".to_string(), "30".to_string()),
//!             ],
//!         }
//!     ];
//!
//!     let result = extract(
//!         "Alice Smith is 25 years old and works as a doctor",
//!         Some("Extract person names and ages from the text"),
//!         &examples,
//!         Default::default(),
//!     ).await?;
//!
//!     println!("{:?}", result);
//!     Ok(())
//! }
//! ```

// Core modules
pub mod data;
pub mod exceptions;
pub mod schema;

// Processing modules
pub mod alignment;
pub mod annotation;
pub mod chunking;
pub mod inference;
pub mod multipass;
pub mod tokenizer;

// Provider modules
pub mod providers;
pub mod factory;

// Utility modules
pub mod io;
pub mod progress;
pub mod prompting;
pub mod resolver;
pub mod visualization;

// Re-export key types for convenience
pub use data::{
    AlignmentStatus, AnnotatedDocument, CharInterval, Document, ExampleData, Extraction,
    FormatType,
};
pub use exceptions::{LangExtractError, LangExtractResult};
pub use inference::{BaseLanguageModel, ScoredOutput};
pub use providers::{ProviderConfig, ProviderType, UniversalProvider};
pub use resolver::{ValidationConfig, ValidationResult, ValidationError, ValidationWarning, CoercionSummary, CoercionDetail, CoercionTargetType};
pub use visualization::{ExportFormat, ExportConfig, export_document};

use std::collections::HashMap;

/// Configuration for the extract function
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// The model ID to use (e.g., "gemini-2.5-flash", "gpt-4o")
    pub model_id: String,
    /// API key for the language model service
    pub api_key: Option<String>,
    /// Output format type
    pub format_type: FormatType,
    /// Maximum characters per chunk for processing
    pub max_char_buffer: usize,
    /// Sampling temperature (0.0 to 1.0)
    pub temperature: f32,
    /// Whether to wrap output in code fences
    pub fence_output: Option<bool>,
    /// Whether to use schema constraints
    pub use_schema_constraints: bool,
    /// Batch size for processing chunks
    pub batch_length: usize,
    /// Maximum number of concurrent workers
    pub max_workers: usize,
    /// Additional context for the prompt
    pub additional_context: Option<String>,
    /// Custom resolver parameters
    pub resolver_params: HashMap<String, serde_json::Value>,
    /// Custom language model parameters
    pub language_model_params: HashMap<String, serde_json::Value>,
    /// Enable debug mode
    pub debug: bool,
    /// Custom model URL for self-hosted models
    pub model_url: Option<String>,
    /// Number of extraction passes to improve recall
    pub extraction_passes: usize,
    /// Enable multi-pass extraction for improved recall
    pub enable_multipass: bool,
    /// Minimum extractions per chunk to avoid re-processing
    pub multipass_min_extractions: usize,
    /// Quality threshold for keeping extractions (0.0 to 1.0)
    pub multipass_quality_threshold: f32,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            model_id: "gemini-2.5-flash".to_string(),
            api_key: None,
            format_type: FormatType::Json,
            max_char_buffer: 1000,
            temperature: 0.5,
            fence_output: None,
            use_schema_constraints: true,
            batch_length: 10,
            max_workers: 10,
            additional_context: None,
            resolver_params: HashMap::new(),
            language_model_params: HashMap::new(),
            debug: true,
            model_url: None,
            extraction_passes: 1,
            enable_multipass: false,
            multipass_min_extractions: 1,
            multipass_quality_threshold: 0.3,
        }
    }
}

/// Main extraction function that mirrors the Python API
///
/// Extracts structured information from text using a language model based on
/// the provided examples and configuration.
///
/// # Arguments
///
/// * `text_or_documents` - The source text to extract information from, or a URL starting with http/https
/// * `prompt_description` - Instructions for what to extract from the text
/// * `examples` - Example data to guide the extraction
/// * `config` - Configuration parameters for the extraction
///
/// # Returns
///
/// An `AnnotatedDocument` with the extracted information
///
/// # Errors
///
/// Returns an error if:
/// * Examples are empty
/// * No API key is provided
/// * URL download fails
/// * Language model inference fails
pub async fn extract(
    text_or_documents: &str,
    prompt_description: Option<&str>,
    examples: &[ExampleData],
    config: ExtractConfig,
) -> LangExtractResult<AnnotatedDocument> {
    // Validate inputs
    if examples.is_empty() {
        return Err(LangExtractError::InvalidInput(
            "Examples are required for reliable extraction. Please provide at least one ExampleData object with sample extractions.".to_string()
        ));
    }

    if config.batch_length < config.max_workers {
        log::warn!(
            "batch_length ({}) < max_workers ({}). Only {} workers will be used. Set batch_length >= max_workers for optimal parallelization.",
            config.batch_length,
            config.max_workers,
            config.batch_length
        );
    }

    // Load environment variables
    dotenvy::dotenv().ok();

    // Handle URL input
    let text = if io::is_url(text_or_documents) {
        io::download_text_from_url(text_or_documents).await?
    } else {
        text_or_documents.to_string()
    };

    // Create prompt template
    let mut prompt_template = prompting::PromptTemplateStructured::new(prompt_description);
    prompt_template.examples.extend(examples.iter().cloned());

    // Create language model
    let language_model = factory::create_model(&config, Some(&prompt_template.examples)).await?;

    // Create resolver
    let resolver = resolver::Resolver::new(&config, language_model.requires_fence_output())?;

    // Create annotator
    let annotator = annotation::Annotator::new(
        language_model,
        prompt_template,
        config.format_type,
        resolver.fence_output(),
    );

    // Perform annotation - use multi-pass if enabled
    if config.enable_multipass && config.extraction_passes > 1 {
        // Use multi-pass extraction
        let multipass_config = multipass::MultiPassConfig {
            max_passes: config.extraction_passes,
            min_extractions_per_chunk: config.multipass_min_extractions,
            enable_targeted_reprocessing: true,
            enable_refinement_passes: true,
            quality_threshold: config.multipass_quality_threshold,
            max_reprocess_chunks: 10,
            temperature_decay: 0.9,
        };

        let processor = multipass::MultiPassProcessor::new(
            multipass_config,
            annotator,
            resolver,
        );

        let (result, _stats) = processor.extract_multipass(
            &text,
            config.additional_context.as_deref(),
            config.debug,
        ).await?;

        if config.debug {
            println!("🎯 Multi-pass extraction completed with {} total extractions", 
                result.extraction_count());
        }

        Ok(result)
    } else {
        // Use single-pass extraction
        annotator
            .annotate_text(
                &text,
                &resolver,
                config.max_char_buffer,
                config.batch_length,
                config.additional_context.as_deref(),
                config.debug,
                config.extraction_passes,
                config.max_workers,
            )
            .await
    }
}

/// Visualize function that mirrors the Python API
pub fn visualize(
    annotated_document: &AnnotatedDocument,
    show_char_intervals: bool,
) -> LangExtractResult<String> {
    visualization::visualize(annotated_document, show_char_intervals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_config_default() {
        let config = ExtractConfig::default();
        assert_eq!(config.model_id, "gemini-2.5-flash");
        assert_eq!(config.format_type, FormatType::Json);
        assert_eq!(config.max_char_buffer, 1000);
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_extraction_validation() {
        let examples: Vec<ExampleData> = vec![];
        let config = ExtractConfig::default();

        tokio_test::block_on(async {
            let result = extract("test text", None, &examples, config).await;
            assert!(result.is_err());
            match result.err().unwrap() {
                LangExtractError::InvalidInput(msg) => {
                    assert!(msg.contains("Examples are required"));
                }
                _ => panic!("Expected InvalidInput error"),
            }
        });
    }
}
