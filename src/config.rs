//! Unified configuration system for LangExtract.
//!
//! This module provides a centralized configuration system that unifies all the
//! various configuration structures used throughout the library.

use crate::{
    data::FormatType,
    logging::ProgressHandler,
    providers::ProviderConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// The main unified configuration for LangExtract operations
#[derive(Clone, Serialize, Deserialize)]
pub struct LangExtractConfig {
    /// Core processing configuration
    pub processing: ProcessingConfig,
    /// Provider configuration
    pub provider: ProviderConfig,
    /// Validation and output processing
    pub validation: ValidationConfig,
    /// Text chunking configuration  
    pub chunking: ChunkingConfig,
    /// Alignment configuration
    pub alignment: AlignmentConfig,
    /// Multi-pass extraction configuration
    pub multipass: MultiPassConfig,
    /// Visualization and export configuration
    pub visualization: VisualizationConfig,
    /// Inference-specific parameters
    pub inference: InferenceConfig,
    /// Progress reporting configuration (not serialized)
    #[serde(skip)]
    pub progress: ProgressConfig,
}

/// Core processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Output format type
    pub format_type: FormatType,
    /// Maximum characters per chunk for processing
    pub max_char_buffer: usize,
    /// Batch size for processing chunks
    pub batch_length: usize,
    /// Maximum number of concurrent workers
    pub max_workers: usize,
    /// Additional context for the prompt
    pub additional_context: Option<String>,
    /// Enable debug mode
    pub debug: bool,
    /// Whether to wrap output in code fences
    pub fence_output: Option<bool>,
    /// Whether to use schema constraints
    pub use_schema_constraints: bool,
    /// Custom parameters for extensibility
    pub custom_params: HashMap<String, serde_json::Value>,
}

/// Configuration for validation and output processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Whether to enable schema validation
    pub enable_schema_validation: bool,
    /// Whether to enable type coercion (e.g., string "25" -> number 25)
    pub enable_type_coercion: bool,
    /// Whether to require all expected fields to be present
    pub require_all_fields: bool,
    /// Whether to save raw model outputs to files
    pub save_raw_outputs: bool,
    /// Directory to save raw outputs (defaults to "./raw_outputs")
    pub raw_outputs_dir: String,
    /// Quality threshold for extractions (0.0 to 1.0)
    pub quality_threshold: f32,
}

/// Configuration for text chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Chunking strategy to use
    pub strategy: ChunkingStrategy,
    /// Target chunk size in characters
    pub target_size: usize,
    /// Maximum chunk size in characters
    pub max_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Minimum chunk size in characters
    pub min_size: usize,
    /// Whether to preserve sentence boundaries
    pub preserve_sentences: bool,
    /// Whether to preserve paragraph boundaries
    pub preserve_paragraphs: bool,
}

/// Chunking strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkingStrategy {
    /// Token-based chunking (recommended)
    Token,
    /// Semantic chunking using embeddings
    Semantic,
    /// Sentence-based chunking
    Sentence,
    /// Paragraph-based chunking
    Paragraph,
    /// Fixed character-based chunking
    Fixed,
}

/// Configuration for text alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Configuration for multi-pass extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPassConfig {
    /// Enable multi-pass extraction for improved recall
    pub enable_multipass: bool,
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
}

/// Configuration for visualization and export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Default export format
    pub default_format: ExportFormat,
    /// Show character intervals in output
    pub show_char_intervals: bool,
    /// Include original text in export
    pub include_text: bool,
    /// Highlight extractions in text (for HTML/Markdown)
    pub highlight_extractions: bool,
    /// Include extraction statistics
    pub include_statistics: bool,
    /// Custom CSS for HTML export
    pub custom_css: Option<String>,
    /// Default title for exports
    pub default_title: Option<String>,
}

/// Export format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Text,
    Html,
    Markdown,
    Json,
    Csv,
}

/// Configuration for language model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Sampling temperature (0.0 to 1.0)
    pub temperature: f32,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Number of candidate outputs to generate
    pub num_candidates: usize,
    /// Stop sequences to halt generation
    pub stop_sequences: Vec<String>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Configuration for progress reporting
#[derive(Clone)]
pub struct ProgressConfig {
    /// Progress handler for reporting extraction progress
    pub handler: Option<Arc<dyn ProgressHandler>>,
    /// Whether to show progress messages
    pub show_progress: bool,
    /// Whether to show debug information
    pub show_debug: bool,
    /// Whether to use emoji and colors in output
    pub use_styling: bool,
}

impl Default for LangExtractConfig {
    fn default() -> Self {
        Self {
            processing: ProcessingConfig::default(),
            provider: ProviderConfig::ollama("mistral", None), // Safe default
            validation: ValidationConfig::default(),
            chunking: ChunkingConfig::default(),
            alignment: AlignmentConfig::default(),
            multipass: MultiPassConfig::default(),
            visualization: VisualizationConfig::default(),
            inference: InferenceConfig::default(),
            progress: ProgressConfig::default(),
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            format_type: FormatType::Json,
            max_char_buffer: 8000,
            batch_length: 4,
            max_workers: 6,
            additional_context: None,
            debug: false,
            fence_output: None,
            use_schema_constraints: true,
            custom_params: HashMap::new(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_schema_validation: true,
            enable_type_coercion: true,
            require_all_fields: false,
            save_raw_outputs: true,
            raw_outputs_dir: "./raw_outputs".to_string(),
            quality_threshold: 0.0,
        }
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Token,
            target_size: 8000,
            max_size: 10000,
            overlap: 200,
            min_size: 500,
            preserve_sentences: true,
            preserve_paragraphs: true,
        }
    }
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: 0.4,
            accept_match_lesser: true,
            case_sensitive: false,
            max_search_window: 100,
        }
    }
}

impl Default for MultiPassConfig {
    fn default() -> Self {
        Self {
            enable_multipass: false,
            max_passes: 2,
            min_extractions_per_chunk: 1,
            enable_targeted_reprocessing: true,
            enable_refinement_passes: true,
            quality_threshold: 0.3,
            max_reprocess_chunks: 10,
            temperature_decay: 0.9,
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            default_format: ExportFormat::Text,
            show_char_intervals: false,
            include_text: true,
            highlight_extractions: true,
            include_statistics: true,
            custom_css: None,
            default_title: None,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            max_tokens: None,
            num_candidates: 1,
            stop_sequences: vec![],
            extra_params: HashMap::new(),
        }
    }
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            handler: None,
            show_progress: true,
            show_debug: false,
            use_styling: true,
        }
    }
}

// Builder pattern implementation for easier configuration
impl LangExtractConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider configuration
    pub fn with_provider(mut self, provider: ProviderConfig) -> Self {
        self.provider = provider;
        self
    }

    /// Set the processing configuration
    pub fn with_processing(mut self, processing: ProcessingConfig) -> Self {
        self.processing = processing;
        self
    }

    /// Set validation configuration
    pub fn with_validation(mut self, validation: ValidationConfig) -> Self {
        self.validation = validation;
        self
    }

    /// Set chunking configuration
    pub fn with_chunking(mut self, chunking: ChunkingConfig) -> Self {
        self.chunking = chunking;
        self
    }

    /// Set alignment configuration
    pub fn with_alignment(mut self, alignment: AlignmentConfig) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set multi-pass configuration
    pub fn with_multipass(mut self, multipass: MultiPassConfig) -> Self {
        self.multipass = multipass;
        self
    }

    /// Set visualization configuration
    pub fn with_visualization(mut self, visualization: VisualizationConfig) -> Self {
        self.visualization = visualization;
        self
    }

    /// Set inference configuration
    pub fn with_inference(mut self, inference: InferenceConfig) -> Self {
        self.inference = inference;
        self
    }

    /// Set progress configuration
    pub fn with_progress(mut self, progress: ProgressConfig) -> Self {
        self.progress = progress;
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.processing.debug = enabled;
        self.progress.show_debug = enabled;
        self
    }

    /// Set maximum characters per chunk
    pub fn with_max_char_buffer(mut self, size: usize) -> Self {
        self.processing.max_char_buffer = size;
        self.chunking.target_size = size;
        self
    }

    /// Set the number of workers
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.processing.max_workers = workers;
        self
    }

    /// Set temperature for inference
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.inference.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    /// Enable multi-pass extraction
    pub fn with_multipass_enabled(mut self, enabled: bool) -> Self {
        self.multipass.enable_multipass = enabled;
        self
    }

    /// Set progress handler
    pub fn with_progress_handler(mut self, handler: Arc<dyn ProgressHandler>) -> Self {
        self.progress.handler = Some(handler);
        self
    }

    /// Enable quiet mode (no progress output)
    pub fn with_quiet_mode(mut self) -> Self {
        self.progress.show_progress = false;
        self.progress.show_debug = false;
        self
    }

    /// Enable verbose mode (show all output)
    pub fn with_verbose_mode(mut self) -> Self {
        self.progress.show_progress = true;
        self.progress.show_debug = true;
        self
    }
}

// Specialized builder methods for common configurations
impl LangExtractConfig {
    /// Create a configuration optimized for OpenAI
    pub fn for_openai(model: &str, api_key: Option<String>) -> Self {
        Self::new()
            .with_provider(ProviderConfig::openai(model, api_key))
            .with_inference(InferenceConfig {
                temperature: 0.2,
                max_tokens: Some(2000),
                ..Default::default()
            })
    }

    /// Create a configuration optimized for Ollama
    pub fn for_ollama(model: &str, base_url: Option<String>) -> Self {
        Self::new()
            .with_provider(ProviderConfig::ollama(model, base_url))
            .with_inference(InferenceConfig {
                temperature: 0.3,
                max_tokens: Some(1500),
                ..Default::default()
            })
            .with_chunking(ChunkingConfig {
                target_size: 6000, // Smaller chunks for local models
                max_size: 8000,
                ..Default::default()
            })
    }

    /// Create a configuration for high-performance processing
    pub fn for_high_performance() -> Self {
        Self::new()
            .with_processing(ProcessingConfig {
                max_workers: 12,
                batch_length: 8,
                max_char_buffer: 10000,
                ..Default::default()
            })
            .with_multipass(MultiPassConfig {
                enable_multipass: true,
                max_passes: 3,
                ..Default::default()
            })
    }

    /// Create a configuration for memory-efficient processing
    pub fn for_memory_efficient() -> Self {
        Self::new()
            .with_processing(ProcessingConfig {
                max_workers: 4,
                batch_length: 2,
                max_char_buffer: 6000,
                ..Default::default()
            })
            .with_chunking(ChunkingConfig {
                target_size: 4000,
                max_size: 6000,
                overlap: 100,
                ..Default::default()
            })
    }
}

// Conversion traits for backward compatibility
impl From<LangExtractConfig> for crate::ExtractConfig {
    fn from(config: LangExtractConfig) -> Self {
        let provider_config_value = serde_json::to_value(&config.provider).unwrap_or_default();
        
        Self {
            model_id: config.provider.model.clone(),
            api_key: config.provider.api_key.clone(),
            format_type: config.processing.format_type,
            max_char_buffer: config.processing.max_char_buffer,
            temperature: config.inference.temperature,
            fence_output: config.processing.fence_output,
            use_schema_constraints: config.processing.use_schema_constraints,
            batch_length: config.processing.batch_length,
            max_workers: config.processing.max_workers,
            additional_context: config.processing.additional_context.clone(),
            resolver_params: HashMap::new(), // Legacy field
            language_model_params: {
                let mut params = HashMap::new();
                params.insert("provider_config".to_string(), provider_config_value);
                params
            },
            debug: config.processing.debug,
            model_url: Some(config.provider.base_url.clone()),
            enable_multipass: config.multipass.enable_multipass,
            multipass_max_passes: config.multipass.max_passes,
            multipass_min_extractions: config.multipass.min_extractions_per_chunk,
            multipass_quality_threshold: config.multipass.quality_threshold,
            progress_handler: config.progress.handler,
        }
    }
}

impl std::fmt::Debug for LangExtractConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangExtractConfig")
            .field("processing", &self.processing)
            .field("provider", &self.provider)
            .field("validation", &self.validation)
            .field("chunking", &self.chunking)
            .field("alignment", &self.alignment)
            .field("multipass", &self.multipass)
            .field("visualization", &self.visualization)
            .field("inference", &self.inference)
            .field("progress", &"<ProgressConfig>")
            .finish()
    }
}

impl std::fmt::Debug for ProgressConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressConfig")
            .field("handler", &"<ProgressHandler>")
            .field("show_progress", &self.show_progress)
            .field("show_debug", &self.show_debug)
            .field("use_styling", &self.use_styling)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LangExtractConfig::default();
        assert_eq!(config.processing.format_type, FormatType::Json);
        assert_eq!(config.processing.max_char_buffer, 8000);
        assert_eq!(config.chunking.strategy, ChunkingStrategy::Token);
    }

    #[test]
    fn test_builder_pattern() {
        let config = LangExtractConfig::new()
            .with_debug(true)
            .with_max_char_buffer(10000)
            .with_workers(8)
            .with_temperature(0.5);

        assert!(config.processing.debug);
        assert_eq!(config.processing.max_char_buffer, 10000);
        assert_eq!(config.processing.max_workers, 8);
        assert_eq!(config.inference.temperature, 0.5);
    }

    #[test]
    fn test_specialized_configs() {
        use crate::providers::ProviderType;
        
        let openai_config = LangExtractConfig::for_openai("gpt-4o", Some("test-key".to_string()));
        assert_eq!(openai_config.provider.provider_type, ProviderType::OpenAI);
        assert_eq!(openai_config.inference.temperature, 0.2);

        let ollama_config = LangExtractConfig::for_ollama("mistral", None);
        assert_eq!(ollama_config.provider.provider_type, ProviderType::Ollama);
        assert_eq!(ollama_config.chunking.target_size, 6000);

        let hp_config = LangExtractConfig::for_high_performance();
        assert_eq!(hp_config.processing.max_workers, 12);
        assert!(hp_config.multipass.enable_multipass);
    }

    #[test]
    fn test_backward_compatibility() {
        let new_config = LangExtractConfig::for_ollama("mistral", None)
            .with_debug(true)
            .with_temperature(0.4);

        let old_config: crate::ExtractConfig = new_config.into();
        assert_eq!(old_config.model_id, "mistral");
        assert!(old_config.debug);
        assert_eq!(old_config.temperature, 0.4);
    }

    #[test]
    fn test_serialization() {
        let config = LangExtractConfig::for_openai("gpt-4o", Some("test-key".to_string()));
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: LangExtractConfig = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.provider.model, deserialized.provider.model);
        assert_eq!(config.processing.format_type, deserialized.processing.format_type);
    }
}
