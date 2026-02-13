//! LangExtract CLI
//! 
//! A command-line interface for the LangExtract library that provides
//! structured information extraction from text using Large Language Models.

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("This binary requires the 'cli' feature to be enabled.");
    eprintln!("Install with: cargo install langextract-rust --features cli");
    std::process::exit(1);
}

#[cfg(feature = "cli")]
mod cli {
    use clap::{Args, Parser, Subcommand, ValueEnum};

    use console::style;
    use indicatif::{ProgressBar, ProgressStyle};
    use langextract_rust::{
        extract, ExampleData, Extraction, ExtractConfig, FormatType,
        ProviderConfig, ProviderType, LangExtractError,
        visualization::{export_document, ExportConfig, ExportFormat},
    };

    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    /// CLI for LangExtract - Extract structured information from text using LLMs
    #[derive(Parser)]
    #[command(name = "lx-rs")]
    #[command(about = "Extract structured information from text using Large Language Models")]
    #[command(version, long_about = None)]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,

        /// Enable verbose output
        #[arg(short, long, global = true)]
        pub verbose: bool,

        /// Disable colored output
        #[arg(long, global = true)]
        pub no_color: bool,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// Extract information from text or files
        Extract(ExtractArgs),
        /// Initialize configuration templates
        Init(InitArgs),
        /// Test provider connectivity
        Test(TestArgs),
        /// List available models and providers
        Providers,
        /// Show example configurations
        Examples,
        /// Convert extractions between formats
        Convert(ConvertArgs),
        /// Execute a multi-step extraction pipeline
        Pipeline(PipelineArgs),
    }

    #[derive(Args)]
    pub struct ExtractArgs {
        /// Input text, file path, or URL to process
        #[arg(value_name = "INPUT")]
        pub input: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        pub output: Option<PathBuf>,

        /// Examples file (JSON/YAML format)
        #[arg(short, long)]
        pub examples: Option<PathBuf>,

        /// Prompt description for extraction
        #[arg(short, long)]
        pub prompt: Option<String>,

        /// Model to use (e.g., 'gpt-4o', 'mistral', 'gemini-2.5-flash')
        #[arg(short, long, default_value = "gemini-2.5-flash")]
        pub model: String,

        /// Provider type (required)
        #[arg(long, value_enum)]
        pub provider: ProviderType,

        /// API key (overrides environment variables)
        #[arg(long)]
        pub api_key: Option<String>,

        /// Model URL for custom/self-hosted models
        #[arg(long)]
        pub model_url: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value = "json")]
        pub format: OutputFormat,

        /// Export format for visualization
        #[arg(long, value_enum)]
        pub export: Option<ExportFormat>,

        /// Maximum characters per chunk
        #[arg(long, default_value = "8000")]
        pub max_chars: usize,

        /// Number of parallel workers
        #[arg(long, default_value = "6")]
        pub workers: usize,

        /// Batch size for processing
        #[arg(long, default_value = "4")]
        pub batch_size: usize,

        /// Sampling temperature (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        pub temperature: f32,

        /// Enable multi-pass extraction
        #[arg(long)]
        pub multipass: bool,

        /// Number of extraction passes (implies --multipass if > 1)
        #[arg(long, default_value = "2")]
        pub passes: usize,

        /// Show character intervals in output
        #[arg(long)]
        pub show_intervals: bool,

        /// Enable debug mode
        #[arg(long)]
        pub debug: bool,

        /// Suppress progress output (quiet mode)
        #[arg(short, long)]
        pub quiet: bool,

        /// Additional context for the prompt
        #[arg(long)]
        pub context: Option<String>,
    }

    #[derive(Args)]
    pub struct InitArgs {
        /// Directory to create configuration files
        #[arg(default_value = ".")]
        pub directory: PathBuf,

        /// Provider to configure (required)
        #[arg(short, long, value_enum)]
        pub provider: ProviderType,

        /// Force overwrite existing files
        #[arg(short, long)]
        pub force: bool,
    }

    #[derive(Args)]
    pub struct TestArgs {
        /// Provider to test (required)
        #[arg(short, long, value_enum)]
        pub provider: ProviderType,

        /// Model to test
        #[arg(short, long)]
        pub model: Option<String>,

        /// API key to use
        #[arg(long)]
        pub api_key: Option<String>,

        /// Model URL for custom providers
        #[arg(long)]
        pub model_url: Option<String>,
    }

    #[derive(Args)]
    pub struct ConvertArgs {
        /// Input file to convert
        pub input: PathBuf,

        /// Output file
        #[arg(short, long)]
        pub output: PathBuf,

        /// Target format
        #[arg(short, long, value_enum)]
        pub format: ExportFormat,

        /// Include character intervals
        #[arg(long)]
        pub show_intervals: bool,
    }

    #[derive(Args)]
    pub struct PipelineArgs {
        /// Pipeline configuration file (YAML)
        #[arg(short, long)]
        pub config: PathBuf,

        /// Input text, file path, or URL to process
        #[arg(value_name = "INPUT")]
        pub input: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        pub output: Option<PathBuf>,

        /// Create sample pipeline configuration
        #[arg(long)]
        pub create_sample: bool,

        /// Sample pipeline type (requirements, medical, etc.)
        #[arg(long, default_value = "requirements")]
        pub sample_type: String,

        /// Aggregate layered highlights (HTML export)
        #[arg(long, default_value_t = false)]
        pub aggregate_highlights: bool,

        /// Allow overlapping highlights in layered HTML
        #[arg(long, default_value_t = false)]
        pub allow_overlaps: bool,

        /// Expand nested JSON extraction_text into atomic highlights
        #[arg(long, default_value_t = false)]
        pub expand_nested_json: bool,

        /// Export layered HTML to file (with flags)
        #[arg(long)]
        pub export_html: Option<PathBuf>,

        /// Export flattened JSON to file (Rust-exported, replaces python flatten)
        #[arg(long)]
        pub export_flattened: Option<PathBuf>,
    }

    #[derive(ValueEnum, Clone, Debug)]
    pub enum OutputFormat {
        Json,
        Yaml,
        Text,
    }

    /// Initialize the CLI application
    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::parse();

        // Set up colored output
        if cli.no_color {
            colored::control::set_override(false);
        }

        // Initialize logging
        if cli.verbose {
            env_logger::Builder::from_default_env()
                .filter_level(log::LevelFilter::Debug)
                .init();
        } else {
            env_logger::Builder::from_default_env()
                .filter_level(log::LevelFilter::Warn)
                .init();
        }

        match cli.command {
            Commands::Extract(args) => extract_command(args, cli.verbose).await,
            Commands::Init(args) => init_command(args).await,
            Commands::Test(args) => test_command(args).await,
            Commands::Providers => providers_command().await,
            Commands::Examples => examples_command().await,
            Commands::Convert(args) => convert_command(args).await,
            Commands::Pipeline(args) => pipeline_command(args).await,
        }
    }

    async fn extract_command(args: ExtractArgs, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Load environment variables
        dotenvy::dotenv().ok();

        if !args.quiet {
            println!("{}", style("LangExtract - starting extraction").bold().cyan());
        }

        // Create progress bar (only in non-quiet mode)
        let pb = if !args.quiet {
            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .expect("Failed to set progress bar template"));
            pb.set_message("Loading configuration...");
            Some(pb)
        } else {
            None
        };

        // Load examples
        let examples = if let Some(examples_path) = &args.examples {
            if let Some(ref pb) = pb {
                pb.set_message("Loading examples...");
            }
            load_examples(examples_path)?
        } else {
            if !args.quiet {
                println!("{}", style("Warning: no examples provided, using default person extraction examples.").yellow());
            }
            get_default_examples()
        };

        if verbose {
            println!("Loaded {} examples", examples.len());
        }

        // Read input
        if let Some(ref pb) = pb {
            pb.set_message("Reading input...");
        }
        let text = if args.input.starts_with("http://") || args.input.starts_with("https://") {
            if !args.quiet {
                println!("Downloading from URL: {}", args.input);
            }
            langextract_rust::io::download_text_from_url(&args.input).await?
        } else if std::path::Path::new(&args.input).exists() {
            if !args.quiet {
                println!("Reading file: {}", args.input);
            }
            fs::read_to_string(&args.input)?
        } else {
            // Treat as literal text
            args.input.clone()
        };

        if verbose {
            println!("Input text length: {} characters", text.len());
        }

        // Configure extraction
        if let Some(ref pb) = pb {
            pb.set_message("Configuring extraction...");
        }
        let mut config = ExtractConfig {
            model_id: args.model.clone(),
            api_key: args.api_key.clone(),
            model_url: args.model_url.clone(),
            format_type: match args.format {
                OutputFormat::Json => FormatType::Json,
                OutputFormat::Yaml => FormatType::Yaml,
                OutputFormat::Text => FormatType::Json, // Default to JSON for processing
            },
            max_char_buffer: args.max_chars,
            max_workers: args.workers,
            batch_length: args.batch_size,
            temperature: args.temperature,
            enable_multipass: args.multipass || args.passes > 1,
            multipass_max_passes: args.passes,
            debug: args.debug || verbose,
            additional_context: args.context.clone(),
            ..Default::default()
        };

        // Configure progress handling based on CLI options
        if args.quiet {
            config = config.with_quiet_mode();
        } else if args.debug || verbose {
            config = config.with_verbose_progress();
        } else {
            config = config.with_console_progress();
        }

        // Set up provider configuration (required)
        let provider_config = match args.provider {
            ProviderType::OpenAI => {
                let mut config = ProviderConfig::openai(&args.model, args.api_key.clone());
                if let Some(model_url) = &args.model_url {
                    config = config.with_base_url(model_url.clone());
                }
                config
            },
            ProviderType::Ollama => ProviderConfig::ollama(&args.model, args.model_url.clone()),
            ProviderType::Custom => ProviderConfig::custom(
                &args.model_url.clone().unwrap_or_else(|| "http://localhost:8000".to_string()),
                &args.model
            ),
        };

        config.language_model_params.insert(
            "provider_config".to_string(),
            serde_json::to_value(&provider_config)?
        );

        if let Some(ref pb) = pb {
            pb.set_message("Performing extraction...");
        }

        // Perform extraction
        let result = match extract(
            &text,
            args.prompt.as_deref(),
            &examples,
            config,
        ).await {
            Ok(result) => {
                if let Some(ref pb) = pb {
                    pb.finish_with_message("Extraction completed");
                }
                result
            }
            Err(e) => {
                if let Some(ref pb) = pb {
                    pb.finish_with_message("Extraction failed");
                }
                return Err(handle_extraction_error(e));
            }
        };

        let elapsed = start_time.elapsed();
        if !args.quiet {
            println!("{} Found {} extractions in {:.2}s", 
                style(">>").green(), 
                result.extraction_count(), 
                elapsed.as_secs_f64()
            );
        }

        // Output results
        if let Some(output_path) = &args.output {
            write_output(&result, output_path, &args)?;
            if !args.quiet {
                println!("Results saved to: {}", output_path.display());
            }
        } else {
            print_output(&result, &args)?;
        }

        // Export visualization if requested
        if let Some(export_format) = args.export {
            let export_config = ExportConfig {
                format: export_format.clone(),
                title: Some("LangExtract Results".to_string()),
                highlight_extractions: true,
                show_char_intervals: args.show_intervals,
                include_statistics: true,
                ..Default::default()
            };

            let exported = export_document(&result, &export_config)?;
            let filename = format!("langextract_results.{}", 
                match export_format {
                    ExportFormat::Html => "html",
                    ExportFormat::Markdown => "md",
                    ExportFormat::Json => "json",
                    ExportFormat::Csv => "csv",
                    ExportFormat::Text => "txt",
                }
            );
            
            fs::write(&filename, exported)?;
            if !args.quiet {
                println!("Visualization exported to: {}", filename);
            }
        }

        Ok(())
    }

    async fn init_command(args: InitArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", style("Initializing LangExtract configuration...").bold().cyan());

        let config_dir = &args.directory;
        fs::create_dir_all(config_dir)?;

        // Create examples template
        let examples_path = config_dir.join("examples.json");
        if !examples_path.exists() || args.force {
            let examples = get_default_examples();
            let examples_json = serde_json::to_string_pretty(&examples)?;
            fs::write(&examples_path, examples_json)?;
            println!("Created examples template: {}", examples_path.display());
        }

        // Create .env template
        let env_path = config_dir.join(".env.example");
        if !env_path.exists() || args.force {
            let env_content = r#"# LangExtract Environment Configuration
# Copy this to .env and fill in your API keys

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Gemini Configuration  
GEMINI_API_KEY=your_gemini_api_key_here

# Custom provider configuration
CUSTOM_API_KEY=your_custom_api_key_here
CUSTOM_MODEL_URL=http://localhost:8000

# Ollama Configuration (no API key needed for local)
OLLAMA_BASE_URL=http://localhost:11434
"#;
            fs::write(&env_path, env_content)?;
            println!("Created environment template: {}", env_path.display());
        }

        // Create config file based on provider
        let provider = args.provider;
        let config_path = config_dir.join("langextract.yaml");
        if !config_path.exists() || args.force {
            let config_content = generate_config_template(provider);
            fs::write(&config_path, config_content)?;
            println!("Created configuration template: {}", config_path.display());
        }

        println!("\n{}", style("Configuration initialized successfully.").green().bold());
        println!("\nNext steps:");
        println!("1. Edit {} with your API keys", style(".env.example").cyan());
        println!("2. Customize {} with your extraction examples", style("examples.json").cyan());
        println!("3. Run: {} to test your setup", style("lx-rs test").yellow());

        Ok(())
    }

    async fn test_command(args: TestArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", style("Testing provider connectivity...").bold().cyan());

        dotenvy::dotenv().ok();

        let provider = args.provider;
        let model = args.model.unwrap_or_else(|| match provider {
            ProviderType::OpenAI => "gpt-3.5-turbo".to_string(),
            ProviderType::Ollama => "mistral".to_string(),
            ProviderType::Custom => "test-model".to_string(),
        });

        println!("Provider: {}", style(format!("{:?}", provider)).cyan());
        println!("Model: {}", style(&model).cyan());

        let mut config = ExtractConfig {
            model_id: model.clone(),
            api_key: args.api_key.clone(),
            model_url: args.model_url.clone(),
            debug: true,
            max_char_buffer: 1000,
            max_workers: 1,
            batch_length: 1,
            ..Default::default()
        };

        // Set up provider configuration (required)
        let provider_config = match args.provider {
            ProviderType::OpenAI => {
                let mut config = ProviderConfig::openai(&model, args.api_key.clone());
                if let Some(model_url) = &args.model_url {
                    config = config.with_base_url(model_url.clone());
                }
                config
            },
            ProviderType::Ollama => ProviderConfig::ollama(&model, args.model_url.clone()),
            ProviderType::Custom => ProviderConfig::custom(
                &args.model_url.clone().unwrap_or_else(|| "http://localhost:8000".to_string()),
                &model
            ),
        };

        config.language_model_params.insert(
            "provider_config".to_string(),
            serde_json::to_value(&provider_config).unwrap()
        );

        let examples = vec![
            ExampleData::new(
                "Test message".to_string(),
                vec![Extraction::new("test".to_string(), "test".to_string())],
            )
        ];

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Failed to set progress bar template"));
        pb.set_message("Testing connection...");

        match extract("This is a test message", Some("Extract test information"), &examples, config).await {
            Ok(_) => {
                pb.finish_with_message("Provider test successful");
                println!("{}", style("Connection to provider working correctly.").green().bold());
            }
            Err(e) => {
                pb.finish_with_message("Provider test failed");
                println!("{}", style("Provider test failed:").red().bold());
                println!("{}", e);
                
                match provider {
                    ProviderType::Ollama => {
                        println!("\n{}", style("Troubleshooting tips for Ollama:").yellow());
                        println!("1. Start Ollama: {}", style("ollama serve").cyan());
                        println!("2. Pull model: {}", style(&format!("ollama pull {}", model)).cyan());
                        println!("3. Check status: {}", style("curl http://localhost:11434/api/tags").cyan());
                    }
                    ProviderType::OpenAI => {
                        println!("\n{}", style("Troubleshooting tips for OpenAI:").yellow());
                        println!("1. Set API key: {}", style("export OPENAI_API_KEY=your_key").cyan());
                        println!("2. Check account: https://platform.openai.com/account/api-keys");
                    }
                    ProviderType::Custom => {
                        println!("\n{}", style("Troubleshooting tips for Custom provider:").yellow());
                        println!("1. Check URL: {}", style("--model-url http://your-server").cyan());
                        println!("2. Verify API compatibility with OpenAI format");
                    }
                }
                
                return Err(e.into());
            }
        }

        Ok(())
    }

    async fn providers_command() -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", style("Available Providers and Models").bold().cyan());
        println!();

        let providers = vec![
            ("OpenAI", vec!["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], "High accuracy, JSON mode support"),
            ("Ollama", vec!["mistral", "llama2", "qwen", "codellama"], "Local inference, privacy-focused"),
            ("Custom", vec!["any-model"], "OpenAI-compatible HTTP APIs"),
        ];

        for (provider, models, description) in providers {
            println!("{}", style(provider).bold().green());
            println!("  {}", description);
            println!("  Models: {}", models.join(", "));
            
            match provider {
                "OpenAI" => println!("  Requires: OPENAI_API_KEY environment variable"),
                "Ollama" => println!("  Requires: Local Ollama installation (ollama.ai)"),
                "Custom" => println!("  Requires: --model-url parameter"),
                _ => {}
            }
            println!();
        }

        println!("{}", style("Example usage:").bold().yellow());
        println!("  lx-rs extract 'Hello world' --provider openai --model gpt-4o");
        println!("  lx-rs extract 'Hello world' --provider ollama --model mistral");
        println!("  lx-rs extract 'Hello world' --provider custom --model-url http://localhost:8000");

        Ok(())
    }

    async fn examples_command() -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", style("Example Configurations").bold().cyan());
        println!();

        let examples = vec![
            ("Basic Person Extraction", r#"lx-rs extract "John Doe is 30 years old" --prompt "Extract names and ages""#),
            ("From File", r#"lx-rs extract document.txt --examples examples.json --output results.json"#),
            ("With Ollama", r#"lx-rs extract text.txt --provider ollama --model mistral"#),
            ("Multi-pass Extraction", r#"lx-rs extract large_doc.txt --multipass --passes 3 --workers 8"#),
            ("Export to HTML", r#"lx-rs extract article.txt --export html --show-intervals"#),
            ("Custom Provider", r#"lx-rs extract text.txt --provider custom --model-url http://localhost:8000"#),
        ];

        for (title, command) in examples {
            println!("{}", style(title).bold().green());
            println!("  {}", style(command).cyan());
            println!();
        }

        println!("{}", style("Configuration Examples:").bold().yellow());
        println!();
        println!("{}", style("# examples.json").green());
        println!("{}", serde_json::to_string_pretty(&get_default_examples())?);

        Ok(())
    }

    async fn convert_command(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", style("Converting extraction results...").bold().cyan());

        // Read input file
        let input_content = fs::read_to_string(&args.input)?;
        let result: langextract_rust::AnnotatedDocument = serde_json::from_str(&input_content)?;

        // Export to requested format
        let export_config = ExportConfig {
            format: args.format,
            title: Some("Converted Results".to_string()),
            highlight_extractions: true,
            show_char_intervals: args.show_intervals,
            include_statistics: true,
            ..Default::default()
        };

        let exported = export_document(&result, &export_config)?;
        fs::write(&args.output, exported)?;

        println!("Converted {} to {}", 
            args.input.display(), 
            args.output.display()
        );

        Ok(())
    }

    async fn pipeline_command(args: PipelineArgs) -> Result<(), Box<dyn std::error::Error>> {
        use langextract_rust::pipeline::{PipelineExecutor, utils};

        // Handle sample pipeline creation
        if args.create_sample {
            println!("{}", style("Creating sample pipeline configuration...").bold().cyan());

            let sample_config = match args.sample_type.as_str() {
                "requirements" => utils::create_requirements_pipeline(),
                _ => {
                    println!("Unknown sample type: {}. Using requirements as default.", args.sample_type);
                    utils::create_requirements_pipeline()
                }
            };

            utils::save_pipeline_to_file(&sample_config, &args.config)?;
            println!("Created sample pipeline configuration: {}", args.config.display());

            // Print usage example
            println!("\n{}", style("Usage Example:").bold().yellow());
            println!("  lx-rs pipeline --config {} \"The system shall process 100 transactions per second.\"",
                     args.config.display());

            return Ok(());
        }

        // Execute pipeline
        println!("{}", style("Executing pipeline...").bold().cyan());

        // Load pipeline configuration
        let executor = PipelineExecutor::from_yaml_file(&args.config)?;

        // Read input text
        let input_text = if args.input.starts_with("http") {
            // Handle URL input
            println!("Downloading content from URL...");
            langextract_rust::io::download_text_from_url(&args.input).await?
        } else if std::path::Path::new(&args.input).exists() {
            // Handle file input
            println!("Reading content from file...");
            std::fs::read_to_string(&args.input)?
        } else {
            // Handle direct text input
            args.input.clone()
        };

        // Execute the pipeline
        let result = executor.execute(&input_text).await?;

        // Output results
        let output_content = serde_json::to_string_pretty(&result.nested_output)?;

        if let Some(output_path) = &args.output {
            std::fs::write(output_path, &output_content)?;
            println!("Pipeline results saved to: {}", output_path.display());
        } else {
            println!("{}", output_content);
        }

        // Optional layered HTML export
        if let Some(html_path) = &args.export_html {
            use langextract_rust::visualization::{ExportConfig, ExportFormat, export_pipeline_html};
            let cfg = ExportConfig {
                format: ExportFormat::Html,
                aggregate_pipeline_highlights: args.aggregate_highlights,
                allow_overlapping_highlights: args.allow_overlaps,
                expand_nested_json: args.expand_nested_json,
                show_pipeline_legend: true,
                ..Default::default()
            };
            let html = export_pipeline_html(&result, &input_text, &cfg)?;
            std::fs::write(html_path, html)?;
            println!("Layered HTML exported to: {}", html_path.display());
        }

        // Optional flattened JSON export (Rust-generated)
        if let Some(json_path) = &args.export_flattened {
            use langextract_rust::visualization::export_pipeline_flattened_json;
            let flat = export_pipeline_flattened_json(&result, &input_text, args.expand_nested_json)?;
            std::fs::write(json_path, flat)?;
            println!("Flattened JSON exported to: {}", json_path.display());
        }

        // Print summary
        println!("\n{}", style("Pipeline Summary:").bold().green());
        println!("  Total processing time: {}ms", result.total_time_ms);
        println!("  Steps executed: {}", result.step_results.len());
        println!("  Successful steps: {}", result.step_results.iter().filter(|s| s.success).count());

        // Show step details
        for step_result in &result.step_results {
            let status = if step_result.success { "ok" } else { "FAILED" };
            println!("  {} {}: {} extractions in {}ms",
                     status,
                     step_result.step_name,
                     step_result.extractions.len(),
                     step_result.processing_time_ms);
        }

        Ok(())
    }

    // Helper functions

    fn load_examples(path: &PathBuf) -> Result<Vec<ExampleData>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        
        if path.extension().and_then(|s| s.to_str()) == Some("yaml") || 
           path.extension().and_then(|s| s.to_str()) == Some("yml") {
            Ok(serde_yaml::from_str(&content)?)
        } else {
            Ok(serde_json::from_str(&content)?)
        }
    }

    fn get_default_examples() -> Vec<ExampleData> {
        vec![
            ExampleData::new(
                "John Doe is 30 years old and works as a software engineer".to_string(),
                vec![
                    Extraction::new("person".to_string(), "John Doe".to_string()),
                    Extraction::new("age".to_string(), "30".to_string()),
                    Extraction::new("profession".to_string(), "software engineer".to_string()),
                ],
            ),
            ExampleData::new(
                "Dr. Sarah Johnson, 35, is a cardiologist at Mayo Clinic".to_string(),
                vec![
                    Extraction::new("person".to_string(), "Dr. Sarah Johnson".to_string()),
                    Extraction::new("age".to_string(), "35".to_string()),
                    Extraction::new("profession".to_string(), "cardiologist".to_string()),
                    Extraction::new("workplace".to_string(), "Mayo Clinic".to_string()),
                ],
            ),
        ]
    }

    fn generate_config_template(provider: ProviderType) -> String {
        match provider {
            ProviderType::OpenAI => r#"# OpenAI Configuration
model: "gpt-4o-mini"
provider: "openai"
temperature: 0.3
max_char_buffer: 8000
max_workers: 6
batch_length: 4
"#,
            ProviderType::Ollama => r#"# Ollama Configuration
model: "mistral"
provider: "ollama"
model_url: "http://localhost:11434"
temperature: 0.3
max_char_buffer: 8000
max_workers: 6
batch_length: 4
"#,
            ProviderType::Custom => r#"# Custom Provider Configuration
model: "your-model"
provider: "custom"
model_url: "http://localhost:8000"
temperature: 0.3
max_char_buffer: 8000
max_workers: 6
batch_length: 4
"#,
        }.to_string()
    }

    fn write_output(
        result: &langextract_rust::AnnotatedDocument, 
        path: &PathBuf, 
        args: &ExtractArgs
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = match args.format {
            OutputFormat::Json => serde_json::to_string_pretty(result)?,
            OutputFormat::Yaml => serde_yaml::to_string(result)?,
            OutputFormat::Text => {
                if let Some(extractions) = &result.extractions {
                    extractions.iter()
                        .map(|e| format!("{}: {}", e.extraction_class, e.extraction_text))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    "No extractions found".to_string()
                }
            }
        };

        fs::write(path, content)?;
        Ok(())
    }

    fn print_output(
        result: &langextract_rust::AnnotatedDocument, 
        args: &ExtractArgs
    ) -> Result<(), Box<dyn std::error::Error>> {
        match args.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(result)?);
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(result)?);
            }
            OutputFormat::Text => {
                if let Some(extractions) = &result.extractions {
                    for (i, extraction) in extractions.iter().enumerate() {
                        println!("{}. {}: {}", 
                            i + 1, 
                            style(&extraction.extraction_class).bold().green(),
                            style(&extraction.extraction_text).cyan()
                        );
                        
                        if args.show_intervals {
                            if let Some(interval) = &extraction.char_interval {
                                println!("   Position: {}:{}", 
                                    interval.start_pos.unwrap_or(0), 
                                    interval.end_pos.unwrap_or(0)
                                );
                            }
                        }
                    }
                } else {
                    println!("{}", style("No extractions found").yellow());
                }
            }
        }
        Ok(())
    }

    fn handle_extraction_error(error: LangExtractError) -> Box<dyn std::error::Error> {
        match &error {
            LangExtractError::NetworkError(_) => {
                eprintln!("{}", style("Network Error:").red().bold());
                eprintln!("   Check your internet connection and API endpoints");
            }
            LangExtractError::ConfigurationError(_) => {
                eprintln!("{}", style("Configuration Error:").red().bold());
                eprintln!("   Check your API keys and model settings");
            }
            LangExtractError::InferenceError { provider, .. } => {
                eprintln!("{}", style("Inference Error:").red().bold());
                if let Some(provider) = provider {
                    eprintln!("   Provider: {}", provider);
                }
                eprintln!("   Check model availability and API limits");
            }
            _ => {
                eprintln!("{}", style("Extraction Error:").red().bold());
            }
        }
        Box::new(error)
    }
}

#[cfg(feature = "cli")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cli::run().await
}
