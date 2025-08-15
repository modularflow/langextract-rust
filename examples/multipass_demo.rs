use langextract_rust::{
    data::{ExampleData, Extraction, FormatType},
    extract, ExtractConfig,
    visualization::visualize,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    dotenvy::dotenv().ok();

    println!("🚀 Multi-Pass Extraction Demo");
    println!("=============================\n");

    // Complex text with many named entities that might be missed in a single pass
    let source_text = r#"
In the bustling heart of San Francisco, California, Dr. Sarah Chen, a renowned neuroscientist at Stanford University, 
was preparing for her groundbreaking presentation at the Global AI Conference 2024. Her research team, including 
Dr. Michael Rodriguez from MIT and Prof. Lisa Zhang from UC Berkeley, had been working on revolutionary neural 
network architectures for the past three years.

The conference, sponsored by TechCorp Inc., Google AI, and Microsoft Research, was being held at the 
Moscone Convention Center from March 15-18, 2024. Over 5,000 attendees from around the world were expected, 
including representatives from major tech companies like Apple, Meta, and OpenAI.

Dr. Chen's presentation, titled "Adaptive Neural Pathways: Bridging Biological and Artificial Intelligence," 
was scheduled for March 16th at 2:30 PM in Hall B. The research had been funded by a $2.3 million grant 
from the National Science Foundation (NSF) and had resulted in 15 peer-reviewed publications.

Key findings included the development of the ChenNet architecture, which achieved 94.7% accuracy on 
benchmark datasets, outperforming previous models by 12%. The team's work on synaptic plasticity 
simulation had also caught the attention of pharmaceutical companies like Pfizer and Johnson & Johnson, 
who were exploring applications in drug discovery.

Contact information for the research team:
- Dr. Sarah Chen: s.chen@stanford.edu, (650) 555-0123
- Dr. Michael Rodriguez: m.rodriguez@mit.edu, (617) 555-0456  
- Prof. Lisa Zhang: l.zhang@berkeley.edu, (510) 555-0789

The research lab is located at 450 Jane Stanford Way, Stanford, CA 94305, Building 160, Room 282.
"#.trim();

    println!("📖 Source Text Preview:");
    println!("{}", &source_text[..300]);
    println!("... (text continues for {} total characters)\n", source_text.len());

    // Create example data for comprehensive entity extraction
    let examples = vec![
        ExampleData::new(
            "Dr. John Smith from Harvard University presented his research on AI at the Tech Summit 2023 in Boston.".to_string(),
            vec![
                Extraction::new("person".to_string(), "Dr. John Smith".to_string()),
                Extraction::new("organization".to_string(), "Harvard University".to_string()),
                Extraction::new("event".to_string(), "Tech Summit 2023".to_string()),
                Extraction::new("location".to_string(), "Boston".to_string()),
                Extraction::new("research_topic".to_string(), "AI".to_string()),
            ]
        ),
        ExampleData::new(
            "Contact Prof. Jane Doe at j.doe@mit.edu or call (617) 555-1234. Her office is at 77 Massachusetts Ave, Room 102.".to_string(),
            vec![
                Extraction::new("person".to_string(), "Prof. Jane Doe".to_string()),
                Extraction::new("email".to_string(), "j.doe@mit.edu".to_string()),
                Extraction::new("phone".to_string(), "(617) 555-1234".to_string()),
                Extraction::new("address".to_string(), "77 Massachusetts Ave, Room 102".to_string()),
            ]
        ),
        ExampleData::new(
            "The conference received $1.5 million in funding from Google and Microsoft, with 3,000 attendees expected.".to_string(),
            vec![
                Extraction::new("funding_amount".to_string(), "$1.5 million".to_string()),
                Extraction::new("organization".to_string(), "Google".to_string()),
                Extraction::new("organization".to_string(), "Microsoft".to_string()),
                Extraction::new("number".to_string(), "3,000 attendees".to_string()),
            ]
        ),
    ];

    println!("🔧 Configurations:");
    
    // Test 1: Single-pass extraction (baseline)
    println!("\n1️⃣ Single-Pass Extraction (Baseline):");
    let single_pass_config = ExtractConfig {
        model_id: "mistral".to_string(),
        api_key: None,
        format_type: FormatType::Json,
        max_char_buffer: 1000, // Smaller buffer to force chunking
        temperature: 0.1,
        fence_output: Some(true),
        use_schema_constraints: false,
        batch_length: 1,
        max_workers: 1,
        additional_context: None,
        resolver_params: HashMap::new(),
        language_model_params: HashMap::new(),
        debug: true,
        model_url: Some("http://localhost:11434".to_string()),
        extraction_passes: 1,
        enable_multipass: false, // Single pass
        multipass_min_extractions: 1,
        multipass_quality_threshold: 0.3,
    };

    match extract(source_text, None, &examples, single_pass_config).await {
        Ok(result) => {
            println!("✅ Single-pass extraction completed!");
            println!("   Found {} extractions", result.extraction_count());
            
            if let Some(extractions) = &result.extractions {
                let mut by_class: HashMap<String, Vec<&Extraction>> = HashMap::new();
                for extraction in extractions {
                    by_class.entry(extraction.extraction_class.clone())
                        .or_default()
                        .push(extraction);
                }
                
                for (class, extractions) in by_class {
                    println!("   📋 {}: {} found", class, extractions.len());
                }
            }
        },
        Err(e) => {
            println!("❌ Single-pass extraction failed: {}", e);
        }
    }

    println!("\n{}", "=".repeat(60));

    // Test 2: Multi-pass extraction with enhanced settings
    println!("\n2️⃣ Multi-Pass Extraction (Enhanced):");
    let multipass_config = ExtractConfig {
        model_id: "mistral".to_string(),
        api_key: None,
        format_type: FormatType::Json,
        max_char_buffer: 1000, // Smaller buffer to force chunking
        temperature: 0.1,
        fence_output: Some(true),
        use_schema_constraints: false,
        batch_length: 1,
        max_workers: 1,
        additional_context: Some("Please look for people, organizations, locations, dates, contact information, financial amounts, and research topics. Be thorough and check for entities that might be easy to miss.".to_string()),
        resolver_params: HashMap::new(),
        language_model_params: HashMap::new(),
        debug: true,
        model_url: Some("http://localhost:11434".to_string()),
        extraction_passes: 3, // Multiple passes
        enable_multipass: true, // Enable multi-pass
        multipass_min_extractions: 2, // Re-process chunks with < 2 extractions
        multipass_quality_threshold: 0.2, // Lower threshold for more inclusion
    };

    match extract(source_text, None, &examples, multipass_config).await {
        Ok(result) => {
            println!("✅ Multi-pass extraction completed!");
            println!("   Found {} total extractions", result.extraction_count());
            
            if let Some(extractions) = &result.extractions {
                let mut by_class: HashMap<String, Vec<&Extraction>> = HashMap::new();
                for extraction in extractions {
                    by_class.entry(extraction.extraction_class.clone())
                        .or_default()
                        .push(extraction);
                }
                
                println!("   📊 Detailed breakdown:");
                for (class, extractions) in by_class {
                    println!("   📋 {}: {} found", class, extractions.len());
                    for extraction in extractions.iter().take(3) { // Show first 3
                        if let Some(interval) = &extraction.char_interval {
                            if let (Some(start), Some(end)) = (interval.start_pos, interval.end_pos) {
                                println!("      - \"{}\" ({}..{}, {:?})", 
                                    extraction.extraction_text, start, end, 
                                    extraction.alignment_status);
                            }
                        } else {
                            println!("      - \"{}\" (not aligned)", extraction.extraction_text);
                        }
                    }
                    if extractions.len() > 3 {
                        println!("      ... and {} more", extractions.len() - 3);
                    }
                }
            }

            println!("\n🎨 Text Visualization:");
            println!("=====================");
            visualize(&result, true);
        },
        Err(e) => {
            println!("❌ Multi-pass extraction failed: {}", e);
        }
    }

    println!("\n🎯 Summary:");
    println!("============");
    println!("Multi-pass extraction should find significantly more entities than single-pass,");
    println!("especially for complex documents with many entities that are easy to miss.");
    println!("The refinement passes help catch entities that were overlooked in earlier passes.");

    Ok(())
}
