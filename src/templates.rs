//! Template engine and utilities for LangExtract.
//!
//! This module provides a unified template system that eliminates duplication
//! across different prompt templates and formats.

use crate::{data::{ExampleData, FormatType}, exceptions::{LangExtractError, LangExtractResult}};
use std::collections::HashMap;

/// Template error types
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("Missing required variable: {variable}")]
    MissingVariable { variable: String },
    #[error("Invalid template syntax: {message}")]
    InvalidSyntax { message: String },
    #[error("Variable substitution failed: {message}")]
    SubstitutionError { message: String },
}

impl From<TemplateError> for LangExtractError {
    fn from(err: TemplateError) -> Self {
        LangExtractError::InvalidInput(err.to_string())
    }
}

/// A simple but flexible template engine
#[derive(Debug, Clone)]
pub struct TemplateEngine {
    /// Variable delimiter start (default: "{")
    pub var_start: String,
    /// Variable delimiter end (default: "}")
    pub var_end: String,
    /// Whether to allow missing variables (replaces with empty string)
    pub allow_missing: bool,
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self {
            var_start: "{".to_string(),
            var_end: "}".to_string(),
            allow_missing: false,
        }
    }
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lenient template engine that allows missing variables
    pub fn lenient() -> Self {
        Self {
            allow_missing: true,
            ..Default::default()
        }
    }

    /// Render a template with variables
    pub fn render(&self, template: &str, variables: &HashMap<String, String>) -> LangExtractResult<String> {
        let mut result = template.to_string();
        let mut pos = 0;

        while pos < result.len() {
            if let Some(start) = result[pos..].find(&self.var_start) {
                let abs_start = pos + start;
                let search_from = abs_start + self.var_start.len();
                
                if let Some(end) = result[search_from..].find(&self.var_end) {
                    let abs_end = search_from + end;
                    let var_name = &result[abs_start + self.var_start.len()..abs_end];
                    
                    if let Some(value) = variables.get(var_name) {
                        result.replace_range(abs_start..abs_end + self.var_end.len(), value);
                        pos = abs_start + value.len();
                    } else if self.allow_missing {
                        result.replace_range(abs_start..abs_end + self.var_end.len(), "");
                        pos = abs_start;
                    } else {
                        return Err(TemplateError::MissingVariable {
                            variable: var_name.to_string(),
                        }.into());
                    }
                } else {
                    return Err(TemplateError::InvalidSyntax {
                        message: format!("Unclosed variable at position {}", abs_start),
                    }.into());
                }
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Extract all variable names from a template
    pub fn extract_variables(&self, template: &str) -> Vec<String> {
        let mut variables = Vec::new();
        let mut pos = 0;

        while pos < template.len() {
            if let Some(start) = template[pos..].find(&self.var_start) {
                let abs_start = pos + start;
                let search_from = abs_start + self.var_start.len();
                
                if let Some(end) = template[search_from..].find(&self.var_end) {
                    let abs_end = search_from + end;
                    let var_name = &template[abs_start + self.var_start.len()..abs_end];
                    
                    if !var_name.is_empty() && !variables.contains(&var_name.to_string()) {
                        variables.push(var_name.to_string());
                    }
                    pos = abs_end + self.var_end.len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        variables
    }

    /// Validate that all required variables are present
    pub fn validate(&self, template: &str, variables: &HashMap<String, String>) -> LangExtractResult<()> {
        if self.allow_missing {
            return Ok(());
        }

        let required = self.extract_variables(template);
        for var in required {
            if !variables.contains_key(&var) {
                return Err(TemplateError::MissingVariable { variable: var }.into());
            }
        }
        Ok(())
    }
}

/// Common template fragments for reuse
pub struct TemplateFragments;

impl TemplateFragments {
    /// Standard instruction prefix
    pub fn instruction_prefix() -> &'static str {
        "You are an expert information extraction assistant. "
    }

    /// JSON format instruction
    pub fn json_format_instruction() -> &'static str {
        "Respond with valid JSON that matches the structure shown in the examples."
    }

    /// YAML format instruction
    pub fn yaml_format_instruction() -> &'static str {
        "Respond with valid YAML that matches the structure shown in the examples."
    }

    /// Reasoning instruction for local models
    pub fn reasoning_instruction() -> &'static str {
        "\n\nThink step by step:\n1. Read the text carefully\n2. Identify the requested information\n3. Extract it in the exact format shown in examples"
    }

    /// Example section header
    pub fn examples_header() -> &'static str {
        "\n\nExamples:\n"
    }

    /// Input section header  
    pub fn input_header() -> &'static str {
        "\n\nNow extract information from this text:\n\nInput: "
    }

    /// Output section header
    pub fn output_header(format: FormatType) -> String {
        match format {
            FormatType::Json => "\n\nOutput (JSON format):".to_string(),
            FormatType::Yaml => "\n\nOutput (YAML format):".to_string(),
        }
    }
}

/// Example formatter that handles different output formats consistently
pub struct ExampleFormatter {
    format_type: FormatType,
    max_examples: Option<usize>,
}

impl ExampleFormatter {
    pub fn new(format_type: FormatType) -> Self {
        Self {
            format_type,
            max_examples: None,
        }
    }

    pub fn with_max_examples(mut self, max: usize) -> Self {
        self.max_examples = Some(max);
        self
    }

    /// Format examples for inclusion in prompts
    pub fn format_examples(&self, examples: &[ExampleData]) -> LangExtractResult<String> {
        if examples.is_empty() {
            return Ok(String::new());
        }

        let examples_to_use = if let Some(max) = self.max_examples {
            &examples[..examples.len().min(max)]
        } else {
            examples
        };

        let mut result = String::new();
        result.push_str(TemplateFragments::examples_header());

        for (i, example) in examples_to_use.iter().enumerate() {
            result.push_str(&format!("\nExample {}:\n", i + 1));
            result.push_str(&format!("Input: {}\n", example.text));
            result.push_str("Output: ");
            result.push_str(&self.format_single_example(example)?);
            result.push('\n');
        }

        Ok(result)
    }

    /// Format a single example in the specified format
    fn format_single_example(&self, example: &ExampleData) -> LangExtractResult<String> {
        match self.format_type {
            FormatType::Json => self.format_as_json(example),
            FormatType::Yaml => self.format_as_yaml(example),
        }
    }

    fn format_as_json(&self, example: &ExampleData) -> LangExtractResult<String> {
        let mut json_obj = serde_json::Map::new();
        
        for extraction in &example.extractions {
            json_obj.insert(
                extraction.extraction_class.clone(),
                serde_json::Value::String(extraction.extraction_text.clone()),
            );
        }

        serde_json::to_string_pretty(&json_obj)
            .map_err(|e| TemplateError::SubstitutionError {
                message: format!("Failed to format JSON: {}", e),
            }.into())
    }

    fn format_as_yaml(&self, example: &ExampleData) -> LangExtractResult<String> {
        let mut yaml_map = std::collections::BTreeMap::new();
        
        for extraction in &example.extractions {
            yaml_map.insert(
                extraction.extraction_class.clone(),
                extraction.extraction_text.clone(),
            );
        }

        serde_yaml::to_string(&yaml_map)
            .map_err(|e| TemplateError::SubstitutionError {
                message: format!("Failed to format YAML: {}", e),
            }.into())
    }
}

/// Template builder for creating common prompt templates
pub struct TemplateBuilder {
    instruction: String,
    format_instruction: String,
    reasoning: String,
    examples_section: String,
    context_section: String,
    input_section: String,
    _output_section: String,
    engine: TemplateEngine,
}

impl TemplateBuilder {
    pub fn new(format_type: FormatType) -> Self {
        let format_instruction = match format_type {
            FormatType::Json => TemplateFragments::json_format_instruction(),
            FormatType::Yaml => TemplateFragments::yaml_format_instruction(),
        };

        Self {
            instruction: TemplateFragments::instruction_prefix().to_string(),
            format_instruction: format_instruction.to_string(),
            reasoning: String::new(),
            examples_section: "{examples}".to_string(),
            context_section: "{additional_context}".to_string(),
            input_section: format!("{}{}{}", 
                TemplateFragments::input_header(),
                "{input_text}",
                TemplateFragments::output_header(format_type)
            ),
            _output_section: String::new(),
            engine: TemplateEngine::lenient(),
        }
    }

    pub fn with_instruction(mut self, instruction: &str) -> Self {
        self.instruction = instruction.to_string();
        self
    }

    pub fn with_reasoning(mut self, include: bool) -> Self {
        if include {
            self.reasoning = TemplateFragments::reasoning_instruction().to_string();
        } else {
            self.reasoning.clear();
        }
        self
    }

    pub fn with_custom_examples_section(mut self, section: &str) -> Self {
        self.examples_section = section.to_string();
        self
    }

    pub fn build(&self) -> String {
        format!(
            "{{task_description}}\n\n{}{}{}{}{}{}\n",
            self.instruction,
            self.format_instruction,
            self.context_section,
            self.examples_section,
            self.reasoning,
            self.input_section,
        )
    }

    pub fn build_with_variables(self, variables: HashMap<String, String>) -> LangExtractResult<String> {
        let template = self.build();
        self.engine.render(&template, &variables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Extraction;

    #[test]
    fn test_template_engine_basic() {
        let engine = TemplateEngine::new();
        let template = "Hello {name}, welcome to {place}!";
        
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "John".to_string());
        vars.insert("place".to_string(), "LangExtract".to_string());

        let result = engine.render(template, &vars).unwrap();
        assert_eq!(result, "Hello John, welcome to LangExtract!");
    }

    #[test]
    fn test_template_engine_missing_var() {
        let engine = TemplateEngine::new();
        let template = "Hello {name}, welcome to {place}!";
        
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "John".to_string());
        // Missing "place" variable

        let result = engine.render(template, &vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_engine_lenient() {
        let engine = TemplateEngine::lenient();
        let template = "Hello {name}, welcome to {place}!";
        
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "John".to_string());
        // Missing "place" variable

        let result = engine.render(template, &vars).unwrap();
        assert_eq!(result, "Hello John, welcome to !");
    }

    #[test]
    fn test_variable_extraction() {
        let engine = TemplateEngine::new();
        let template = "Hello {name}, welcome to {place}! Your ID is {id}.";
        
        let vars = engine.extract_variables(template);
        assert_eq!(vars, vec!["name", "place", "id"]);
    }

    #[test]
    fn test_example_formatter_json() {
        let formatter = ExampleFormatter::new(FormatType::Json);
        
        let example = ExampleData::new(
            "John Doe is 30 years old".to_string(),
            vec![
                Extraction::new("person".to_string(), "John Doe".to_string()),
                Extraction::new("age".to_string(), "30".to_string()),
            ],
        );

        let result = formatter.format_examples(&[example]).unwrap();
        assert!(result.contains("Examples:"));
        assert!(result.contains("John Doe"));
        assert!(result.contains("person"));
        assert!(result.contains("age"));
    }

    #[test]
    fn test_template_builder() {
        let template = TemplateBuilder::new(FormatType::Json)
            .with_reasoning(true)
            .build();

        assert!(template.contains("You are an expert"));
        assert!(template.contains("JSON"));
        assert!(template.contains("Think step by step"));
        assert!(template.contains("{task_description}"));
        assert!(template.contains("{examples}"));
        assert!(template.contains("{input_text}"));
    }

    #[test]
    fn test_template_builder_with_variables() {
        let mut vars = HashMap::new();
        vars.insert("task_description".to_string(), "Extract names".to_string());
        vars.insert("examples".to_string(), "Example: John -> person: John".to_string());
        vars.insert("input_text".to_string(), "Alice Smith".to_string());
        vars.insert("additional_context".to_string(), "".to_string());

        let result = TemplateBuilder::new(FormatType::Json)
            .build_with_variables(vars)
            .unwrap();

        assert!(result.contains("Extract names"));
        assert!(result.contains("Alice Smith"));
        assert!(result.contains("Example: John"));
    }
}
