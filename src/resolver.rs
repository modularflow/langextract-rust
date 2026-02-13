//! Output resolution and parsing functionality.

use crate::{
    data::{FormatType, Extraction}, 
    exceptions::{LangExtractError, LangExtractResult}, 
    ExtractConfig
};
use serde_json::Value;
use std::fs;
use std::path::Path;
use uuid::Uuid;
use regex::Regex;

/// Configuration for validation behavior
#[derive(Debug, Clone)]
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

/// Results of validation process
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors encountered
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Corrected/coerced data (if any)
    pub corrected_data: Option<Value>,
    /// Path to saved raw output file
    pub raw_output_file: Option<String>,
    /// Type coercion details
    pub coercion_summary: Option<CoercionSummary>,
}

/// Validation error details
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Field path where error occurred
    pub field_path: Option<String>,
    /// Expected value or type
    pub expected: Option<String>,
    /// Actual value found
    pub actual: Option<String>,
}

/// Validation warning details
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Field path where warning occurred
    pub field_path: Option<String>,
}

/// Summary of type coercion operations performed
#[derive(Debug, Clone)]
pub struct CoercionSummary {
    /// Number of successful coercions
    pub successful_coercions: usize,
    /// Number of failed coercion attempts
    pub failed_coercions: usize,
    /// Details of each coercion attempt
    pub coercion_details: Vec<CoercionDetail>,
}

/// Details of a single coercion operation
#[derive(Debug, Clone)]
pub struct CoercionDetail {
    /// Field name being coerced
    pub field_name: String,
    /// Original value
    pub original_value: String,
    /// Coerced value (if successful)
    pub coerced_value: Option<Value>,
    /// Target type attempted
    pub target_type: CoercionTargetType,
    /// Whether coercion was successful
    pub success: bool,
    /// Error message if coercion failed
    pub error_message: Option<String>,
}

/// Types that can be coerced to
#[derive(Debug, Clone, PartialEq)]
pub enum CoercionTargetType {
    Integer,
    Float,
    Boolean,
    Currency,
    Percentage,
    Email,
    PhoneNumber,
    Date,
    Url,
}

/// Type coercion engine
pub struct TypeCoercer {
    enable_coercion: bool,
    // Pre-compiled regex patterns for performance
    integer_regex: Regex,
    float_regex: Regex,
    currency_regex: Regex,
    percentage_regex: Regex,
    email_regex: Regex,
    phone_regex: Regex,
    date_regex: Regex,
    url_regex: Regex,
}

impl TypeCoercer {
    /// Create a new type coercer
    pub fn new(enable_coercion: bool) -> Self {
        Self {
            enable_coercion,
            integer_regex: Regex::new(r"^[+-]?\d+$").unwrap(),
            float_regex: Regex::new(r"^[+-]?\d*\.?\d+([eE][+-]?\d+)?$").unwrap(),
            currency_regex: Regex::new(r"^\$+([\d,]+(?:\.\d{1,2})?)\s*(?:million|M|billion|B|thousand|K)?$|^([\d,]+(?:\.\d{1,2})?)\s*(?:million|M|billion|B|thousand|K)$").unwrap(),
            percentage_regex: Regex::new(r"^(\d*\.?\d+)%$").unwrap(),
            email_regex: Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap(),
            phone_regex: Regex::new(r"^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$").unwrap(),
            date_regex: Regex::new(r"^\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{4}|\w+ \d{1,2}, \d{4}$").unwrap(),
            url_regex: Regex::new(r"^https?://[^\s/$.?#].[^\s]*$").unwrap(),
        }
    }

    /// Attempt to coerce a string value to a more appropriate type
    pub fn coerce_value(&self, field_name: &str, value: &str) -> CoercionDetail {
        if !self.enable_coercion {
            return CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: None,
                target_type: CoercionTargetType::Integer, // Default
                success: false,
                error_message: Some("Type coercion disabled".to_string()),
            };
        }

        let trimmed_value = value.trim();

        // Try different coercion types in order of specificity
        
        // 1. Try percentage first (very specific pattern)
        if let Some(result) = self.try_coerce_percentage(field_name, trimmed_value) {
            return result;
        }

        // 2. Try email (very specific pattern)
        if let Some(result) = self.try_coerce_email(field_name, trimmed_value) {
            return result;
        }

        // 3. Try phone number (very specific pattern)
        if let Some(result) = self.try_coerce_phone(field_name, trimmed_value) {
            return result;
        }

        // 4. Try URL (very specific pattern)
        if let Some(result) = self.try_coerce_url(field_name, trimmed_value) {
            return result;
        }

        // 5. Try date (very specific pattern)
        if let Some(result) = self.try_coerce_date(field_name, trimmed_value) {
            return result;
        }

        // 6. Try currency (specific patterns with $ or units)
        if let Some(result) = self.try_coerce_currency(field_name, trimmed_value) {
            return result;
        }

        // 7. Try integer (before boolean so "1"/"0" become integers, not booleans)
        if let Some(result) = self.try_coerce_integer(field_name, trimmed_value) {
            return result;
        }

        // 8. Try float (more general numeric pattern)
        if let Some(result) = self.try_coerce_float(field_name, trimmed_value) {
            return result;
        }

        // 9. Try boolean (keyword strings only, after numeric coercion)
        if let Some(result) = self.try_coerce_boolean(field_name, trimmed_value) {
            return result;
        }

        // No coercion possible
        CoercionDetail {
            field_name: field_name.to_string(),
            original_value: value.to_string(),
            coerced_value: None,
            target_type: CoercionTargetType::Integer, // Default
            success: false,
            error_message: Some("No applicable coercion found".to_string()),
        }
    }

    fn try_coerce_integer(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if self.integer_regex.is_match(value) {
            match value.parse::<i64>() {
                Ok(num) => Some(CoercionDetail {
                    field_name: field_name.to_string(),
                    original_value: value.to_string(),
                    coerced_value: Some(Value::Number(serde_json::Number::from(num))),
                    target_type: CoercionTargetType::Integer,
                    success: true,
                    error_message: None,
                }),
                Err(e) => Some(CoercionDetail {
                    field_name: field_name.to_string(),
                    original_value: value.to_string(),
                    coerced_value: None,
                    target_type: CoercionTargetType::Integer,
                    success: false,
                    error_message: Some(format!("Integer parse error: {}", e)),
                }),
            }
        } else {
            None
        }
    }

    fn try_coerce_float(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if self.float_regex.is_match(value) {
            match value.parse::<f64>() {
                Ok(num) => Some(CoercionDetail {
                    field_name: field_name.to_string(),
                    original_value: value.to_string(),
                    coerced_value: Some(Value::Number(serde_json::Number::from_f64(num).unwrap_or_else(|| serde_json::Number::from(0)))),
                    target_type: CoercionTargetType::Float,
                    success: true,
                    error_message: None,
                }),
                Err(e) => Some(CoercionDetail {
                    field_name: field_name.to_string(),
                    original_value: value.to_string(),
                    coerced_value: None,
                    target_type: CoercionTargetType::Float,
                    success: false,
                    error_message: Some(format!("Float parse error: {}", e)),
                }),
            }
        } else {
            None
        }
    }

    fn try_coerce_boolean(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        let lower_value = value.to_lowercase();
        match lower_value.as_str() {
            "true" | "yes" | "y" | "on" | "enabled" => Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Bool(true)),
                target_type: CoercionTargetType::Boolean,
                success: true,
                error_message: None,
            }),
            "false" | "no" | "n" | "off" | "disabled" => Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Bool(false)),
                target_type: CoercionTargetType::Boolean,
                success: true,
                error_message: None,
            }),
            _ => None,
        }
    }

    fn try_coerce_currency(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if let Some(captures) = self.currency_regex.captures(value) {
            // Try first capture group (for $amounts), then second (for amounts with units)
            let amount_str = captures.get(1).or_else(|| captures.get(2))?;
            let amount_clean = amount_str.as_str().replace(",", "");
            if let Ok(mut amount) = amount_clean.parse::<f64>() {
                // Handle suffixes
                let lower_value = value.to_lowercase();
                if lower_value.contains("million") || lower_value.contains("m") {
                    amount *= 1_000_000.0;
                } else if lower_value.contains("billion") || lower_value.contains("b") {
                    amount *= 1_000_000_000.0;
                } else if lower_value.contains("thousand") || lower_value.contains("k") {
                    amount *= 1_000.0;
                }

                return Some(CoercionDetail {
                    field_name: field_name.to_string(),
                    original_value: value.to_string(),
                    coerced_value: Some(Value::Number(serde_json::Number::from_f64(amount).unwrap_or_else(|| serde_json::Number::from(0)))),
                    target_type: CoercionTargetType::Currency,
                    success: true,
                    error_message: None,
                });
            }
        }
        None
    }

    fn try_coerce_percentage(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if let Some(captures) = self.percentage_regex.captures(value) {
            if let Some(percent_str) = captures.get(1) {
                if let Ok(percent) = percent_str.as_str().parse::<f64>() {
                    return Some(CoercionDetail {
                        field_name: field_name.to_string(),
                        original_value: value.to_string(),
                        coerced_value: Some(Value::Number(serde_json::Number::from_f64(percent / 100.0).unwrap_or_else(|| serde_json::Number::from(0)))),
                        target_type: CoercionTargetType::Percentage,
                        success: true,
                        error_message: None,
                    });
                }
            }
        }
        None
    }

    fn try_coerce_email(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if self.email_regex.is_match(value) {
            Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Object({
                    let mut obj = serde_json::Map::new();
                    obj.insert("email".to_string(), Value::String(value.to_string()));
                    obj.insert("type".to_string(), Value::String("email".to_string()));
                    obj
                })),
                target_type: CoercionTargetType::Email,
                success: true,
                error_message: None,
            })
        } else {
            None
        }
    }

    fn try_coerce_phone(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if let Some(captures) = self.phone_regex.captures(value) {
            let area = captures.get(1)?.as_str();
            let exchange = captures.get(2)?.as_str();
            let number = captures.get(3)?.as_str();
            let formatted = format!("({}) {}-{}", area, exchange, number);

            Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Object({
                    let mut obj = serde_json::Map::new();
                    obj.insert("phone".to_string(), Value::String(formatted));
                    obj.insert("area_code".to_string(), Value::String(area.to_string()));
                    obj.insert("type".to_string(), Value::String("phone".to_string()));
                    obj
                })),
                target_type: CoercionTargetType::PhoneNumber,
                success: true,
                error_message: None,
            })
        } else {
            None
        }
    }

    fn try_coerce_date(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if self.date_regex.is_match(value) {
            Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Object({
                    let mut obj = serde_json::Map::new();
                    obj.insert("date".to_string(), Value::String(value.to_string()));
                    obj.insert("type".to_string(), Value::String("date".to_string()));
                    obj
                })),
                target_type: CoercionTargetType::Date,
                success: true,
                error_message: None,
            })
        } else {
            None
        }
    }

    fn try_coerce_url(&self, field_name: &str, value: &str) -> Option<CoercionDetail> {
        if self.url_regex.is_match(value) {
            Some(CoercionDetail {
                field_name: field_name.to_string(),
                original_value: value.to_string(),
                coerced_value: Some(Value::Object({
                    let mut obj = serde_json::Map::new();
                    obj.insert("url".to_string(), Value::String(value.to_string()));
                    obj.insert("type".to_string(), Value::String("url".to_string()));
                    obj
                })),
                target_type: CoercionTargetType::Url,
                success: true,
                error_message: None,
            })
        } else {
            None
        }
    }
}

/// Resolver for parsing language model outputs with validation
pub struct Resolver {
    /// Whether to expect fenced output
    fence_output: bool,
    /// Output format type
    format_type: FormatType,
    /// Validation configuration
    validation_config: ValidationConfig,
    /// Type coercion engine
    type_coercer: TypeCoercer,
}

impl Resolver {
    /// Create a new resolver
    pub fn new(config: &ExtractConfig, fence_output: bool) -> LangExtractResult<Self> {
        let validation_config = ValidationConfig {
            save_raw_outputs: config.debug, // Enable for debug mode by default
            ..Default::default()
        };

        // Create raw outputs directory if it doesn't exist
        if validation_config.save_raw_outputs {
            if let Err(e) = fs::create_dir_all(&validation_config.raw_outputs_dir) {
                log::warn!("Failed to create raw outputs directory: {}", e);
            }
        }

        let type_coercer = TypeCoercer::new(validation_config.enable_type_coercion);

        Ok(Self {
            fence_output,
            format_type: config.format_type,
            validation_config,
            type_coercer,
        })
    }

    /// Create a new resolver with custom validation config
    pub fn with_validation_config(
        config: &ExtractConfig, 
        fence_output: bool, 
        validation_config: ValidationConfig
    ) -> LangExtractResult<Self> {
        // Create raw outputs directory if it doesn't exist
        if validation_config.save_raw_outputs {
            if let Err(e) = fs::create_dir_all(&validation_config.raw_outputs_dir) {
                log::warn!("Failed to create raw outputs directory: {}", e);
            }
        }

        let type_coercer = TypeCoercer::new(validation_config.enable_type_coercion);

        Ok(Self {
            fence_output,
            format_type: config.format_type,
            validation_config,
            type_coercer,
        })
    }

    /// Get whether this resolver expects fenced output
    pub fn fence_output(&self) -> bool {
        self.fence_output
    }

    /// Save raw model output to a file for debugging/recovery
    pub fn save_raw_output(&self, raw_output: &str, metadata: Option<&str>) -> LangExtractResult<String> {
        if !self.validation_config.save_raw_outputs {
            return Err(LangExtractError::configuration("Raw output saving is disabled"));
        }

        // Ensure output directory exists
        let output_dir = Path::new(&self.validation_config.raw_outputs_dir);
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(|e| {
                LangExtractError::IoError(e)
            })?;
        }

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let unique_id = Uuid::new_v4().to_string()[..8].to_string();
        let filename = format!("raw_output_{}_{}.txt", timestamp, unique_id);
        let filepath = output_dir.join(&filename);

        let mut content = String::new();
        content.push_str(&format!("=== Raw Model Output ===\n"));
        content.push_str(&format!("Timestamp: {}\n", chrono::Utc::now().to_rfc3339()));
        if let Some(meta) = metadata {
            content.push_str(&format!("Metadata: {}\n", meta));
        }
        content.push_str(&format!("Format: {:?}\n", self.format_type));
        content.push_str(&format!("Content Length: {} chars\n", raw_output.len()));
        content.push_str(&format!("=== Output Content ===\n"));
        content.push_str(raw_output);
        content.push_str("\n=== End Output ===\n");

        fs::write(&filepath, content).map_err(|e| {
            LangExtractError::IoError(e)
        })?;

        let path_str = filepath.to_string_lossy().to_string();
        log::info!("Saved raw output to: {}", path_str);
        Ok(path_str)
    }

    /// Validate and parse model response with raw data preservation
    pub fn validate_and_parse(&self, raw_response: &str, expected_fields: &[String]) -> LangExtractResult<(Vec<Extraction>, ValidationResult)> {
        // Step 1: Always save raw output first if enabled
        let raw_file_path = if self.validation_config.save_raw_outputs {
            match self.save_raw_output(raw_response, Some("validation_parse")) {
                Ok(path) => {
                    log::debug!("Raw output saved to: {}", path);
                    Some(path)
                }
                Err(e) => {
                    log::warn!("Failed to save raw output: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 2: Attempt to parse the response with enhanced cleaning and repair
        log::debug!("Parsing model response...");
        let parse_result = self.parse_response_with_repair(raw_response, expected_fields);
        
        // Step 3: Validate the parsed data
        let mut validation_result = match &parse_result {
            Ok(extractions) => {
                log::debug!("Successfully parsed {} potential extractions", extractions.len());
                self.validate_extractions(extractions, expected_fields)
            }
            Err(parse_error) => {
                log::debug!("Failed to parse model response");
                // If parsing failed, create validation result with error
                ValidationResult {
                    is_valid: false,
                    errors: vec![ValidationError {
                        message: format!("Failed to parse response: {}", parse_error),
                        field_path: None,
                        expected: Some("Valid JSON structure".to_string()),
                        actual: Some("Unparseable content".to_string()),
                    }],
                    warnings: vec![],
                    corrected_data: None,
                    raw_output_file: raw_file_path.clone(), // Set the path here
                    coercion_summary: None,
                }
            }
        };

        // Step 4: Set the raw output file path in the validation result (update if not already set)
        if validation_result.raw_output_file.is_none() {
            validation_result.raw_output_file = raw_file_path.clone();
        }

        // Step 5: Return results - even if validation fails, we preserve the raw data
        match parse_result {
            Ok(extractions) => Ok((extractions, validation_result)),
            Err(e) => {
                // Improved error reporting
                match &validation_result.raw_output_file {
                    Some(path) => {
                        log::warn!("Parse failed but raw data saved to: {}", path);
                        log::warn!("Parse failed - check raw output at: {}", path);
                    }
                    None => {
                        log::warn!("Parse failed and no raw data was saved");
                        log::warn!("Parse failed and raw data could not be saved");
                    }
                }
                Err(e)
            }
        }
    }

    /// Clean and preprocess response before parsing
    fn clean_response(&self, response: &str) -> String {
        let mut cleaned = response.to_string();

        // Simple approach: remove code fence markers step by step
        // First, handle fenced blocks with language specifiers
        cleaned = cleaned.replace("```json", "");
        cleaned = cleaned.replace("```yaml", "");
        cleaned = cleaned.replace("```python", "");
        cleaned = cleaned.replace("```javascript", "");
        cleaned = cleaned.replace("```rust", "");
        cleaned = cleaned.replace("```", "");

        // Trim whitespace
        cleaned.trim().to_string()
    }

    /// Detect and repair malformed JSON where multiple extraction classes are crammed into a single extraction_text
    fn detect_and_repair_malformed_json(&self, json: &serde_json::Value, expected_fields: &[String]) -> Option<serde_json::Value> {
        // Check if this looks like malformed JSON where multiple classes are in a single extraction_text
        if let Some(obj) = json.as_object() {
            // Only proceed if we have exactly one key-value pair
            if obj.len() == 1 {
                if let Some((single_key, single_value)) = obj.iter().next() {
                    if let Some(extraction_text) = single_value.as_str() {
                        // Check if the extraction_text contains multiple expected field names
                        let mut found_fields = Vec::new();

                        for field in expected_fields {
                            // Look for patterns like "field_name: value" or "field_name - value" or "field_name=value"
                            let patterns = [
                                format!(r"(?i){}[:\-=]\s*([^\n\r,]*)", regex::escape(field)),
                                format!(r"(?i){}\s*[:\-=]\s*([^\n\r,]*)", regex::escape(field)),
                                format!(r"(?i){}[:\-=]\s*([^,\n\r]+)", regex::escape(field)),
                            ];

                            for pattern in &patterns {
                                if let Ok(regex) = Regex::new(pattern) {
                                    if regex.is_match(extraction_text) {
                                        found_fields.push(field.clone());
                                        break; // Found this field, move to next
                                    }
                                }
                            }
                        }

                        // If we found multiple expected fields in the single extraction_text,
                        // this is likely malformed and should be re-parsed
                        if found_fields.len() > 1 {
                            log::debug!("Detected malformed JSON: {} extraction classes found in single extraction_text '{}'",
                                     found_fields.len(), single_key);

                            // Try to extract individual field values
                            let mut repaired_obj = serde_json::Map::new();

                            for field in &found_fields {
                                let patterns = [
                                    format!(r"(?i){}[:\-=]\s*([^\n\r,]*)", regex::escape(field)),
                                    format!(r"(?i){}\s*[:\-=]\s*([^\n\r,]*)", regex::escape(field)),
                                ];

                                for pattern in &patterns {
                                    if let Ok(regex) = Regex::new(pattern) {
                                        if let Some(captures) = regex.captures(extraction_text) {
                                            if let Some(value_match) = captures.get(1) {
                                                let value = value_match.as_str().trim();
                                                if !value.is_empty() {
                                                    repaired_obj.insert(field.clone(), serde_json::Value::String(value.to_string()));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if !repaired_obj.is_empty() {
                                log::debug!("Successfully repaired malformed JSON, extracted {} fields", repaired_obj.len());
                                return Some(serde_json::Value::Object(repaired_obj));
                            }
                        }
                    }
                }
            }
        }

        None // No repair needed
    }

    /// Parse response with enhanced cleaning and repair capabilities
    fn parse_response_with_repair(&self, response: &str, expected_fields: &[String]) -> LangExtractResult<Vec<Extraction>> {
        // First, clean the response (remove code fences, etc.)
        let cleaned_response = self.clean_response(response);
        log::debug!("Cleaned response length: {} chars", cleaned_response.len());

        // Try to parse as JSON first
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&cleaned_response) {
            log::debug!("Parsed JSON successfully");

            // Check if the JSON needs repair (malformed case with multiple classes in single extraction_text)
            if let Some(repaired_json) = self.detect_and_repair_malformed_json(&json_value, expected_fields) {
                log::debug!("Applied JSON repair logic");
                return self.parse_json_response(&repaired_json);
            } else {
                return self.parse_json_response(&json_value);
            }
        }

        // If that fails, try to extract JSON from the response (in case it's wrapped)
        if let Some(json_start) = cleaned_response.find('{') {
            if let Some(json_end) = cleaned_response.rfind('}') {
                let json_str = &cleaned_response[json_start..=json_end];
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    log::debug!("Extracted and parsed JSON from wrapped content");

                    // Check if the extracted JSON needs repair
                    if let Some(repaired_json) = self.detect_and_repair_malformed_json(&json_value, expected_fields) {
                        log::debug!("Applied JSON repair logic to extracted content");
                        return self.parse_json_response(&repaired_json);
                    } else {
                        return self.parse_json_response(&json_value);
                    }
                }
            }
        }

        Err(LangExtractError::parsing(
            format!("Could not parse response as JSON after cleaning: {}", cleaned_response)
        ))
    }



    /// Parse JSON response into extractions
    fn parse_json_response(&self, json: &serde_json::Value) -> LangExtractResult<Vec<Extraction>> {
        let mut extractions = Vec::new();

        // Handle array at top level
        if let Some(array) = json.as_array() {
            for (index, item) in array.iter().enumerate() {
                extractions.extend(self.parse_single_item(item, Some(index))?);
            }
            return Ok(extractions);
        }

        // Handle object with data/results wrapper
        if let Some(obj) = json.as_object() {
            if let Some(data_array) = obj.get("data").and_then(|v| v.as_array()) {
                for (index, item) in data_array.iter().enumerate() {
                    extractions.extend(self.parse_single_item(item, Some(index))?);
                }
                return Ok(extractions);
            }
            if let Some(results_array) = obj.get("results").and_then(|v| v.as_array()) {
                for (index, item) in results_array.iter().enumerate() {
                    extractions.extend(self.parse_single_item(item, Some(index))?);
                }
                return Ok(extractions);
            }

            // Handle flat JSON structure like {"name": "John", "age": "25"}
            extractions.extend(self.parse_single_item(json, None)?);
        }

        Ok(extractions)
    }

    /// Parse a single item (object or primitive) into extractions
    fn parse_single_item(&self, item: &serde_json::Value, index: Option<usize>) -> LangExtractResult<Vec<Extraction>> {
        let mut extractions = Vec::new();

        match item {
            Value::Object(obj) => {
                for (key, value) in obj {
                    let extraction_text = match value {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        Value::Array(_) | Value::Object(_) => value.to_string(),
                        Value::Null => continue,
                    };

                    let mut extraction = Extraction::new(key.clone(), extraction_text);
                    if let Some(idx) = index {
                        extraction.group_index = Some(idx);
                    }
                    extractions.push(extraction);
                }
            }
            Value::String(s) => {
                let extraction_class = if let Some(idx) = index {
                    format!("item_{}", idx)
                } else {
                    "text".to_string()
                };
                extractions.push(Extraction::new(extraction_class, s.clone()));
            }
            _ => {
                return Err(LangExtractError::parsing(
                    format!("Unsupported item type: {:?}", item)
                ));
            }
        }

        Ok(extractions)
    }

    /// Validate extractions against expected schema
    fn validate_extractions(&self, extractions: &[Extraction], expected_fields: &[String]) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut is_valid = true;
        let mut coercion_details = Vec::new();

        // Check for required fields if enabled
        if self.validation_config.require_all_fields {
            let extraction_classes: std::collections::HashSet<_> = 
                extractions.iter().map(|e| &e.extraction_class).collect();
            
            for expected_field in expected_fields {
                if !extraction_classes.contains(expected_field) {
                    errors.push(ValidationError {
                        message: format!("Required field '{}' is missing", expected_field),
                        field_path: Some(expected_field.clone()),
                        expected: Some("Present".to_string()),
                        actual: Some("Missing".to_string()),
                    });
                    is_valid = false;
                }
            }
        }

        // Validate individual extractions and attempt type coercion
        for extraction in extractions {
            // Check for empty extraction text
            if extraction.extraction_text.trim().is_empty() {
                warnings.push(ValidationWarning {
                    message: format!("Empty extraction text for field '{}'", extraction.extraction_class),
                    field_path: Some(extraction.extraction_class.clone()),
                });
            }

            // Check extraction text length
            if extraction.extraction_text.len() > 1000 {
                warnings.push(ValidationWarning {
                    message: format!("Very long extraction text ({} chars) for field '{}'", 
                        extraction.extraction_text.len(), extraction.extraction_class),
                    field_path: Some(extraction.extraction_class.clone()),
                });
            }

            // Attempt type coercion if enabled
            if self.validation_config.enable_type_coercion {
                let coercion_result = self.type_coercer.coerce_value(
                    &extraction.extraction_class, 
                    &extraction.extraction_text
                );
                coercion_details.push(coercion_result);
            }
        }

        // Quality check - too few extractions might indicate poor model performance
        if extractions.len() < expected_fields.len() / 2 {
            warnings.push(ValidationWarning {
                message: format!("Low extraction count: found {} but expected around {}", 
                    extractions.len(), expected_fields.len()),
                field_path: None,
            });
        }

        // Build corrected data from coerced values
        let corrected_data = if !coercion_details.is_empty() && coercion_details.iter().any(|d| d.success) {
            let mut corrected_obj = serde_json::Map::new();
            
            for detail in &coercion_details {
                if detail.success {
                    if let Some(ref coerced_value) = detail.coerced_value {
                        corrected_obj.insert(detail.field_name.clone(), coerced_value.clone());
                    }
                } else {
                    // Keep original value as string if coercion failed
                    corrected_obj.insert(detail.field_name.clone(), Value::String(detail.original_value.clone()));
                }
            }
            
            Some(Value::Object(corrected_obj))
        } else {
            None
        };

        // Create coercion summary
        let coercion_summary = if !coercion_details.is_empty() {
            let successful_coercions = coercion_details.iter().filter(|d| d.success).count();
            let failed_coercions = coercion_details.len() - successful_coercions;
            
            Some(CoercionSummary {
                successful_coercions,
                failed_coercions,
                coercion_details,
            })
        } else {
            None
        };

        ValidationResult {
            is_valid: is_valid && errors.is_empty(),
            errors,
            warnings,
            corrected_data,
            raw_output_file: None, // Set by caller
            coercion_summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExtractConfig;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_config() -> ExtractConfig {
        ExtractConfig {
            debug: true,
            ..Default::default()
        }
    }

    fn create_test_resolver() -> Resolver {
        let config = create_test_config();
        Resolver::new(&config, true).unwrap()
    }

    fn create_test_resolver_with_temp_dir(temp_dir: &TempDir) -> Resolver {
        let config = create_test_config();
        let validation_config = ValidationConfig {
            save_raw_outputs: true,
            raw_outputs_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        Resolver::with_validation_config(&config, true, validation_config).unwrap()
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.enable_schema_validation);
        assert!(config.enable_type_coercion);
        assert!(!config.require_all_fields);
        assert!(config.save_raw_outputs);
        assert_eq!(config.raw_outputs_dir, "./raw_outputs");
        assert_eq!(config.quality_threshold, 0.0);
    }

    #[test]
    fn test_raw_output_saving() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);
        
        let test_output = r#"{"person": "John Doe", "age": "30"}"#;
        let result = resolver.save_raw_output(test_output, Some("test_metadata"));
        
        assert!(result.is_ok());
        let file_path = result.unwrap();
        assert!(std::path::Path::new(&file_path).exists());
        
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("Raw Model Output"));
        assert!(content.contains("test_metadata"));
        assert!(content.contains(test_output));
    }

    #[test]
    fn test_parse_valid_json() {
        let resolver = create_test_resolver();
        let json_response = r#"[{"person": "John Doe", "age": "30"}]"#;
        let expected_fields = vec!["person".to_string(), "age".to_string()];

        let result = resolver.parse_response_with_repair(json_response, &expected_fields);
        assert!(result.is_ok());
        
        let extractions = result.unwrap();
        assert_eq!(extractions.len(), 2);
        
        // Check that we have both fields (order may vary)
        let classes: std::collections::HashSet<_> = extractions.iter()
            .map(|e| e.extraction_class.as_str()).collect();
        assert!(classes.contains("person"));
        assert!(classes.contains("age"));
        
        // Check the values
        let person_extraction = extractions.iter().find(|e| e.extraction_class == "person").unwrap();
        assert_eq!(person_extraction.extraction_text, "John Doe");
        let age_extraction = extractions.iter().find(|e| e.extraction_class == "age").unwrap();
        assert_eq!(age_extraction.extraction_text, "30");
    }

    #[test]
    fn test_parse_wrapped_json() {
        let resolver = create_test_resolver();
        let json_response = r#"{"data": [{"name": "Alice", "city": "NYC"}]}"#;
        let expected_fields = vec!["name".to_string(), "city".to_string()];

        let result = resolver.parse_response_with_repair(json_response, &expected_fields);
        assert!(result.is_ok());
        
        let extractions = result.unwrap();
        assert_eq!(extractions.len(), 2);
        
        // Check that we have both fields (order may vary)
        let classes: std::collections::HashSet<_> = extractions.iter()
            .map(|e| e.extraction_class.as_str()).collect();
        assert!(classes.contains("name"));
        assert!(classes.contains("city"));
        
        // Check the values
        let name_extraction = extractions.iter().find(|e| e.extraction_class == "name").unwrap();
        assert_eq!(name_extraction.extraction_text, "Alice");
        let city_extraction = extractions.iter().find(|e| e.extraction_class == "city").unwrap();
        assert_eq!(city_extraction.extraction_text, "NYC");
    }

    #[test]
    fn test_parse_invalid_json() {
        let resolver = create_test_resolver();
        let invalid_response = r#"This is not JSON at all!"#;
        let expected_fields = vec!["name".to_string()];

        let result = resolver.parse_response_with_repair(invalid_response, &expected_fields);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_required_fields() {
        let resolver = create_test_resolver();
        let extractions = vec![
            Extraction::new("person".to_string(), "John".to_string()),
        ];
        let expected_fields = vec!["person".to_string(), "age".to_string()];
        
        // Test with require_all_fields = false (default)
        let result = resolver.validate_extractions(&extractions, &expected_fields);
        assert!(result.is_valid); // Should pass because require_all_fields is false
        
        // Test with require_all_fields = true
        let config = create_test_config();
        let validation_config = ValidationConfig {
            require_all_fields: true,
            save_raw_outputs: false,
            ..Default::default()
        };
        let resolver = Resolver::with_validation_config(&config, true, validation_config).unwrap();
        let result = resolver.validate_extractions(&extractions, &expected_fields);
        assert!(!result.is_valid); // Should fail because "age" is missing
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].message.contains("age"));
    }

    #[test]
    fn test_validation_empty_extractions() {
        let resolver = create_test_resolver();
        let extractions = vec![
            Extraction::new("person".to_string(), "".to_string()), // Empty text
            Extraction::new("age".to_string(), "25".to_string()),
        ];
        let expected_fields = vec!["person".to_string(), "age".to_string()];
        
        let result = resolver.validate_extractions(&extractions, &expected_fields);
        assert!(result.is_valid); // Valid despite warnings
        assert_eq!(result.warnings.len(), 1); // One warning for empty text
        assert!(result.warnings[0].message.contains("Empty extraction text"));
    }

    #[test]
    fn test_validation_low_extraction_count() {
        let resolver = create_test_resolver();
        let extractions = vec![
            Extraction::new("person".to_string(), "John".to_string()),
        ];
        let expected_fields = vec![
            "person".to_string(), 
            "age".to_string(), 
            "city".to_string(), 
            "email".to_string()
        ]; // 4 expected, only 1 found
        
        let result = resolver.validate_extractions(&extractions, &expected_fields);
        assert!(result.is_valid); // Still valid, just warnings
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.message.contains("Low extraction count")));
    }

    #[test]
    fn test_validate_and_parse_success() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);
        
        let valid_json = r#"{"person": "John Doe", "age": "30"}"#;
        let expected_fields = vec!["person".to_string(), "age".to_string()];
        
        let result = resolver.validate_and_parse(valid_json, &expected_fields);
        assert!(result.is_ok());
        
        let (extractions, validation_result) = result.unwrap();
        assert_eq!(extractions.len(), 2);
        assert!(validation_result.is_valid);
        assert!(validation_result.raw_output_file.is_some());
        
        // Verify the raw output file was created
        let raw_file = validation_result.raw_output_file.unwrap();
        assert!(std::path::Path::new(&raw_file).exists());
    }

    #[test]
    fn test_validate_and_parse_parse_failure() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);

        let invalid_json = "This is definitely not JSON!";
        let expected_fields = vec!["person".to_string()];

        let result = resolver.validate_and_parse(invalid_json, &expected_fields);
        assert!(result.is_err());
        // Raw output should still be saved even on parse failure
    }

    #[test]
    fn test_clean_response_removes_code_fences() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);

        // Test various code fence patterns
        let test_cases = vec![
            (r#"```json{"name": "John"}```"#, r#"{"name": "John"}"#),
            (r#"```yaml{"name": "John"}```"#, r#"{"name": "John"}"#),
            (r#"```{"name": "John"}```"#, r#"{"name": "John"}"#),
            (r#"```python{"name": "John"}```"#, r#"{"name": "John"}"#),
            (r#"Some text ```json{"name": "John"}``` more text"#, r#"Some text {"name": "John"} more text"#),
        ];

        for (input, expected) in test_cases {
            let cleaned = resolver.clean_response(input);
            assert_eq!(cleaned, expected, "Failed to clean: {}", input);
        }
    }

    #[test]
    fn test_detect_and_repair_malformed_json() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);
        let expected_fields = vec!["name".to_string(), "age".to_string(), "city".to_string()];

        // Test case: malformed JSON with multiple classes in single extraction_text
        let malformed_json: serde_json::Value = serde_json::json!({
            "person": "name: John Doe, age: 30, city: New York"
        });

        let repaired = resolver.detect_and_repair_malformed_json(&malformed_json, &expected_fields);
        assert!(repaired.is_some(), "Should detect malformed JSON");

        let repaired = repaired.unwrap();
        if let Some(obj) = repaired.as_object() {
            // Should extract name, age, and city as separate fields
            assert!(obj.contains_key("name"), "Should extract name field");
            assert!(obj.contains_key("age"), "Should extract age field");
            assert!(obj.contains_key("city"), "Should extract city field");

            assert_eq!(obj.get("name").unwrap().as_str().unwrap(), "John Doe");
            assert_eq!(obj.get("age").unwrap().as_str().unwrap(), "30");
            assert_eq!(obj.get("city").unwrap().as_str().unwrap(), "New York");
        } else {
            panic!("Repaired JSON should be an object");
        }

        // Test case: well-formed JSON should not be repaired
        let well_formed_json: serde_json::Value = serde_json::json!({
            "name": "John Doe",
            "age": "30",
            "city": "New York"
        });

        let repaired = resolver.detect_and_repair_malformed_json(&well_formed_json, &expected_fields);
        assert!(repaired.is_none(), "Well-formed JSON should not be repaired");
    }

    #[test]
    fn test_parse_response_with_code_fences() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);
        let expected_fields = vec!["name".to_string(), "age".to_string()];

        // Test parsing response wrapped in code fences
        let fenced_response = r#"```json
{
  "name": "Alice",
  "age": "25"
}
```"#;

        let result = resolver.parse_response_with_repair(fenced_response, &expected_fields);
        assert!(result.is_ok(), "Should parse fenced JSON successfully");

        let extractions = result.unwrap();
        assert_eq!(extractions.len(), 2, "Should extract 2 fields");

        let names: Vec<_> = extractions.iter().filter(|e| e.extraction_class == "name").collect();
        let ages: Vec<_> = extractions.iter().filter(|e| e.extraction_class == "age").collect();

        assert_eq!(names.len(), 1);
        assert_eq!(ages.len(), 1);
        assert_eq!(names[0].extraction_text, "Alice");
        assert_eq!(ages[0].extraction_text, "25");
    }

    #[test]
    fn test_parse_response_with_malformed_repair() {
        let temp_dir = TempDir::new().unwrap();
        let resolver = create_test_resolver_with_temp_dir(&temp_dir);
        let expected_fields = vec!["name".to_string(), "age".to_string(), "profession".to_string()];

        // Test parsing malformed response that should be repaired
        let malformed_response = r#"{
  "person": "name: Bob Smith, age: 35, profession: engineer"
}"#;

        let result = resolver.parse_response_with_repair(malformed_response, &expected_fields);
        assert!(result.is_ok(), "Should parse and repair malformed JSON successfully");

        let extractions = result.unwrap();
        assert_eq!(extractions.len(), 3, "Should extract 3 separate fields after repair");

        let name_found = extractions.iter().any(|e| e.extraction_class == "name" && e.extraction_text == "Bob Smith");
        let age_found = extractions.iter().any(|e| e.extraction_class == "age" && e.extraction_text == "35");
        let profession_found = extractions.iter().any(|e| e.extraction_class == "profession" && e.extraction_text == "engineer");

        assert!(name_found, "Should find extracted name");
        assert!(age_found, "Should find extracted age");
        assert!(profession_found, "Should find extracted profession");
    }

    // Type Coercion Tests
    mod type_coercion_tests {
        use super::*;

        fn create_coercion_resolver() -> Resolver {
            let config = create_test_config();
            let validation_config = ValidationConfig {
                enable_type_coercion: true,
                save_raw_outputs: false,
                ..Default::default()
            };
            Resolver::with_validation_config(&config, true, validation_config).unwrap()
        }

        #[test]
        fn test_integer_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("age".to_string(), "25".to_string()),
                Extraction::new("count".to_string(), "-10".to_string()),
                Extraction::new("year".to_string(), "2024".to_string()),
            ];
            let expected_fields = vec!["age".to_string(), "count".to_string(), "year".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 3);
            assert_eq!(coercion_summary.failed_coercions, 0);
            
            // Check specific coercions
            let age_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "age").unwrap();
            assert!(age_coercion.success);
            assert_eq!(age_coercion.target_type, CoercionTargetType::Integer);
            assert_eq!(age_coercion.coerced_value.as_ref().unwrap().as_i64().unwrap(), 25);
        }

        #[test]
        fn test_float_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("score".to_string(), "94.7".to_string()),
                Extraction::new("percentage".to_string(), "-12.5".to_string()),
                Extraction::new("scientific".to_string(), "1.23e-4".to_string()),
            ];
            let expected_fields = vec!["score".to_string(), "percentage".to_string(), "scientific".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 3);
            
            let score_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "score").unwrap();
            assert!(score_coercion.success);
            assert_eq!(score_coercion.target_type, CoercionTargetType::Float);
            assert!((score_coercion.coerced_value.as_ref().unwrap().as_f64().unwrap() - 94.7).abs() < 0.01);
        }

        #[test]
        fn test_boolean_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("active".to_string(), "true".to_string()),
                Extraction::new("enabled".to_string(), "yes".to_string()),
                Extraction::new("disabled".to_string(), "false".to_string()),
                Extraction::new("off".to_string(), "no".to_string()),
                Extraction::new("binary".to_string(), "1".to_string()),
                Extraction::new("zero".to_string(), "0".to_string()),
            ];
            let expected_fields = vec!["active".to_string(), "enabled".to_string(), "disabled".to_string(), "off".to_string(), "binary".to_string(), "zero".to_string()];

            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);

            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 6);

            let active_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "active").unwrap();
            assert!(active_coercion.success);
            assert_eq!(active_coercion.target_type, CoercionTargetType::Boolean);
            assert_eq!(active_coercion.coerced_value.as_ref().unwrap().as_bool().unwrap(), true);

            // "1" and "0" should now coerce to integers, not booleans
            let binary_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "binary").unwrap();
            assert!(binary_coercion.success);
            assert_eq!(binary_coercion.target_type, CoercionTargetType::Integer);
            assert_eq!(binary_coercion.coerced_value.as_ref().unwrap().as_i64().unwrap(), 1);

            let zero_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "zero").unwrap();
            assert!(zero_coercion.success);
            assert_eq!(zero_coercion.target_type, CoercionTargetType::Integer);
            assert_eq!(zero_coercion.coerced_value.as_ref().unwrap().as_i64().unwrap(), 0);
        }

        #[test]
        fn test_currency_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("funding".to_string(), "$1.5 million".to_string()),
                Extraction::new("budget".to_string(), "$2.3M".to_string()),
                Extraction::new("salary".to_string(), "$75,000".to_string()),
                Extraction::new("value".to_string(), "500K".to_string()),
                Extraction::new("debt".to_string(), "$1.2 billion".to_string()),
            ];
            let expected_fields = vec!["funding".to_string(), "budget".to_string(), "salary".to_string(), "value".to_string(), "debt".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 5);
            
            let funding_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "funding").unwrap();
            assert!(funding_coercion.success);
            assert_eq!(funding_coercion.target_type, CoercionTargetType::Currency);
            assert!((funding_coercion.coerced_value.as_ref().unwrap().as_f64().unwrap() - 1_500_000.0).abs() < 1.0);
        }

        #[test]
        fn test_percentage_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("accuracy".to_string(), "94.7%".to_string()),
                Extraction::new("completion".to_string(), "100%".to_string()),
                Extraction::new("error_rate".to_string(), "0.5%".to_string()),
            ];
            let expected_fields = vec!["accuracy".to_string(), "completion".to_string(), "error_rate".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 3);
            
            let accuracy_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "accuracy").unwrap();
            assert!(accuracy_coercion.success);
            assert_eq!(accuracy_coercion.target_type, CoercionTargetType::Percentage);
            assert!((accuracy_coercion.coerced_value.as_ref().unwrap().as_f64().unwrap() - 0.947).abs() < 0.001);
        }

        #[test]
        fn test_email_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("contact".to_string(), "john.doe@example.com".to_string()),
                Extraction::new("support".to_string(), "support@company.org".to_string()),
                Extraction::new("invalid".to_string(), "not-an-email".to_string()),
            ];
            let expected_fields = vec!["contact".to_string(), "support".to_string(), "invalid".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 2); // Only 2 valid emails
            assert_eq!(coercion_summary.failed_coercions, 1);
            
            let contact_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "contact").unwrap();
            assert!(contact_coercion.success);
            assert_eq!(contact_coercion.target_type, CoercionTargetType::Email);
            let coerced_obj = contact_coercion.coerced_value.as_ref().unwrap().as_object().unwrap();
            assert_eq!(coerced_obj.get("email").unwrap().as_str().unwrap(), "john.doe@example.com");
        }

        #[test]
        fn test_phone_coercion() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("phone1".to_string(), "(617) 555-1234".to_string()),
                Extraction::new("phone2".to_string(), "617-555-1234".to_string()),
                Extraction::new("phone3".to_string(), "617.555.1234".to_string()),
                Extraction::new("phone4".to_string(), "6175551234".to_string()),
                Extraction::new("invalid".to_string(), "123-45".to_string()),
            ];
            let expected_fields = vec!["phone1".to_string(), "phone2".to_string(), "phone3".to_string(), "phone4".to_string(), "invalid".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 4); // 4 valid phone numbers
            assert_eq!(coercion_summary.failed_coercions, 1);
            
            let phone1_coercion = coercion_summary.coercion_details.iter()
                .find(|d| d.field_name == "phone1").unwrap();
            assert!(phone1_coercion.success);
            assert_eq!(phone1_coercion.target_type, CoercionTargetType::PhoneNumber);
            let coerced_obj = phone1_coercion.coerced_value.as_ref().unwrap().as_object().unwrap();
            assert_eq!(coerced_obj.get("phone").unwrap().as_str().unwrap(), "(617) 555-1234");
        }

        #[test]
        fn test_no_coercion_when_disabled() {
            let config = create_test_config();
            let validation_config = ValidationConfig {
                enable_type_coercion: false, // Disabled
                save_raw_outputs: false,
                ..Default::default()
            };
            let resolver = Resolver::with_validation_config(&config, true, validation_config).unwrap();
            
            let extractions = vec![
                Extraction::new("age".to_string(), "25".to_string()),
            ];
            let expected_fields = vec!["age".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            assert!(result.coercion_summary.is_none()); // No coercion attempted
        }

        #[test]
        fn test_mixed_coercion_results() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("age".to_string(), "25".to_string()), // Should coerce to integer
                Extraction::new("name".to_string(), "John Doe".to_string()), // No coercion
                Extraction::new("email".to_string(), "john@example.com".to_string()), // Should coerce to email
                Extraction::new("invalid_number".to_string(), "abc123".to_string()), // Should fail coercion
                Extraction::new("percentage".to_string(), "95%".to_string()), // Should coerce to percentage
            ];
            let expected_fields = vec!["age".to_string(), "name".to_string(), "email".to_string(), "invalid_number".to_string(), "percentage".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            let coercion_summary = result.coercion_summary.unwrap();
            assert_eq!(coercion_summary.successful_coercions, 3); // age, email, percentage
            assert_eq!(coercion_summary.failed_coercions, 2); // name, invalid_number
            
            // Check that successful coercions have the right types
            let successful_types: Vec<_> = coercion_summary.coercion_details.iter()
                .filter(|d| d.success)
                .map(|d| &d.target_type)
                .collect();
            assert!(successful_types.contains(&&CoercionTargetType::Integer));
            assert!(successful_types.contains(&&CoercionTargetType::Email));
            assert!(successful_types.contains(&&CoercionTargetType::Percentage));
        }

        #[test]
        fn test_corrected_data_generation() {
            let resolver = create_coercion_resolver();
            let extractions = vec![
                Extraction::new("age".to_string(), "25".to_string()),
                Extraction::new("price".to_string(), "$19.99".to_string()),
                Extraction::new("active".to_string(), "true".to_string()),
                Extraction::new("invalid".to_string(), "not_a_number".to_string()),
            ];
            let expected_fields = vec!["age".to_string(), "price".to_string(), "active".to_string(), "invalid".to_string()];
            
            let result = resolver.validate_extractions(&extractions, &expected_fields);
            assert!(result.is_valid);
            
            // Check that corrected data was generated
            let corrected_data = result.corrected_data.unwrap();
            let corrected_obj = corrected_data.as_object().unwrap();
            
            // Check coerced values
            assert_eq!(corrected_obj.get("age").unwrap().as_i64().unwrap(), 25);
            assert_eq!(corrected_obj.get("price").unwrap().as_f64().unwrap(), 19.99);
            assert_eq!(corrected_obj.get("active").unwrap().as_bool().unwrap(), true);
            
            // Check that failed coercion keeps original string value
            assert_eq!(corrected_obj.get("invalid").unwrap().as_str().unwrap(), "not_a_number");
        }
    }
}
