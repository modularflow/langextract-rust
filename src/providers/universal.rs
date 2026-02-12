//! Universal provider implementation.

use super::config::{ProviderConfig, ProviderType};
use crate::{
    data::FormatType,
    exceptions::{LangExtractError, LangExtractResult},
    inference::{BaseLanguageModel, ScoredOutput},
    logging::{report_progress, ProgressEvent},
    schema::BaseSchema,
};
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::time::Duration;

/// Universal language model provider
pub struct UniversalProvider {
    config: ProviderConfig,
    format_type: FormatType,
    client: reqwest::Client,
    #[cfg(feature = "openai")]
    openai_client: Option<async_openai::Client<async_openai::config::OpenAIConfig>>,
    schema: Option<Box<dyn BaseSchema>>,
    fence_output_override: Option<bool>,
}

impl UniversalProvider {
    /// Retry helper function with exponential backoff
    /// Retries at least 3 times with 30-second delays between attempts
    pub async fn retry_with_backoff<T, F, Fut>(
        &self,
        mut operation: F,
        operation_name: &str,
    ) -> LangExtractResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = LangExtractResult<T>>,
    {
        let max_retries = 3;
        let base_delay = Duration::from_secs(30);

        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt == max_retries {
                        // Last attempt failed, return the error
                        return Err(LangExtractError::inference_simple(
                            format!("{} failed after {} attempts. Last error: {}", operation_name, max_retries + 1, e)
                        ));
                    }

                    // Calculate delay with exponential backoff (30s, 60s, 90s)
                    let delay = base_delay * (attempt + 1) as u32;
                    report_progress(ProgressEvent::RetryAttempt {
                        operation: operation_name.to_string(),
                        attempt: attempt + 1,
                        max_attempts: max_retries + 1,
                        delay_seconds: delay.as_secs(),
                    });

                    // Add a small delay before the main sleep to ensure logs are printed
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                    // Use tokio::time::sleep explicitly
                    tokio::time::sleep(delay).await;
                }
            }
        }

        unreachable!("Should have returned from the loop")
    }

    /// Create a new universal provider
    pub fn new(config: ProviderConfig) -> LangExtractResult<Self> {
        let client = reqwest::Client::new();

        #[cfg(feature = "openai")]
        let openai_client = if config.provider_type == ProviderType::OpenAI {
            if let Some(api_key) = &config.api_key {
                let openai_config = async_openai::config::OpenAIConfig::new()
                    .with_api_key(api_key)
                    .with_api_base(&config.base_url);
                Some(async_openai::Client::with_config(openai_config))
            } else {
                return Err(LangExtractError::configuration(
                    "API key is required for OpenAI provider",
                ));
            }
        } else {
            None
        };

        #[cfg(not(feature = "openai"))]
        let openai_client = None;

        Ok(Self {
            config,
            format_type: FormatType::Json,
            client,
            openai_client,
            schema: None,
            fence_output_override: None,
        })
    }

    /// Process a single OpenAI prompt
    #[cfg(feature = "openai")]
    async fn infer_openai_single(
        &self,
        prompt: &str,
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> LangExtractResult<Vec<ScoredOutput>> {
        use async_openai::types::{
            ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
            ChatCompletionRequestSystemMessageContent, CreateChatCompletionRequest,
        };

        let client = self.openai_client.as_ref().ok_or_else(|| {
            LangExtractError::configuration("OpenAI client not initialized")
        })?;

        // Create system message for format instructions
        let system_message = match self.format_type {
            FormatType::Json => "You are a helpful assistant that responds in JSON format. Always return valid JSON that matches the expected structure from the examples.",
            FormatType::Yaml => "You are a helpful assistant that responds in YAML format. Always return valid YAML that matches the expected structure from the examples.",
        };

        // Create messages for the chat completion
        let messages = vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(system_message.to_string()),
                name: None,
            }),
            ChatCompletionRequestMessage::User(async_openai::types::ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                name: None,
            }),
        ];

        // Build the request
        let mut request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            temperature: None,
            max_tokens: None,
            ..Default::default()
        };

        // Apply parameters from kwargs
        if let Some(temp) = kwargs.get("temperature") {
            if let Some(temp_f64) = temp.as_f64() {
                request.temperature = Some(temp_f64 as f32);
            }
        }
        
        if let Some(max_tokens) = kwargs.get("max_tokens") {
            if let Some(max_tokens_u64) = max_tokens.as_u64() {
                request.max_tokens = Some(max_tokens_u64 as u32);
            }
        }

        // Make the API call with retry logic
        report_progress(ProgressEvent::ModelCall {
            provider: "OpenAI".to_string(),
            model: self.config.model.clone(),
            input_length: prompt.len(),
        });
        
        let response = self.retry_with_backoff(
            || async {
                let result = client.chat().create(request.clone()).await.map_err(|e| {
                    report_progress(ProgressEvent::Error {
                        operation: "OpenAI API request".to_string(),
                        error: format!("OpenAI API error: {}", e),
                    });
                    LangExtractError::inference_simple(format!("OpenAI API error: {}", e))
                });
                result
            },
            &format!("OpenAI API call for prompt len {}", prompt.len())
        ).await?;

        // Extract the response content
        let content = response
            .choices
            .get(0)
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| {
                LangExtractError::parsing("No content in OpenAI response")
            })?;

        Ok(vec![ScoredOutput::from_text(content.clone())])
    }

    /// Inference implementation for OpenAI-compatible APIs — processes prompts concurrently
    #[cfg(feature = "openai")]
    async fn infer_openai(
        &self,
        batch_prompts: &[String],
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> LangExtractResult<Vec<Vec<ScoredOutput>>> {
        use futures::future::join_all;

        // Process all prompts concurrently instead of sequentially
        let futures: Vec<_> = batch_prompts.iter()
            .map(|prompt| self.infer_openai_single(prompt, kwargs))
            .collect();

        let results = join_all(futures).await;
        
        // Collect results, propagating errors
        results.into_iter().collect()
    }

    /// Process a single Ollama prompt
    async fn infer_ollama_single(
        &self,
        prompt: &str,
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> LangExtractResult<Vec<ScoredOutput>> {
        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
        });

        // Set format for JSON output if needed
        if self.format_type == FormatType::Json {
            request_body["format"] = serde_json::json!("json");
        }

        // Apply parameters from kwargs
        let mut options = serde_json::Map::new();
        if let Some(temp) = kwargs.get("temperature") {
            options.insert("temperature".to_string(), temp.clone());
        }
        if let Some(max_tokens) = kwargs.get("max_tokens") {
            options.insert("num_predict".to_string(), max_tokens.clone());
        }
        if !options.is_empty() {
            request_body["options"] = serde_json::Value::Object(options);
        }

        let url = format!("{}/api/generate", self.config.base_url);

        // Make the API call with retry logic
        report_progress(ProgressEvent::ModelCall {
            provider: "Ollama".to_string(),
            model: self.config.model.clone(),
            input_length: prompt.len(),
        });
        
        let response_body = self.retry_with_backoff(
            || async {
                let mut request = self.client.post(&url).json(&request_body);

                // Add headers
                for (key, value) in &self.config.headers {
                    request = request.header(key, value);
                }

                let response = request.send().await.map_err(|e| {
                    report_progress(ProgressEvent::Error {
                        operation: "Ollama HTTP request".to_string(),
                        error: format!("HTTP request failed: {}", e),
                    });
                    LangExtractError::NetworkError(e)
                })?;

                if !response.status().is_success() {
                    let status = response.status();
                    report_progress(ProgressEvent::Error {
                        operation: "Ollama HTTP status".to_string(),
                        error: format!("HTTP error status: {}", status),
                    });
                    return Err(LangExtractError::inference_simple(format!(
                        "Ollama API error: HTTP {}",
                        status
                    )));
                }

                let response_body: serde_json::Value = response.json().await.map_err(|e| {
                    report_progress(ProgressEvent::Error {
                        operation: "Ollama JSON parsing".to_string(),
                        error: format!("JSON parsing failed: {}", e),
                    });
                    LangExtractError::parsing(format!("Failed to parse Ollama response: {}", e))
                })?;

                Ok(response_body)
            },
            &format!("Ollama API call for prompt len {}", prompt.len())
        ).await?;

        let content = response_body
            .get("response")
            .and_then(|r| r.as_str())
            .ok_or_else(|| {
                LangExtractError::parsing("Missing 'response' field in Ollama response")
            })?;

        Ok(vec![ScoredOutput::from_text(content.to_string())])
    }

    /// Inference implementation for Ollama — processes prompts concurrently
    async fn infer_ollama(
        &self,
        batch_prompts: &[String],
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> LangExtractResult<Vec<Vec<ScoredOutput>>> {
        use futures::future::join_all;

        // Process all prompts concurrently instead of sequentially
        let futures: Vec<_> = batch_prompts.iter()
            .map(|prompt| self.infer_ollama_single(prompt, kwargs))
            .collect();

        let results = join_all(futures).await;
        
        // Collect results, propagating errors
        results.into_iter().collect()
    }
}

#[async_trait]
impl BaseLanguageModel for UniversalProvider {
    fn get_schema_class(&self) -> Option<Box<dyn BaseSchema>> {
        // Return a format mode schema for now
        crate::schema::FormatModeSchema::from_examples(&[], "_attributes").ok()
    }

    fn apply_schema(&mut self, schema: Option<Box<dyn BaseSchema>>) {
        self.schema = schema;
    }

    fn set_fence_output(&mut self, fence_output: Option<bool>) {
        self.fence_output_override = fence_output;
    }

    fn requires_fence_output(&self) -> bool {
        if let Some(override_val) = self.fence_output_override {
            return override_val;
        }

        // OpenAI with JSON mode doesn't need fences, Ollama might
        match self.config.provider_type {
            ProviderType::OpenAI if self.schema.is_some() => false,
            _ => true,
        }
    }

    async fn infer(
        &self,
        batch_prompts: &[String],
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> LangExtractResult<Vec<Vec<ScoredOutput>>> {
        match self.config.provider_type {
            #[cfg(feature = "openai")]
            ProviderType::OpenAI => self.infer_openai(batch_prompts, kwargs).await,
            ProviderType::Ollama => self.infer_ollama(batch_prompts, kwargs).await,
            ProviderType::Custom => {
                Err(LangExtractError::configuration(
                    "Custom provider inference not yet implemented"
                ))
            }
            #[cfg(not(feature = "openai"))]
            ProviderType::OpenAI => {
                Err(LangExtractError::configuration(
                    "OpenAI feature not enabled. Enable with --features openai"
                ))
            }
        }
    }

    fn format_type(&self) -> FormatType {
        self.format_type
    }

    fn model_id(&self) -> &str {
        &self.config.model
    }

    fn provider_name(&self) -> &str {
        match self.config.provider_type {
            ProviderType::OpenAI => "openai",
            ProviderType::Ollama => "ollama",
            ProviderType::Custom => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::config::ProviderConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_with_backoff_success_after_failures() {
        let config = ProviderConfig::ollama("test-model", None);
        let provider = UniversalProvider::new(config).unwrap();

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = provider.retry_with_backoff(
            move || {
                let attempt_count = attempt_count_clone.clone();
                async move {
                    let current = attempt_count.fetch_add(1, Ordering::SeqCst);
                    if current < 2 {
                        Err::<String, _>(LangExtractError::inference_simple(format!("Attempt {} failed", current + 1)))
                    } else {
                        Ok("Success!".to_string())
                    }
                }
            },
            "Test operation"
        ).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success!");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_immediate_success() {
        let config = ProviderConfig::ollama("test-model", None);
        let provider = UniversalProvider::new(config).unwrap();

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = provider.retry_with_backoff(
            move || {
                let attempt_count = attempt_count_clone.clone();
                async move {
                    attempt_count.fetch_add(1, Ordering::SeqCst);
                    Ok("Immediate success!".to_string())
                }
            },
            "Test operation"
        ).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Immediate success!");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }
}
