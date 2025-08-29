//! OpenAI Compatible Provider Adapter
//!
//! This module provides a flexible adapter for OpenAI-compatible providers like
//! SiliconFlow, DeepSeek, OpenRouter, etc. It handles the differences in response
//! formats and parameter requirements between providers.

use async_trait::async_trait;
use serde_json::Value;
use std::any::Any;

use super::*;
use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
use crate::types::*;

/// OpenAI compatible provider configuration trait
pub trait OpenAiCompatibleConfig: Clone + Send + Sync + 'static {
    /// Base URL for the provider
    fn base_url(&self) -> &str;
    
    /// Default model for the provider
    fn default_model(&self) -> &str;
    
    /// API key
    fn api_key(&self) -> &str;
    
    /// Transform provider-specific parameters
    fn transform_params(&self, params: &mut Value) -> Result<(), LlmError>;
    
    /// Get field mappings for response parsing
    fn field_mappings(&self) -> FieldMappings;
    
    /// Get custom headers
    fn custom_headers(&self) -> reqwest::header::HeaderMap {
        reqwest::header::HeaderMap::new()
    }
    
    /// Provider-specific capabilities
    fn capabilities(&self) -> ProviderCapabilities;
}

/// OpenAI compatible adapter
pub struct OpenAiCompatibleAdapter<C: OpenAiCompatibleConfig> {
    config: C,
}

impl<C: OpenAiCompatibleConfig> OpenAiCompatibleAdapter<C> {
    pub fn new(config: C) -> Self {
        Self { config }
    }
}

impl<C: OpenAiCompatibleConfig> ProviderAdapter for OpenAiCompatibleAdapter<C> {
    type Config = C;
    type RequestParams = Value;
    type ResponseChunk = Value;
    
    fn provider_id(&self) -> &'static str {
        "openai_compatible"
    }
    
    fn request_transformer(&self) -> Box<dyn RequestTransformer<Self::RequestParams>> {
        Box::new(OpenAiCompatibleRequestTransformer {
            config: self.config.clone(),
        })
    }
    
    fn response_adapter(&self) -> Box<dyn ResponseAdapter<Self::ResponseChunk>> {
        Box::new(OpenAiCompatibleResponseAdapter {
            field_mappings: self.config.field_mappings(),
        })
    }
    
    fn capabilities(&self) -> ProviderCapabilities {
        self.config.capabilities()
    }
    
    fn validate_config(&self, config: &Self::Config) -> Result<(), LlmError> {
        if config.base_url().is_empty() {
            return Err(LlmError::ConfigurationError("Base URL is required".to_string()));
        }
        if config.api_key().is_empty() {
            return Err(LlmError::ConfigurationError("API key is required".to_string()));
        }
        Ok(())
    }
}

/// Request transformer for OpenAI compatible providers
struct OpenAiCompatibleRequestTransformer<C: OpenAiCompatibleConfig> {
    config: C,
}

impl<C: OpenAiCompatibleConfig> RequestTransformer<Value> for OpenAiCompatibleRequestTransformer<C> {
    fn transform_chat_request(&self, request: &ChatRequest) -> Result<Value, LlmError> {
        let mut params = serde_json::json!({
            "model": request.model.as_deref().unwrap_or(self.config.default_model()),
            "messages": request.messages,
            "stream": request.stream.unwrap_or(false),
        });
        
        // Add optional parameters
        if let Some(temp) = request.temperature {
            params["temperature"] = serde_json::Value::Number(
                serde_json::Number::from_f64(temp).unwrap_or_else(|| serde_json::Number::from(0))
            );
        }
        
        if let Some(max_tokens) = request.max_tokens {
            params["max_tokens"] = serde_json::Value::Number(serde_json::Number::from(max_tokens));
        }
        
        if let Some(top_p) = request.top_p {
            params["top_p"] = serde_json::Value::Number(
                serde_json::Number::from_f64(top_p).unwrap_or_else(|| serde_json::Number::from(1))
            );
        }
        
        // Apply provider-specific transformations
        self.config.transform_params(&mut params)?;
        
        Ok(params)
    }
    
    fn transform_embedding_request(&self, texts: &[String]) -> Result<Value, LlmError> {
        let params = serde_json::json!({
            "model": self.config.default_model(),
            "input": texts,
        });
        
        Ok(params)
    }
    
    fn transform_rerank_request(&self, request: &RerankRequest) -> Result<Value, LlmError> {
        let params = serde_json::json!({
            "model": request.model.as_deref().unwrap_or(self.config.default_model()),
            "query": request.query,
            "documents": request.documents,
            "top_k": request.top_k,
        });
        
        Ok(params)
    }
    
    fn add_provider_params(&self, params: &mut Value, _config: &dyn Any) -> Result<(), LlmError> {
        // Provider-specific parameter additions can be implemented here
        Ok(())
    }
    
    fn add_custom_headers(&self, headers: &mut reqwest::header::HeaderMap) -> Result<(), LlmError> {
        // Add authorization header
        let auth_value = format!("Bearer {}", self.config.api_key());
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&auth_value)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {}", e)))?,
        );
        
        // Add custom headers from config
        let custom_headers = self.config.custom_headers();
        for (key, value) in custom_headers.iter() {
            headers.insert(key, value.clone());
        }
        
        Ok(())
    }
}

/// Response adapter for OpenAI compatible providers
struct OpenAiCompatibleResponseAdapter {
    field_mappings: FieldMappings,
}

impl ResponseAdapter<Value> for OpenAiCompatibleResponseAdapter {
    fn adapt_chunk(&self, chunk: Value) -> Result<Vec<ChatStreamEvent>, LlmError> {
        let mut events = Vec::new();
        
        // Extract delta from the standard OpenAI response structure
        let delta = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("delta"));
        
        if let Some(delta) = delta {
            // Handle thinking content with multiple possible field names
            if let Some(thinking) = self.extract_thinking_from_delta(delta) {
                events.push(ChatStreamEvent::ThinkingDelta { content: thinking });
            }
            
            // Handle regular content
            if let Some(content) = self.extract_content_from_delta(delta) {
                events.push(ChatStreamEvent::ContentDelta { content });
            }
            
            // Handle tool calls
            if let Some(tool_calls) = self.extract_tool_calls_from_delta(delta) {
                for tool_call in tool_calls {
                    events.push(ChatStreamEvent::ToolCallDelta { tool_call });
                }
            }
            
            // Handle role
            if let Some(role) = self.extract_role_from_delta(delta) {
                events.push(ChatStreamEvent::RoleAssigned { role });
            }
        }
        
        // Handle usage information
        if let Some(usage) = chunk.get("usage") {
            if let Ok(usage_info) = serde_json::from_value::<Usage>(usage.clone()) {
                events.push(ChatStreamEvent::UsageUpdate { usage: usage_info });
            }
        }
        
        Ok(events)
    }
    
    fn extract_thinking(&self, chunk: &Value) -> Option<String> {
        let delta = chunk
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?;
        self.extract_thinking_from_delta(delta)
    }
    
    fn extract_content(&self, chunk: &Value) -> Option<String> {
        let delta = chunk
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?;
        self.extract_content_from_delta(delta)
    }
    
    fn extract_tool_calls(&self, chunk: &Value) -> Option<Vec<ToolCall>> {
        let delta = chunk
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?;
        self.extract_tool_calls_from_delta(delta)
    }
    
    fn extract_role(&self, chunk: &Value) -> Option<String> {
        let delta = chunk
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?;
        self.extract_role_from_delta(delta)
    }
    
    fn handle_stream_end(&self) -> Option<ChatStreamEvent> {
        Some(ChatStreamEvent::StreamEnd {
            response: ChatResponse::default(),
        })
    }
    
    fn handle_error(&self, error: LlmError) -> ChatStreamEvent {
        ChatStreamEvent::Error {
            error: error.to_string(),
        }
    }
}

impl OpenAiCompatibleResponseAdapter {
    fn extract_thinking_from_delta(&self, delta: &Value) -> Option<String> {
        // Try each thinking field in priority order
        for field_name in &self.field_mappings.thinking_fields {
            if let Some(thinking) = delta
                .get(field_name)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                return Some(thinking.to_string());
            }
        }
        None
    }
    
    fn extract_content_from_delta(&self, delta: &Value) -> Option<String> {
        delta
            .get(self.field_mappings.content_field)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_tool_calls_from_delta(&self, delta: &Value) -> Option<Vec<ToolCall>> {
        delta
            .get(self.field_mappings.tool_calls_field)
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                let tool_calls: Result<Vec<ToolCall>, _> = arr
                    .iter()
                    .map(|tc| serde_json::from_value(tc.clone()))
                    .collect();
                tool_calls.ok()
            })
    }
    
    fn extract_role_from_delta(&self, delta: &Value) -> Option<String> {
        delta
            .get(self.field_mappings.role_field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}
