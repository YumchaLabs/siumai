//! Provider Adapter System
//!
//! This module implements a flexible adapter system for handling different AI providers
//! with their specific request/response formats. Inspired by Cherry Studio's middleware
//! architecture but designed for Rust's type system.

use async_trait::async_trait;
use serde_json::Value;
use std::any::Any;

use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
use crate::types::*;

pub mod openai_compatible;

/// Core provider adapter trait
///
/// This trait defines the interface for adapting different AI providers to our unified system.
/// Each provider implements this trait to handle request transformation and response adaptation.
pub trait ProviderAdapter: Send + Sync + 'static {
    type Config: Clone + Send + Sync;
    type RequestParams: Send + Sync;
    type ResponseChunk: Send + Sync;
    
    /// Provider identifier
    fn provider_id(&self) -> &'static str;
    
    /// Get request transformer
    fn request_transformer(&self) -> Box<dyn RequestTransformer<Self::RequestParams>>;
    
    /// Get response adapter
    fn response_adapter(&self) -> Box<dyn ResponseAdapter<Self::ResponseChunk>>;
    
    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities;
    
    /// Validate configuration
    fn validate_config(&self, config: &Self::Config) -> Result<(), LlmError>;
}

/// Request transformer trait
///
/// Handles conversion from our unified request format to provider-specific formats
pub trait RequestTransformer<T>: Send + Sync {
    /// Transform chat request
    fn transform_chat_request(&self, request: &ChatRequest) -> Result<T, LlmError>;
    
    /// Transform embedding request
    fn transform_embedding_request(&self, texts: &[String]) -> Result<T, LlmError>;
    
    /// Transform rerank request
    fn transform_rerank_request(&self, request: &RerankRequest) -> Result<T, LlmError>;
    
    /// Add provider-specific parameters
    fn add_provider_params(&self, params: &mut T, config: &dyn Any) -> Result<(), LlmError>;
    
    /// Add custom headers
    fn add_custom_headers(&self, headers: &mut reqwest::header::HeaderMap) -> Result<(), LlmError>;
}

/// Response adapter trait
///
/// Handles conversion from provider-specific response chunks to our unified events
pub trait ResponseAdapter<TChunk>: Send + Sync {
    /// Adapt a response chunk to stream events
    fn adapt_chunk(&self, chunk: TChunk) -> Result<Vec<ChatStreamEvent>, LlmError>;
    
    /// Extract thinking content from chunk
    fn extract_thinking(&self, chunk: &TChunk) -> Option<String>;
    
    /// Extract regular content from chunk
    fn extract_content(&self, chunk: &TChunk) -> Option<String>;
    
    /// Extract tool calls from chunk
    fn extract_tool_calls(&self, chunk: &TChunk) -> Option<Vec<ToolCall>>;
    
    /// Extract role information
    fn extract_role(&self, chunk: &TChunk) -> Option<String>;
    
    /// Handle stream end
    fn handle_stream_end(&self) -> Option<ChatStreamEvent>;
    
    /// Handle errors
    fn handle_error(&self, error: LlmError) -> ChatStreamEvent;
}

/// Field mappings for OpenAI-compatible providers
///
/// This allows different providers to use different field names for the same concepts
#[derive(Debug, Clone)]
pub struct FieldMappings {
    /// Fields that contain thinking/reasoning content (in priority order)
    pub thinking_fields: Vec<&'static str>,
    /// Field that contains regular content
    pub content_field: &'static str,
    /// Field that contains tool calls
    pub tool_calls_field: &'static str,
    /// Field that contains role information
    pub role_field: &'static str,
}

impl Default for FieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec!["thinking"],
            content_field: "content",
            tool_calls_field: "tool_calls",
            role_field: "role",
        }
    }
}

/// Provider capabilities information
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub streaming: bool,
    pub thinking: bool,
    pub tools: bool,
    pub vision: bool,
    pub audio: bool,
    pub embedding: bool,
    pub rerank: bool,
    pub image_generation: bool,
    pub file_management: bool,
}

/// Adapter registry for managing different provider adapters
#[derive(Debug)]
pub struct AdapterRegistry {
    adapters: std::collections::HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl AdapterRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            adapters: std::collections::HashMap::new(),
        }
    }
    
    /// Register an adapter
    pub fn register<A: ProviderAdapter>(&mut self, adapter: A) {
        self.adapters.insert(
            adapter.provider_id().to_string(),
            Box::new(adapter),
        );
    }
    
    /// Get an adapter by provider ID
    pub fn get_adapter<A: ProviderAdapter + 'static>(&self, provider_id: &str) -> Option<&A> {
        self.adapters
            .get(provider_id)
            .and_then(|adapter| adapter.downcast_ref::<A>())
    }
    
    /// List all registered provider IDs
    pub fn list_providers(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified provider client that uses adapters
pub struct UnifiedProviderClient<A: ProviderAdapter> {
    adapter: A,
    http_client: reqwest::Client,
    config: A::Config,
}

impl<A: ProviderAdapter> UnifiedProviderClient<A> {
    /// Create a new unified client
    pub fn new(adapter: A, config: A::Config, http_client: Option<reqwest::Client>) -> Result<Self, LlmError> {
        // Validate configuration
        adapter.validate_config(&config)?;
        
        let client = Self {
            adapter,
            http_client: http_client.unwrap_or_else(|| reqwest::Client::new()),
            config,
        };
        
        Ok(client)
    }
    
    /// Get provider ID
    pub fn provider_id(&self) -> &'static str {
        self.adapter.provider_id()
    }
    
    /// Get provider capabilities
    pub fn capabilities(&self) -> ProviderCapabilities {
        self.adapter.capabilities()
    }
    
    /// Get HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
    
    /// Get configuration
    pub fn config(&self) -> &A::Config {
        &self.config
    }
}

/// Helper trait for creating adapters
pub trait AdapterBuilder<A: ProviderAdapter> {
    type Config;
    
    /// Build the adapter with configuration
    fn build(config: Self::Config) -> Result<A, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_mappings_default() {
        let mappings = FieldMappings::default();
        assert_eq!(mappings.thinking_fields, vec!["thinking"]);
        assert_eq!(mappings.content_field, "content");
        assert_eq!(mappings.tool_calls_field, "tool_calls");
        assert_eq!(mappings.role_field, "role");
    }
    
    #[test]
    fn test_adapter_registry() {
        let mut registry = AdapterRegistry::new();
        assert_eq!(registry.list_providers().len(), 0);
        
        // Note: We would need concrete adapter implementations to test registration
    }
    
    #[test]
    fn test_provider_capabilities_default() {
        let caps = ProviderCapabilities::default();
        assert!(!caps.chat);
        assert!(!caps.streaming);
        assert!(!caps.thinking);
    }
}
