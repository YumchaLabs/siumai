//! OpenAI Response Adapter System
//!
//! This module provides a flexible system for adapting different OpenAI-compatible
//! provider responses to our unified format. This allows us to handle provider-specific
//! response formats without polluting the core streaming logic.

use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
use serde_json::Value;
use std::sync::Arc;

/// Trait for adapting provider-specific response formats
pub trait ResponseAdapter: Send + Sync {
    /// Extract thinking/reasoning content from a delta object
    /// Different providers may use different field names (thinking, reasoning_content, etc.)
    fn extract_thinking_content(&self, delta: &Value) -> Option<String>;
    
    /// Extract regular content from a delta object
    fn extract_content(&self, delta: &Value) -> Option<String>;
    
    /// Extract tool calls from a delta object
    fn extract_tool_calls(&self, delta: &Value) -> Option<Vec<Value>>;
    
    /// Extract role information from a delta object
    fn extract_role(&self, delta: &Value) -> Option<String>;
    
    /// Provider identifier for debugging/logging
    fn provider_id(&self) -> &str;
}

/// Standard OpenAI response adapter
#[derive(Debug, Clone)]
pub struct StandardOpenAiAdapter;

impl ResponseAdapter for StandardOpenAiAdapter {
    fn extract_thinking_content(&self, delta: &Value) -> Option<String> {
        delta
            .get("thinking")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_content(&self, delta: &Value) -> Option<String> {
        delta
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_tool_calls(&self, delta: &Value) -> Option<Vec<Value>> {
        delta
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
    }
    
    fn extract_role(&self, delta: &Value) -> Option<String> {
        delta
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
    
    fn provider_id(&self) -> &str {
        "openai"
    }
}

/// SiliconFlow response adapter for handling DeepSeek models
#[derive(Debug, Clone)]
pub struct SiliconFlowAdapter;

impl ResponseAdapter for SiliconFlowAdapter {
    fn extract_thinking_content(&self, delta: &Value) -> Option<String> {
        // First try the standard 'thinking' field
        if let Some(thinking) = delta
            .get("thinking")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            return Some(thinking.to_string());
        }
        
        // Then try SiliconFlow's 'reasoning_content' field for DeepSeek models
        delta
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_content(&self, delta: &Value) -> Option<String> {
        delta
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_tool_calls(&self, delta: &Value) -> Option<Vec<Value>> {
        delta
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
    }
    
    fn extract_role(&self, delta: &Value) -> Option<String> {
        delta
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
    
    fn provider_id(&self) -> &str {
        "siliconflow"
    }
}

/// DeepSeek response adapter
#[derive(Debug, Clone)]
pub struct DeepSeekAdapter;

impl ResponseAdapter for DeepSeekAdapter {
    fn extract_thinking_content(&self, delta: &Value) -> Option<String> {
        // DeepSeek may use 'reasoning_content' field
        if let Some(reasoning) = delta
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            return Some(reasoning.to_string());
        }
        
        // Fallback to standard 'thinking' field
        delta
            .get("thinking")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_content(&self, delta: &Value) -> Option<String> {
        delta
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
    
    fn extract_tool_calls(&self, delta: &Value) -> Option<Vec<Value>> {
        delta
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
    }
    
    fn extract_role(&self, delta: &Value) -> Option<String> {
        delta
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
    
    fn provider_id(&self) -> &str {
        "deepseek"
    }
}

/// Response adapter registry for managing different adapters
#[derive(Debug, Clone)]
pub struct ResponseAdapterRegistry {
    adapters: std::collections::HashMap<String, Arc<dyn ResponseAdapter>>,
}

impl ResponseAdapterRegistry {
    /// Create a new registry with default adapters
    pub fn new() -> Self {
        let mut registry = Self {
            adapters: std::collections::HashMap::new(),
        };
        
        // Register default adapters
        registry.register("openai", Arc::new(StandardOpenAiAdapter));
        registry.register("siliconflow", Arc::new(SiliconFlowAdapter));
        registry.register("deepseek", Arc::new(DeepSeekAdapter));
        
        registry
    }
    
    /// Register a new adapter
    pub fn register(&mut self, provider_id: &str, adapter: Arc<dyn ResponseAdapter>) {
        self.adapters.insert(provider_id.to_string(), adapter);
    }
    
    /// Get an adapter by provider ID
    pub fn get_adapter(&self, provider_id: &str) -> Option<Arc<dyn ResponseAdapter>> {
        self.adapters.get(provider_id).cloned()
    }
    
    /// Get the default OpenAI adapter
    pub fn get_default_adapter(&self) -> Arc<dyn ResponseAdapter> {
        self.adapters
            .get("openai")
            .cloned()
            .unwrap_or_else(|| Arc::new(StandardOpenAiAdapter))
    }
}

impl Default for ResponseAdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}
