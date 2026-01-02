//! Chat response types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::content::{ContentPart, MessageContent};
use super::message::{ChatMessage, MessageRole};
use super::metadata::MessageMetadata;
use crate::types::{FinishReason, Usage, Warning};

/// Audio output from the model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioOutput {
    /// Unique identifier for this audio response
    pub id: String,
    /// Unix timestamp (in seconds) when this audio expires
    pub expires_at: i64,
    /// Base64-encoded audio data
    pub data: String,
    /// Transcript of the audio
    pub transcript: String,
}

/// Chat response from the provider
///
/// The response content can include text, tool calls, reasoning, and other content types.
/// Tool calls and reasoning are now part of the content, not separate fields.
///
/// # Examples
///
/// ```rust
/// use siumai::types::{ChatResponse, MessageContent, ContentPart};
/// use serde_json::json;
///
/// // Response with tool calls
/// let response = ChatResponse::new(MessageContent::MultiModal(vec![
///     ContentPart::text("Let me search for that..."),
///     ContentPart::tool_call("call_123", "search", json!({"query":"rust"}), None),
/// ]));
///
/// // Check for tool calls
/// if response.has_tool_calls() {
///     let tool_calls = response.tool_calls();
///     println!("Found {} tool calls", tool_calls.len());
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Response ID
    pub id: Option<String>,
    /// The response content (can include text, tool calls, reasoning, etc.)
    pub content: MessageContent,
    /// Model used for the response
    pub model: Option<String>,
    /// Usage statistics
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Audio output (if audio modality was requested)
    pub audio: Option<AudioOutput>,
    /// System fingerprint (backend configuration identifier)
    ///
    /// This fingerprint represents the backend configuration that the model runs with.
    /// Can be used in conjunction with the `seed` request parameter to understand when
    /// backend changes have been made that might impact determinism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// Service tier used for processing the request
    ///
    /// Indicates the actual service tier used (e.g., "default", "scale", "flex", "priority").
    /// May differ from the requested service tier based on availability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Warnings from the model provider (e.g., unsupported settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Warning>>,
    /// Provider-specific metadata (nested structure for namespace isolation)
    ///
    /// This field contains provider-specific metadata in a nested structure:
    /// `{ "provider_id": { "key": value, ... }, ... }`
    ///
    /// For type-safe access to common provider metadata, use the helper methods:
    /// - `provider_metadata_as::<T>(provider_id)` for provider-agnostic typed extraction
    ///
    /// For OpenAI-specific typed metadata, prefer the provider extension trait:
    /// `siumai::provider_ext::openai::OpenAiChatResponseExt::openai_metadata()`.
    ///
    /// For Anthropic-specific typed metadata, prefer the provider extension trait:
    /// `siumai::provider_ext::anthropic::AnthropicChatResponseExt::anthropic_metadata()`.
    ///
    /// For Gemini-specific typed metadata, prefer the provider extension trait:
    /// `siumai::provider_ext::gemini::GeminiChatResponseExt::gemini_metadata()`.
    ///
    /// # Example
    /// ```rust,ignore
    /// use siumai::provider_ext::anthropic::AnthropicChatResponseExt;
    ///
    /// if let Some(meta) = response.anthropic_metadata() {
    ///     if let Some(cache_tokens) = meta.cache_read_input_tokens {
    ///         println!("Cache hit! Saved {} tokens", cache_tokens);
    ///     }
    /// }
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<HashMap<String, HashMap<String, serde_json::Value>>>,
}

impl ChatResponse {
    /// Create a new chat response
    pub fn new(content: MessageContent) -> Self {
        Self {
            id: None,
            content,
            model: None,
            usage: None,
            finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        }
    }

    /// Create an empty response (typically used for stream end events)
    pub fn empty() -> Self {
        Self {
            id: None,
            content: MessageContent::Text(String::new()),
            model: None,
            usage: None,
            finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        }
    }

    /// Create an empty response with a specific finish reason
    pub fn empty_with_finish_reason(reason: FinishReason) -> Self {
        Self {
            id: None,
            content: MessageContent::Text(String::new()),
            model: None,
            usage: None,
            finish_reason: Some(reason),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        }
    }

    /// Get the text content of the response
    pub fn content_text(&self) -> Option<&str> {
        self.content.text()
    }

    /// Get all text content of the response
    pub fn text(&self) -> Option<String> {
        Some(self.content.all_text())
    }

    /// Extract all tool calls from response content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, MessageContent, ContentPart};
    /// use serde_json::json;
    ///
    /// let response = ChatResponse::new(MessageContent::MultiModal(vec![
    ///     ContentPart::text("Let me search..."),
    ///     ContentPart::tool_call("call_123", "search", json!({}), None),
    /// ]));
    ///
    /// let tool_calls = response.tool_calls();
    /// assert_eq!(tool_calls.len(), 1);
    /// ```
    pub fn tool_calls(&self) -> Vec<&ContentPart> {
        match &self.content {
            MessageContent::MultiModal(parts) => {
                parts.iter().filter(|p| p.is_tool_call()).collect()
            }
            _ => vec![],
        }
    }

    /// Check if the response has tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls().is_empty()
    }

    /// Get tool calls (deprecated - use tool_calls() instead)
    #[deprecated(since = "0.12.0", note = "Use `tool_calls()` instead")]
    pub fn get_tool_calls(&self) -> Option<Vec<&ContentPart>> {
        let calls = self.tool_calls();
        if calls.is_empty() { None } else { Some(calls) }
    }

    /// Extract all reasoning/thinking content from response
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, MessageContent, ContentPart};
    ///
    /// let response = ChatResponse::new(MessageContent::MultiModal(vec![
    ///     ContentPart::reasoning("Let me think..."),
    ///     ContentPart::text("The answer is 42"),
    /// ]));
    ///
    /// let reasoning = response.reasoning();
    /// assert_eq!(reasoning.len(), 1);
    /// ```
    pub fn reasoning(&self) -> Vec<&str> {
        match &self.content {
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Reasoning { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Check if the response has reasoning/thinking content
    pub fn has_reasoning(&self) -> bool {
        !self.reasoning().is_empty()
    }

    /// Convert response to messages for conversation history
    ///
    /// Returns an assistant message containing the response content
    /// (including tool calls, reasoning, and other content if present).
    ///
    /// This is useful for building conversation history in multi-step tool calling scenarios,
    /// similar to Vercel AI SDK's `response.messages` property.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::types::{ChatResponse, ChatMessage};
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder().openai().build().await?;
    /// let mut messages = vec![ChatMessage::user("What's the weather?").build()];
    ///
    /// let (response, _) = client.chat().messages(&messages).execute().await?;
    ///
    /// // Add response messages to conversation history
    /// messages.extend(response.to_messages());
    ///
    /// // Now messages contains both user message and assistant response
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: MessageRole::Assistant,
            content: self.content.clone(),
            metadata: MessageMetadata::default(),
        }]
    }

    /// Convert response to a single assistant message
    ///
    /// This is equivalent to `to_messages()[0]` but more explicit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ChatResponse;
    ///
    /// # let response = ChatResponse::empty();
    /// let assistant_msg = response.to_assistant_message();
    /// ```
    pub fn to_assistant_message(&self) -> ChatMessage {
        ChatMessage {
            role: MessageRole::Assistant,
            content: self.content.clone(),
            metadata: MessageMetadata::default(),
        }
    }

    /// Get thinking content if available (deprecated - use reasoning() instead)
    #[deprecated(since = "0.12.0", note = "Use `reasoning()` instead")]
    pub fn has_thinking(&self) -> bool {
        self.has_reasoning()
    }

    /// Get thinking content if available (deprecated - use reasoning() instead)
    #[deprecated(since = "0.12.0", note = "Use `reasoning()` instead")]
    pub fn get_thinking(&self) -> Option<String> {
        let reasoning = self.reasoning();
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning.join("\n"))
        }
    }

    /// Get thinking content with fallback to empty string
    pub fn thinking_or_empty(&self) -> String {
        let reasoning_parts = self.reasoning();
        reasoning_parts
            .first()
            .map(|s| s.to_string())
            .unwrap_or_default()
    }

    /// Get the response ID for use in multi-turn conversations (OpenAI Responses API)
    ///
    /// This is particularly useful for OpenAI's Responses API, where you can chain
    /// conversations by passing the previous response ID to the next request.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Siumai::builder().openai().api_key("key").model("gpt-4o-mini").build().await?;
    /// // Turn 1
    /// let response1 = client.chat(vec![user!("What is Rust?")]).await?;
    /// let response_id = response1.response_id().expect("Response ID not found");
    ///
    /// // Turn 2 - automatically loads context from Turn 1
    /// let request2 = ChatRequest::new(vec![user!("Can you give me a code example?")])
    ///     .with_openai_options(
    ///         OpenAiOptions::new().with_responses_api(
    ///             ResponsesApiConfig::new()
    ///                 .with_previous_response(response_id.to_string())
    ///         )
    ///     );
    /// let response2 = client.chat_request(request2).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn response_id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Get provider-specific metadata value by provider and key
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Siumai::builder().openai().api_key("key").model("gpt-4o-mini").build().await?;
    /// let response = client.chat(vec![user!("Hello")]).await?;
    /// if let Some(value) = response.get_metadata("openai", "response_id") {
    ///     println!("OpenAI Response ID: {:?}", value);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_metadata(&self, provider: &str, key: &str) -> Option<&serde_json::Value> {
        self.provider_metadata.as_ref()?.get(provider)?.get(key)
    }

    /// Check if response has any metadata
    pub fn has_metadata(&self) -> bool {
        self.provider_metadata
            .as_ref()
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    /// Deserialize provider metadata into a typed struct.
    ///
    /// This is a provider-agnostic helper that allows provider crates (or user code)
    /// to define typed views over `provider_metadata` without requiring `siumai-core`
    /// to own provider-specific metadata types.
    pub fn provider_metadata_as<T: serde::de::DeserializeOwned>(
        &self,
        provider: &str,
    ) -> Option<T> {
        let meta = self.provider_metadata.as_ref()?.get(provider)?;
        serde_json::from_value(serde_json::to_value(meta).ok()?).ok()
    }

    // Provider-specific typed metadata helpers live in provider crates via extension traits.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_response_helper_methods() {
        // Test response_id()
        let mut response = ChatResponse::new(MessageContent::Text("Hello".to_string()));
        assert_eq!(response.response_id(), None);

        response.id = Some("resp_123".to_string());
        assert_eq!(response.response_id(), Some("resp_123"));

        // Test get_metadata() and has_metadata()
        assert!(!response.has_metadata());
        assert_eq!(response.get_metadata("openai", "key1"), None);

        // Create nested metadata structure
        let mut openai_meta = HashMap::new();
        openai_meta.insert("key1".to_string(), serde_json::json!("value1"));
        openai_meta.insert("key2".to_string(), serde_json::json!(42));

        let mut provider_metadata = HashMap::new();
        provider_metadata.insert("openai".to_string(), openai_meta);
        response.provider_metadata = Some(provider_metadata);

        assert!(response.has_metadata());
        assert_eq!(
            response.get_metadata("openai", "key1"),
            Some(&serde_json::json!("value1"))
        );
        assert_eq!(
            response.get_metadata("openai", "key2"),
            Some(&serde_json::json!(42))
        );
        assert_eq!(response.get_metadata("openai", "nonexistent"), None);
        assert_eq!(response.get_metadata("anthropic", "key1"), None);
    }
}
