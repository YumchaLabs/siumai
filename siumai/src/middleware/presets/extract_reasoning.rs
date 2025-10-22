//! Extract reasoning/thinking content from LLM responses.
//!
//! This middleware extracts reasoning content from various tag formats used by
//! different LLM providers (e.g., `<think>`, `<thought>`, `<reasoning>`).

use crate::LlmError;
use crate::middleware::{LanguageModelMiddleware, TagConfig, TagExtractor};
use crate::types::{ChatRequest, ChatResponse, MessageContent};

/// Preset reasoning tag configurations for different models.
pub struct ReasoningTagPresets;

impl ReasoningTagPresets {
    /// `<think>...</think>` tag (default, used by DeepSeek, Qwen, etc.)
    pub fn think() -> TagConfig {
        TagConfig::new("<think>", "</think>").with_separator("\n")
    }

    /// `<thought>...</thought>` tag (used by Gemini)
    pub fn thought() -> TagConfig {
        TagConfig::new("<thought>", "</thought>").with_separator("\n")
    }

    /// `<reasoning>...</reasoning>` tag (used by some OpenAI models)
    pub fn reasoning() -> TagConfig {
        TagConfig::new("<reasoning>", "</reasoning>").with_separator("\n")
    }

    /// `<seed:think>...</seed:think>` tag (used by Seed models)
    pub fn seed_think() -> TagConfig {
        TagConfig::new("<seed:think>", "</seed:think>").with_separator("\n")
    }

    /// `<thinking>...</thinking>` tag (generic)
    pub fn thinking() -> TagConfig {
        TagConfig::new("<thinking>", "</thinking>").with_separator("\n")
    }

    /// Get appropriate tag configuration based on model ID.
    ///
    /// This automatically selects the best tag format based on the model name.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier (e.g., "gemini-2.5-pro", "qwen-3")
    ///
    /// # Returns
    ///
    /// The appropriate tag configuration for the model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ReasoningTagPresets::for_model("gemini-2.5-pro");
    /// // Returns TagConfig for <thought>...</thought>
    /// ```
    pub fn for_model(model_id: &str) -> TagConfig {
        let model_lower = model_id.to_lowercase();

        if model_lower.contains("gemini") {
            Self::thought()
        } else if model_lower.contains("qwen") {
            Self::think()
        } else if model_lower.contains("seed-oss") || model_lower.contains("seed_oss") {
            Self::seed_think()
        } else if model_lower.contains("gpt-oss") || model_lower.contains("gpt_oss") {
            Self::reasoning()
        } else {
            // Default to <think> for most models
            Self::think()
        }
    }
}

/// Configuration for the extract reasoning middleware.
#[derive(Debug, Clone)]
pub struct ExtractReasoningConfig {
    /// Tag configuration to use for extraction
    pub tag_config: TagConfig,
    /// Whether to remove the tag content from the response text
    pub remove_from_text: bool,
}

impl Default for ExtractReasoningConfig {
    fn default() -> Self {
        Self {
            tag_config: ReasoningTagPresets::think(),
            remove_from_text: true,
        }
    }
}

impl ExtractReasoningConfig {
    /// Create a new configuration with the given tag config.
    pub fn new(tag_config: TagConfig) -> Self {
        Self {
            tag_config,
            remove_from_text: true,
        }
    }

    /// Set whether to remove tag content from the response text.
    pub fn with_remove_from_text(mut self, remove: bool) -> Self {
        self.remove_from_text = remove;
        self
    }
}

/// Middleware for extracting reasoning/thinking content from LLM responses.
///
/// This middleware extracts reasoning content using a three-layer fallback strategy:
///
/// 1. **Provider-extracted**: Check if the provider already extracted thinking content
///    into the `response.thinking` field
/// 2. **Metadata**: Check if thinking content is in `response.metadata["thinking"]`
///    (used by some providers like Anthropic)
/// 3. **Tag extraction**: Extract from response content using XML-style tags
///    (e.g., `<think>...</think>`)
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::presets::ExtractReasoningMiddleware;
/// use std::sync::Arc;
///
/// // Use default configuration (<think> tags)
/// let middleware = Arc::new(ExtractReasoningMiddleware::default());
///
/// // Use custom tag
/// let middleware = Arc::new(ExtractReasoningMiddleware::with_tag(
///     TagConfig::new("<thought>", "</thought>")
/// ));
///
/// // Auto-select based on model
/// let middleware = Arc::new(ExtractReasoningMiddleware::for_model("gemini-2.5-pro"));
/// ```
pub struct ExtractReasoningMiddleware {
    config: ExtractReasoningConfig,
}

impl ExtractReasoningMiddleware {
    /// Create a new middleware with the given configuration.
    pub fn new(config: ExtractReasoningConfig) -> Self {
        Self { config }
    }

    /// Create a middleware with a specific tag configuration.
    ///
    /// # Arguments
    ///
    /// * `tag_config` - The tag configuration to use
    pub fn with_tag(tag_config: TagConfig) -> Self {
        Self {
            config: ExtractReasoningConfig::new(tag_config),
        }
    }

    /// Create a middleware that auto-selects tags based on model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let middleware = ExtractReasoningMiddleware::for_model("gemini-2.5-pro");
    /// ```
    pub fn for_model(model_id: &str) -> Self {
        Self {
            config: ExtractReasoningConfig::new(ReasoningTagPresets::for_model(model_id)),
        }
    }
}

impl Default for ExtractReasoningMiddleware {
    fn default() -> Self {
        Self {
            config: ExtractReasoningConfig::default(),
        }
    }
}

impl LanguageModelMiddleware for ExtractReasoningMiddleware {
    fn post_generate(
        &self,
        _req: &ChatRequest,
        mut resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        use crate::types::ContentPart;

        // 1. Priority: check if reasoning already exists in content
        if resp.has_reasoning() {
            return Ok(resp);
        }

        // 2. Extract from metadata (Anthropic etc.)
        if let Some(thinking_value) = resp.metadata.get("thinking") {
            if let Some(thinking_str) = thinking_value.as_str() {
                // Add reasoning to content
                let mut parts = match &resp.content {
                    MessageContent::Text(text) if !text.is_empty() => {
                        vec![ContentPart::text(text)]
                    }
                    MessageContent::MultiModal(parts) => parts.clone(),
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(v) => {
                        vec![ContentPart::text(
                            &serde_json::to_string(v).unwrap_or_default(),
                        )]
                    }
                    _ => vec![],
                };
                parts.push(ContentPart::reasoning(thinking_str));
                resp.content = MessageContent::MultiModal(parts);
                return Ok(resp);
            }
        }

        // 3. Extract from content using TagExtractor
        if let Some(text) = resp.content.text() {
            let mut extractor = TagExtractor::new(self.config.tag_config.clone());
            let results = extractor.process_text(text);

            // Find complete extraction
            for result in &results {
                if result.complete {
                    if let Some(thinking) = &result.tag_content_extracted {
                        let thinking_trimmed = thinking.trim().to_string();

                        // Build new content
                        let mut parts = Vec::new();

                        // Remove from text if configured
                        if self.config.remove_from_text {
                            // Collect all non-tag content
                            let clean_text: String = results
                                .iter()
                                .filter(|r| !r.is_tag_content && !r.complete)
                                .map(|r| r.content.as_str())
                                .collect();

                            let trimmed = clean_text.trim();
                            if !trimmed.is_empty() {
                                parts.push(ContentPart::text(trimmed));
                            }
                        } else {
                            parts.push(ContentPart::text(text));
                        }

                        // Add reasoning
                        parts.push(ContentPart::reasoning(&thinking_trimmed));

                        resp.content = if parts.len() == 1 && parts[0].is_text() {
                            MessageContent::Text(text.to_string())
                        } else {
                            MessageContent::MultiModal(parts)
                        };
                        break;
                    }
                }
            }

            // Check for any remaining content in finalize
            if !resp.has_reasoning() {
                if let Some(final_result) = extractor.finalize() {
                    if let Some(thinking) = final_result.tag_content_extracted {
                        let mut parts = match &resp.content {
                            MessageContent::Text(text) if !text.is_empty() => {
                                vec![ContentPart::text(text)]
                            }
                            MessageContent::MultiModal(parts) => parts.clone(),
                            _ => vec![],
                        };
                        parts.push(ContentPart::reasoning(thinking.trim()));
                        resp.content = MessageContent::MultiModal(parts);
                    }
                }
            }
        }

        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, Usage};

    fn create_test_response(content: &str) -> ChatResponse {
        ChatResponse {
            id: Some("test".to_string()),
            content: MessageContent::Text(content.to_string()),
            model: Some("test-model".to_string()),
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                cached_tokens: None,
                reasoning_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            finish_reason: Some(FinishReason::Stop),
            system_fingerprint: None,
            service_tier: None,
            audio: None,
            warnings: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_extract_thinking_with_think_tags() {
        let middleware = ExtractReasoningMiddleware::default();
        let req = ChatRequest::default();
        let resp = create_test_response("Hello <think>This is thinking</think> World");

        let result = middleware.post_generate(&req, resp).unwrap();

        // Check reasoning was extracted
        let reasoning = result.reasoning();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0], "This is thinking");
        // Note: separator (\n) is added between tag content and regular text
        assert_eq!(result.content.text(), Some("Hello \n World"));
    }

    #[test]
    fn test_extract_thinking_with_thought_tags() {
        let middleware = ExtractReasoningMiddleware::with_tag(ReasoningTagPresets::thought());
        let req = ChatRequest::default();
        let resp = create_test_response("Hello <thought>This is thinking</thought> World");

        let result = middleware.post_generate(&req, resp).unwrap();

        // Check reasoning was extracted
        let reasoning = result.reasoning();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0], "This is thinking");
        // Note: separator (\n) is added between tag content and regular text
        assert_eq!(result.content.text(), Some("Hello \n World"));
    }

    #[test]
    fn test_no_thinking_tags() {
        let middleware = ExtractReasoningMiddleware::default();
        let req = ChatRequest::default();
        let resp = create_test_response("Hello World");

        let result = middleware.post_generate(&req, resp).unwrap();

        // No reasoning should be extracted
        assert_eq!(result.reasoning().len(), 0);
        assert_eq!(result.content.text(), Some("Hello World"));
    }

    #[test]
    fn test_provider_already_extracted() {
        use crate::types::{ContentPart, MessageContent};

        let middleware = ExtractReasoningMiddleware::default();
        let req = ChatRequest::default();
        let mut resp = create_test_response("Hello World");

        // Simulate provider already extracted reasoning
        resp.content = MessageContent::MultiModal(vec![
            ContentPart::text("Hello World"),
            ContentPart::reasoning("Provider extracted"),
        ]);

        let result = middleware.post_generate(&req, resp).unwrap();

        // Should keep provider's extraction
        let reasoning = result.reasoning();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0], "Provider extracted");
    }

    #[test]
    fn test_for_model_gemini() {
        let config = ReasoningTagPresets::for_model("gemini-2.5-pro");
        assert_eq!(config.opening_tag, "<thought>");
        assert_eq!(config.closing_tag, "</thought>");
    }

    #[test]
    fn test_for_model_qwen() {
        let config = ReasoningTagPresets::for_model("qwen-3-turbo");
        assert_eq!(config.opening_tag, "<think>");
        assert_eq!(config.closing_tag, "</think>");
    }

    #[test]
    fn test_for_model_default() {
        let config = ReasoningTagPresets::for_model("unknown-model");
        assert_eq!(config.opening_tag, "<think>");
        assert_eq!(config.closing_tag, "</think>");
    }
}
