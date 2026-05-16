//! Extract reasoning/thinking content from LLM responses.
//!
//! This middleware extracts reasoning content from provider-agnostic tag and metadata shapes.

use crate::LlmError;
use crate::execution::middleware::{LanguageModelMiddleware, TagConfig, TagExtractor};
use crate::types::{ChatRequest, ChatResponse, MessageContent};

/// Preset reasoning tag configurations for different models.
pub struct ReasoningTagPresets;

impl ReasoningTagPresets {
    /// `<think>...</think>` tag.
    pub fn think() -> TagConfig {
        TagConfig::new("<think>", "</think>").with_separator("\n")
    }

    /// `<thought>...</thought>` tag.
    pub fn thought() -> TagConfig {
        TagConfig::new("<thought>", "</thought>").with_separator("\n")
    }

    /// `<reasoning>...</reasoning>` tag.
    pub fn reasoning() -> TagConfig {
        TagConfig::new("<reasoning>", "</reasoning>").with_separator("\n")
    }

    /// `<seed:think>...</seed:think>` tag.
    pub fn seed_think() -> TagConfig {
        TagConfig::new("<seed:think>", "</seed:think>").with_separator("\n")
    }

    /// `<thinking>...</thinking>` tag (generic)
    pub fn thinking() -> TagConfig {
        TagConfig::new("<thinking>", "</thinking>").with_separator("\n")
    }

    /// Return the provider-agnostic default tag configuration for a model ID.
    ///
    /// Provider-specific tag routing belongs in provider/facade extension code. Core keeps this
    /// helper as a stable fallback and intentionally does not inspect concrete provider names.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier.
    ///
    /// # Returns
    ///
    /// The appropriate tag configuration for the model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ReasoningTagPresets::for_model("model-id");
    /// // Returns the provider-agnostic default <think> tag config.
    /// ```
    pub fn for_model(_model_id: &str) -> TagConfig {
        Self::think()
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
/// 1. **Provider-extracted**: Check if the provider already extracted reasoning content
///    into the `response.thinking` field
/// 2. **Metadata**: Check if reasoning content is in provider metadata under generic
///    `thinking` or `reasoning` keys
/// 3. **Tag extraction**: Extract from response content using XML-style tags
///    (e.g., `<think>...</think>`)
///
/// # Example
///
/// ```rust,ignore
/// use siumai::experimental::execution::middleware::presets::ExtractReasoningMiddleware;
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
/// // Use the provider-agnostic default for a model ID
/// let middleware = Arc::new(ExtractReasoningMiddleware::for_model("model-id"));
/// ```
#[derive(Default)]
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

    /// Create a middleware that uses the provider-agnostic default for a model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let middleware = ExtractReasoningMiddleware::for_model("model-id");
    /// ```
    pub fn for_model(model_id: &str) -> Self {
        Self {
            config: ExtractReasoningConfig::new(ReasoningTagPresets::for_model(model_id)),
        }
    }
}

// Default is derived

fn reasoning_metadata_text(resp: &ChatResponse) -> Option<&str> {
    resp.provider_metadata
        .as_ref()?
        .values()
        .find_map(|metadata| {
            metadata
                .get("thinking")
                .or_else(|| metadata.get("reasoning"))
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
}

impl LanguageModelMiddleware for ExtractReasoningMiddleware {
    #[allow(clippy::collapsible_if)]
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

        // 2. Extract from generic provider metadata keys.
        if let Some(thinking_str) = reasoning_metadata_text(&resp) {
            let thinking = thinking_str.to_string();
            // Add reasoning to content
            let mut parts = match &resp.content {
                MessageContent::Text(text) if !text.is_empty() => {
                    vec![ContentPart::text(text)]
                }
                MessageContent::MultiModal(parts) => parts.clone(),
                #[cfg(feature = "structured-messages")]
                MessageContent::Json(v) => {
                    vec![ContentPart::text(
                        serde_json::to_string(v).unwrap_or_default(),
                    )]
                }
                _ => vec![],
            };
            parts.push(ContentPart::reasoning(thinking));
            resp.content = MessageContent::MultiModal(parts);
            return Ok(resp);
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

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    fn create_test_response(content: &str) -> ChatResponse {
        ChatResponse {
            id: Some("test".to_string()),
            content: MessageContent::Text(content.to_string()),
            model: Some("test-model".to_string()),
            usage: Some(Usage::new(10, 20)),
            finish_reason: Some(FinishReason::Stop),
            raw_finish_reason: None,
            system_fingerprint: None,
            service_tier: None,
            audio: None,
            warnings: None,
            request: None,
            response: None,
            provider_metadata: None,
        }
    }

    #[test]
    fn extract_reasoning_middleware_source_stays_provider_agnostic() {
        let source = include_str!("extract_reasoning.rs");
        let production_source =
            source_section(source, "pub struct ReasoningTagPresets", "#[cfg(test)]");

        let disallowed = [
            format!("\"{}\"", ["an", "thropic"].concat()),
            format!("\"{}\"", ["ge", "mini"].concat()),
            format!("\"{}\"", ["qw", "en"].concat()),
            format!("\"{}-oss\"", ["gp", "t"].concat()),
            format!("\"{}_oss\"", ["gp", "t"].concat()),
            format!("\"{}-oss\"", ["se", "ed"].concat()),
            format!("\"{}_oss\"", ["se", "ed"].concat()),
            ["Deep", "Seek"].concat(),
            ["An", "thropic"].concat(),
            ["Ge", "mini"].concat(),
            ["Open", "AI"].concat(),
        ];

        for disallowed in disallowed {
            assert!(
                !production_source.contains(&disallowed),
                "core reasoning extraction middleware must stay provider-agnostic"
            );
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
    fn test_extract_thinking_from_generic_provider_metadata() {
        let middleware = ExtractReasoningMiddleware::default();
        let req = ChatRequest::default();
        let mut resp = create_test_response("Hello World");
        resp.provider_metadata = Some(std::collections::HashMap::from([(
            "provider".to_string(),
            serde_json::json!({
                "thinking": "metadata thinking"
            }),
        )]));

        let result = middleware.post_generate(&req, resp).unwrap();

        let reasoning = result.reasoning();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0], "metadata thinking");
    }

    #[test]
    fn test_for_model_uses_provider_agnostic_default() {
        let config = ReasoningTagPresets::for_model("provider-model-id");
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
