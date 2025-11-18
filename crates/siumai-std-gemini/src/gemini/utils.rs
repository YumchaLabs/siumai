//! Gemini standard helper functions (provider-agnostic).
//!
//! Built on top of `siumai-core` Chat abstractions, this module provides
//! a minimal, unified content model for Gemini responses that can be
//! reused across non-streaming and streaming paths.
//!
//! The goal is similar to the Anthropic standard: expose a provider-
//! independent structure that captures:
//! - aggregated assistant text
//! - tool calls (functionCall)
//! - thinking/thought content (for reasoning models)

/// Core representation of a single Gemini tool call.
#[derive(Debug, Clone)]
pub struct GeminiToolCallCore {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Core representation of Gemini message content.
///
/// This mirrors the unified content model used by higher-level layers:
/// - `text`: aggregated assistant text (excluding thinking parts)
/// - `tool_calls`: functionCall parts with JSON arguments
/// - `thinking`: optional concatenated thinking/thought content
#[derive(Debug, Clone, Default)]
pub struct GeminiParsedContentCore {
    pub text: String,
    pub tool_calls: Vec<GeminiToolCallCore>,
    pub thinking: Option<String>,
}

/// Parse a Gemini chat response JSON into the core content structure.
///
/// This helper operates on the raw JSON shape returned by Gemini:
///
/// ```json
/// {
///   "candidates": [
///     {
///       "content": {
///         "parts": [
///           { "text": "...", "thought": true|false },
///           { "functionCall": { "name": "...", "args": { ... } } },
///           ...
///         ]
///       }
///     }
///   ]
/// }
/// ```
pub fn parse_content_core(resp: &serde_json::Value) -> GeminiParsedContentCore {
    let mut parsed = GeminiParsedContentCore::default();

    // Navigate to candidates[0].content.parts
    let parts = resp
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|cand| cand.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array());

    if let Some(parts) = parts {
        for part in parts {
            // Text (with optional thought flag)
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                let is_thought = part
                    .get("thought")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);

                if is_thought {
                    // Aggregate thinking content
                    if !text.is_empty() {
                        match &mut parsed.thinking {
                            Some(acc) => {
                                if !acc.is_empty() {
                                    acc.push('\n');
                                }
                                acc.push_str(text);
                            }
                            None => parsed.thinking = Some(text.to_string()),
                        }
                    }
                } else if !text.is_empty() {
                    // Aggregate main assistant text (non-thinking)
                    if !parsed.text.is_empty() {
                        parsed.text.push('\n');
                    }
                    parsed.text.push_str(text);
                }
            }

            // Function calls (tool calls)
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = fc
                    .get("args")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));

                if !name.is_empty() {
                    parsed
                        .tool_calls
                        .push(GeminiToolCallCore { name, arguments });
                }
            }
        }
    }

    parsed
}
