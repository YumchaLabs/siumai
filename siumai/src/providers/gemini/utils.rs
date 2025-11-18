//! Gemini provider utility helpers.
//!
//! This module bridges the std-gemini core content model
//! (`GeminiParsedContentCore`) with the aggregator's `MessageContent`
//! representation. It mirrors the Anthropic utilities so that both
//! providers share a consistent content pipeline:
//! - std crate parses provider JSON into a core content struct
//! - aggregator converts the core struct into `ContentPart`s

use crate::types::ContentPart;
use siumai_std_gemini::gemini::utils::{GeminiParsedContentCore, GeminiToolCallCore};

/// Internal representation of Gemini content parts built from the core model.
///
/// This keeps the aggregated assistant text alongside the multimodal parts
/// so callers can decide whether to collapse into `MessageContent::Text`
/// or keep a multimodal structure.
#[derive(Debug, Clone)]
pub struct GeminiContentParts {
    /// Aggregated assistant text (excluding thinking parts).
    pub text: String,
    /// Multimodal content parts (text, tool calls, reasoning, media, etc.).
    pub parts: Vec<ContentPart>,
}

/// Convert core-level `GeminiParsedContentCore` into the aggregator's
/// content parts (text + tool calls + reasoning).
///
/// Media parts (images, audio, files) are handled separately because they
/// are not represented in the core content model.
pub fn core_parsed_to_content_parts(parsed: &GeminiParsedContentCore) -> GeminiContentParts {
    let mut parts = Vec::new();

    // Primary assistant text content
    if !parsed.text.is_empty() {
        parts.push(ContentPart::text(&parsed.text));
    }

    // Tool calls
    for GeminiToolCallCore { name, arguments } in &parsed.tool_calls {
        // Gemini tool calls do not provide a stable ID in the core model,
        // so we generate one here (same behavior as the legacy transformer).
        let id = format!("call_{}", uuid::Uuid::new_v4());
        parts.push(ContentPart::tool_call(
            id,
            name.clone(),
            arguments.clone(),
            None,
        ));
    }

    // Thinking / reasoning content
    if let Some(thinking) = &parsed.thinking {
        if !thinking.is_empty() {
            parts.push(ContentPart::reasoning(thinking));
        }
    }

    GeminiContentParts {
        text: parsed.text.clone(),
        parts,
    }
}
