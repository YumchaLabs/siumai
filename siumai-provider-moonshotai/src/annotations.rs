//! Moonshot AI-owned durable message annotations.

use serde::{Deserialize, Serialize};
use siumai_core::{MessageAnnotationTarget, TypedProviderAnnotation};
use siumai_protocol_openai::chat_completions::API_MODE_ID;

/// Mark the final assistant message as the text prefix for Kimi Partial Mode.
///
/// The annotation belongs to the semantic assistant node instead of a request-wide numeric index.
/// The Moonshot AI codec validates that it appears exactly once, on the final assistant message,
/// and projects it to Kimi's `partial: true` wire field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiAssistantPartial {}

impl KimiAssistantPartial {
    pub const fn new() -> Self {
        Self {}
    }
}

impl TypedProviderAnnotation for KimiAssistantPartial {
    type Target = MessageAnnotationTarget;

    const NAMESPACE: &'static str = "moonshotai";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}
