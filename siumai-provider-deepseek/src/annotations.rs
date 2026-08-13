//! DeepSeek-owned durable message annotations.

use serde::{Deserialize, Serialize};
use siumai_core::{MessageAnnotationTarget, TypedProviderAnnotation};
use siumai_protocol_openai::chat_completions::API_MODE_ID;

/// Mark the final assistant message as a DeepSeek beta prefix-completion seed.
///
/// The beta Chat handle requires this marker to appear on exactly the final message, and that
/// message must have the assistant role. Stable Chat handles reject the marker before transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepSeekAssistantPrefix {}

impl DeepSeekAssistantPrefix {
    pub const fn new() -> Self {
        Self {}
    }
}

impl TypedProviderAnnotation for DeepSeekAssistantPrefix {
    type Target = MessageAnnotationTarget;

    const NAMESPACE: &'static str = "deepseek";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}
