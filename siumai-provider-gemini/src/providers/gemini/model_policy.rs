//! Curated, provider-owned Gemini model policies.
//!
//! Google keeps unknown model identifiers callable so newly released models do not require a
//! client release. Capability and limit claims are intentionally stricter: only exact, audited
//! identifiers receive a policy.

use serde_json::Value;

use super::model_constants::{gemini_2_5_flash, gemini_2_5_flash_lite, gemini_2_5_pro, gemini_3};
use super::types::GenerateContentRequest;

/// Primary official source for the curated catalog.
pub const MODEL_CATALOG_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/models";
/// Date on which the curated catalog was checked against the official source.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-05";

/// Whether a curated model policy supports a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupport {
    /// The audited policy explicitly supports the capability.
    Supported,
    /// The audited policy explicitly does not support the capability.
    Unsupported,
    /// The model or capability name is not covered by the audited policy table.
    Unknown,
}

/// How the provider must encode legacy sampling controls for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingParameterPolicy {
    /// Preserve `temperature`, `topP`, and `topK` in the request.
    Supported,
    /// Omit `temperature`, `topP`, and `topK` from the wire payload.
    Omit,
}

/// Audited facts for an exact Gemini model identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeminiModelPolicy {
    /// Maximum accepted input tokens when documented by Google.
    pub context_window: Option<u32>,
    /// Maximum generated output tokens when documented by Google.
    pub max_output_tokens: Option<u32>,
    /// Explicit provider capabilities supported by this model.
    pub capabilities: &'static [&'static str],
    /// Model-specific sampling parameter behavior.
    pub sampling_parameters: SamplingParameterPolicy,
}

const KNOWN_CAPABILITIES: &[&str] = &[
    "chat",
    "streaming",
    "vision",
    "audio_input",
    "video_input",
    "pdf_input",
    "function_calling",
    "code_execution",
    "thinking",
    "prompt_caching",
    "file_search",
    "search_grounding",
    "maps_grounding",
    "structured_outputs",
    "url_context",
    "computer_use",
    "image_generation",
    "audio_generation",
    "live_api",
];

const GENERAL_CAPABILITIES: &[&str] = &[
    "chat",
    "streaming",
    "vision",
    "audio_input",
    "video_input",
    "pdf_input",
    "function_calling",
    "code_execution",
    "thinking",
    "prompt_caching",
    "file_search",
    "search_grounding",
    "maps_grounding",
    "structured_outputs",
    "url_context",
];

const GENERAL_CAPABILITIES_WITH_COMPUTER_USE: &[&str] = &[
    "chat",
    "streaming",
    "vision",
    "audio_input",
    "video_input",
    "pdf_input",
    "function_calling",
    "code_execution",
    "thinking",
    "prompt_caching",
    "file_search",
    "search_grounding",
    "maps_grounding",
    "structured_outputs",
    "url_context",
    "computer_use",
];

const IMAGE_CAPABILITIES: &[&str] = &["vision", "thinking", "image_generation"];
const IMAGE_CAPABILITIES_WITH_SEARCH: &[&str] =
    &["vision", "thinking", "search_grounding", "image_generation"];
const LIVE_CAPABILITIES: &[&str] = &[
    "vision",
    "audio_input",
    "video_input",
    "function_calling",
    "audio_generation",
    "live_api",
];
const TTS_CAPABILITIES: &[&str] = &["audio_generation"];

const GEMINI_3_6_FLASH_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES_WITH_COMPUTER_USE,
    sampling_parameters: SamplingParameterPolicy::Omit,
};

const GEMINI_3_5_FLASH_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES_WITH_COMPUTER_USE,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_5_FLASH_LITE_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Omit,
};

const GEMINI_3_1_PRO_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES_WITH_COMPUTER_USE,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_1_FLASH_LITE_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_1_FLASH_IMAGE_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(65_536),
    max_output_tokens: Some(32_768),
    capabilities: IMAGE_CAPABILITIES_WITH_SEARCH,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_1_FLASH_LITE_IMAGE_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(32_768),
    max_output_tokens: Some(4_096),
    capabilities: IMAGE_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_1_FLASH_LIVE_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(131_072),
    max_output_tokens: Some(8_192),
    capabilities: LIVE_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_3_1_FLASH_TTS_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: None,
    max_output_tokens: None,
    capabilities: TTS_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

const GEMINI_2_5_POLICY: GeminiModelPolicy = GeminiModelPolicy {
    context_window: Some(1_048_576),
    max_output_tokens: Some(65_536),
    capabilities: GENERAL_CAPABILITIES,
    sampling_parameters: SamplingParameterPolicy::Supported,
};

fn normalize_model_id(model_id: &str) -> &str {
    model_id
        .trim()
        .strip_prefix("models/")
        .unwrap_or(model_id.trim())
}

/// Return the audited policy for an exact model identifier.
pub fn model_policy(model_id: &str) -> Option<&'static GeminiModelPolicy> {
    match normalize_model_id(model_id) {
        gemini_3::GEMINI_3_6_FLASH => Some(&GEMINI_3_6_FLASH_POLICY),
        gemini_3::GEMINI_3_5_FLASH => Some(&GEMINI_3_5_FLASH_POLICY),
        gemini_3::GEMINI_3_5_FLASH_LITE => Some(&GEMINI_3_5_FLASH_LITE_POLICY),
        gemini_3::GEMINI_3_1_PRO_PREVIEW => Some(&GEMINI_3_1_PRO_POLICY),
        gemini_3::GEMINI_3_1_FLASH_LITE => Some(&GEMINI_3_1_FLASH_LITE_POLICY),
        gemini_3::GEMINI_3_1_FLASH_IMAGE => Some(&GEMINI_3_1_FLASH_IMAGE_POLICY),
        gemini_3::GEMINI_3_1_FLASH_LITE_IMAGE => Some(&GEMINI_3_1_FLASH_LITE_IMAGE_POLICY),
        gemini_3::GEMINI_3_1_FLASH_LIVE_PREVIEW => Some(&GEMINI_3_1_FLASH_LIVE_POLICY),
        gemini_3::GEMINI_3_1_FLASH_TTS_PREVIEW => Some(&GEMINI_3_1_FLASH_TTS_POLICY),
        gemini_2_5_pro::GEMINI_2_5_PRO
        | gemini_2_5_flash::GEMINI_2_5_FLASH
        | gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE => Some(&GEMINI_2_5_POLICY),
        _ => None,
    }
}

/// Return explicit capability support without guessing from a model name.
pub fn model_capability_support(model_id: &str, capability: &str) -> CapabilitySupport {
    if !KNOWN_CAPABILITIES.contains(&capability) {
        return CapabilitySupport::Unknown;
    }

    match model_policy(model_id) {
        Some(policy) if policy.capabilities.contains(&capability) => CapabilitySupport::Supported,
        Some(_) => CapabilitySupport::Unsupported,
        None => CapabilitySupport::Unknown,
    }
}

/// Return whether legacy sampling controls must be omitted for this exact model.
pub(crate) fn omits_sampling_parameters(model_id: &str) -> bool {
    model_policy(model_id)
        .is_some_and(|policy| policy.sampling_parameters == SamplingParameterPolicy::Omit)
}

/// Remove sampling controls that Google has deprecated for the selected model.
pub(crate) fn sanitize_sampling_parameters(model_id: &str, body: &mut Value) {
    if !omits_sampling_parameters(model_id) {
        return;
    }

    if let Some(object) = body.as_object_mut() {
        object.remove("temperature");
        object.remove("topP");
        object.remove("top_p");
        object.remove("topK");
        object.remove("top_k");
    }

    if let Some(generation_config) = body
        .get_mut("generationConfig")
        .and_then(Value::as_object_mut)
    {
        generation_config.remove("temperature");
        generation_config.remove("topP");
        generation_config.remove("topK");
    }
}

/// Apply the same sampling policy to the typed request compatibility surface.
pub(crate) fn sanitize_generate_content_request(
    model_id: &str,
    request: &mut GenerateContentRequest,
) {
    if !omits_sampling_parameters(model_id) {
        return;
    }

    if let Some(generation_config) = request.generation_config.as_mut() {
        generation_config.temperature = None;
        generation_config.top_p = None;
        generation_config.top_k = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_policies_do_not_guess_for_unknown_models() {
        assert_eq!(model_policy("gemini-4-future"), None);
        assert_eq!(
            model_capability_support("gemini-4-future", "vision"),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            model_capability_support(gemini_3::GEMINI_3_6_FLASH, "future_capability"),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn sampling_policy_is_exact_and_model_specific() {
        assert!(omits_sampling_parameters(gemini_3::GEMINI_3_6_FLASH));
        assert!(omits_sampling_parameters(gemini_3::GEMINI_3_5_FLASH_LITE));
        assert!(!omits_sampling_parameters(gemini_3::GEMINI_3_5_FLASH));
        assert!(!omits_sampling_parameters("gemini-flash-latest"));
    }

    #[test]
    fn sanitizer_removes_only_deprecated_sampling_fields() {
        let mut body = serde_json::json!({
            "temperature": 0.4,
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 1024
            }
        });

        sanitize_sampling_parameters(gemini_3::GEMINI_3_6_FLASH, &mut body);

        assert!(body.get("temperature").is_none());
        assert!(body["generationConfig"].get("temperature").is_none());
        assert!(body["generationConfig"].get("topP").is_none());
        assert!(body["generationConfig"].get("topK").is_none());
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 1024);
    }
}
