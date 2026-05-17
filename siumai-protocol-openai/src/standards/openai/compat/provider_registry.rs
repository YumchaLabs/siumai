//! Configuration-driven adapter helpers for OpenAI-compatible providers.
//!
//! This module is protocol-owned and can be reused by any OpenAI-like provider crate that needs a
//! lightweight, configuration-based `ProviderAdapter` implementation.

use super::adapter::ProviderAdapter;
use super::metadata::{
    NestedProviderMetadata, extract_openai_compatible_provider_metadata,
    extract_perplexity_provider_metadata, provider_options_key,
};
use super::types::{FieldAccessor, FieldMappings, JsonFieldAccessor, RequestType};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use serde::{Deserialize, Serialize};

/// Provider configuration entry (best-effort hints, not exhaustive).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Base URL for API
    pub base_url: String,
    /// Field mappings for response parsing
    pub field_mappings: ProviderFieldMappings,
    /// Supported capabilities
    pub capabilities: Vec<String>,
    /// Default model (optional)
    pub default_model: Option<String>,
    /// Whether this provider supports reasoning/thinking
    pub supports_reasoning: bool,
    /// Environment variable name to read the API key from (optional).
    ///
    /// When not set, callers typically fall back to `${PROVIDER_ID}_API_KEY`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Additional environment variable names to try (fallbacks).
    #[serde(default)]
    pub api_key_env_aliases: Vec<String>,
}

/// Field mappings configuration (string-based, suitable for deserialization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFieldMappings {
    /// Fields that contain thinking/reasoning content (in priority order)
    pub thinking_fields: Vec<String>,
    /// Field that contains regular content
    pub content_field: String,
    /// Field that contains tool calls
    pub tool_calls_field: String,
    /// Field that contains role information
    pub role_field: String,
}

impl Default for ProviderFieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec![
                "reasoning_content".to_string(),
                "thinking".to_string(),
                "reasoning".to_string(),
            ],
            content_field: "content".to_string(),
            tool_calls_field: "tool_calls".to_string(),
            role_field: "role".to_string(),
        }
    }
}

/// Generic adapter that uses configuration.
#[derive(Debug, Clone)]
pub struct ConfigurableAdapter {
    config: ProviderConfig,
}

impl ConfigurableAdapter {
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
    }
}

pub fn provider_capabilities_declare_chat_surface(capabilities: &[String]) -> bool {
    const NON_CHAT_ONLY_CAPABILITIES: &[&str] = &[
        "embedding",
        "rerank",
        "image_generation",
        "speech",
        "transcription",
        "audio",
        "tts",
        "stt",
    ];

    if capabilities.is_empty() {
        return true;
    }

    if capabilities
        .iter()
        .any(|cap| matches!(cap.as_str(), "tools" | "vision" | "reasoning"))
    {
        return true;
    }

    !capabilities
        .iter()
        .all(|cap| NON_CHAT_ONLY_CAPABILITIES.contains(&cap.as_str()))
}

pub fn provider_config_declares_chat_surface(config: &ProviderConfig) -> bool {
    provider_capabilities_declare_chat_surface(&config.capabilities)
}

pub fn provider_config_declares_completion_surface(config: &ProviderConfig) -> bool {
    if !provider_config_declares_chat_surface(config) {
        return false;
    }

    // Some AI SDK-packaged compat providers intentionally expose only chat/language models
    // even though they reuse OpenAI-style chat-completions transport underneath.
    !matches!(config.id.as_str(), "mistral" | "perplexity" | "moonshotai")
}

impl ProviderAdapter for ConfigurableAdapter {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.config.id.clone())
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        _model: &str,
        _request_type: RequestType,
    ) -> Result<(), LlmError> {
        fn take_any(
            obj: &mut serde_json::Map<String, serde_json::Value>,
            keys: &[&str],
        ) -> Option<serde_json::Value> {
            for k in keys {
                if let Some(v) = obj.remove(*k) {
                    return Some(v);
                }
            }
            None
        }

        fn rename_field(
            obj: &mut serde_json::Map<String, serde_json::Value>,
            from: &str,
            to: &str,
        ) {
            if let Some(v) = obj.remove(from) {
                obj.entry(to.to_string()).or_insert(v);
            }
        }

        fn normalize_xai_search_parameters(v: &mut serde_json::Value) {
            let Some(obj) = v.as_object_mut() else {
                return;
            };

            rename_field(obj, "returnCitations", "return_citations");
            rename_field(obj, "maxSearchResults", "max_search_results");
            rename_field(obj, "fromDate", "from_date");
            rename_field(obj, "toDate", "to_date");

            rename_field(obj, "searchParameters", "search_parameters");

            if let Some(arr) = obj.get_mut("sources").and_then(|v| v.as_array_mut()) {
                for src in arr {
                    let Some(src_obj) = src.as_object_mut() else {
                        continue;
                    };

                    rename_field(src_obj, "allowedWebsites", "allowed_websites");
                    rename_field(src_obj, "excludedWebsites", "excluded_websites");
                    rename_field(src_obj, "safeSearch", "safe_search");

                    rename_field(src_obj, "excludedXHandles", "excluded_x_handles");
                    rename_field(src_obj, "includedXHandles", "included_x_handles");
                    rename_field(src_obj, "postFavoriteCount", "post_favorite_count");
                    rename_field(src_obj, "postViewCount", "post_view_count");
                    rename_field(src_obj, "xHandles", "x_handles");
                }
            }
        }

        fn normalize_moonshot_thinking(v: &mut serde_json::Value) {
            let Some(obj) = v.as_object_mut() else {
                return;
            };

            rename_field(obj, "budgetTokens", "budget_tokens");
        }

        fn normalize_deepseek_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
            let legacy_enable = take_any(obj, &["enableReasoning", "enable_reasoning"])
                .and_then(|value| value.as_bool());

            obj.remove("reasoningBudget");
            obj.remove("reasoning_budget");

            if let Some(thinking) = obj
                .get_mut("thinking")
                .and_then(|value| value.as_object_mut())
            {
                rename_field(thinking, "thinkingType", "type");
                rename_field(thinking, "thinking_type", "type");
                thinking.remove("budgetTokens");
                thinking.remove("budget_tokens");
            } else if let Some(enable) = legacy_enable {
                obj.insert(
                    "thinking".to_string(),
                    serde_json::json!({
                        "type": if enable { "enabled" } else { "disabled" }
                    }),
                );
            }
        }

        fn normalize_alibaba_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
            rename_field(obj, "enableThinking", "enable_thinking");
            rename_field(obj, "thinkingBudget", "thinking_budget");
            rename_field(obj, "parallelToolCalls", "parallel_tool_calls");
        }

        // Most OpenAI-compatible providers don't need parameter transformation.
        //
        // For a small set of vendors, we centralize well-known OpenAI-compat quirks here
        // to keep provider crates thin and avoid copy/paste drift.
        match self.config.id.as_str() {
            // Groq is OpenAI-compatible but has a few request differences:
            // - "developer" role is not supported (treat as "system")
            // - "stream_options" is not supported (omit it)
            // - "max_completion_tokens" is not supported (use "max_tokens")
            "groq" => {
                if let Some(msgs) = params.get_mut("messages").and_then(|v| v.as_array_mut()) {
                    for m in msgs {
                        if m.get("role").and_then(|v| v.as_str()) == Some("developer") {
                            m["role"] = serde_json::Value::String("system".to_string());
                        }
                    }
                }

                if let Some(obj) = params.as_object_mut() {
                    obj.remove("stream_options");
                    if let Some(v) = obj.remove("max_completion_tokens") {
                        obj.entry("max_tokens".to_string()).or_insert(v);
                    }
                }
            }
            // xAI (Grok) Chat Completions quirks (Vercel-aligned):
            // - stop sequences are not supported (omit "stop")
            // - accept Vercel-style camelCase provider options and normalize to snake_case
            "xai" => {
                if let Some(obj) = params.as_object_mut() {
                    obj.remove("stop");

                    if let Some(v) = take_any(obj, &["reasoningEffort", "reasoning_effort"]) {
                        obj.entry("reasoning_effort".to_string()).or_insert(v);
                    }

                    let legacy_reasoning_enabled = take_any(
                        obj,
                        &[
                            "enableReasoning",
                            "enable_reasoning",
                            "enableThinking",
                            "enable_thinking",
                        ],
                    )
                    .and_then(|value| value.as_bool());
                    for key in [
                        "enableReasoning",
                        "enable_reasoning",
                        "enableThinking",
                        "enable_thinking",
                    ] {
                        obj.remove(key);
                    }
                    let legacy_reasoning_budget = take_any(
                        obj,
                        &[
                            "reasoningBudget",
                            "reasoning_budget",
                            "thinkingBudget",
                            "thinking_budget",
                        ],
                    );
                    for key in [
                        "reasoningBudget",
                        "reasoning_budget",
                        "thinkingBudget",
                        "thinking_budget",
                    ] {
                        obj.remove(key);
                    }
                    if obj.get("reasoning_effort").is_none() {
                        if legacy_reasoning_budget.is_some() {
                            obj.insert("reasoning_effort".to_string(), serde_json::json!("high"));
                        } else if matches!(legacy_reasoning_enabled, Some(true)) {
                            obj.insert("reasoning_effort".to_string(), serde_json::json!("low"));
                        }
                    }

                    if let Some(mut v) = take_any(obj, &["searchParameters", "search_parameters"]) {
                        normalize_xai_search_parameters(&mut v);
                        obj.entry("search_parameters".to_string()).or_insert(v);
                    }
                }
            }
            // MoonshotAI uses AI SDK camelCase request options on the public surface but expects
            // snake_case on the wire for `reasoning_history` and `thinking.budget_tokens`.
            "moonshot" | "moonshotai" => {
                if let Some(obj) = params.as_object_mut() {
                    if let Some(v) = take_any(obj, &["reasoningHistory", "reasoning_history"]) {
                        obj.entry("reasoning_history".to_string()).or_insert(v);
                    }

                    if let Some(mut thinking) = take_any(obj, &["thinking"]) {
                        normalize_moonshot_thinking(&mut thinking);
                        obj.entry("thinking".to_string()).or_insert(thinking);
                    }
                }
            }
            // DeepSeek uses the AI SDK `thinking.type` shape. Legacy reasoning aliases remain
            // accepted as input but are not forwarded to the wire.
            "deepseek" => {
                if let Some(obj) = params.as_object_mut() {
                    normalize_deepseek_options(obj);
                }
            }
            // Alibaba/Qwen follows OpenAI chat-completions transport but uses snake_case
            // thinking controls on the wire.
            "qwen" | "alibaba" => {
                if let Some(obj) = params.as_object_mut() {
                    normalize_alibaba_options(obj);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        let config_mappings = &self.config.field_mappings;
        FieldMappings {
            thinking_fields: config_mappings
                .thinking_fields
                .iter()
                .map(|s| std::borrow::Cow::Owned(s.clone()))
                .collect(),
            content_field: std::borrow::Cow::Owned(config_mappings.content_field.clone()),
            tool_calls_field: std::borrow::Cow::Owned(config_mappings.tool_calls_field.clone()),
            role_field: std::borrow::Cow::Owned(config_mappings.role_field.clone()),
        }
    }

    fn get_model_config(&self, _model: &str) -> super::types::ModelConfig {
        super::types::ModelConfig {
            supports_thinking: self.config.supports_reasoning,
            ..Default::default()
        }
    }

    fn get_field_accessor(&self) -> Box<dyn FieldAccessor> {
        Box::new(JsonFieldAccessor)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new();
        let has_audio = self.config.capabilities.iter().any(|cap| cap == "audio");
        let has_speech = self
            .config
            .capabilities
            .iter()
            .any(|cap| matches!(cap.as_str(), "speech" | "tts"));
        let has_transcription = self
            .config
            .capabilities
            .iter()
            .any(|cap| matches!(cap.as_str(), "transcription" | "stt"));

        if provider_config_declares_chat_surface(&self.config) {
            caps = caps.with_chat().with_streaming();
        }
        if provider_config_declares_completion_surface(&self.config) {
            caps = caps.with_completion();
        }

        if self.config.capabilities.contains(&"tools".to_string()) {
            caps = caps.with_tools();
        }
        if self.config.capabilities.contains(&"vision".to_string()) {
            caps = caps.with_vision();
        }
        if self.config.capabilities.contains(&"embedding".to_string()) {
            caps = caps.with_embedding();
        }
        if self.config.capabilities.contains(&"rerank".to_string()) {
            caps = caps.with_rerank();
        }
        if self
            .config
            .capabilities
            .contains(&"image_generation".to_string())
        {
            caps = caps.with_image_generation();
        }
        if self.config.supports_reasoning {
            caps = caps.with_custom_feature("reasoning", true);
        }
        if has_audio {
            caps = caps.with_audio();
        } else {
            if has_speech {
                caps = caps.with_speech();
            }
            if has_transcription {
                caps = caps.with_transcription();
            }
        }

        caps
    }

    fn base_url(&self) -> &str {
        &self.config.base_url
    }

    fn audio_base_url(&self) -> Option<&str> {
        match self.config.id.as_str() {
            "fireworks" => Some("https://audio.fireworks.ai/v1"),
            _ => None,
        }
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }

    fn extract_response_provider_metadata(
        &self,
        raw: &serde_json::Value,
    ) -> Option<NestedProviderMetadata> {
        let metadata_key = provider_options_key(&self.config.id);
        match self.config.id.as_str() {
            "perplexity" => extract_perplexity_provider_metadata(&metadata_key, raw),
            _ => extract_openai_compatible_provider_metadata(&metadata_key, raw),
        }
    }

    fn supports_stream_usage_hints(&self) -> bool {
        !matches!(self.config.id.as_str(), "groq")
    }

    fn supports_image_generation(&self) -> bool {
        self.config
            .capabilities
            .contains(&"image_generation".to_string())
    }

    fn transform_image_request(
        &self,
        _request: &mut crate::types::ImageGenerationRequest,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn get_supported_image_sizes(&self) -> Vec<String> {
        vec![
            "256x256".to_string(),
            "512x512".to_string(),
            "1024x1024".to_string(),
            "1024x1792".to_string(),
            "1792x1024".to_string(),
        ]
    }

    fn get_supported_image_formats(&self) -> Vec<String> {
        vec!["url".to_string(), "b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        self.supports_image_generation()
    }

    fn supports_image_variations(&self) -> bool {
        self.supports_image_generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configurable_adapter_capabilities_include_image_generation_and_rerank() {
        let cfg = ProviderConfig {
            id: "test".to_string(),
            name: "Test".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "rerank".to_string(),
                "image_generation".to_string(),
            ],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();
        assert!(caps.supports("tools"));
        assert!(caps.supports("vision"));
        assert!(caps.supports("embedding"));
        assert!(caps.supports("rerank"));
        assert!(caps.supports("image_generation"));
        assert!(caps.supports("reasoning"));
        assert!(adapter.supports_image_generation());
    }

    #[test]
    fn configurable_adapter_capabilities_include_audio_family_aliases() {
        let cfg = ProviderConfig {
            id: "audio-test".to_string(),
            name: "Audio Test".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tts".to_string(), "stt".to_string()],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
    }

    #[test]
    fn configurable_adapter_focused_non_text_caps_do_not_imply_chat_surface() {
        let cfg = ProviderConfig {
            id: "focused".to_string(),
            name: "Focused".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string(), "rerank".to_string()],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(!caps.supports("chat"));
        assert!(!caps.supports("streaming"));
        assert!(caps.supports("embedding"));
        assert!(caps.supports("rerank"));
    }

    #[test]
    fn configurable_adapter_mixed_caps_keep_chat_surface() {
        let cfg = ProviderConfig {
            id: "mixed".to_string(),
            name: "Mixed".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("completion"));
        assert!(caps.supports("embedding"));
    }

    #[test]
    fn configurable_adapter_mistral_keeps_chat_surface_but_not_completion_surface() {
        let cfg = ProviderConfig {
            id: "mistral".to_string(),
            name: "Mistral AI".to_string(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("mistral-large-latest".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(!caps.supports("completion"));
        assert!(caps.supports("embedding"));
    }

    #[test]
    fn configurable_adapter_perplexity_keeps_chat_surface_but_not_completion_surface() {
        let cfg = ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("sonar".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(!caps.supports("completion"));
        assert!(caps.supports("tools"));
    }

    #[test]
    fn configurable_adapter_fireworks_exposes_transcription_only_audio_base() {
        let cfg = ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["transcription".to_string()],
            default_model: Some("whisper-v3".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };

        let adapter = ConfigurableAdapter::new(cfg);
        let caps = adapter.capabilities();

        assert!(!caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
        assert_eq!(
            adapter.audio_base_url(),
            Some("https://audio.fireworks.ai/v1")
        );
    }

    #[test]
    fn configurable_adapter_extracts_standard_metadata_for_openai_compatible_providers() {
        let openai = ConfigurableAdapter::new(ProviderConfig {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("gpt-4.1-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        });
        let raw = serde_json::json!({
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "hello",
                        "logprob": -0.1,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": []
                    }]
                }
            }],
            "usage": {
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 15,
                    "rejected_prediction_tokens": 5
                }
            }
        });
        assert!(openai.extract_response_provider_metadata(&raw).is_some());

        let generic = ConfigurableAdapter::new(ProviderConfig {
            id: "test-provider".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        });
        let metadata = generic
            .extract_response_provider_metadata(&raw)
            .expect("generic metadata");
        let generic = metadata
            .get("test-provider")
            .expect("generic provider namespace");
        assert_eq!(generic["logprobs"][0]["token"], serde_json::json!("hello"));
        assert_eq!(
            generic.get("acceptedPredictionTokens"),
            Some(&serde_json::json!(15))
        );
        assert_eq!(
            generic.get("rejectedPredictionTokens"),
            Some(&serde_json::json!(5))
        );
    }

    #[test]
    fn configurable_adapter_canonicalizes_moonshot_alias_metadata_namespace() {
        let adapter = ConfigurableAdapter::new(ProviderConfig {
            id: "moonshot".to_string(),
            name: "Moonshot".to_string(),
            base_url: "https://api.moonshot.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("kimi-k2-0905".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        });
        let raw = serde_json::json!({
            "usage": {
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 9,
                    "rejected_prediction_tokens": 3
                }
            }
        });

        let metadata = adapter
            .extract_response_provider_metadata(&raw)
            .expect("moonshot metadata");
        assert!(metadata.contains_key("moonshotai"));
        assert!(!metadata.contains_key("moonshot"));
        assert_eq!(
            metadata
                .get("moonshotai")
                .and_then(|value| value.get("acceptedPredictionTokens")),
            Some(&serde_json::json!(9))
        );
        assert_eq!(
            metadata
                .get("moonshotai")
                .and_then(|value| value.get("rejectedPredictionTokens")),
            Some(&serde_json::json!(3))
        );
    }

    #[test]
    fn configurable_adapter_transforms_groq_compat_quirks() {
        let cfg = ProviderConfig {
            id: "groq".to_string(),
            name: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("llama-3.3-70b-versatile".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };
        let adapter = ConfigurableAdapter::new(cfg);

        let mut params = serde_json::json!({
            "model": "llama-3.3-70b-versatile",
            "messages": [
                { "role": "developer", "content": "dev" },
                { "role": "user", "content": "hi" }
            ],
            "stream_options": { "include_usage": true },
            "max_completion_tokens": 123
        });

        adapter
            .transform_request_params(&mut params, "llama-3.3-70b-versatile", RequestType::Chat)
            .expect("transform ok");

        assert_eq!(params["messages"][0]["role"], "system");
        assert!(params.get("stream_options").is_none());
        assert_eq!(params["max_tokens"], 123);
    }

    #[test]
    fn configurable_adapter_transforms_deepseek_thinking_compat_quirks() {
        let cfg = ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("deepseek-chat".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };
        let adapter = ConfigurableAdapter::new(cfg);

        let mut params = serde_json::json!({
            "model": "deepseek-chat",
            "enableReasoning": true,
            "reasoningBudget": 2048
        });

        adapter
            .transform_request_params(&mut params, "deepseek-chat", RequestType::Chat)
            .expect("transform ok");

        assert_eq!(
            params["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert!(params.get("enableReasoning").is_none());
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoningBudget").is_none());
        assert!(params.get("reasoning_budget").is_none());
    }

    #[test]
    fn configurable_adapter_transforms_xai_reasoning_compat_quirks() {
        let cfg = ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("grok-4".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        };
        let adapter = ConfigurableAdapter::new(cfg);

        let mut params = serde_json::json!({
            "model": "grok-4",
            "stream_options": { "include_usage": true },
            "enableReasoning": true,
            "reasoningBudget": 2048,
            "enable_thinking": true,
            "thinking_budget": 1024
        });

        adapter
            .transform_request_params(&mut params, "grok-4", RequestType::Chat)
            .expect("transform ok");

        assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            params["stream_options"],
            serde_json::json!({ "include_usage": true })
        );
        assert!(params.get("enableReasoning").is_none());
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoningBudget").is_none());
        assert!(params.get("reasoning_budget").is_none());
        assert!(params.get("enableThinking").is_none());
        assert!(params.get("enable_thinking").is_none());
        assert!(params.get("thinkingBudget").is_none());
        assert!(params.get("thinking_budget").is_none());
    }

    #[test]
    fn configurable_adapter_reports_stream_usage_hint_support_by_provider() {
        let deepseek = ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("deepseek-chat".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        });
        let xai = ConfigurableAdapter::new(ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string(), "streaming".to_string()],
            default_model: Some("grok-4".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        });

        assert!(deepseek.supports_stream_usage_hints());
        assert!(xai.supports_stream_usage_hints());
    }
}
