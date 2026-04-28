use crate::types::ChatRequest;

fn merge_provider_option_object_for(
    map: &mut crate::types::ProviderOptionsMap,
    provider_id: &str,
    value: serde_json::Value,
) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get(provider_id)
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert(provider_id, serde_json::Value::Object(merged));
    } else {
        map.insert(provider_id, value);
    }
}

fn merge_provider_option_object(
    mut request: ChatRequest,
    provider_id: &str,
    value: serde_json::Value,
) -> ChatRequest {
    merge_provider_option_object_for(&mut request.provider_options_map, provider_id, value);
    request
}

/// Generic OpenAI-compatible request option helpers for `ChatRequest`.
///
/// This targets the shared `provider_options_map["openaiCompatible"]` namespace.
pub trait OpenAiCompatibleChatRequestExt {
    /// Convenience: attach generic OpenAI-compatible options to
    /// `provider_options_map["openaiCompatible"]`.
    fn with_openai_compatible_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenAiCompatibleChatRequestExt for ChatRequest {
    fn with_openai_compatible_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object(self, "openaiCompatible", value)
    }
}

/// Generic OpenAI-compatible request option helpers for `CompletionRequest`.
///
/// This targets the shared `provider_options_map["openaiCompatible"]` namespace.
pub trait OpenAiCompatibleCompletionRequestExt {
    /// Convenience: attach generic OpenAI-compatible options to
    /// `provider_options_map["openaiCompatible"]`.
    fn with_openai_compatible_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenAiCompatibleCompletionRequestExt for crate::types::CompletionRequest {
    fn with_openai_compatible_options<T: serde::Serialize>(mut self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object_for(&mut self.provider_options_map, "openaiCompatible", value);
        self
    }
}

/// Generic OpenAI-compatible request option helpers for `EmbeddingRequest`.
///
/// This targets the shared `provider_options_map["openaiCompatible"]` namespace.
pub trait OpenAiCompatibleEmbeddingRequestExt {
    /// Convenience: attach generic OpenAI-compatible options to
    /// `provider_options_map["openaiCompatible"]`.
    fn with_openai_compatible_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenAiCompatibleEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_openai_compatible_options<T: serde::Serialize>(mut self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object_for(&mut self.provider_options_map, "openaiCompatible", value);
        self
    }
}

/// Alibaba request option helpers for `ChatRequest`.
///
/// This targets the AI SDK-aligned `provider_options_map["alibaba"]` namespace.
pub trait AlibabaChatRequestExt {
    /// Convenience: attach Alibaba-specific options to `provider_options_map["alibaba"]`.
    fn with_alibaba_options(self, options: crate::provider_options::AlibabaChatOptions) -> Self;
}

impl AlibabaChatRequestExt for ChatRequest {
    fn with_alibaba_options(self, options: crate::provider_options::AlibabaChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize AlibabaChatOptions");
        merge_provider_option_object(self, "alibaba", value)
    }
}

/// Qwen request option helpers for `ChatRequest`.
///
/// This targets the local `provider_options_map["qwen"]` namespace.
pub trait QwenChatRequestExt {
    /// Convenience: attach Qwen-specific options to `provider_options_map["qwen"]`.
    fn with_qwen_options(self, options: crate::provider_options::QwenChatOptions) -> Self;
}

impl QwenChatRequestExt for ChatRequest {
    fn with_qwen_options(self, options: crate::provider_options::QwenChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize QwenChatOptions");
        merge_provider_option_object(self, "qwen", value)
    }
}

/// Mistral request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait MistralChatRequestExt {
    /// Convenience: attach Mistral-specific options to `provider_options_map["mistral"]`.
    fn with_mistral_options(self, options: crate::provider_options::MistralChatOptions) -> Self;
}

impl MistralChatRequestExt for ChatRequest {
    fn with_mistral_options(self, options: crate::provider_options::MistralChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize MistralChatOptions");
        merge_provider_option_object(self, "mistral", value)
    }
}

/// Fireworks request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait FireworksChatRequestExt {
    /// Convenience: attach Fireworks-specific options to `provider_options_map["fireworks"]`.
    fn with_fireworks_options(self, options: crate::provider_options::FireworksChatOptions)
    -> Self;
}

impl FireworksChatRequestExt for ChatRequest {
    fn with_fireworks_options(
        self,
        options: crate::provider_options::FireworksChatOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize FireworksChatOptions");
        merge_provider_option_object(self, "fireworks", value)
    }
}

/// MoonshotAI request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait MoonshotAIChatRequestExt {
    /// Convenience: attach MoonshotAI-specific options to `provider_options_map["moonshotai"]`.
    fn with_moonshotai_options(
        self,
        options: crate::provider_options::MoonshotAIChatOptions,
    ) -> Self;
}

impl MoonshotAIChatRequestExt for ChatRequest {
    fn with_moonshotai_options(
        self,
        options: crate::provider_options::MoonshotAIChatOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize MoonshotAIChatOptions");
        merge_provider_option_object(self, "moonshotai", value)
    }
}

/// OpenRouter request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait OpenRouterChatRequestExt {
    /// Convenience: attach OpenRouter-specific options to `provider_options_map["openrouter"]`.
    fn with_openrouter_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenRouterChatRequestExt for ChatRequest {
    fn with_openrouter_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object(self, "openrouter", value)
    }
}

/// Perplexity request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait PerplexityChatRequestExt {
    /// Convenience: attach Perplexity-specific options to `provider_options_map["perplexity"]`.
    fn with_perplexity_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl PerplexityChatRequestExt for ChatRequest {
    fn with_perplexity_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        merge_provider_option_object(self, "perplexity", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        AlibabaChatOptions, FireworksChatOptions, FireworksReasoningHistory,
        FireworksThinkingConfig, FireworksThinkingType, MistralChatOptions, MistralReasoningEffort,
        MoonshotAIChatOptions, MoonshotAIReasoningHistory, MoonshotAIThinkingConfig,
        MoonshotAIThinkingType, OpenAiCompatibleEmbeddingModelOptions,
        OpenAiCompatibleLanguageModelChatOptions, OpenAiCompatibleLanguageModelCompletionOptions,
        OpenRouterOptions, OpenRouterTransform, PerplexityOptions, PerplexitySearchMode,
    };
    use crate::types::{ChatMessage, CompletionRequest, EmbeddingRequest};

    #[test]
    fn chat_request_ext_attaches_openai_compatible_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "openaiCompatible",
                serde_json::json!({ "existing": true, "textVerbosity": "low" }),
            )
            .with_openai_compatible_options(
                OpenAiCompatibleLanguageModelChatOptions::new()
                    .with_user("user-123")
                    .with_reasoning_effort("high")
                    .with_strict_json_schema(true),
            );

        let value = request
            .provider_options_map
            .get("openaiCompatible")
            .expect("openai-compatible options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["user"], serde_json::json!("user-123"));
        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["textVerbosity"], serde_json::json!("low"));
        assert_eq!(value["strictJsonSchema"], serde_json::json!(true));
    }

    #[test]
    fn completion_request_ext_attaches_openai_compatible_options() {
        let request = CompletionRequest::new("hi")
            .with_provider_option(
                "openaiCompatible",
                serde_json::json!({ "existing": true, "echo": false }),
            )
            .with_openai_compatible_options(
                OpenAiCompatibleLanguageModelCompletionOptions::new()
                    .with_echo(true)
                    .with_logit_bias_token("42", 1.5)
                    .with_suffix(" after")
                    .with_user("user-456"),
            );

        let value = request
            .provider_options_map
            .get("openaiCompatible")
            .expect("openai-compatible options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["echo"], serde_json::json!(true));
        assert_eq!(value["logitBias"]["42"], serde_json::json!(1.5));
        assert_eq!(value["suffix"], serde_json::json!(" after"));
        assert_eq!(value["user"], serde_json::json!("user-456"));
    }

    #[test]
    fn embedding_request_ext_attaches_openai_compatible_options() {
        let request = EmbeddingRequest::single("hello")
            .with_provider_option("openaiCompatible", serde_json::json!({ "existing": true }))
            .with_openai_compatible_options(
                OpenAiCompatibleEmbeddingModelOptions::new()
                    .with_dimensions(256)
                    .with_user("user-789"),
            );

        let value = request
            .provider_options_map
            .get("openaiCompatible")
            .expect("openai-compatible options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["dimensions"], serde_json::json!(256));
        assert_eq!(value["user"], serde_json::json!("user-789"));
    }

    #[test]
    fn chat_request_ext_attaches_alibaba_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("alibaba", serde_json::json!({ "existing": true }))
            .with_alibaba_options(
                AlibabaChatOptions::new()
                    .with_enable_thinking(true)
                    .with_thinking_budget(2048)
                    .with_parallel_tool_calls(false),
            );

        let value = request
            .provider_options_map
            .get("alibaba")
            .expect("alibaba options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["enableThinking"], serde_json::json!(true));
        assert_eq!(value["thinkingBudget"], serde_json::json!(2048));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
    }

    #[test]
    fn chat_request_ext_attaches_qwen_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("qwen", serde_json::json!({ "existing": true }))
            .with_qwen_options(
                AlibabaChatOptions::new()
                    .with_enable_thinking(false)
                    .with_thinking_budget(1024),
            );

        let value = request
            .provider_options_map
            .get("qwen")
            .expect("qwen options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["enableThinking"], serde_json::json!(false));
        assert_eq!(value["thinkingBudget"], serde_json::json!(1024));
    }

    #[test]
    fn chat_request_ext_attaches_mistral_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("mistral", serde_json::json!({ "reasoningEffort": "none" }))
            .with_mistral_options(
                MistralChatOptions::new()
                    .with_safe_prompt(true)
                    .with_parallel_tool_calls(false)
                    .with_reasoning_effort(MistralReasoningEffort::High),
            );

        let value = request
            .provider_options_map
            .get("mistral")
            .expect("mistral options present");
        assert_eq!(value["safePrompt"], serde_json::json!(true));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
    }

    #[test]
    fn chat_request_ext_attaches_fireworks_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "fireworks",
                serde_json::json!({ "reasoningEffort": "minimal" }),
            )
            .with_fireworks_options(
                FireworksChatOptions::new()
                    .with_thinking(
                        FireworksThinkingConfig::new()
                            .with_type(FireworksThinkingType::Enabled)
                            .with_budget_tokens(2048),
                    )
                    .with_reasoning_history(FireworksReasoningHistory::Interleaved),
            );

        let value = request
            .provider_options_map
            .get("fireworks")
            .expect("fireworks options present");
        assert_eq!(value["reasoningEffort"], serde_json::json!("minimal"));
        assert_eq!(value["reasoningHistory"], serde_json::json!("interleaved"));
        assert_eq!(value["thinking"]["type"], serde_json::json!("enabled"));
        assert_eq!(value["thinking"]["budgetTokens"], serde_json::json!(2048));
    }

    #[test]
    fn chat_request_ext_attaches_moonshotai_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("moonshotai", serde_json::json!({ "existing": true }))
            .with_moonshotai_options(
                MoonshotAIChatOptions::new()
                    .with_thinking(
                        MoonshotAIThinkingConfig::new()
                            .with_type(MoonshotAIThinkingType::Enabled)
                            .with_budget_tokens(2048),
                    )
                    .with_reasoning_history(MoonshotAIReasoningHistory::Interleaved),
            );

        let value = request
            .provider_options_map
            .get("moonshotai")
            .expect("moonshotai options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["reasoningHistory"], serde_json::json!("interleaved"));
        assert_eq!(value["thinking"]["type"], serde_json::json!("enabled"));
        assert_eq!(value["thinking"]["budgetTokens"], serde_json::json!(2048));
    }

    #[test]
    fn chat_request_ext_attaches_openrouter_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let value = request
            .provider_options_map
            .get("openrouter")
            .expect("openrouter options present");
        assert_eq!(value["transforms"], serde_json::json!(["middle-out"]));
        assert_eq!(value["someVendorParam"], serde_json::json!(true));
    }

    #[test]
    fn chat_request_ext_merges_openrouter_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "openrouter",
                serde_json::json!({
                    "existing": true,
                    "transforms": ["legacy"]
                }),
            )
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let value = request
            .provider_options_map
            .get("openrouter")
            .expect("openrouter options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["transforms"], serde_json::json!(["middle-out"]));
        assert_eq!(value["someVendorParam"], serde_json::json!(true));
    }

    #[test]
    fn chat_request_ext_attaches_perplexity_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_return_images(true),
            );

        let value = request
            .provider_options_map
            .get("perplexity")
            .expect("perplexity options present");
        assert_eq!(value["searchMode"], serde_json::json!("academic"));
        assert_eq!(value["returnImages"], serde_json::json!(true));
    }

    #[test]
    fn chat_request_ext_merges_perplexity_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "perplexity",
                serde_json::json!({
                    "existing": true
                }),
            )
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_return_images(true),
            );

        let value = request
            .provider_options_map
            .get("perplexity")
            .expect("perplexity options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["searchMode"], serde_json::json!("academic"));
        assert_eq!(value["returnImages"], serde_json::json!(true));
    }
}
