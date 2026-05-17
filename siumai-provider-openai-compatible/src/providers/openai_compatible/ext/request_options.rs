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

/// DeepSeek request option helpers for `ChatRequest`.
///
/// This targets the AI SDK-aligned `provider_options_map["deepseek"]` namespace.
pub trait DeepSeekChatRequestExt {
    /// Convenience: attach DeepSeek-specific options to `provider_options_map["deepseek"]`.
    fn with_deepseek_options(
        self,
        options: crate::provider_options::DeepSeekLanguageModelOptions,
    ) -> Self;
}

impl DeepSeekChatRequestExt for ChatRequest {
    fn with_deepseek_options(
        self,
        options: crate::provider_options::DeepSeekLanguageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize DeepSeekLanguageModelOptions");
        merge_provider_option_object(self, "deepseek", value)
    }
}

/// Groq request option helpers for `ChatRequest`.
///
/// This targets the AI SDK-aligned `provider_options_map["groq"]` namespace.
pub trait GroqChatRequestExt {
    /// Convenience: attach Groq-specific options to `provider_options_map["groq"]`.
    fn with_groq_options(self, options: crate::provider_options::GroqLanguageModelOptions) -> Self;
}

impl GroqChatRequestExt for ChatRequest {
    fn with_groq_options(self, options: crate::provider_options::GroqLanguageModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GroqLanguageModelOptions");
        merge_provider_option_object(self, "groq", value)
    }
}

/// Groq transcription option helpers for `SttRequest`.
///
/// This targets the AI SDK-aligned `provider_options_map["groq"]` namespace.
pub trait GroqTranscriptionRequestExt {
    /// Convenience: attach Groq transcription options to `provider_options_map["groq"]`.
    fn with_groq_transcription_options(
        self,
        options: crate::provider_options::GroqTranscriptionModelOptions,
    ) -> Self;
}

impl GroqTranscriptionRequestExt for crate::types::SttRequest {
    fn with_groq_transcription_options(
        mut self,
        options: crate::provider_options::GroqTranscriptionModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize GroqTranscriptionModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "groq", value);
        self
    }
}

/// xAI request option helpers for `ChatRequest`.
///
/// This targets the AI SDK-aligned `provider_options_map["xai"]` namespace and accepts both
/// chat-completions and Responses option structs.
pub trait XaiChatRequestExt {
    /// Convenience: attach xAI-specific options to `provider_options_map["xai"]`.
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl XaiChatRequestExt for ChatRequest {
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).expect("serialize xAI options");
        merge_provider_option_object(self, "xai", value)
    }
}

/// xAI image option helpers.
///
/// This targets the AI SDK-aligned `provider_options_map["xai"]` namespace.
pub trait XaiImageRequestExt {
    /// Convenience: attach xAI image options to `provider_options_map["xai"]`.
    fn with_xai_image_options(self, options: crate::provider_options::XaiImageModelOptions)
    -> Self;
}

impl XaiImageRequestExt for crate::types::ImageGenerationRequest {
    fn with_xai_image_options(
        mut self,
        options: crate::provider_options::XaiImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "xai", value);
        self
    }
}

impl XaiImageRequestExt for crate::types::ImageEditRequest {
    fn with_xai_image_options(
        mut self,
        options: crate::provider_options::XaiImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "xai", value);
        self
    }
}

impl XaiImageRequestExt for crate::types::ImageVariationRequest {
    fn with_xai_image_options(
        mut self,
        options: crate::provider_options::XaiImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "xai", value);
        self
    }
}

impl XaiImageRequestExt for crate::types::GenerateImageRequest {
    fn with_xai_image_options(
        mut self,
        options: crate::provider_options::XaiImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize XaiImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "xai", value);
        self
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

/// TogetherAI image option helpers.
///
/// This targets the AI SDK-aligned `provider_options_map["togetherai"]` namespace.
pub trait TogetherAIImageRequestExt {
    /// Convenience: attach TogetherAI image options to `provider_options_map["togetherai"]`.
    fn with_togetherai_image_options(
        self,
        options: crate::provider_options::TogetherAIImageModelOptions,
    ) -> Self;
}

impl TogetherAIImageRequestExt for crate::types::ImageGenerationRequest {
    fn with_togetherai_image_options(
        mut self,
        options: crate::provider_options::TogetherAIImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize TogetherAIImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "togetherai", value);
        self
    }
}

impl TogetherAIImageRequestExt for crate::types::ImageEditRequest {
    fn with_togetherai_image_options(
        mut self,
        options: crate::provider_options::TogetherAIImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize TogetherAIImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "togetherai", value);
        self
    }
}

impl TogetherAIImageRequestExt for crate::types::ImageVariationRequest {
    fn with_togetherai_image_options(
        mut self,
        options: crate::provider_options::TogetherAIImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize TogetherAIImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "togetherai", value);
        self
    }
}

impl TogetherAIImageRequestExt for crate::types::GenerateImageRequest {
    fn with_togetherai_image_options(
        mut self,
        options: crate::provider_options::TogetherAIImageModelOptions,
    ) -> Self {
        let value = serde_json::to_value(options).expect("serialize TogetherAIImageModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "togetherai", value);
        self
    }
}

/// TogetherAI rerank option helpers.
///
/// This targets the AI SDK-aligned `provider_options_map["togetherai"]` namespace.
pub trait TogetherAIRerankRequestExt {
    /// Convenience: attach TogetherAI reranking options to `provider_options_map["togetherai"]`.
    fn with_togetherai_options(
        self,
        options: crate::provider_options::TogetherAIRerankingModelOptions,
    ) -> Self;
}

impl TogetherAIRerankRequestExt for crate::types::RerankRequest {
    fn with_togetherai_options(
        mut self,
        options: crate::provider_options::TogetherAIRerankingModelOptions,
    ) -> Self {
        let value =
            serde_json::to_value(options).expect("serialize TogetherAIRerankingModelOptions");
        merge_provider_option_object_for(&mut self.provider_options_map, "togetherai", value);
        self
    }
}

#[cfg(test)]
mod tests;
