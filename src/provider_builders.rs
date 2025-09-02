//! SiumaiBuilder Provider Methods
//!
//! This module contains all provider-specific methods for SiumaiBuilder to keep the main provider.rs clean.
//! Each provider gets its own method that sets the appropriate provider type and name.

use crate::provider::SiumaiBuilder;
use crate::types::ProviderType;

/// Provider builder methods for SiumaiBuilder
impl SiumaiBuilder {
    // ========================================================================
    // Core Providers (with native implementations)
    // ========================================================================

    /// Create an `OpenAI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn openai(mut self) -> Self {
        self.provider_type = Some(ProviderType::OpenAi);
        self.provider_name = Some("openai".to_string());
        self
    }

    /// Create an Anthropic provider (convenience method)
    #[cfg(feature = "anthropic")]
    pub fn anthropic(mut self) -> Self {
        self.provider_type = Some(ProviderType::Anthropic);
        self.provider_name = Some("anthropic".to_string());
        self
    }

    /// Create a Gemini provider (convenience method)
    #[cfg(feature = "google")]
    pub fn gemini(mut self) -> Self {
        self.provider_type = Some(ProviderType::Gemini);
        self.provider_name = Some("gemini".to_string());
        self
    }

    /// Create an Ollama provider (convenience method)
    #[cfg(feature = "ollama")]
    pub fn ollama(mut self) -> Self {
        self.provider_type = Some(ProviderType::Ollama);
        self.provider_name = Some("ollama".to_string());
        self
    }

    /// Create an xAI provider (convenience method)
    #[cfg(feature = "xai")]
    pub fn xai(mut self) -> Self {
        self.provider_type = Some(ProviderType::XAI);
        self.provider_name = Some("xai".to_string());
        self
    }

    /// Create a Groq provider (convenience method)
    #[cfg(feature = "groq")]
    pub fn groq(mut self) -> Self {
        self.provider_type = Some(ProviderType::Groq);
        self.provider_name = Some("groq".to_string());
        self
    }

    // ========================================================================
    // OpenAI-Compatible Providers
    // ========================================================================

    /// Create a `SiliconFlow` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn siliconflow(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("siliconflow".to_string()));
        self.provider_name = Some("siliconflow".to_string());
        self
    }

    /// Create a `DeepSeek` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn deepseek(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("deepseek".to_string()));
        self.provider_name = Some("deepseek".to_string());
        self
    }

    /// Create an `OpenRouter` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn openrouter(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("openrouter".to_string()));
        self.provider_name = Some("openrouter".to_string());
        self
    }

    /// Create a `Together AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn together(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("together".to_string()));
        self.provider_name = Some("together".to_string());
        self
    }

    /// Create a `Fireworks AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn fireworks(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("fireworks".to_string()));
        self.provider_name = Some("fireworks".to_string());
        self
    }

    /// Create a `GitHub Copilot` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn github_copilot(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("github_copilot".to_string()));
        self.provider_name = Some("github_copilot".to_string());
        self
    }

    /// Create a `Perplexity` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn perplexity(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("perplexity".to_string()));
        self.provider_name = Some("perplexity".to_string());
        self
    }

    /// Create a `Mistral AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn mistral(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("mistral".to_string()));
        self.provider_name = Some("mistral".to_string());
        self
    }

    /// Create a `Cohere` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn cohere(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("cohere".to_string()));
        self.provider_name = Some("cohere".to_string());
        self
    }

    /// Create a `Zhipu AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn zhipu(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("zhipu".to_string()));
        self.provider_name = Some("zhipu".to_string());
        self
    }

    /// Create a `Moonshot AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn moonshot(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("moonshot".to_string()));
        self.provider_name = Some("moonshot".to_string());
        self
    }

    /// Create a `Doubao` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn doubao(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("doubao".to_string()));
        self.provider_name = Some("doubao".to_string());
        self
    }

    /// Create a `Qwen` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn qwen(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("qwen".to_string()));
        self.provider_name = Some("qwen".to_string());
        self
    }

    /// Create a `01.AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn yi(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("yi".to_string()));
        self.provider_name = Some("yi".to_string());
        self
    }

    /// Create a `Baichuan AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn baichuan(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("baichuan".to_string()));
        self.provider_name = Some("baichuan".to_string());
        self
    }

    // ========================================================================
    // OpenAI-Compatible Versions of Native Providers
    // ========================================================================

    /// Create a `Groq` provider (OpenAI-compatible, convenience method)
    #[cfg(feature = "openai")]
    pub fn groq_openai_compatible(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("groq_openai_compatible".to_string()));
        self.provider_name = Some("groq_openai_compatible".to_string());
        self
    }

    /// Create an `xAI` provider (OpenAI-compatible, convenience method)
    #[cfg(feature = "openai")]
    pub fn xai_openai_compatible(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("xai_openai_compatible".to_string()));
        self.provider_name = Some("xai_openai_compatible".to_string());
        self
    }

    // ========================================================================
    // International Providers
    // ========================================================================

    /// Create a `Nvidia` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn nvidia(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("nvidia".to_string()));
        self.provider_name = Some("nvidia".to_string());
        self
    }

    /// Create a `Hyperbolic` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn hyperbolic(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("hyperbolic".to_string()));
        self.provider_name = Some("hyperbolic".to_string());
        self
    }

    /// Create a `Jina AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn jina(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("jina".to_string()));
        self.provider_name = Some("jina".to_string());
        self
    }

    /// Create a `GitHub Models` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn github(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("github".to_string()));
        self.provider_name = Some("github".to_string());
        self
    }

    /// Create a `VoyageAI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn voyageai(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("voyageai".to_string()));
        self.provider_name = Some("voyageai".to_string());
        self
    }

    /// Create a `Poe` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn poe(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("poe".to_string()));
        self.provider_name = Some("poe".to_string());
        self
    }

    // ========================================================================
    // Chinese Providers
    // ========================================================================

    /// Create a `StepFun` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn stepfun(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("stepfun".to_string()));
        self.provider_name = Some("stepfun".to_string());
        self
    }

    /// Create a `MiniMax` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn minimax(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("minimax".to_string()));
        self.provider_name = Some("minimax".to_string());
        self
    }

    /// Create an `Infini AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn infini(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("infini".to_string()));
        self.provider_name = Some("infini".to_string());
        self
    }

    /// Create a `ModelScope` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn modelscope(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("modelscope".to_string()));
        self.provider_name = Some("modelscope".to_string());
        self
    }

    /// Create a `Hunyuan` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn hunyuan(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("hunyuan".to_string()));
        self.provider_name = Some("hunyuan".to_string());
        self
    }

    /// Create a `Baidu Cloud` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn baidu_cloud(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("baidu_cloud".to_string()));
        self.provider_name = Some("baidu_cloud".to_string());
        self
    }

    /// Create a `Tencent Cloud TI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn tencent_cloud_ti(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("tencent_cloud_ti".to_string()));
        self.provider_name = Some("tencent_cloud_ti".to_string());
        self
    }

    /// Create a `Xirang` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn xirang(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("xirang".to_string()));
        self.provider_name = Some("xirang".to_string());
        self
    }

    // ========================================================================
    // Platform Aggregation Providers
    // ========================================================================

    /// Create a `302.AI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn ai302(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("302ai".to_string()));
        self.provider_name = Some("302ai".to_string());
        self
    }

    /// Create an `AiHubMix` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn aihubmix(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("aihubmix".to_string()));
        self.provider_name = Some("aihubmix".to_string());
        self
    }

    /// Create a `PPIO` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn ppio(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("ppio".to_string()));
        self.provider_name = Some("ppio".to_string());
        self
    }

    /// Create an `OcoolAI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn ocoolai(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("ocoolai".to_string()));
        self.provider_name = Some("ocoolai".to_string());
        self
    }
}
