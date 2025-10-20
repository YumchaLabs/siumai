//! Provider Builder Methods
//!
//! This module contains all provider-specific builder methods to keep the main builder clean.
//! Each provider gets its own builder method that returns the appropriate builder type.

use crate::builder::LlmBuilder;

/// OpenAI-compatible provider builder methods
impl LlmBuilder {
    // ========================================================================
    // Core Providers (with native implementations)
    // ========================================================================

    /// Create an `OpenAI` client builder.
    #[cfg(feature = "openai")]
    pub fn openai(self) -> crate::providers::openai::OpenAiBuilder {
        crate::providers::openai::OpenAiBuilder::new(self)
    }

    /// Create an `Anthropic` client builder.
    #[cfg(feature = "anthropic")]
    pub fn anthropic(self) -> crate::providers::anthropic::AnthropicBuilder {
        crate::providers::anthropic::AnthropicBuilder::new(self)
    }

    /// Create a `Google Gemini` client builder.
    #[cfg(feature = "google")]
    pub fn gemini(self) -> crate::providers::gemini::GeminiBuilder {
        crate::providers::gemini::GeminiBuilder::new(self)
    }

    /// Create an `Ollama` client builder.
    #[cfg(feature = "ollama")]
    pub fn ollama(self) -> crate::providers::ollama::OllamaBuilder {
        crate::providers::ollama::OllamaBuilder::new(self)
    }

    /// Create a `Groq` client builder (native implementation).
    #[cfg(feature = "groq")]
    pub fn groq(self) -> crate::providers::groq::GroqBuilderWrapper {
        crate::providers::groq::GroqBuilderWrapper::new(self)
    }

    /// Create an `xAI` client builder (native implementation).
    #[cfg(feature = "xai")]
    pub fn xai(self) -> crate::providers::xai::XaiBuilderWrapper {
        crate::providers::xai::XaiBuilderWrapper::new(self)
    }

    // ========================================================================
    // OpenAI-Compatible Providers
    // ========================================================================

    /// Create a `DeepSeek` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn deepseek(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "deepseek")
    }

    /// Create an `OpenRouter` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn openrouter(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "openrouter")
    }

    /// Create a `SiliconFlow` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn siliconflow(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "siliconflow")
    }

    /// Create a `Together AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn together(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "together")
    }

    /// Create a `Fireworks AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn fireworks(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "fireworks")
    }

    /// Create a `GitHub Copilot` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn github_copilot(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "github_copilot")
    }

    /// Create a `Perplexity` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn perplexity(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "perplexity")
    }

    /// Create a `Mistral AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn mistral(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "mistral")
    }

    /// Create a `Cohere` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn cohere(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "cohere")
    }

    /// Create a `Zhipu AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn zhipu(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "zhipu")
    }

    /// Create a `Moonshot AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn moonshot(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "moonshot")
    }

    /// Create a `01.AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn yi(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "yi")
    }

    /// Create a `Doubao` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn doubao(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "doubao")
    }

    /// Create a `Baichuan AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn baichuan(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "baichuan")
    }

    /// Create a `Qwen` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn qwen(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "qwen")
    }

    // ========================================================================
    // OpenAI-Compatible Versions of Native Providers
    // ========================================================================

    /// Create a `Groq` client builder (OpenAI-compatible).
    ///
    /// This is the OpenAI-compatible version that uses the unified system.
    /// For the native implementation, use `.groq()` instead.
    #[cfg(feature = "openai")]
    pub fn groq_openai_compatible(
        self,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            self,
            "groq_openai_compatible",
        )
    }

    /// Create an `xAI` client builder (OpenAI-compatible).
    ///
    /// This is the OpenAI-compatible version that uses the unified system.
    /// For the native implementation, use `.xai()` instead.
    #[cfg(feature = "openai")]
    pub fn xai_openai_compatible(
        self,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            self,
            "xai_openai_compatible",
        )
    }

    // ========================================================================
    // International Providers
    // ========================================================================

    /// Create a `Nvidia` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn nvidia(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "nvidia")
    }

    /// Create a `Hyperbolic` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn hyperbolic(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "hyperbolic")
    }

    /// Create a `Jina AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn jina(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "jina")
    }

    /// Create a `GitHub Models` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn github(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "github")
    }

    /// Create a `VoyageAI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn voyageai(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "voyageai")
    }

    /// Create a `Poe` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn poe(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "poe")
    }

    // ========================================================================
    // Chinese Providers
    // ========================================================================

    /// Create a `StepFun` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn stepfun(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "stepfun")
    }

    /// Create a `MiniMax` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn minimax(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "minimax")
    }

    /// Create an `Infini AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn infini(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "infini")
    }

    /// Create a `ModelScope` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn modelscope(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "modelscope")
    }

    /// Create a `Hunyuan` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn hunyuan(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "hunyuan")
    }

    /// Create a `Baidu Cloud` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn baidu_cloud(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "baidu_cloud")
    }

    /// Create a `Tencent Cloud TI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn tencent_cloud_ti(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "tencent_cloud_ti")
    }

    /// Create a `Xirang` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn xirang(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "xirang")
    }

    // ========================================================================
    // Platform Aggregation Providers
    // ========================================================================

    /// Create a `302.AI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn ai302(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "302ai")
    }

    /// Create an `AiHubMix` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn aihubmix(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "aihubmix")
    }

    /// Create a `PPIO` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn ppio(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "ppio")
    }

    /// Create an `OcoolAI` client builder (OpenAI-compatible).
    #[cfg(feature = "openai")]
    pub fn ocoolai(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, "ocoolai")
    }
}
