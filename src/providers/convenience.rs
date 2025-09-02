//! Provider Convenience Functions
//!
//! This module provides convenient top-level functions for creating provider builders.
//! These functions are re-exported at the crate root for easy access.

/// Core provider convenience functions
pub mod core {
    /// Create an `OpenAI` client builder
    #[cfg(feature = "openai")]
    pub fn openai() -> crate::providers::openai::OpenAiBuilder {
        crate::builder::LlmBuilder::new().openai()
    }

    /// Create an `Anthropic` client builder
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> crate::providers::anthropic::AnthropicBuilder {
        crate::builder::LlmBuilder::new().anthropic()
    }

    /// Create a `Google Gemini` client builder
    #[cfg(feature = "google")]
    pub fn gemini() -> crate::providers::gemini::GeminiBuilder {
        crate::builder::LlmBuilder::new().gemini()
    }

    /// Create an `Ollama` client builder
    #[cfg(feature = "ollama")]
    pub fn ollama() -> crate::providers::ollama::OllamaBuilder {
        crate::builder::LlmBuilder::new().ollama()
    }

    /// Create a `Groq` client builder (native implementation)
    #[cfg(feature = "groq")]
    pub fn groq() -> crate::providers::groq::GroqBuilderWrapper {
        crate::builder::LlmBuilder::new().groq()
    }

    /// Create an `xAI` client builder (native implementation)
    #[cfg(feature = "xai")]
    pub fn xai() -> crate::providers::xai::XaiBuilderWrapper {
        crate::builder::LlmBuilder::new().xai()
    }
}

/// OpenAI-compatible provider convenience functions
pub mod openai_compatible {
    /// Create a `DeepSeek` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn deepseek() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().deepseek()
    }

    /// Create an `OpenRouter` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn openrouter() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().openrouter()
    }

    /// Create a `SiliconFlow` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn siliconflow() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().siliconflow()
    }

    /// Create a `Together AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn together() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().together()
    }

    /// Create a `Fireworks AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn fireworks() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().fireworks()
    }

    /// Create a `GitHub Copilot` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn github_copilot() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().github_copilot()
    }

    /// Create a `Perplexity` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn perplexity() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().perplexity()
    }

    /// Create a `Mistral AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn mistral() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().mistral()
    }

    /// Create a `Cohere` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn cohere() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().cohere()
    }

    /// Create a `Zhipu AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn zhipu() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().zhipu()
    }

    /// Create a `Moonshot AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn moonshot() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().moonshot()
    }

    /// Create a `01.AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn yi() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().yi()
    }

    /// Create a `Doubao` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn doubao() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().doubao()
    }

    /// Create a `Baichuan AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn baichuan() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().baichuan()
    }

    /// Create a `Qwen` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn qwen() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().qwen()
    }

    /// Create a `Groq` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn groq_openai_compatible() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder
    {
        crate::builder::LlmBuilder::new().groq_openai_compatible()
    }

    /// Create an `xAI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn xai_openai_compatible() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().xai_openai_compatible()
    }

    /// Create a `Nvidia` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn nvidia() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().nvidia()
    }

    /// Create a `Hyperbolic` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn hyperbolic() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().hyperbolic()
    }

    /// Create a `Jina AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn jina() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().jina()
    }

    /// Create a `GitHub Models` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn github() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().github()
    }

    /// Create a `VoyageAI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn voyageai() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().voyageai()
    }

    /// Create a `Poe` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn poe() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().poe()
    }

    /// Create a `StepFun` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn stepfun() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().stepfun()
    }

    /// Create a `MiniMax` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn minimax() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().minimax()
    }

    /// Create an `Infini AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn infini() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().infini()
    }

    /// Create a `ModelScope` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn modelscope() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().modelscope()
    }

    /// Create a `Hunyuan` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn hunyuan() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().hunyuan()
    }

    /// Create a `Baidu Cloud` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn baidu_cloud() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().baidu_cloud()
    }

    /// Create a `Tencent Cloud TI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn tencent_cloud_ti() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().tencent_cloud_ti()
    }

    /// Create a `Xirang` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn xirang() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().xirang()
    }

    /// Create a `302.AI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn ai302() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().ai302()
    }

    /// Create an `AiHubMix` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn aihubmix() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().aihubmix()
    }

    /// Create a `PPIO` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn ppio() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().ppio()
    }

    /// Create an `OcoolAI` client builder (OpenAI-compatible)
    #[cfg(feature = "openai")]
    pub fn ocoolai() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
        crate::builder::LlmBuilder::new().ocoolai()
    }
}
