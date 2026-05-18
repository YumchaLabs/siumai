use super::OpenAiCompatibleBuilder;

impl OpenAiCompatibleBuilder {
    fn apply_reasoning_patch(
        mut self,
        patch: crate::standards::openai::compat::reasoning::OpenAiCompatibleReasoningPatch,
    ) -> Self {
        patch.apply_to_hash_map(&mut self.provider_specific_config);
        self
    }

    /// Enable thinking mode for supported models.
    ///
    /// When enabled, models that support thinking (like DeepSeek V3.1, Qwen 3, etc.)
    /// will include their reasoning process in the response.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable thinking mode
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking(true)
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking(mut self, enable: bool) -> Self {
        let patch =
            crate::standards::openai::compat::reasoning::OpenAiCompatibleReasoningPolicy::for_provider(
                &self.provider_id,
            )
            .thinking(enable);
        self = self.apply_reasoning_patch(patch);
        self
    }

    /// Set the thinking budget (maximum tokens for reasoning) for supported models.
    ///
    /// This controls how many tokens the model can use for its internal reasoning process.
    /// Higher values allow for more detailed reasoning but consume more tokens.
    ///
    /// # Arguments
    /// * `budget` - Number of tokens (128-32768, default varies by model size)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking_budget(8192)  // 8K tokens for complex reasoning
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        let patch =
            crate::standards::openai::compat::reasoning::OpenAiCompatibleReasoningPolicy::for_provider(
                &self.provider_id,
            )
            .thinking_budget(budget);
        self = self.apply_reasoning_patch(patch);
        self
    }

    /// Enable reasoning mode for supported models
    ///
    /// This is the unified reasoning interface that works across all OpenAI-compatible providers.
    /// Different providers handle reasoning differently:
    /// - DeepSeek: Maps to `thinking.type`
    /// - OpenRouter: Passes through to underlying model's reasoning capabilities
    /// - SiliconFlow: Maps to `enable_thinking` parameter
    /// - Qwen/Alibaba: Maps to `enable_thinking` parameter
    /// - MoonshotAI: Maps to `thinking.type`
    ///
    /// # Arguments
    /// * `enable` - Whether to enable reasoning output
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .deepseek()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-reasoner")
    ///         .reasoning(true)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning(mut self, enable: bool) -> Self {
        let patch =
            crate::standards::openai::compat::reasoning::OpenAiCompatibleReasoningPolicy::for_provider(
                &self.provider_id,
            )
            .reasoning(enable);
        self = self.apply_reasoning_patch(patch);
        self
    }

    /// Set reasoning budget
    ///
    /// This controls how many tokens the model can use for its internal reasoning process.
    /// Different providers interpret this differently:
    /// - SiliconFlow: Maps to `thinking_budget` parameter
    /// - Qwen/Alibaba: Maps to `thinking_budget` parameter
    /// - DeepSeek: Maps to `thinking.type` and ignores token budget
    /// - OpenRouter: Passed through to underlying model
    /// - MoonshotAI: Maps to `thinking.budget_tokens` and enables thinking
    ///
    /// # Arguments
    /// * `budget` - Number of tokens for reasoning (128-32768)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .reasoning_budget(8192)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning_budget(mut self, budget: i32) -> Self {
        let patch =
            crate::standards::openai::compat::reasoning::OpenAiCompatibleReasoningPolicy::for_provider(
                &self.provider_id,
            )
            .reasoning_budget(budget);
        self = self.apply_reasoning_patch(patch);
        self
    }
}
