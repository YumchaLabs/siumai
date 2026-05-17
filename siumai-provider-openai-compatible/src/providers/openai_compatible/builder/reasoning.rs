use super::OpenAiCompatibleBuilder;

fn provider_thinking_value(enable: bool, budget_tokens: Option<u32>) -> serde_json::Value {
    let mut thinking = serde_json::Map::new();
    thinking.insert(
        "type".to_string(),
        serde_json::Value::String(if enable { "enabled" } else { "disabled" }.to_string()),
    );
    if let Some(budget_tokens) = budget_tokens {
        thinking.insert(
            "budget_tokens".to_string(),
            serde_json::Value::Number(serde_json::Number::from(budget_tokens)),
        );
    }
    serde_json::Value::Object(thinking)
}

impl OpenAiCompatibleBuilder {
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
        match self.provider_id.as_str() {
            "deepseek" | "moonshot" | "moonshotai" => {
                self.provider_specific_config.insert(
                    "thinking".to_string(),
                    provider_thinking_value(enable, None),
                );
            }
            "xai" => {
                if enable {
                    self.provider_specific_config
                        .insert("reasoning_effort".to_string(), serde_json::json!("low"));
                } else {
                    self.provider_specific_config.remove("reasoning_effort");
                    self.provider_specific_config.remove("reasoningEffort");
                }
            }
            _ => {
                self.provider_specific_config.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
        }
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
        let clamped_budget = budget.clamp(128, 32768);
        match self.provider_id.as_str() {
            "deepseek" => {
                self.provider_specific_config
                    .insert("thinking".to_string(), provider_thinking_value(true, None));
            }
            "moonshot" | "moonshotai" => {
                self.provider_specific_config.insert(
                    "thinking".to_string(),
                    provider_thinking_value(true, Some(clamped_budget)),
                );
            }
            "xai" => {
                self.provider_specific_config
                    .insert("reasoning_effort".to_string(), serde_json::json!("high"));
            }
            _ => {
                self.provider_specific_config.insert(
                    "thinking_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
            }
        }
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
        match self.provider_id.as_str() {
            "siliconflow" | "qwen" | "alibaba" => {
                self.provider_specific_config.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
            "moonshot" | "moonshotai" => {
                let mut thinking = self
                    .provider_specific_config
                    .remove("thinking")
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                thinking.insert(
                    "type".to_string(),
                    serde_json::Value::String(if enable { "enabled" } else { "disabled" }.into()),
                );
                self.provider_specific_config
                    .insert("thinking".to_string(), serde_json::Value::Object(thinking));
            }
            "deepseek" => {
                self.provider_specific_config.insert(
                    "thinking".to_string(),
                    provider_thinking_value(enable, None),
                );
            }
            "xai" => {
                if enable {
                    self.provider_specific_config
                        .insert("reasoning_effort".to_string(), serde_json::json!("low"));
                } else {
                    self.provider_specific_config.remove("reasoning_effort");
                    self.provider_specific_config.remove("reasoningEffort");
                }
            }
            "openrouter" => {
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
        }
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
        let clamped_budget = budget.clamp(128, 32768) as u32;

        match self.provider_id.as_str() {
            "siliconflow" | "qwen" | "alibaba" => {
                self.provider_specific_config.insert(
                    "thinking_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
                self.provider_specific_config
                    .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            }
            "moonshot" | "moonshotai" => {
                let mut thinking = self
                    .provider_specific_config
                    .remove("thinking")
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                thinking.insert(
                    "type".to_string(),
                    serde_json::Value::String("enabled".to_string()),
                );
                thinking.insert(
                    "budget_tokens".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
                self.provider_specific_config
                    .insert("thinking".to_string(), serde_json::Value::Object(thinking));
            }
            "deepseek" => {
                self.provider_specific_config
                    .insert("thinking".to_string(), provider_thinking_value(true, None));
            }
            "xai" => {
                self.provider_specific_config
                    .insert("reasoning_effort".to_string(), serde_json::json!("high"));
            }
            "openrouter" => {
                self.provider_specific_config.insert(
                    "reasoning_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(true),
                );
            }
            _ => {
                self.provider_specific_config.insert(
                    "reasoning_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
            }
        }
        self
    }
}
