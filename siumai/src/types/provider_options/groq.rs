//! Groq Provider Options
//!
//! This module contains types for Groq-specific features.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Groq-specific options (typed providerOptions).
///
/// 这一层对齐 Vercel 的 `groqProviderOptions` 设计：
/// - `reasoning_effort`: "none" | "default" | "low" | "medium" | "high"
/// - `reasoning_format`: "hidden" | "raw" | "parsed"
/// - `parallel_tool_calls`: bool
/// - `service_tier`: "on_demand" | "flex" | "auto"
///
/// 对 Groq 原生 API 而言，这些字段直接映射到 JSON 顶层；额外的自定义参数
/// 可以通过 `extra_params` 传入，并最终展开到请求体中。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroqOptions {
    /// Reasoning effort level for qwen3/reasoning-style models.
    ///
    /// 官方允许的取值见：https://console.groq.com/docs/reasoning#reasoning-effort
    pub reasoning_effort: Option<String>,
    /// Reasoning output format ("hidden" | "raw" | "parsed").
    pub reasoning_format: Option<String>,
    /// Whether to enable parallel function calling.
    pub parallel_tool_calls: Option<bool>,
    /// Service tier ("on_demand" | "flex" | "auto").
    pub service_tier: Option<String>,
    /// Additional Groq-specific parameters (escape hatch).
    ///
    /// 仅在确有需要时使用，类型安全的字段优先。
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GroqOptions {
    /// Create new Groq options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning effort
    pub fn with_reasoning_effort<S: Into<String>>(mut self, effort: S) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Set reasoning format
    pub fn with_reasoning_format<S: Into<String>>(mut self, fmt: S) -> Self {
        self.reasoning_format = Some(fmt.into());
        self
    }

    /// Enable or disable parallel tool calls
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set service tier
    pub fn with_service_tier<S: Into<String>>(mut self, tier: S) -> Self {
        self.service_tier = Some(tier.into());
        self
    }

    /// Add a raw custom parameter (escape hatch)
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}
