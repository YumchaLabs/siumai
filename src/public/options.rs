//! Typed ProviderOptions (optional)
//!
//! Optional typed options that convert into `ProviderParams` for use with the
//! unified API. Mirrors an options-factory pattern without forcing typed
//! accessors at the top level.

use crate::types::ProviderParams;

#[derive(Debug, Clone, Default)]
pub struct OpenAiOptions {
    pub responses_api: Option<bool>,
    pub response_format: Option<String>,
}

impl From<OpenAiOptions> for ProviderParams {
    fn from(o: OpenAiOptions) -> Self {
        let mut p = ProviderParams::openai();
        if let Some(v) = o.responses_api {
            p = p.with_param("responses_api", v);
        }
        if let Some(v) = o.response_format {
            p = p.with_param("response_format", v);
        }
        p
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    pub thinking_budget: Option<u32>,
    pub cache_control: Option<serde_json::Value>,
}

impl From<AnthropicOptions> for ProviderParams {
    fn from(o: AnthropicOptions) -> Self {
        let mut p = ProviderParams::anthropic();
        if let Some(v) = o.thinking_budget {
            p = p.with_param("thinking_budget", v);
        }
        if let Some(v) = o.cache_control {
            p = p.with_param("cache_control", v);
        }
        p
    }
}

#[derive(Debug, Clone, Default)]
pub struct GeminiOptions {
    pub thinking_budget: Option<i32>,
}

impl From<GeminiOptions> for ProviderParams {
    fn from(o: GeminiOptions) -> Self {
        let mut p = ProviderParams::gemini();
        if let Some(v) = o.thinking_budget {
            p = p.with_param("thinking_budget", v);
        }
        p
    }
}

#[derive(Debug, Clone, Default)]
pub struct OpenAiCompatOptions {
    pub provider_id: Option<String>,
}

impl From<OpenAiCompatOptions> for ProviderParams {
    fn from(_o: OpenAiCompatOptions) -> Self {
        // Placeholder; compat options often flow through adapter/config
        ProviderParams::new()
    }
}

/// Convenience: create OpenAI provider params from typed options
pub fn create_openai_options(opts: OpenAiOptions) -> ProviderParams {
    opts.into()
}

/// Convenience: create Anthropic provider params from typed options
pub fn create_anthropic_options(opts: AnthropicOptions) -> ProviderParams {
    opts.into()
}

/// Convenience: create Gemini provider params from typed options
pub fn create_gemini_options(opts: GeminiOptions) -> ProviderParams {
    opts.into()
}

/// Merge multiple ProviderParams with right-most precedence
pub fn merge_provider_options<I>(iter: I) -> ProviderParams
where
    I: IntoIterator<Item = ProviderParams>,
{
    let mut acc = ProviderParams::new();
    for p in iter {
        acc = acc.merge(p);
    }
    acc
}
