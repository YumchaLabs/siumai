//! Function tool schema.

use serde::{Deserialize, Serialize};

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Function name
    pub name: String,
    /// Function description
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// JSON schema for function parameters
    pub parameters: serde_json::Value,

    /// Tool input examples (Vercel-aligned).
    ///
    /// Vercel's AI SDK accepts `inputExamples: [{ input: {...} }]` on function tools and
    /// forwards them to Anthropic as `input_examples: [{...}]`.
    #[serde(
        default,
        rename = "inputExamples",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_examples: Option<Vec<serde_json::Value>>,

    /// Strict mode setting for the tool (Vercel-aligned).
    ///
    /// Providers that support strict mode will use this setting to determine
    /// how the input should be generated. Strict mode will always produce
    /// valid inputs, but it might limit what input schemas the model can use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    /// Tool-level provider options (Vercel-aligned).
    ///
    /// This is useful for provider-specific tool configuration knobs such as
    /// Anthropic's `defer_loading` for function tools.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "crate::types::ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: crate::types::ProviderOptionsMap,
}
