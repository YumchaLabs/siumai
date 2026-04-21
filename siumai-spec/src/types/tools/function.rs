//! Function tool schema.

use serde::{Deserialize, Serialize};

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Function name
    pub name: String,
    /// Optional display title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Function description
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// JSON schema for function parameters
    #[serde(rename = "inputSchema", alias = "parameters", alias = "input_schema")]
    pub parameters: serde_json::Value,

    /// Optional JSON schema for function outputs (AI SDK-aligned user-facing tool metadata).
    ///
    /// Provider request converters intentionally do not forward this field as-is today; it is
    /// stored on the shared stable tool shape so higher-level helpers can keep parity with AI
    /// SDK's `Tool.outputSchema` contract without changing provider wire payloads.
    #[serde(
        default,
        rename = "outputSchema",
        alias = "output_schema",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_schema: Option<serde_json::Value>,

    /// Tool input examples (Vercel-aligned).
    ///
    /// Vercel's AI SDK accepts `inputExamples: [{ input: {...} }]` on function tools and
    /// forwards them to Anthropic as `input_examples: [{...}]`.
    #[serde(
        default,
        rename = "inputExamples",
        alias = "input_examples",
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

impl ToolFunction {
    /// Create a new function-tool schema with the canonical stable storage fields.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            title: None,
            description: description.into(),
            parameters,
            output_schema: None,
            input_examples: None,
            strict: None,
            provider_options_map: crate::types::ProviderOptionsMap::default(),
        }
    }

    /// Optional display title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Attach a display title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// AI SDK-style view over the input schema.
    pub fn input_schema(&self) -> &serde_json::Value {
        &self.parameters
    }

    /// Mutable AI SDK-style view over the input schema.
    pub fn input_schema_mut(&mut self) -> &mut serde_json::Value {
        &mut self.parameters
    }

    /// Replace the input schema using AI SDK naming.
    pub fn with_input_schema(mut self, input_schema: serde_json::Value) -> Self {
        self.parameters = input_schema;
        self
    }

    /// Optional AI SDK-style output schema metadata.
    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        self.output_schema.as_ref()
    }

    /// Mutable AI SDK-style output schema metadata.
    pub fn output_schema_mut(&mut self) -> Option<&mut serde_json::Value> {
        self.output_schema.as_mut()
    }

    /// Attach AI SDK-style output schema metadata.
    pub fn with_output_schema(mut self, output_schema: serde_json::Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    /// Optional AI SDK-style input examples metadata.
    pub fn input_examples(&self) -> Option<&[serde_json::Value]> {
        self.input_examples.as_deref()
    }

    /// Attach AI SDK-style input examples metadata.
    pub fn with_input_examples(
        mut self,
        input_examples: impl IntoIterator<Item = serde_json::Value>,
    ) -> Self {
        self.input_examples = Some(input_examples.into_iter().collect());
        self
    }

    /// Optional strict-mode setting.
    pub const fn strict(&self) -> Option<bool> {
        self.strict
    }

    /// Attach a strict-mode setting.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Borrow the tool-level provider options map.
    pub fn provider_options_map(&self) -> &crate::types::ProviderOptionsMap {
        &self.provider_options_map
    }

    /// Mutably borrow the tool-level provider options map.
    pub fn provider_options_map_mut(&mut self) -> &mut crate::types::ProviderOptionsMap {
        &mut self.provider_options_map
    }

    /// Replace the tool-level provider options map.
    pub fn with_provider_options_map(
        mut self,
        provider_options_map: crate::types::ProviderOptionsMap,
    ) -> Self {
        self.provider_options_map = provider_options_map;
        self
    }
}
