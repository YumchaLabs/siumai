//! Anthropic standard parameters for streaming conversion.

/// Structured output mode inferred from the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputMode {
    /// Request-level `responseFormat: { type: "json", schema }` is implemented via the reserved
    /// `json` tool (unsupported models).
    JsonTool,
    /// Request-level `responseFormat: { type: "json", schema }` is implemented via `output_format`
    /// (supported models).
    OutputFormat,
}

/// Config for Anthropic streaming conversion.
#[derive(Debug, Clone, Default)]
pub struct AnthropicParams {
    pub structured_output_mode: Option<StructuredOutputMode>,
}

impl AnthropicParams {
    pub fn with_structured_output_mode(mut self, mode: StructuredOutputMode) -> Self {
        self.structured_output_mode = Some(mode);
        self
    }
}
