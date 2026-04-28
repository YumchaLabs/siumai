//! Completion model family types.

use serde::{Deserialize, Serialize};

use crate::types::{
    ChatMessage, CommonParams, FinishReason, HttpConfig, ProviderMetadataMap, ProviderOptionsMap,
    ResponseFormat, ResponseMetadata, StreamRequestOptions, Tool, ToolChoice, Usage, Warning,
};

/// Completion request aligned with AI SDK completion-model call options.
///
/// Unlike the chat family, completion models materialize a text prompt from a prompt-shaped
/// message list before sending the provider request. The stable Rust request keeps that prompt in
/// structured form so providers can apply AI SDK-compatible materialization and unsupported-option
/// warnings on the provider boundary.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionRequest {
    /// Prompt messages to materialize into a text completion prompt.
    pub prompt: Vec<ChatMessage>,

    /// Optional tools carried for AI SDK call-option parity.
    ///
    /// Providers whose completion endpoints do not support tools should surface warnings or errors
    /// rather than silently inventing chat-specific behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Tool choice carried for AI SDK call-option parity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Shared generation parameters such as model, temperature, stop sequences, and seed.
    pub common_params: CommonParams,

    /// Request-level response format hints.
    #[serde(
        default,
        rename = "responseFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub response_format: Option<ResponseFormat>,

    /// Open provider options map (AI SDK-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,

    /// Per-request HTTP configuration.
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,

    /// Runtime-only stream options.
    #[serde(skip)]
    pub stream_options: StreamRequestOptions,
}

impl CompletionRequest {
    /// Create a completion request from a single user prompt string.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: vec![ChatMessage::user(prompt).build()],
            tools: None,
            tool_choice: None,
            common_params: CommonParams::default(),
            response_format: None,
            provider_options_map: ProviderOptionsMap::default(),
            http_config: None,
            stream_options: StreamRequestOptions::default(),
        }
    }

    /// Create a completion request from a structured prompt.
    pub fn from_prompt(prompt: Vec<ChatMessage>) -> Self {
        Self {
            prompt,
            ..Default::default()
        }
    }

    /// Replace the prompt messages.
    pub fn with_prompt(mut self, prompt: Vec<ChatMessage>) -> Self {
        self.prompt = prompt;
        self
    }

    /// Set the target model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p sampling.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    pub fn with_top_k(mut self, top_k: f64) -> Self {
        self.common_params.top_k = Some(top_k);
        self
    }

    /// Set deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set frequency penalty.
    pub fn with_frequency_penalty(mut self, frequency_penalty: f64) -> Self {
        self.common_params.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set presence penalty.
    pub fn with_presence_penalty(mut self, presence_penalty: f64) -> Self {
        self.common_params.presence_penalty = Some(presence_penalty);
        self
    }

    /// Attach tools.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set tool choice.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set response format.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Replace the full provider options map.
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set per-request HTTP config.
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Set runtime-only stream options.
    pub fn with_stream_options(mut self, options: StreamRequestOptions) -> Self {
        self.stream_options = options;
        self
    }

    /// Enable or disable raw chunk emission for streaming calls.
    pub fn with_include_raw_chunks(mut self, include_raw_chunks: bool) -> Self {
        self.stream_options.include_raw_chunks = include_raw_chunks;
        self
    }

    /// Add a single request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut http = self.http_config.take().unwrap_or_else(HttpConfig::empty);
        http.headers.insert(key.into(), value.into());
        self.http_config = Some(http);
        self
    }

    /// Fill the model if the request does not already specify one.
    pub fn with_model_if_missing(mut self, model: impl Into<String>) -> Self {
        if self.common_params.model.trim().is_empty() {
            self.common_params.model = model.into();
        }
        self
    }
}

/// Completion response produced by a completion-family model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionResponse {
    /// Generated text.
    pub text: String,

    /// Finish reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Provider-native finish reason when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,

    /// Token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// AI SDK-style response metadata.
    #[serde(
        default,
        rename = "responseMetadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub response_metadata: Option<ResponseMetadata>,

    /// Provider warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Warning>>,

    /// Provider-scoped response metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadataMap>,
}

impl CompletionResponse {
    /// Create a new completion response from text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// Borrow the generated text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Borrow the response id if present.
    pub fn id(&self) -> Option<&str> {
        self.response_metadata
            .as_ref()
            .and_then(|metadata| metadata.id.as_deref())
    }

    /// Borrow the resolved model id if present.
    pub fn model(&self) -> Option<&str> {
        self.response_metadata
            .as_ref()
            .and_then(|metadata| metadata.model.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn completion_response_serializes_raw_finish_reason() {
        let response = CompletionResponse {
            text: "done".to_string(),
            finish_reason: Some(FinishReason::Stop),
            raw_finish_reason: Some("stop".to_string()),
            ..Default::default()
        };

        let value = serde_json::to_value(response).expect("serialize completion response");

        assert_eq!(value["finish_reason"], json!("stop"));
        assert_eq!(value["raw_finish_reason"], json!("stop"));
    }

    #[test]
    fn completion_response_deserializes_missing_raw_finish_reason() {
        let response: CompletionResponse =
            serde_json::from_value(json!({ "text": "done" })).expect("completion response");

        assert_eq!(response.raw_finish_reason, None);
    }
}
