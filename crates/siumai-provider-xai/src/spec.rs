//! xAI core-level provider spec implementation.
//!
//! This module provides a `CoreProviderSpec` implementation based on
//! `siumai-core` / `siumai-std-openai`, which can be consumed by the
//! aggregator via feature gates.

use siumai_core::error::LlmError;
use siumai_core::execution::chat::ChatInput;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
use siumai_core::traits::ProviderCapabilities;
use siumai_std_openai::openai::chat::{OpenAiChatAdapter, OpenAiChatStandard};
use std::sync::Arc;

/// xAI `CoreProviderSpec` implementation.
#[derive(Clone)]
pub struct XaiCoreSpec {
    chat_standard: OpenAiChatStandard,
}

impl Default for XaiCoreSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl XaiCoreSpec {
    /// Create a new spec using the OpenAI Chat standard with an xAI-specific adapter.
    pub fn new() -> Self {
        let adapter = Arc::new(XaiOpenAiChatAdapter::default());
        Self {
            chat_standard: OpenAiChatStandard::with_adapter(adapter),
        }
    }
}

/// xAI-specific OpenAI Chat adapter.
///
/// This adapter reads xAI-related parameters from `ChatInput::extra`:
/// - `xai_search_parameters`: pre-built search configuration object
/// - `xai_reasoning_effort`: reasoning effort string
/// and injects them into the final request JSON.
#[derive(Clone, Default)]
struct XaiOpenAiChatAdapter;

impl OpenAiChatAdapter for XaiOpenAiChatAdapter {
    fn transform_request(
        &self,
        input: &ChatInput,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // Merge pre-encoded search_parameters from ChatInput::extra into the request body.
        if let Some(params) = input.extra.get("xai_search_parameters") {
            body["search_parameters"] = params.clone();
        }

        // Inject reasoning_effort if present.
        if let Some(effort) = input.extra.get("xai_reasoning_effort") {
            body["reasoning_effort"] = effort.clone();
        }

        Ok(())
    }
}

impl CoreProviderSpec for XaiCoreSpec {
    fn id(&self) -> &'static str {
        "xai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(
        &self,
        ctx: &CoreProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("xAI API key not provided".into()))?;

        crate::headers::build_xai_json_headers(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, ctx: &CoreProviderContext) -> String {
        // Base URL already includes "/v1" (see aggregator config). Append only the
        // operation path to avoid duplicating the prefix.
        format!("{}/chat/completions", ctx.base_url.trim_end_matches('/'))
    }

    fn choose_chat_transformers(
        &self,
        _input: &ChatInput,
        ctx: &CoreProviderContext,
    ) -> CoreChatTransformers {
        // Reuse the OpenAI Chat standard for xAI. Provider-specific options
        // (search_parameters, reasoning_effort, etc.) are handled via
        // higher-level hooks in the aggregator.
        let req = self
            .chat_standard
            .create_request_transformer(&ctx.provider_id);
        let resp = self
            .chat_standard
            .create_response_transformer(&ctx.provider_id);
        let stream = self.chat_standard.create_stream_converter(&ctx.provider_id);

        CoreChatTransformers {
            request: req,
            response: resp,
            stream: Some(stream),
        }
    }

    fn map_core_stream_event(&self, event: ChatStreamEventCore) -> ChatStreamEventCore {
        // No additional processing for now; pass through.
        event
    }
}
