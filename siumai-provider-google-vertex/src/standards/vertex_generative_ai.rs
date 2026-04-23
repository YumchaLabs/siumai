//! Vertex AI chat standard (Gemini generateContent via Vertex endpoints).
//!
//! This module intentionally reuses `siumai-protocol-gemini` for request/response mapping
//! while implementing Vertex-specific auth behavior:
//! - Express mode: API key is passed as `?key=...` query param (no `x-goog-api-key` header).
//! - Enterprise mode: Bearer token via `Authorization: Bearer ...`.

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
use siumai_protocol_gemini::standards::gemini::types::GeminiConfig;
use std::collections::HashMap;

const VERTEX_USER_AGENT: &str = concat!("siumai/google-vertex/", env!("CARGO_PKG_VERSION"));

fn has_auth_header(headers: &HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"))
}

fn append_api_key_query(url: String, api_key: &str) -> String {
    let key = urlencoding::encode(api_key);
    if url.contains('?') {
        format!("{url}&key={key}")
    } else {
        format!("{url}?key={key}")
    }
}

fn build_vertex_headers(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_user_agent(VERTEX_USER_AGENT)?
        .with_custom_headers(custom_headers)?;
    Ok(builder.build())
}

/// Vertex Generative AI (Gemini via Vertex) chat standard.
#[derive(Clone, Default)]
pub struct VertexGenerativeAiStandard {
    gemini_config: Option<GeminiConfig>,
}

impl VertexGenerativeAiStandard {
    pub fn new() -> Self {
        Self {
            gemini_config: None,
        }
    }

    pub fn with_gemini_config(mut self, gemini_config: GeminiConfig) -> Self {
        self.gemini_config = Some(gemini_config);
        self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> VertexGenerativeAiSpec {
        VertexGenerativeAiSpec {
            provider_id,
            gemini_config: self.gemini_config.clone(),
        }
    }
}

pub struct VertexGenerativeAiSpec {
    provider_id: &'static str,
    gemini_config: Option<GeminiConfig>,
}

impl ProviderSpec for VertexGenerativeAiSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_vertex_headers(&ctx.http_extra_headers)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let mut standard = siumai_protocol_gemini::standards::gemini::GeminiChatStandard::new();
        if let Some(gemini_config) = self.gemini_config.clone() {
            standard = standard.with_base_config(gemini_config);
        }
        standard.create_transformers_with_model_and_stream_options(
            &ctx.provider_id,
            Some(&req.common_params.model),
            req.stream_options.include_raw_chunks,
        )
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let base = siumai_protocol_gemini::standards::gemini::GeminiChatStandard::new()
            .create_spec(self.provider_id)
            .chat_url(stream, req, ctx);

        if let Some(key) = ctx.api_key.as_deref()
            && !key.is_empty()
            && !has_auth_header(&ctx.http_extra_headers)
        {
            append_api_key_query(base, key)
        } else {
            base
        }
    }
}
