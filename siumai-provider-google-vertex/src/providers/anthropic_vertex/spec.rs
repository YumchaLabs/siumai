//! Vertex Anthropic Provider Spec
//!
//! Defines the ProviderSpec implementation for Anthropic on Vertex AI.

use std::sync::Arc;

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::ChatRequest;
use reqwest::header::HeaderMap;
use std::collections::{HashMap, HashSet};

use siumai_protocol_anthropic::standards::anthropic;

/// Vertex Anthropic provider specification
#[derive(Clone)]
pub struct VertexAnthropicSpec {
    pub base_url: String,
    pub model: String,
    pub extra_headers: std::collections::HashMap<String, String>,
}

impl VertexAnthropicSpec {
    pub fn new(
        base_url: String,
        model: String,
        extra_headers: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            base_url,
            model,
            extra_headers,
        }
    }
}

impl ProviderSpec for VertexAnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic-vertex"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }

    fn build_headers(
        &self,
        _ctx: &ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let builder = crate::execution::http::headers::HttpHeaderBuilder::new()
            .with_json_content_type()
            .with_custom_headers(&self.extra_headers)?;
        Ok(builder.build())
    }

    fn merge_request_headers(
        &self,
        mut base: HeaderMap,
        extra: &HashMap<String, String>,
    ) -> HeaderMap {
        fn merge_comma_separated_tokens(a: &str, b: &str) -> String {
            let mut seen: HashSet<String> = HashSet::new();
            let mut out: Vec<String> = Vec::new();

            for raw in a.split(',').chain(b.split(',')) {
                let token = raw.trim();
                if token.is_empty() {
                    continue;
                }
                if seen.insert(token.to_string()) {
                    out.push(token.to_string());
                }
            }

            out.join(",")
        }

        for (k, v) in extra {
            if k.eq_ignore_ascii_case("anthropic-beta") {
                let existing = base
                    .get("anthropic-beta")
                    .and_then(|hv| hv.to_str().ok())
                    .unwrap_or("");
                let merged = merge_comma_separated_tokens(existing, v);
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::from_bytes(b"anthropic-beta"),
                    reqwest::header::HeaderValue::from_str(&merged),
                ) {
                    base.insert(name, val);
                }
                continue;
            }

            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                base.insert(name, val);
            }
        }

        base
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        anthropic::errors::classify_anthropic_http_error(self.id(), status, body_text)
    }

    fn chat_url(&self, stream: bool, _req: &ChatRequest, _ctx: &ProviderContext) -> String {
        if stream {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:streamRawPredict?alt=sse", self.model),
            )
        } else {
            crate::utils::url::join_url(
                &self.base_url,
                &format!("models/{}:rawPredict", self.model),
            )
        }
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let stream_transformer = if req.stream {
            let converter = anthropic::streaming::AnthropicEventConverter::new(
                anthropic::params::AnthropicParams::default(),
            );
            let stream_tx = anthropic::transformers::AnthropicStreamChunkTransformer {
                provider_id: "anthropic-vertex".to_string(),
                inner: converter,
            };
            Some(Arc::new(stream_tx)
                as Arc<
                    dyn crate::execution::transformers::stream::StreamChunkTransformer,
                >)
        } else {
            None
        };

        ChatTransformers {
            request: Arc::new(anthropic::transformers::AnthropicRequestTransformer::new(
                None,
            )),
            response: Arc::new(anthropic::transformers::AnthropicResponseTransformer),
            stream: stream_transformer,
            json: None,
        }
    }
}
