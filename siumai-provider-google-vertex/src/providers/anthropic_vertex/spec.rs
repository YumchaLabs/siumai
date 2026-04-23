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

const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

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

    fn base_url_ends_with_models(&self) -> bool {
        let base = self.base_url.trim_end_matches('/');
        base.ends_with("/models")
    }

    fn chat_path(&self, stream: bool) -> String {
        if self.base_url_ends_with_models() {
            format!(
                "{}:{}",
                self.model,
                if stream {
                    "streamRawPredict"
                } else {
                    "rawPredict"
                }
            )
        } else {
            format!(
                "models/{}:{}",
                self.model,
                if stream {
                    "streamRawPredict"
                } else {
                    "rawPredict"
                }
            )
        }
    }

    fn anthropic_thinking_config_from_provider_options_map(
        &self,
        req: &ChatRequest,
    ) -> Option<anthropic::thinking::ThinkingConfig> {
        let value = req.provider_options_map.get("anthropic")?;
        Self::anthropic_thinking_config_from_provider_options_value(value)
    }

    fn anthropic_thinking_config_from_provider_options_value(
        value: &serde_json::Value,
    ) -> Option<anthropic::thinking::ThinkingConfig> {
        let obj = value.as_object()?;

        if let Some(thinking) = obj.get("thinking").and_then(|v| v.as_object()) {
            let enabled = thinking
                .get("type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == "enabled");
            if enabled {
                let budget_tokens = thinking
                    .get("budget_tokens")
                    .or_else(|| thinking.get("budgetTokens"))
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(1024);

                return Some(anthropic::thinking::ThinkingConfig {
                    r#type: "enabled".to_string(),
                    budget_tokens,
                });
            }
        }

        let thinking_mode = obj
            .get("thinking_mode")
            .or_else(|| obj.get("thinkingMode"))
            .and_then(|v| v.as_object())?;

        let enabled = thinking_mode
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !enabled {
            return None;
        }

        let budget_tokens = thinking_mode
            .get("thinking_budget")
            .or_else(|| thinking_mode.get("thinkingBudget"))
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(1024);

        Some(anthropic::thinking::ThinkingConfig {
            r#type: "enabled".to_string(),
            budget_tokens,
        })
    }

    fn configured_structured_output_mode(
        req: &ChatRequest,
    ) -> Option<anthropic::params::StructuredOutputMode> {
        let obj = req.provider_options_map.get("anthropic")?.as_object()?;
        let mode = obj
            .get("structured_output_mode")
            .or_else(|| obj.get("structuredOutputMode"))
            .and_then(|value| value.as_str())?;

        match mode {
            "outputFormat" | "output_format" | "output-format" => {
                Some(anthropic::params::StructuredOutputMode::OutputFormat)
            }
            "jsonTool" | "json_tool" | "json-tool" => {
                Some(anthropic::params::StructuredOutputMode::JsonTool)
            }
            _ => None,
        }
    }

    fn effective_structured_output_mode(
        req: &ChatRequest,
    ) -> Option<anthropic::params::StructuredOutputMode> {
        if !matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::Json { .. })
        ) {
            return None;
        }

        match Self::configured_structured_output_mode(req) {
            Some(anthropic::params::StructuredOutputMode::OutputFormat) => {
                Some(anthropic::params::StructuredOutputMode::OutputFormat)
            }
            _ => Some(anthropic::params::StructuredOutputMode::JsonTool),
        }
    }

    fn request_with_effective_structured_output_mode(req: &ChatRequest) -> ChatRequest {
        let Some(mode) = Self::effective_structured_output_mode(req) else {
            return req.clone();
        };
        if mode != anthropic::params::StructuredOutputMode::JsonTool {
            return req.clone();
        }

        let mut out = req.clone();
        match out.provider_options_map.get("anthropic").cloned() {
            Some(serde_json::Value::Object(mut obj)) => {
                obj.insert(
                    "structured_output_mode".to_string(),
                    serde_json::json!("jsonTool"),
                );
                out.provider_options_map
                    .insert("anthropic", serde_json::Value::Object(obj));
            }
            _ => {
                out.provider_options_map.insert(
                    "anthropic",
                    serde_json::json!({
                        "structured_output_mode": "jsonTool"
                    }),
                );
            }
        }

        out
    }
}

#[derive(Clone)]
struct VertexAnthropicRequestTransformer {
    provider_id: &'static str,
    inner: anthropic::transformers::AnthropicRequestTransformer,
}

impl VertexAnthropicRequestTransformer {
    fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: anthropic::transformers::AnthropicRequestTransformer::new(None),
        }
    }
}

impl crate::execution::transformers::request::RequestTransformer
    for VertexAnthropicRequestTransformer
{
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let req = VertexAnthropicSpec::request_with_effective_structured_output_mode(req);
        let mut body = self.inner.transform_chat(&req)?;
        let Some(obj) = body.as_object_mut() else {
            return Err(LlmError::ParseError(
                "Anthropic transformer did not return a JSON object".to_string(),
            ));
        };

        // Vertex Anthropic: model id is carried in the URL, not in the request body.
        obj.remove("model");
        // Vertex Anthropic: uses a request-body field instead of the `anthropic-version` header.
        obj.insert(
            "anthropic_version".to_string(),
            serde_json::json!(VERTEX_ANTHROPIC_VERSION),
        );

        Ok(body)
    }
}

impl ProviderSpec for VertexAnthropicSpec {
    fn id(&self) -> &'static str {
        "anthropic-vertex"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = self.extra_headers.clone();
        headers.extend(ctx.http_extra_headers.clone());
        let builder = crate::execution::http::headers::HttpHeaderBuilder::new()
            .with_json_content_type()
            .with_custom_headers(&headers)?;
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
        crate::utils::url::join_url(&self.base_url, &self.chat_path(stream))
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let stream_transformer = if req.stream {
            let mut stream_params = anthropic::params::AnthropicParams::default();
            if let Some(mode) = Self::effective_structured_output_mode(req) {
                stream_params = stream_params.with_structured_output_mode(mode);
            }
            let converter = anthropic::streaming::AnthropicEventConverter::new(stream_params);
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
            request: Arc::new(VertexAnthropicRequestTransformer::new("anthropic-vertex")),
            response: Arc::new(anthropic::transformers::AnthropicResponseTransformer::default()),
            stream: stream_transformer,
            json: None,
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        let thinking = self.anthropic_thinking_config_from_provider_options_map(req)?;

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            out["thinking"] = thinking.to_request_params();

            if let Some(obj) = out.as_object_mut() {
                obj.remove("temperature");
                obj.remove("top_p");
                obj.remove("top_k");
            }

            if let Some(max_tokens) = out.get("max_tokens").and_then(|v| v.as_u64()) {
                out["max_tokens"] =
                    serde_json::json!(max_tokens.saturating_add(thinking.budget_tokens as u64));
            }

            Ok(out)
        };

        Some(Arc::new(hook))
    }

    fn models_url(&self, ctx: &ProviderContext) -> String {
        if self.base_url_ends_with_models() {
            ctx.base_url.trim_end_matches('/').to_string()
        } else {
            format!("{}/models", ctx.base_url.trim_end_matches('/'))
        }
    }

    fn model_url(&self, model_id: &str, ctx: &ProviderContext) -> String {
        if self.base_url_ends_with_models() {
            format!("{}/{}", ctx.base_url.trim_end_matches('/'), model_id)
        } else {
            format!("{}/models/{}", ctx.base_url.trim_end_matches('/'), model_id)
        }
    }
}
