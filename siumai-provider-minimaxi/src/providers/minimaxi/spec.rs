//! MiniMaxi ProviderSpec Implementation

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::transformers::response::ResponseTransformer;
use crate::execution::transformers::stream::{StreamChunkTransformer, StreamEventFuture};
use crate::provider_options::{MinimaxiOptions, MinimaxiResponseFormat};
use crate::standards::anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
use crate::traits::ProviderCapabilities;
use reqwest::header::HeaderMap;
use std::sync::Arc;

fn resolve_openai_base_url(base_url: &str) -> String {
    format!(
        "{}/v1",
        super::utils::resolve_api_root_base_url(base_url).trim_end_matches('/')
    )
}

fn build_openai_like_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let api_key = ctx
        .api_key
        .as_ref()
        .ok_or_else(|| LlmError::MissingApiKey("MiniMaxi API key not provided".into()))?;

    let mut builder = HttpHeaderBuilder::new()
        .with_bearer_auth(api_key)?
        .with_json_content_type();

    if let Some(org) = ctx.organization.as_deref() {
        builder = builder.with_header("OpenAI-Organization", org)?;
    }
    if let Some(proj) = ctx.project.as_deref() {
        builder = builder.with_header("OpenAI-Project", proj)?;
    }

    builder = builder.with_custom_headers(&ctx.http_extra_headers)?;
    Ok(builder.build())
}

const ANTHROPIC_METADATA_KEY: &str = "anthropic";
const MINIMAXI_METADATA_KEY: &str = "minimaxi";
const MINIMAXI_OPTIONS_KEY: &str = "minimaxi";
const LEGACY_ANTHROPIC_OPTIONS_KEY: &str = "anthropic";

fn normalize_response_provider_metadata(response: &mut crate::types::ChatResponse) {
    let Some(provider_metadata) = response.provider_metadata.as_mut() else {
        return;
    };

    if provider_metadata.contains_key(MINIMAXI_METADATA_KEY) {
        provider_metadata.remove(ANTHROPIC_METADATA_KEY);
    } else if let Some(anthropic) = provider_metadata.remove(ANTHROPIC_METADATA_KEY) {
        provider_metadata.insert(MINIMAXI_METADATA_KEY.to_string(), anthropic);
    }

    if provider_metadata.is_empty() {
        response.provider_metadata = None;
    }
}

fn normalize_custom_provider_metadata(data: &mut serde_json::Value) {
    let Some(provider_metadata) = data
        .get_mut("providerMetadata")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };

    if provider_metadata.contains_key(MINIMAXI_METADATA_KEY) {
        provider_metadata.remove(ANTHROPIC_METADATA_KEY);
    } else if let Some(anthropic) = provider_metadata.remove(ANTHROPIC_METADATA_KEY) {
        provider_metadata.insert(MINIMAXI_METADATA_KEY.to_string(), anthropic);
    }
}

fn normalize_stream_event(
    event: crate::streaming::ChatStreamEvent,
) -> crate::streaming::ChatStreamEvent {
    match event {
        crate::streaming::ChatStreamEvent::Custom {
            event_type,
            mut data,
        } => {
            normalize_custom_provider_metadata(&mut data);
            crate::streaming::ChatStreamEvent::Custom { event_type, data }
        }
        crate::streaming::ChatStreamEvent::StreamEnd { mut response } => {
            normalize_response_provider_metadata(&mut response);
            crate::streaming::ChatStreamEvent::StreamEnd { response }
        }
        other => other,
    }
}

fn normalize_stream_event_result(
    result: Result<crate::streaming::ChatStreamEvent, LlmError>,
) -> Result<crate::streaming::ChatStreamEvent, LlmError> {
    result.map(normalize_stream_event)
}

#[derive(Clone)]
struct MinimaxiResponseTransformer {
    inner: Arc<dyn ResponseTransformer>,
}

impl MinimaxiResponseTransformer {
    fn new(inner: Arc<dyn ResponseTransformer>) -> Self {
        Self { inner }
    }
}

impl ResponseTransformer for MinimaxiResponseTransformer {
    fn provider_id(&self) -> &str {
        MINIMAXI_METADATA_KEY
    }

    fn transform_chat_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        let mut response = self.inner.transform_chat_response(raw)?;
        normalize_response_provider_metadata(&mut response);
        Ok(response)
    }
}

#[derive(Clone)]
struct MinimaxiStreamTransformer {
    inner: Arc<dyn StreamChunkTransformer>,
}

impl MinimaxiStreamTransformer {
    fn new(inner: Arc<dyn StreamChunkTransformer>) -> Self {
        Self { inner }
    }
}

impl StreamChunkTransformer for MinimaxiStreamTransformer {
    fn provider_id(&self) -> &str {
        MINIMAXI_METADATA_KEY
    }

    fn convert_event(&self, event: eventsource_stream::Event) -> StreamEventFuture<'_> {
        let future = self.inner.convert_event(event);
        Box::pin(async move {
            future
                .await
                .into_iter()
                .map(normalize_stream_event_result)
                .collect()
        })
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.inner
            .handle_stream_end()
            .map(normalize_stream_event_result)
    }

    fn handle_stream_end_events(&self) -> Vec<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.inner
            .handle_stream_end_events()
            .into_iter()
            .map(normalize_stream_event_result)
            .collect()
    }

    fn finalize_on_disconnect(&self) -> bool {
        self.inner.finalize_on_disconnect()
    }
}

/// MiniMaxi ProviderSpec implementation
///
/// MiniMaxi supports both OpenAI and Anthropic API formats.
/// We use Anthropic format (recommended by MiniMaxi) for better support of:
/// - Thinking content blocks (reasoning process)
/// - Tool Use and Interleaved Thinking
/// - Extended thinking capabilities
#[derive(Clone)]
pub struct MinimaxiChatSpec {
    /// Anthropic Chat standard for request/response transformation
    chat_standard: AnthropicChatStandard,
}

#[derive(Debug, Default)]
struct MinimaxiAnthropicAdapter;

impl AnthropicChatAdapter for MinimaxiAnthropicAdapter {
    fn build_headers(
        &self,
        api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey(
                "MiniMaxi API key not provided".into(),
            ));
        }
        Ok(())
    }
}

impl MinimaxiChatSpec {
    pub fn new() -> Self {
        Self {
            chat_standard: AnthropicChatStandard::with_adapter(Arc::new(MinimaxiAnthropicAdapter)),
        }
    }

    fn chat_spec(&self) -> crate::standards::anthropic::chat::AnthropicChatSpec {
        self.chat_standard.create_spec("minimaxi")
    }
}

impl Default for MinimaxiChatSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderSpec for MinimaxiChatSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::anthropic::errors::classify_anthropic_http_error(
            "minimaxi", status, body_text,
        )
        .or_else(|| {
            crate::standards::openai::errors::classify_openai_compatible_http_error(
                "minimaxi", status, body_text,
            )
        })
    }

    fn chat_url(
        &self,
        stream: bool,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> String {
        self.chat_spec().chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let mut bundle = self.chat_spec().choose_chat_transformers(req, ctx);
        bundle.response = Arc::new(MinimaxiResponseTransformer::new(bundle.response.clone()));
        if let Some(stream) = bundle.stream.clone() {
            bundle.stream = Some(Arc::new(MinimaxiStreamTransformer::new(stream)));
        }
        bundle
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        let base_hook = self.chat_spec().chat_before_send(req, ctx);
        let options = self.minimaxi_options_from_provider_options_map(req);
        let thinking_mode = options.as_ref().and_then(|o| o.thinking_mode.clone());
        let response_format = self
            .stable_response_format_from_request(req)
            .or_else(|| options.as_ref().and_then(|o| o.response_format.clone()));

        if thinking_mode.is_none() && response_format.is_none() {
            return base_hook;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = if let Some(base) = &base_hook {
                base(body)?
            } else {
                body.clone()
            };

            if let Some(ref thinking) = thinking_mode
                && thinking.enabled
            {
                let budget = thinking.thinking_budget.unwrap_or(1024);

                out["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                });

                if let Some(obj) = out.as_object_mut() {
                    obj.remove("temperature");
                    obj.remove("top_p");
                    obj.remove("top_k");
                }

                if let Some(mt) = out.get("max_tokens").and_then(|v| v.as_u64()) {
                    out["max_tokens"] = serde_json::json!(mt.saturating_add(budget as u64));
                }
            }

            if let Some(ref rf) = response_format {
                match rf {
                    MinimaxiResponseFormat::JsonObject => {
                        out["output_format"] = serde_json::json!({ "type": "json_object" });
                    }
                    MinimaxiResponseFormat::JsonSchema {
                        name,
                        schema,
                        strict,
                    } => {
                        let _ = (name, strict);
                        out["output_format"] = serde_json::json!({
                            "type": "json_schema",
                            "schema": schema
                        });
                    }
                }

                if let Some(obj) = out.as_object_mut() {
                    obj.remove("tools");
                    obj.remove("tool_choice");
                }
            }

            Ok(out)
        };

        Some(Arc::new(hook))
    }
}

impl MinimaxiChatSpec {
    fn minimaxi_options_from_provider_options_map(
        &self,
        req: &crate::types::ChatRequest,
    ) -> Option<MinimaxiOptions> {
        let value = req
            .provider_options_map
            .get(MINIMAXI_OPTIONS_KEY)
            .or_else(|| req.provider_options_map.get(LEGACY_ANTHROPIC_OPTIONS_KEY))?;
        self.minimaxi_options_from_provider_options_value(value)
    }

    fn minimaxi_options_from_provider_options_value(
        &self,
        value: &serde_json::Value,
    ) -> Option<MinimaxiOptions> {
        let normalized = Self::normalize_minimaxi_provider_options_json(value);
        serde_json::from_value(normalized).ok()
    }

    fn normalize_minimaxi_provider_options_json(value: &serde_json::Value) -> serde_json::Value {
        fn normalize_key(key: &str) -> Option<&'static str> {
            Some(match key {
                "thinkingMode" => "thinking_mode",
                "responseFormat" => "response_format",
                "thinkingBudget" => "thinking_budget",
                _ => return None,
            })
        }

        fn inner(value: &serde_json::Value) -> serde_json::Value {
            match value {
                serde_json::Value::Object(map) => {
                    let mut out = serde_json::Map::new();
                    for (key, value) in map {
                        if key == "thinking"
                            && let Some(obj) = value.as_object()
                        {
                            let enabled = obj
                                .get("type")
                                .and_then(|t| t.as_str())
                                .map(|t| t == "enabled")
                                .unwrap_or(false);
                            let budget = obj
                                .get("budgetTokens")
                                .or_else(|| obj.get("budget_tokens"))
                                .and_then(|b| b.as_u64())
                                .and_then(|b| u32::try_from(b).ok());

                            let mut thinking_mode = serde_json::Map::new();
                            thinking_mode
                                .insert("enabled".to_string(), serde_json::Value::Bool(enabled));
                            if let Some(budget) = budget {
                                thinking_mode.insert(
                                    "thinking_budget".to_string(),
                                    serde_json::json!(budget),
                                );
                            }
                            out.insert(
                                "thinking_mode".to_string(),
                                serde_json::Value::Object(thinking_mode),
                            );
                            continue;
                        }

                        let normalized_key = normalize_key(key).unwrap_or(key);
                        out.insert(normalized_key.to_string(), inner(value));
                    }
                    serde_json::Value::Object(out)
                }
                serde_json::Value::Array(items) => {
                    serde_json::Value::Array(items.iter().map(inner).collect())
                }
                other => other.clone(),
            }
        }

        inner(value)
    }

    fn stable_response_format_from_request(
        &self,
        req: &crate::types::ChatRequest,
    ) -> Option<MinimaxiResponseFormat> {
        match req.response_format.as_ref()? {
            crate::types::chat::ResponseFormat::Json {
                schema,
                name,
                strict,
                ..
            } => Some(MinimaxiResponseFormat::JsonSchema {
                name: name.clone().unwrap_or_else(|| "response".to_string()),
                schema: schema.clone(),
                strict: strict.unwrap_or(true),
            }),
        }
    }
}

/// MiniMaxi audio spec (OpenAI-compatible endpoint).
///
/// Important: MiniMaxi uses Anthropic-compatible endpoints for chat, but OpenAI-compatible auth
/// (Bearer) for audio endpoints. Split specs keep each endpoint consistent and avoid mixing headers.
#[derive(Clone, Default)]
pub(crate) struct MinimaxiAudioSpec;

impl MinimaxiAudioSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxiAudioSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_audio()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        // MiniMaxi TTS/STT endpoints use OpenAI-compatible format under /v1.
        resolve_openai_base_url(&ctx.base_url)
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(super::transformers::audio::MinimaxiAudioTransformer),
        }
    }
}

/// MiniMaxi image generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiImageSpec;

impl MinimaxiImageSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxiImageSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        // Use OpenAI image protocol transformers with MiniMaxi response adapter.
        let standard =
            crate::providers::minimaxi::transformers::image::create_minimaxi_image_standard();
        let transformers = standard.create_transformers("minimaxi");
        crate::core::ImageTransformers {
            request: transformers.request,
            response: transformers.response,
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        format!(
            "{}/image_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }
}

/// MiniMaxi video generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiVideoSpec;

impl MinimaxiVideoSpec {
    pub fn new() -> Self {
        Self
    }

    pub fn video_generation_url(&self, ctx: &ProviderContext) -> String {
        format!(
            "{}/video_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }

    pub fn video_query_url(&self, ctx: &ProviderContext, task_id: &str) -> String {
        let base_url = resolve_openai_base_url(&ctx.base_url);
        format!(
            "{}/query/video_generation?task_id={}",
            base_url.trim_end_matches('/'),
            task_id
        )
    }
}

impl ProviderSpec for MinimaxiVideoSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("video", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// MiniMaxi music generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxiMusicSpec;

impl MinimaxiMusicSpec {
    pub fn new() -> Self {
        Self
    }

    pub fn music_generation_url(&self, ctx: &ProviderContext) -> String {
        format!(
            "{}/music_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        )
    }
}

impl ProviderSpec for MinimaxiMusicSpec {
    fn id(&self) -> &'static str {
        "minimaxi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("music", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// Backward compatible name: historically referenced as `MinimaxiSpec`.
pub type MinimaxiSpec = MinimaxiChatSpec;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderContext;
    use crate::core::ProviderSpec;
    use crate::provider_metadata::minimaxi::MinimaxiChatResponseExt;
    use crate::providers::minimaxi::MinimaxiConfig;
    use crate::streaming::ChatStreamEvent;
    use crate::types::{ChatMessage, ChatRequest};
    use std::collections::HashMap;

    #[test]
    fn minimaxi_chat_spec_build_headers_use_bearer_auth() {
        let ctx = ProviderContext::new(
            "minimaxi",
            MinimaxiConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );
        let headers = MinimaxiChatSpec::new()
            .build_headers(&ctx)
            .expect("headers");

        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        assert!(headers.get("x-api-key").is_none());
        assert!(headers.get("anthropic-version").is_none());
    }

    #[test]
    fn minimaxi_chat_spec_rekeys_anthropic_metadata_to_minimaxi() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "minimaxi",
            MinimaxiConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = MinimaxiChatSpec::new().choose_chat_transformers(&request, &ctx);
        let response = bundle
            .response
            .transform_chat_response(&serde_json::json!({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M2",
                "content": [{ "type": "text", "text": "hello" }],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1
                }
            }))
            .expect("transform response");

        let provider_metadata = response
            .provider_metadata
            .clone()
            .expect("provider metadata");
        assert!(provider_metadata.get("minimaxi").is_some());
        assert!(provider_metadata.get("anthropic").is_none());

        let meta = response
            .minimaxi_metadata()
            .expect("typed minimaxi metadata");
        assert!(meta.sources.is_none());
    }

    #[tokio::test]
    async fn minimaxi_chat_stream_rekeys_anthropic_metadata_to_minimaxi() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "minimaxi",
            MinimaxiConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = MinimaxiChatSpec::new().choose_chat_transformers(&request, &ctx);
        let stream = bundle.stream.expect("stream transformer");

        let _ = stream
            .convert_event(eventsource_stream::Event {
                event: "".to_string(),
                data: r#"{"type":"message_start","message":{"id":"msg_test","model":"MiniMax-M2","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
                id: "".to_string(),
                retry: None,
            })
            .await;

        let out = stream
            .convert_event(eventsource_stream::Event {
                event: "".to_string(),
                data: r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":5,"cleared_input_tokens":10000}]}},"usage":{"input_tokens":1,"output_tokens":1}}"#.to_string(),
                id: "".to_string(),
                retry: None,
            })
            .await;

        let finish = out
            .iter()
            .find_map(|event| match event.as_ref().ok() {
                Some(ChatStreamEvent::Custom { data, .. })
                    if data.get("type") == Some(&serde_json::json!("finish")) =>
                {
                    Some(data.clone())
                }
                _ => None,
            })
            .expect("finish event");
        assert!(finish["providerMetadata"].get("minimaxi").is_some());
        assert!(finish["providerMetadata"].get("anthropic").is_none());

        let end = out
            .iter()
            .find_map(|event| match event.as_ref().ok() {
                Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
                _ => None,
            })
            .expect("stream end");
        let provider_metadata = end.provider_metadata.clone().expect("provider metadata");
        assert!(provider_metadata.get("minimaxi").is_some());
        assert!(provider_metadata.get("anthropic").is_none());

        let meta = end.minimaxi_metadata().expect("typed minimaxi metadata");
        assert!(meta.context_management.is_some());
    }

    #[test]
    fn minimaxi_chat_spec_injects_thinking_and_structured_output_from_provider_owned_options() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .temperature(0.5)
            .max_tokens(256)
            .messages(vec![ChatMessage::user("hi").build()])
            .provider_option(
                "minimaxi",
                serde_json::json!({
                    "thinking": {
                        "type": "enabled",
                        "budgetTokens": 512
                    }
                }),
            )
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            ))
            .build();
        let ctx = ProviderContext::new(
            "minimaxi",
            MinimaxiConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );
        let spec = MinimaxiChatSpec::new();
        let transformers = spec.choose_chat_transformers(&request, &ctx);
        let body = transformers
            .request
            .transform_chat(&request)
            .expect("transform request");
        let body = spec
            .chat_before_send(&request, &ctx)
            .expect("before send hook")(&body)
        .expect("apply before send hook");

        assert_eq!(body["thinking"]["type"], serde_json::json!("enabled"));
        assert_eq!(body["thinking"]["budget_tokens"], serde_json::json!(512));
        assert_eq!(body["max_tokens"], serde_json::json!(768));
        assert!(body.get("temperature").is_none());
        assert_eq!(
            body["output_format"]["type"],
            serde_json::json!("json_schema")
        );
    }

    #[test]
    fn minimaxi_chat_spec_accepts_legacy_anthropic_provider_options_key() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .provider_option(
                "anthropic",
                serde_json::json!({
                    "thinking": {
                        "type": "enabled",
                        "budgetTokens": 256
                    }
                }),
            )
            .build();
        let ctx = ProviderContext::new(
            "minimaxi",
            MinimaxiConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );
        let spec = MinimaxiChatSpec::new();
        let transformers = spec.choose_chat_transformers(&request, &ctx);
        let body = transformers
            .request
            .transform_chat(&request)
            .expect("transform request");
        let body = spec
            .chat_before_send(&request, &ctx)
            .expect("before send hook")(&body)
        .expect("apply before send hook");

        assert_eq!(body["thinking"]["budget_tokens"], serde_json::json!(256));
    }
}
