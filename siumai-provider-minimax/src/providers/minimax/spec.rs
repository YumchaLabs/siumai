//! MiniMax ProviderSpec Implementation

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::transformers::response::ResponseTransformer;
use crate::execution::transformers::stream::{StreamChunkTransformer, StreamEventFuture};
use crate::provider_options::{MinimaxOptions, MinimaxThinking};
use crate::standards::anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
use crate::standards::openai::chat::OpenAiChatStandard;
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
        .ok_or_else(|| LlmError::MissingApiKey("MiniMax API key not provided".into()))?;

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
const MINIMAX_METADATA_KEY: &str = "minimax";
const MINIMAX_OPTIONS_KEY: &str = "minimax";

fn normalize_provider_metadata_map(provider_metadata: &mut crate::types::ProviderMetadataMap) {
    if provider_metadata.contains_key(MINIMAX_METADATA_KEY) {
        provider_metadata.remove(ANTHROPIC_METADATA_KEY);
    } else if let Some(anthropic) = provider_metadata.remove(ANTHROPIC_METADATA_KEY) {
        provider_metadata.insert(MINIMAX_METADATA_KEY.to_string(), anthropic);
    }
}

fn normalize_provider_metadata_object(
    provider_metadata: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_metadata.contains_key(MINIMAX_METADATA_KEY) {
        provider_metadata.remove(ANTHROPIC_METADATA_KEY);
    } else if let Some(anthropic) = provider_metadata.remove(ANTHROPIC_METADATA_KEY) {
        provider_metadata.insert(MINIMAX_METADATA_KEY.to_string(), anthropic);
    }
}

fn normalize_optional_provider_metadata(
    provider_metadata: &mut Option<crate::types::ProviderMetadataMap>,
) {
    let should_clear = match provider_metadata.as_mut() {
        Some(provider_metadata) => {
            normalize_provider_metadata_map(provider_metadata);
            provider_metadata.is_empty()
        }
        None => return,
    };

    if should_clear {
        *provider_metadata = None;
    }
}

fn normalize_response_provider_metadata(response: &mut crate::types::ChatResponse) {
    normalize_optional_provider_metadata(&mut response.provider_metadata);
}

fn normalize_custom_provider_metadata(data: &mut serde_json::Value) {
    let Some(provider_metadata) = data
        .get_mut("providerMetadata")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };

    normalize_provider_metadata_object(provider_metadata);
}

fn normalize_stream_part_provider_metadata(part: &mut crate::types::ChatStreamPart) {
    use crate::types::ChatStreamPart;

    match part {
        ChatStreamPart::TextStart {
            provider_metadata, ..
        }
        | ChatStreamPart::TextDelta {
            provider_metadata, ..
        }
        | ChatStreamPart::TextEnd {
            provider_metadata, ..
        }
        | ChatStreamPart::ReasoningStart {
            provider_metadata, ..
        }
        | ChatStreamPart::ReasoningDelta {
            provider_metadata, ..
        }
        | ChatStreamPart::ReasoningEnd {
            provider_metadata, ..
        }
        | ChatStreamPart::ToolInputStart {
            provider_metadata, ..
        }
        | ChatStreamPart::ToolInputDelta {
            provider_metadata, ..
        }
        | ChatStreamPart::ToolInputEnd {
            provider_metadata, ..
        }
        | ChatStreamPart::Source {
            provider_metadata, ..
        }
        | ChatStreamPart::Finish {
            provider_metadata, ..
        } => normalize_optional_provider_metadata(provider_metadata),
        ChatStreamPart::ToolApprovalRequest(request) => {
            normalize_optional_provider_metadata(&mut request.provider_metadata)
        }
        ChatStreamPart::ToolCall(call) => {
            normalize_optional_provider_metadata(&mut call.provider_metadata)
        }
        ChatStreamPart::ToolResult(result) => {
            normalize_optional_provider_metadata(&mut result.provider_metadata)
        }
        ChatStreamPart::Custom(content) => {
            normalize_optional_provider_metadata(&mut content.provider_metadata)
        }
        ChatStreamPart::File(file) | ChatStreamPart::ReasoningFile(file) => {
            normalize_optional_provider_metadata(&mut file.provider_metadata)
        }
        ChatStreamPart::StreamStart { .. }
        | ChatStreamPart::ResponseMetadata(..)
        | ChatStreamPart::Raw { .. }
        | ChatStreamPart::Error { .. } => {}
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
        crate::streaming::ChatStreamEvent::Part { mut part } => {
            normalize_stream_part_provider_metadata(&mut part);
            crate::streaming::ChatStreamEvent::Part { part }
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
struct MinimaxResponseTransformer {
    inner: Arc<dyn ResponseTransformer>,
}

impl MinimaxResponseTransformer {
    fn new(inner: Arc<dyn ResponseTransformer>) -> Self {
        Self { inner }
    }
}

impl ResponseTransformer for MinimaxResponseTransformer {
    fn provider_id(&self) -> &str {
        MINIMAX_METADATA_KEY
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
struct MinimaxStreamTransformer {
    inner: Arc<dyn StreamChunkTransformer>,
}

impl MinimaxStreamTransformer {
    fn new(inner: Arc<dyn StreamChunkTransformer>) -> Self {
        Self { inner }
    }
}

impl StreamChunkTransformer for MinimaxStreamTransformer {
    fn provider_id(&self) -> &str {
        MINIMAX_METADATA_KEY
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

    fn is_stream_end_event(&self, event: &eventsource_stream::Event) -> bool {
        self.inner.is_stream_end_event(event)
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

/// MiniMax ProviderSpec implementation
///
/// MiniMax supports both OpenAI and Anthropic API formats.
/// We use Anthropic format (recommended by MiniMax) for better support of:
/// - Thinking content blocks (reasoning process)
/// - Tool Use and Interleaved Thinking
/// - Extended thinking capabilities
#[derive(Clone)]
pub struct MinimaxChatSpec {
    /// Anthropic Chat standard for request/response transformation
    anthropic_standard: AnthropicChatStandard,
    /// OpenAI Chat standard used by the native and OpenAI-compatible endpoints.
    openai_standard: OpenAiChatStandard,
    /// Selected provider endpoint.
    endpoint: super::models::MinimaxChatEndpoint,
}

#[derive(Debug, Default)]
struct MinimaxAnthropicAdapter;

impl AnthropicChatAdapter for MinimaxAnthropicAdapter {
    fn build_headers(
        &self,
        api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey(
                "MiniMax API key not provided".into(),
            ));
        }
        Ok(())
    }
}

impl MinimaxChatSpec {
    pub fn new() -> Self {
        Self::for_endpoint(super::models::MinimaxChatEndpoint::AnthropicMessages)
    }

    /// Create a MiniMax chat spec for an explicit provider endpoint.
    pub fn for_endpoint(endpoint: super::models::MinimaxChatEndpoint) -> Self {
        Self {
            anthropic_standard: AnthropicChatStandard::with_adapter(Arc::new(
                MinimaxAnthropicAdapter,
            )),
            openai_standard: OpenAiChatStandard::new(),
            endpoint,
        }
    }

    fn anthropic_spec(&self) -> crate::standards::anthropic::chat::AnthropicChatSpec {
        self.anthropic_standard.create_spec("minimax")
    }

    fn openai_spec(&self) -> crate::standards::openai::chat::OpenAiChatSpec {
        self.openai_standard.create_spec("minimax")
    }
}

impl Default for MinimaxChatSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderSpec for MinimaxChatSpec {
    fn id(&self) -> &'static str {
        "minimax"
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
            "minimax", status, body_text,
        )
        .or_else(|| {
            crate::standards::openai::errors::classify_openai_compatible_http_error(
                "minimax", status, body_text,
            )
        })
    }

    fn try_chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let root = super::utils::resolve_api_root_base_url(&ctx.base_url);
        let path = match self.endpoint {
            super::models::MinimaxChatEndpoint::NativeText => "/v1/text/chatcompletion_v2",
            super::models::MinimaxChatEndpoint::AnthropicMessages => "/anthropic/v1/messages",
            super::models::MinimaxChatEndpoint::OpenAiChatCompletions => "/v1/chat/completions",
        };
        Ok(format!("{}{path}", root.trim_end_matches('/')))
    }

    fn choose_chat_transformers(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        let mut bundle = match self.endpoint {
            super::models::MinimaxChatEndpoint::AnthropicMessages => {
                self.anthropic_spec().choose_chat_transformers(req, ctx)
            }
            super::models::MinimaxChatEndpoint::NativeText
            | super::models::MinimaxChatEndpoint::OpenAiChatCompletions => {
                self.openai_spec().choose_chat_transformers(req, ctx)
            }
        };
        bundle.response = Arc::new(MinimaxResponseTransformer::new(bundle.response.clone()));
        if let Some(stream) = bundle.stream.clone() {
            bundle.stream = Some(Arc::new(MinimaxStreamTransformer::new(stream)));
        }
        bundle
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        let base_hook = match self.endpoint {
            super::models::MinimaxChatEndpoint::AnthropicMessages => {
                self.anthropic_spec().chat_before_send(req, ctx)
            }
            super::models::MinimaxChatEndpoint::NativeText
            | super::models::MinimaxChatEndpoint::OpenAiChatCompletions => {
                self.openai_spec().chat_before_send(req, ctx)
            }
        };
        let options = req.provider_options_map.get(MINIMAX_OPTIONS_KEY).cloned();
        let has_response_format = req.response_format.is_some();
        let model = req.common_params.model.clone();
        let endpoint = self.endpoint;
        let needs_openai_replay = endpoint != super::models::MinimaxChatEndpoint::AnthropicMessages;

        if options.is_none() && !has_response_format && !needs_openai_replay {
            return base_hook;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            if has_response_format {
                return Err(LlmError::InvalidParameter(
                    "MiniMax current models do not document JSON object or JSON schema output"
                        .to_string(),
                ));
            }

            let mut out = if let Some(base) = &base_hook {
                base(body)?
            } else {
                body.clone()
            };

            let options = options
                .as_ref()
                .map(|value| {
                    serde_json::from_value::<MinimaxOptions>(value.clone()).map_err(|error| {
                        LlmError::InvalidParameter(format!(
                            "invalid MiniMax provider options: {error}"
                        ))
                    })
                })
                .transpose()?;

            if let Some(options) = options {
                if options.thinking == Some(MinimaxThinking::Disabled)
                    && super::models::thinking_policy(&model, endpoint)
                        == Some(super::models::MinimaxThinkingPolicy::AlwaysOn)
                {
                    return Err(LlmError::InvalidParameter(format!(
                        "MiniMax model '{model}' always emits thinking and cannot be disabled"
                    )));
                }

                if let Some(thinking) = options.thinking {
                    out["thinking"] = serde_json::to_value(thinking).map_err(|error| {
                        LlmError::InvalidParameter(format!(
                            "failed to serialize MiniMax thinking options: {error}"
                        ))
                    })?;
                }
                if let Some(service_tier) = options.service_tier {
                    out["service_tier"] = serde_json::to_value(service_tier).map_err(|error| {
                        LlmError::InvalidParameter(format!(
                            "failed to serialize MiniMax service tier: {error}"
                        ))
                    })?;
                }

                let protected = [
                    "model",
                    "messages",
                    "system",
                    "stream",
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "max_completion_tokens",
                    "thinking",
                    "service_tier",
                    "reasoning_split",
                    "response_format",
                    "output_format",
                ];
                let object = out.as_object_mut().ok_or_else(|| {
                    LlmError::InvalidParameter(
                        "MiniMax request transformer produced a non-object body".to_string(),
                    )
                })?;
                for (key, value) in options.extra_params {
                    if protected.contains(&key.as_str()) {
                        return Err(LlmError::InvalidParameter(format!(
                            "MiniMax provider option '{key}' must use the stable or typed field"
                        )));
                    }
                    object.insert(key, value);
                }
            }

            if needs_openai_replay {
                let object = out.as_object_mut().ok_or_else(|| {
                    LlmError::InvalidParameter(
                        "MiniMax request transformer produced a non-object body".to_string(),
                    )
                })?;
                object.insert("reasoning_split".to_string(), serde_json::Value::Bool(true));
                if let Some(max_tokens) = object.remove("max_tokens") {
                    object
                        .entry("max_completion_tokens".to_string())
                        .or_insert(max_tokens);
                }
            }

            Ok(out)
        };

        Some(Arc::new(hook))
    }
}

/// MiniMax audio spec (OpenAI-compatible endpoint).
///
/// Important: MiniMax uses Anthropic-compatible endpoints for chat, but OpenAI-compatible auth
/// (Bearer) for audio endpoints. Split specs keep each endpoint consistent and avoid mixing headers.
#[derive(Clone, Default)]
pub(crate) struct MinimaxAudioSpec;

impl MinimaxAudioSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxAudioSpec {
    fn id(&self) -> &'static str {
        "minimax"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_audio()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        // MiniMax TTS/STT endpoints use OpenAI-compatible format under /v1.
        resolve_openai_base_url(&ctx.base_url)
    }

    fn choose_audio_transformer(&self, _ctx: &ProviderContext) -> crate::core::AudioTransformer {
        crate::core::AudioTransformer {
            transformer: Arc::new(super::transformers::audio::MinimaxAudioTransformer),
        }
    }
}

/// MiniMax image generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxImageSpec;

impl MinimaxImageSpec {
    pub fn new() -> Self {
        Self
    }
}

impl ProviderSpec for MinimaxImageSpec {
    fn id(&self) -> &'static str {
        "minimax"
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
        // Use OpenAI image protocol transformers with MiniMax response adapter.
        let standard =
            crate::providers::minimax::transformers::image::create_minimax_image_standard();
        let transformers = standard.create_transformers("minimax");
        crate::core::ImageTransformers {
            request: transformers.request,
            response: transformers.response,
        }
    }

    fn try_image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(format!(
            "{}/image_generation",
            resolve_openai_base_url(&ctx.base_url).trim_end_matches('/')
        ))
    }
}

/// MiniMax video generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxVideoSpec;

impl MinimaxVideoSpec {
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

impl ProviderSpec for MinimaxVideoSpec {
    fn id(&self) -> &'static str {
        "minimax"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("video", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// MiniMax music generation spec (OpenAI-compatible endpoint).
#[derive(Clone, Default)]
pub(crate) struct MinimaxMusicSpec;

impl MinimaxMusicSpec {
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

impl ProviderSpec for MinimaxMusicSpec {
    fn id(&self) -> &'static str {
        "minimax"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("music", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_openai_like_headers(ctx)
    }
}

/// Backward compatible name: historically referenced as `MinimaxSpec`.
pub type MinimaxSpec = MinimaxChatSpec;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderContext;
    use crate::core::ProviderSpec;
    use crate::provider_metadata::minimax::MinimaxChatResponseExt;
    use crate::providers::minimax::MinimaxConfig;
    use crate::providers::minimax::models::MinimaxChatEndpoint;
    use crate::streaming::ChatStreamEvent;
    use crate::types::{ChatMessage, ChatRequest};
    use std::collections::HashMap;

    fn production_source() -> &'static str {
        include_str!("spec.rs")
    }

    fn source_between(start_marker: &str, end_marker: &str) -> &'static str {
        let source = production_source();
        let (_, after_start) = source
            .split_once(start_marker)
            .expect("source start marker should exist");
        let (section, _) = after_start
            .split_once(end_marker)
            .expect("source end marker should exist");
        section
    }

    #[test]
    fn response_metadata_normalization_source_does_not_read_request_provider_options() {
        let response_source =
            source_between("fn normalize_provider_metadata_map", "impl MinimaxChatSpec");

        assert!(
            !response_source.contains("providerOptions"),
            "MiniMax response metadata normalization must not read camelCase request provider options"
        );
        assert!(
            !response_source.contains("provider_options_map"),
            "MiniMax response metadata normalization must not read request provider options maps"
        );
    }

    #[test]
    fn request_option_resolution_source_does_not_read_response_provider_metadata() {
        let request_source = source_between("impl MinimaxChatSpec", "#[cfg(test)]");

        assert!(
            !request_source.contains("providerMetadata"),
            "MiniMax request option resolution must not read camelCase response provider metadata"
        );
        assert!(
            !request_source.contains("provider_metadata"),
            "MiniMax request option resolution must not read legacy response provider metadata"
        );
    }

    #[test]
    fn minimax_chat_spec_build_headers_use_bearer_auth() {
        let ctx = ProviderContext::new(
            "minimax",
            MinimaxConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );
        let headers = MinimaxChatSpec::new().build_headers(&ctx).expect("headers");

        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        assert!(headers.get("x-api-key").is_none());
        assert!(headers.get("anthropic-version").is_none());
    }

    #[test]
    fn minimax_chat_spec_rekeys_anthropic_metadata_to_minimax() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "minimax",
            MinimaxConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = MinimaxChatSpec::new().choose_chat_transformers(&request, &ctx);
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
        assert!(provider_metadata.contains_key("minimax"));
        assert!(!provider_metadata.contains_key("anthropic"));

        let meta = response.minimax_metadata().expect("typed minimax metadata");
        assert!(meta.sources.is_none());
    }

    #[tokio::test]
    async fn minimax_chat_stream_rekeys_anthropic_metadata_to_minimax() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = ProviderContext::new(
            "minimax",
            MinimaxConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let bundle = MinimaxChatSpec::new().choose_chat_transformers(&request, &ctx);
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

        let finish_provider_metadata = out
            .iter()
            .find_map(|event| match event.as_ref().ok() {
                Some(ChatStreamEvent::Part {
                    part:
                        crate::types::ChatStreamPart::Finish {
                            provider_metadata: Some(provider_metadata),
                            ..
                        },
                }) => Some(provider_metadata.clone()),
                _ => None,
            })
            .expect("finish part");
        assert!(finish_provider_metadata.contains_key("minimax"));
        assert!(!finish_provider_metadata.contains_key("anthropic"));

        let end = out
            .iter()
            .find_map(|event| match event.as_ref().ok() {
                Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
                _ => None,
            })
            .expect("stream end");
        let provider_metadata = end.provider_metadata.clone().expect("provider metadata");
        assert!(provider_metadata.contains_key("minimax"));
        assert!(!provider_metadata.contains_key("anthropic"));

        let meta = end.minimax_metadata().expect("typed minimax metadata");
        assert!(meta.context_management.is_some());
    }

    fn context() -> ProviderContext {
        ProviderContext::new(
            "minimax",
            MinimaxConfig::DEFAULT_BASE_URL,
            Some("test-key".to_string()),
            HashMap::new(),
        )
    }

    fn transformed_body(
        spec: &MinimaxChatSpec,
        request: &ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let ctx = context();
        let transformers = spec.choose_chat_transformers(request, &ctx);
        let body = transformers.request.transform_chat(request)?;
        spec.chat_before_send(request, &ctx)
            .map(|hook| hook(&body))
            .unwrap_or(Ok(body))
    }

    #[test]
    fn minimax_chat_spec_preserves_official_thinking_and_sampling_fields() {
        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .temperature(0.5)
            .max_tokens(256)
            .messages(vec![ChatMessage::user("hi").build()])
            .provider_option(
                "minimax",
                serde_json::json!({
                    "thinking": { "type": "adaptive" },
                    "service_tier": "priority"
                }),
            )
            .build();

        let body = transformed_body(&MinimaxChatSpec::new(), &request).expect("transform request");
        assert_eq!(body["thinking"], serde_json::json!({ "type": "adaptive" }));
        assert_eq!(body["service_tier"], serde_json::json!("priority"));
        assert_eq!(body["max_tokens"], serde_json::json!(256));
        assert_eq!(body["temperature"], serde_json::json!(0.5));
    }

    #[test]
    fn minimax_chat_spec_rejects_unsupported_stable_response_format() {
        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(crate::types::chat::ResponseFormat::json_object())
            .build();

        let error = transformed_body(&MinimaxChatSpec::new(), &request).expect_err("must reject");
        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn minimax_chat_spec_rejects_disabling_thinking_for_m2_models() {
        let request = ChatRequest::builder()
            .model("MiniMax-M2")
            .messages(vec![ChatMessage::user("hi").build()])
            .provider_option(
                "minimax",
                serde_json::json!({ "thinking": { "type": "disabled" } }),
            )
            .build();

        let error = transformed_body(&MinimaxChatSpec::new(), &request).expect_err("must reject");
        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn minimax_chat_spec_uses_exact_endpoint_paths() {
        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let ctx = context();

        assert_eq!(
            MinimaxChatSpec::for_endpoint(MinimaxChatEndpoint::AnthropicMessages)
                .try_chat_url(false, &request, &ctx)
                .expect("anthropic url"),
            "https://api.minimax.io/anthropic/v1/messages"
        );
        assert_eq!(
            MinimaxChatSpec::for_endpoint(MinimaxChatEndpoint::OpenAiChatCompletions)
                .try_chat_url(false, &request, &ctx)
                .expect("openai url"),
            "https://api.minimax.io/v1/chat/completions"
        );
        assert_eq!(
            MinimaxChatSpec::for_endpoint(MinimaxChatEndpoint::NativeText)
                .try_chat_url(false, &request, &ctx)
                .expect("native url"),
            "https://api.minimax.io/v1/text/chatcompletion_v2"
        );
    }

    #[test]
    fn minimax_chat_spec_adds_reasoning_split_for_openai_compatible_routes() {
        let request = ChatRequest::builder()
            .model("MiniMax-M3")
            .max_tokens(256)
            .messages(vec![ChatMessage::user("hi").build()])
            .build();
        let body = transformed_body(
            &MinimaxChatSpec::for_endpoint(MinimaxChatEndpoint::OpenAiChatCompletions),
            &request,
        )
        .expect("transform request");

        assert_eq!(body["reasoning_split"], serde_json::json!(true));
        assert_eq!(body["max_completion_tokens"], serde_json::json!(256));
        assert!(body.get("max_tokens").is_none());
    }
}
