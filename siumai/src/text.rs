//! Text model family APIs.
//!
//! This is the recommended Rust-first surface for text generation:
//! - `generate` for non-streaming
//! - `stream` for streaming
//! - `stream_with_cancel` for streaming with cancellation

use crate::request_options::{
    EffectiveRequestOptions, link_stream_handle_abort, retry_or_call_with_abort,
    wrap_stream_with_abort,
};
use crate::retry_api::RetryOptions;
use chrono::Utc;
use siumai_core::error::LlmError;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::text::{
    LanguageModel, LanguageModelV4, LanguageModelV4DoStreamResult, LanguageModelV4Stream,
    TextModelV3, TextRequest, TextResponse, TextStream, TextStreamHandle,
};
pub use siumai_core::types::StreamRequestOptions;
use siumai_core::types::{
    AssistantContent, AssistantModelMessage, CallWarning, ChatMessage, ContentPart, CustomOutput,
    FileOutput, FinishReason, GenerateTextContentPart, GenerateTextModelInfo,
    GenerateTextReasoningPart, GenerateTextResponseMetadata, GenerateTextResult,
    GenerateTextStepReasoningPart, GenerateTextStepResult, GeneratedFile, HttpConfig, JSONValue,
    LanguageModelRequestMetadata, LanguageModelResponseMetadata, MessageContent, MessageMetadata,
    MessageRole, ModelMessage, ProviderOptionsMap, ReasoningFileOutput, ReasoningOutput,
    RequestOptions, ResponseMessage, Source, TextOutput, Tool, ToolCall as GenerateTextToolCall,
    ToolChoice, ToolResult as GenerateTextToolResult, ToolResultOutput,
};

/// AI SDK-style non-streaming `generateText` result returned by `text::generate_text`.
pub type GenerateTextProjectionResult = GenerateTextResult<()>;

type GenerateTextOutputPart = GenerateTextContentPart<String, JSONValue, ToolResultOutput>;
type GenerateTextProjectedToolCall = GenerateTextToolCall<String, JSONValue>;
type GenerateTextProjectedToolResult = GenerateTextToolResult<String, JSONValue, ToolResultOutput>;

/// AI SDK-style inclusion controls for `generateText` result metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerateTextInclude {
    /// Whether to retain the provider request body in result and step metadata.
    pub request_body: bool,
    /// Whether to retain the provider response body in result and step metadata.
    pub response_body: bool,
}

impl GenerateTextInclude {
    /// Include all metadata bodies, matching AI SDK defaults.
    pub const fn all() -> Self {
        Self {
            request_body: true,
            response_body: true,
        }
    }

    /// Omit large request and response bodies while keeping the rest of the metadata.
    pub const fn metadata_only() -> Self {
        Self {
            request_body: false,
            response_body: false,
        }
    }
}

impl Default for GenerateTextInclude {
    fn default() -> Self {
        Self::all()
    }
}

/// Options for `text::generate`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `ChatRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `ChatRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
    /// Optional tools to add to the request for this call.
    ///
    /// When the request already has tools, these are appended.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice override for this call.
    pub tool_choice: Option<ToolChoice>,
    /// Optional telemetry config for this call.
    ///
    /// This is applied to `ChatRequest.telemetry` (runtime-only; not serialized).
    pub telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
    /// Controls whether large provider request/response bodies are retained in `generate_text`.
    pub include: GenerateTextInclude,
}

/// Options for `text::stream`.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Optional retry policy applied when establishing the stream.
    ///
    /// Note: this retries stream *creation* only. It does not retry mid-stream failures.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    pub headers: HashMap<String, String>,
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
    /// Optional tools to add to the request for this call.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice override for this call.
    pub tool_choice: Option<ToolChoice>,
    /// Optional telemetry config for this call (runtime-only).
    pub telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
    /// Include provider raw chunks on the stream part lane.
    pub include_raw_chunks: bool,
}

fn apply_text_call_options(
    mut request: TextRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
) -> TextRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }

    if let Some(ts) = tools {
        match request.tools.as_mut() {
            Some(existing) => existing.extend(ts),
            None => request.tools = Some(ts),
        }
    }

    if let Some(choice) = tool_choice {
        request.tool_choice = Some(choice);
    }

    if let Some(tel) = telemetry {
        request.telemetry = Some(tel);
    }

    request
}

pub(crate) fn prepare_generate_request(
    request: TextRequest,
    options: GenerateOptions,
) -> (TextRequest, EffectiveRequestOptions) {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    (request, effective)
}

pub(crate) async fn generate_prepared<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    effective: EffectiveRequestOptions,
) -> Result<TextResponse, LlmError> {
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = request.clone();
        async move { model.generate(req).await }
    })
    .await
}

/// Generate a non-streaming text response.
pub async fn generate<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: GenerateOptions,
) -> Result<TextResponse, LlmError> {
    let (request, effective) = prepare_generate_request(request, options);
    generate_prepared(model, request, effective).await
}

/// Generate text and project the provider response into an AI SDK-style `GenerateTextResult`.
///
/// This is a single-step projection over Siumai's existing `TextResponse`. It intentionally does
/// not emulate AI SDK's full tool-loop/agent runtime; callers that need multi-step execution should
/// keep using explicit tool orchestration and can still inspect the projected `steps[0]`.
pub async fn generate_text<M: LanguageModel + ?Sized>(
    model: &M,
    request: TextRequest,
    options: GenerateOptions,
) -> Result<GenerateTextProjectionResult, LlmError> {
    let include = options.include;
    let (request, effective) = prepare_generate_request(request, options);
    let response = generate_prepared(model, request, effective).await?;
    let mut request_metadata = response
        .request
        .as_ref()
        .map(LanguageModelRequestMetadata::from)
        .unwrap_or_default();
    if !include.request_body {
        request_metadata.body = None;
    }
    Ok(project_generate_text_response(
        model,
        response,
        request_metadata,
        include,
    ))
}

/// Generate a streaming text response.
pub async fn stream<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStream, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }

    let abort_signal = effective.abort_signal();
    let stream = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let req = request.clone();
        async move { model.stream(req).await }
    })
    .await?;
    Ok(wrap_stream_with_abort(stream, abort_signal))
}

/// Generate a streaming text response with cancellation support.
pub async fn stream_with_cancel<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStreamHandle, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }

    let abort_signal = effective.abort_signal();
    let handle = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let req = request.clone();
        async move { model.stream_with_cancel(req).await }
    })
    .await?;
    Ok(link_stream_handle_abort(handle, abort_signal))
}

fn project_generate_text_response<M: LanguageModel + ?Sized>(
    model: &M,
    response: TextResponse,
    request_metadata: LanguageModelRequestMetadata,
    include: GenerateTextInclude,
) -> GenerateTextProjectionResult {
    let call_id = response.id.clone().unwrap_or_else(crate::generate_id);
    let http_response = response.response.as_ref();
    let model_id = http_response
        .and_then(|response| response.model_id.clone())
        .or_else(|| response.model.clone())
        .unwrap_or_else(|| model.model_id().to_string());
    let response_headers = http_response
        .map(|response| response.headers.clone())
        .filter(|headers| !headers.is_empty());
    let response_body = http_response.and_then(|response| response.body.clone());
    let response_timestamp = http_response
        .map(|response| response.timestamp)
        .unwrap_or_else(Utc::now);
    let response_id = response.id.clone().unwrap_or_else(|| call_id.clone());
    let model_info_model_id = response
        .model
        .clone()
        .unwrap_or_else(|| model.model_id().to_string());
    let model_info = GenerateTextModelInfo::new(model.provider_id(), model_info_model_id);
    let mut response_metadata = GenerateTextResponseMetadata::new(LanguageModelResponseMetadata {
        id: response_id,
        timestamp: response_timestamp,
        model_id,
        headers: response_headers,
    })
    .with_messages(response_messages_from_content(&response.content));
    if include.response_body
        && let Some(body) = response_body
    {
        response_metadata = response_metadata.with_body(body);
    }

    let projection = project_generate_text_content(&response.content);
    let usage = response
        .usage
        .as_ref()
        .map(siumai_core::types::as_language_model_usage)
        .unwrap_or_default();
    let finish_reason = response
        .finish_reason
        .clone()
        .unwrap_or(FinishReason::Unknown);
    let warnings = response
        .warnings
        .clone()
        .map(|warnings| warnings.into_iter().map(CallWarning::from).collect());

    let step = GenerateTextStepResult {
        call_id,
        step_number: 0,
        model: model_info,
        tools_context: HashMap::new(),
        runtime_context: HashMap::new(),
        content: projection.content.clone(),
        text: projection.text.clone(),
        reasoning: projection.step_reasoning.clone(),
        reasoning_text: projection.reasoning_text.clone(),
        files: projection.files.clone(),
        sources: projection.sources.clone(),
        tool_calls: projection.tool_calls.clone(),
        static_tool_calls: projection.static_tool_calls.clone(),
        dynamic_tool_calls: projection.dynamic_tool_calls.clone(),
        tool_results: projection.tool_results.clone(),
        static_tool_results: projection.static_tool_results.clone(),
        dynamic_tool_results: projection.dynamic_tool_results.clone(),
        finish_reason: finish_reason.clone(),
        raw_finish_reason: response.raw_finish_reason.clone(),
        usage: usage.clone(),
        warnings: warnings.clone(),
        request: request_metadata.clone(),
        response: response_metadata.clone(),
        provider_metadata: response.provider_metadata.clone(),
    };

    GenerateTextResult {
        content: projection.content,
        text: projection.text,
        reasoning: projection.reasoning,
        reasoning_text: projection.reasoning_text,
        files: projection.files,
        sources: projection.sources,
        tool_calls: projection.tool_calls,
        static_tool_calls: projection.static_tool_calls,
        dynamic_tool_calls: projection.dynamic_tool_calls,
        tool_results: projection.tool_results,
        static_tool_results: projection.static_tool_results,
        dynamic_tool_results: projection.dynamic_tool_results,
        finish_reason,
        raw_finish_reason: response.raw_finish_reason,
        usage: usage.clone(),
        total_usage: usage,
        warnings,
        request: request_metadata,
        response: response_metadata,
        provider_metadata: response.provider_metadata,
        steps: vec![step],
        output: (),
    }
}

#[derive(Debug, Default)]
struct GenerateTextContentProjection {
    content: Vec<GenerateTextOutputPart>,
    text: String,
    reasoning: Vec<GenerateTextReasoningPart>,
    step_reasoning: Vec<GenerateTextStepReasoningPart>,
    reasoning_text: Option<String>,
    files: Vec<GeneratedFile>,
    sources: Vec<Source>,
    tool_calls: Vec<GenerateTextProjectedToolCall>,
    static_tool_calls: Vec<GenerateTextProjectedToolCall>,
    dynamic_tool_calls: Vec<GenerateTextProjectedToolCall>,
    tool_results: Vec<GenerateTextProjectedToolResult>,
    static_tool_results: Vec<GenerateTextProjectedToolResult>,
    dynamic_tool_results: Vec<GenerateTextProjectedToolResult>,
}

fn project_generate_text_content(content: &MessageContent) -> GenerateTextContentProjection {
    let mut projection = GenerateTextContentProjection {
        text: content.all_text(),
        ..Default::default()
    };

    match content {
        MessageContent::Text(text) => {
            push_text_output(&mut projection, TextOutput::new(text.clone()));
        }
        MessageContent::MultiModal(parts) => {
            for part in parts {
                project_generate_text_content_part(part, &mut projection);
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            push_text_output(&mut projection, TextOutput::new(value.to_string()));
        }
    }

    if !projection.step_reasoning.is_empty() {
        let reasoning_text = projection
            .step_reasoning
            .iter()
            .filter_map(|part| match part {
                GenerateTextStepReasoningPart::Reasoning { text, .. } => Some(text.as_str()),
                GenerateTextStepReasoningPart::ReasoningFile { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("");

        if !reasoning_text.is_empty() {
            projection.reasoning_text = Some(reasoning_text);
        }
    }

    projection
}

fn project_generate_text_content_part(
    part: &ContentPart,
    projection: &mut GenerateTextContentProjection,
) {
    match part {
        ContentPart::Text {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = TextOutput::new(text.clone());
            if let Some(metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(metadata);
            }
            push_text_output(projection, output);
        }
        ContentPart::Custom {
            kind,
            provider_metadata,
            ..
        } => {
            let mut output = CustomOutput::new(kind.clone());
            if let Some(metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(metadata);
            }
            projection.content.push(output.into());
        }
        ContentPart::File {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            if let Some(base64) = source.as_base64() {
                let file = GeneratedFile::from_base64(base64, media_type.clone());
                let mut output = FileOutput::new(file.clone());
                if let Some(metadata) = provider_metadata.clone() {
                    output = output.with_provider_metadata(metadata);
                }
                projection.files.push(file);
                projection.content.push(output.into());
            }
        }
        ContentPart::Reasoning {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = ReasoningOutput::new(text.clone());
            if let Some(metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(metadata);
            }
            push_reasoning_output(projection, output);
        }
        ContentPart::ReasoningFile {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            if let Some(base64) = source.as_base64() {
                let mut output = ReasoningFileOutput::new(GeneratedFile::from_base64(
                    base64,
                    media_type.clone(),
                ));
                if let Some(metadata) = provider_metadata.clone() {
                    output = output.with_provider_metadata(metadata);
                }
                push_reasoning_file_output(projection, output);
            }
        }
        ContentPart::Source { .. } => {
            if let Ok(source) = Source::try_from(part) {
                projection.sources.push(source.clone());
                projection.content.push(source.into());
            }
        }
        ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            dynamic,
            invalid,
            error,
            title,
            provider_metadata,
            ..
        } => {
            let mut tool_call = GenerateTextToolCall::new(
                tool_call_id.clone(),
                tool_name.clone(),
                arguments.clone(),
            );
            if let Some(provider_executed) = provider_executed {
                tool_call = tool_call.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                tool_call = tool_call.with_dynamic(*dynamic);
            }
            if let Some(invalid) = invalid {
                tool_call = tool_call.with_invalid(*invalid);
            }
            if let Some(error) = error.clone() {
                tool_call = tool_call.with_error(error);
            }
            if let Some(title) = title.clone() {
                tool_call = tool_call.with_title(title);
            }
            if let Some(metadata) = provider_metadata.clone() {
                tool_call = tool_call.with_provider_metadata(metadata);
            }

            if tool_call.dynamic == Some(true) {
                projection.dynamic_tool_calls.push(tool_call.clone());
            } else {
                projection.static_tool_calls.push(tool_call.clone());
            }
            projection.tool_calls.push(tool_call.clone());
            projection.content.push(tool_call.into());
        }
        ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            input,
            provider_executed,
            dynamic,
            preliminary,
            title,
            provider_metadata,
            ..
        } => {
            let mut tool_result = GenerateTextToolResult::new(
                tool_call_id.clone(),
                tool_name.clone(),
                input.clone().unwrap_or(JSONValue::Null),
                output.clone(),
            );
            if let Some(provider_executed) = provider_executed {
                tool_result = tool_result.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                tool_result = tool_result.with_dynamic(*dynamic);
            }
            if let Some(preliminary) = preliminary {
                tool_result = tool_result.with_preliminary(*preliminary);
            }
            if let Some(title) = title.clone() {
                tool_result = tool_result.with_title(title);
            }
            if let Some(metadata) = provider_metadata.clone() {
                tool_result = tool_result.with_provider_metadata(metadata);
            }

            if tool_result.dynamic == Some(true) {
                projection.dynamic_tool_results.push(tool_result.clone());
            } else {
                projection.static_tool_results.push(tool_result.clone());
            }
            projection.tool_results.push(tool_result.clone());
            projection.content.push(tool_result.into());
        }
        ContentPart::Audio { .. }
        | ContentPart::Image { .. }
        | ContentPart::ToolApprovalRequest { .. }
        | ContentPart::ToolApprovalResponse { .. } => {}
    }
}

fn push_text_output(projection: &mut GenerateTextContentProjection, output: TextOutput) {
    projection.content.push(output.into());
}

fn push_reasoning_output(projection: &mut GenerateTextContentProjection, output: ReasoningOutput) {
    projection.reasoning.push(output.clone().into());
    projection.step_reasoning.push(output.clone().into());
    projection.content.push(output.into());
}

fn push_reasoning_file_output(
    projection: &mut GenerateTextContentProjection,
    output: ReasoningFileOutput,
) {
    projection.reasoning.push(output.clone().into());
    projection.step_reasoning.push(output.clone().into());
    projection.content.push(output.into());
}

fn response_messages_from_content(content: &MessageContent) -> Vec<ResponseMessage> {
    let content = assistant_response_message_content(content);
    let message = ChatMessage {
        role: MessageRole::Assistant,
        content: content.clone(),
        provider_options: ProviderOptionsMap::default(),
        metadata: MessageMetadata::default(),
    };

    match ModelMessage::try_from(&message) {
        Ok(ModelMessage::Assistant(message)) => vec![message.into()],
        _ => vec![AssistantModelMessage::new(AssistantContent::text(content.all_text())).into()],
    }
}

fn assistant_response_message_content(content: &MessageContent) -> MessageContent {
    match content {
        MessageContent::Text(text) => MessageContent::Text(text.clone()),
        MessageContent::MultiModal(parts) => {
            let parts = parts
                .iter()
                .filter(|part| {
                    matches!(
                        part,
                        ContentPart::Text { .. }
                            | ContentPart::Custom { .. }
                            | ContentPart::File { .. }
                            | ContentPart::Reasoning { .. }
                            | ContentPart::ReasoningFile { .. }
                            | ContentPart::ToolCall { .. }
                            | ContentPart::ToolResult { .. }
                            | ContentPart::ToolApprovalRequest { .. }
                    )
                })
                .cloned()
                .collect::<Vec<_>>();

            if parts.is_empty() {
                MessageContent::Text(content.all_text())
            } else {
                MessageContent::MultiModal(parts)
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => MessageContent::Text(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use siumai_core::traits::ModelMetadata;
    use siumai_core::types::{
        ChatRequest, ChatResponse, HttpRequestInfo, HttpResponseInfo, Usage, Warning,
    };

    struct FakeLanguageModel;

    impl ModelMetadata for FakeLanguageModel {
        fn provider_id(&self) -> &str {
            "test-provider"
        }

        fn model_id(&self) -> &str {
            "test-model"
        }
    }

    #[async_trait]
    impl TextModelV3 for FakeLanguageModel {
        async fn generate(&self, _request: TextRequest) -> Result<TextResponse, LlmError> {
            let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
                ContentPart::reasoning("thinking"),
                ContentPart::text("Hello"),
                ContentPart::text("world"),
                ContentPart::source_url("src-1", "https://example.com", "Example"),
                ContentPart::tool_call("call-1", "lookup", json!({ "q": "rust" }), Some(true)),
                ContentPart::tool_result_json("call-1", "lookup", json!({ "ok": true })),
            ]));
            response.id = Some("resp-1".to_string());
            response.model = Some("provider-model".to_string());
            response.finish_reason = Some(FinishReason::ToolCalls);
            response.raw_finish_reason = Some("tool_calls".to_string());
            response.usage = Some(
                Usage::builder()
                    .with_input_total_tokens(7)
                    .with_output_total_tokens(5)
                    .with_output_reasoning_tokens(2)
                    .build(),
            );
            response.provider_metadata = Some(HashMap::from([(
                "test-provider".to_string(),
                json!({ "trace": "abc" }),
            )]));
            response.warnings = Some(vec![Warning::UnsupportedSetting {
                setting: "topK".to_string(),
                details: Some("not supported".to_string()),
            }]);
            response.request = Some(HttpRequestInfo {
                body: Some(
                    json!({
                        "model": "provider-model",
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ),
            });
            response.response = Some(HttpResponseInfo {
                timestamp: chrono::DateTime::parse_from_rfc3339("2026-04-30T00:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&chrono::Utc),
                model_id: Some("provider-model".to_string()),
                headers: HashMap::from([("x-request-id".to_string(), "req_123".to_string())]),
                body: Some(json!({ "id": "resp-1", "raw": true })),
            });
            Ok(response)
        }

        async fn stream(&self, _request: TextRequest) -> Result<TextStream, LlmError> {
            unreachable!("generate_text does not call stream")
        }

        async fn stream_with_cancel(
            &self,
            _request: TextRequest,
        ) -> Result<TextStreamHandle, LlmError> {
            unreachable!("generate_text does not call stream_with_cancel")
        }
    }

    #[tokio::test]
    async fn generate_text_projects_single_step_ai_sdk_result() {
        let model = FakeLanguageModel;
        let result = generate_text(
            &model,
            ChatRequest::new(vec![ChatMessage::user("hi").build()]),
            GenerateOptions::default(),
        )
        .await
        .expect("generate text");

        assert_eq!(result.text, "Hello world");
        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
        assert_eq!(result.raw_finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(result.usage.input_tokens, Some(7));
        assert_eq!(result.usage.output_tokens, Some(5));
        assert_eq!(result.usage.reasoning_tokens, Some(2));
        assert_eq!(result.reasoning_text.as_deref(), Some("thinking"));
        assert_eq!(result.sources.len(), 1);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].provider_executed, Some(true));
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].call_id, "resp-1");
        assert_eq!(result.steps[0].model.provider, "test-provider");
        assert_eq!(result.steps[0].model.model_id, "provider-model");
        assert_eq!(result.steps[0].request, result.request);
        assert_eq!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("model")),
            Some(&json!("provider-model"))
        );
        assert_eq!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("messages"))
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(1)
        );
        assert!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("common_params"))
                .is_none()
        );
        assert_eq!(result.warnings, result.steps[0].warnings);
        assert_eq!(
            serde_json::to_value(result.warnings.as_ref().expect("warnings"))
                .expect("serialize warnings"),
            json!([{
                "type": "unsupported",
                "feature": "topK",
                "details": "not supported"
            }])
        );
        assert_eq!(result.response.metadata.id, "resp-1");
        assert_eq!(result.response.metadata.model_id, "provider-model");
        assert_eq!(
            result
                .response
                .metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("req_123")
        );
        assert_eq!(
            result.response.body,
            Some(json!({ "id": "resp-1", "raw": true }))
        );
        assert_eq!(
            result.steps[0].response.body,
            Some(json!({ "id": "resp-1", "raw": true }))
        );
        assert_eq!(result.response.messages.len(), 1);
        assert_eq!(
            result
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata
                    .get("test-provider")
                    .and_then(|value| value.get("trace"))
                    .and_then(serde_json::Value::as_str)),
            Some("abc")
        );
    }

    #[tokio::test]
    async fn generate_text_include_can_omit_large_metadata_bodies() {
        let model = FakeLanguageModel;
        let result = generate_text(
            &model,
            ChatRequest::new(vec![ChatMessage::user("hi").build()]),
            GenerateOptions {
                include: GenerateTextInclude::metadata_only(),
                ..GenerateOptions::default()
            },
        )
        .await
        .expect("generate text");

        assert_eq!(result.request.body, None);
        assert_eq!(result.steps[0].request.body, None);
        assert_eq!(result.response.body, None);
        assert_eq!(result.steps[0].response.body, None);
        assert_eq!(
            result
                .response
                .metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("req_123")
        );
    }
}
