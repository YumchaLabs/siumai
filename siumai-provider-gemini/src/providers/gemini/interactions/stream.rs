use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use eventsource_stream::Event;
use futures_util::Stream;
use serde_json::{Map, Value};
use std::pin::Pin;

use crate::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig, execute_json_request};
use crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers;
use crate::execution::http::interceptor::{HttpRequestContext, generate_request_id};
use crate::execution::http::transport::HttpTransportGetRequest;
use crate::streaming::{
    ChatStream, ChatStreamEvent, ChatStreamHandle, SseEventConverter, SseEventFuture, SseStreamExt,
    StreamStateTracker,
};
use crate::types::{
    CancelHandle, ChatRequest, ChatResponse, ChatStreamFileData, ChatStreamFilePart,
    ChatStreamFinishInfo, ChatStreamPart, ChatStreamToolCall, ChatStreamToolResult, ContentPart,
    HttpRequestInfo, HttpResponseInfo, MessageContent, ResponseMetadata, SourcePart,
    StreamProviderMetadata, Usage,
};
use futures_util::{StreamExt, TryStreamExt};
use reqwest::header::{ACCEPT, CACHE_CONTROL, CONNECTION, HeaderMap};

use super::GoogleInteractionsLanguageModel;
use super::response::{
    BUILTIN_TOOL_CALL_TYPES, BUILTIN_TOOL_RESULT_TYPES, annotations_to_sources, arguments_value,
    builtin_tool_result_to_sources, convert_usage, map_finish_reason, part_provider_metadata,
    response_provider_metadata, source_dedupe_key, string_field,
};
use super::runtime::{
    build_execution_config, interaction_cancel_url, interaction_id, interaction_stream_url,
    interactions_http_config, interactions_url, is_terminal_response,
};

const AGENT_STREAM_MAX_RETRIES: u32 = 3;
const AGENT_STREAM_RETRY_DELAY_MS: u64 = 500;

type AgentStreamBody = Pin<Box<dyn Stream<Item = Result<Vec<u8>, LlmError>> + Send + 'static>>;

struct AgentStreamResponse {
    headers: HeaderMap,
    body: AgentStreamBody,
}

pub(super) async fn execute_interactions_stream(
    model: &GoogleInteractionsLanguageModel,
    request: ChatRequest,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> Result<ChatStream, LlmError> {
    let prepared = model.prepare_request_body(&request, true)?;
    let request_body = serde_json::to_value(prepared.body).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to serialize google.interactions streaming request body: {error}"
        ))
    })?;
    let request_info = serde_json::to_string(&request_body)
        .ok()
        .map(|body| HttpRequestInfo { body: Some(body) });

    let http_config =
        interactions_http_config(&model.config().http_config, request.http_config.as_ref());
    let execution_config = build_execution_config(model, http_client, retry_options.clone()).await;
    let headers_base = execution_config
        .provider_spec
        .build_headers(&execution_config.provider_context)?;
    let converter = GoogleInteractionsEventConverter::new(
        model.provider(),
        model.model_id().to_string(),
        model.config().generate_id.clone(),
        prepared.warnings,
    );
    let request_id = generate_request_id();
    let stream = execute_sse_stream_request_with_headers(
        &execution_config.http_client,
        &execution_config.provider_id,
        Some(execution_config.provider_spec.as_ref()),
        &interactions_url(model.base_url()),
        request_id,
        headers_base,
        request_body,
        &execution_config.interceptors,
        retry_options,
        Some(http_config.clone()),
        converter,
        model.config().http_config.stream_disable_compression
            || request
                .http_config
                .as_ref()
                .map(|config| config.stream_disable_compression)
                .unwrap_or(false),
        execution_config.transport.clone(),
    )
    .await?;

    Ok(attach_interactions_stream_request_metadata(
        stream,
        request_info,
    ))
}

pub(super) async fn execute_interactions_agent_stream(
    model: &GoogleInteractionsLanguageModel,
    request: ChatRequest,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
    cancel: Option<CancelHandle>,
) -> Result<ChatStream, LlmError> {
    let prepared = model.prepare_request_body(&request, false)?;
    let request_body = serde_json::to_value(prepared.body).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to serialize google.interactions agent streaming request body: {error}"
        ))
    })?;
    let request_info = serde_json::to_string(&request_body)
        .ok()
        .map(|body| HttpRequestInfo { body: Some(body) });

    let http_config =
        interactions_http_config(&model.config().http_config, request.http_config.as_ref());
    let execution_config = build_execution_config(model, http_client, retry_options.clone()).await;
    let post_result = execute_json_request(
        &execution_config,
        &interactions_url(model.base_url()),
        HttpBody::Json(request_body.clone()),
        Some(&http_config),
        false,
    )
    .await?;

    let interaction_id = interaction_id(&post_result.json).ok_or_else(|| {
        LlmError::ParseError(
            "google.interactions: background POST response did not include an interaction id; cannot stream the result."
                .to_string(),
        )
    })?;

    let converter = GoogleInteractionsEventConverter::new(
        model.provider(),
        model.model_id().to_string(),
        model.config().generate_id.clone(),
        prepared.warnings,
    );
    converter.seed_from_interaction_response(&post_result.json);

    let headers_base = execution_config
        .provider_spec
        .build_headers(&execution_config.provider_context)?;
    let stream = if is_terminal_response(&post_result.json)? {
        converter
            .stream_from_terminal_response(post_result.json.clone(), post_result.headers.clone())
    } else {
        stream_agent_interaction_events(
            execution_config,
            model.base_url().to_string(),
            interaction_id.to_string(),
            headers_base,
            http_config.clone(),
            converter,
            cancel,
            model.config().http_config.stream_disable_compression
                || request
                    .http_config
                    .as_ref()
                    .map(|config| config.stream_disable_compression)
                    .unwrap_or(false),
        )
    };

    Ok(attach_interactions_stream_request_metadata(
        stream,
        request_info,
    ))
}

pub(super) async fn execute_interactions_agent_stream_handle(
    model: &GoogleInteractionsLanguageModel,
    request: ChatRequest,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> Result<ChatStreamHandle, LlmError> {
    let cancel = CancelHandle::new();
    let stream = execute_interactions_agent_stream(
        model,
        request,
        http_client,
        retry_options,
        Some(cancel.clone()),
    )
    .await?;
    Ok(ChatStreamHandle { stream, cancel })
}

#[allow(clippy::too_many_arguments)]
fn stream_agent_interaction_events(
    execution_config: HttpExecutionConfig,
    base_url: String,
    interaction_id: String,
    headers_base: HeaderMap,
    http_config: crate::types::HttpConfig,
    converter: GoogleInteractionsEventConverter,
    cancel: Option<CancelHandle>,
    disable_compression: bool,
) -> ChatStream {
    let stream = async_stream::try_stream! {
        let mut last_event_id: Option<String> = None;
        let mut attempts = 0_u32;
        let mut completed = false;
        let mut cancelled = false;

        while !completed {
            if cancel.as_ref().is_some_and(CancelHandle::is_cancelled) {
                cancelled = true;
                break;
            }

            let url = interaction_stream_url(&base_url, &interaction_id, last_event_id.as_deref());
            let response = match open_agent_stream_response(
                &execution_config,
                &url,
                &headers_base,
                &http_config,
                disable_compression,
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    if cancel.as_ref().is_some_and(CancelHandle::is_cancelled) {
                        cancelled = true;
                        break;
                    }
                    attempts += 1;
                    if attempts >= AGENT_STREAM_MAX_RETRIES {
                        Err(error)?;
                    }
                    cancel_aware_retry_delay(attempts, cancel.as_ref()).await?;
                    continue;
                }
            };

            let response_headers = response.headers.clone();
            let mut received_any_event = false;
            let mut body = response.body.into_sse_stream();

            while let Some(next) = next_agent_sse_event(&mut body, cancel.as_ref()).await {
                let event = next?;
                if converter.is_stream_end_event(&event) {
                    completed = true;
                    for item in converter.handle_stream_end_events() {
                        yield attach_response_headers_to_result(item, &response_headers)?;
                    }
                    break;
                }
                if event.data.trim().is_empty() {
                    continue;
                }

                received_any_event = true;
                if let Some(event_id) =
                    event_id_from_sse_data(&event.data).or_else(|| {
                        (!event.id.is_empty()).then(|| event.id.clone())
                    })
                {
                    last_event_id = Some(event_id);
                }

                for item in converter.convert_event(event).await {
                    yield item?;
                }

                if converter.is_complete() {
                    completed = true;
                    for item in converter.handle_stream_end_events() {
                        yield attach_response_headers_to_result(item, &response_headers)?;
                    }
                    break;
                }

                if cancel.as_ref().is_some_and(CancelHandle::is_cancelled) {
                    cancelled = true;
                    break;
                }
            }

            if completed || cancelled {
                break;
            }

            if received_any_event {
                attempts = 0;
            } else {
                attempts += 1;
                if attempts >= AGENT_STREAM_MAX_RETRIES {
                    Err(LlmError::StreamError(
                        "google.interactions: SSE stream closed without producing any events."
                            .to_string(),
                    ))?;
                }
                cancel_aware_retry_delay(attempts, cancel.as_ref()).await?;
            }
        }

        if cancelled {
            best_effort_cancel_interaction(&execution_config, &base_url, &interaction_id, &http_config).await;
        }
    };

    Box::pin(stream)
}

async fn next_agent_sse_event<S>(
    stream: &mut crate::streaming::SseStream<S>,
    cancel: Option<&CancelHandle>,
) -> Option<Result<Event, LlmError>>
where
    S: futures_util::Stream<Item = Result<Vec<u8>, LlmError>> + Unpin,
{
    if let Some(cancel) = cancel {
        tokio::select! {
            _ = cancel.cancelled() => None,
            item = stream.next() => item.map(|result| {
                result.map_err(|error| LlmError::StreamError(format!("SSE parsing error: {error}")))
            }),
        }
    } else {
        stream.next().await.map(|result| {
            result.map_err(|error| LlmError::StreamError(format!("SSE parsing error: {error}")))
        })
    }
}

async fn cancel_aware_retry_delay(
    attempt: u32,
    cancel: Option<&CancelHandle>,
) -> Result<(), LlmError> {
    crate::utils::cancel::delay(
        Some(AGENT_STREAM_RETRY_DELAY_MS * u64::from(attempt)),
        cancel,
    )
    .await
}

fn event_id_from_sse_data(data: &str) -> Option<String> {
    serde_json::from_str::<Value>(data).ok().and_then(|raw| {
        raw.get("event_id")
            .and_then(Value::as_str)
            .filter(|event_id| !event_id.is_empty())
            .map(ToOwned::to_owned)
    })
}

async fn open_agent_stream_response(
    execution_config: &HttpExecutionConfig,
    url: &str,
    headers_base: &HeaderMap,
    http_config: &crate::types::HttpConfig,
    disable_compression: bool,
) -> Result<AgentStreamResponse, LlmError> {
    let effective_headers = execution_config
        .provider_spec
        .merge_request_headers(headers_base.clone(), &http_config.headers);
    let mut headers = effective_headers.clone();
    headers.insert(ACCEPT, "text/event-stream".parse().expect("valid accept"));
    headers.insert(
        CACHE_CONTROL,
        "no-cache".parse().expect("valid cache-control"),
    );
    headers.insert(CONNECTION, "keep-alive".parse().expect("valid connection"));
    if disable_compression {
        headers.insert(
            reqwest::header::ACCEPT_ENCODING,
            "identity".parse().expect("valid accept-encoding"),
        );
    }

    let ctx = HttpRequestContext {
        request_id: generate_request_id(),
        provider_id: execution_config.provider_id.clone(),
        url: url.to_string(),
        stream: true,
    };
    let empty_body = serde_json::json!({});

    if let Some(transport) = &execution_config.transport {
        let response = transport
            .execute_get_stream(HttpTransportGetRequest {
                ctx: ctx.clone(),
                url: url.to_string(),
                headers,
            })
            .await?;
        if !(200..300).contains(&response.status) {
            let bytes = response.body.into_stream().try_concat().await?;
            let text = String::from_utf8_lossy(&bytes);
            let fallback_message = reqwest::StatusCode::from_u16(response.status)
                .ok()
                .and_then(|status| status.canonical_reason());
            let error = crate::execution::executors::errors::classify_http_error(
                &execution_config.provider_id,
                Some(execution_config.provider_spec.as_ref()),
                response.status,
                &text,
                &response.headers,
                fallback_message,
            );
            for interceptor in &execution_config.interceptors {
                interceptor.on_error(&ctx, &error);
            }
            return Err(error);
        }
        return Ok(AgentStreamResponse {
            headers: response.headers,
            body: Box::pin(response.body.into_stream()),
        });
    }

    let mut rb = execution_config
        .http_client
        .get(url)
        .headers(headers.clone());
    if let Some(timeout) = http_config.timeout {
        rb = rb.timeout(timeout);
    }
    rb = rb
        .header(ACCEPT, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive");
    if disable_compression {
        rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
    }
    for interceptor in &execution_config.interceptors {
        rb = interceptor.on_before_send(&ctx, rb, &empty_body, &headers)?;
    }
    let response = rb
        .send()
        .await
        .map_err(|error| LlmError::HttpError(format!("Failed to send request: {error}")))?;
    if !response.status().is_success() {
        return Err(
            crate::execution::executors::errors::classify_error_with_text(
                &execution_config.provider_id,
                Some(execution_config.provider_spec.as_ref()),
                response,
                &ctx,
                &execution_config.interceptors,
            )
            .await,
        );
    }
    for interceptor in &execution_config.interceptors {
        interceptor.on_response(&ctx, &response)?;
    }
    let headers = response.headers().clone();
    Ok(AgentStreamResponse {
        headers,
        body: Box::pin(response.bytes_stream().map(|chunk| {
            chunk
                .map(|bytes| bytes.to_vec())
                .map_err(|error| LlmError::HttpError(format!("Stream error: {error}")))
        })),
    })
}

async fn best_effort_cancel_interaction(
    execution_config: &HttpExecutionConfig,
    base_url: &str,
    interaction_id: &str,
    http_config: &crate::types::HttpConfig,
) {
    let url = interaction_cancel_url(base_url, interaction_id);
    let _ = execute_json_request(
        execution_config,
        &url,
        HttpBody::Json(serde_json::json!({})),
        Some(http_config),
        false,
    )
    .await;
}

fn attach_response_headers_to_result(
    result: Result<ChatStreamEvent, LlmError>,
    headers: &HeaderMap,
) -> Result<ChatStreamEvent, LlmError> {
    result.map(|event| attach_response_headers_to_event(event, headers))
}

fn attach_response_headers_to_stream(stream: ChatStream, headers: HeaderMap) -> ChatStream {
    if headers.is_empty() {
        return stream;
    }

    Box::pin(stream.map(move |event| {
        let headers = headers.clone();
        event.map(|event| attach_response_headers_to_event(event, &headers))
    }))
}

fn attach_response_headers_to_event(
    event: ChatStreamEvent,
    headers: &HeaderMap,
) -> ChatStreamEvent {
    let headers = crate::execution::http::headers::headermap_to_hashmap(headers);
    match event {
        ChatStreamEvent::StreamEnd { mut response } => {
            if let Some(info) = response.response.as_mut() {
                if info.headers.is_empty() {
                    info.headers = headers;
                }
            } else {
                response.response = Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: response.model.clone(),
                    headers,
                    body: None,
                });
            }
            ChatStreamEvent::StreamEnd { response }
        }
        other => other,
    }
}

fn attach_interactions_stream_request_metadata(
    stream: ChatStream,
    request_info: Option<HttpRequestInfo>,
) -> ChatStream {
    let Some(request_info) = request_info else {
        return stream;
    };

    Box::pin(stream.map(move |event| {
        let request_info = request_info.clone();
        event.map(|event| match event {
            ChatStreamEvent::StreamEnd { mut response } => {
                if response.request.is_none() {
                    response.request = Some(request_info);
                }
                ChatStreamEvent::StreamEnd { response }
            }
            other => other,
        })
    }))
}

#[derive(Clone)]
pub(super) struct GoogleInteractionsEventConverter {
    provider: String,
    fallback_model_id: String,
    generate_id: Option<super::super::SharedIdGenerator>,
    warnings: Vec<crate::types::Warning>,
    state: StreamStateTracker,
    inner: Arc<Mutex<GoogleInteractionsStreamState>>,
}

#[derive(Debug, Default)]
struct GoogleInteractionsStreamState {
    interaction_id: Option<String>,
    model_id: Option<String>,
    created: Option<chrono::DateTime<chrono::Utc>>,
    service_tier: Option<String>,
    finish_status: Option<String>,
    usage: Option<Usage>,
    has_function_call: bool,
    open_blocks: HashMap<i64, OpenBlockState>,
    emitted_source_keys: HashSet<String>,
    error_payloads: Vec<Value>,
}

#[derive(Debug, Clone)]
enum OpenBlockState {
    PendingModelOutput {
        id: String,
    },
    Text {
        id: String,
    },
    Reasoning {
        id: String,
        signature: Option<String>,
    },
    Image {
        data: Option<String>,
        mime_type: Option<String>,
        uri: Option<String>,
    },
    FunctionCall {
        tool_call_id: String,
        tool_name: String,
        arguments_accum: String,
        signature: Option<String>,
    },
    BuiltinToolCall {
        block_type: String,
        tool_call_id: String,
        tool_name: String,
        arguments: Value,
        call_emitted: bool,
    },
    BuiltinToolResult {
        block_type: String,
        call_id: String,
        tool_name: String,
        result: Value,
        is_error: Option<bool>,
        result_emitted: bool,
    },
    Unknown,
}

impl GoogleInteractionsEventConverter {
    fn new(
        provider: String,
        fallback_model_id: String,
        generate_id: Option<super::super::SharedIdGenerator>,
        warnings: Vec<crate::types::Warning>,
    ) -> Self {
        Self {
            provider,
            fallback_model_id,
            generate_id,
            warnings,
            state: StreamStateTracker::new(),
            inner: Arc::new(Mutex::new(GoogleInteractionsStreamState::default())),
        }
    }

    fn generate_id(&self) -> String {
        self.generate_id
            .as_ref()
            .map(|generate_id| generate_id())
            .unwrap_or_else(generate_request_id)
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut GoogleInteractionsStreamState) -> R) -> R {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut state)
    }

    fn seed_from_interaction_response(&self, raw: &Value) {
        if let Some(object) = raw.as_object() {
            self.with_state(|state| {
                if let Some(id) = string_field(object, "id").filter(|id| !id.is_empty()) {
                    state.interaction_id = Some(id.to_string());
                }
                if let Some(model) = string_field(object, "model")
                    .or_else(|| string_field(object, "agent"))
                    .filter(|id| !id.is_empty())
                {
                    state.model_id = Some(model.to_string());
                }
                if let Some(status) = string_field(object, "status") {
                    state.finish_status = Some(status.to_string());
                }
                if let Some(service_tier) =
                    string_field(object, "service_tier").filter(|id| !id.is_empty())
                {
                    state.service_tier = Some(service_tier.to_string());
                }
                if let Some(usage) = object.get("usage").and_then(convert_usage) {
                    state.usage = Some(usage);
                }
            });
        }
    }

    fn stream_from_terminal_response(&self, raw: Value, headers: HeaderMap) -> ChatStream {
        self.seed_from_interaction_response(&raw);
        let mut out = self.start_events();
        if let Some(steps) = raw
            .as_object()
            .and_then(|object| object.get("steps"))
            .and_then(Value::as_array)
        {
            out.extend(self.events_from_completed_steps(steps));
        }
        out.extend(self.final_events().into_iter().filter_map(Result::ok));
        attach_response_headers_to_stream(
            Box::pin(futures_util::stream::iter(out.into_iter().map(Ok))),
            headers,
        )
    }

    fn events_from_completed_steps(&self, steps: &[Value]) -> Vec<ChatStreamEvent> {
        let mut out = Vec::new();
        let mut block_index = 0_i64;

        for step in steps {
            let Some(step_obj) = step.as_object() else {
                continue;
            };
            let Some(step_type) = string_field(step_obj, "type") else {
                continue;
            };

            match step_type {
                "model_output" => {
                    if let Some(blocks) = step_obj.get("content").and_then(Value::as_array) {
                        for block in blocks {
                            let Some(block_obj) = block.as_object() else {
                                continue;
                            };
                            match string_field(block_obj, "type") {
                                Some("text") => {
                                    let id = self.block_id(block_index);
                                    out.push(part(ChatStreamPart::TextStart {
                                        id: id.clone(),
                                        provider_metadata: None,
                                    }));
                                    if let Some(text) = string_field(block_obj, "text")
                                        .filter(|text| !text.is_empty())
                                    {
                                        out.push(part(ChatStreamPart::TextDelta {
                                            id: id.clone(),
                                            delta: text.to_string(),
                                            provider_metadata: None,
                                        }));
                                    }
                                    if let Some(sources) =
                                        block_obj.get("annotations").and_then(Value::as_array)
                                    {
                                        out.extend(self.annotation_source_events(sources));
                                    }
                                    out.push(part(ChatStreamPart::TextEnd {
                                        id,
                                        provider_metadata: self.interaction_provider_metadata(),
                                    }));
                                    block_index += 1;
                                }
                                Some("image") => {
                                    if let Some(event) = self.image_event(
                                        string_field(block_obj, "data"),
                                        string_field(block_obj, "mime_type"),
                                        string_field(block_obj, "uri"),
                                    ) {
                                        out.push(event);
                                    }
                                    block_index += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "thought" => {
                    let id = self.block_id(block_index);
                    out.push(part(ChatStreamPart::ReasoningStart {
                        id: id.clone(),
                        provider_metadata: None,
                    }));
                    if let Some(summary) = step_obj.get("summary").and_then(Value::as_array) {
                        let text = summary
                            .iter()
                            .filter_map(Value::as_object)
                            .filter(|item| string_field(item, "type") == Some("text"))
                            .filter_map(|item| string_field(item, "text"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !text.is_empty() {
                            out.push(part(ChatStreamPart::ReasoningDelta {
                                id: id.clone(),
                                delta: text,
                                provider_metadata: None,
                            }));
                        }
                    }
                    out.push(part(ChatStreamPart::ReasoningEnd {
                        id,
                        provider_metadata: self
                            .part_provider_metadata(string_field(step_obj, "signature")),
                    }));
                    block_index += 1;
                }
                "function_call" => {
                    let tool_call_id = string_field(step_obj, "id")
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| self.block_id(block_index));
                    let tool_name = string_field(step_obj, "name")
                        .unwrap_or("unknown")
                        .to_string();
                    let input = serde_json::to_string(&arguments_value(step_obj.get("arguments")))
                        .unwrap_or_else(|_| "{}".to_string());
                    self.with_state(|state| {
                        state.has_function_call = true;
                    });
                    out.push(part(ChatStreamPart::ToolInputStart {
                        id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        provider_metadata: None,
                        provider_executed: None,
                        dynamic: None,
                        title: None,
                    }));
                    out.push(part(ChatStreamPart::ToolInputDelta {
                        id: tool_call_id.clone(),
                        delta: input.clone(),
                        provider_metadata: None,
                    }));
                    out.push(part(ChatStreamPart::ToolInputEnd {
                        id: tool_call_id.clone(),
                        provider_metadata: None,
                    }));
                    out.push(part(ChatStreamPart::ToolCall(ChatStreamToolCall {
                        tool_call_id,
                        tool_name,
                        input,
                        provider_executed: None,
                        dynamic: None,
                        provider_metadata: self
                            .part_provider_metadata(string_field(step_obj, "signature")),
                    })));
                    block_index += 1;
                }
                other if BUILTIN_TOOL_CALL_TYPES.contains(&other) => {
                    let tool_call_id = string_field(step_obj, "id")
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| self.block_id(block_index));
                    out.push(part(ChatStreamPart::ToolCall(ChatStreamToolCall {
                        tool_call_id,
                        tool_name: builtin_tool_call_name(other, step_obj),
                        input: serde_json::to_string(&arguments_value(step_obj.get("arguments")))
                            .unwrap_or_else(|_| "{}".to_string()),
                        provider_executed: Some(true),
                        dynamic: None,
                        provider_metadata: None,
                    })));
                    block_index += 1;
                }
                other if BUILTIN_TOOL_RESULT_TYPES.contains(&other) => {
                    let call_id = string_field(step_obj, "call_id")
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| self.block_id(block_index));
                    let result = step_obj.get("result").cloned().unwrap_or(Value::Null);
                    let is_error = step_obj.get("is_error").and_then(Value::as_bool);
                    out.push(part(ChatStreamPart::ToolResult(ChatStreamToolResult {
                        tool_call_id: call_id.clone(),
                        tool_name: builtin_tool_result_name(other, step_obj),
                        result: non_null_json(result.clone()),
                        is_error,
                        preliminary: None,
                        dynamic: None,
                        provider_metadata: None,
                    })));
                    out.extend(
                        self.builtin_result_source_events(other, &call_id, result, is_error),
                    );
                    block_index += 1;
                }
                _ => {}
            }
        }

        out
    }

    fn is_complete(&self) -> bool {
        self.with_state(|state| {
            state.finish_status.as_deref().is_some_and(|status| {
                matches!(status, "completed" | "failed" | "cancelled" | "incomplete")
            })
        })
    }

    fn start_events(&self) -> Vec<ChatStreamEvent> {
        if !self.state.needs_stream_start() {
            return Vec::new();
        }

        vec![
            ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: None,
                    model: Some(self.fallback_model_id.clone()),
                    created: Some(chrono::Utc::now()),
                    provider: self.provider.clone(),
                    request_id: None,
                    headers: None,
                    body: None,
                },
            },
            ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart {
                    warnings: self.warnings.clone(),
                },
            },
        ]
    }

    fn convert_json_event(&self, raw: Value) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let mut out = self.start_events();
        let Some(object) = raw.as_object() else {
            out.push(ChatStreamEvent::Part {
                part: ChatStreamPart::Error {
                    error: serde_json::json!({
                        "message": "google.interactions SSE event must be a JSON object"
                    }),
                },
            });
            self.with_state(|state| {
                state.finish_status = Some("failed".to_string());
                state.error_payloads.push(raw);
            });
            return ok_events(out);
        };

        match string_field(object, "event_type") {
            Some("interaction.created") => {
                out.extend(self.handle_interaction_created(object));
            }
            Some("step.start") => {
                out.extend(self.handle_step_start(object));
            }
            Some("step.delta") => {
                out.extend(self.handle_step_delta(object));
            }
            Some("step.stop") => {
                out.extend(self.handle_step_stop(object));
            }
            Some("interaction.status_update")
            | Some("interaction.in_progress")
            | Some("interaction.requires_action") => {
                self.handle_status_update(object);
            }
            Some("interaction.completed") => {
                self.handle_interaction_completed(object);
            }
            Some("error") => {
                self.handle_error_event(object, &mut out);
            }
            _ => {}
        }

        ok_events(out)
    }

    fn handle_interaction_created(&self, object: &Map<String, Value>) -> Vec<ChatStreamEvent> {
        let interaction = object.get("interaction").and_then(Value::as_object);
        let interaction_id = interaction
            .and_then(|value| string_field(value, "id"))
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let model_id = interaction
            .and_then(|value| string_field(value, "model").or_else(|| string_field(value, "agent")))
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let created = interaction
            .and_then(|value| string_field(value, "created"))
            .and_then(parse_timestamp);

        self.with_state(|state| {
            if interaction_id.is_some() {
                state.interaction_id = interaction_id.clone();
            }
            if model_id.is_some() {
                state.model_id = model_id.clone();
            }
            if created.is_some() {
                state.created = created;
            }
        });

        vec![ChatStreamEvent::Part {
            part: ChatStreamPart::ResponseMetadata(ResponseMetadata {
                id: interaction_id,
                model: model_id,
                created,
                provider: self.provider.clone(),
                request_id: None,
                headers: None,
                body: None,
            }),
        }]
    }

    fn handle_step_start(&self, object: &Map<String, Value>) -> Vec<ChatStreamEvent> {
        let Some(index) = object.get("index").and_then(Value::as_i64) else {
            return Vec::new();
        };
        let Some(step) = object.get("step").and_then(Value::as_object) else {
            return Vec::new();
        };
        let block_id = self.block_id(index);
        let Some(step_type) = string_field(step, "type") else {
            self.with_state(|state| {
                state.open_blocks.insert(index, OpenBlockState::Unknown);
            });
            return Vec::new();
        };

        let mut out = Vec::new();
        if step_type == "model_output" {
            let initial = step
                .get("content")
                .and_then(Value::as_array)
                .and_then(|content| content.first())
                .and_then(Value::as_object);
            match initial.and_then(|block| string_field(block, "type")) {
                Some("text") => {
                    self.with_state(|state| {
                        state.open_blocks.insert(
                            index,
                            OpenBlockState::Text {
                                id: block_id.clone(),
                            },
                        );
                    });
                    out.push(part(ChatStreamPart::TextStart {
                        id: block_id,
                        provider_metadata: None,
                    }));
                    if let Some(sources) =
                        initial.and_then(|block| block.get("annotations").and_then(Value::as_array))
                    {
                        out.extend(self.annotation_source_events(sources));
                    }
                }
                Some("image") => {
                    self.with_state(|state| {
                        state.open_blocks.insert(
                            index,
                            OpenBlockState::Image {
                                data: initial
                                    .and_then(|block| string_field(block, "data"))
                                    .map(ToOwned::to_owned),
                                mime_type: initial
                                    .and_then(|block| string_field(block, "mime_type"))
                                    .map(ToOwned::to_owned),
                                uri: initial
                                    .and_then(|block| string_field(block, "uri"))
                                    .map(ToOwned::to_owned),
                            },
                        );
                    });
                }
                _ => {
                    self.with_state(|state| {
                        state
                            .open_blocks
                            .insert(index, OpenBlockState::PendingModelOutput { id: block_id });
                    });
                }
            }
        } else if step_type == "thought" {
            let signature = string_field(step, "signature").map(ToOwned::to_owned);
            self.with_state(|state| {
                state.open_blocks.insert(
                    index,
                    OpenBlockState::Reasoning {
                        id: block_id.clone(),
                        signature,
                    },
                );
            });
            out.push(part(ChatStreamPart::ReasoningStart {
                id: block_id.clone(),
                provider_metadata: None,
            }));
            out.extend(
                step.get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_object)
                    .filter(|item| string_field(item, "type") == Some("text"))
                    .filter_map(|item| string_field(item, "text"))
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        part(ChatStreamPart::ReasoningDelta {
                            id: block_id.clone(),
                            delta: text.to_string(),
                            provider_metadata: None,
                        })
                    }),
            );
        } else if step_type == "function_call" {
            let tool_call_id = string_field(step, "id").unwrap_or(&block_id).to_string();
            let tool_name = string_field(step, "name").unwrap_or("unknown").to_string();
            let signature = string_field(step, "signature").map(ToOwned::to_owned);
            self.with_state(|state| {
                state.has_function_call = true;
                state.open_blocks.insert(
                    index,
                    OpenBlockState::FunctionCall {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        arguments_accum: String::new(),
                        signature,
                    },
                );
            });
            out.push(part(ChatStreamPart::ToolInputStart {
                id: tool_call_id,
                tool_name,
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            }));
        } else if BUILTIN_TOOL_CALL_TYPES.contains(&step_type) {
            let tool_name = builtin_tool_call_name(step_type, step);
            let tool_call_id = string_field(step, "id").unwrap_or(&block_id).to_string();
            self.with_state(|state| {
                state.open_blocks.insert(
                    index,
                    OpenBlockState::BuiltinToolCall {
                        block_type: step_type.to_string(),
                        tool_call_id,
                        tool_name,
                        arguments: arguments_value(step.get("arguments")),
                        call_emitted: false,
                    },
                );
            });
        } else if BUILTIN_TOOL_RESULT_TYPES.contains(&step_type) {
            let tool_name = builtin_tool_result_name(step_type, step);
            let call_id = string_field(step, "call_id")
                .unwrap_or(&block_id)
                .to_string();
            self.with_state(|state| {
                state.open_blocks.insert(
                    index,
                    OpenBlockState::BuiltinToolResult {
                        block_type: step_type.to_string(),
                        call_id,
                        tool_name,
                        result: step.get("result").cloned().unwrap_or(Value::Null),
                        is_error: step.get("is_error").and_then(Value::as_bool),
                        result_emitted: false,
                    },
                );
            });
        } else {
            self.with_state(|state| {
                state.open_blocks.insert(index, OpenBlockState::Unknown);
            });
        }

        out
    }

    fn handle_step_delta(&self, object: &Map<String, Value>) -> Vec<ChatStreamEvent> {
        let Some(index) = object.get("index").and_then(Value::as_i64) else {
            return Vec::new();
        };
        let Some(delta) = object.get("delta").and_then(Value::as_object) else {
            return Vec::new();
        };
        let delta_type = string_field(delta, "type");
        let mut out = Vec::new();

        if matches!(
            delta_type,
            Some("text") | Some("text_annotation") | Some("text_annotation_delta")
        ) {
            let promoted = self.with_state(|state| match state.open_blocks.get(&index).cloned() {
                Some(OpenBlockState::PendingModelOutput { id }) => {
                    state
                        .open_blocks
                        .insert(index, OpenBlockState::Text { id: id.clone() });
                    Some(id)
                }
                _ => None,
            });
            if let Some(id) = promoted {
                out.push(part(ChatStreamPart::TextStart {
                    id,
                    provider_metadata: None,
                }));
            }
        }

        if delta_type == Some("image") {
            let event = self.inline_image_delta_event(delta);
            let should_emit = self.with_state(|state| {
                matches!(
                    state.open_blocks.get(&index),
                    Some(OpenBlockState::PendingModelOutput { .. })
                        | Some(OpenBlockState::Text { .. })
                        | Some(OpenBlockState::Image { .. })
                )
            });
            if should_emit {
                if let Some(event) = event {
                    out.push(event);
                }
                self.with_state(|state| {
                    if let Some(OpenBlockState::Image { data, uri, .. }) =
                        state.open_blocks.get_mut(&index)
                    {
                        *data = None;
                        *uri = None;
                    }
                });
            }
            return out;
        }

        let block = self.with_state(|state| state.open_blocks.get(&index).cloned());
        match block {
            Some(OpenBlockState::Text { id }) if delta_type == Some("text") => {
                if let Some(text) = string_field(delta, "text").filter(|text| !text.is_empty()) {
                    out.push(part(ChatStreamPart::TextDelta {
                        id,
                        delta: text.to_string(),
                        provider_metadata: None,
                    }));
                }
            }
            Some(OpenBlockState::Text { .. })
                if matches!(
                    delta_type,
                    Some("text_annotation") | Some("text_annotation_delta")
                ) =>
            {
                if let Some(annotations) = delta.get("annotations").and_then(Value::as_array) {
                    out.extend(self.annotation_source_events(annotations));
                }
            }
            Some(OpenBlockState::Image { .. }) if delta_type == Some("image") => {
                self.with_state(|state| {
                    if let Some(OpenBlockState::Image {
                        data,
                        mime_type,
                        uri,
                        ..
                    }) = state.open_blocks.get_mut(&index)
                    {
                        if let Some(value) = string_field(delta, "data") {
                            *data = Some(value.to_string());
                        }
                        if let Some(value) = string_field(delta, "mime_type") {
                            *mime_type = Some(value.to_string());
                        }
                        if let Some(value) = string_field(delta, "uri") {
                            *uri = Some(value.to_string());
                        }
                    }
                });
            }
            Some(OpenBlockState::Reasoning { id, .. }) => match delta_type {
                Some("thought_summary") => {
                    if let Some(content) = delta.get("content").and_then(Value::as_object)
                        && string_field(content, "type") == Some("text")
                        && let Some(text) =
                            string_field(content, "text").filter(|text| !text.is_empty())
                    {
                        out.push(part(ChatStreamPart::ReasoningDelta {
                            id,
                            delta: text.to_string(),
                            provider_metadata: None,
                        }));
                    }
                }
                Some("thought_signature") => {
                    let signature = string_field(delta, "signature").map(ToOwned::to_owned);
                    self.with_state(|state| {
                        if let Some(OpenBlockState::Reasoning {
                            signature: target, ..
                        }) = state.open_blocks.get_mut(&index)
                        {
                            *target = signature;
                        }
                    });
                }
                _ => {}
            },
            Some(OpenBlockState::FunctionCall { tool_call_id, .. })
                if delta_type == Some("arguments_delta") =>
            {
                if let Some(slice) =
                    string_field(delta, "arguments").filter(|value| !value.is_empty())
                {
                    out.push(part(ChatStreamPart::ToolInputDelta {
                        id: tool_call_id,
                        delta: slice.to_string(),
                        provider_metadata: None,
                    }));
                    self.with_state(|state| {
                        state.has_function_call = true;
                        if let Some(OpenBlockState::FunctionCall {
                            arguments_accum,
                            tool_call_id,
                            signature,
                            ..
                        }) = state.open_blocks.get_mut(&index)
                        {
                            arguments_accum.push_str(slice);
                            if let Some(id) = string_field(delta, "id") {
                                *tool_call_id = id.to_string();
                            }
                            if let Some(sig) = string_field(delta, "signature") {
                                *signature = Some(sig.to_string());
                            }
                        }
                    });
                }
            }
            Some(OpenBlockState::BuiltinToolCall { block_type, .. })
                if delta_type == Some(block_type.as_str()) =>
            {
                self.with_state(|state| {
                    if let Some(OpenBlockState::BuiltinToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    }) = state.open_blocks.get_mut(&index)
                    {
                        if let Some(id) = string_field(delta, "id") {
                            *tool_call_id = id.to_string();
                        }
                        if let Some(value) = delta.get("arguments") {
                            *arguments = arguments_value(Some(value));
                        }
                        if let Some(name) = string_field(delta, "name") {
                            *tool_name = name.to_string();
                        }
                    }
                });
            }
            Some(OpenBlockState::BuiltinToolResult { block_type, .. })
                if delta_type == Some(block_type.as_str()) =>
            {
                self.with_state(|state| {
                    if let Some(OpenBlockState::BuiltinToolResult {
                        call_id,
                        tool_name,
                        result,
                        is_error,
                        ..
                    }) = state.open_blocks.get_mut(&index)
                    {
                        if let Some(id) = string_field(delta, "call_id") {
                            *call_id = id.to_string();
                        }
                        if let Some(value) = delta.get("result") {
                            *result = value.clone();
                        }
                        if let Some(value) = delta.get("is_error").and_then(Value::as_bool) {
                            *is_error = Some(value);
                        }
                        if let Some(name) = string_field(delta, "name") {
                            *tool_name = name.to_string();
                        }
                    }
                });
            }
            _ => {}
        }

        out
    }

    fn handle_step_stop(&self, object: &Map<String, Value>) -> Vec<ChatStreamEvent> {
        let Some(index) = object.get("index").and_then(Value::as_i64) else {
            return Vec::new();
        };
        let block = self.with_state(|state| state.open_blocks.remove(&index));
        let mut out = Vec::new();

        match block {
            Some(OpenBlockState::Text { id }) => {
                out.push(part(ChatStreamPart::TextEnd {
                    id,
                    provider_metadata: self.interaction_provider_metadata(),
                }));
            }
            Some(OpenBlockState::Reasoning { id, signature }) => {
                out.push(part(ChatStreamPart::ReasoningEnd {
                    id,
                    provider_metadata: self.part_provider_metadata(signature.as_deref()),
                }));
            }
            Some(OpenBlockState::Image {
                data,
                mime_type,
                uri,
                ..
            }) => {
                if let Some(event) =
                    self.image_event(data.as_deref(), mime_type.as_deref(), uri.as_deref())
                {
                    out.push(event);
                }
            }
            Some(OpenBlockState::FunctionCall {
                tool_call_id,
                tool_name,
                arguments_accum,
                signature,
                ..
            }) => {
                let input = if arguments_accum.is_empty() {
                    "{}".to_string()
                } else {
                    arguments_accum
                };
                out.push(part(ChatStreamPart::ToolInputEnd {
                    id: tool_call_id.clone(),
                    provider_metadata: None,
                }));
                out.push(part(ChatStreamPart::ToolCall(ChatStreamToolCall {
                    tool_call_id,
                    tool_name,
                    input,
                    provider_executed: None,
                    dynamic: None,
                    provider_metadata: self.part_provider_metadata(signature.as_deref()),
                })));
            }
            Some(OpenBlockState::BuiltinToolCall {
                tool_call_id,
                tool_name,
                arguments,
                call_emitted,
                ..
            }) if !call_emitted => {
                out.push(part(ChatStreamPart::ToolCall(ChatStreamToolCall {
                    tool_call_id,
                    tool_name,
                    input: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string()),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_metadata: None,
                })));
            }
            Some(OpenBlockState::BuiltinToolResult {
                block_type,
                call_id,
                tool_name,
                result,
                is_error,
                result_emitted,
                ..
            }) if !result_emitted => {
                out.push(part(ChatStreamPart::ToolResult(ChatStreamToolResult {
                    tool_call_id: call_id.clone(),
                    tool_name,
                    result: non_null_json(result.clone()),
                    is_error,
                    preliminary: None,
                    dynamic: None,
                    provider_metadata: None,
                })));
                out.extend(self.builtin_result_source_events(
                    &block_type,
                    &call_id,
                    result,
                    is_error,
                ));
            }
            _ => {}
        }

        out
    }

    fn handle_status_update(&self, object: &Map<String, Value>) {
        let status = string_field(object, "status")
            .map(ToOwned::to_owned)
            .or_else(|| match string_field(object, "event_type") {
                Some("interaction.requires_action") => Some("requires_action".to_string()),
                Some("interaction.in_progress") => Some("in_progress".to_string()),
                _ => None,
            });
        if let Some(status) = status {
            self.with_state(|state| {
                state.finish_status = Some(status);
            });
        }
    }

    fn handle_interaction_completed(&self, object: &Map<String, Value>) {
        let interaction = object.get("interaction").and_then(Value::as_object);
        self.with_state(|state| {
            if let Some(interaction) = interaction {
                if let Some(id) = string_field(interaction, "id").filter(|id| !id.is_empty()) {
                    state.interaction_id = Some(id.to_string());
                }
                if let Some(model) = string_field(interaction, "model")
                    .or_else(|| string_field(interaction, "agent"))
                    .filter(|id| !id.is_empty())
                {
                    state.model_id = Some(model.to_string());
                }
                if let Some(status) = string_field(interaction, "status") {
                    state.finish_status = Some(status.to_string());
                }
                if let Some(service_tier) =
                    string_field(interaction, "service_tier").filter(|id| !id.is_empty())
                {
                    state.service_tier = Some(service_tier.to_string());
                }
                if let Some(usage) = interaction.get("usage").and_then(convert_usage) {
                    state.usage = Some(usage);
                }
            }
        });
    }

    fn handle_error_event(&self, object: &Map<String, Value>, out: &mut Vec<ChatStreamEvent>) {
        let error_payload = object
            .get("error")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "message": "Unknown interaction error" }));
        out.push(part(ChatStreamPart::Error {
            error: error_payload.clone(),
        }));
        self.with_state(|state| {
            state.finish_status = Some("failed".to_string());
            state.error_payloads.push(error_payload);
        });
    }

    fn annotation_source_events(&self, annotations: &[Value]) -> Vec<ChatStreamEvent> {
        let sources =
            annotations_to_sources(Some(&annotations.to_vec()), &mut || self.generate_id());
        self.source_events(sources)
    }

    fn builtin_result_source_events(
        &self,
        block_type: &str,
        call_id: &str,
        result: Value,
        is_error: Option<bool>,
    ) -> Vec<ChatStreamEvent> {
        let mut step = Map::new();
        step.insert("type".to_string(), Value::String(block_type.to_string()));
        step.insert("call_id".to_string(), Value::String(call_id.to_string()));
        step.insert("result".to_string(), result);
        if let Some(is_error) = is_error {
            step.insert("is_error".to_string(), Value::Bool(is_error));
        }
        let sources = builtin_tool_result_to_sources(block_type, &step, &mut || self.generate_id());
        self.source_events(sources)
    }

    fn source_events(&self, sources: Vec<ContentPart>) -> Vec<ChatStreamEvent> {
        let mut out = Vec::new();
        for source in sources {
            let key = source_dedupe_key(&source);
            let should_emit = self.with_state(|state| {
                if state.emitted_source_keys.contains(&key) {
                    false
                } else {
                    state.emitted_source_keys.insert(key);
                    true
                }
            });
            if !should_emit {
                continue;
            }
            if let ContentPart::Source {
                id,
                source,
                provider_metadata,
            } = source
            {
                out.push(part(ChatStreamPart::Source {
                    id,
                    source,
                    provider_metadata,
                }));
            }
        }
        out
    }

    fn inline_image_delta_event(&self, delta: &Map<String, Value>) -> Option<ChatStreamEvent> {
        self.image_event(
            string_field(delta, "data"),
            string_field(delta, "mime_type"),
            string_field(delta, "uri"),
        )
    }

    fn image_event(
        &self,
        data: Option<&str>,
        mime_type: Option<&str>,
        uri: Option<&str>,
    ) -> Option<ChatStreamEvent> {
        let media_type = mime_type.unwrap_or("image/png").to_string();
        if let Some(data) = data.filter(|value| !value.is_empty()) {
            return Some(part(ChatStreamPart::File(ChatStreamFilePart {
                media_type,
                data: ChatStreamFileData::Base64(data.to_string()),
                provider_metadata: self.interaction_provider_metadata(),
            })));
        }
        if let Some(uri) = uri.filter(|value| !value.is_empty()) {
            return Some(part(ChatStreamPart::Source {
                id: self.generate_id(),
                source: SourcePart::Url {
                    url: uri.to_string(),
                    title: None,
                },
                provider_metadata: self.interaction_provider_metadata(),
            }));
        }
        None
    }

    fn block_id(&self, index: i64) -> String {
        let interaction_id = self.with_state(|state| state.interaction_id.clone());
        format!(
            "{}:{index}",
            interaction_id.unwrap_or_else(|| "interaction".to_string())
        )
    }

    fn interaction_provider_metadata(&self) -> Option<StreamProviderMetadata> {
        let interaction_id = self.with_state(|state| state.interaction_id.clone());
        part_provider_metadata(None, interaction_id.as_deref())
    }

    fn part_provider_metadata(&self, signature: Option<&str>) -> Option<StreamProviderMetadata> {
        let interaction_id = self.with_state(|state| state.interaction_id.clone());
        part_provider_metadata(signature, interaction_id.as_deref())
    }

    fn final_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        if !self.state.needs_stream_end() {
            return vec![];
        }

        let mut out = self.close_remaining_blocks();
        let snapshot = self.with_state(|state| FinalSnapshot {
            interaction_id: state.interaction_id.clone(),
            model_id: state.model_id.clone(),
            service_tier: state.service_tier.clone(),
            finish_status: state.finish_status.clone(),
            usage: state.usage.clone(),
            has_function_call: state.has_function_call,
            errors: state.error_payloads.clone(),
        });
        let raw_status = snapshot
            .finish_status
            .clone()
            .unwrap_or_else(|| "in_progress".to_string());
        let finish_reason = map_finish_reason(&raw_status, snapshot.has_function_call);
        let usage = snapshot
            .usage
            .clone()
            .unwrap_or_else(|| Usage::builder().build());
        let provider_metadata = response_provider_metadata(
            snapshot.interaction_id.as_deref(),
            snapshot.service_tier.as_deref(),
        );

        out.push(part(ChatStreamPart::Finish {
            usage: usage.clone(),
            finish_reason: ChatStreamFinishInfo {
                unified: finish_reason.clone(),
                raw: Some(raw_status.clone()),
            },
            provider_metadata: provider_metadata.clone(),
        }));

        if let Some(error) = snapshot.errors.last() {
            out.push(ChatStreamEvent::Error {
                error: error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| error.to_string()),
            });
        }

        let final_model_id = snapshot
            .model_id
            .clone()
            .or_else(|| Some(self.fallback_model_id.clone()));
        out.push(ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: snapshot.interaction_id,
                model: final_model_id.clone(),
                content: MessageContent::MultiModal(Vec::new()),
                usage: snapshot.usage,
                finish_reason: Some(finish_reason),
                raw_finish_reason: Some(raw_status),
                audio: None,
                system_fingerprint: None,
                warnings: if self.warnings.is_empty() {
                    None
                } else {
                    Some(self.warnings.clone())
                },
                service_tier: snapshot.service_tier,
                request: None,
                provider_metadata,
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: final_model_id,
                    headers: HashMap::new(),
                    body: None,
                }),
            },
        });

        ok_events(out)
    }

    fn close_remaining_blocks(&self) -> Vec<ChatStreamEvent> {
        let indexes =
            self.with_state(|state| state.open_blocks.keys().copied().collect::<Vec<_>>());
        let mut out = Vec::new();
        for index in indexes {
            let mut object = Map::new();
            object.insert("index".to_string(), Value::from(index));
            out.extend(self.handle_step_stop(&object));
        }
        out
    }
}

struct FinalSnapshot {
    interaction_id: Option<String>,
    model_id: Option<String>,
    service_tier: Option<String>,
    finish_status: Option<String>,
    usage: Option<Usage>,
    has_function_call: bool,
    errors: Vec<Value>,
}

impl SseEventConverter for GoogleInteractionsEventConverter {
    fn convert_event(&self, event: Event) -> SseEventFuture<'_> {
        Box::pin(async move {
            if event.data.trim().is_empty() {
                return vec![];
            }
            match serde_json::from_str::<Value>(&event.data) {
                Ok(raw) => self.convert_json_event(raw),
                Err(error) => {
                    self.with_state(|state| {
                        state.finish_status = Some("failed".to_string());
                    });
                    let mut out = self.start_events();
                    out.push(part(ChatStreamPart::Error {
                        error: serde_json::json!({
                            "message": format!("Failed to parse google.interactions SSE JSON: {error}"),
                            "raw": event.data,
                        }),
                    }));
                    ok_events(out)
                }
            }
        })
    }

    fn is_stream_end_event(&self, event: &Event) -> bool {
        event.data.trim() == "[DONE]"
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        self.final_events()
    }

    fn finalize_on_disconnect(&self) -> bool {
        self.with_state(|state| {
            state.finish_status.as_deref().is_some_and(|status| {
                matches!(
                    status,
                    "completed" | "requires_action" | "failed" | "cancelled" | "incomplete"
                )
            })
        })
    }
}

fn ok_events(events: Vec<ChatStreamEvent>) -> Vec<Result<ChatStreamEvent, LlmError>> {
    events.into_iter().map(Ok).collect()
}

fn part(part: ChatStreamPart) -> ChatStreamEvent {
    ChatStreamEvent::Part { part }
}

fn parse_timestamp(value: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.with_timezone(&chrono::Utc))
}

fn builtin_tool_call_name(step_type: &str, step: &Map<String, Value>) -> String {
    if step_type == "mcp_server_tool_call" {
        string_field(step, "name")
            .unwrap_or("mcp_server_tool")
            .to_string()
    } else {
        step_type
            .strip_suffix("_call")
            .unwrap_or(step_type)
            .to_string()
    }
}

fn builtin_tool_result_name(step_type: &str, step: &Map<String, Value>) -> String {
    if step_type == "mcp_server_tool_result" {
        string_field(step, "name")
            .unwrap_or("mcp_server_tool")
            .to_string()
    } else {
        step_type
            .strip_suffix("_result")
            .unwrap_or(step_type)
            .to_string()
    }
}

fn non_null_json(value: Value) -> Value {
    if value.is_null() {
        serde_json::json!({})
    } else {
        value
    }
}
