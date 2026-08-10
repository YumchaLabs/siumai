use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde_json::Value;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageCallError, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamDecoder, LanguageStreamEvent, Model,
    ModelDescriptor, ModelFamily, ModelId, ModelOperation, ProviderOptionError, ProviderScope,
    StreamTerminal,
};
use siumai_protocol_openai::chat_completions::{
    CHAT_COMPLETIONS_TARGET, ChatCompletionsDialect, ChatCompletionsStreamDecoder,
    ChatRequestEncodingOptions, MaxOutputTokensField, decode_response as decode_chat_response,
    encode_request_with_options_and_resolver as encode_chat_request_with_options,
};
use siumai_protocol_openai::responses::{
    RequestEncodingOptions, ResponsesStreamDecoder, decode_response as decode_responses_response,
    encode_request_with_options_and_resolver as encode_request_with_options,
};
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    RequestBody, RequestHeaders, RequestPlan, RequestTarget, TransportByteStream, TransportLimits,
    TransportResponse, TransportStreamResponse,
};

use super::annotations::{OpenAiAnnotationResolver, validate_prompt_cache_annotations};
use super::http_error;
use super::mode::OpenAiApiMode;
use super::provider::{OpenAiMergedOptions, OpenAiRuntime};
use super::responses_native::{
    OpenAiResponsesResponse, OpenAiResponsesStream, OpenAiResponsesStreamFrame,
};
use super::responses_resource::OpenAiBackgroundResponse;
#[cfg(feature = "openai-responses-websocket")]
use super::responses_websocket::{
    OpenAiResponsesWebSocketConfig, OpenAiResponsesWebSocketConfigError,
};

const RESPONSES_TARGET: &str = "responses";

#[cfg(feature = "openai-responses-websocket")]
pub(crate) struct PreparedOpenAiResponsesWebSocketCall {
    pub(crate) body: Value,
}

#[derive(Debug, Clone, Copy)]
enum OpenAiResponsesBodyMode {
    Http {
        stream: bool,
        background: bool,
    },
    #[cfg(feature = "openai-responses-websocket")]
    WebSocket {
        generate: bool,
    },
}

/// Lightweight model handle for the recommended OpenAI Responses API.
#[derive(Clone)]
pub struct OpenAiResponsesModel {
    pub(crate) runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
}

impl OpenAiResponsesModel {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope_arc(OpenAiApiMode::Responses),
            model,
            ModelFamily::Language,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    pub const fn api_mode(&self) -> OpenAiApiMode {
        OpenAiApiMode::Responses
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        stream: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        self.plan_internal(request, stream, false, merged)
    }

    fn background_plan(
        &self,
        request: &LanguageRequest,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        self.plan_internal(request, false, true, merged)
    }

    fn plan_internal(
        &self,
        request: &LanguageRequest,
        stream: bool,
        background: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        let body = self.encode_body(
            self.runtime.scope(OpenAiApiMode::Responses),
            request,
            OpenAiResponsesBodyMode::Http { stream, background },
            merged,
        )?;
        request_plan(
            RESPONSES_TARGET,
            body,
            stream,
            self.runtime.replay_safety.clone(),
            OpenAiApiMode::Responses,
        )
    }

    fn encode_body(
        &self,
        scope: &ProviderScope,
        request: &LanguageRequest,
        mode: OpenAiResponsesBodyMode,
        merged: OpenAiMergedOptions,
    ) -> Result<Value, Error> {
        let resolver = OpenAiAnnotationResolver;
        let mut encoding = match mode {
            OpenAiResponsesBodyMode::Http { stream, .. } => RequestEncodingOptions::new(stream),
            #[cfg(feature = "openai-responses-websocket")]
            OpenAiResponsesBodyMode::WebSocket { .. } => RequestEncodingOptions::websocket(),
        }
        .with_extra(merged.wire);
        for tool in merged.native_tools {
            encoding = encoding.with_native_tool(tool);
        }
        for (name, options) in merged.function_tools {
            encoding = encoding.with_function_tool_options(name, options);
        }
        let mut body =
            encode_request_with_options(scope, self.model_id(), request, &encoding, &resolver)?;
        let object = body.as_object_mut().ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses encoder produced a non-object request",
            )
        })?;
        match mode {
            OpenAiResponsesBodyMode::Http {
                background: true, ..
            } => {
                object.insert("background".to_string(), Value::Bool(true));
            }
            OpenAiResponsesBodyMode::Http { .. } => {}
            #[cfg(feature = "openai-responses-websocket")]
            OpenAiResponsesBodyMode::WebSocket { generate } => {
                object.insert(
                    "type".to_string(),
                    Value::String("response.create".to_string()),
                );
                if !generate {
                    object.insert("generate".to_string(), Value::Bool(false));
                }
            }
        }
        Ok(body)
    }

    #[cfg(feature = "openai-responses-websocket")]
    /// Configure one persistent provider-owned Responses WebSocket session.
    pub fn websocket(
        &self,
    ) -> Result<OpenAiResponsesWebSocketConfig, OpenAiResponsesWebSocketConfigError> {
        OpenAiResponsesWebSocketConfig::from_model(self.clone())
    }

    #[cfg(feature = "openai-responses-websocket")]
    pub(crate) fn prepare_websocket_call(
        &self,
        scope: &ProviderScope,
        request: LanguageRequest,
        options: &CallOptions,
        generate: bool,
    ) -> Result<PreparedOpenAiResponsesWebSocketCall, Error> {
        let operation = ModelOperation::Stream;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::Responses, options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let request = normalize_request(OpenAiApiMode::Responses, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let body = self
            .encode_body(
                scope,
                &request,
                OpenAiResponsesBodyMode::WebSocket { generate },
                merged,
            )
            .map_err(|error| self.contextualize(operation, error))?;
        Ok(PreparedOpenAiResponsesWebSocketCall { body })
    }

    pub(crate) fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        contextualize(self, operation, error)
    }

    /// Create a provider-native background response without forcing a terminal projection.
    pub async fn create_background(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiBackgroundResponse, Error> {
        let operation = ModelOperation::Generate;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let request = normalize_request(OpenAiApiMode::Responses, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .background_plan(&request, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(OpenAiApiMode::Responses, response),
            ));
        }
        let resource = siumai_protocol_openai::responses::decode_response_resource(response.body())
            .map_err(|error| self.contextualize(operation, error))?;
        Ok(OpenAiBackgroundResponse::new(resource))
    }

    /// Generate one Responses result while retaining the provider-native resource.
    pub async fn generate_native(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesResponse, Error> {
        let operation = ModelOperation::Generate;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let request = normalize_request(OpenAiApiMode::Responses, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, false, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(OpenAiApiMode::Responses, response),
            ));
        }
        let decoded = decode_responses_response(
            response.body(),
            self.runtime.scope(OpenAiApiMode::Responses),
            self.model_id(),
        )
        .map_err(|error| self.contextualize(operation, error))?;
        let (native, portable) = decoded.into_parts();
        let portable = portable.map_err(|error| contextualize_call_error(self, operation, error));
        Ok(OpenAiResponsesResponse::new(native, portable))
    }

    /// Establish one native Responses stream.
    ///
    /// Converting the returned carrier with [`OpenAiResponsesStream::into_portable`] projects the
    /// same HTTP response and decoder state; it never issues a second provider request.
    pub async fn stream_native(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesStream, Error> {
        let operation = ModelOperation::Stream;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::Responses, &options)
            .map_err(|source| {
                self.contextualize(operation, option_error(OpenAiApiMode::Responses, source))
            })?;
        let request = normalize_request(OpenAiApiMode::Responses, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, true, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(OpenAiApiMode::Responses, response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        let diagnostics = headers.diagnostics().with_status(status.as_u16());
        let context = model_error_context(self, operation);
        let decoder = ResponsesStreamDecoder::new(
            self.runtime.scope(OpenAiApiMode::Responses).clone(),
            self.model_id().clone(),
        )
        .with_wire_dialect(self.runtime.responses_wire_dialect)
        .with_response_diagnostics(diagnostics);

        Ok(decode_responses_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            decoder,
            context,
        ))
    }
}

impl std::fmt::Debug for OpenAiResponsesModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiResponsesModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for OpenAiResponsesModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.generate_native(request, options)
            .await?
            .into_portable()
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        Ok(self.stream_native(request, options).await?.into_portable())
    }
}

/// Lightweight model handle for explicit OpenAI Chat Completions compatibility.
#[derive(Clone)]
pub struct OpenAiChatCompletionsModel {
    pub(crate) runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
}

impl OpenAiChatCompletionsModel {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.scope_arc(OpenAiApiMode::ChatCompletions),
            model,
            ModelFamily::Language,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    pub const fn api_mode(&self) -> OpenAiApiMode {
        OpenAiApiMode::ChatCompletions
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        stream: bool,
        merged: OpenAiMergedOptions,
    ) -> Result<RequestPlan, Error> {
        let resolver = OpenAiAnnotationResolver;
        let encoding = ChatRequestEncodingOptions::new(stream).with_extra(merged.wire);
        let body = encode_chat_request_with_options(
            self.runtime.scope(OpenAiApiMode::ChatCompletions),
            self.model_id(),
            request,
            &official_chat_dialect(),
            &encoding,
            &resolver,
        )?;
        request_plan(
            CHAT_COMPLETIONS_TARGET,
            body,
            stream,
            self.runtime.replay_safety.clone(),
            OpenAiApiMode::ChatCompletions,
        )
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        contextualize(self, operation, error)
    }
}

impl std::fmt::Debug for OpenAiChatCompletionsModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiChatCompletionsModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiChatCompletionsModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatCompletionsModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        let operation = ModelOperation::Generate;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::ChatCompletions, &options)
            .map_err(|source| {
                self.contextualize(
                    operation,
                    option_error(OpenAiApiMode::ChatCompletions, source),
                )
            })?;
        let request = normalize_request(OpenAiApiMode::ChatCompletions, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, false, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self
                .contextualize(
                    operation,
                    response_error(OpenAiApiMode::ChatCompletions, response),
                )
                .into());
        }
        let response = decode_chat_response(
            self.runtime.scope(OpenAiApiMode::ChatCompletions),
            self.model_id(),
            response.body(),
            &official_chat_dialect(),
        )
        .map_err(|error| self.contextualize(operation, error))?;
        Ok(response)
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let merged = self
            .runtime
            .merge_options_for(self, OpenAiApiMode::ChatCompletions, &options)
            .map_err(|source| {
                self.contextualize(
                    operation,
                    option_error(OpenAiApiMode::ChatCompletions, source),
                )
            })?;
        let request = normalize_request(OpenAiApiMode::ChatCompletions, request, &merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, true, merged)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(OpenAiApiMode::ChatCompletions, response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        let diagnostics = headers.diagnostics().with_status(status.as_u16());

        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            ChatCompletionsStreamDecoder::new(
                self.runtime.scope(OpenAiApiMode::ChatCompletions).clone(),
                self.model_id().clone(),
                official_chat_dialect(),
            )
            .with_response_diagnostics(diagnostics),
            OpenAiApiMode::ChatCompletions,
            model_error_context(self, operation),
        ))
    }
}

fn official_chat_dialect() -> ChatCompletionsDialect {
    ChatCompletionsDialect::generic()
        .with_developer_role(true)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens)
        .with_stream_usage(true)
}

fn normalize_request(
    mode: OpenAiApiMode,
    request: LanguageRequest,
    merged: &OpenAiMergedOptions,
) -> Result<LanguageRequest, Error> {
    if mode == OpenAiApiMode::Responses {
        if request.generation.seed.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI Responses cannot encode the neutral seed control",
            ));
        }
        if !request.generation.stop_sequences.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI Responses cannot encode neutral stop sequences",
            ));
        }
    }

    let cache_mode = prompt_cache_mode(&merged.wire);
    validate_prompt_cache_annotations(&request, cache_mode).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "OpenAI prompt-cache annotations are invalid",
        )
        .with_source(source)
    })?;
    Ok(request)
}

fn prompt_cache_mode(
    wire: &std::collections::BTreeMap<String, Value>,
) -> super::options::OpenAiPromptCacheMode {
    if wire
        .get("prompt_cache_options")
        .and_then(Value::as_object)
        .and_then(|options| options.get("mode"))
        .and_then(Value::as_str)
        == Some("explicit")
    {
        super::options::OpenAiPromptCacheMode::Explicit
    } else {
        super::options::OpenAiPromptCacheMode::Implicit
    }
}

fn request_plan(
    target: &'static str,
    body: Value,
    stream: bool,
    replay_safety: siumai_transport::ReplaySafety,
    mode: OpenAiApiMode,
) -> Result<RequestPlan, Error> {
    let accept = if stream {
        HeaderValue::from_static("text/event-stream")
    } else {
        HeaderValue::from_static("application/json")
    };
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, accept)
        .map_err(|source| request_build_error(mode, source))?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(target).map_err(|source| request_build_error(mode, source))?,
    )
    .with_headers(headers)
    .with_body(RequestBody::json(&body).map_err(|source| request_build_error(mode, source))?)
    .with_replay_safety(replay_safety)
    .map_err(|source| request_build_error(mode, source))
}

fn contextualize(model: &impl Model, operation: ModelOperation, error: Error) -> Error {
    error.with_context(model_error_context(model, operation))
}

fn contextualize_call_error(
    model: &impl Model,
    operation: ModelOperation,
    error: LanguageCallError,
) -> LanguageCallError {
    let (error, partial) = error.into_parts();
    LanguageCallError::new(contextualize(model, operation, error), partial)
}

pub(crate) fn model_error_context(model: &impl Model, operation: ModelOperation) -> ErrorContext {
    ErrorContext {
        operation: Some(operation),
        provider: Some(model.provider_id().clone()),
        route: None,
        model: Some(model.model_id().clone()),
    }
}

fn decode_responses_sse_stream(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: TransportLimits,
    mut protocol: ResponsesStreamDecoder,
    context: ErrorContext,
) -> OpenAiResponsesStream {
    let cancellation_error =
        Error::cancelled("OpenAI Responses stream was cancelled").with_context(context.clone());
    let source = async_stream::try_stream! {
        let mut body = body;
        let mut framing = SseDecoder::new(&limits);
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|error| error.with_context(context.clone()))?;
            let frames = framing
                .push(&chunk)
                .map_err(|source| {
                    sse_error(OpenAiApiMode::Responses, source).with_context(context.clone())
                })?;
            let mut pending_frames = Vec::with_capacity(frames.len());
            let mut terminal_in_batch = false;
            for frame in frames {
                if terminal_in_batch && frame.data().trim() == "[DONE]" {
                    continue;
                }
                let decoded = protocol
                    .decode_native(frame.data())
                    .map_err(|error| error.with_context(context.clone()))?;
                let (native, mut portable_events, replay_status) = decoded.into_parts();
                let terminal_position = portable_events
                    .iter()
                    .position(|event| event.terminal().is_some());
                if terminal_position.is_some_and(|index| index + 1 != portable_events.len()) {
                    Err(Error::new(
                        ErrorKind::Protocol,
                        "OpenAI Responses decoder emitted events after terminal settlement",
                    )
                    .with_context(context.clone()))?;
                }
                for event in &mut portable_events {
                    contextualize_terminal_error(event, &context);
                }
                let canonical_terminal_response = portable_events
                    .iter()
                    .any(|event| event.terminal().is_some())
                    .then(|| protocol.terminal_response().cloned())
                    .flatten();
                let frame = OpenAiResponsesStreamFrame::new(
                    native,
                    portable_events,
                    canonical_terminal_response,
                    replay_status,
                );
                terminal_in_batch |= frame.is_terminal();
                pending_frames.push(frame);
            }
            for frame in pending_frames {
                yield frame;
            }
            if terminal_in_batch {
                return;
            }
        }
        framing
            .finish()
            .map_err(|source| {
                sse_error(OpenAiApiMode::Responses, source).with_context(context.clone())
            })?;
        let trailing = protocol
            .finish()
            .map_err(|error| error.with_context(context.clone()))?;
        if !trailing.is_empty() {
            Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses decoder produced portable events without a native frame",
            )
            .with_context(context.clone()))?;
        }
    };
    OpenAiResponsesStream::established(cancellation, cancellation_error, source)
}

fn decode_sse_stream<D>(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: TransportLimits,
    mut protocol: D,
    mode: OpenAiApiMode,
    context: ErrorContext,
) -> LanguageStream
where
    D: LanguageStreamDecoder<ProtocolFrame = str> + Send + 'static,
{
    established_stream(cancellation, move |_| {
        async_stream::try_stream! {
            let mut body = body;
            let mut framing = SseDecoder::new(&limits);
            while let Some(chunk) = body.next().await {
                let chunk = chunk.map_err(|error| error.with_context(context.clone()))?;
                let frames = framing
                    .push(&chunk)
                    .map_err(|source| sse_error(mode, source).with_context(context.clone()))?;
                let mut pending_events = Vec::new();
                let mut terminal_in_batch = false;
                for frame in frames {
                    if terminal_in_batch && frame.data().trim() == "[DONE]" {
                        continue;
                    }
                    let events = protocol
                        .decode(frame.data())
                        .map_err(|error| error.with_context(context.clone()))?;
                    let terminal_position = events
                        .iter()
                        .position(|event| event.terminal().is_some());
                    if terminal_position.is_some_and(|index| index + 1 != events.len()) {
                        Err(Error::new(
                            ErrorKind::Protocol,
                            "OpenAI stream decoder emitted events after terminal settlement",
                        )
                        .with_context(context.clone()))?;
                    }
                    terminal_in_batch |= terminal_position.is_some();
                    for mut event in events {
                        contextualize_terminal_error(&mut event, &context);
                        pending_events.push(event);
                    }
                }
                for event in pending_events {
                    yield event;
                }
                if terminal_in_batch {
                    return;
                }
            }
            framing
                .finish()
                .map_err(|source| sse_error(mode, source).with_context(context.clone()))?;
            let events = protocol
                .finish()
                .map_err(|error| error.with_context(context.clone()))?;
            for mut event in events {
                contextualize_terminal_error(&mut event, &context);
                let terminal = event.terminal().is_some();
                yield event;
                if terminal {
                    return;
                }
            }
        }
    })
}

pub(crate) fn contextualize_terminal_error(
    event: &mut LanguageStreamEvent,
    context: &ErrorContext,
) {
    let LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. }) = event else {
        return;
    };
    let original = std::mem::replace(
        error,
        Error::new(
            ErrorKind::Internal,
            "stream error context replacement failed",
        ),
    );
    *error = original.with_context(context.clone());
}

fn option_error(mode: OpenAiApiMode, source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        match mode {
            OpenAiApiMode::Responses => "provider options are invalid for the OpenAI Responses API",
            OpenAiApiMode::ChatCompletions => {
                "provider options are invalid for OpenAI Chat Completions"
            }
        },
    )
    .with_source(source)
}

pub(crate) fn request_build_error(
    mode: OpenAiApiMode,
    source: siumai_transport::RequestBuildError,
) -> Error {
    http_error::request_build_error(
        match mode {
            OpenAiApiMode::Responses => "OpenAI Responses request violates the transport contract",
            OpenAiApiMode::ChatCompletions => {
                "OpenAI Chat Completions request violates the transport contract"
            }
        },
        source,
    )
}

fn sse_error(mode: OpenAiApiMode, source: SseFrameError) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        SseFrameError::InvalidUtf8 | SseFrameError::UnexpectedEof => ErrorKind::Protocol,
        _ => ErrorKind::Protocol,
    };
    Error::new(
        kind,
        match mode {
            OpenAiApiMode::Responses => "provider returned an invalid OpenAI Responses SSE stream",
            OpenAiApiMode::ChatCompletions => {
                "provider returned an invalid OpenAI Chat Completions SSE stream"
            }
        },
    )
    .with_source(source)
}

pub(crate) fn response_error(mode: OpenAiApiMode, response: TransportResponse) -> Error {
    http_error::response_error(
        match mode {
            OpenAiApiMode::Responses => "OpenAI rejected the Responses request",
            OpenAiApiMode::ChatCompletions => "OpenAI rejected the Chat Completions request",
        },
        response,
    )
}

async fn stream_response_error(mode: OpenAiApiMode, response: TransportStreamResponse) -> Error {
    http_error::stream_response_error(
        match mode {
            OpenAiApiMode::Responses => "OpenAI rejected the Responses request",
            OpenAiApiMode::ChatCompletions => "OpenAI rejected the Chat Completions request",
        },
        response,
    )
    .await
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        ContentPart, Message, MessagePart, MessageRole, ReplayDomain, ReplayDomainId, ToolSpec,
    };
    use siumai_protocol_openai::responses::ResponsesWireDialect;
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::configured::{
        GPT_5_5, GPT_5_6_SOL, OpenAiChatCompletionsOptions, OpenAiContentOptions, OpenAiCredential,
        OpenAiFunctionToolOptions, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
        OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, OpenAiProvider, OpenAiReasoning,
        OpenAiReasoningEffort, OpenAiResponsesOptions, OpenAiResponsesTool, OpenAiTextVerbosity,
    };

    fn provider() -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-model-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    fn request_with_cache_marker(annotation: OpenAiContentOptions) -> LanguageRequest {
        LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [MessagePart::text("hello")
                .with_provider_annotation(&annotation)
                .expect("OpenAI annotation")],
        )])
    }

    async fn provider_for(server: &MockServer) -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-model-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    fn body_json(plan: &RequestPlan) -> Value {
        match plan.body() {
            RequestBody::Bytes { data, .. } => serde_json::from_slice(data).unwrap(),
            _ => panic!("expected a JSON request body"),
        }
    }

    #[test]
    fn neutral_prompt_uses_distinct_native_routes_and_wire_shapes() {
        let provider = provider();
        let responses = provider.responses(GPT_5_6_SOL).unwrap();
        let chat = provider.chat_completions(GPT_5_6_SOL).unwrap();
        let responses_options = responses
            .runtime
            .merge_options_for(
                &responses,
                OpenAiApiMode::Responses,
                &CallOptions::default(),
            )
            .unwrap();
        let chat_options = chat
            .runtime
            .merge_options_for(
                &chat,
                OpenAiApiMode::ChatCompletions,
                &CallOptions::default(),
            )
            .unwrap();
        let responses_plan = responses
            .plan(&request(), false, responses_options)
            .unwrap();
        let chat_plan = chat.plan(&request(), false, chat_options).unwrap();
        let responses_body = body_json(&responses_plan);
        let chat_body = body_json(&chat_plan);

        assert_eq!(responses_plan.target().as_str(), "responses");
        assert_eq!(chat_plan.target().as_str(), "chat/completions");
        assert!(responses_body.get("input").is_some());
        assert!(responses_body.get("messages").is_none());
        assert!(chat_body.get("messages").is_some());
        assert!(chat_body.get("input").is_none());
    }

    #[test]
    fn branded_openai_responses_always_use_the_openai_baseline() {
        let caller_selected = provider().responses(GPT_5_6_SOL).unwrap();
        assert_eq!(
            caller_selected.runtime.responses_wire_dialect,
            ResponsesWireDialect::openai()
        );

        let official = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .build()
            .unwrap()
            .responses(GPT_5_6_SOL)
            .unwrap();
        assert_eq!(
            official.runtime.responses_wire_dialect,
            ResponsesWireDialect::openai()
        );

        let custom = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("dialect-fixture").unwrap(),
            ))
            .build()
            .unwrap()
            .responses(GPT_5_6_SOL)
            .unwrap();
        assert_eq!(
            custom.runtime.responses_wire_dialect,
            ResponsesWireDialect::openai()
        );
    }

    #[test]
    fn typed_responses_options_shape_cache_annotations_and_native_tools() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let typed = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: Some(OpenAiPromptCacheMode::Explicit),
                ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
            }),
            top_logprobs: Some(5),
            reasoning: Some(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::None)),
            text_verbosity: Some(OpenAiTextVerbosity::High),
            tools: vec![
                OpenAiResponsesTool::web_search(),
                OpenAiResponsesTool::programmatic_tool_calling(),
            ],
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options_for(&model, &typed)
            .unwrap();
        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model
            .plan(
                &request_with_cache_marker(OpenAiContentOptions::cache_write_candidate()),
                false,
                merged,
            )
            .unwrap();
        let body = body_json(&plan);

        assert_eq!(body["text"]["verbosity"], "high");
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(body["top_logprobs"], 5);
        assert!(
            body["include"]
                .as_array()
                .unwrap()
                .contains(&json!("message.output_text.logprobs"))
        );
        assert_eq!(
            body["input"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert_eq!(body["tools"][1]["type"], "programmatic_tool_calling");
        assert!(body.get("native_tools").is_none());
        assert!(body.get("prompt_cache_breakpoints").is_none());
    }

    #[test]
    fn exact_target_options_reach_wire_and_do_not_cross_provider_instances() {
        let configured = provider();
        let model = configured.responses("private-reasoning-model").unwrap();
        let typed = OpenAiResponsesOptions::default()
            .with_reasoning(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::Max));
        let call_options = CallOptions::default()
            .with_provider_options_for(&model, &typed)
            .unwrap();

        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        assert_eq!(merged.wire["reasoning"]["effort"], "max");

        let other = provider().responses("private-reasoning-model").unwrap();
        let error =
            match other
                .runtime
                .merge_options_for(&other, OpenAiApiMode::Responses, &call_options)
            {
                Ok(_) => panic!("options bound to another provider instance unexpectedly matched"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            ProviderOptionError::ExactTargetMismatch { .. }
        ));
    }

    #[test]
    fn typed_chat_options_shape_explicit_cache_annotation() {
        let provider = provider();
        let model = provider.chat_completions(GPT_5_6_SOL).unwrap();
        let typed = OpenAiChatCompletionsOptions::default()
            .with_prompt_cache(OpenAiPromptCacheOptions::explicit_30_minutes());
        let call_options = CallOptions::default()
            .with_provider_options_for(&model, &typed)
            .unwrap();
        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::ChatCompletions, &call_options)
            .unwrap();
        let plan = model
            .plan(
                &request_with_cache_marker(OpenAiContentOptions::cache_write_candidate()),
                false,
                merged,
            )
            .unwrap();
        let body = body_json(&plan);

        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
        assert!(body.get("prompt_cache_breakpoints").is_none());
    }

    #[test]
    fn checked_raw_options_forward_future_values_but_not_canonical_fields() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let call_options = CallOptions::default()
            .with_raw_provider_options_for(
                &model,
                json!({
                    "future_feature": {"mode": "next"},
                    "service_tier": "future-priority",
                    "reasoning": {"effort": "future-max", "summary": "future-brief"},
                    "include": ["future.output.metadata"]
                }),
            )
            .unwrap();
        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model.plan(&request(), false, merged).unwrap();
        let body = body_json(&plan);
        assert_eq!(body["future_feature"], json!({"mode": "next"}));
        assert_eq!(body["service_tier"], "future-priority");
        assert_eq!(body["reasoning"]["effort"], "future-max");
        assert_eq!(body["reasoning"]["summary"], "future-brief");
        assert_eq!(body["include"], json!(["future.output.metadata"]));

        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"model": "gpt-5.6-luna"}))
            .unwrap();
        let result = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &options);
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "model")
        );

        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"background": true}))
            .unwrap();
        let result = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &options);
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "background")
        );

        let options = CallOptions::default()
            .with_raw_provider_options_for(
                &model,
                json!({"prompt_cache_options": {"mode": "explicit"}}),
            )
            .unwrap();
        let result = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &options);
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "prompt_cache_options")
        );

        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"service_tier": {"future": true}}))
            .unwrap();
        let result = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &options);
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "service_tier")
        );

        let chat = provider.chat_completions("future-chat-model").unwrap();
        let options = CallOptions::default()
            .with_raw_provider_options_for(&chat, json!({"top_logprobs": 5}))
            .unwrap();
        let result =
            chat.runtime
                .merge_options_for(&chat, OpenAiApiMode::ChatCompletions, &options);
        assert!(
            matches!(result, Err(ProviderOptionError::Rejected { path, .. }) if path == "top_logprobs")
        );
    }

    #[test]
    fn explicit_max_reasoning_survives_future_model_ids_without_implicit_summary() {
        let provider = provider();
        let responses = provider.responses("private-reasoning-model").unwrap();
        let responses_options = OpenAiResponsesOptions::default()
            .with_reasoning(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::Max));
        let responses_call = CallOptions::default()
            .with_provider_options_for(&responses, &responses_options)
            .unwrap();
        let merged = responses
            .runtime
            .merge_options_for(&responses, OpenAiApiMode::Responses, &responses_call)
            .unwrap();
        let normalized = normalize_request(OpenAiApiMode::Responses, request(), &merged).unwrap();
        let responses_body = body_json(&responses.plan(&normalized, false, merged).unwrap());
        assert_eq!(responses_body["reasoning"]["effort"], "max");
        assert!(responses_body["reasoning"].get("summary").is_none());

        let chat = provider
            .chat_completions("private-reasoning-model")
            .unwrap();
        let chat_options = OpenAiChatCompletionsOptions {
            reasoning_effort: Some(OpenAiReasoningEffort::Max),
            ..OpenAiChatCompletionsOptions::default()
        };
        let chat_call = CallOptions::default()
            .with_provider_options_for(&chat, &chat_options)
            .unwrap();
        let merged = chat
            .runtime
            .merge_options_for(&chat, OpenAiApiMode::ChatCompletions, &chat_call)
            .unwrap();
        let normalized =
            normalize_request(OpenAiApiMode::ChatCompletions, request(), &merged).unwrap();
        let chat_body = body_json(&chat.plan(&normalized, false, merged).unwrap());
        assert_eq!(chat_body["reasoning_effort"], "max");
    }

    #[test]
    fn prompt_cache_controls_are_not_filtered_by_model_identity() {
        let provider = provider();

        let responses_5_6 = provider.responses(GPT_5_6_SOL).unwrap();
        let formerly_rejected_5_6 = OpenAiResponsesOptions {
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::InMemory),
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options_for(&responses_5_6, &formerly_rejected_5_6)
            .unwrap();
        let merged = responses_5_6
            .runtime
            .merge_options_for(&responses_5_6, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let normalized = normalize_request(OpenAiApiMode::Responses, request(), &merged).unwrap();
        let body = body_json(&responses_5_6.plan(&normalized, false, merged).unwrap());
        assert_eq!(body["prompt_cache_retention"], "in_memory");

        let responses_5_5 = provider.responses(GPT_5_5).unwrap();
        let formerly_rejected_5_5 = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit_30_minutes()),
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options_for(&responses_5_5, &formerly_rejected_5_5)
            .unwrap();
        let merged = responses_5_5
            .runtime
            .merge_options_for(&responses_5_5, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let normalized = normalize_request(
            OpenAiApiMode::Responses,
            request_with_cache_marker(OpenAiContentOptions::cache_write_candidate()),
            &merged,
        )
        .unwrap();
        let body = body_json(&responses_5_5.plan(&normalized, false, merged).unwrap());
        assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(
            body["input"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
    }

    #[test]
    fn programmatic_tool_calling_can_invoke_typed_function_tools() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let mut request = request();
        request.tools.push(
            ToolSpec::new(
                "get_inventory",
                Some("Read inventory".to_string()),
                json!({"type": "object"}),
            )
            .unwrap(),
        );
        let typed = OpenAiResponsesOptions::default()
            .with_tool(OpenAiResponsesTool::programmatic_tool_calling())
            .with_function_tool_options(
                "get_inventory",
                OpenAiFunctionToolOptions::programmatic()
                    .with_strict(true)
                    .with_output_schema(json!({"type": "object"})),
            );
        let call_options = CallOptions::default()
            .with_provider_options_for(&model, &typed)
            .unwrap();
        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &call_options)
            .unwrap();
        let plan = model.plan(&request, false, merged).unwrap();
        let body = body_json(&plan);

        assert_eq!(body["tools"][0]["name"], "get_inventory");
        assert_eq!(body["tools"][0]["allowed_callers"], json!(["programmatic"]));
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(body["tools"][0]["output_schema"], json!({"type": "object"}));
        assert_eq!(body["tools"][1]["type"], "programmatic_tool_calling");
        assert!(body.get("function_tool_options").is_none());
    }

    #[test]
    fn mode_descriptors_use_protocol_specific_provenance() {
        let provider = provider();
        let responses = provider.responses(GPT_5_6_SOL).unwrap();
        let chat = provider.chat_completions(GPT_5_6_SOL).unwrap();

        assert_eq!(responses.descriptor().protocol(), Some("openai.responses"));
        assert_eq!(chat.descriptor().protocol(), Some("openai"));
    }

    #[tokio::test]
    async fn background_creation_preserves_non_terminal_native_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_background",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "queued",
                "output": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let model = provider_for(&server).await.responses(GPT_5_6_SOL).unwrap();
        let background = model
            .create_background(request(), CallOptions::default())
            .await
            .unwrap();
        assert_eq!(background.resource().status.as_str(), "queued");

        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["background"], true);
        assert_eq!(body["model"], GPT_5_6_SOL);
    }

    #[tokio::test]
    async fn native_generation_pairs_lossless_and_portable_views_from_one_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_native",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_native",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "hello from native",
                            "annotations": []
                        }]
                    },
                    {
                        "id": "future_native",
                        "type": "future_provider_tool_call",
                        "status": "completed",
                        "private_payload": "native-secret"
                    }
                ],
                "usage": null,
                "error": null,
                "incomplete_details": null,
                "reasoning": null
            })))
            .expect(1)
            .mount(&server)
            .await;

        let model = provider_for(&server).await.responses(GPT_5_6_SOL).unwrap();
        let response = model
            .generate_native(request(), CallOptions::default())
            .await
            .unwrap();
        assert_eq!(response.native().output.len(), 2);
        assert!(
            response.portable().unwrap().content().iter().any(
                |part| matches!(part, ContentPart::Text { text } if text == "hello from native")
            )
        );
        assert!(!format!("{response:?}").contains("native-secret"));
    }

    #[tokio::test]
    async fn native_generation_preserves_failed_resource_and_portable_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_failed_native",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "failed",
                "output": [
                    {
                        "id": "msg_partial",
                        "type": "message",
                        "role": "assistant",
                        "status": "incomplete",
                        "content": [{
                            "type": "output_text",
                            "text": "bounded partial",
                            "annotations": []
                        }]
                    },
                    {
                        "id": "call_unsafe",
                        "type": "function_call",
                        "status": "incomplete",
                        "call_id": "call_unsafe",
                        "name": "must_not_execute",
                        "arguments": "{\"unsafe\":true}"
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                "error": {"code": "server_error", "message": "provider detail"},
                "incomplete_details": null,
                "reasoning": null
            })))
            .expect(1)
            .mount(&server)
            .await;

        let model = provider_for(&server).await.responses(GPT_5_6_SOL).unwrap();
        let response = model
            .generate_native(request(), CallOptions::default())
            .await
            .unwrap();
        assert_eq!(response.native().status.as_str(), "failed");
        assert_eq!(response.native().output.len(), 2);
        let error = response.portable().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unavailable);
        assert_eq!(error.context().operation, Some(ModelOperation::Generate));
        let partial = error.partial().expect("bounded portable partial output");
        assert!(matches!(
            partial.content(),
            [siumai_core::PartialLanguageOutputPart::Text { text }] if text == "bounded partial"
        ));
        assert_eq!(partial.usage().input_tokens.value(), Some(7));
    }

    #[tokio::test]
    async fn native_stream_projects_to_portable_without_a_second_request() {
        let server = MockServer::start().await;
        let created = json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_stream_native",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "in_progress",
                "output": [],
                "usage": null,
                "error": null,
                "incomplete_details": null,
                "reasoning": null
            }
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_stream_native",
                "created_at": 1,
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [{
                    "id": "msg_stream_native",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "one request",
                        "annotations": []
                    }]
                }],
                "usage": null,
                "error": null,
                "incomplete_details": null,
                "reasoning": null
            }
        });
        let body = format!("data: {created}\n\ndata: {completed}\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let model = provider_for(&server).await.responses(GPT_5_6_SOL).unwrap();
        let frames = model
            .stream_native(request(), CallOptions::default())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let mut events = Vec::new();
        for frame in frames {
            let frame = frame.unwrap();
            if frame.is_terminal() {
                assert!(frame.replay_status().is_available());
            }
            events.extend(frame.into_portable_events());
        }
        assert_eq!(
            events
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
                if response.content().iter().any(
                    |part| matches!(part, ContentPart::Text { text } if text == "one request")
                )
        ));
    }

    #[test]
    fn responses_preserve_encodable_intent_and_reject_unencodable_fields() {
        let provider = provider();
        let model = provider.responses(GPT_5_6_SOL).unwrap();
        let mut caller_request = request();
        caller_request.generation.temperature = Some(0.7);
        caller_request.generation.top_p = Some(0.9);
        let typed = OpenAiResponsesOptions {
            top_logprobs: Some(5),
            ..OpenAiResponsesOptions::default()
        };
        let call_options = CallOptions::default()
            .with_provider_options_for(&model, &typed)
            .unwrap();
        let merged = model
            .runtime
            .merge_options_for(&model, OpenAiApiMode::Responses, &call_options)
            .unwrap();

        let normalized_request =
            normalize_request(OpenAiApiMode::Responses, caller_request, &merged).unwrap();

        assert_eq!(normalized_request.generation.temperature, Some(0.7));
        assert_eq!(normalized_request.generation.top_p, Some(0.9));
        assert_eq!(merged.wire["top_logprobs"], 5);

        let mut with_seed = request();
        with_seed.generation.seed = Some(42);
        let error = normalize_request(OpenAiApiMode::Responses, with_seed, &merged).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);

        let mut with_stop = request();
        with_stop.generation.stop_sequences = vec!["stop".to_string()];
        let error = normalize_request(OpenAiApiMode::Responses, with_stop, &merged).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);

        let body = body_json(
            &model
                .plan(&normalized_request, false, merged)
                .expect("encodable caller intent reaches the Responses request"),
        );
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["top_logprobs"], 5);
    }
}
