use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageCallError, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Model, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ProviderOptionError, PublicDiagnosticText, SensitiveResponse,
    StreamTerminal, Warning,
};
use siumai_protocol_openai::chat_completions::CHAT_COMPLETIONS_TARGET;
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_protocol_openai::responses::RESPONSES_TARGET;
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    RequestBody, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportByteStream,
    TransportLimits, TransportResponse, TransportStreamResponse,
};

use super::codec_policy::{CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall};
use super::mode::OpenAiCompatibleApiMode;
use super::profile::LanguageModeProfile;
use super::provider::{CompatibleCallOptions, ProviderRuntime};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Lightweight language model handle sharing one configured provider runtime.
#[derive(Clone)]
pub struct OpenAiCompatibleLanguageModel {
    pub(crate) runtime: Arc<ProviderRuntime>,
    mode: LanguageModeProfile,
    descriptor: ModelDescriptor,
}

impl OpenAiCompatibleLanguageModel {
    pub(crate) fn new(
        runtime: Arc<ProviderRuntime>,
        mode: LanguageModeProfile,
        model: ModelId,
    ) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            mode.scope().clone(),
            model,
            ModelFamily::Language,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            mode,
            descriptor,
        }
    }

    pub const fn api_mode(&self) -> OpenAiCompatibleApiMode {
        self.mode.api_mode()
    }

    fn prepare_chat(
        &self,
        request: LanguageRequest,
        extra: std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<PreparedChatCall, Error> {
        match &self.mode {
            LanguageModeProfile::ChatCompletions {
                dialect,
                codec_policy,
                ..
            } => codec_policy.prepare(self.model_id(), request, dialect.clone(), extra),
            LanguageModeProfile::Responses { .. } => Err(inconsistent_mode_error(
                "Chat Completions preparation requires a Chat Completions model",
            )),
        }
    }

    fn prepare_responses(
        &self,
        request: LanguageRequest,
        extra: std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        match &self.mode {
            LanguageModeProfile::Responses { codec_policy, .. } => {
                codec_policy.prepare(self.model_id(), request, extra)
            }
            LanguageModeProfile::ChatCompletions { .. } => Err(inconsistent_mode_error(
                "Responses preparation requires a Responses model",
            )),
        }
    }

    fn chat_plan(
        &self,
        prepared: &PreparedChatCall,
        raw: Option<&serde_json::Map<String, serde_json::Value>>,
        stream: bool,
    ) -> Result<RequestPlan, Error> {
        let LanguageModeProfile::ChatCompletions {
            scope,
            codec_policy,
            ..
        } = &self.mode
        else {
            return Err(inconsistent_mode_error(
                "Chat Completions encoding requires a Chat Completions model",
            ));
        };
        let mut body = codec_policy.encode_request(scope, self.model_id(), prepared, stream)?;
        codec_policy
            .apply_raw_body_overlay(&mut body, raw)
            .map_err(|source| option_error(self.api_mode(), source))?;
        request_plan(
            self.api_mode(),
            CHAT_COMPLETIONS_TARGET,
            body,
            stream,
            prepared.headers.clone(),
            self.runtime.replay_safety.clone(),
        )
    }

    fn responses_plan(
        &self,
        prepared: &PreparedResponsesCall,
        raw: Option<&serde_json::Map<String, serde_json::Value>>,
        stream: bool,
    ) -> Result<RequestPlan, Error> {
        let LanguageModeProfile::Responses {
            scope,
            codec_policy,
            ..
        } = &self.mode
        else {
            return Err(inconsistent_mode_error(
                "Responses encoding requires a Responses model",
            ));
        };
        let mut body = codec_policy.encode_request(scope, self.model_id(), prepared, stream)?;
        codec_policy
            .apply_raw_body_overlay(&mut body, raw)
            .map_err(|source| option_error(self.api_mode(), source))?;
        request_plan(
            self.api_mode(),
            RESPONSES_TARGET,
            body,
            stream,
            prepared.headers.clone(),
            self.runtime.replay_safety.clone(),
        )
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        error.with_context(self.error_context(operation))
    }

    fn contextualize_call_error(
        &self,
        operation: ModelOperation,
        error: LanguageCallError,
    ) -> LanguageCallError {
        let (error, partial) = error.into_parts();
        LanguageCallError::new(self.contextualize(operation, error), partial)
    }

    fn error_context(&self, operation: ModelOperation) -> ErrorContext {
        ErrorContext {
            operation: Some(operation),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        }
    }
}

impl std::fmt::Debug for OpenAiCompatibleLanguageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleLanguageModel")
            .field("mode", &self.api_mode())
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiCompatibleLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for OpenAiCompatibleLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        let operation = ModelOperation::Generate;
        let mut warnings = Vec::new();
        let mode = self.api_mode();
        let CompatibleCallOptions { typed, raw } =
            self.runtime
                .options_for(self, mode, &options)
                .map_err(|source| self.contextualize(operation, option_error(mode, source)))?;

        let response = match &self.mode {
            LanguageModeProfile::ChatCompletions {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_chat(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                warnings.extend(prepared.warnings.iter().cloned());
                let plan = self
                    .chat_plan(&prepared, raw.as_ref(), false)
                    .map_err(|error| self.contextualize(operation, error))?;
                let response = self
                    .runtime
                    .transport
                    .execute(plan, options)
                    .await
                    .map_err(|error| self.contextualize(operation, error))?;
                if !response.status().is_success() {
                    return Err(self
                        .contextualize(operation, response_error(mode, response))
                        .into());
                }
                codec_policy
                    .decode_response(
                        scope,
                        self.model_id(),
                        response.headers(),
                        response.body(),
                        &prepared.dialect,
                    )
                    .map_err(|error| self.contextualize(operation, error))?
            }
            LanguageModeProfile::Responses {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_responses(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                warnings.extend(prepared.warnings.iter().cloned());
                let plan = self
                    .responses_plan(&prepared, raw.as_ref(), false)
                    .map_err(|error| self.contextualize(operation, error))?;
                let response = self
                    .runtime
                    .transport
                    .execute(plan, options)
                    .await
                    .map_err(|error| self.contextualize(operation, error))?;
                if !response.status().is_success() {
                    return Err(self
                        .contextualize(operation, response_error(mode, response))
                        .into());
                }
                codec_policy
                    .decode_response(scope, self.model_id(), response.headers(), response.body())
                    .map_err(|error| self.contextualize_call_error(operation, error))?
            }
        };
        Ok(append_warnings(response, &warnings))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let mut warnings = Vec::new();
        let mode = self.api_mode();
        let CompatibleCallOptions { typed, raw } =
            self.runtime
                .options_for(self, mode, &options)
                .map_err(|source| self.contextualize(operation, option_error(mode, source)))?;

        let (plan, mut decoder) = match &self.mode {
            LanguageModeProfile::ChatCompletions {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_chat(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                warnings.extend(prepared.warnings.iter().cloned());
                let plan = self
                    .chat_plan(&prepared, raw.as_ref(), true)
                    .map_err(|error| self.contextualize(operation, error))?;
                let decoder = codec_policy.stream_decoder(
                    scope.as_ref().clone(),
                    self.model_id().clone(),
                    prepared.dialect,
                );
                (plan, decoder)
            }
            LanguageModeProfile::Responses {
                scope,
                wire_dialect,
                codec_policy,
            } => {
                let prepared = self
                    .prepare_responses(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                warnings.extend(prepared.warnings.iter().cloned());
                let plan = self
                    .responses_plan(&prepared, raw.as_ref(), true)
                    .map_err(|error| self.contextualize(operation, error))?;
                let decoder = codec_policy.stream_decoder(
                    scope.as_ref().clone(),
                    self.model_id().clone(),
                    *wire_dialect,
                );
                (plan, decoder)
            }
        };
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(mode, response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        decoder.set_response_diagnostics(headers.diagnostics().with_status(status.as_u16()));

        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            decoder,
            warnings,
            mode,
            self.error_context(operation),
        ))
    }
}

fn request_plan(
    mode: OpenAiCompatibleApiMode,
    target: &'static str,
    body: serde_json::Value,
    stream: bool,
    mut headers: RequestHeaders,
    replay_safety: siumai_transport::ReplaySafety,
) -> Result<RequestPlan, Error> {
    let accept = if stream {
        HeaderValue::from_static("text/event-stream")
    } else {
        HeaderValue::from_static("application/json")
    };
    headers = headers
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

fn append_warnings(mut response: LanguageResponse, warnings: &[Warning]) -> LanguageResponse {
    if warnings.is_empty() {
        return response;
    }
    let mut combined = response.warnings().to_vec();
    combined.extend_from_slice(warnings);
    response = response.with_warnings(combined);
    response
}

fn attach_warnings(event: &mut LanguageStreamEvent, warnings: &[Warning]) {
    if warnings.is_empty() {
        return;
    }
    if let LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) = event {
        **response = append_warnings(response.as_ref().clone(), warnings);
    }
}

fn decode_sse_stream(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: TransportLimits,
    mut protocol: CompatibleStreamDecoder,
    warnings: Vec<Warning>,
    mode: OpenAiCompatibleApiMode,
    context: ErrorContext,
) -> LanguageStream {
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
                            "compatible stream decoder emitted events after terminal settlement",
                        )
                        .with_context(context.clone()))?;
                    }
                    terminal_in_batch |= terminal_position.is_some();
                    for mut event in events {
                        contextualize_terminal_error(&mut event, &context);
                        attach_warnings(&mut event, &warnings);
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
                attach_warnings(&mut event, &warnings);
                let terminal = event.terminal().is_some();
                yield event;
                if terminal {
                    return;
                }
            }
        }
    })
}

fn contextualize_terminal_error(event: &mut LanguageStreamEvent, context: &ErrorContext) {
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

fn inconsistent_mode_error(message: &'static str) -> Error {
    Error::new(ErrorKind::Internal, message)
}

fn option_error(mode: OpenAiCompatibleApiMode, source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                "provider options are invalid for the Responses API"
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                "provider options are invalid for Chat Completions"
            }
        },
    )
    .with_source(source)
}

fn request_build_error(
    mode: OpenAiCompatibleApiMode,
    source: siumai_transport::RequestBuildError,
) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                "Responses request violates the transport contract"
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                "Chat Completions request violates the transport contract"
            }
        },
    )
    .with_source(source)
}

fn sse_error(mode: OpenAiCompatibleApiMode, source: SseFrameError) -> Error {
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
            OpenAiCompatibleApiMode::Responses => {
                "provider returned an invalid Responses SSE stream"
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                "provider returned an invalid Chat Completions SSE stream"
            }
        },
    )
    .with_source(source)
}

fn response_error(mode: OpenAiCompatibleApiMode, response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    provider_status_error(mode, status, headers, captured, truncated)
}

async fn stream_response_error(
    mode: OpenAiCompatibleApiMode,
    response: TransportStreamResponse,
) -> Error {
    let (status, headers, mut body) = response.into_parts();
    let mut bytes = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = ERROR_CAPTURE_BYTES.saturating_sub(bytes.len());
                if remaining == 0 {
                    truncated = true;
                    break;
                }
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if chunk.len() > remaining {
                    truncated = true;
                    break;
                }
            }
            Err(error) => return error,
        }
    }
    provider_status_error(mode, status, headers, bytes, truncated)
}

fn provider_status_error(
    mode: OpenAiCompatibleApiMode,
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
) -> Error {
    let metadata = decode_error_metadata(&body);
    let provider_code = metadata
        .as_ref()
        .and_then(|metadata| metadata.code())
        .and_then(public_provider_identifier);
    let provider_type = metadata
        .as_ref()
        .and_then(|metadata| metadata.error_type())
        .and_then(public_provider_identifier);
    let provider_param = metadata
        .as_ref()
        .and_then(|metadata| metadata.param())
        .and_then(public_provider_identifier);
    let kind = classify_http_error(
        status.as_u16(),
        provider_code.as_ref().map(PublicDiagnosticText::as_str),
        provider_type.as_ref().map(PublicDiagnosticText::as_str),
    );
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body_truncated);
    if let Some(code) = provider_code {
        diagnostics = diagnostics.with_provider_code(code);
    }
    if let Some(error_type) = provider_type {
        diagnostics = diagnostics.with_provider_type(error_type);
    }
    if let Some(param) = provider_param {
        diagnostics = diagnostics.with_provider_param(param);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    Error::new(
        kind,
        match mode {
            OpenAiCompatibleApiMode::Responses => "provider rejected the Responses request",
            OpenAiCompatibleApiMode::ChatCompletions => {
                "provider rejected the Chat Completions request"
            }
        },
    )
    .with_diagnostics(diagnostics)
    .with_sensitive_response(SensitiveResponse::new(raw_headers, body))
}

fn public_provider_identifier(value: &str) -> Option<PublicDiagnosticText> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return None;
    }
    PublicDiagnosticText::new(value.to_string()).ok()
}
