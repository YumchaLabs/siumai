use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::Deserialize;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamDecoder, LanguageStreamEvent, Model, ModelAdvisory,
    ModelDescriptor, ModelFamily, ModelId, ModelOperation, ModelPolicyContext, ProviderOptionError,
    PublicDiagnosticText, SensitiveResponse, StreamTerminal, SupportState, Warning, WarningKind,
};
use siumai_protocol_anthropic::messages::{
    MessagesStreamDecoder, decode_response, encode_request_for_scope_with_resolver_and_rules,
};
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    RequestBody, RequestPlan, ResponseHeaders, TransportByteStream, TransportLimits,
    TransportResponse, TransportStreamResponse,
};

use crate::profile::MessagesRequestRequirements;
use crate::projection::MessagesRequestProjectionContext;
use crate::provider::ProviderRuntime;

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Lightweight language-model handle sharing one configured provider runtime.
#[derive(Clone)]
pub struct AnthropicCompatibleLanguageModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl AnthropicCompatibleLanguageModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Language);
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(&self, operation: ModelOperation) -> Result<Vec<Warning>, Error> {
        let decision = self.runtime.policy.evaluate(&ModelPolicyContext::new(
            self.runtime.scope.clone(),
            self.model_id().clone(),
            operation,
        ));
        if matches!(decision.state(), SupportState::Unsupported { .. }) {
            return Err(self.contextualize(
                operation,
                Error::new(
                    ErrorKind::Unsupported,
                    "model policy rejected the Anthropic Messages operation",
                ),
            ));
        }
        Ok(decision.advisories().iter().map(advisory_warning).collect())
    }

    fn request_plan(
        &self,
        body: serde_json::Value,
        stream: bool,
        requirements: &MessagesRequestRequirements,
    ) -> Result<RequestPlan, Error> {
        let beta_header = self.runtime.profile.beta_header(requirements)?;
        let context = MessagesRequestProjectionContext::new(
            self.model_id(),
            stream,
            self.runtime.profile.api_version(),
            self.runtime.profile.messages_target(),
            beta_header.as_deref(),
        );
        let projected = self
            .runtime
            .profile
            .request_projection()
            .project(&context, body)?;
        let (target, body, headers) = projected.into_parts();
        let accept = if stream {
            HeaderValue::from_static("text/event-stream")
        } else {
            HeaderValue::from_static("application/json")
        };
        let headers = headers
            .try_insert(ACCEPT, accept)
            .map_err(request_build_error)?;
        RequestPlan::new(Method::POST, target)
            .with_headers(headers)
            .with_body(RequestBody::json(&body).map_err(request_build_error)?)
            .with_replay_safety(self.runtime.replay_safety.clone())
            .map_err(request_build_error)
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        error.with_context(self.error_context(operation))
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

impl std::fmt::Debug for AnthropicCompatibleLanguageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicCompatibleLanguageModel")
            .field("descriptor", &self.descriptor)
            .field("runtime", &"shared")
            .finish()
    }
}

impl Model for AnthropicCompatibleLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for AnthropicCompatibleLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        let operation = ModelOperation::Generate;
        let warnings = self.policy(operation)?;
        let mut call_options = self
            .runtime
            .merge_options(&options)
            .map_err(|source| self.contextualize(operation, option_error(source)))?;
        let requirements = self
            .runtime
            .profile
            .request_policy()
            .prepare(self.model_id(), &request, &mut call_options)
            .map_err(|error| self.contextualize(operation, error))?;
        let protocol_options = call_options.into_protocol(false);
        let body = encode_request_for_scope_with_resolver_and_rules(
            &self.runtime.scope,
            self.model_id(),
            &request,
            &protocol_options,
            self.runtime.profile.annotation_resolver().as_ref(),
            self.runtime.profile.encoding_rules(),
        )
        .map_err(Error::from)
        .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .request_plan(body, false, &requirements)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(operation, response_error(response)));
        }
        let response = decode_response(response.body(), &self.runtime.scope, self.model_id())
            .map_err(Error::from)
            .map_err(|error| self.contextualize(operation, error))?;
        Ok(append_warnings(response, &warnings))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let warnings = self.policy(operation)?;
        let mut call_options = self
            .runtime
            .merge_options(&options)
            .map_err(|source| self.contextualize(operation, option_error(source)))?;
        let requirements = self
            .runtime
            .profile
            .request_policy()
            .prepare(self.model_id(), &request, &mut call_options)
            .map_err(|error| self.contextualize(operation, error))?;
        let protocol_options = call_options.into_protocol(true);
        let body = encode_request_for_scope_with_resolver_and_rules(
            &self.runtime.scope,
            self.model_id(),
            &request,
            &protocol_options,
            self.runtime.profile.annotation_resolver().as_ref(),
            self.runtime.profile.encoding_rules(),
        )
        .map_err(Error::from)
        .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .request_plan(body, true, &requirements)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(response).await;
            return Err(self.contextualize(operation, error));
        }
        let (status, headers, body) = response.into_parts();
        let diagnostics = headers.diagnostics().with_status(status.as_u16());
        let decoder = MessagesStreamDecoder::new(
            self.runtime.scope.as_ref().clone(),
            self.model_id().clone(),
        )
        .with_response_diagnostics(diagnostics);
        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.transport.limits().clone(),
            decoder,
            warnings,
            self.error_context(operation),
        ))
    }
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
    match event {
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })
        | LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            response: Some(response),
            ..
        })
        | LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            response: Some(response),
            ..
        }) => {
            **response = append_warnings(response.as_ref().clone(), warnings);
        }
        _ => {}
    }
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the verified advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "model is deprecated; inspect the compatible profile for its replacement",
        ),
        ModelAdvisory::Retired { .. } => Warning::new(
            WarningKind::RetiredModel,
            "model is retired in the compatible profile",
        ),
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "model ID is a rolling alias whose behavior may change",
        ),
        _ => Warning::provider(
            "model_advisory",
            "compatible profile returned a model advisory",
        ),
    }
}

fn decode_sse_stream(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: TransportLimits,
    mut protocol: MessagesStreamDecoder,
    warnings: Vec<Warning>,
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
                    .map_err(|source| sse_error(source).with_context(context.clone()))?;
                for frame in frames {
                    let events = protocol
                        .decode(frame.data())
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
            }
            framing
                .finish()
                .map_err(|source| sse_error(source).with_context(context.clone()))?;
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

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Anthropic Messages",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Anthropic Messages request violates the transport contract",
    )
    .with_source(source)
}

fn sse_error(source: SseFrameError) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        _ => ErrorKind::Protocol,
    };
    Error::new(
        kind,
        "provider returned an invalid Anthropic Messages SSE stream",
    )
    .with_source(source)
}

fn response_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    provider_status_error(status, headers, captured, truncated)
}

async fn stream_response_error(response: TransportStreamResponse) -> Error {
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
    provider_status_error(status, headers, bytes, truncated)
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
    #[serde(default)]
    request_id: Option<String>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: Option<String>,
}

fn provider_status_error(
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
) -> Error {
    let envelope = serde_json::from_slice::<ErrorEnvelope>(&body).ok();
    let provider_type = envelope
        .as_ref()
        .and_then(|envelope| envelope.error.as_ref())
        .and_then(|error| error.kind.as_deref())
        .and_then(public_provider_identifier);
    let kind = classify_http_error(
        status,
        provider_type.as_ref().map(PublicDiagnosticText::as_str),
    );
    let body_request_id = envelope
        .as_ref()
        .and_then(|envelope| envelope.request_id.as_deref())
        .and_then(public_provider_identifier);
    let mut diagnostics = headers
        .diagnostics()
        .with_status(status.as_u16())
        .with_body_truncated(body_truncated);
    if let Some(provider_type) = provider_type {
        diagnostics = diagnostics.with_provider_type(provider_type);
    }
    if diagnostics.request_id().is_none()
        && let Some(request_id) = body_request_id
    {
        diagnostics = diagnostics.with_request_id(request_id);
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
    Error::new(kind, "provider rejected the Anthropic Messages request")
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body))
}

fn classify_http_error(status: StatusCode, provider_type: Option<&str>) -> ErrorKind {
    match provider_type {
        Some("authentication_error") => return ErrorKind::Authentication,
        Some("billing_error") => return ErrorKind::QuotaExceeded,
        Some("permission_error") => return ErrorKind::Authorization,
        Some("rate_limit_error") => return ErrorKind::RateLimited,
        Some("invalid_request_error" | "not_found_error") => return ErrorKind::InvalidInput,
        Some("request_too_large") => return ErrorKind::LimitExceeded,
        Some("timeout_error") => return ErrorKind::Timeout,
        Some("api_error" | "overloaded_error") => return ErrorKind::Unavailable,
        _ => {}
    }
    match status.as_u16() {
        400 | 404 | 405 | 409 | 422 => ErrorKind::InvalidInput,
        401 => ErrorKind::Authentication,
        402 => ErrorKind::QuotaExceeded,
        403 => ErrorKind::Authorization,
        408 | 504 => ErrorKind::Timeout,
        413 => ErrorKind::LimitExceeded,
        429 => ErrorKind::RateLimited,
        500..=599 => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    }
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
