use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamEvent, Model, ModelAdvisory, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ModelPolicy, ModelPolicyDecision, ProviderOptionError,
    ResponseDiagnostics, SensitiveResponse, SupportState, Warning, WarningKind,
};
use siumai_protocol_openai::chat_completions::{
    CHAT_COMPLETIONS_TARGET, ChatCompletionsStreamDecoder, decode_response, encode_request,
};
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    RequestBody, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders, TransportResponse,
    TransportStreamResponse,
};

use super::provider::ProviderRuntime;

/// Lightweight language model handle sharing one configured provider runtime.
#[derive(Clone)]
pub struct OpenAiCompatibleLanguageModel {
    pub(crate) runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl OpenAiCompatibleLanguageModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Language);
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(
        &self,
        operation: ModelOperation,
    ) -> Result<(ModelPolicyDecision, Vec<Warning>), Error> {
        let decision = self
            .runtime
            .policy
            .evaluate(&siumai_core::ModelPolicyContext {
                scope: self.runtime.scope.clone(),
                model: self.model_id().clone(),
                family: ModelFamily::Language,
                operation,
            });
        if let SupportState::Unsupported { .. } = decision.state() {
            return Err(self.contextualize(
                operation,
                Error::new(
                    ErrorKind::Unsupported,
                    "model policy rejected the requested Chat Completions operation",
                ),
            ));
        }
        let warnings = decision.advisories().iter().map(advisory_warning).collect();
        Ok((decision, warnings))
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        stream: bool,
        extra: &BTreeMap<String, serde_json::Value>,
    ) -> Result<RequestPlan, Error> {
        let body = encode_request(
            self.model_id(),
            request,
            stream,
            self.runtime.profile.dialect(),
            extra,
        )?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(CHAT_COMPLETIONS_TARGET).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(self.runtime.replay_safety.clone())
        .map_err(request_build_error)
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(operation),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for OpenAiCompatibleLanguageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleLanguageModel")
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
    ) -> Result<LanguageResponse, Error> {
        let (_, warnings) = self.policy(ModelOperation::Generate)?;
        let extra = self.runtime.merge_options(&options).map_err(option_error)?;
        let plan = self
            .plan(&request, false, &extra)
            .map_err(|error| self.contextualize(ModelOperation::Generate, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(ModelOperation::Generate, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(ModelOperation::Generate, response_error(response)));
        }
        let response = decode_response(
            &self.runtime.scope,
            self.model_id(),
            response.body(),
            self.runtime.profile.dialect(),
        )
        .map_err(|error| self.contextualize(ModelOperation::Generate, error))?;
        Ok(append_warnings(response, warnings))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let (_, warnings) = self.policy(ModelOperation::Stream)?;
        let extra = self.runtime.merge_options(&options).map_err(option_error)?;
        let plan = self
            .plan(&request, true, &extra)
            .map_err(|error| self.contextualize(ModelOperation::Stream, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(ModelOperation::Stream, error))?;
        if !response.status().is_success() {
            let error = stream_response_error(response).await;
            return Err(self.contextualize(ModelOperation::Stream, error));
        }

        let limits = self.runtime.transport.limits().clone();
        let scope = self.runtime.scope.as_ref().clone();
        let model = self.model_id().clone();
        let dialect = self.runtime.profile.dialect().clone();
        let body = response.into_body();
        Ok(established_stream(cancellation, move |_| {
            async_stream::try_stream! {
                let mut body = body;
                let mut framing = SseDecoder::new(&limits);
                let mut protocol = ChatCompletionsStreamDecoder::new(scope, model, dialect);
                while let Some(chunk) = body.next().await {
                    let chunk = chunk?;
                    let frames = framing.push(&chunk).map_err(sse_error)?;
                    for frame in frames {
                        let events = protocol.decode(frame.data())?;
                        for mut event in events {
                            if let LanguageStreamEvent::Terminal(
                                siumai_core::StreamTerminal::Completed { response }
                            ) = &mut event
                            {
                                **response = append_warnings(response.as_ref().clone(), warnings.clone());
                            }
                            let terminal = event.terminal().is_some();
                            yield event;
                            if terminal {
                                return;
                            }
                        }
                    }
                }
                framing.finish().map_err(sse_error)?;
            }
        }))
    }
}

fn append_warnings(mut response: LanguageResponse, warnings: Vec<Warning>) -> LanguageResponse {
    if warnings.is_empty() {
        return response;
    }
    let mut combined = response.warnings().to_vec();
    combined.extend(warnings);
    response = response.with_warnings(combined);
    response
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the verified advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "model is deprecated; inspect the provider profile for its replacement",
        ),
        ModelAdvisory::Retired { .. } => Warning::new(
            WarningKind::RetiredModel,
            "model is retired but this configured profile permits advisory use",
        ),
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "model ID is a rolling alias whose behavior may change",
        ),
        _ => Warning::provider(
            "model_advisory",
            "provider profile returned a model advisory",
        ),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Chat Completions",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Chat Completions request violates the transport contract",
    )
    .with_source(source)
}

fn sse_error(source: SseFrameError) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        SseFrameError::InvalidUtf8 | SseFrameError::UnexpectedEof => ErrorKind::Protocol,
        _ => ErrorKind::Protocol,
    };
    Error::new(
        kind,
        "provider returned an invalid Chat Completions SSE stream",
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
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                truncated |= chunk.len() > remaining;
            }
            Err(error) => return error,
        }
    }
    provider_status_error(status, headers, bytes, truncated)
}

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

fn provider_status_error(
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
) -> Error {
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        status if status.is_server_error() => ErrorKind::Provider,
        _ => ErrorKind::Provider,
    };
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
    Error::new(kind, "provider rejected the Chat Completions request")
        .with_diagnostics(
            ResponseDiagnostics::default()
                .with_status(status.as_u16())
                .with_body_truncated(body_truncated),
        )
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body))
}
