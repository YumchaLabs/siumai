use std::sync::Arc;

use async_trait::async_trait;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageCallError, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Model, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ProviderOptionError, ProviderScope, StreamTerminal, Warning,
};
use siumai_protocol_openai::chat_completions::{CHAT_COMPLETIONS_TARGET, ChatCompletionsDialect};
use siumai_protocol_openai::responses::RESPONSES_TARGET;
use siumai_transport::{RequestHeaders, RequestTarget};

use super::codec_policy::{
    ChatCodecPolicy, CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall,
    ResponsesCodecPolicy,
};
use super::execution::{
    DirectDecoder, DirectResponse, ExecutionContext, PreparedCall, PreparedJsonBody,
    SseStreamDecoder, StreamResponseContext, execute_direct, execute_sse,
};
use super::mode::OpenAiCompatibleApiMode;
use super::profile::LanguageModeProfile;
use super::provider::{CompatibleCallOptions, ProviderRuntime};

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

    fn encode_chat(
        &self,
        prepared: &PreparedChatCall,
        raw: Option<&serde_json::Map<String, serde_json::Value>>,
        stream: bool,
    ) -> Result<EncodedCall, Error> {
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
        Ok(EncodedCall {
            target: RequestTarget::new(CHAT_COMPLETIONS_TARGET)
                .map_err(|source| request_build_error(self.api_mode(), source))?,
            body,
            headers: prepared.headers.clone(),
            warnings: prepared.warnings.clone(),
        })
    }

    fn encode_responses(
        &self,
        prepared: &PreparedResponsesCall,
        raw: Option<&serde_json::Map<String, serde_json::Value>>,
        stream: bool,
    ) -> Result<EncodedCall, Error> {
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
        Ok(EncodedCall {
            target: RequestTarget::new(RESPONSES_TARGET)
                .map_err(|source| request_build_error(self.api_mode(), source))?,
            body,
            headers: prepared.headers.clone(),
            warnings: prepared.warnings.clone(),
        })
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

    fn execution_context(&self, operation: ModelOperation) -> ExecutionContext {
        execution_context(self.api_mode(), self.error_context(operation))
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
        let mode = self.api_mode();
        let CompatibleCallOptions { typed, raw } =
            self.runtime
                .options_for(self, mode, &options)
                .map_err(|source| self.contextualize(operation, option_error(mode, source)))?;
        let context = self.execution_context(operation);
        let call = match &self.mode {
            LanguageModeProfile::ChatCompletions {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_chat(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                let encoded = self
                    .encode_chat(&prepared, raw.as_ref(), false)
                    .map_err(|error| self.contextualize(operation, error))?;
                encoded
                    .into_prepared(
                        &self.runtime.transport,
                        self.runtime.replay_safety.clone(),
                        CompatibleDirectDecoder::Chat {
                            scope: scope.clone(),
                            model: self.model_id().clone(),
                            dialect: prepared.dialect,
                            codec_policy: codec_policy.clone(),
                        },
                        context,
                    )
                    .map_err(|source| {
                        self.contextualize(operation, request_build_error(mode, source))
                    })?
            }
            LanguageModeProfile::Responses {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_responses(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                let encoded = self
                    .encode_responses(&prepared, raw.as_ref(), false)
                    .map_err(|error| self.contextualize(operation, error))?;
                encoded
                    .into_prepared(
                        &self.runtime.transport,
                        self.runtime.replay_safety.clone(),
                        CompatibleDirectDecoder::Responses {
                            scope: scope.clone(),
                            model: self.model_id().clone(),
                            codec_policy: codec_policy.clone(),
                        },
                        context,
                    )
                    .map_err(|source| {
                        self.contextualize(operation, request_build_error(mode, source))
                    })?
            }
        };
        execute_direct(&self.runtime.transport, call, options)
            .await
            .map_err(LanguageCallError::from)?
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let mode = self.api_mode();
        let CompatibleCallOptions { typed, raw } =
            self.runtime
                .options_for(self, mode, &options)
                .map_err(|source| self.contextualize(operation, option_error(mode, source)))?;
        let context = self.execution_context(operation);
        let call = match &self.mode {
            LanguageModeProfile::ChatCompletions {
                scope,
                codec_policy,
                ..
            } => {
                let prepared = self
                    .prepare_chat(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                let encoded = self
                    .encode_chat(&prepared, raw.as_ref(), true)
                    .map_err(|error| self.contextualize(operation, error))?;
                let decoder = codec_policy.stream_decoder(
                    scope.as_ref().clone(),
                    self.model_id().clone(),
                    prepared.dialect,
                );
                encoded
                    .into_prepared(
                        &self.runtime.transport,
                        self.runtime.replay_safety.clone(),
                        CompatibleSseDecoder::new(decoder),
                        context,
                    )
                    .map_err(|source| {
                        self.contextualize(operation, request_build_error(mode, source))
                    })?
            }
            LanguageModeProfile::Responses {
                scope,
                wire_dialect,
                codec_policy,
            } => {
                let prepared = self
                    .prepare_responses(request, typed)
                    .map_err(|error| self.contextualize(operation, error))?;
                let encoded = self
                    .encode_responses(&prepared, raw.as_ref(), true)
                    .map_err(|error| self.contextualize(operation, error))?;
                let decoder = codec_policy.stream_decoder(
                    scope.as_ref().clone(),
                    self.model_id().clone(),
                    *wire_dialect,
                );
                encoded
                    .into_prepared(
                        &self.runtime.transport,
                        self.runtime.replay_safety.clone(),
                        CompatibleSseDecoder::new(decoder),
                        context,
                    )
                    .map_err(|source| {
                        self.contextualize(operation, request_build_error(mode, source))
                    })?
            }
        };
        let cancellation = options.cancellation().clone();
        let stream = execute_sse(&self.runtime.transport, call, options).await?;
        Ok(established_stream(cancellation, move |_| stream))
    }
}

struct EncodedCall {
    target: RequestTarget,
    body: serde_json::Value,
    headers: RequestHeaders,
    warnings: Vec<Warning>,
}

impl EncodedCall {
    fn into_prepared<D>(
        self,
        transport: &siumai_transport::ProviderTransport,
        replay_safety: siumai_transport::ReplaySafety,
        decoder: D,
        context: ExecutionContext,
    ) -> Result<PreparedCall<D>, siumai_transport::RequestBuildError> {
        let body = PreparedJsonBody::new(transport, &self.body)?;
        Ok(
            PreparedCall::new(self.target, body, replay_safety, decoder, context)
                .with_headers(self.headers)
                .with_warnings(self.warnings),
        )
    }
}

enum CompatibleDirectDecoder {
    Chat {
        scope: Arc<ProviderScope>,
        model: ModelId,
        dialect: ChatCompletionsDialect,
        codec_policy: Arc<dyn ChatCodecPolicy>,
    },
    Responses {
        scope: Arc<ProviderScope>,
        model: ModelId,
        codec_policy: Arc<dyn ResponsesCodecPolicy>,
    },
}

impl DirectDecoder for CompatibleDirectDecoder {
    type Output = Result<LanguageResponse, LanguageCallError>;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        match self {
            Self::Chat {
                scope,
                model,
                dialect,
                codec_policy,
            } => codec_policy
                .decode_response(
                    &scope,
                    &model,
                    response.headers(),
                    response.body(),
                    &dialect,
                )
                .map(|response_value| Ok(append_warnings(response_value, response.warnings()))),
            Self::Responses {
                scope,
                model,
                codec_policy,
            } => Ok(codec_policy
                .decode_response(&scope, &model, response.headers(), response.body())
                .map(|response_value| append_warnings(response_value, response.warnings()))
                .map_err(|error| {
                    contextualize_call_error(error, response.error_context().clone())
                })),
        }
    }
}

struct CompatibleSseDecoder {
    inner: CompatibleStreamDecoder,
    warnings: Vec<Warning>,
    error_context: Option<ErrorContext>,
}

impl CompatibleSseDecoder {
    fn new(inner: CompatibleStreamDecoder) -> Self {
        Self {
            inner,
            warnings: Vec::new(),
            error_context: None,
        }
    }

    fn prepare_events(
        &self,
        mut events: Vec<LanguageStreamEvent>,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let context = self.error_context.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "compatible SSE decoder was not initialized",
            )
        })?;
        for event in &mut events {
            contextualize_terminal_error(event, context);
            attach_warnings(event, &self.warnings);
        }
        Ok(events)
    }
}

impl SseStreamDecoder for CompatibleSseDecoder {
    type Event = LanguageStreamEvent;

    fn start(&mut self, context: StreamResponseContext) -> Result<(), Error> {
        self.inner
            .set_response_diagnostics(context.diagnostics().clone());
        self.warnings = context.warnings().to_vec();
        self.error_context = Some(context.error_context().clone());
        Ok(())
    }

    fn decode(&mut self, data: &str) -> Result<Vec<Self::Event>, Error> {
        let events = self.inner.decode(data)?;
        self.prepare_events(events)
    }

    fn is_terminal(&self, event: &Self::Event) -> bool {
        event.terminal().is_some()
    }

    fn finish(&mut self) -> Result<Vec<Self::Event>, Error> {
        let events = self.inner.finish()?;
        self.prepare_events(events)
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
    if let LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) = event {
        **response = append_warnings(response.as_ref().clone(), warnings);
    }
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

fn contextualize_call_error(error: LanguageCallError, context: ErrorContext) -> LanguageCallError {
    let (error, partial) = error.into_parts();
    LanguageCallError::new(error.with_context(context), partial)
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

fn execution_context(
    mode: OpenAiCompatibleApiMode,
    error_context: ErrorContext,
) -> ExecutionContext {
    match mode {
        OpenAiCompatibleApiMode::Responses => ExecutionContext::new(
            error_context,
            "Responses request violates the transport contract",
            "provider rejected the Responses request",
            "provider returned an invalid Responses SSE stream",
        ),
        OpenAiCompatibleApiMode::ChatCompletions => ExecutionContext::new(
            error_context,
            "Chat Completions request violates the transport contract",
            "provider rejected the Chat Completions request",
            "provider returned an invalid Chat Completions SSE stream",
        ),
    }
}
