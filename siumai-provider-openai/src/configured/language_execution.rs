//! OpenAI-specific adapters for the shared stateless HTTP/SSE execution kernel.

use std::sync::Arc;

use serde_json::Value;
use siumai_core::{
    Error, ErrorContext, ErrorKind, LanguageCallError, LanguageResponse, LanguageStreamDecoder,
    LanguageStreamEvent, Model, ModelId, ModelOperation, ProviderScope, StreamTerminal,
};
use siumai_openai_compatible::extension::v2::{
    DirectDecoder, DirectResponse, ExecutionContext, PreparedCall, PreparedJsonBody,
    SseStreamDecoder, StreamResponseContext,
};
use siumai_protocol_openai::chat_completions::{
    ChatCompletionsDialect, ChatCompletionsStreamDecoder, decode_response as decode_chat_response,
};
use siumai_protocol_openai::responses::{
    ResponsesStreamDecoder, ResponsesWireDialect, decode_response as decode_responses_response,
    decode_response_resource,
};
use siumai_transport::{ProviderTransport, ReplaySafety, RequestBuildError, RequestTarget};

use super::http_error;
use super::mode::OpenAiApiMode;
use super::responses_native::{OpenAiResponsesResponse, OpenAiResponsesStreamFrame};
use super::responses_resource::OpenAiBackgroundResponse;

pub(crate) fn prepare_call<D>(
    transport: &ProviderTransport,
    mode: OpenAiApiMode,
    target: &'static str,
    body: Value,
    replay_safety: ReplaySafety,
    decoder: D,
    error_context: ErrorContext,
) -> Result<PreparedCall<D>, Error> {
    let context = execution_context(mode, error_context.clone());
    let target = RequestTarget::new(target)
        .map_err(|source| request_build_error(mode, source).with_context(error_context.clone()))?;
    let body = PreparedJsonBody::new(transport, &body)
        .map_err(|source| request_build_error(mode, source).with_context(error_context))?;
    Ok(PreparedCall::new(
        target,
        body,
        replay_safety,
        decoder,
        context,
    ))
}

fn execution_context(mode: OpenAiApiMode, error_context: ErrorContext) -> ExecutionContext {
    match mode {
        OpenAiApiMode::Responses => ExecutionContext::new(
            error_context,
            "OpenAI Responses request violates the transport contract",
            "OpenAI rejected the Responses request",
            "provider returned an invalid OpenAI Responses SSE stream",
        ),
        OpenAiApiMode::ChatCompletions => ExecutionContext::new(
            error_context,
            "OpenAI Chat Completions request violates the transport contract",
            "OpenAI rejected the Chat Completions request",
            "provider returned an invalid OpenAI Chat Completions SSE stream",
        ),
    }
}

fn request_build_error(mode: OpenAiApiMode, source: RequestBuildError) -> Error {
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

pub(crate) struct ResponsesDirectDecoder {
    scope: Arc<ProviderScope>,
    model: ModelId,
}

impl ResponsesDirectDecoder {
    pub(crate) fn new(scope: Arc<ProviderScope>, model: ModelId) -> Self {
        Self { scope, model }
    }
}

impl DirectDecoder for ResponsesDirectDecoder {
    type Output = OpenAiResponsesResponse;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        let decoded = decode_responses_response(response.body(), &self.scope, &self.model)?;
        let (native, portable) = decoded.into_parts();
        let portable = portable
            .map_err(|error| contextualize_call_error(error, response.error_context().clone()));
        Ok(OpenAiResponsesResponse::new(native, portable))
    }
}

pub(crate) struct BackgroundDirectDecoder;

impl DirectDecoder for BackgroundDirectDecoder {
    type Output = OpenAiBackgroundResponse;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        decode_response_resource(response.body()).map(OpenAiBackgroundResponse::new)
    }
}

pub(crate) struct ChatDirectDecoder {
    scope: Arc<ProviderScope>,
    model: ModelId,
    dialect: ChatCompletionsDialect,
}

impl ChatDirectDecoder {
    pub(crate) fn new(
        scope: Arc<ProviderScope>,
        model: ModelId,
        dialect: ChatCompletionsDialect,
    ) -> Self {
        Self {
            scope,
            model,
            dialect,
        }
    }
}

impl DirectDecoder for ChatDirectDecoder {
    type Output = LanguageResponse;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error> {
        decode_chat_response(&self.scope, &self.model, response.body(), &self.dialect)
    }
}

pub(crate) struct ChatSseDecoder {
    inner: ChatCompletionsStreamDecoder,
    error_context: Option<ErrorContext>,
}

impl ChatSseDecoder {
    pub(crate) fn new(inner: ChatCompletionsStreamDecoder) -> Self {
        Self {
            inner,
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
                "OpenAI SSE decoder was not initialized",
            )
        })?;
        for event in &mut events {
            contextualize_terminal_error(event, context);
        }
        Ok(events)
    }
}

impl SseStreamDecoder for ChatSseDecoder {
    type Event = LanguageStreamEvent;

    fn start(&mut self, context: StreamResponseContext) -> Result<(), Error> {
        self.inner
            .set_response_diagnostics(context.diagnostics().clone());
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

pub(crate) struct ResponsesSseDecoder {
    inner: ResponsesStreamDecoder,
    error_context: Option<ErrorContext>,
}

impl ResponsesSseDecoder {
    pub(crate) fn new(
        scope: ProviderScope,
        model: ModelId,
        wire_dialect: ResponsesWireDialect,
    ) -> Self {
        Self {
            inner: ResponsesStreamDecoder::new(scope, model).with_wire_dialect(wire_dialect),
            error_context: None,
        }
    }

    fn context(&self) -> Result<&ErrorContext, Error> {
        self.error_context.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "OpenAI Responses SSE decoder was not initialized",
            )
        })
    }
}

impl SseStreamDecoder for ResponsesSseDecoder {
    type Event = OpenAiResponsesStreamFrame;

    fn start(&mut self, context: StreamResponseContext) -> Result<(), Error> {
        self.inner
            .set_response_diagnostics(context.diagnostics().clone());
        self.error_context = Some(context.error_context().clone());
        Ok(())
    }

    fn decode(&mut self, data: &str) -> Result<Vec<Self::Event>, Error> {
        let decoded = self.inner.decode_native(data)?;
        let (native, mut portable_events, replay_status) = decoded.into_parts();
        validate_portable_terminal_order(&portable_events)?;
        let context = self.context()?.clone();
        for event in &mut portable_events {
            contextualize_terminal_error(event, &context);
        }
        let canonical_terminal_response = portable_events
            .iter()
            .any(|event| event.terminal().is_some())
            .then(|| self.inner.terminal_response().cloned())
            .flatten();
        Ok(vec![OpenAiResponsesStreamFrame::new(
            native,
            portable_events,
            canonical_terminal_response,
            replay_status,
        )])
    }

    fn is_terminal(&self, event: &Self::Event) -> bool {
        event.is_terminal()
    }

    fn finish(&mut self) -> Result<Vec<Self::Event>, Error> {
        let trailing = self.inner.finish()?;
        if trailing.is_empty() {
            Ok(Vec::new())
        } else {
            Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses decoder produced portable events without a native frame",
            ))
        }
    }
}

fn validate_portable_terminal_order(events: &[LanguageStreamEvent]) -> Result<(), Error> {
    let mut terminal_index = None;
    for (index, event) in events.iter().enumerate() {
        if event.terminal().is_none() {
            continue;
        }
        if terminal_index.replace(index).is_some() {
            return Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses decoder emitted duplicate terminal events",
            ));
        }
    }
    if terminal_index.is_some_and(|index| index + 1 != events.len()) {
        return Err(Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses decoder emitted events after terminal settlement",
        ));
    }
    Ok(())
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

fn contextualize_call_error(error: LanguageCallError, context: ErrorContext) -> LanguageCallError {
    let (error, partial) = error.into_parts();
    LanguageCallError::new(error.with_context(context), partial)
}

pub(crate) fn model_error_context(model: &impl Model, operation: ModelOperation) -> ErrorContext {
    ErrorContext {
        operation: Some(operation),
        provider: Some(model.provider_id().clone()),
        route: None,
        model: Some(model.model_id().clone()),
    }
}
