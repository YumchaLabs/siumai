//! Stateless OpenAI-family HTTP and SSE execution mechanics.
//!
//! Provider code owns request encoding, identity, options, credentials, endpoint selection, and
//! response semantics. This module accepts only an already validated relative target, JSON body,
//! non-credential headers, replay proof, warnings, diagnostics context, and provider-owned
//! decoders. It constructs the immutable transport plan and owns bounded response handling,
//! OpenAI error classification, SSE framing, terminal ordering, EOF, and child cancellation.

use std::fmt;
use std::io::{self, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::{Stream, StreamExt};
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::Serialize;
use siumai_core::{
    CallOptions, Cancellation, Error, ErrorContext, ErrorKind, PublicDiagnosticText,
    ResponseDiagnostics, SensitiveResponse, Warning,
};
use siumai_protocol_openai::openai_error::{classify_http_error, decode_error_metadata};
use siumai_transport::framing::{SseDecoder, SseFrameError};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan,
    RequestTarget, ResponseHeaders, TransportLimits, TransportResponse, TransportStreamResponse,
};

const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Sanitized provider-authored diagnostics for one stateless execution.
///
/// The message arguments must be public, static diagnostic text. Runtime data, credentials,
/// endpoints, headers, request bodies, and provider response bodies must never be interpolated.
#[derive(Clone)]
pub struct ExecutionContext {
    error_context: ErrorContext,
    request_contract_message: &'static str,
    rejected_response_message: &'static str,
    invalid_sse_message: &'static str,
}

/// JSON body serialized within the selected transport's request-byte limit.
///
/// Construction aborts before retaining bytes beyond `max_request_bytes`. The transport still
/// validates the encoded body again at execution, so using a different transport with a lower
/// limit fails before network submission.
pub struct PreparedJsonBody {
    body: RequestBody,
    encoded_bytes: usize,
}

impl PreparedJsonBody {
    pub fn new<T>(transport: &ProviderTransport, value: &T) -> Result<Self, RequestBuildError>
    where
        T: Serialize + ?Sized,
    {
        let maximum = transport.limits().max_request_bytes;
        let mut writer = BoundedJsonWriter::new(maximum);
        let serialized = serde_json::to_writer(&mut writer, value);
        if writer.exceeded() {
            return Err(RequestBuildError::BodyTooLarge { maximum });
        }
        serialized.map_err(|_| RequestBuildError::JsonSerialization)?;
        let bytes = writer.into_bytes();
        let encoded_bytes = bytes.len();
        Ok(Self {
            body: RequestBody::bytes_with_content_type(
                bytes,
                HeaderValue::from_static("application/json"),
            ),
            encoded_bytes,
        })
    }

    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    fn into_request_body(self) -> RequestBody {
        self.body
    }
}

impl fmt::Debug for PreparedJsonBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedJsonBody")
            .field("encoded_bytes", &self.encoded_bytes)
            .finish()
    }
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    maximum: usize,
    exceeded: bool,
}

impl BoundedJsonWriter {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(maximum.min(8 * 1024)),
            maximum,
            exceeded: false,
        }
    }

    fn exceeded(&self) -> bool {
        self.exceeded
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.len() > self.maximum.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other(
                "JSON request exceeds the configured byte limit",
            ));
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ExecutionContext {
    pub fn new(
        error_context: ErrorContext,
        request_contract_message: &'static str,
        rejected_response_message: &'static str,
        invalid_sse_message: &'static str,
    ) -> Self {
        Self {
            error_context,
            request_contract_message,
            rejected_response_message,
            invalid_sse_message,
        }
    }

    pub fn error_context(&self) -> &ErrorContext {
        &self.error_context
    }
}

impl fmt::Debug for ExecutionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionContext")
            .field("error_context", &self.error_context)
            .field(
                "request_contract_message_bytes",
                &self.request_contract_message.len(),
            )
            .field(
                "rejected_response_message_bytes",
                &self.rejected_response_message.len(),
            )
            .field("invalid_sse_message_bytes", &self.invalid_sse_message.len())
            .finish()
    }
}

/// One provider-prepared stateless OpenAI-family call.
///
/// The kernel fixes the method to `POST`, serializes the body as JSON, and owns the `Accept`
/// header. The selected [`ProviderTransport`] remains the sole owner of endpoint, credentials,
/// signing, retry policy, timeouts, admission, and resource limits.
pub struct PreparedCall<D> {
    target: RequestTarget,
    headers: RequestHeaders,
    body: PreparedJsonBody,
    replay_safety: ReplaySafety,
    warnings: Vec<Warning>,
    context: ExecutionContext,
    decoder: D,
}

impl<D> PreparedCall<D> {
    pub fn new(
        target: RequestTarget,
        body: PreparedJsonBody,
        replay_safety: ReplaySafety,
        decoder: D,
        context: ExecutionContext,
    ) -> Self {
        Self {
            target,
            headers: RequestHeaders::new(),
            body,
            replay_safety,
            warnings: Vec::new(),
            context,
            decoder,
        }
    }

    pub fn with_headers(mut self, headers: RequestHeaders) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }
}

impl<D> fmt::Debug for PreparedCall<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCall")
            .field("target", &self.target)
            .field("headers", &self.headers)
            .field("body", &self.body)
            .field("replay_safety", &self.replay_safety)
            .field("warning_count", &self.warnings.len())
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

/// Borrowed successful direct response passed to a provider-owned decoder.
pub struct DirectResponse<'a> {
    status: StatusCode,
    headers: &'a ResponseHeaders,
    body: &'a [u8],
    attempts: u8,
    warnings: &'a [Warning],
    error_context: &'a ErrorContext,
}

impl DirectResponse<'_> {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &ResponseHeaders {
        self.headers
    }

    pub fn body(&self) -> &[u8] {
        self.body
    }

    pub fn attempts(&self) -> u8 {
        self.attempts
    }

    pub fn warnings(&self) -> &[Warning] {
        self.warnings
    }

    pub fn error_context(&self) -> &ErrorContext {
        self.error_context
    }
}

impl fmt::Debug for DirectResponse<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body_bytes", &self.body.len())
            .field("body", &"[REDACTED]")
            .field("attempts", &self.attempts)
            .field("warning_count", &self.warnings.len())
            .field("error_context", &self.error_context)
            .finish()
    }
}

/// Provider-owned decoder for one successful buffered response.
///
/// A provider-specific failure carrier may be preserved as part of `Output`; for example, a
/// decoder can use `Result<LanguageResponse, LanguageCallError>` as its associated output without
/// teaching the execution kernel about portable language semantics.
pub trait DirectDecoder: Send + 'static {
    type Output: Send + 'static;

    fn decode(self, response: DirectResponse<'_>) -> Result<Self::Output, Error>;
}

/// Owned successful-stream context delivered before the first SSE event.
#[derive(Clone)]
pub struct StreamResponseContext {
    status: StatusCode,
    headers: ResponseHeaders,
    diagnostics: ResponseDiagnostics,
    attempts: u8,
    warnings: Arc<[Warning]>,
    error_context: ErrorContext,
}

impl StreamResponseContext {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &ResponseHeaders {
        &self.headers
    }

    pub fn diagnostics(&self) -> &ResponseDiagnostics {
        &self.diagnostics
    }

    pub fn attempts(&self) -> u8 {
        self.attempts
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    pub fn error_context(&self) -> &ErrorContext {
        &self.error_context
    }
}

impl fmt::Debug for StreamResponseContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamResponseContext")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("diagnostics", &self.diagnostics)
            .field("attempts", &self.attempts)
            .field("warning_count", &self.warnings.len())
            .field("error_context", &self.error_context)
            .finish()
    }
}

/// Provider-owned decoder for framed OpenAI-family SSE data.
///
/// `start` is called exactly once after a successful HTTP stream is established. `finish` is
/// called exactly once at clean framing EOF. The kernel validates terminal ordering independently
/// of decoder-internal state and treats EOF without an authoritative terminal event as failure.
pub trait SseStreamDecoder: Send + 'static {
    type Event: Send + 'static;

    fn start(&mut self, _context: StreamResponseContext) -> Result<(), Error> {
        Ok(())
    }

    fn decode(&mut self, data: &str) -> Result<Vec<Self::Event>, Error>;

    fn is_terminal(&self, event: &Self::Event) -> bool;

    fn finish(&mut self) -> Result<Vec<Self::Event>, Error>;
}

/// Established generic SSE stream returned by the execution kernel.
///
/// Setup failures are returned by [`execute_sse`]. After establishment, the stream emits decoded
/// events or one final error and then EOF. Dropping it cancels only the child operation created for
/// this call, not the caller's parent cancellation token.
pub struct SseStream<E> {
    inner: Pin<Box<dyn Stream<Item = Result<E, Error>> + Send + 'static>>,
    cancellation: Cancellation,
}

impl<E> fmt::Debug for SseStream<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SseStream")
            .field("is_cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl<E> Stream for SseStream<E> {
    type Item = Result<E, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

impl<E> Drop for SseStream<E> {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Execute one prepared direct call through the selected transport.
pub async fn execute_direct<D>(
    transport: &ProviderTransport,
    call: PreparedCall<D>,
    options: CallOptions,
) -> Result<D::Output, Error>
where
    D: DirectDecoder,
{
    let PreparedCall {
        target,
        headers,
        body,
        replay_safety,
        warnings,
        context,
        decoder,
    } = call;
    let plan = request_plan(target, headers, body, replay_safety, false, &context)?;
    let response = transport
        .execute(plan, options)
        .await
        .map_err(|error| contextualize(error, &context))?;
    if !response.status().is_success() {
        return Err(response_error(response, &context));
    }
    let response = DirectResponse {
        status: response.status(),
        headers: response.headers(),
        body: response.body(),
        attempts: response.attempts(),
        warnings: &warnings,
        error_context: context.error_context(),
    };
    decoder
        .decode(response)
        .map_err(|error| contextualize(error, &context))
}

/// Establish one prepared SSE call through the selected transport.
pub async fn execute_sse<D>(
    transport: &ProviderTransport,
    call: PreparedCall<D>,
    options: CallOptions,
) -> Result<SseStream<D::Event>, Error>
where
    D: SseStreamDecoder,
{
    let PreparedCall {
        target,
        headers,
        body,
        replay_safety,
        warnings,
        context,
        mut decoder,
    } = call;
    let operation_cancellation = options.cancellation().child();
    let options = options.with_cancellation(operation_cancellation.clone());
    let plan = request_plan(target, headers, body, replay_safety, true, &context)?;
    let response = transport
        .execute_stream(plan, options)
        .await
        .map_err(|error| contextualize(error, &context))?;
    if !response.status().is_success() {
        let error = stream_response_error(response, &context).await;
        operation_cancellation.cancel();
        return Err(error);
    }
    let attempts = response.attempts();
    let (status, headers, body) = response.into_parts();
    let response_context = StreamResponseContext {
        status,
        diagnostics: headers.diagnostics().with_status(status.as_u16()),
        headers,
        attempts,
        warnings: warnings.into(),
        error_context: context.error_context.clone(),
    };
    if let Err(error) = decoder.start(response_context) {
        operation_cancellation.cancel();
        return Err(contextualize(error, &context));
    }

    let limits = transport.limits().clone();
    let source = decode_sse(
        body,
        limits,
        decoder,
        context,
        operation_cancellation.clone(),
    );

    Ok(SseStream {
        inner: Box::pin(source),
        cancellation: operation_cancellation,
    })
}

fn decode_sse<S, B, D>(
    body: S,
    limits: TransportLimits,
    mut decoder: D,
    stream_context: ExecutionContext,
    cancellation: Cancellation,
) -> impl Stream<Item = Result<D::Event, Error>> + Send + 'static
where
    S: Stream<Item = Result<B, Error>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    D: SseStreamDecoder,
{
    async_stream::try_stream! {
        let mut body = Box::pin(body);
        let mut framing = SseDecoder::new(&limits);
        while let Some(chunk) = body.as_mut().next().await {
            let chunk = chunk.map_err(|error| contextualize(error, &stream_context))?;
            let frames = framing
                .push(chunk.as_ref())
                .map_err(|source| sse_error(source, &stream_context))?;
            let mut pending_events = Vec::new();
            let mut terminal_in_batch = false;
            let mut terminal_source_was_done = false;
            let mut done_after_terminal = false;
            for frame in frames {
                let is_done = frame.data().trim() == "[DONE]";
                if terminal_in_batch {
                    if is_done && !terminal_source_was_done && !done_after_terminal {
                        done_after_terminal = true;
                        continue;
                    }
                    Err(terminal_order_error(&stream_context))?;
                }
                let events = decoder
                    .decode(frame.data())
                    .map_err(|error| contextualize(error, &stream_context))?;
                if validate_event_batch(&decoder, &events, &stream_context)? {
                    terminal_in_batch = true;
                    terminal_source_was_done = is_done;
                }
                pending_events.extend(events);
            }
            if terminal_in_batch {
                framing
                    .finish()
                    .map_err(|source| sse_error(source, &stream_context))?;
                for event in pending_events {
                    if cancellation.is_cancelled() {
                        Err(stream_cancelled_error(&stream_context))?;
                    }
                    yield event;
                }
                return;
            }
            for event in pending_events {
                if cancellation.is_cancelled() {
                    Err(stream_cancelled_error(&stream_context))?;
                }
                yield event;
            }
        }
        framing
            .finish()
            .map_err(|source| sse_error(source, &stream_context))?;
        let events = decoder
            .finish()
            .map_err(|error| contextualize(error, &stream_context))?;
        let terminal = validate_event_batch(&decoder, &events, &stream_context)?;
        for event in events {
            if cancellation.is_cancelled() {
                Err(stream_cancelled_error(&stream_context))?;
            }
            yield event;
        }
        if terminal {
            return;
        }
        Err(Error::unexpected_eof().with_context(stream_context.error_context.clone()))?;
    }
}

fn stream_cancelled_error(context: &ExecutionContext) -> Error {
    Error::cancelled("call cancelled").with_context(context.error_context.clone())
}

fn request_plan(
    target: RequestTarget,
    mut headers: RequestHeaders,
    body: PreparedJsonBody,
    replay_safety: ReplaySafety,
    stream: bool,
    context: &ExecutionContext,
) -> Result<RequestPlan, Error> {
    let accept = if stream {
        HeaderValue::from_static("text/event-stream")
    } else {
        HeaderValue::from_static("application/json")
    };
    headers = headers
        .try_insert(ACCEPT, accept)
        .map_err(|source| request_contract_error(source, context))?;
    RequestPlan::new(Method::POST, target)
        .with_headers(headers)
        .with_body(body.into_request_body())
        .with_replay_safety(replay_safety)
        .map_err(|source| request_contract_error(source, context))
}

fn validate_event_batch<D>(
    decoder: &D,
    events: &[D::Event],
    context: &ExecutionContext,
) -> Result<bool, Error>
where
    D: SseStreamDecoder,
{
    let mut terminal_index = None;
    for (index, event) in events.iter().enumerate() {
        if !decoder.is_terminal(event) {
            continue;
        }
        if terminal_index.replace(index).is_some() {
            return Err(terminal_order_error(context));
        }
    }
    if terminal_index.is_some_and(|index| index + 1 != events.len()) {
        return Err(terminal_order_error(context));
    }
    Ok(terminal_index.is_some())
}

fn terminal_order_error(context: &ExecutionContext) -> Error {
    Error::new(ErrorKind::Protocol, context.invalid_sse_message)
        .with_context(context.error_context.clone())
}

fn contextualize(error: Error, context: &ExecutionContext) -> Error {
    error.with_context(context.error_context.clone())
}

fn request_contract_error(
    source: siumai_transport::RequestBuildError,
    context: &ExecutionContext,
) -> Error {
    Error::new(ErrorKind::InvalidInput, context.request_contract_message)
        .with_context(context.error_context.clone())
        .with_source(source)
}

fn sse_error(source: SseFrameError, context: &ExecutionContext) -> Error {
    let kind = match source {
        SseFrameError::FrameTooLarge
        | SseFrameError::EventTooLarge
        | SseFrameError::TooManyEvents => ErrorKind::ResponseLimit,
        SseFrameError::InvalidUtf8 | SseFrameError::UnexpectedEof => ErrorKind::Protocol,
        _ => ErrorKind::Protocol,
    };
    Error::new(kind, context.invalid_sse_message)
        .with_context(context.error_context.clone())
        .with_source(source)
}

fn response_error(response: TransportResponse, context: &ExecutionContext) -> Error {
    let (status, headers, body) = response.into_parts();
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    provider_status_error(status, headers, captured, truncated, context)
}

async fn stream_response_error(
    response: TransportStreamResponse,
    context: &ExecutionContext,
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
            Err(error) => return contextualize(error, context),
        }
    }
    provider_status_error(status, headers, bytes, truncated, context)
}

fn provider_status_error(
    status: StatusCode,
    headers: ResponseHeaders,
    body: Vec<u8>,
    body_truncated: bool,
    context: &ExecutionContext,
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
    Error::new(kind, context.rejected_response_message)
        .with_context(context.error_context.clone())
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

#[cfg(test)]
mod tests {
    use futures_util::stream;
    use serde_json::json;
    use siumai_core::{ErrorKind, WarningKind};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    enum TestEvent {
        Data(String),
        Terminal(String),
    }

    struct TestDecoder {
        finish_events: Vec<TestEvent>,
    }

    impl TestDecoder {
        fn new(finish_events: Vec<TestEvent>) -> Self {
            Self { finish_events }
        }
    }

    impl SseStreamDecoder for TestDecoder {
        type Event = TestEvent;

        fn decode(&mut self, data: &str) -> Result<Vec<Self::Event>, Error> {
            Ok(match data {
                "terminal" | "[DONE]" => vec![TestEvent::Terminal(data.to_string())],
                "terminal-extra" => vec![
                    TestEvent::Terminal("terminal".to_string()),
                    TestEvent::Data("extra".to_string()),
                ],
                "double-terminal" => vec![
                    TestEvent::Terminal("first".to_string()),
                    TestEvent::Terminal("second".to_string()),
                ],
                value => vec![TestEvent::Data(value.to_string())],
            })
        }

        fn is_terminal(&self, event: &Self::Event) -> bool {
            matches!(event, TestEvent::Terminal(_))
        }

        fn finish(&mut self) -> Result<Vec<Self::Event>, Error> {
            Ok(std::mem::take(&mut self.finish_events))
        }
    }

    fn context() -> ExecutionContext {
        ExecutionContext::new(
            ErrorContext::default(),
            "test request violates the transport contract",
            "test provider rejected the request",
            "test provider returned invalid SSE",
        )
    }

    #[tokio::test]
    async fn fragmented_sse_preserves_native_events_and_allows_done_after_terminal() {
        let body = stream::iter(vec![
            Ok::<_, Error>(b"data: fir".to_vec()),
            Ok(b"st\n\ndata: terminal\n\ndata: [DONE]\n\n".to_vec()),
        ]);
        let events = decode_sse(
            body,
            TransportLimits::default(),
            TestDecoder::new(Vec::new()),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0].as_ref().unwrap(),
            &TestEvent::Data("first".to_string())
        );
        assert_eq!(
            events[1].as_ref().unwrap(),
            &TestEvent::Terminal("terminal".to_string())
        );
    }

    #[tokio::test]
    async fn event_after_terminal_in_one_framing_batch_fails_atomically() {
        let body = stream::iter(vec![Ok::<_, Error>(
            b"data: terminal\n\ndata: later\n\n".to_vec(),
        )]);
        let events = decode_sse(
            body,
            TransportLimits::default(),
            TestDecoder::new(Vec::new()),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap_err().kind(), ErrorKind::Protocol);
    }

    #[tokio::test]
    async fn partial_frame_after_terminal_fails_before_terminal_publication() {
        let body = stream::iter(vec![Ok::<_, Error>(
            b"data: terminal\n\ndata: later".to_vec(),
        )]);
        let events = decode_sse(
            body,
            TransportLimits::default(),
            TestDecoder::new(Vec::new()),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap_err().kind(), ErrorKind::Protocol);
    }

    #[tokio::test]
    async fn repeated_done_terminal_or_done_suffix_is_rejected_atomically() {
        for body in [
            b"data: [DONE]\n\ndata: [DONE]\n\n".as_slice(),
            b"data: terminal\n\ndata: [DONE]\n\ndata: [DONE]\n\n".as_slice(),
        ] {
            let events = decode_sse(
                stream::iter(vec![Ok::<_, Error>(body.to_vec())]),
                TransportLimits::default(),
                TestDecoder::new(Vec::new()),
                context(),
                Cancellation::new(),
            )
            .collect::<Vec<_>>()
            .await;

            assert_eq!(events.len(), 1);
            assert_eq!(events[0].as_ref().unwrap_err().kind(), ErrorKind::Protocol);
        }
    }

    #[tokio::test]
    async fn decoder_terminal_followed_by_event_or_duplicate_terminal_is_rejected() {
        for data in ["terminal-extra", "double-terminal"] {
            let body = stream::iter(vec![Ok::<_, Error>(
                format!("data: {data}\n\n").into_bytes(),
            )]);
            let events = decode_sse(
                body,
                TransportLimits::default(),
                TestDecoder::new(Vec::new()),
                context(),
                Cancellation::new(),
            )
            .collect::<Vec<_>>()
            .await;

            assert_eq!(events.len(), 1, "fixture {data}");
            assert_eq!(
                events[0].as_ref().unwrap_err().kind(),
                ErrorKind::Protocol,
                "fixture {data}"
            );
        }
    }

    #[tokio::test]
    async fn decoder_finish_must_settle_or_is_followed_by_unexpected_eof() {
        let settled = decode_sse(
            stream::empty::<Result<Vec<u8>, Error>>(),
            TransportLimits::default(),
            TestDecoder::new(vec![TestEvent::Terminal("finish".to_string())]),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(settled.len(), 1);
        assert_eq!(
            settled[0].as_ref().unwrap(),
            &TestEvent::Terminal("finish".to_string())
        );

        let incomplete = decode_sse(
            stream::empty::<Result<Vec<u8>, Error>>(),
            TransportLimits::default(),
            TestDecoder::new(vec![TestEvent::Data("finish".to_string())]),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(incomplete.len(), 2);
        assert_eq!(
            incomplete[0].as_ref().unwrap(),
            &TestEvent::Data("finish".to_string())
        );
        assert_eq!(
            incomplete[1].as_ref().unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );

        let invalid = decode_sse(
            stream::empty::<Result<Vec<u8>, Error>>(),
            TransportLimits::default(),
            TestDecoder::new(vec![
                TestEvent::Terminal("finish".to_string()),
                TestEvent::Data("extra".to_string()),
            ]),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(invalid.len(), 1);
        assert_eq!(invalid[0].as_ref().unwrap_err().kind(), ErrorKind::Protocol);
    }

    #[tokio::test]
    async fn malformed_utf8_is_one_protocol_error() {
        let body = stream::iter(vec![Ok::<_, Error>(vec![
            b'd', b'a', b't', b'a', b':', b' ', 0xff, b'\n', b'\n',
        ])]);
        let events = decode_sse(
            body,
            TransportLimits::default(),
            TestDecoder::new(Vec::new()),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap_err().kind(), ErrorKind::Protocol);
    }

    #[tokio::test]
    async fn sse_resource_limits_map_to_response_limit() {
        let limits = TransportLimits {
            max_event_bytes: 8,
            ..TransportLimits::default()
        };
        let body = stream::iter(vec![Ok::<_, Error>(
            b"data: payload-that-is-too-large\n\n".to_vec(),
        )]);
        let events = decode_sse(
            body,
            limits,
            TestDecoder::new(Vec::new()),
            context(),
            Cancellation::new(),
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].as_ref().unwrap_err().kind(),
            ErrorKind::ResponseLimit
        );
    }

    #[tokio::test]
    async fn cancellation_preempts_buffered_event_from_the_same_chunk() {
        let parent = Cancellation::new();
        let operation = parent.child();
        let body = stream::iter(vec![Ok::<_, Error>(
            b"data: first\n\ndata: terminal\n\n".to_vec(),
        )]);
        let mut events = Box::pin(decode_sse(
            body,
            TransportLimits::default(),
            TestDecoder::new(Vec::new()),
            context(),
            operation,
        ));

        assert_eq!(
            events.next().await.unwrap().unwrap(),
            TestEvent::Data("first".to_string())
        );
        parent.cancel();

        let error = events.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(events.next().await.is_none());
    }

    #[test]
    fn dropping_generic_stream_cancels_only_its_child() {
        let parent = Cancellation::new();
        let child = parent.child();
        let observed_child = child.clone();
        let stream = SseStream::<TestEvent> {
            inner: Box::pin(stream::pending()),
            cancellation: child,
        };

        drop(stream);

        assert!(observed_child.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn prepared_call_debug_redacts_body_headers_warnings_and_messages() {
        let transport = ProviderTransport::builder(
            siumai_transport::EndpointConfig::local_explicit("http://127.0.0.1:1").unwrap(),
        )
        .build()
        .unwrap();
        let headers = RequestHeaders::new()
            .try_insert(
                http::header::HeaderName::from_static("x-provider-canary"),
                HeaderValue::from_static("header-secret-canary"),
            )
            .unwrap();
        let call = PreparedCall::new(
            RequestTarget::new("responses").unwrap(),
            PreparedJsonBody::new(&transport, &json!({"secret": "body-secret-canary"})).unwrap(),
            ReplaySafety::Never,
            (),
            ExecutionContext::new(
                ErrorContext::default(),
                "request-message-canary",
                "response-message-canary",
                "sse-message-canary",
            ),
        )
        .with_headers(headers)
        .with_warnings(vec![Warning::new(
            WarningKind::IgnoredOption,
            "warning-secret-canary",
        )]);

        let debug = format!("{call:?}");
        for secret in [
            "header-secret-canary",
            "body-secret-canary",
            "warning-secret-canary",
            "request-message-canary",
            "response-message-canary",
            "sse-message-canary",
        ] {
            assert!(!debug.contains(secret), "debug leaked {secret}");
        }
        assert!(debug.contains("warning_count: 1"));
    }
}
