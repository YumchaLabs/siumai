//! Persistent OpenAI Responses WebSocket sessions.

use std::collections::BTreeSet;
use std::error::Error as StdError;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::Stream;
use siumai_core::experimental::{
    SessionCloseMetadata, SessionCloseRequest, SessionFailure, SessionLineageId, SessionTerminal,
};
use siumai_core::{
    CallOptions, Cancellation, Error, ErrorContext, ErrorKind, LanguageRequest,
    LanguageStreamEvent, Model, ModelOperation, ProviderScope, PublicDiagnosticText,
    StreamTerminal,
};
use siumai_protocol_openai::responses::{
    DecodedResponsesStreamFrame, ResponseWire, ResponsesReplayStatus, ResponsesStreamDecoder,
    ResponsesStreamEvent,
};
use siumai_transport::framing::WebSocketFrame;
use siumai_transport::{
    RequestHeaders, WebSocketEndpoint, WebSocketReceiver, WebSocketSender, WebSocketTransport,
};
use thiserror::Error as ThisError;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::{AbortHandle, JoinError, JoinHandle};
use uuid::Uuid;

use super::language_execution::{contextualize_terminal_error, model_error_context};
use super::mode::OpenAiApiMode;
use super::model::OpenAiResponsesModel;
use super::{effective_deadline, wait_for_deadline};

/// Current provider-owned endpoint for persistent Responses sessions.
pub const OPENAI_RESPONSES_WEBSOCKET_URL: &str = "wss://api.openai.com/v1/responses";

const MAX_SESSION_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_COMMAND_QUEUE_CAPACITY: usize = 16;
const DEFAULT_TURN_EVENT_QUEUE_CAPACITY: usize = 128;
const MAX_QUEUE_CAPACITY: usize = 4096;
const MAX_ACTOR_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub(crate) struct OpenAiResponsesWebSocketRuntime {
    transport: WebSocketTransport,
    turn_timeout: Duration,
}

impl OpenAiResponsesWebSocketRuntime {
    pub(crate) fn new(
        transport: WebSocketTransport,
        turn_timeout: Option<Duration>,
    ) -> Result<Self, OpenAiResponsesWebSocketConfigError> {
        let turn_timeout =
            turn_timeout.unwrap_or(DEFAULT_TURN_TIMEOUT.min(transport.session_timeout()));
        validate_timeout("turn timeout", turn_timeout)?;
        if transport.session_timeout() > MAX_SESSION_TIMEOUT {
            return Err(OpenAiResponsesWebSocketConfigError::SessionTimeoutTooLarge);
        }
        if turn_timeout > transport.session_timeout() {
            return Err(OpenAiResponsesWebSocketConfigError::TurnTimeoutExceedsSession);
        }
        Ok(Self {
            transport,
            turn_timeout,
        })
    }
}

/// Invalid static configuration for a persistent Responses WebSocket session.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum OpenAiResponsesWebSocketConfigError {
    #[error(
        "Responses WebSocket is not configured; custom HTTP providers require an explicit WebSocket endpoint"
    )]
    EndpointNotConfigured,
    #[error("Responses WebSocket queue capacity must be between 1 and {MAX_QUEUE_CAPACITY}")]
    InvalidQueueCapacity,
    #[error("Responses WebSocket timeout must be non-zero and representable")]
    InvalidTimeout,
    #[error("Responses WebSocket sessions cannot exceed 60 minutes")]
    SessionTimeoutTooLarge,
    #[error("Responses WebSocket turn timeout cannot exceed the session timeout")]
    TurnTimeoutExceedsSession,
}

struct ConnectRequest {
    transport: WebSocketTransport,
    headers: RequestHeaders,
    options: CallOptions,
}

#[async_trait]
trait SocketSenderAdapter: Send + 'static {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error>;
    async fn close(&mut self) -> Result<(), Error>;
}

#[async_trait]
trait SocketReceiverAdapter: Send + 'static {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error>;
}

#[async_trait]
impl SocketSenderAdapter for WebSocketSender {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        WebSocketSender::send(self, frame).await
    }

    async fn close(&mut self) -> Result<(), Error> {
        WebSocketSender::close(self).await
    }
}

#[async_trait]
impl SocketReceiverAdapter for WebSocketReceiver {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        WebSocketReceiver::next(self).await
    }
}

struct SocketAdapter {
    sender: Box<dyn SocketSenderAdapter>,
    receiver: Box<dyn SocketReceiverAdapter>,
}

impl SocketAdapter {
    fn new<S, R>(sender: S, receiver: R) -> Self
    where
        S: SocketSenderAdapter,
        R: SocketReceiverAdapter,
    {
        Self {
            sender: Box::new(sender),
            receiver: Box::new(receiver),
        }
    }

    fn into_parts(self) -> (Box<dyn SocketSenderAdapter>, Box<dyn SocketReceiverAdapter>) {
        (self.sender, self.receiver)
    }
}

impl fmt::Debug for SocketAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SocketAdapter")
            .finish_non_exhaustive()
    }
}

#[async_trait]
trait SessionConnector: Send + Sync + 'static {
    async fn connect(&self, request: ConnectRequest) -> Result<SocketAdapter, Error>;
}

#[derive(Debug, Clone, Copy, Default)]
struct TransportSessionConnector;

#[async_trait]
impl SessionConnector for TransportSessionConnector {
    async fn connect(&self, request: ConnectRequest) -> Result<SocketAdapter, Error> {
        let connection = request
            .transport
            .connect(request.headers, request.options)
            .await?;
        let (sender, receiver) = connection.split();
        Ok(SocketAdapter::new(sender, receiver))
    }
}

/// Construction configuration for one persistent Responses WebSocket lineage.
#[derive(Clone)]
pub struct OpenAiResponsesWebSocketConfig {
    model: OpenAiResponsesModel,
    transport: WebSocketTransport,
    turn_timeout: Duration,
    command_queue_capacity: usize,
    turn_event_queue_capacity: usize,
    connector: Arc<dyn SessionConnector>,
}

impl OpenAiResponsesWebSocketConfig {
    pub(crate) fn from_model(
        model: OpenAiResponsesModel,
    ) -> Result<Self, OpenAiResponsesWebSocketConfigError> {
        let runtime = model
            .runtime
            .responses_websocket
            .clone()
            .ok_or(OpenAiResponsesWebSocketConfigError::EndpointNotConfigured)?;
        Ok(Self {
            model,
            transport: runtime.transport,
            turn_timeout: runtime.turn_timeout,
            command_queue_capacity: DEFAULT_COMMAND_QUEUE_CAPACITY,
            turn_event_queue_capacity: DEFAULT_TURN_EVENT_QUEUE_CAPACITY,
            connector: Arc::new(TransportSessionConnector),
        })
    }

    pub fn model(&self) -> &OpenAiResponsesModel {
        &self.model
    }

    pub fn endpoint(&self) -> &WebSocketEndpoint {
        self.transport.endpoint()
    }

    pub fn with_turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = timeout;
        self
    }

    pub fn with_command_queue_capacity(mut self, capacity: usize) -> Self {
        self.command_queue_capacity = capacity;
        self
    }

    pub fn with_turn_event_queue_capacity(mut self, capacity: usize) -> Self {
        self.turn_event_queue_capacity = capacity;
        self
    }

    #[cfg(test)]
    fn with_test_transport(mut self, connector: Arc<dyn SessionConnector>) -> Self {
        self.connector = connector;
        self
    }

    pub fn validate(&self) -> Result<(), OpenAiResponsesWebSocketConfigError> {
        validate_queue_capacity(self.command_queue_capacity)?;
        validate_queue_capacity(self.turn_event_queue_capacity)?;
        validate_timeout("turn timeout", self.turn_timeout)?;
        if self.transport.session_timeout() > MAX_SESSION_TIMEOUT {
            return Err(OpenAiResponsesWebSocketConfigError::SessionTimeoutTooLarge);
        }
        if self.turn_timeout > self.transport.session_timeout() {
            return Err(OpenAiResponsesWebSocketConfigError::TurnTimeoutExceedsSession);
        }
        Ok(())
    }

    /// Open one new persistent connection. Failed opens are never retried implicitly.
    pub async fn connect(
        &self,
        options: CallOptions,
    ) -> Result<OpenAiResponsesWebSocketSession, Error> {
        self.validate().map_err(configuration_error)?;
        if options.has_provider_options() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI Responses WebSocket connect does not accept language-model provider options",
            ));
        }
        let lineage_id =
            SessionLineageId::new(format!("openai-responses-ws:{}", Uuid::new_v4().simple()))
                .map_err(|source| {
                    Error::new(
                        ErrorKind::Internal,
                        "failed to construct an OpenAI Responses WebSocket lineage ID",
                    )
                    .with_source(source)
                })?;
        let model = Arc::<str>::from(self.model.model_id().as_str());
        let session_cancellation = options.cancellation().clone();
        let session_deadline =
            effective_deadline(options.deadline(), self.transport.session_timeout());
        let request = ConnectRequest {
            transport: self.transport.clone(),
            headers: RequestHeaders::new(),
            options,
        };
        let socket = self.connector.connect(request).await?;
        let (sender, receiver) = socket.into_parts();
        let lifecycle = Arc::new(ActorLifecycle::new());
        let (commands, command_rx) = mpsc::channel(self.command_queue_capacity);
        let scope = self.model.runtime.scope_arc(OpenAiApiMode::Responses);
        let actor = SessionActor {
            sender,
            receiver,
            commands: command_rx,
            lifecycle: lifecycle.clone(),
            scope,
            model: self.model.clone(),
            session_cancellation: session_cancellation.clone(),
            session_deadline,
            max_event_bytes: self.transport.limits().max_event_bytes,
            max_settled_response_ids: self.transport.limits().max_events_per_stream,
            settled_response_ids: BTreeSet::new(),
            active: None,
        };
        lifecycle.attach(tokio::spawn(actor.run()));
        Ok(OpenAiResponsesWebSocketSession {
            inner: Arc::new(SessionHandle {
                lineage_id,
                model,
                model_handle: self.model.clone(),
                commands,
                lifecycle,
                turn_timeout: self.turn_timeout,
                session_deadline,
                session_cancellation,
                turn_event_queue_capacity: self.turn_event_queue_capacity,
                next_turn_id: AtomicU64::new(1),
            }),
        })
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWebSocketConfig")
            .field("model", self.model.model_id())
            .field("endpoint", self.transport.endpoint())
            .field("turn_timeout", &self.turn_timeout)
            .field("command_queue_capacity", &self.command_queue_capacity)
            .field("turn_event_queue_capacity", &self.turn_event_queue_capacity)
            .field("connector", &"configured")
            .finish()
    }
}

/// Whether one turn generates output or only warms provider state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiResponsesWebSocketTurnKind {
    Generate,
    WarmUp,
}

/// What the client can prove about one WebSocket turn's submission.
///
/// `NotSubmitted` is safe to retry because the payload was proven not to reach
/// the socket sender. `Indeterminate` means the command entered the session
/// control path or the socket sender was polled, so replay requires caller
/// policy. `Settled` is recorded only after an authoritative provider terminal
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenAiResponsesWebSocketSubmissionState {
    NotSubmitted,
    Indeterminate,
    Settled,
}

impl OpenAiResponsesWebSocketSubmissionState {
    /// Read submission certainty from an error returned by this session API.
    pub fn from_error(error: &Error) -> Option<Self> {
        error
            .sensitive_source()
            .and_then(|source| source.expose().downcast_ref::<SubmissionStateErrorSource>())
            .map(|source| source.state)
    }
}

struct SubmissionStateErrorSource {
    state: OpenAiResponsesWebSocketSubmissionState,
    source: Error,
}

impl fmt::Debug for SubmissionStateErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubmissionStateErrorSource")
            .field("state", &self.state)
            .field("source", &"[REDACTED]")
            .finish()
    }
}

impl fmt::Display for SubmissionStateErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Responses WebSocket submission is {:?}",
            self.state
        )
    }
}

impl StdError for SubmissionStateErrorSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source)
    }
}

/// Native-only settlement for a `generate: false` warm-up.
#[derive(Debug)]
#[non_exhaustive]
pub enum OpenAiResponsesWarmUpOutcome {
    Completed {
        response: ResponseWire,
    },
    Failed {
        error: Error,
        response: Option<ResponseWire>,
    },
    Cancelled {
        response: Option<ResponseWire>,
    },
}

/// One native warm-up frame. Portable generation output is intentionally suppressed.
pub struct OpenAiResponsesWarmUpFrame {
    native: ResponsesStreamEvent,
    outcome: Option<OpenAiResponsesWarmUpOutcome>,
    replay_status: ResponsesReplayStatus,
}

impl OpenAiResponsesWarmUpFrame {
    pub fn native(&self) -> &ResponsesStreamEvent {
        &self.native
    }

    pub fn outcome(&self) -> Option<&OpenAiResponsesWarmUpOutcome> {
        self.outcome.as_ref()
    }

    /// Return whether the native warm-up result remains safe to replay.
    pub const fn replay_status(&self) -> ResponsesReplayStatus {
        self.replay_status
    }

    pub fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }
}

impl fmt::Debug for OpenAiResponsesWarmUpFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWarmUpFrame")
            .field("native", &self.native)
            .field("terminal", &self.is_terminal())
            .field("replay_status", &self.replay_status)
            .finish()
    }
}

/// One event from a generated or warm-up turn.
#[non_exhaustive]
pub enum OpenAiResponsesWebSocketEvent {
    Generated(super::responses_native::OpenAiResponsesStreamFrame),
    WarmUp(OpenAiResponsesWarmUpFrame),
}

impl OpenAiResponsesWebSocketEvent {
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Generated(frame) => frame.is_terminal(),
            Self::WarmUp(frame) => frame.is_terminal(),
        }
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generated(frame) => formatter.debug_tuple("Generated").field(frame).finish(),
            Self::WarmUp(frame) => formatter.debug_tuple("WarmUp").field(frame).finish(),
        }
    }
}

/// Stream handle for exactly one WebSocket turn.
pub struct OpenAiResponsesWebSocketTurn {
    id: u64,
    kind: OpenAiResponsesWebSocketTurnKind,
    receiver: mpsc::Receiver<Result<OpenAiResponsesWebSocketEvent, Error>>,
    lifecycle: Arc<TurnLifecycle>,
    cancellation: Cancellation,
    _session: Arc<SessionHandle>,
}

impl OpenAiResponsesWebSocketTurn {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub const fn kind(&self) -> OpenAiResponsesWebSocketTurnKind {
        self.kind
    }

    /// Return the latest safe submission classification for this turn.
    pub fn submission_state(&self) -> OpenAiResponsesWebSocketSubmissionState {
        self.lifecycle.submission_state()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl Stream for OpenAiResponsesWebSocketTurn {
    type Item = Result<OpenAiResponsesWebSocketEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.receiver).poll_recv(context) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => Poll::Ready(self.lifecycle.consumer_channel_closed().map(Err)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketTurn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWebSocketTurn")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("terminal", &self.lifecycle.has_terminal())
            .finish()
    }
}

impl Drop for OpenAiResponsesWebSocketTurn {
    fn drop(&mut self) {
        if !self.lifecycle.has_terminal() {
            self.cancellation.cancel();
        }
    }
}

/// Concurrent handle for one persistent Responses WebSocket lineage.
#[derive(Clone)]
pub struct OpenAiResponsesWebSocketSession {
    inner: Arc<SessionHandle>,
}

impl OpenAiResponsesWebSocketSession {
    pub fn lineage_id(&self) -> &SessionLineageId {
        &self.inner.lineage_id
    }

    pub fn model(&self) -> &str {
        self.inner.model.as_ref()
    }

    pub fn terminal(&self) -> Option<SessionTerminal> {
        self.inner.lifecycle.terminal()
    }

    pub async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesWebSocketTurn, Error> {
        self.start(OpenAiResponsesWebSocketTurnKind::Generate, request, options)
            .await
    }

    pub async fn warm_up(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesWebSocketTurn, Error> {
        self.start(OpenAiResponsesWebSocketTurnKind::WarmUp, request, options)
            .await
    }

    pub async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        validate_close_request(&request)?;
        if let Some(terminal) = self.terminal() {
            wait_for_actor_cleanup(&self.inner.lifecycle, self.inner.turn_timeout).await;
            return Ok(terminal);
        }
        let (ack, response) = oneshot::channel();
        let deadline = effective_deadline(None, self.inner.turn_timeout);
        let command = ActorCommand::Close { request, ack };
        let permit = tokio::select! {
            biased;
            terminal = self.inner.lifecycle.wait_for_terminal() => {
                wait_for_actor_cleanup(&self.inner.lifecycle, self.inner.turn_timeout).await;
                return Ok(terminal);
            }
            _ = wait_for_deadline(deadline) => {
                self.inner.lifecycle.abort_actor();
                return Err(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket close command deadline elapsed",
                ));
            }
            result = self.inner.commands.reserve() => {
                result.map_err(|_| session_closed_error())?
            }
        };
        permit.send(command);
        let terminal = tokio::select! {
            biased;
            terminal = self.inner.lifecycle.wait_for_terminal() => terminal,
            _ = wait_for_deadline(deadline) => {
                self.inner.lifecycle.abort_actor();
                return Err(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket close acknowledgement deadline elapsed",
                ));
            }
            result = response => match result {
                Ok(result) => result?,
                Err(_) => self.terminal().ok_or_else(actor_stopped_error)?,
            }
        };
        wait_for_actor_cleanup(&self.inner.lifecycle, self.inner.turn_timeout).await;
        Ok(terminal)
    }

    async fn start(
        &self,
        kind: OpenAiResponsesWebSocketTurnKind,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesWebSocketTurn, Error> {
        if let Some(terminal) = self.terminal() {
            return Err(with_submission_state(
                session_terminal_error(terminal),
                OpenAiResponsesWebSocketSubmissionState::NotSubmitted,
            ));
        }
        let mut cancellation = CancelOnDrop::new();
        let caller_cancellation = options.cancellation().child();
        let deadline = effective_deadline(options.deadline(), self.inner.turn_timeout);
        let (turn_lifecycle, receiver) = TurnLifecycle::new(self.inner.turn_event_queue_capacity);
        let control = StartCallControl {
            turn_cancellation: cancellation.cancellation().clone(),
            caller_cancellation: caller_cancellation.clone(),
            turn_deadline: deadline,
            session_deadline: self.inner.session_deadline,
            session_cancellation: self.inner.session_cancellation.clone(),
            lifecycle: self.inner.lifecycle.clone(),
        };
        if let Some(signal) = control.immediate_signal() {
            return Err(control_error(
                signal,
                SubmissionPhase::Queue,
                &turn_lifecycle,
            ));
        }
        let generate = kind == OpenAiResponsesWebSocketTurnKind::Generate;
        let scope = self
            .inner
            .model_handle
            .runtime
            .scope(OpenAiApiMode::Responses);
        let prepared = self
            .inner
            .model_handle
            .prepare_websocket_call(scope, request, &options, generate)
            .map_err(|error| turn_lifecycle.classify_error(error))?;
        let payload = serde_json::to_string(&prepared.body)
            .map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "failed to serialize an OpenAI Responses WebSocket command",
                )
                .with_source(source)
            })
            .map_err(|error| turn_lifecycle.classify_error(error))?;
        let id = self.inner.next_turn_id.fetch_add(1, Ordering::Relaxed);
        let (ack, response) = oneshot::channel();
        let command = ActorCommand::Start(StartCommand {
            kind,
            payload,
            lifecycle: turn_lifecycle.clone(),
            queue_guard: QueuedStartGuard::new(turn_lifecycle.clone()),
            cancellation: cancellation.cancellation().clone(),
            caller_cancellation,
            deadline,
            ack,
        });
        let permit = tokio::select! {
            biased;
            signal = control.wait() => {
                return Err(control_error(
                    signal,
                    SubmissionPhase::Queue,
                    &turn_lifecycle,
                ));
            }
            result = self.inner.commands.reserve() => {
                match result {
                    Ok(permit) => permit,
                    Err(_) => {
                        let error = self
                            .terminal()
                            .map(session_terminal_error)
                            .unwrap_or_else(actor_stopped_error);
                        return Err(with_submission_state(
                            error,
                            OpenAiResponsesWebSocketSubmissionState::NotSubmitted,
                        ));
                    }
                }
            }
        };
        permit.send(command);
        turn_lifecycle.mark_control_accepted();
        tokio::select! {
            biased;
            signal = control.wait() => {
                // Cancellation is the only control signal that should inject a new
                // cancellation into the actor. For a deadline, leave the command's
                // original deadline visible so the actor classifies the session as
                // timed out rather than observing our cleanup cancellation as a
                // different outcome. The local guard is disarmed because dropping it
                // would otherwise cancel the same token before the actor can observe
                // the deadline branch.
                if !matches!(signal, CallControlSignal::TurnCancelled) {
                    let _ = cancellation.disarm();
                }
                return Err(control_error(
                    signal,
                    SubmissionPhase::Acknowledgement,
                    &turn_lifecycle,
                ));
            }
            result = response => match result {
                Ok(result) => result?,
                Err(_) => {
                    let error = self
                        .terminal()
                        .map(session_terminal_error)
                        .unwrap_or_else(actor_stopped_error);
                    return Err(with_submission_state(
                        error,
                        turn_lifecycle.submission_state(),
                    ));
                }
            }
        }
        Ok(OpenAiResponsesWebSocketTurn {
            id,
            kind,
            receiver,
            lifecycle: turn_lifecycle,
            cancellation: cancellation.disarm(),
            _session: self.inner.clone(),
        })
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWebSocketSession")
            .field("lineage_id", &self.inner.lineage_id)
            .field("model", &self.inner.model)
            .field("terminal", &self.terminal().is_some())
            .finish()
    }
}

struct SessionHandle {
    lineage_id: SessionLineageId,
    model: Arc<str>,
    model_handle: OpenAiResponsesModel,
    commands: mpsc::Sender<ActorCommand>,
    lifecycle: Arc<ActorLifecycle>,
    turn_timeout: Duration,
    session_deadline: Option<Instant>,
    session_cancellation: Cancellation,
    turn_event_queue_capacity: usize,
    next_turn_id: AtomicU64,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.lifecycle.abort_actor();
    }
}

#[derive(Clone)]
struct StartCallControl {
    turn_cancellation: Cancellation,
    caller_cancellation: Cancellation,
    turn_deadline: Option<Instant>,
    session_deadline: Option<Instant>,
    session_cancellation: Cancellation,
    lifecycle: Arc<ActorLifecycle>,
}

impl StartCallControl {
    fn immediate_signal(&self) -> Option<CallControlSignal> {
        if self.turn_cancellation.is_cancelled() || self.caller_cancellation.is_cancelled() {
            return Some(CallControlSignal::TurnCancelled);
        }
        if self.session_cancellation.is_cancelled() {
            return Some(CallControlSignal::SessionCancelled);
        }
        let now = Instant::now();
        if self.turn_deadline.is_some_and(|deadline| deadline <= now) {
            return Some(CallControlSignal::TurnDeadline);
        }
        if self
            .session_deadline
            .is_some_and(|deadline| deadline <= now)
        {
            return Some(CallControlSignal::SessionDeadline);
        }
        self.lifecycle
            .terminal()
            .map(CallControlSignal::SessionTerminal)
    }

    async fn wait(&self) -> CallControlSignal {
        tokio::select! {
            biased;
            _ = self.turn_cancellation.cancelled() => CallControlSignal::TurnCancelled,
            _ = self.caller_cancellation.cancelled() => CallControlSignal::TurnCancelled,
            _ = self.session_cancellation.cancelled() => CallControlSignal::SessionCancelled,
            _ = wait_for_deadline(self.turn_deadline) => CallControlSignal::TurnDeadline,
            _ = wait_for_deadline(self.session_deadline) => CallControlSignal::SessionDeadline,
            terminal = self.lifecycle.wait_for_terminal() => CallControlSignal::SessionTerminal(terminal),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CallControlSignal {
    TurnCancelled,
    SessionCancelled,
    TurnDeadline,
    SessionDeadline,
    SessionTerminal(SessionTerminal),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmissionPhase {
    Queue,
    Acknowledgement,
}

struct ActorLifecycle {
    terminal: Mutex<Option<SessionTerminal>>,
    active_turn: Mutex<Option<Arc<TurnLifecycle>>>,
    terminal_notify: Notify,
    actor_abort: Mutex<Option<AbortHandle>>,
    actor_monitor: Mutex<Option<JoinHandle<()>>>,
    actor_finished: AtomicBool,
    actor_finished_notify: Notify,
}

impl ActorLifecycle {
    fn new() -> Self {
        Self {
            terminal: Mutex::new(None),
            active_turn: Mutex::new(None),
            terminal_notify: Notify::new(),
            actor_abort: Mutex::new(None),
            actor_monitor: Mutex::new(None),
            actor_finished: AtomicBool::new(false),
            actor_finished_notify: Notify::new(),
        }
    }

    fn attach(self: &Arc<Self>, actor: JoinHandle<()>) {
        *lock(&self.actor_abort) = Some(actor.abort_handle());
        let lifecycle = Arc::downgrade(self);
        let monitor = tokio::spawn(async move {
            let result = actor.await;
            if let Some(lifecycle) = lifecycle.upgrade() {
                lifecycle.observe_actor_exit(result);
            }
        });
        *lock(&self.actor_monitor) = Some(monitor);
    }

    fn terminal(&self) -> Option<SessionTerminal> {
        lock(&self.terminal).clone()
    }

    fn settle_terminal(&self, terminal: SessionTerminal) -> bool {
        let settled = {
            let mut slot = lock(&self.terminal);
            if slot.is_some() {
                false
            } else {
                *slot = Some(terminal);
                true
            }
        };
        if settled {
            self.terminal_notify.notify_waiters();
        }
        settled
    }

    async fn wait_for_terminal(&self) -> SessionTerminal {
        loop {
            let notified = self.terminal_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(terminal) = self.terminal() {
                return terminal;
            }
            notified.await;
        }
    }

    fn set_active_turn(&self, lifecycle: Arc<TurnLifecycle>) {
        *lock(&self.active_turn) = Some(lifecycle);
    }

    fn clear_active_turn(&self) {
        lock(&self.active_turn).take();
    }

    fn abort_actor(&self) {
        if let Some(actor) = lock(&self.actor_abort).as_ref() {
            actor.abort();
        }
    }

    async fn wait_for_actor(&self) {
        loop {
            let notified = self.actor_finished_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.actor_finished.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn observe_actor_exit(&self, result: Result<(), JoinError>) {
        if result.is_err() || self.terminal().is_none() {
            let error = Error::new(
                ErrorKind::UnexpectedEof,
                "OpenAI Responses WebSocket actor stopped before session settlement",
            );
            if self.settle_terminal(failed_terminal(
                ErrorKind::UnexpectedEof,
                "OpenAI Responses WebSocket actor stopped before session settlement",
            )) && let Some(turn) = lock(&self.active_turn).take()
            {
                turn.publish_failure(error);
            }
        }
        self.actor_finished.store(true, Ordering::Release);
        self.actor_finished_notify.notify_waiters();
    }
}

impl Drop for ActorLifecycle {
    fn drop(&mut self) {
        if let Some(actor) = self
            .actor_abort
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            actor.abort();
        }
        if let Some(monitor) = self
            .actor_monitor
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            monitor.abort();
        }
    }
}

enum ActorCommand {
    Start(StartCommand),
    Close {
        request: SessionCloseRequest,
        ack: oneshot::Sender<Result<SessionTerminal, Error>>,
    },
}

struct StartCommand {
    kind: OpenAiResponsesWebSocketTurnKind,
    payload: String,
    lifecycle: Arc<TurnLifecycle>,
    queue_guard: QueuedStartGuard,
    cancellation: Cancellation,
    caller_cancellation: Cancellation,
    deadline: Option<Instant>,
    ack: oneshot::Sender<Result<(), Error>>,
}

struct QueuedStartGuard {
    lifecycle: Arc<TurnLifecycle>,
    armed: bool,
}

impl QueuedStartGuard {
    fn new(lifecycle: Arc<TurnLifecycle>) -> Self {
        Self {
            lifecycle,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for QueuedStartGuard {
    fn drop(&mut self) {
        if self.armed {
            self.lifecycle.prove_not_submitted();
        }
    }
}

struct ActorTurnContext {
    kind: OpenAiResponsesWebSocketTurnKind,
    decoder: ResponsesStreamDecoder,
    context: ErrorContext,
    lifecycle: Arc<TurnLifecycle>,
    cancellation: Cancellation,
    caller_cancellation: Cancellation,
    deadline: Option<Instant>,
}

struct SessionActor {
    sender: Box<dyn SocketSenderAdapter>,
    receiver: Box<dyn SocketReceiverAdapter>,
    commands: mpsc::Receiver<ActorCommand>,
    lifecycle: Arc<ActorLifecycle>,
    scope: Arc<ProviderScope>,
    model: OpenAiResponsesModel,
    session_cancellation: Cancellation,
    session_deadline: Option<Instant>,
    max_event_bytes: usize,
    max_settled_response_ids: usize,
    settled_response_ids: BTreeSet<String>,
    active: Option<ActorTurnContext>,
}

impl SessionActor {
    async fn run(mut self) {
        loop {
            let terminal = if self.active.is_some() {
                self.run_active_once().await
            } else {
                self.run_ready_once().await
            };
            if let Some(terminal) = terminal {
                self.finish(terminal).await;
                return;
            }
        }
    }

    async fn run_ready_once(&mut self) -> Option<SessionTerminal> {
        tokio::select! {
            biased;
            _ = self.session_cancellation.cancelled() => Some(cancelled_terminal(
                "OpenAI Responses WebSocket session was cancelled",
            )),
            _ = wait_for_deadline(self.session_deadline) => Some(expired_terminal()),
            frame = self.receiver.receive() => self.handle_ready_frame(frame).await,
            command = self.commands.recv() => self.handle_ready_command(command).await,
        }
    }

    async fn run_active_once(&mut self) -> Option<SessionTerminal> {
        let active = self.active.as_ref().expect("active branch owns a turn");
        let cancellation = active.cancellation.clone();
        let caller_cancellation = active.caller_cancellation.clone();
        let deadline = active.deadline;
        tokio::select! {
            biased;
            _ = self.session_cancellation.cancelled() => {
                self.fail_active(Error::cancelled("OpenAI Responses WebSocket session was cancelled"));
                Some(cancelled_terminal("OpenAI Responses WebSocket session was cancelled"))
            }
            _ = cancellation.cancelled() => {
                self.fail_active(Error::cancelled("OpenAI Responses WebSocket turn was cancelled"));
                Some(cancelled_terminal("OpenAI Responses WebSocket turn was cancelled"))
            }
            _ = caller_cancellation.cancelled() => {
                self.fail_active(Error::cancelled("OpenAI Responses WebSocket turn was cancelled"));
                Some(cancelled_terminal("OpenAI Responses WebSocket turn was cancelled"))
            }
            _ = wait_for_deadline(deadline) => {
                self.fail_active(Error::new(ErrorKind::Timeout, "OpenAI Responses WebSocket turn deadline elapsed"));
                Some(failed_terminal(ErrorKind::Timeout, "OpenAI Responses WebSocket turn deadline elapsed"))
            }
            _ = wait_for_deadline(self.session_deadline) => {
                self.fail_active(Error::new(ErrorKind::Timeout, "OpenAI Responses WebSocket session deadline elapsed"));
                Some(expired_terminal())
            }
            command = self.commands.recv() => self.handle_active_command(command).await,
            frame = self.receiver.receive() => self.handle_active_frame(frame).await,
        }
    }

    async fn handle_ready_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Option<SessionTerminal> {
        match command {
            Some(ActorCommand::Start(command)) => self.start_turn(command).await,
            Some(ActorCommand::Close { request, ack }) => {
                let terminal = match self.send_close(&request).await {
                    Ok(()) => self.local_close_terminal(&request),
                    Err(error) => failed_terminal(
                        error.kind(),
                        "failed to close the OpenAI Responses WebSocket session",
                    ),
                };
                let _ = ack.send(Ok(terminal.clone()));
                Some(terminal)
            }
            None => Some(SessionTerminal::Closed(SessionCloseMetadata::local())),
        }
    }

    async fn handle_active_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Option<SessionTerminal> {
        match command {
            Some(ActorCommand::Start(command)) => {
                command.lifecycle.prove_not_submitted();
                let error = command.lifecycle.classify_error(Error::new(
                    ErrorKind::InvalidInput,
                    "an OpenAI Responses WebSocket turn is already active",
                ));
                let _ = command.ack.send(Err(error));
                None
            }
            Some(ActorCommand::Close { request, ack }) => {
                self.fail_active(Error::cancelled(
                    "OpenAI Responses WebSocket session closed during an active turn",
                ));
                let terminal = match self.send_close(&request).await {
                    Ok(()) => self.local_close_terminal(&request),
                    Err(error) => failed_terminal(
                        error.kind(),
                        "failed to close the OpenAI Responses WebSocket session",
                    ),
                };
                let _ = ack.send(Ok(terminal.clone()));
                Some(terminal)
            }
            None => {
                self.fail_active(Error::cancelled(
                    "OpenAI Responses WebSocket session handle was dropped",
                ));
                Some(cancelled_terminal(
                    "OpenAI Responses WebSocket session handle was dropped",
                ))
            }
        }
    }

    async fn start_turn(&mut self, mut command: StartCommand) -> Option<SessionTerminal> {
        command.queue_guard.disarm();
        if command.cancellation.is_cancelled() || command.caller_cancellation.is_cancelled() {
            command.lifecycle.prove_not_submitted();
            let error = command.lifecycle.classify_error(Error::cancelled(
                "OpenAI Responses WebSocket turn was cancelled before submission",
            ));
            let _ = command.ack.send(Err(error));
            return None;
        }
        if self.session_cancellation.is_cancelled() {
            command.lifecycle.prove_not_submitted();
            let error = command.lifecycle.classify_error(Error::cancelled(
                "OpenAI Responses WebSocket session was cancelled before submission",
            ));
            let _ = command.ack.send(Err(error));
            return Some(cancelled_terminal(
                "OpenAI Responses WebSocket session was cancelled",
            ));
        }
        let deadline = command.deadline;
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            command.lifecycle.prove_not_submitted();
            let error = command.lifecycle.classify_error(Error::new(
                ErrorKind::Timeout,
                "OpenAI Responses WebSocket turn deadline elapsed before submission",
            ));
            let _ = command.ack.send(Err(error));
            return None;
        }
        if self
            .session_deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            command.lifecycle.prove_not_submitted();
            let error = command.lifecycle.classify_error(Error::new(
                ErrorKind::Timeout,
                "OpenAI Responses WebSocket session deadline elapsed before submission",
            ));
            let _ = command.ack.send(Err(error));
            return Some(expired_terminal());
        }

        enum SubmissionOutcome {
            Sent(Result<(), Error>),
            SessionCancelled,
            TurnCancelled,
            TurnTimedOut,
            SessionTimedOut,
        }

        let session_cancellation = self.session_cancellation.clone();
        command.lifecycle.mark_sender_polled();
        self.lifecycle.set_active_turn(command.lifecycle.clone());
        let submission = tokio::select! {
            biased;
            _ = session_cancellation.cancelled() => SubmissionOutcome::SessionCancelled,
            _ = command.cancellation.cancelled() => SubmissionOutcome::TurnCancelled,
            _ = command.caller_cancellation.cancelled() => SubmissionOutcome::TurnCancelled,
            _ = wait_for_deadline(deadline) => SubmissionOutcome::TurnTimedOut,
            _ = wait_for_deadline(self.session_deadline) => SubmissionOutcome::SessionTimedOut,
            result = self.sender.send(WebSocketFrame::Text(command.payload)) => {
                SubmissionOutcome::Sent(result)
            }
        };
        let send_result = match submission {
            SubmissionOutcome::Sent(result) => result,
            SubmissionOutcome::SessionCancelled => {
                self.lifecycle.clear_active_turn();
                let error = command.lifecycle.classify_error(Error::cancelled(
                    "OpenAI Responses WebSocket session was cancelled during submission",
                ));
                let _ = command.ack.send(Err(error));
                return Some(cancelled_terminal(
                    "OpenAI Responses WebSocket session was cancelled",
                ));
            }
            SubmissionOutcome::TurnCancelled => {
                self.lifecycle.clear_active_turn();
                let error = command.lifecycle.classify_error(Error::cancelled(
                    "OpenAI Responses WebSocket turn was cancelled during submission",
                ));
                let _ = command.ack.send(Err(error));
                return Some(cancelled_terminal(
                    "OpenAI Responses WebSocket turn was cancelled during submission",
                ));
            }
            SubmissionOutcome::TurnTimedOut => {
                self.lifecycle.clear_active_turn();
                let error = command.lifecycle.classify_error(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket turn deadline elapsed during submission",
                ));
                let _ = command.ack.send(Err(error));
                return Some(failed_terminal(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket turn deadline elapsed during submission",
                ));
            }
            SubmissionOutcome::SessionTimedOut => {
                self.lifecycle.clear_active_turn();
                let error = command.lifecycle.classify_error(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket session deadline elapsed during submission",
                ));
                let _ = command.ack.send(Err(error));
                return Some(expired_terminal());
            }
        };
        if let Err(error) = send_result {
            let kind = error.kind();
            self.lifecycle.clear_active_turn();
            let _ = command
                .ack
                .send(Err(command.lifecycle.classify_error(error)));
            return Some(failed_terminal(
                kind,
                "failed to send an OpenAI Responses WebSocket turn",
            ));
        }
        let context = model_error_context(&self.model, ModelOperation::Stream);
        self.active = Some(ActorTurnContext {
            kind: command.kind,
            decoder: ResponsesStreamDecoder::new(
                self.scope.as_ref().clone(),
                self.model.model_id().clone(),
            )
            .with_wire_dialect(self.model.runtime.responses_wire_dialect),
            context,
            lifecycle: command.lifecycle,
            cancellation: command.cancellation,
            caller_cancellation: command.caller_cancellation,
            deadline,
        });
        if command.ack.send(Ok(())).is_err()
            && let Some(active) = &self.active
        {
            active.cancellation.cancel();
        }
        None
    }

    async fn handle_ready_frame(
        &mut self,
        frame: Result<Option<WebSocketFrame>, Error>,
    ) -> Option<SessionTerminal> {
        match frame {
            Ok(Some(WebSocketFrame::Ping(payload))) => {
                if let Err(error) = self.sender.send(WebSocketFrame::Pong(payload)).await {
                    return Some(failed_terminal(
                        error.kind(),
                        "failed to reply to an OpenAI Responses WebSocket ping",
                    ));
                }
                None
            }
            Ok(Some(WebSocketFrame::Pong(_))) => None,
            Ok(Some(WebSocketFrame::Close { code, reason })) => {
                Some(SessionTerminal::Closed(remote_close_metadata(code, reason)))
            }
            Ok(None) => Some(SessionTerminal::Closed(SessionCloseMetadata::remote())),
            Ok(Some(WebSocketFrame::Text(_))) | Ok(Some(WebSocketFrame::Binary(_))) => {
                Some(failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket emitted an event without an active turn",
                ))
            }
            Ok(Some(_)) => Some(failed_terminal(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket returned an unsupported frame",
            )),
            Err(error) => Some(failed_terminal(
                error.kind(),
                "OpenAI Responses WebSocket receive failed",
            )),
        }
    }

    async fn handle_active_frame(
        &mut self,
        frame: Result<Option<WebSocketFrame>, Error>,
    ) -> Option<SessionTerminal> {
        match frame {
            Ok(Some(WebSocketFrame::Ping(payload))) => {
                if let Err(error) = self.sender.send(WebSocketFrame::Pong(payload)).await {
                    self.fail_active(Error::new(
                        error.kind(),
                        "failed to reply to an OpenAI Responses WebSocket ping",
                    ));
                    return Some(failed_terminal(
                        error.kind(),
                        "failed to reply to an OpenAI Responses WebSocket ping",
                    ));
                }
                None
            }
            Ok(Some(WebSocketFrame::Pong(_))) => None,
            Ok(Some(WebSocketFrame::Text(text))) => self.decode_active_text(&text),
            Ok(Some(WebSocketFrame::Binary(_))) => {
                self.fail_active(Error::new(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket returned an unsupported binary frame",
                ));
                Some(failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket returned an unsupported binary frame",
                ))
            }
            Ok(Some(WebSocketFrame::Close { code, .. })) => {
                let (kind, message) = classify_unsettled_close(code);
                self.fail_active(Error::new(kind, message));
                Some(failed_terminal(kind, message))
            }
            Ok(None) => {
                self.fail_active(Error::unexpected_eof());
                Some(failed_terminal(
                    ErrorKind::UnexpectedEof,
                    "OpenAI Responses WebSocket closed before turn settlement",
                ))
            }
            Err(error) => {
                let kind = error.kind();
                self.fail_active(error);
                Some(failed_terminal(
                    kind,
                    "OpenAI Responses WebSocket receive failed during an active turn",
                ))
            }
            Ok(Some(_)) => {
                self.fail_active(Error::new(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket returned an unsupported frame",
                ));
                Some(failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket returned an unsupported frame",
                ))
            }
        }
    }

    fn decode_active_text(&mut self, text: &str) -> Option<SessionTerminal> {
        if text.len() > self.max_event_bytes {
            self.fail_active(Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI Responses WebSocket event exceeds the configured limit",
            ));
            return Some(failed_terminal(
                ErrorKind::ResponseLimit,
                "OpenAI Responses WebSocket event exceeds the configured limit",
            ));
        }
        let active = self.active.as_mut().expect("active text owns a turn");
        let decoded = match active.decoder.decode_native(text) {
            Ok(decoded) => decoded,
            Err(error) => {
                let error = error.with_context(active.context.clone());
                self.fail_active(error);
                return Some(failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket event violated the protocol",
                ));
            }
        };
        let canonical = decoded
            .is_terminal()
            .then(|| active.decoder.terminal_response().cloned())
            .flatten();
        let observed_response_id = decoded
            .native()
            .response_resource()
            .map(|response| response.id.clone())
            .or_else(|| canonical.as_ref().map(|response| response.id.clone()));
        if observed_response_id
            .as_deref()
            .is_some_and(|response_id| self.settled_response_ids.contains(response_id))
        {
            self.fail_active(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket repeated a settled response ID",
            ));
            return Some(failed_terminal(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket repeated a settled response ID",
            ));
        }
        let identity = active
            .lifecycle
            .validate_response_identity(&decoded, canonical.as_ref());
        if let Err(error) = identity {
            self.fail_active(error);
            return Some(failed_terminal(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket event violated turn identity",
            ));
        }
        if canonical.as_ref().is_some_and(|response| {
            self.settled_response_ids.len() >= self.max_settled_response_ids
                && !self.settled_response_ids.contains(response.id.as_str())
        }) {
            self.fail_active(Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI Responses WebSocket response identity ledger exceeds the configured limit",
            ));
            return Some(failed_terminal(
                ErrorKind::ResponseLimit,
                "OpenAI Responses WebSocket response identity ledger exceeds the configured limit",
            ));
        }
        let active = self.active.as_mut().expect("active text owns a turn");
        let session_fatal = is_session_fatal_provider_event(&decoded, canonical.as_ref());
        let terminal = decoded.is_terminal();
        let settled_response_id = canonical.as_ref().map(|response| response.id.clone());
        let event = match active.kind {
            OpenAiResponsesWebSocketTurnKind::Generate => {
                generated_event(decoded, canonical, &active.context)
            }
            OpenAiResponsesWebSocketTurnKind::WarmUp => {
                warm_up_event(decoded, canonical, &active.context)
            }
        };
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                self.fail_active(error);
                return Some(failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket terminal projection failed",
                ));
            }
        };
        if let Err(failure) = active.lifecycle.publish_event(event) {
            let terminal = match failure {
                EmitFailure::Closed => {
                    cancelled_terminal("OpenAI Responses WebSocket turn consumer was dropped")
                }
                EmitFailure::Saturated => failed_terminal(
                    ErrorKind::ResponseLimit,
                    "OpenAI Responses WebSocket turn queue was saturated",
                ),
                EmitFailure::AfterTerminal => failed_terminal(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket emitted an event after turn settlement",
                ),
            };
            self.active = None;
            self.lifecycle.clear_active_turn();
            return Some(terminal);
        }
        if terminal {
            if let Some(response_id) = settled_response_id {
                self.settled_response_ids.insert(response_id);
            }
            self.active = None;
            self.lifecycle.clear_active_turn();
            if session_fatal {
                return Some(failed_terminal(
                    ErrorKind::RateLimited,
                    "OpenAI Responses WebSocket connection limit was reached",
                ));
            }
        }
        None
    }

    fn fail_active(&mut self, mut error: Error) {
        let Some(active) = self.active.take() else {
            return;
        };
        self.lifecycle.clear_active_turn();
        if error.context().operation.is_none() {
            error = error.with_context(active.context.clone());
        }
        active.lifecycle.publish_failure(error);
    }

    async fn send_close(&mut self, request: &SessionCloseRequest) -> Result<(), Error> {
        self.sender
            .send(WebSocketFrame::Close {
                code: request.transport_code.map(|code| code as u16),
                reason: request
                    .reason
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
            })
            .await
    }

    fn local_close_terminal(&self, request: &SessionCloseRequest) -> SessionTerminal {
        let mut metadata = SessionCloseMetadata::local();
        if let Some(code) = request.transport_code {
            metadata = metadata.with_transport_code(code);
        }
        if let Some(reason) = &request.reason {
            metadata = metadata.with_reason(reason.clone());
        }
        SessionTerminal::Closed(metadata)
    }

    async fn finish(&mut self, terminal: SessionTerminal) {
        self.lifecycle.settle_terminal(terminal);
        let deadline = effective_deadline(None, MAX_ACTOR_CLEANUP_TIMEOUT);
        tokio::select! {
            biased;
            _ = wait_for_deadline(deadline) => {}
            _ = self.sender.close() => {}
        }
    }
}

fn generated_event(
    decoded: DecodedResponsesStreamFrame,
    canonical_terminal_response: Option<ResponseWire>,
    context: &ErrorContext,
) -> Result<OpenAiResponsesWebSocketEvent, Error> {
    let (native, mut portable_events, replay_status) = decoded.into_parts();
    for event in &mut portable_events {
        contextualize_terminal_error(event, context);
    }
    Ok(OpenAiResponsesWebSocketEvent::Generated(
        super::responses_native::OpenAiResponsesStreamFrame::new(
            native,
            portable_events,
            canonical_terminal_response,
            replay_status,
        ),
    ))
}

fn warm_up_event(
    decoded: DecodedResponsesStreamFrame,
    canonical_terminal_response: Option<ResponseWire>,
    context: &ErrorContext,
) -> Result<OpenAiResponsesWebSocketEvent, Error> {
    let (native, portable_events, replay_status) = decoded.into_parts();
    let mut outcome = None;
    for event in portable_events {
        if let LanguageStreamEvent::Terminal(mut terminal) = event {
            match &mut terminal {
                StreamTerminal::Failed { error, .. } => {
                    let original = std::mem::replace(
                        error,
                        Error::new(ErrorKind::Internal, "warm-up error replacement failed"),
                    );
                    *error = original.with_context(context.clone());
                }
                StreamTerminal::Completed { .. } | StreamTerminal::Cancelled { .. } => {}
                _ => {}
            }
            outcome = Some(match terminal {
                StreamTerminal::Completed { .. } => OpenAiResponsesWarmUpOutcome::Completed {
                    response: canonical_terminal_response.clone().ok_or_else(|| {
                        Error::new(
                            ErrorKind::Protocol,
                            "OpenAI Responses warm-up completed without a canonical response",
                        )
                    })?,
                },
                StreamTerminal::Failed { error, .. } => OpenAiResponsesWarmUpOutcome::Failed {
                    error,
                    response: canonical_terminal_response.clone(),
                },
                StreamTerminal::Cancelled { .. } => OpenAiResponsesWarmUpOutcome::Cancelled {
                    response: canonical_terminal_response.clone(),
                },
                _ => {
                    return Err(Error::new(
                        ErrorKind::Protocol,
                        "OpenAI Responses warm-up returned an unsupported terminal state",
                    ));
                }
            });
        }
    }
    Ok(OpenAiResponsesWebSocketEvent::WarmUp(
        OpenAiResponsesWarmUpFrame {
            native,
            outcome,
            replay_status,
        },
    ))
}

fn is_session_fatal_provider_event(
    decoded: &DecodedResponsesStreamFrame,
    canonical: Option<&ResponseWire>,
) -> bool {
    let native_code = decoded
        .native()
        .error()
        .and_then(|error| error.code.as_deref());
    let response_code = canonical
        .and_then(|response| response.error.as_ref())
        .and_then(|error| error.code.as_deref());
    [native_code, response_code]
        .into_iter()
        .flatten()
        .any(|code| code == "websocket_connection_limit_reached")
}

enum EmitFailure {
    Closed,
    Saturated,
    AfterTerminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnSubmissionPhase {
    Queued,
    ControlAccepted,
    SenderPolled,
    ProvenNotSubmitted,
    Settled,
}

impl TurnSubmissionPhase {
    fn public_state(self) -> OpenAiResponsesWebSocketSubmissionState {
        match self {
            Self::Queued | Self::ProvenNotSubmitted => {
                OpenAiResponsesWebSocketSubmissionState::NotSubmitted
            }
            Self::ControlAccepted | Self::SenderPolled => {
                OpenAiResponsesWebSocketSubmissionState::Indeterminate
            }
            Self::Settled => OpenAiResponsesWebSocketSubmissionState::Settled,
        }
    }
}

struct TurnLifecycleState {
    submission: TurnSubmissionPhase,
    response_id: Option<String>,
    delivery: TurnDeliveryState,
}

enum TurnDeliveryState {
    Open(mpsc::Sender<Result<OpenAiResponsesWebSocketEvent, Error>>),
    Selected { fallback: Option<Error> },
    Closed,
}

struct TurnLifecycle {
    state: Mutex<TurnLifecycleState>,
}

impl TurnLifecycle {
    fn new(
        event_queue_capacity: usize,
    ) -> (
        Arc<Self>,
        mpsc::Receiver<Result<OpenAiResponsesWebSocketEvent, Error>>,
    ) {
        let (events, receiver) = mpsc::channel(event_queue_capacity);
        (
            Arc::new(Self {
                state: Mutex::new(TurnLifecycleState {
                    submission: TurnSubmissionPhase::Queued,
                    response_id: None,
                    delivery: TurnDeliveryState::Open(events),
                }),
            }),
            receiver,
        )
    }

    fn submission_state(&self) -> OpenAiResponsesWebSocketSubmissionState {
        lock(&self.state).submission.public_state()
    }

    fn mark_control_accepted(&self) {
        let mut state = lock(&self.state);
        if state.submission == TurnSubmissionPhase::Queued {
            state.submission = TurnSubmissionPhase::ControlAccepted;
        }
    }

    fn mark_sender_polled(&self) {
        let mut state = lock(&self.state);
        if state.submission != TurnSubmissionPhase::Settled {
            state.submission = TurnSubmissionPhase::SenderPolled;
        }
    }

    fn prove_not_submitted(&self) {
        let mut state = lock(&self.state);
        if matches!(
            state.submission,
            TurnSubmissionPhase::Queued | TurnSubmissionPhase::ControlAccepted
        ) {
            state.submission = TurnSubmissionPhase::ProvenNotSubmitted;
        }
    }

    fn classify_error(&self, error: Error) -> Error {
        with_submission_state(error, self.submission_state())
    }

    fn has_terminal(&self) -> bool {
        !matches!(lock(&self.state).delivery, TurnDeliveryState::Open(_))
    }

    fn validate_response_identity(
        &self,
        decoded: &DecodedResponsesStreamFrame,
        canonical: Option<&ResponseWire>,
    ) -> Result<(), Error> {
        let mut state = lock(&self.state);
        if !matches!(state.delivery, TurnDeliveryState::Open(_)) {
            return Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket emitted an event after turn settlement",
            ));
        }
        if let Some(response) = decoded.native().response_resource() {
            if state
                .response_id
                .as_deref()
                .is_some_and(|existing| existing != response.id)
            {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "OpenAI Responses WebSocket turn changed its response ID",
                ));
            }
            state.response_id.get_or_insert_with(|| response.id.clone());
        }
        if let Some(response) = canonical
            && state.response_id.as_deref() != Some(response.id.as_str())
        {
            return Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket terminal did not match a response created for this turn",
            ));
        }
        Ok(())
    }

    fn publish_event(&self, event: OpenAiResponsesWebSocketEvent) -> Result<(), EmitFailure> {
        let terminal = event.is_terminal();
        let mut state = lock(&self.state);
        let sender = match &state.delivery {
            TurnDeliveryState::Open(sender) => sender.clone(),
            TurnDeliveryState::Selected { .. } | TurnDeliveryState::Closed => {
                return Err(EmitFailure::AfterTerminal);
            }
        };
        if terminal {
            state.submission = TurnSubmissionPhase::Settled;
        }
        let submission = state.submission.public_state();
        let result = match sender.try_send(Ok(event)) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(EmitFailure::Closed),
            Err(mpsc::error::TrySendError::Full(_)) => Err(EmitFailure::Saturated),
        };
        state.delivery = match result {
            Ok(()) if !terminal => return Ok(()),
            Ok(()) | Err(EmitFailure::Closed) => TurnDeliveryState::Selected { fallback: None },
            Err(EmitFailure::Saturated) => TurnDeliveryState::Selected {
                fallback: Some(with_submission_state(
                    Error::new(
                        ErrorKind::ResponseLimit,
                        "OpenAI Responses WebSocket turn queue was saturated",
                    ),
                    submission,
                )),
            },
            Err(EmitFailure::AfterTerminal) => unreachable!("delivery state was open"),
        };
        result
    }

    fn publish_failure(&self, error: Error) {
        let mut state = lock(&self.state);
        let sender = match &state.delivery {
            TurnDeliveryState::Open(sender) => sender.clone(),
            TurnDeliveryState::Selected { .. } | TurnDeliveryState::Closed => return,
        };
        let error = with_submission_state(error, state.submission.public_state());
        let fallback = match sender.try_send(Err(error)) {
            Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => None,
            Err(mpsc::error::TrySendError::Full(item)) => match item {
                Err(error) => Some(error),
                Ok(_) => unreachable!("failure publication always sends an error"),
            },
        };
        state.delivery = TurnDeliveryState::Selected { fallback };
    }

    fn consumer_channel_closed(&self) -> Option<Error> {
        let mut state = lock(&self.state);
        match std::mem::replace(&mut state.delivery, TurnDeliveryState::Closed) {
            TurnDeliveryState::Open(_) => Some(with_submission_state(
                Error::new(
                    ErrorKind::UnexpectedEof,
                    "OpenAI Responses WebSocket turn channel closed before settlement",
                ),
                state.submission.public_state(),
            )),
            TurnDeliveryState::Selected { fallback } => fallback,
            TurnDeliveryState::Closed => None,
        }
    }
}

struct CancelOnDrop {
    cancellation: Option<Cancellation>,
}

impl CancelOnDrop {
    fn new() -> Self {
        Self {
            cancellation: Some(Cancellation::new()),
        }
    }

    fn cancellation(&self) -> &Cancellation {
        self.cancellation
            .as_ref()
            .expect("start cancellation remains armed")
    }

    fn disarm(&mut self) -> Cancellation {
        self.cancellation
            .take()
            .expect("start cancellation is disarmed once")
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
        }
    }
}

fn control_error(
    signal: CallControlSignal,
    phase: SubmissionPhase,
    lifecycle: &TurnLifecycle,
) -> Error {
    match phase {
        SubmissionPhase::Queue => lifecycle.prove_not_submitted(),
        SubmissionPhase::Acknowledgement => {
            if !matches!(&signal, CallControlSignal::SessionTerminal(_)) {
                lifecycle.mark_control_accepted();
            }
        }
    }
    let error = match signal {
        CallControlSignal::TurnCancelled => Error::cancelled(
            "OpenAI Responses WebSocket turn was cancelled while awaiting submission",
        ),
        CallControlSignal::SessionCancelled => Error::cancelled(
            "OpenAI Responses WebSocket session was cancelled while awaiting submission",
        ),
        CallControlSignal::TurnDeadline => Error::new(
            ErrorKind::Timeout,
            "OpenAI Responses WebSocket turn deadline elapsed while awaiting submission",
        ),
        CallControlSignal::SessionDeadline => Error::new(
            ErrorKind::Timeout,
            "OpenAI Responses WebSocket session deadline elapsed while awaiting submission",
        ),
        CallControlSignal::SessionTerminal(terminal) => session_terminal_error(terminal),
    };
    lifecycle.classify_error(error)
}

fn with_submission_state(error: Error, state: OpenAiResponsesWebSocketSubmissionState) -> Error {
    let kind = error.kind();
    let context = error.context().clone();
    let detail = error.detail().cloned();
    let diagnostics = error.diagnostics().cloned();
    let message = PublicDiagnosticText::new(error.message().to_owned())
        .unwrap_or_else(|_| PublicDiagnosticText::from("Responses WebSocket call failed"));
    let mut classified = Error::new(kind, message).with_context(context);
    if let Some(detail) = detail {
        classified = classified.with_detail(detail);
    }
    if let Some(diagnostics) = diagnostics {
        classified = classified.with_diagnostics(diagnostics);
    }
    classified.with_source(SubmissionStateErrorSource {
        state,
        source: error,
    })
}

async fn wait_for_actor_cleanup(lifecycle: &ActorLifecycle, timeout: Duration) {
    let timeout = timeout.min(MAX_ACTOR_CLEANUP_TIMEOUT);
    let hard_deadline = effective_deadline(None, timeout);
    let abort_grace = timeout.min(Duration::from_secs(1));
    let graceful_deadline = hard_deadline.and_then(|deadline| deadline.checked_sub(abort_grace));
    let needs_abort = tokio::select! {
        biased;
        _ = lifecycle.wait_for_actor() => false,
        _ = wait_for_deadline(graceful_deadline) => true,
    };
    if !needs_abort {
        return;
    }
    lifecycle.abort_actor();
    tokio::select! {
        biased;
        _ = lifecycle.wait_for_actor() => {}
        _ = wait_for_deadline(hard_deadline) => {}
    }
}

fn validate_queue_capacity(capacity: usize) -> Result<(), OpenAiResponsesWebSocketConfigError> {
    if !(1..=MAX_QUEUE_CAPACITY).contains(&capacity) {
        return Err(OpenAiResponsesWebSocketConfigError::InvalidQueueCapacity);
    }
    Ok(())
}

fn validate_timeout(
    _name: &'static str,
    timeout: Duration,
) -> Result<(), OpenAiResponsesWebSocketConfigError> {
    if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
        return Err(OpenAiResponsesWebSocketConfigError::InvalidTimeout);
    }
    Ok(())
}

fn configuration_error(source: OpenAiResponsesWebSocketConfigError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "OpenAI Responses WebSocket configuration is invalid",
    )
    .with_source(source)
}

fn validate_close_request(request: &SessionCloseRequest) -> Result<(), Error> {
    if request
        .transport_code
        .is_some_and(|code| code > u16::MAX as u32)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "WebSocket close code exceeds the transport range",
        ));
    }
    Ok(())
}

fn remote_close_metadata(code: Option<u16>, reason: String) -> SessionCloseMetadata {
    let mut metadata = SessionCloseMetadata::remote();
    if let Some(code) = code {
        metadata = metadata.with_transport_code(u32::from(code));
    }
    if !reason.trim().is_empty()
        && let Ok(reason) = PublicDiagnosticText::new(reason)
    {
        metadata = metadata.with_reason(reason);
    }
    metadata
}

fn classify_unsettled_close(code: Option<u16>) -> (ErrorKind, &'static str) {
    match code {
        // Going Away, Internal Error, Service Restart, Try Again Later, and
        // Bad Gateway all carry an explicit retryable availability signal.
        Some(1001 | 1011..=1014) => (
            ErrorKind::Unavailable,
            "OpenAI Responses WebSocket became unavailable before turn settlement",
        ),
        _ => (
            ErrorKind::UnexpectedEof,
            "OpenAI Responses WebSocket closed before turn settlement",
        ),
    }
}

fn failed_terminal(kind: ErrorKind, message: &'static str) -> SessionTerminal {
    SessionTerminal::Failed(
        SessionFailure::new(kind, PublicDiagnosticText::from(message)).with_retryable(matches!(
            kind,
            ErrorKind::RateLimited | ErrorKind::Unavailable
        )),
    )
}

fn cancelled_terminal(message: &'static str) -> SessionTerminal {
    SessionTerminal::Cancelled {
        reason: Some(PublicDiagnosticText::from(message)),
    }
}

fn expired_terminal() -> SessionTerminal {
    SessionTerminal::Expired {
        reason: Some(PublicDiagnosticText::from(
            "OpenAI Responses WebSocket session deadline elapsed",
        )),
    }
}

fn session_closed_error() -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "OpenAI Responses WebSocket session is closed",
    )
}

fn actor_stopped_error() -> Error {
    Error::new(
        ErrorKind::UnexpectedEof,
        "OpenAI Responses WebSocket actor stopped before acknowledgement",
    )
}

fn session_terminal_error(terminal: SessionTerminal) -> Error {
    match terminal {
        SessionTerminal::Failed(failure) => Error::new(failure.kind, failure.message),
        SessionTerminal::Cancelled { reason } => Error::new(
            ErrorKind::Cancelled,
            reason.unwrap_or_else(|| {
                PublicDiagnosticText::from("OpenAI Responses WebSocket session was cancelled")
            }),
        ),
        SessionTerminal::Expired { reason } => Error::new(
            ErrorKind::Timeout,
            reason.unwrap_or_else(|| {
                PublicDiagnosticText::from("OpenAI Responses WebSocket session expired")
            }),
        ),
        SessionTerminal::Closed(_) => session_closed_error(),
        _ => session_closed_error(),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures_util::StreamExt;
    use serde_json::{Value, json};
    use siumai_core::{Message, MessageRole, ReplayDomain, ReplayDomainId};
    use siumai_transport::{EndpointConfig, TransportLimits};

    use super::*;
    use crate::configured::{OpenAiCredential, OpenAiProvider, OpenAiResponsesOptions};

    struct MockSender {
        outgoing: mpsc::UnboundedSender<WebSocketFrame>,
    }

    #[async_trait]
    impl SocketSenderAdapter for MockSender {
        async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
            self.outgoing
                .send(frame)
                .map_err(|_| Error::new(ErrorKind::Transport, "mock sender is closed"))
        }

        async fn close(&mut self) -> Result<(), Error> {
            Ok(())
        }
    }

    struct MockReceiver {
        incoming: mpsc::UnboundedReceiver<Result<Option<WebSocketFrame>, Error>>,
    }

    #[async_trait]
    impl SocketReceiverAdapter for MockReceiver {
        async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
            self.incoming.recv().await.unwrap_or(Ok(None))
        }
    }

    struct BlockingSender {
        started: Option<oneshot::Sender<()>>,
    }

    #[async_trait]
    impl SocketSenderAdapter for BlockingSender {
        async fn send(&mut self, _frame: WebSocketFrame) -> Result<(), Error> {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<(), Error> {
            Ok(())
        }
    }

    struct PanicReceiver {
        trigger: oneshot::Receiver<()>,
    }

    #[async_trait]
    impl SocketReceiverAdapter for PanicReceiver {
        async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
            let _ = (&mut self.trigger).await;
            panic!("intentional Responses WebSocket actor panic");
        }
    }

    struct SingleSocketConnector {
        socket: Mutex<Option<SocketAdapter>>,
    }

    #[async_trait]
    impl SessionConnector for SingleSocketConnector {
        async fn connect(&self, _request: ConnectRequest) -> Result<SocketAdapter, Error> {
            self.socket
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .ok_or_else(|| Error::new(ErrorKind::Internal, "mock socket was already consumed"))
        }
    }

    struct Harness {
        connector: Arc<SingleSocketConnector>,
        incoming: mpsc::UnboundedSender<Result<Option<WebSocketFrame>, Error>>,
        outgoing: mpsc::UnboundedReceiver<WebSocketFrame>,
    }

    fn harness() -> Harness {
        let (outgoing_tx, outgoing) = mpsc::unbounded_channel();
        let (incoming, incoming_rx) = mpsc::unbounded_channel();
        Harness {
            connector: Arc::new(SingleSocketConnector {
                socket: Mutex::new(Some(SocketAdapter::new(
                    MockSender {
                        outgoing: outgoing_tx,
                    },
                    MockReceiver {
                        incoming: incoming_rx,
                    },
                ))),
            }),
            incoming,
            outgoing,
        }
    }

    fn provider(with_websocket: bool) -> Result<OpenAiProvider, super::super::OpenAiConfigError> {
        provider_with_limits(with_websocket, TransportLimits::default())
    }

    fn provider_with_limits(
        with_websocket: bool,
        limits: TransportLimits,
    ) -> Result<OpenAiProvider, super::super::OpenAiConfigError> {
        let mut builder = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
            .with_responses_websocket_transport_limits(limits)
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("responses-ws-test").unwrap(),
            ));
        if with_websocket {
            builder = builder.with_responses_websocket_endpoint(
                WebSocketEndpoint::local_explicit("ws://127.0.0.1:43192/v1/responses").unwrap(),
            );
        }
        builder.build()
    }

    fn request(text: &str) -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, text)])
    }

    fn server_event(value: Value) -> Result<Option<WebSocketFrame>, Error> {
        Ok(Some(WebSocketFrame::Text(value.to_string())))
    }

    fn created(response_id: &str, sequence: u64) -> Value {
        json!({
            "type": "response.created",
            "sequence_number": sequence,
            "response": {
                "id": response_id,
                "model": "gpt-5.6",
                "status": "in_progress",
                "output": []
            }
        })
    }

    fn completed(response_id: &str, sequence: u64) -> Value {
        json!({
            "type": "response.completed",
            "sequence_number": sequence,
            "response": {
                "id": response_id,
                "model": "gpt-5.6",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4
                }
            }
        })
    }

    fn failed(response_id: &str, sequence: u64) -> Value {
        json!({
            "type": "response.failed",
            "sequence_number": sequence,
            "response": {
                "id": response_id,
                "model": "gpt-5.6",
                "status": "failed",
                "output": [],
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "provider detail",
                    "type": "rate_limit_error"
                }
            }
        })
    }

    fn connection_limit(sequence: u64) -> Value {
        json!({
            "type": "error",
            "sequence_number": sequence,
            "error": {
                "code": "websocket_connection_limit_reached",
                "message": "provider detail",
                "type": "rate_limit_error"
            }
        })
    }

    async fn connect(harness: &Harness) -> OpenAiResponsesWebSocketSession {
        provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .connect(CallOptions::default())
            .await
            .unwrap()
    }

    async fn receive_terminal(
        turn: &mut OpenAiResponsesWebSocketTurn,
    ) -> OpenAiResponsesWebSocketEvent {
        loop {
            let event = turn.next().await.unwrap().unwrap();
            if event.is_terminal() {
                return event;
            }
        }
    }

    async fn complete_turn(
        harness: &mut Harness,
        session: &OpenAiResponsesWebSocketSession,
        prompt: &str,
        response_id: &str,
    ) {
        let mut turn = session
            .generate(request(prompt), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created(response_id, 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed(response_id, 1)))
            .unwrap();
        receive_terminal(&mut turn).await;
        assert!(turn.next().await.is_none());
    }

    async fn wait_for_session_terminal(
        session: &OpenAiResponsesWebSocketSession,
    ) -> SessionTerminal {
        for _ in 0..32 {
            if let Some(terminal) = session.terminal() {
                return terminal;
            }
            tokio::task::yield_now().await;
        }
        panic!("session did not settle");
    }

    async fn wait_for_turn_queue_len(turn: &OpenAiResponsesWebSocketTurn, expected: usize) {
        for _ in 0..32 {
            if turn.receiver.len() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(turn.receiver.len(), expected, "turn queue did not fill");
    }

    fn outgoing_json(frame: WebSocketFrame) -> Value {
        let WebSocketFrame::Text(text) = frame else {
            panic!("expected text frame");
        };
        serde_json::from_str(&text).unwrap()
    }

    #[tokio::test]
    async fn generated_turns_are_single_flight_and_support_continuation() {
        let mut harness = harness();
        let session = connect(&harness).await;

        let mut first = session
            .generate(request("first"), CallOptions::default())
            .await
            .unwrap();
        let first_body = outgoing_json(harness.outgoing.recv().await.unwrap());
        assert_eq!(first_body["type"], "response.create");
        assert!(first_body.get("stream").is_none());
        assert!(first_body.get("background").is_none());
        assert!(first_body.get("generate").is_none());

        let error = session
            .generate(request("overlap"), CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(harness.outgoing.try_recv().is_err());

        harness
            .incoming
            .send(server_event(created("resp_1", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_1", 1)))
            .unwrap();
        let terminal = receive_terminal(&mut first).await;
        let OpenAiResponsesWebSocketEvent::Generated(frame) = terminal else {
            panic!("generated turn must expose a response frame");
        };
        assert_eq!(frame.canonical_terminal_response().unwrap().id, "resp_1");
        assert!(frame.replay_status().is_available());
        assert_eq!(
            first.submission_state(),
            OpenAiResponsesWebSocketSubmissionState::Settled
        );

        let options = OpenAiResponsesOptions {
            previous_response_id: Some("resp_1".to_string()),
            ..OpenAiResponsesOptions::default()
        };
        let call = CallOptions::default()
            .with_provider_options_for(&session.inner.model_handle, &options)
            .unwrap();
        let mut second = session.generate(request("second"), call).await.unwrap();
        let second_body = outgoing_json(harness.outgoing.recv().await.unwrap());
        assert_eq!(second_body["previous_response_id"], "resp_1");
        harness
            .incoming
            .send(server_event(created("resp_2", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_2", 1)))
            .unwrap();
        receive_terminal(&mut second).await;
        assert!(session.terminal().is_none());
    }

    #[tokio::test]
    async fn websocket_rejects_http_transport_fields_before_send() {
        let mut harness = harness();
        let model = provider(true).unwrap().responses("gpt-5.6").unwrap();
        let session = model
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .connect(CallOptions::default())
            .await
            .unwrap();

        for raw in [json!({"stream": true}), json!({"background": true})] {
            let options = CallOptions::default()
                .with_raw_provider_options_for(&session.inner.model_handle, raw)
                .unwrap();
            let error = session
                .generate(request("reject HTTP transport field"), options)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert!(harness.outgoing.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn a_turn_keeps_its_session_alive_after_the_external_handle_is_dropped() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("detached"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        drop(session);

        harness
            .incoming
            .send(server_event(created("resp_detached", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_detached", 1)))
            .unwrap();

        let terminal = receive_terminal(&mut turn).await;
        let OpenAiResponsesWebSocketEvent::Generated(frame) = terminal else {
            panic!("generated turn must expose a response frame");
        };
        assert_eq!(
            frame.canonical_terminal_response().unwrap().id,
            "resp_detached"
        );
    }

    #[tokio::test]
    async fn dropping_the_last_session_handle_aborts_and_joins_the_actor() {
        let harness = harness();
        let session = connect(&harness).await;
        let lifecycle = session.inner.lifecycle.clone();

        drop(session);

        tokio::time::timeout(Duration::from_secs(1), lifecycle.wait_for_actor())
            .await
            .expect("dropping the last session handle must join the actor monitor");
    }

    #[tokio::test]
    async fn warm_up_is_native_only_and_does_not_close_the_session() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut warm_up = session
            .warm_up(request("warm"), CallOptions::default())
            .await
            .unwrap();
        let body = outgoing_json(harness.outgoing.recv().await.unwrap());
        assert_eq!(body["generate"], false);
        harness
            .incoming
            .send(server_event(created("resp_warm", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_warm", 1)))
            .unwrap();
        let terminal = receive_terminal(&mut warm_up).await;
        let OpenAiResponsesWebSocketEvent::WarmUp(frame) = terminal else {
            panic!("warm-up must not fabricate a portable response frame");
        };
        assert!(matches!(
            frame.outcome(),
            Some(OpenAiResponsesWarmUpOutcome::Completed { response })
                if response.id == "resp_warm"
        ));
        assert!(frame.replay_status().is_available());
        assert!(session.terminal().is_none());
    }

    #[tokio::test]
    async fn a_well_formed_failed_turn_leaves_the_socket_synchronized() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut first = session
            .generate(request("fail"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created("resp_fail", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(failed("resp_fail", 1)))
            .unwrap();
        let terminal = receive_terminal(&mut first).await;
        let OpenAiResponsesWebSocketEvent::Generated(frame) = terminal else {
            panic!("failed generated turn must retain its response frame");
        };
        assert!(matches!(
            frame.terminal(),
            Some(StreamTerminal::Failed { error, .. })
                if error.kind() == ErrorKind::RateLimited
        ));

        let mut second = session
            .generate(request("recover"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created("resp_ok", 0)))
            .unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_ok", 1)))
            .unwrap();
        receive_terminal(&mut second).await;
        assert!(session.terminal().is_none());
    }

    #[tokio::test]
    async fn active_eof_is_a_typed_turn_and_session_failure() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("truncate"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness.incoming.send(Ok(None)).unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
        assert!(turn.next().await.is_none());
        tokio::task::yield_now().await;
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::UnexpectedEof,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn active_retryable_close_is_a_sanitized_unavailable_failure() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("temporary close"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(Ok(Some(WebSocketFrame::Close {
                code: Some(1013),
                reason: "private tenant detail".to_string(),
            })))
            .unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unavailable);
        assert!(!error.message().contains("private tenant detail"));
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Unavailable,
                retryable: Some(true),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn session_cancellation_settles_an_active_turn_as_cancelled() {
        let mut harness = harness();
        let cancellation = Cancellation::new();
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .connect(CallOptions::default().with_cancellation(cancellation.clone()))
            .await
            .unwrap();
        let mut turn = session
            .generate(request("cancel session"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();

        cancellation.cancel();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Cancelled { .. }
        ));
    }

    #[tokio::test]
    async fn active_turn_timeout_settles_once_and_closes_the_session() {
        let mut harness = harness();
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .with_turn_timeout(Duration::from_millis(10))
            .connect(CallOptions::default())
            .await
            .unwrap();
        let mut turn = session
            .generate(request("timeout"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();

        let error = tokio::time::timeout(Duration::from_secs(1), turn.next())
            .await
            .expect("turn timeout must settle")
            .expect("turn must emit one terminal error")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Timeout,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn cancellation_before_enqueue_is_not_submitted() {
        let harness = harness();
        let session = connect(&harness).await;
        let cancellation = Cancellation::new();
        cancellation.cancel();

        let error = session
            .generate(
                request("cancel before enqueue"),
                CallOptions::default().with_cancellation(cancellation),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::NotSubmitted)
        );
        assert!(session.terminal().is_none());
    }

    #[tokio::test]
    async fn turn_cancellation_during_submission_fails_closed() {
        let (started_tx, started_rx) = oneshot::channel();
        let (_incoming, incoming_rx) = mpsc::unbounded_channel();
        let connector = Arc::new(SingleSocketConnector {
            socket: Mutex::new(Some(SocketAdapter::new(
                BlockingSender {
                    started: Some(started_tx),
                },
                MockReceiver {
                    incoming: incoming_rx,
                },
            ))),
        });
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(connector)
            .connect(CallOptions::default())
            .await
            .unwrap();
        let cancellation = Cancellation::new();
        let caller = session.clone();
        let task = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                caller
                    .generate(
                        request("cancel submission"),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        started_rx.await.unwrap();

        cancellation.cancel();

        let error = task.await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Indeterminate)
        );
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Cancelled { .. }
        ));
    }

    #[tokio::test]
    async fn deadline_after_sender_acceptance_is_indeterminate() {
        let (started_tx, started_rx) = oneshot::channel();
        let (_incoming, incoming_rx) = mpsc::unbounded_channel();
        let connector = Arc::new(SingleSocketConnector {
            socket: Mutex::new(Some(SocketAdapter::new(
                BlockingSender {
                    started: Some(started_tx),
                },
                MockReceiver {
                    incoming: incoming_rx,
                },
            ))),
        });
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(connector)
            .connect(CallOptions::default())
            .await
            .unwrap();
        let caller = session.clone();
        let task = tokio::spawn(async move {
            caller
                .generate(
                    request("deadline during submission"),
                    CallOptions::default()
                        .with_deadline(Instant::now() + Duration::from_millis(10)),
                )
                .await
        });
        started_rx.await.unwrap();

        let error = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("submission deadline must settle")
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Indeterminate)
        );
        let terminal = wait_for_session_terminal(&session).await;
        assert!(matches!(
            terminal,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Timeout,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn queue_full_wait_honors_cancellation_and_deadline_without_submission() {
        let (started_tx, started_rx) = oneshot::channel();
        let (_incoming, incoming_rx) = mpsc::unbounded_channel();
        let connector = Arc::new(SingleSocketConnector {
            socket: Mutex::new(Some(SocketAdapter::new(
                BlockingSender {
                    started: Some(started_tx),
                },
                MockReceiver {
                    incoming: incoming_rx,
                },
            ))),
        });
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(connector)
            .with_command_queue_capacity(1)
            .connect(CallOptions::default())
            .await
            .unwrap();

        let first_cancellation = Cancellation::new();
        let first = tokio::spawn({
            let caller = session.clone();
            let cancellation = first_cancellation.clone();
            async move {
                caller
                    .generate(
                        request("occupy actor"),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        started_rx.await.unwrap();

        let second_cancellation = Cancellation::new();
        let second = tokio::spawn({
            let caller = session.clone();
            let cancellation = second_cancellation.clone();
            async move {
                caller
                    .generate(
                        request("fill queue"),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        for _ in 0..32 {
            if session.inner.commands.capacity() == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(session.inner.commands.capacity(), 0);

        let third_cancellation = Cancellation::new();
        let third = tokio::spawn({
            let caller = session.clone();
            let cancellation = third_cancellation.clone();
            async move {
                caller
                    .generate(
                        request("cancel while queue is full"),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        tokio::task::yield_now().await;
        third_cancellation.cancel();

        let error = tokio::time::timeout(Duration::from_secs(1), third)
            .await
            .expect("queue wait must observe cancellation")
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::NotSubmitted)
        );

        let error = session
            .generate(
                request("deadline while queue is full"),
                CallOptions::default().with_deadline(Instant::now() + Duration::from_millis(10)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::NotSubmitted)
        );

        first_cancellation.cancel();
        second_cancellation.cancel();
        let _ = first.await;
        let _ = second.await;
    }

    #[tokio::test]
    async fn actor_abort_emits_one_failed_terminal_then_eof() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("abort actor"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();

        session.inner.lifecycle.abort_actor();

        let error = tokio::time::timeout(Duration::from_secs(1), turn.next())
            .await
            .expect("actor abort must settle the turn")
            .expect("actor abort must emit one terminal error")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Indeterminate)
        );
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::UnexpectedEof,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn queued_start_drop_proves_the_turn_was_not_submitted() {
        let (lifecycle, _events) = TurnLifecycle::new(1);
        lifecycle.mark_control_accepted();
        let (ack, _response) = oneshot::channel();
        let (commands, receiver) = mpsc::channel(1);
        let command = ActorCommand::Start(StartCommand {
            kind: OpenAiResponsesWebSocketTurnKind::Generate,
            payload: "{}".to_owned(),
            lifecycle: lifecycle.clone(),
            queue_guard: QueuedStartGuard::new(lifecycle.clone()),
            cancellation: Cancellation::new(),
            caller_cancellation: Cancellation::new(),
            deadline: None,
            ack,
        });
        assert!(commands.try_send(command).is_ok());

        drop(receiver);

        assert_eq!(
            lifecycle.submission_state(),
            OpenAiResponsesWebSocketSubmissionState::NotSubmitted
        );
    }

    #[tokio::test]
    async fn cleanup_timeout_aborts_and_waits_for_actor_resources() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let lifecycle = Arc::new(ActorLifecycle::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let probe = DropProbe(dropped.clone());
        let actor = tokio::spawn(async move {
            let _probe = probe;
            std::future::pending::<()>().await;
        });
        lifecycle.attach(actor);

        wait_for_actor_cleanup(&lifecycle, Duration::from_secs(1)).await;

        assert!(dropped.load(Ordering::Acquire));
        assert!(lifecycle.actor_finished.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn actor_panic_emits_one_failed_terminal_then_eof() {
        let (outgoing_tx, mut outgoing) = mpsc::unbounded_channel();
        let (panic_tx, panic_rx) = oneshot::channel();
        let connector = Arc::new(SingleSocketConnector {
            socket: Mutex::new(Some(SocketAdapter::new(
                MockSender {
                    outgoing: outgoing_tx,
                },
                PanicReceiver { trigger: panic_rx },
            ))),
        });
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(connector)
            .connect(CallOptions::default())
            .await
            .unwrap();
        let mut turn = session
            .generate(request("panic actor"), CallOptions::default())
            .await
            .unwrap();
        outgoing.recv().await.unwrap();

        panic_tx.send(()).unwrap();

        let error = tokio::time::timeout(Duration::from_secs(1), turn.next())
            .await
            .expect("actor panic must settle the turn")
            .expect("actor panic must emit one terminal error")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::UnexpectedEof,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn cancellation_and_socket_eof_race_emits_one_terminal() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let cancellation = Cancellation::new();
        let mut turn = session
            .generate(
                request("race cancellation and EOF"),
                CallOptions::default().with_cancellation(cancellation.clone()),
            )
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();

        cancellation.cancel();
        harness.incoming.send(Ok(None)).unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert!(matches!(
            error.kind(),
            ErrorKind::Cancelled | ErrorKind::UnexpectedEof
        ));
        assert!(turn.next().await.is_none());
    }

    #[tokio::test]
    async fn terminal_response_must_follow_identity_created_for_the_turn() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("identity"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(completed("resp_stale", 0)))
            .unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Protocol,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn settled_response_identity_cannot_reopen_on_a_later_turn() {
        let mut harness = harness();
        let session = connect(&harness).await;
        complete_turn(&mut harness, &session, "first", "resp_a").await;
        complete_turn(&mut harness, &session, "second", "resp_b").await;

        let mut turn = session
            .generate(request("third"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created("resp_a", 0)))
            .unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Protocol,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn buffered_frame_is_rejected_before_the_next_turn_is_submitted() {
        let mut harness = harness();
        let session = connect(&harness).await;
        complete_turn(&mut harness, &session, "first", "resp_a").await;
        complete_turn(&mut harness, &session, "second", "resp_b").await;

        harness
            .incoming
            .send(server_event(created("resp_a", 0)))
            .unwrap();
        let error = session
            .generate(request("third"), CallOptions::default())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::NotSubmitted)
        );
        assert!(harness.outgoing.try_recv().is_err());
    }

    #[tokio::test]
    async fn oversized_event_fails_once_before_json_decode() {
        let oversized = created("resp_oversized", 0).to_string();
        let limits = TransportLimits {
            max_frame_bytes: oversized.len() + 128,
            max_event_bytes: oversized.len() - 1,
            ..TransportLimits::default()
        };
        let mut harness = harness();
        let session = provider_with_limits(true, limits)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .connect(CallOptions::default())
            .await
            .unwrap();
        let lifecycle = session.inner.lifecycle.clone();
        let mut turn = session
            .generate(request("oversized"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(Ok(Some(WebSocketFrame::Text(oversized))))
            .unwrap();

        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Indeterminate)
        );
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::ResponseLimit,
                ..
            })
        ));
        tokio::time::timeout(Duration::from_secs(1), lifecycle.wait_for_actor())
            .await
            .expect("oversized events must release the actor");
    }

    #[tokio::test]
    async fn connection_limit_error_settles_both_turn_and_session() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let mut turn = session
            .generate(request("limit"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(connection_limit(0)))
            .unwrap();

        let terminal = receive_terminal(&mut turn).await;
        let OpenAiResponsesWebSocketEvent::Generated(frame) = terminal else {
            panic!("provider error must retain the native response frame");
        };
        assert!(matches!(
            frame.terminal(),
            Some(StreamTerminal::Failed { error, .. })
                if error.kind() == ErrorKind::RateLimited
        ));
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::RateLimited,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn dropping_an_active_turn_cancels_the_session_conservatively() {
        let mut harness = harness();
        let session = connect(&harness).await;
        let lifecycle = session.inner.lifecycle.clone();
        let turn = session
            .generate(request("cancel"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        drop(turn);

        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Cancelled { .. }
        ));
        tokio::time::timeout(Duration::from_secs(1), lifecycle.wait_for_actor())
            .await
            .expect("dropping a turn must not leave the actor detached");
    }

    #[tokio::test]
    async fn saturated_turn_queue_fails_closed_with_one_fallback_error() {
        let mut harness = harness();
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .with_turn_event_queue_capacity(1)
            .connect(CallOptions::default())
            .await
            .unwrap();
        let mut turn = session
            .generate(request("saturate"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created("resp_q", 0)))
            .unwrap();
        wait_for_turn_queue_len(&turn, 1).await;
        harness
            .incoming
            .send(server_event(json!({
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": {
                    "id": "resp_q",
                    "model": "gpt-5.6",
                    "status": "in_progress",
                    "output": []
                }
            })))
            .unwrap();
        let session_terminal = wait_for_session_terminal(&session).await;

        assert!(turn.next().await.unwrap().is_ok());
        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Indeterminate)
        );
        assert!(turn.next().await.is_none());
        assert!(matches!(
            session_terminal,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::ResponseLimit,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn saturated_terminal_queue_preserves_provider_settlement() {
        let mut harness = harness();
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .with_turn_event_queue_capacity(1)
            .connect(CallOptions::default())
            .await
            .unwrap();
        let mut turn = session
            .generate(request("saturate terminal"), CallOptions::default())
            .await
            .unwrap();
        harness.outgoing.recv().await.unwrap();
        harness
            .incoming
            .send(server_event(created("resp_terminal_q", 0)))
            .unwrap();
        wait_for_turn_queue_len(&turn, 1).await;
        harness
            .incoming
            .send(server_event(completed("resp_terminal_q", 1)))
            .unwrap();
        let session_terminal = wait_for_session_terminal(&session).await;

        assert!(turn.next().await.unwrap().is_ok());
        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        assert_eq!(
            OpenAiResponsesWebSocketSubmissionState::from_error(&error),
            Some(OpenAiResponsesWebSocketSubmissionState::Settled)
        );
        assert_eq!(
            turn.submission_state(),
            OpenAiResponsesWebSocketSubmissionState::Settled
        );
        assert!(turn.next().await.is_none());
        assert!(matches!(
            session_terminal,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::ResponseLimit,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn invalid_queue_bounds_fail_before_connector_use() {
        for capacity in [0, MAX_QUEUE_CAPACITY + 1] {
            let command_harness = harness();
            let error = provider(true)
                .unwrap()
                .responses("gpt-5.6")
                .unwrap()
                .websocket()
                .unwrap()
                .with_test_transport(command_harness.connector.clone())
                .with_command_queue_capacity(capacity)
                .connect(CallOptions::default())
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Configuration);
            assert!(
                command_harness
                    .connector
                    .socket
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            );

            let event_harness = harness();
            let error = provider(true)
                .unwrap()
                .responses("gpt-5.6")
                .unwrap()
                .websocket()
                .unwrap()
                .with_test_transport(event_harness.connector.clone())
                .with_turn_event_queue_capacity(capacity)
                .connect(CallOptions::default())
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Configuration);
            assert!(
                event_harness
                    .connector
                    .socket
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            );
        }
    }

    #[test]
    fn custom_provider_requires_an_explicit_websocket_endpoint() {
        let provider = provider(false).unwrap();
        let error = provider
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap_err();
        assert!(matches!(
            error,
            OpenAiResponsesWebSocketConfigError::EndpointNotConfigured
        ));
    }

    #[test]
    fn official_provider_owns_the_default_websocket_endpoint() {
        let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-key"))
            .build()
            .unwrap();
        let config = provider.responses("gpt-5.6").unwrap().websocket().unwrap();
        assert_eq!(
            config.endpoint().expose_url().as_str(),
            OPENAI_RESPONSES_WEBSOCKET_URL
        );
    }

    #[tokio::test]
    async fn connect_rejects_language_model_provider_options_before_opening_a_socket() {
        let harness = harness();
        let model = provider(true).unwrap().responses("gpt-5.6").unwrap();
        let options = CallOptions::default()
            .with_provider_options_for(&model, &OpenAiResponsesOptions::default())
            .unwrap();
        let error = model
            .websocket()
            .unwrap()
            .with_test_transport(harness.connector.clone())
            .connect(options)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(
            harness
                .connector
                .socket
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        );
    }

    #[test]
    fn official_provider_rejects_a_caller_controlled_websocket_relay() {
        let error = OpenAiProvider::builder(OpenAiCredential::api_key("test-key"))
            .with_responses_websocket_endpoint(
                WebSocketEndpoint::local_explicit("ws://127.0.0.1:43192/v1/responses").unwrap(),
            )
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            super::super::OpenAiConfigError::CallerControlledResponsesWebSocketOnOfficialProvider
        ));
    }
}
