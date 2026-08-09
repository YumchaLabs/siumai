//! Persistent OpenAI Responses WebSocket sessions.

use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
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
    StreamTerminal, Warning,
};
use siumai_protocol_openai::responses::{
    DecodedResponsesStreamFrame, ResponseWire, ResponsesStreamDecoder, ResponsesStreamEvent,
    ResponsesTerminalPolicy,
};
use siumai_transport::framing::WebSocketFrame;
use siumai_transport::{
    RequestHeaders, WebSocketEndpoint, WebSocketReceiver, WebSocketSender, WebSocketTransport,
};
use thiserror::Error as ThisError;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::mode::OpenAiApiMode;
use super::model::{
    OpenAiResponsesModel, attach_policy_warnings, contextualize_terminal_error,
    model_error_context, responses_terminal_policy,
};

/// Current provider-owned endpoint for persistent Responses sessions.
pub const OPENAI_RESPONSES_WEBSOCKET_URL: &str = "wss://api.openai.com/v1/responses";

const MAX_SESSION_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_COMMAND_QUEUE_CAPACITY: usize = 16;
const DEFAULT_TURN_EVENT_QUEUE_CAPACITY: usize = 128;
const MAX_QUEUE_CAPACITY: usize = 4096;

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

/// A connector request that exposes no credential value.
pub struct OpenAiResponsesWebSocketConnectRequest {
    lineage_id: SessionLineageId,
    model: Arc<str>,
    transport: WebSocketTransport,
    headers: RequestHeaders,
    options: CallOptions,
}

impl OpenAiResponsesWebSocketConnectRequest {
    pub fn lineage_id(&self) -> &SessionLineageId {
        &self.lineage_id
    }

    pub fn model(&self) -> &str {
        self.model.as_ref()
    }

    pub fn endpoint(&self) -> &WebSocketEndpoint {
        self.transport.endpoint()
    }

    pub fn call_options(&self) -> &CallOptions {
        &self.options
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketConnectRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWebSocketConnectRequest")
            .field("lineage_id", &self.lineage_id)
            .field("model", &self.model)
            .field("endpoint", self.transport.endpoint())
            .field("headers", &self.headers)
            .field("options", &self.options)
            .finish()
    }
}

/// Sending half used by the deterministic connector seam.
#[async_trait]
pub trait OpenAiResponsesWebSocketSocketSender: Send + 'static {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error>;
    async fn close(&mut self) -> Result<(), Error>;
}

/// Receiving half used by the deterministic connector seam.
#[async_trait]
pub trait OpenAiResponsesWebSocketSocketReceiver: Send + 'static {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error>;
}

#[async_trait]
impl OpenAiResponsesWebSocketSocketSender for WebSocketSender {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        WebSocketSender::send(self, frame).await
    }

    async fn close(&mut self) -> Result<(), Error> {
        WebSocketSender::close(self).await
    }
}

#[async_trait]
impl OpenAiResponsesWebSocketSocketReceiver for WebSocketReceiver {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        WebSocketReceiver::next(self).await
    }
}

/// One established socket split into actor-owned halves.
pub struct OpenAiResponsesWebSocketSocket {
    sender: Box<dyn OpenAiResponsesWebSocketSocketSender>,
    receiver: Box<dyn OpenAiResponsesWebSocketSocketReceiver>,
}

impl OpenAiResponsesWebSocketSocket {
    pub fn new<S, R>(sender: S, receiver: R) -> Self
    where
        S: OpenAiResponsesWebSocketSocketSender,
        R: OpenAiResponsesWebSocketSocketReceiver,
    {
        Self {
            sender: Box::new(sender),
            receiver: Box::new(receiver),
        }
    }

    fn into_parts(
        self,
    ) -> (
        Box<dyn OpenAiResponsesWebSocketSocketSender>,
        Box<dyn OpenAiResponsesWebSocketSocketReceiver>,
    ) {
        (self.sender, self.receiver)
    }
}

impl fmt::Debug for OpenAiResponsesWebSocketSocket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesWebSocketSocket")
            .finish_non_exhaustive()
    }
}

/// Mockable seam for opening one persistent Responses WebSocket.
#[async_trait]
pub trait OpenAiResponsesWebSocketConnector: Send + Sync + 'static {
    async fn connect(
        &self,
        request: OpenAiResponsesWebSocketConnectRequest,
    ) -> Result<OpenAiResponsesWebSocketSocket, Error>;
}

/// Production connector backed by `siumai-transport` endpoint policy and bounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiResponsesWebSocketTransportConnector;

#[async_trait]
impl OpenAiResponsesWebSocketConnector for OpenAiResponsesWebSocketTransportConnector {
    async fn connect(
        &self,
        request: OpenAiResponsesWebSocketConnectRequest,
    ) -> Result<OpenAiResponsesWebSocketSocket, Error> {
        let connection = request
            .transport
            .connect(request.headers, request.options)
            .await?;
        let (sender, receiver) = connection.split();
        Ok(OpenAiResponsesWebSocketSocket::new(sender, receiver))
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
    connector: Arc<dyn OpenAiResponsesWebSocketConnector>,
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
            connector: Arc::new(OpenAiResponsesWebSocketTransportConnector),
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

    pub fn with_connector(mut self, connector: Arc<dyn OpenAiResponsesWebSocketConnector>) -> Self {
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
        let request = OpenAiResponsesWebSocketConnectRequest {
            lineage_id: lineage_id.clone(),
            model: model.clone(),
            transport: self.transport.clone(),
            headers: RequestHeaders::new(),
            options,
        };
        let socket = self.connector.connect(request).await?;
        let (sender, receiver) = socket.into_parts();
        let terminal = Arc::new(Mutex::new(None));
        let (commands, command_rx) = mpsc::channel(self.command_queue_capacity);
        let scope = self.model.runtime.scope_arc(OpenAiApiMode::Responses);
        let actor = SessionActor {
            sender,
            receiver,
            commands: command_rx,
            terminal: terminal.clone(),
            scope,
            model: self.model.clone(),
            terminal_policy: responses_terminal_policy(&self.model.runtime),
            turn_timeout: self.turn_timeout,
            session_cancellation,
            session_deadline,
            last_settled_response_id: None,
            active: None,
        };
        tokio::spawn(actor.run());
        Ok(OpenAiResponsesWebSocketSession {
            inner: Arc::new(SessionHandle {
                lineage_id,
                model,
                model_handle: self.model.clone(),
                commands,
                terminal,
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
    warnings: Arc<[Warning]>,
}

impl OpenAiResponsesWarmUpFrame {
    pub fn native(&self) -> &ResponsesStreamEvent {
        &self.native
    }

    pub fn outcome(&self) -> Option<&OpenAiResponsesWarmUpOutcome> {
        self.outcome.as_ref()
    }

    pub fn warnings(&self) -> &[Warning] {
        self.warnings.as_ref()
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
            .field("warning_count", &self.warnings.len())
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
    shared: Arc<TurnShared>,
    cancellation: Cancellation,
    _session: Arc<SessionHandle>,
    settled: bool,
}

impl OpenAiResponsesWebSocketTurn {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub const fn kind(&self) -> OpenAiResponsesWebSocketTurnKind {
        self.kind
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl Stream for OpenAiResponsesWebSocketTurn {
    type Item = Result<OpenAiResponsesWebSocketEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.receiver).poll_recv(context) {
            Poll::Ready(Some(item)) => {
                if item
                    .as_ref()
                    .map_or(true, OpenAiResponsesWebSocketEvent::is_terminal)
                {
                    self.settled = true;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                if let Some(error) = self.shared.take_fallback_error() {
                    self.settled = true;
                    return Poll::Ready(Some(Err(error)));
                }
                self.settled = true;
                Poll::Ready(None)
            }
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
            .field("settled", &self.settled)
            .finish()
    }
}

impl Drop for OpenAiResponsesWebSocketTurn {
    fn drop(&mut self) {
        if !self.settled {
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
        lock_terminal(&self.inner.terminal).clone()
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
            return Ok(terminal);
        }
        let (ack, response) = oneshot::channel();
        self.inner
            .commands
            .send(ActorCommand::Close { request, ack })
            .await
            .map_err(|_| session_closed_error())?;
        match response.await {
            Ok(result) => result,
            Err(_) => self.terminal().ok_or_else(session_closed_error),
        }
    }

    async fn start(
        &self,
        kind: OpenAiResponsesWebSocketTurnKind,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<OpenAiResponsesWebSocketTurn, Error> {
        if self.terminal().is_some() {
            return Err(session_closed_error());
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
            .prepare_websocket_call(scope, request, &options, generate)?;
        let payload = serde_json::to_string(&prepared.body).map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "failed to serialize an OpenAI Responses WebSocket command",
            )
            .with_source(source)
        })?;
        let id = self.inner.next_turn_id.fetch_add(1, Ordering::Relaxed);
        let (events, receiver) = mpsc::channel(self.inner.turn_event_queue_capacity);
        let shared = Arc::new(TurnShared::default());
        let mut cancellation = CancelOnDrop::new();
        let caller_cancellation = options.cancellation().child();
        let caller_deadline = options.deadline();
        let (ack, response) = oneshot::channel();
        self.inner
            .commands
            .send(ActorCommand::Start(StartCommand {
                kind,
                payload,
                warnings: prepared.warnings,
                events,
                shared: shared.clone(),
                cancellation: cancellation.cancellation().clone(),
                caller_cancellation,
                caller_deadline,
                ack,
            }))
            .await
            .map_err(|_| session_closed_error())?;
        response.await.map_err(|_| session_closed_error())??;
        Ok(OpenAiResponsesWebSocketTurn {
            id,
            kind,
            receiver,
            shared,
            cancellation: cancellation.disarm(),
            _session: self.inner.clone(),
            settled: false,
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
    terminal: Arc<Mutex<Option<SessionTerminal>>>,
    turn_event_queue_capacity: usize,
    next_turn_id: AtomicU64,
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
    warnings: Vec<Warning>,
    events: mpsc::Sender<Result<OpenAiResponsesWebSocketEvent, Error>>,
    shared: Arc<TurnShared>,
    cancellation: Cancellation,
    caller_cancellation: Cancellation,
    caller_deadline: Option<Instant>,
    ack: oneshot::Sender<Result<(), Error>>,
}

struct ActiveTurn {
    kind: OpenAiResponsesWebSocketTurnKind,
    decoder: ResponsesStreamDecoder,
    response_id: Option<String>,
    warnings: Arc<[Warning]>,
    context: ErrorContext,
    events: mpsc::Sender<Result<OpenAiResponsesWebSocketEvent, Error>>,
    shared: Arc<TurnShared>,
    cancellation: Cancellation,
    caller_cancellation: Cancellation,
    deadline: Option<Instant>,
}

struct SessionActor {
    sender: Box<dyn OpenAiResponsesWebSocketSocketSender>,
    receiver: Box<dyn OpenAiResponsesWebSocketSocketReceiver>,
    commands: mpsc::Receiver<ActorCommand>,
    terminal: Arc<Mutex<Option<SessionTerminal>>>,
    scope: Arc<ProviderScope>,
    model: OpenAiResponsesModel,
    terminal_policy: ResponsesTerminalPolicy,
    turn_timeout: Duration,
    session_cancellation: Cancellation,
    session_deadline: Option<Instant>,
    last_settled_response_id: Option<String>,
    active: Option<ActiveTurn>,
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
            command = self.commands.recv() => self.handle_ready_command(command).await,
            frame = self.receiver.receive() => self.handle_ready_frame(frame).await,
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
            None => {
                let _ = self.sender.close().await;
                Some(SessionTerminal::Closed(SessionCloseMetadata::local()))
            }
        }
    }

    async fn handle_active_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Option<SessionTerminal> {
        match command {
            Some(ActorCommand::Start(command)) => {
                let _ = command.ack.send(Err(Error::new(
                    ErrorKind::InvalidInput,
                    "an OpenAI Responses WebSocket turn is already active",
                )));
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
                let _ = self.sender.close().await;
                Some(cancelled_terminal(
                    "OpenAI Responses WebSocket session handle was dropped",
                ))
            }
        }
    }

    async fn start_turn(&mut self, command: StartCommand) -> Option<SessionTerminal> {
        if command.cancellation.is_cancelled() || command.caller_cancellation.is_cancelled() {
            let _ = command.ack.send(Err(Error::cancelled(
                "OpenAI Responses WebSocket turn was cancelled before submission",
            )));
            return None;
        }
        let deadline = effective_deadline(command.caller_deadline, self.turn_timeout);
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let _ = command.ack.send(Err(Error::new(
                ErrorKind::Timeout,
                "OpenAI Responses WebSocket turn deadline elapsed before submission",
            )));
            return None;
        }

        enum SubmissionOutcome {
            Sent(Result<(), Error>),
            SessionCancelled,
            TurnCancelled,
            TurnTimedOut,
            SessionTimedOut,
        }

        let session_cancellation = self.session_cancellation.clone();
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
                let _ = command.ack.send(Err(Error::cancelled(
                    "OpenAI Responses WebSocket session was cancelled during submission",
                )));
                return Some(cancelled_terminal(
                    "OpenAI Responses WebSocket session was cancelled",
                ));
            }
            SubmissionOutcome::TurnCancelled => {
                let _ = command.ack.send(Err(Error::cancelled(
                    "OpenAI Responses WebSocket turn was cancelled during submission",
                )));
                return Some(cancelled_terminal(
                    "OpenAI Responses WebSocket turn was cancelled during submission",
                ));
            }
            SubmissionOutcome::TurnTimedOut => {
                let _ = command.ack.send(Err(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket turn deadline elapsed during submission",
                )));
                return Some(failed_terminal(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket turn deadline elapsed during submission",
                ));
            }
            SubmissionOutcome::SessionTimedOut => {
                let _ = command.ack.send(Err(Error::new(
                    ErrorKind::Timeout,
                    "OpenAI Responses WebSocket session deadline elapsed during submission",
                )));
                return Some(expired_terminal());
            }
        };
        if let Err(error) = send_result {
            let kind = error.kind();
            let _ = command.ack.send(Err(error));
            return Some(failed_terminal(
                kind,
                "failed to send an OpenAI Responses WebSocket turn",
            ));
        }
        let context = model_error_context(&self.model, ModelOperation::Stream);
        self.active = Some(ActiveTurn {
            kind: command.kind,
            decoder: ResponsesStreamDecoder::new(
                self.scope.as_ref().clone(),
                self.model.model_id().clone(),
            )
            .with_terminal_policy(self.terminal_policy),
            response_id: None,
            warnings: command.warnings.into(),
            context,
            events: command.events,
            shared: command.shared,
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
            Ok(Some(WebSocketFrame::Close { .. })) | Ok(None) => {
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
        let identity = validate_active_response_identity(active, &decoded, canonical.as_ref());
        if let Err(error) = identity {
            self.fail_active(error);
            return Some(failed_terminal(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket event violated turn identity",
            ));
        }
        let active = self.active.as_mut().expect("active text owns a turn");
        if let Some(response) = &canonical
            && self
                .last_settled_response_id
                .as_deref()
                .is_some_and(|last| last == response.id)
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
        let session_fatal = is_session_fatal_provider_event(&decoded, canonical.as_ref());
        let terminal = decoded.is_terminal();
        let settled_response_id = canonical.as_ref().map(|response| response.id.clone());
        let event = match active.kind {
            OpenAiResponsesWebSocketTurnKind::Generate => {
                generated_event(decoded, canonical, &active.warnings, &active.context)
            }
            OpenAiResponsesWebSocketTurnKind::WarmUp => {
                warm_up_event(decoded, canonical, active.warnings.clone(), &active.context)
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
        if let Err(failure) = emit_turn_event(active, event) {
            let terminal = match failure {
                EmitFailure::Closed => {
                    cancelled_terminal("OpenAI Responses WebSocket turn consumer was dropped")
                }
                EmitFailure::Saturated => failed_terminal(
                    ErrorKind::ResponseLimit,
                    "OpenAI Responses WebSocket turn queue was saturated",
                ),
            };
            self.active = None;
            return Some(terminal);
        }
        if terminal {
            if let Some(response_id) = settled_response_id {
                self.last_settled_response_id = Some(response_id);
            }
            self.active = None;
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
        if error.context().operation.is_none() {
            error = error.with_context(active.context.clone());
        }
        match active.events.try_send(Err(error)) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                active.shared.set_fallback_error(Error::new(
                    ErrorKind::ResponseLimit,
                    "OpenAI Responses WebSocket turn queue was saturated",
                ))
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {}
        }
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
        *lock_terminal(&self.terminal) = Some(terminal);
        let _ = self.sender.close().await;
    }
}

fn validate_active_response_identity(
    active: &mut ActiveTurn,
    decoded: &DecodedResponsesStreamFrame,
    canonical: Option<&ResponseWire>,
) -> Result<(), Error> {
    if let Some(response) = decoded.native().response_resource() {
        if active
            .response_id
            .as_deref()
            .is_some_and(|existing| existing != response.id)
        {
            return Err(Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses WebSocket turn changed its response ID",
            ));
        }
        active
            .response_id
            .get_or_insert_with(|| response.id.clone());
    }

    if let Some(response) = canonical
        && active.response_id.as_deref() != Some(response.id.as_str())
    {
        return Err(Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses WebSocket terminal did not match a response created for this turn",
        ));
    }
    Ok(())
}

fn generated_event(
    decoded: DecodedResponsesStreamFrame,
    canonical_terminal_response: Option<ResponseWire>,
    warnings: &[Warning],
    context: &ErrorContext,
) -> Result<OpenAiResponsesWebSocketEvent, Error> {
    let (native, mut portable_events) = decoded.into_parts();
    for event in &mut portable_events {
        contextualize_terminal_error(event, context);
        attach_policy_warnings(event, warnings);
    }
    Ok(OpenAiResponsesWebSocketEvent::Generated(
        super::responses_native::OpenAiResponsesStreamFrame::new(
            native,
            portable_events,
            canonical_terminal_response,
        ),
    ))
}

fn warm_up_event(
    decoded: DecodedResponsesStreamFrame,
    canonical_terminal_response: Option<ResponseWire>,
    warnings: Arc<[Warning]>,
    context: &ErrorContext,
) -> Result<OpenAiResponsesWebSocketEvent, Error> {
    let (native, portable_events) = decoded.into_parts();
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
            warnings,
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

fn emit_turn_event(
    active: &ActiveTurn,
    event: OpenAiResponsesWebSocketEvent,
) -> Result<(), EmitFailure> {
    match active.events.try_send(Ok(event)) {
        Ok(()) => Ok(()),
        Err(mpsc::error::TrySendError::Closed(_)) => Err(EmitFailure::Closed),
        Err(mpsc::error::TrySendError::Full(_)) => {
            active.shared.set_fallback_error(Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI Responses WebSocket turn queue was saturated",
            ));
            Err(EmitFailure::Saturated)
        }
    }
}

enum EmitFailure {
    Closed,
    Saturated,
}

#[derive(Default)]
struct TurnShared {
    fallback_error: Mutex<Option<Error>>,
}

impl TurnShared {
    fn set_fallback_error(&self, error: Error) {
        let mut slot = self
            .fallback_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn take_fallback_error(&self) -> Option<Error> {
        self.fallback_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
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

fn effective_deadline(explicit: Option<Instant>, timeout: Duration) -> Option<Instant> {
    let configured = Instant::now().checked_add(timeout);
    match (explicit, configured) {
        (Some(explicit), Some(configured)) => Some(explicit.min(configured)),
        (Some(explicit), None) => Some(explicit),
        (None, configured) => configured,
    }
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
        None => std::future::pending::<()>().await,
    }
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

fn lock_terminal(
    terminal: &Mutex<Option<SessionTerminal>>,
) -> std::sync::MutexGuard<'_, Option<SessionTerminal>> {
    terminal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::StreamExt;
    use serde_json::{Value, json};
    use siumai_core::{Message, MessageRole, ProviderOptions, ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::configured::{OpenAiCredential, OpenAiProvider, OpenAiResponsesOptions};

    struct MockSender {
        outgoing: mpsc::UnboundedSender<WebSocketFrame>,
    }

    #[async_trait]
    impl OpenAiResponsesWebSocketSocketSender for MockSender {
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
    impl OpenAiResponsesWebSocketSocketReceiver for MockReceiver {
        async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
            self.incoming.recv().await.unwrap_or(Ok(None))
        }
    }

    struct BlockingSender {
        started: Option<oneshot::Sender<()>>,
    }

    #[async_trait]
    impl OpenAiResponsesWebSocketSocketSender for BlockingSender {
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

    struct SingleSocketConnector {
        socket: Mutex<Option<OpenAiResponsesWebSocketSocket>>,
    }

    #[async_trait]
    impl OpenAiResponsesWebSocketConnector for SingleSocketConnector {
        async fn connect(
            &self,
            _request: OpenAiResponsesWebSocketConnectRequest,
        ) -> Result<OpenAiResponsesWebSocketSocket, Error> {
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
                socket: Mutex::new(Some(OpenAiResponsesWebSocketSocket::new(
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
        let mut builder = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
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
            .with_connector(harness.connector.clone())
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

        let options = OpenAiResponsesOptions {
            previous_response_id: Some("resp_1".to_string()),
            ..OpenAiResponsesOptions::default()
        };
        let call =
            CallOptions::default().with_provider_options(ProviderOptions::typed(&options).unwrap());
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
    async fn session_cancellation_settles_an_active_turn_as_cancelled() {
        let mut harness = harness();
        let cancellation = Cancellation::new();
        let session = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_connector(harness.connector.clone())
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
    async fn turn_cancellation_during_submission_fails_closed() {
        let (started_tx, started_rx) = oneshot::channel();
        let (_incoming, incoming_rx) = mpsc::unbounded_channel();
        let connector = Arc::new(SingleSocketConnector {
            socket: Mutex::new(Some(OpenAiResponsesWebSocketSocket::new(
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
            .with_connector(connector)
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
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Cancelled { .. }
        ));
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
            .with_connector(harness.connector.clone())
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

        assert!(turn.next().await.unwrap().is_ok());
        let error = turn.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        assert!(turn.next().await.is_none());
        assert!(matches!(
            wait_for_session_terminal(&session).await,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::ResponseLimit,
                ..
            })
        ));
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
        let options = ProviderOptions::typed(&OpenAiResponsesOptions::default()).unwrap();
        let error = provider(true)
            .unwrap()
            .responses("gpt-5.6")
            .unwrap()
            .websocket()
            .unwrap()
            .with_connector(harness.connector.clone())
            .connect(CallOptions::default().with_provider_options(options))
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
