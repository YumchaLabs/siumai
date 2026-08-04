//! Experimental OpenAI Realtime and Realtime Translation session runtimes.
//!
//! Each established WebSocket is owned by one actor. Public session handles
//! communicate with that actor through a bounded command queue, which keeps
//! concurrent send, receive, and close operations free from socket locks.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use http::header::{HeaderName, HeaderValue};
use reqwest::Url;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use siumai_core::experimental::{
    ProviderSession, SessionCloseMetadata, SessionCloseOrigin, SessionCloseRequest, SessionFailure,
    SessionIncoming, SessionLineageId, SessionTerminal, SessionTransportKind,
};
use siumai_core::{CallOptions, Error, ErrorKind, PublicDiagnosticText};
use siumai_protocol_openai::realtime::{
    DecodedRealtimeEvent, JsonTextFrame, OpenAiRealtimeClientEvent, OpenAiRealtimeDecoder,
    OpenAiRealtimeServerEvent, OpenAiTranslationClientEvent, OpenAiTranslationCodec,
    OpenAiTranslationServerEvent, RealtimeCodecError, RealtimeCodecLimits, RealtimeInputFrame,
};
use siumai_transport::framing::WebSocketFrame;
use siumai_transport::{
    EndpointError, EndpointPolicy, LocalNetworkGrant, OfficialOrigin, RequestBuildError,
    RequestHeaders, TransportConfigError, TransportLimits, WebSocketEndpoint, WebSocketReceiver,
    WebSocketSender, WebSocketTransport,
};
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc, oneshot};

use super::credential::{OpenAiCredential, OpenAiCredentialError};

/// Current official OpenAI Realtime conversation model.
pub const OPENAI_REALTIME_MODEL: &str = "gpt-realtime-2.1";
/// Current official OpenAI Realtime Translation model.
pub const OPENAI_REALTIME_TRANSLATION_MODEL: &str = "gpt-realtime-translate";
/// Official Realtime WebSocket endpoint before the model query is applied.
pub const OPENAI_REALTIME_WEBSOCKET_URL: &str = "wss://api.openai.com/v1/realtime";
/// Official Realtime Translation WebSocket endpoint before the model query is applied.
pub const OPENAI_REALTIME_TRANSLATION_WEBSOCKET_URL: &str =
    "wss://api.openai.com/v1/realtime/translations";
/// Official endpoint that creates Realtime client secrets.
pub const OPENAI_REALTIME_CLIENT_SECRETS_URL: &str =
    "https://api.openai.com/v1/realtime/client_secrets";
/// Official endpoint that creates Realtime Translation client secrets.
pub const OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL: &str =
    "https://api.openai.com/v1/realtime/translations/client_secrets";
/// Official WebRTC SDP endpoint for Realtime conversations.
pub const OPENAI_REALTIME_WEBRTC_CALLS_URL: &str = "https://api.openai.com/v1/realtime/calls";
/// Official WebRTC SDP endpoint for Realtime Translation.
pub const OPENAI_REALTIME_TRANSLATION_WEBRTC_CALLS_URL: &str =
    "https://api.openai.com/v1/realtime/translations/calls";
/// Data-channel label documented for OpenAI Realtime WebRTC sessions.
pub const OPENAI_REALTIME_WEBRTC_DATA_CHANNEL: &str = "oai-events";
/// Official source for the current Realtime WebSocket connection contract.
pub const OPENAI_REALTIME_WEBSOCKET_SOURCE_URL: &str =
    "https://developers.openai.com/api/docs/guides/realtime-websocket";
/// Official source for the current Realtime Translation contract.
pub const OPENAI_REALTIME_TRANSLATION_SOURCE_URL: &str =
    "https://developers.openai.com/api/docs/guides/realtime-translation";

const OFFICIAL_ORIGIN: &str = "https://api.openai.com";
const DEFAULT_COMMAND_QUEUE_CAPACITY: usize = 64;
const DEFAULT_INCOMING_QUEUE_CAPACITY: usize = 256;
const MAX_SESSION_QUEUE_CAPACITY: usize = 100_000;
const MAX_MODEL_BYTES: usize = 512;
const MAX_SAFETY_IDENTIFIER_BYTES: usize = 16 * 1024;
const MAX_CLIENT_SECRET_BYTES: usize = 16 * 1024;
const MIN_CLIENT_SECRET_TTL_SECONDS: u64 = 10;
const MAX_CLIENT_SECRET_TTL_SECONDS: u64 = 7_200;
const SAFETY_IDENTIFIER_HEADER: HeaderName = HeaderName::from_static("openai-safety-identifier");

/// Which native OpenAI Realtime protocol owns a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiRealtimeRoute {
    Conversation,
    Translation,
}

impl OpenAiRealtimeRoute {
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::Conversation => OPENAI_REALTIME_MODEL,
            Self::Translation => OPENAI_REALTIME_TRANSLATION_MODEL,
        }
    }

    pub const fn client_secrets_url(self) -> &'static str {
        match self {
            Self::Conversation => OPENAI_REALTIME_CLIENT_SECRETS_URL,
            Self::Translation => OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL,
        }
    }

    pub const fn webrtc_calls_url(self) -> &'static str {
        match self {
            Self::Conversation => OPENAI_REALTIME_WEBRTC_CALLS_URL,
            Self::Translation => OPENAI_REALTIME_TRANSLATION_WEBRTC_CALLS_URL,
        }
    }

    const fn websocket_url(self) -> &'static str {
        match self {
            Self::Conversation => OPENAI_REALTIME_WEBSOCKET_URL,
            Self::Translation => OPENAI_REALTIME_TRANSLATION_WEBSOCKET_URL,
        }
    }

    const fn lineage_prefix(self) -> &'static str {
        match self {
            Self::Conversation => "openai-realtime",
            Self::Translation => "openai-realtime-translation",
        }
    }
}

/// Whether an endpoint is provider-owned or explicitly caller-authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenAiRealtimeEndpointKind {
    Official,
    PublicCustom,
    Loopback,
    PrivateNetwork,
    LinkLocal,
}

/// A Realtime endpoint with an explicit network-security policy.
#[derive(Clone)]
pub struct OpenAiRealtimeEndpoint {
    location: RealtimeEndpointLocation,
}

#[derive(Clone)]
enum RealtimeEndpointLocation {
    Official,
    Explicit { url: Url, policy: EndpointPolicy },
}

impl OpenAiRealtimeEndpoint {
    /// Use the provider-owned endpoint selected by the typed route.
    pub const fn official() -> Self {
        Self {
            location: RealtimeEndpointLocation::Official,
        }
    }

    /// Use an explicitly supplied public `wss` endpoint.
    pub fn public_custom(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::explicit(url, EndpointPolicy::PublicCustom)
    }

    /// Use an explicitly supplied loopback `ws` or `wss` endpoint.
    pub fn local_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::explicit(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
        )
    }

    /// Use an explicitly supplied RFC 1918 or IPv6 ULA endpoint.
    pub fn private_network_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::explicit(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::PrivateNetwork),
        )
    }

    /// Use an explicitly supplied link-local endpoint.
    pub fn link_local_explicit(url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::explicit(
            url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::LinkLocal),
        )
    }

    pub fn kind(&self) -> OpenAiRealtimeEndpointKind {
        match &self.location {
            RealtimeEndpointLocation::Official => OpenAiRealtimeEndpointKind::Official,
            RealtimeEndpointLocation::Explicit {
                policy: EndpointPolicy::PublicCustom,
                ..
            } => OpenAiRealtimeEndpointKind::PublicCustom,
            RealtimeEndpointLocation::Explicit {
                policy: EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
                ..
            } => OpenAiRealtimeEndpointKind::Loopback,
            RealtimeEndpointLocation::Explicit {
                policy: EndpointPolicy::LocalExplicit(LocalNetworkGrant::PrivateNetwork),
                ..
            } => OpenAiRealtimeEndpointKind::PrivateNetwork,
            RealtimeEndpointLocation::Explicit {
                policy: EndpointPolicy::LocalExplicit(LocalNetworkGrant::LinkLocal),
                ..
            } => OpenAiRealtimeEndpointKind::LinkLocal,
            RealtimeEndpointLocation::Explicit {
                policy: EndpointPolicy::Official(_),
                ..
            } => OpenAiRealtimeEndpointKind::Official,
            RealtimeEndpointLocation::Explicit { .. } => OpenAiRealtimeEndpointKind::PublicCustom,
        }
    }

    pub fn is_official(&self) -> bool {
        matches!(&self.location, RealtimeEndpointLocation::Official)
    }

    fn explicit(url: impl AsRef<str>, policy: EndpointPolicy) -> Result<Self, EndpointError> {
        let endpoint = WebSocketEndpoint::new(url.as_ref(), policy.clone())?;
        Ok(Self {
            location: RealtimeEndpointLocation::Explicit {
                url: endpoint.expose_url().clone(),
                policy,
            },
        })
    }

    fn resolve(
        &self,
        route: OpenAiRealtimeRoute,
        model: &str,
    ) -> Result<WebSocketEndpoint, OpenAiRealtimeConfigError> {
        let (mut url, policy) = match &self.location {
            RealtimeEndpointLocation::Official => {
                let url = Url::parse(route.websocket_url())
                    .map_err(|_| OpenAiRealtimeConfigError::InvalidOfficialMetadata)?;
                let origin = OfficialOrigin::new(OFFICIAL_ORIGIN)?;
                (url, EndpointPolicy::Official(origin))
            }
            RealtimeEndpointLocation::Explicit { url, policy } => (url.clone(), policy.clone()),
        };
        ensure_model_query(&mut url, model)?;
        WebSocketEndpoint::new(url.as_str(), policy).map_err(Into::into)
    }
}

impl Default for OpenAiRealtimeEndpoint {
    fn default() -> Self {
        Self::official()
    }
}

impl fmt::Debug for OpenAiRealtimeEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeEndpoint")
            .field("kind", &self.kind())
            .field("url", &"[REDACTED]")
            .finish()
    }
}

/// Invalid static configuration for an OpenAI Realtime session.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenAiRealtimeConfigError {
    #[error("OpenAI Realtime model ID is empty, malformed, or too large")]
    InvalidModel,
    #[error("OpenAI Realtime queue capacity must be between 1 and {maximum}")]
    InvalidQueueCapacity { maximum: usize },
    #[error("OpenAI Realtime safety identifier is empty, malformed, or too large")]
    InvalidSafetyIdentifier,
    #[error("the explicit Realtime endpoint already selects a different model")]
    EndpointModelConflict,
    #[error("the official OpenAI Realtime endpoint requires authentication")]
    OfficialEndpointRequiresAuthentication,
    #[error("a provider with a custom HTTP endpoint requires an explicit Realtime endpoint")]
    ExplicitEndpointRequiredForCustomProvider,
    #[error("OpenAI Realtime official endpoint metadata is invalid")]
    InvalidOfficialMetadata,
    #[error("OpenAI Realtime client-secret session must be a JSON object")]
    InvalidClientSecretSession,
    #[error("OpenAI Realtime client-secret lifetime must be between 10 and 7200 seconds")]
    InvalidClientSecretLifetime,
    #[error(transparent)]
    Credential(#[from] OpenAiCredentialError),
    #[error(transparent)]
    Endpoint(#[from] EndpointError),
    #[error(transparent)]
    Transport(#[from] TransportConfigError),
    #[error(transparent)]
    Request(#[from] RequestBuildError),
    #[error(transparent)]
    Codec(#[from] RealtimeCodecError),
}

/// A connector request that contains no directly accessible credential value.
pub struct OpenAiRealtimeConnectRequest {
    lineage_id: SessionLineageId,
    route: OpenAiRealtimeRoute,
    model: Arc<str>,
    transport: WebSocketTransport,
    headers: RequestHeaders,
    options: CallOptions,
}

impl OpenAiRealtimeConnectRequest {
    pub fn lineage_id(&self) -> &SessionLineageId {
        &self.lineage_id
    }

    pub const fn route(&self) -> OpenAiRealtimeRoute {
        self.route
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

    /// OpenAI currently exposes no resume token for these provider sessions.
    pub const fn supports_provider_resume(&self) -> bool {
        false
    }
}

impl fmt::Debug for OpenAiRealtimeConnectRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeConnectRequest")
            .field("lineage_id", &self.lineage_id)
            .field("route", &self.route)
            .field("model", &self.model)
            .field("endpoint", self.transport.endpoint())
            .field("headers", &self.headers)
            .field("options", &self.options)
            .field("supports_provider_resume", &false)
            .finish()
    }
}

/// Sending half used by the mockable Realtime connector seam.
#[async_trait]
pub trait OpenAiRealtimeSocketSender: Send + 'static {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error>;
    async fn close(&mut self) -> Result<(), Error>;
}

/// Receiving half used by the mockable Realtime connector seam.
#[async_trait]
pub trait OpenAiRealtimeSocketReceiver: Send + 'static {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error>;
}

#[async_trait]
impl OpenAiRealtimeSocketSender for WebSocketSender {
    async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
        WebSocketSender::send(self, frame).await
    }

    async fn close(&mut self) -> Result<(), Error> {
        WebSocketSender::close(self).await
    }
}

#[async_trait]
impl OpenAiRealtimeSocketReceiver for WebSocketReceiver {
    async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
        WebSocketReceiver::next(self).await
    }
}

/// One established socket split into actor-owned sending and receiving halves.
pub struct OpenAiRealtimeSocket {
    sender: Box<dyn OpenAiRealtimeSocketSender>,
    receiver: Box<dyn OpenAiRealtimeSocketReceiver>,
}

impl OpenAiRealtimeSocket {
    pub fn new<S, R>(sender: S, receiver: R) -> Self
    where
        S: OpenAiRealtimeSocketSender,
        R: OpenAiRealtimeSocketReceiver,
    {
        Self {
            sender: Box::new(sender),
            receiver: Box::new(receiver),
        }
    }

    fn into_parts(
        self,
    ) -> (
        Box<dyn OpenAiRealtimeSocketSender>,
        Box<dyn OpenAiRealtimeSocketReceiver>,
    ) {
        (self.sender, self.receiver)
    }
}

impl fmt::Debug for OpenAiRealtimeSocket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeSocket")
            .finish_non_exhaustive()
    }
}

/// Mockable seam for opening exactly one native Realtime WebSocket.
#[async_trait]
pub trait OpenAiRealtimeConnector: Send + Sync + 'static {
    async fn connect(
        &self,
        request: OpenAiRealtimeConnectRequest,
    ) -> Result<OpenAiRealtimeSocket, Error>;
}

/// Production connector backed by `siumai-transport` WebSocket policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiWebSocketConnector;

#[async_trait]
impl OpenAiRealtimeConnector for OpenAiWebSocketConnector {
    async fn connect(
        &self,
        request: OpenAiRealtimeConnectRequest,
    ) -> Result<OpenAiRealtimeSocket, Error> {
        let connection = request
            .transport
            .connect(request.headers, request.options)
            .await?;
        let (sender, receiver) = connection.split();
        Ok(OpenAiRealtimeSocket::new(sender, receiver))
    }
}

#[derive(Clone)]
struct SessionConfig {
    route: OpenAiRealtimeRoute,
    credential: OpenAiCredential,
    model: String,
    endpoint: OpenAiRealtimeEndpoint,
    organization: Option<String>,
    project: Option<String>,
    safety_identifier: Option<String>,
    limits: TransportLimits,
    connect_timeout: Option<Duration>,
    session_timeout: Option<Duration>,
    io_timeout: Option<Duration>,
    command_queue_capacity: usize,
    incoming_queue_capacity: usize,
    codec_limits: RealtimeCodecLimits,
    connector: Arc<dyn OpenAiRealtimeConnector>,
}

impl SessionConfig {
    fn new(
        route: OpenAiRealtimeRoute,
        credential: OpenAiCredential,
        model: impl Into<String>,
        endpoint: OpenAiRealtimeEndpoint,
    ) -> Self {
        Self {
            route,
            credential,
            model: model.into(),
            endpoint,
            organization: None,
            project: None,
            safety_identifier: None,
            limits: TransportLimits::default(),
            connect_timeout: None,
            session_timeout: None,
            io_timeout: None,
            command_queue_capacity: DEFAULT_COMMAND_QUEUE_CAPACITY,
            incoming_queue_capacity: DEFAULT_INCOMING_QUEUE_CAPACITY,
            codec_limits: RealtimeCodecLimits::default(),
            connector: Arc::new(OpenAiWebSocketConnector),
        }
    }

    fn official(route: OpenAiRealtimeRoute, credential: OpenAiCredential) -> Self {
        Self::new(
            route,
            credential,
            route.default_model(),
            OpenAiRealtimeEndpoint::official(),
        )
    }

    fn prepare(&self) -> Result<PreparedConnection, OpenAiRealtimeConfigError> {
        validate_model(&self.model)?;
        validate_queue_capacity(self.command_queue_capacity)?;
        validate_queue_capacity(self.incoming_queue_capacity)?;
        validate_safety_identifier(self.safety_identifier.as_deref())?;
        self.credential.validate()?;
        if self.endpoint.is_official() && self.credential.is_unauthenticated() {
            return Err(OpenAiRealtimeConfigError::OfficialEndpointRequiresAuthentication);
        }

        let endpoint = self.endpoint.resolve(self.route, &self.model)?;
        let auth = self
            .credential
            .clone()
            .into_auth(self.organization.clone(), self.project.clone())?;
        let mut transport = WebSocketTransport::builder(endpoint)
            .with_auth(auth)
            .with_limits(self.limits.clone());
        if let Some(timeout) = self.connect_timeout {
            transport = transport.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.session_timeout {
            transport = transport.with_session_timeout(timeout);
        }
        if let Some(timeout) = self.io_timeout {
            transport = transport.with_io_timeout(timeout);
        }
        let transport = transport.build()?;

        let mut headers = RequestHeaders::new();
        if let Some(safety_identifier) = &self.safety_identifier {
            let value = HeaderValue::from_str(safety_identifier)
                .map_err(|_| OpenAiRealtimeConfigError::InvalidSafetyIdentifier)?;
            headers = headers.try_insert(SAFETY_IDENTIFIER_HEADER.clone(), value)?;
        }

        Ok(PreparedConnection { transport, headers })
    }

    fn debug(&self, name: &str, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct(name)
            .field("route", &self.route)
            .field("credential", &self.credential)
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("organization_present", &self.organization.is_some())
            .field("project_present", &self.project.is_some())
            .field(
                "safety_identifier_present",
                &self.safety_identifier.is_some(),
            )
            .field("limits", &self.limits)
            .field("connect_timeout", &self.connect_timeout)
            .field("session_timeout", &self.session_timeout)
            .field("io_timeout", &self.io_timeout)
            .field("command_queue_capacity", &self.command_queue_capacity)
            .field("incoming_queue_capacity", &self.incoming_queue_capacity)
            .field("codec_limits", &self.codec_limits)
            .field("connector", &"configured")
            .finish()
    }
}

struct PreparedConnection {
    transport: WebSocketTransport,
    headers: RequestHeaders,
}

/// Construction configuration for a typed Realtime conversation session.
#[derive(Clone)]
pub struct OpenAiRealtimeConfig {
    inner: SessionConfig,
}

impl OpenAiRealtimeConfig {
    /// Use the current official model and endpoint.
    pub fn official(credential: OpenAiCredential) -> Self {
        Self {
            inner: SessionConfig::official(OpenAiRealtimeRoute::Conversation, credential),
        }
    }

    /// Use an explicit model and policy-bound endpoint.
    pub fn new(
        credential: OpenAiCredential,
        model: impl Into<String>,
        endpoint: OpenAiRealtimeEndpoint,
    ) -> Self {
        Self {
            inner: SessionConfig::new(
                OpenAiRealtimeRoute::Conversation,
                credential,
                model,
                endpoint,
            ),
        }
    }

    pub fn model(&self) -> &str {
        &self.inner.model
    }

    pub fn endpoint(&self) -> &OpenAiRealtimeEndpoint {
        &self.inner.endpoint
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = model.into();
        self
    }

    pub fn with_endpoint(mut self, endpoint: OpenAiRealtimeEndpoint) -> Self {
        self.inner.endpoint = endpoint;
        self
    }

    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.inner.organization = Some(organization.into());
        self
    }

    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.inner.project = Some(project.into());
        self
    }

    pub fn with_safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.inner.safety_identifier = Some(safety_identifier.into());
        self
    }

    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Self {
        self.inner.limits = limits;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.inner.connect_timeout = Some(timeout);
        self
    }

    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.inner.session_timeout = Some(timeout);
        self
    }

    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.inner.io_timeout = Some(timeout);
        self
    }

    pub fn with_command_queue_capacity(mut self, capacity: usize) -> Self {
        self.inner.command_queue_capacity = capacity;
        self
    }

    pub fn with_incoming_queue_capacity(mut self, capacity: usize) -> Self {
        self.inner.incoming_queue_capacity = capacity;
        self
    }

    pub fn with_codec_limits(mut self, limits: RealtimeCodecLimits) -> Self {
        self.inner.codec_limits = limits;
        self
    }

    pub fn with_connector(mut self, connector: Arc<dyn OpenAiRealtimeConnector>) -> Self {
        self.inner.connector = connector;
        self
    }

    pub fn validate(&self) -> Result<(), OpenAiRealtimeConfigError> {
        self.inner.prepare()?;
        OpenAiRealtimeDecoder::new(self.inner.codec_limits)?;
        Ok(())
    }

    /// Open one new provider lineage. Failed opens are never resumed implicitly.
    pub async fn connect(&self, options: CallOptions) -> Result<OpenAiRealtimeSession, Error> {
        let protocol = ConversationProtocol::new(self.inner.codec_limits)
            .map_err(OpenAiRealtimeConfigError::from)
            .map_err(configuration_error)?;
        connect_session(&self.inner, protocol, options)
            .await
            .map(|inner| OpenAiRealtimeSession { inner })
    }
}

impl fmt::Debug for OpenAiRealtimeConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.debug("OpenAiRealtimeConfig", formatter)
    }
}

/// Construction configuration for a typed Realtime Translation session.
#[derive(Clone)]
pub struct OpenAiTranslationConfig {
    inner: SessionConfig,
}

impl OpenAiTranslationConfig {
    /// Use the current official translation model and endpoint.
    pub fn official(credential: OpenAiCredential) -> Self {
        Self {
            inner: SessionConfig::official(OpenAiRealtimeRoute::Translation, credential),
        }
    }

    /// Use an explicit translation model and policy-bound endpoint.
    pub fn new(
        credential: OpenAiCredential,
        model: impl Into<String>,
        endpoint: OpenAiRealtimeEndpoint,
    ) -> Self {
        Self {
            inner: SessionConfig::new(
                OpenAiRealtimeRoute::Translation,
                credential,
                model,
                endpoint,
            ),
        }
    }

    pub fn model(&self) -> &str {
        &self.inner.model
    }

    pub fn endpoint(&self) -> &OpenAiRealtimeEndpoint {
        &self.inner.endpoint
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = model.into();
        self
    }

    pub fn with_endpoint(mut self, endpoint: OpenAiRealtimeEndpoint) -> Self {
        self.inner.endpoint = endpoint;
        self
    }

    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.inner.organization = Some(organization.into());
        self
    }

    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.inner.project = Some(project.into());
        self
    }

    pub fn with_safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.inner.safety_identifier = Some(safety_identifier.into());
        self
    }

    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Self {
        self.inner.limits = limits;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.inner.connect_timeout = Some(timeout);
        self
    }

    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.inner.session_timeout = Some(timeout);
        self
    }

    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.inner.io_timeout = Some(timeout);
        self
    }

    pub fn with_command_queue_capacity(mut self, capacity: usize) -> Self {
        self.inner.command_queue_capacity = capacity;
        self
    }

    pub fn with_incoming_queue_capacity(mut self, capacity: usize) -> Self {
        self.inner.incoming_queue_capacity = capacity;
        self
    }

    pub fn with_codec_limits(mut self, limits: RealtimeCodecLimits) -> Self {
        self.inner.codec_limits = limits;
        self
    }

    pub fn with_connector(mut self, connector: Arc<dyn OpenAiRealtimeConnector>) -> Self {
        self.inner.connector = connector;
        self
    }

    pub fn validate(&self) -> Result<(), OpenAiRealtimeConfigError> {
        self.inner.prepare()?;
        OpenAiTranslationCodec::new(self.inner.codec_limits)?;
        Ok(())
    }

    /// Open one new provider lineage. Failed opens are never resumed implicitly.
    pub async fn connect(&self, options: CallOptions) -> Result<OpenAiTranslationSession, Error> {
        let protocol = TranslationProtocol::new(self.inner.codec_limits)
            .map_err(OpenAiRealtimeConfigError::from)
            .map_err(configuration_error)?;
        connect_session(&self.inner, protocol, options)
            .await
            .map(|inner| OpenAiTranslationSession { inner })
    }
}

impl fmt::Debug for OpenAiTranslationConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.debug("OpenAiTranslationConfig", formatter)
    }
}

/// Lossless typed inbound event for a Realtime conversation.
pub type OpenAiRealtimeInbound = DecodedRealtimeEvent<OpenAiRealtimeServerEvent>;
/// Lossless typed inbound event for a Realtime Translation session.
pub type OpenAiTranslationInbound = DecodedRealtimeEvent<OpenAiTranslationServerEvent>;

/// Concurrent handle for one native OpenAI Realtime conversation lineage.
#[derive(Clone)]
pub struct OpenAiRealtimeSession {
    inner: Arc<SessionHandle<OpenAiRealtimeClientEvent, OpenAiRealtimeInbound>>,
}

impl OpenAiRealtimeSession {
    pub fn lineage_id(&self) -> &SessionLineageId {
        &self.inner.lineage_id
    }

    pub fn model(&self) -> &str {
        self.inner.model.as_ref()
    }

    /// OpenAI exposes no provider resume token for this session contract.
    pub const fn supports_provider_resume(&self) -> bool {
        false
    }

    pub async fn send(&self, event: OpenAiRealtimeClientEvent) -> Result<(), Error> {
        self.inner.send(event).await
    }

    pub async fn receive(&self) -> Result<SessionIncoming<OpenAiRealtimeInbound>, Error> {
        self.inner.receive().await
    }

    pub async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        self.inner.close(request).await
    }
}

impl fmt::Debug for OpenAiRealtimeSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeSession")
            .field("lineage_id", &self.inner.lineage_id)
            .field("model", &self.inner.model)
            .field("terminal", &self.inner.incoming.terminal().is_some())
            .field("supports_provider_resume", &false)
            .finish()
    }
}

#[async_trait]
impl ProviderSession<OpenAiRealtimeClientEvent, OpenAiRealtimeInbound> for OpenAiRealtimeSession {
    fn lineage_id(&self) -> &SessionLineageId {
        &self.inner.lineage_id
    }

    fn transport_kind(&self) -> SessionTransportKind {
        SessionTransportKind::WebSocket
    }

    async fn send(&self, event: OpenAiRealtimeClientEvent) -> Result<(), Error> {
        self.inner.send(event).await
    }

    async fn receive(&self) -> Result<SessionIncoming<OpenAiRealtimeInbound>, Error> {
        self.inner.receive().await
    }

    async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        self.inner.close(request).await
    }
}

/// Concurrent handle for one native OpenAI Realtime Translation lineage.
#[derive(Clone)]
pub struct OpenAiTranslationSession {
    inner: Arc<SessionHandle<OpenAiTranslationClientEvent, OpenAiTranslationInbound>>,
}

impl OpenAiTranslationSession {
    pub fn lineage_id(&self) -> &SessionLineageId {
        &self.inner.lineage_id
    }

    pub fn model(&self) -> &str {
        self.inner.model.as_ref()
    }

    /// OpenAI exposes no provider resume token for this session contract.
    pub const fn supports_provider_resume(&self) -> bool {
        false
    }

    pub async fn send(&self, event: OpenAiTranslationClientEvent) -> Result<(), Error> {
        self.inner.send(event).await
    }

    pub async fn receive(&self) -> Result<SessionIncoming<OpenAiTranslationInbound>, Error> {
        self.inner.receive().await
    }

    pub async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        self.inner.close(request).await
    }
}

impl fmt::Debug for OpenAiTranslationSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiTranslationSession")
            .field("lineage_id", &self.inner.lineage_id)
            .field("model", &self.inner.model)
            .field("terminal", &self.inner.incoming.terminal().is_some())
            .field("supports_provider_resume", &false)
            .finish()
    }
}

#[async_trait]
impl ProviderSession<OpenAiTranslationClientEvent, OpenAiTranslationInbound>
    for OpenAiTranslationSession
{
    fn lineage_id(&self) -> &SessionLineageId {
        &self.inner.lineage_id
    }

    fn transport_kind(&self) -> SessionTransportKind {
        SessionTransportKind::WebSocket
    }

    async fn send(&self, event: OpenAiTranslationClientEvent) -> Result<(), Error> {
        self.inner.send(event).await
    }

    async fn receive(&self) -> Result<SessionIncoming<OpenAiTranslationInbound>, Error> {
        self.inner.receive().await
    }

    async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        self.inner.close(request).await
    }
}

struct SessionHandle<Outbound, Inbound> {
    lineage_id: SessionLineageId,
    model: Arc<str>,
    commands: mpsc::Sender<ActorCommand<Outbound>>,
    incoming: Arc<SharedIncoming<Inbound>>,
    closing: Arc<CloseSignal>,
    receive_gate: AsyncMutex<()>,
}

impl<Outbound, Inbound> SessionHandle<Outbound, Inbound>
where
    Outbound: Send + 'static,
    Inbound: Send + 'static,
{
    async fn send(&self, event: Outbound) -> Result<(), Error> {
        if self.incoming.terminal().is_some() {
            return Err(session_is_terminal_error());
        }
        let (acknowledge, result) = oneshot::channel();
        if self
            .commands
            .send(ActorCommand::Send { event, acknowledge })
            .await
            .is_err()
        {
            return Err(self.command_channel_error());
        }
        match result.await {
            Ok(result) => result,
            Err(_) => Err(self.command_channel_error()),
        }
    }

    async fn receive(&self) -> Result<SessionIncoming<Inbound>, Error> {
        Ok(self.incoming.receive(&self.receive_gate).await)
    }

    async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error> {
        if let Some(terminal) = self.incoming.terminal() {
            return Ok(terminal);
        }
        validate_close_request(&request)?;
        self.closing.request(request);
        Ok(self.incoming.wait_terminal().await)
    }

    fn command_channel_error(&self) -> Error {
        if self.incoming.terminal().is_some() {
            session_is_terminal_error()
        } else {
            Error::new(
                ErrorKind::Internal,
                "OpenAI Realtime session actor stopped unexpectedly",
            )
        }
    }
}

enum ActorCommand<Outbound> {
    Send {
        event: Outbound,
        acknowledge: oneshot::Sender<Result<(), Error>>,
    },
}

struct CloseSignal {
    request: Mutex<Option<SessionCloseRequest>>,
    ready: Notify,
}

impl CloseSignal {
    fn new() -> Self {
        Self {
            request: Mutex::new(None),
            ready: Notify::new(),
        }
    }

    fn request(&self, request: SessionCloseRequest) {
        let mut current = self
            .request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if current.is_none() {
            *current = Some(request);
            drop(current);
            self.ready.notify_waiters();
        }
    }

    async fn wait(&self) -> SessionCloseRequest {
        loop {
            let notified = self.ready.notified();
            if let Some(request) = self
                .request
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
            {
                return request;
            }
            notified.await;
        }
    }
}

struct SharedIncoming<Inbound> {
    state: Mutex<IncomingState<Inbound>>,
    ready: Notify,
    space: Notify,
    capacity: usize,
}

struct IncomingState<Inbound> {
    events: VecDeque<Inbound>,
    terminal: Option<SessionTerminal>,
}

impl<Inbound> SharedIncoming<Inbound> {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(IncomingState {
                events: VecDeque::with_capacity(capacity),
                terminal: None,
            }),
            ready: Notify::new(),
            space: Notify::new(),
            capacity,
        }
    }

    fn terminal(&self) -> Option<SessionTerminal> {
        self.lock_state().terminal.clone()
    }

    fn try_push(&self, event: Inbound) -> Result<(), Inbound> {
        let mut state = self.lock_state();
        if state.terminal.is_some() || state.events.len() >= self.capacity {
            return Err(event);
        }
        state.events.push_back(event);
        drop(state);
        self.ready.notify_one();
        Ok(())
    }

    /// Preserve one final provider event or one already-decoded pending event.
    fn push_terminal_tail(&self, event: Inbound) {
        let mut state = self.lock_state();
        if state.terminal.is_none() {
            state.events.push_back(event);
        }
        drop(state);
        self.ready.notify_one();
    }

    fn set_terminal(&self, terminal: SessionTerminal) -> SessionTerminal {
        let mut state = self.lock_state();
        let authoritative = match &state.terminal {
            Some(existing) => existing.clone(),
            None => {
                state.terminal = Some(terminal.clone());
                terminal
            }
        };
        drop(state);
        self.ready.notify_waiters();
        self.space.notify_waiters();
        authoritative
    }

    async fn receive(&self, gate: &AsyncMutex<()>) -> SessionIncoming<Inbound> {
        let _receive = gate.lock().await;
        loop {
            let notified = self.ready.notified();
            let mut released_space = false;
            let next = {
                let mut state = self.lock_state();
                if let Some(event) = state.events.pop_front() {
                    released_space = true;
                    Some(SessionIncoming::Event(event))
                } else {
                    state.terminal.clone().map(SessionIncoming::Terminal)
                }
            };
            if released_space {
                self.space.notify_one();
            }
            if let Some(next) = next {
                return next;
            }
            notified.await;
        }
    }

    async fn wait_terminal(&self) -> SessionTerminal {
        loop {
            let notified = self.ready.notified();
            if let Some(terminal) = self.terminal() {
                return terminal;
            }
            notified.await;
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, IncomingState<Inbound>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

trait SessionProtocol: Send + 'static {
    type Outbound: Send + 'static;
    type Inbound: Send + 'static;

    fn encode(&mut self, event: &Self::Outbound) -> Result<EncodedOutbound, RealtimeCodecError>;

    fn begin_close(&mut self) -> Result<CloseAction, RealtimeCodecError>;

    fn decode(
        &mut self,
        frame: RealtimeInputFrame<'_>,
    ) -> Result<DecodedInbound<Self::Inbound>, RealtimeCodecError>;

    fn finish_transport(&self) -> Result<(), RealtimeCodecError>;
}

struct EncodedOutbound {
    frame: JsonTextFrame,
    begins_close: bool,
}

enum CloseAction {
    Application(JsonTextFrame),
    Transport,
}

struct DecodedInbound<Inbound> {
    event: Option<Inbound>,
    terminal: Option<ProtocolTerminal>,
}

enum ProtocolTerminal {
    Closed { locally_initiated: bool },
}

struct ConversationProtocol {
    decoder: OpenAiRealtimeDecoder,
}

impl ConversationProtocol {
    fn new(limits: RealtimeCodecLimits) -> Result<Self, RealtimeCodecError> {
        Ok(Self {
            decoder: OpenAiRealtimeDecoder::new(limits)?,
        })
    }
}

impl SessionProtocol for ConversationProtocol {
    type Outbound = OpenAiRealtimeClientEvent;
    type Inbound = OpenAiRealtimeInbound;

    fn encode(&mut self, event: &Self::Outbound) -> Result<EncodedOutbound, RealtimeCodecError> {
        Ok(EncodedOutbound {
            frame: event.encode()?,
            begins_close: false,
        })
    }

    fn begin_close(&mut self) -> Result<CloseAction, RealtimeCodecError> {
        Ok(CloseAction::Transport)
    }

    fn decode(
        &mut self,
        frame: RealtimeInputFrame<'_>,
    ) -> Result<DecodedInbound<Self::Inbound>, RealtimeCodecError> {
        Ok(DecodedInbound {
            event: Some(self.decoder.decode(frame)?),
            terminal: None,
        })
    }

    fn finish_transport(&self) -> Result<(), RealtimeCodecError> {
        Ok(())
    }
}

struct TranslationProtocol {
    codec: OpenAiTranslationCodec,
}

impl TranslationProtocol {
    fn new(limits: RealtimeCodecLimits) -> Result<Self, RealtimeCodecError> {
        Ok(Self {
            codec: OpenAiTranslationCodec::new(limits)?,
        })
    }
}

impl SessionProtocol for TranslationProtocol {
    type Outbound = OpenAiTranslationClientEvent;
    type Inbound = OpenAiTranslationInbound;

    fn encode(&mut self, event: &Self::Outbound) -> Result<EncodedOutbound, RealtimeCodecError> {
        Ok(EncodedOutbound {
            frame: self.codec.encode(event)?,
            begins_close: matches!(event, OpenAiTranslationClientEvent::SessionClose { .. }),
        })
    }

    fn begin_close(&mut self) -> Result<CloseAction, RealtimeCodecError> {
        self.codec
            .encode(&OpenAiTranslationClientEvent::SessionClose { event_id: None })
            .map(CloseAction::Application)
    }

    fn decode(
        &mut self,
        frame: RealtimeInputFrame<'_>,
    ) -> Result<DecodedInbound<Self::Inbound>, RealtimeCodecError> {
        let decoded = self.codec.decode(frame)?;
        let terminal = match &decoded.event {
            OpenAiTranslationServerEvent::SessionClosed {
                close_was_requested,
                ..
            } => Some(ProtocolTerminal::Closed {
                locally_initiated: *close_was_requested,
            }),
            _ => None,
        };
        Ok(DecodedInbound {
            event: Some(decoded),
            terminal,
        })
    }

    fn finish_transport(&self) -> Result<(), RealtimeCodecError> {
        self.codec.finish_transport()
    }
}

async fn connect_session<P>(
    config: &SessionConfig,
    protocol: P,
    options: CallOptions,
) -> Result<Arc<SessionHandle<P::Outbound, P::Inbound>>, Error>
where
    P: SessionProtocol,
{
    if options.has_provider_options() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Realtime does not accept language-model provider options",
        ));
    }
    let prepared = config.prepare().map_err(configuration_error)?;
    let cancellation = options.cancellation().clone();
    let session_deadline =
        effective_session_deadline(options.deadline(), prepared.transport.session_timeout());
    let lineage_id = new_lineage_id(config.route)?;
    let request = OpenAiRealtimeConnectRequest {
        lineage_id: lineage_id.clone(),
        route: config.route,
        model: Arc::from(config.model.as_str()),
        transport: prepared.transport,
        headers: prepared.headers,
        options,
    };
    let socket = config.connector.connect(request).await?;
    let (sender, receiver) = socket.into_parts();
    let (commands, command_receiver) = mpsc::channel(config.command_queue_capacity);
    let incoming = Arc::new(SharedIncoming::new(config.incoming_queue_capacity));
    let closing = Arc::new(CloseSignal::new());
    let handle = Arc::new(SessionHandle {
        lineage_id,
        model: Arc::from(config.model.as_str()),
        commands,
        incoming: incoming.clone(),
        closing: closing.clone(),
        receive_gate: AsyncMutex::new(()),
    });
    tokio::spawn(run_session_actor(
        protocol,
        sender,
        receiver,
        command_receiver,
        SessionActorControl {
            incoming,
            closing,
            cancellation,
            session_deadline,
        },
    ));
    Ok(handle)
}

struct SessionActorControl<Inbound> {
    incoming: Arc<SharedIncoming<Inbound>>,
    closing: Arc<CloseSignal>,
    cancellation: siumai_core::Cancellation,
    session_deadline: Option<Instant>,
}

async fn run_session_actor<P>(
    mut protocol: P,
    mut sender: Box<dyn OpenAiRealtimeSocketSender>,
    mut receiver: Box<dyn OpenAiRealtimeSocketReceiver>,
    mut commands: mpsc::Receiver<ActorCommand<P::Outbound>>,
    control: SessionActorControl<P::Inbound>,
) where
    P: SessionProtocol,
{
    let SessionActorControl {
        incoming,
        closing,
        cancellation,
        session_deadline,
    } = control;
    let mut close_started = false;
    let mut close_request = None;
    let mut pending_event = None;

    let terminal = loop {
        if let Some(event) = pending_event.take() {
            let space_available = incoming.space.notified();
            match incoming.try_push(event) {
                Ok(()) => continue,
                Err(event) => pending_event = Some(event),
            }
            let terminal = tokio::select! {
                biased;
                _ = cancellation.cancelled() => Some(cancelled_terminal(
                    "OpenAI Realtime session was cancelled",
                )),
                _ = wait_for_deadline(session_deadline) => Some(expired_terminal()),
                request = closing.wait(), if !close_started => {
                    process_close_request(
                        request,
                        &mut protocol,
                        &mut sender,
                        &mut close_started,
                        &mut close_request,
                    ).await
                }
                command = commands.recv() => {
                    match command {
                        Some(command) => process_actor_command(
                            command,
                            &mut protocol,
                            &mut sender,
                            &mut close_started,
                            &mut close_request,
                        ).await,
                        None => Some(cancelled_terminal("session handle dropped")),
                    }
                }
                _ = space_available => None,
            };
            if let Some(terminal) = terminal {
                break terminal;
            }
            continue;
        }

        let terminal = tokio::select! {
            biased;
            _ = cancellation.cancelled() => Some(cancelled_terminal(
                "OpenAI Realtime session was cancelled",
            )),
            _ = wait_for_deadline(session_deadline) => Some(expired_terminal()),
            request = closing.wait(), if !close_started => {
                process_close_request(
                    request,
                    &mut protocol,
                    &mut sender,
                    &mut close_started,
                    &mut close_request,
                ).await
            }
            command = commands.recv() => {
                match command {
                    Some(command) => process_actor_command(
                        command,
                        &mut protocol,
                        &mut sender,
                        &mut close_started,
                        &mut close_request,
                    ).await,
                    None => Some(cancelled_terminal("session handle dropped")),
                }
            }
            received = receiver.receive() => {
                match received {
                    Ok(Some(frame)) => handle_incoming_frame(
                        frame,
                        &mut protocol,
                        &mut sender,
                        &incoming,
                        &mut pending_event,
                        close_request.as_ref(),
                    ).await,
                    Ok(None) => Some(SessionTerminal::Failed(
                        SessionFailure::new(
                            ErrorKind::UnexpectedEof,
                            PublicDiagnosticText::from(
                                "OpenAI Realtime transport ended without a close frame",
                            ),
                        )
                        .with_retryable(true),
                    )),
                    Err(error) => Some(terminal_from_error(
                        &error,
                        "OpenAI Realtime transport failed",
                    )),
                }
            }
        };
        if let Some(terminal) = terminal {
            break terminal;
        }
    };

    terminate_actor(&mut sender, &incoming, &mut pending_event, terminal).await;
}

async fn process_actor_command<P>(
    command: ActorCommand<P::Outbound>,
    protocol: &mut P,
    sender: &mut Box<dyn OpenAiRealtimeSocketSender>,
    close_started: &mut bool,
    close_request: &mut Option<SessionCloseRequest>,
) -> Option<SessionTerminal>
where
    P: SessionProtocol,
{
    match command {
        ActorCommand::Send { event, acknowledge } => {
            if *close_started {
                let _ = acknowledge.send(Err(Error::new(
                    ErrorKind::InvalidInput,
                    "OpenAI Realtime session is closing",
                )));
                return None;
            }
            let encoded = match protocol.encode(&event) {
                Ok(encoded) => encoded,
                Err(error) => {
                    let _ = acknowledge.send(Err(client_event_error(error)));
                    return None;
                }
            };
            match sender
                .send(WebSocketFrame::Text(encoded.frame.into_string()))
                .await
            {
                Ok(()) => {
                    if encoded.begins_close && !*close_started {
                        *close_started = true;
                        *close_request = Some(SessionCloseRequest::new());
                    }
                    let _ = acknowledge.send(Ok(()));
                    None
                }
                Err(error) => {
                    let terminal =
                        terminal_from_error(&error, "OpenAI Realtime command transport failed");
                    let _ = acknowledge.send(Err(error));
                    Some(terminal)
                }
            }
        }
    }
}

async fn process_close_request<P>(
    request: SessionCloseRequest,
    protocol: &mut P,
    sender: &mut Box<dyn OpenAiRealtimeSocketSender>,
    close_started: &mut bool,
    close_request: &mut Option<SessionCloseRequest>,
) -> Option<SessionTerminal>
where
    P: SessionProtocol,
{
    let action = match protocol.begin_close() {
        Ok(action) => action,
        Err(_error) => {
            return Some(SessionTerminal::Failed(
                SessionFailure::new(
                    ErrorKind::Protocol,
                    PublicDiagnosticText::from(
                        "OpenAI Realtime close command violated the protocol state",
                    ),
                )
                .with_retryable(false),
            ));
        }
    };
    *close_started = true;
    *close_request = Some(request.clone());
    match action {
        CloseAction::Application(frame) => {
            match sender.send(WebSocketFrame::Text(frame.into_string())).await {
                Ok(()) => None,
                Err(error) => Some(terminal_from_error(
                    &error,
                    "OpenAI Realtime close command failed",
                )),
            }
        }
        CloseAction::Transport => {
            let code = request.transport_code.map(|code| code as u16);
            let reason = request
                .reason
                .as_ref()
                .map(PublicDiagnosticText::as_str)
                .unwrap_or_default()
                .to_owned();
            match sender.send(WebSocketFrame::Close { code, reason }).await {
                Ok(()) => Some(SessionTerminal::Closed(local_close_metadata(&request))),
                Err(error) => Some(terminal_from_error(
                    &error,
                    "OpenAI Realtime transport close failed",
                )),
            }
        }
    }
}

async fn handle_incoming_frame<P>(
    frame: WebSocketFrame,
    protocol: &mut P,
    sender: &mut Box<dyn OpenAiRealtimeSocketSender>,
    incoming: &Arc<SharedIncoming<P::Inbound>>,
    pending_event: &mut Option<P::Inbound>,
    close_request: Option<&SessionCloseRequest>,
) -> Option<SessionTerminal>
where
    P: SessionProtocol,
{
    match frame {
        WebSocketFrame::Ping(bytes) => match sender.send(WebSocketFrame::Pong(bytes)).await {
            Ok(()) => None,
            Err(error) => Some(terminal_from_error(
                &error,
                "OpenAI Realtime ping response failed",
            )),
        },
        WebSocketFrame::Pong(_) => None,
        WebSocketFrame::Close { code, reason } => {
            if let Err(_error) = protocol.finish_transport() {
                return Some(SessionTerminal::Failed(
                    SessionFailure::new(
                        ErrorKind::Protocol,
                        PublicDiagnosticText::from(
                            "OpenAI Realtime protocol ended before its terminal event",
                        ),
                    )
                    .with_retryable(true),
                ));
            }
            if !is_clean_remote_close(code) {
                return Some(SessionTerminal::Failed(
                    SessionFailure::new(
                        ErrorKind::Transport,
                        PublicDiagnosticText::from(
                            "OpenAI Realtime peer closed the transport abnormally",
                        ),
                    )
                    .with_retryable(true),
                ));
            }
            Some(SessionTerminal::Closed(remote_close_metadata(code, reason)))
        }
        WebSocketFrame::Text(text) => handle_decoded_inbound(
            protocol.decode(RealtimeInputFrame::Text(&text)),
            incoming,
            pending_event,
            close_request,
        ),
        WebSocketFrame::Binary(bytes) => handle_decoded_inbound(
            protocol.decode(RealtimeInputFrame::Binary(bytes.as_ref())),
            incoming,
            pending_event,
            close_request,
        ),
        _ => Some(SessionTerminal::Failed(
            SessionFailure::new(
                ErrorKind::Protocol,
                PublicDiagnosticText::from(
                    "OpenAI Realtime transport returned an unsupported frame",
                ),
            )
            .with_retryable(false),
        )),
    }
}

fn handle_decoded_inbound<Inbound>(
    decoded: Result<DecodedInbound<Inbound>, RealtimeCodecError>,
    incoming: &Arc<SharedIncoming<Inbound>>,
    pending_event: &mut Option<Inbound>,
    close_request: Option<&SessionCloseRequest>,
) -> Option<SessionTerminal> {
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(_error) => {
            return Some(SessionTerminal::Failed(
                SessionFailure::new(
                    ErrorKind::Protocol,
                    PublicDiagnosticText::from(
                        "OpenAI Realtime server event violated the protocol",
                    ),
                )
                .with_retryable(false),
            ));
        }
    };

    if let Some(terminal) = decoded.terminal {
        if let Some(event) = decoded.event {
            incoming.push_terminal_tail(event);
        }
        return Some(match terminal {
            ProtocolTerminal::Closed { locally_initiated } => {
                let metadata = if locally_initiated {
                    close_request
                        .map(local_close_metadata)
                        .unwrap_or_else(|| local_close_metadata(&SessionCloseRequest::new()))
                } else {
                    SessionCloseMetadata::remote()
                };
                SessionTerminal::Closed(metadata)
            }
        });
    }

    if let Some(event) = decoded.event {
        match incoming.try_push(event) {
            Ok(()) => {}
            Err(event) => *pending_event = Some(event),
        }
    }
    None
}

async fn terminate_actor<Inbound>(
    sender: &mut Box<dyn OpenAiRealtimeSocketSender>,
    incoming: &Arc<SharedIncoming<Inbound>>,
    pending_event: &mut Option<Inbound>,
    terminal: SessionTerminal,
) {
    if let Some(event) = pending_event.take() {
        incoming.push_terminal_tail(event);
    }
    incoming.set_terminal(terminal);
    let _ = sender.close().await;
}

fn terminal_from_error(error: &Error, message: &'static str) -> SessionTerminal {
    if error.kind() == ErrorKind::Cancelled {
        return cancelled_terminal(message);
    }
    SessionTerminal::Failed(
        SessionFailure::new(error.kind(), PublicDiagnosticText::from(message))
            .with_retryable(is_retryable_session_failure(error.kind())),
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
            "OpenAI Realtime session deadline elapsed",
        )),
    }
}

fn effective_session_deadline(explicit: Option<Instant>, timeout: Duration) -> Option<Instant> {
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
        .is_some_and(|code| u16::try_from(code).is_err())
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "session close code does not fit the WebSocket close-code range",
        ));
    }
    Ok(())
}

fn is_retryable_session_failure(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::Transport
            | ErrorKind::Timeout
            | ErrorKind::UnexpectedEof
            | ErrorKind::RateLimited
    )
}

fn is_clean_remote_close(code: Option<u16>) -> bool {
    matches!(code, None | Some(1000 | 1001))
}

fn local_close_metadata(request: &SessionCloseRequest) -> SessionCloseMetadata {
    SessionCloseMetadata {
        origin: SessionCloseOrigin::Local,
        transport_code: request.transport_code,
        reason: request.reason.clone(),
    }
}

fn remote_close_metadata(code: Option<u16>, reason: String) -> SessionCloseMetadata {
    let mut metadata = SessionCloseMetadata::remote();
    if let Some(code) = code {
        metadata = metadata.with_transport_code(u32::from(code));
    }
    if !reason.is_empty()
        && let Ok(reason) = PublicDiagnosticText::new(reason)
    {
        metadata = metadata.with_reason(reason);
    }
    metadata
}

fn session_is_terminal_error() -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "OpenAI Realtime session is already terminal",
    )
}

fn client_event_error(source: RealtimeCodecError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "OpenAI Realtime client event is invalid for the current session state",
    )
    .with_source(source)
}

fn configuration_error(source: OpenAiRealtimeConfigError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "OpenAI Realtime session configuration is invalid",
    )
    .with_source(source)
}

fn new_lineage_id(route: OpenAiRealtimeRoute) -> Result<SessionLineageId, Error> {
    SessionLineageId::new(format!(
        "{}:{}",
        route.lineage_prefix(),
        uuid::Uuid::new_v4()
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "OpenAI Realtime lineage ID could not be created",
        )
        .with_source(source)
    })
}

fn validate_model(model: &str) -> Result<(), OpenAiRealtimeConfigError> {
    if model.trim().is_empty()
        || model != model.trim()
        || model.len() > MAX_MODEL_BYTES
        || model.chars().any(char::is_control)
    {
        return Err(OpenAiRealtimeConfigError::InvalidModel);
    }
    Ok(())
}

fn validate_queue_capacity(capacity: usize) -> Result<(), OpenAiRealtimeConfigError> {
    if capacity == 0 || capacity > MAX_SESSION_QUEUE_CAPACITY {
        return Err(OpenAiRealtimeConfigError::InvalidQueueCapacity {
            maximum: MAX_SESSION_QUEUE_CAPACITY,
        });
    }
    Ok(())
}

fn validate_safety_identifier(
    safety_identifier: Option<&str>,
) -> Result<(), OpenAiRealtimeConfigError> {
    let Some(value) = safety_identifier else {
        return Ok(());
    };
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_SAFETY_IDENTIFIER_BYTES
        || HeaderValue::from_str(value).is_err()
    {
        return Err(OpenAiRealtimeConfigError::InvalidSafetyIdentifier);
    }
    Ok(())
}

fn ensure_model_query(url: &mut Url, model: &str) -> Result<(), OpenAiRealtimeConfigError> {
    let selected = url
        .query_pairs()
        .filter_map(|(name, value)| (name == "model").then_some(value.into_owned()))
        .collect::<Vec<_>>();
    match selected.as_slice() {
        [] => {
            url.query_pairs_mut().append_pair("model", model);
            Ok(())
        }
        [selected] if selected == model => Ok(()),
        _ => Err(OpenAiRealtimeConfigError::EndpointModelConflict),
    }
}

/// Route-specific endpoints and data-channel metadata for browser bootstrap.
#[derive(Clone, PartialEq, Eq)]
pub struct OpenAiRealtimeBootstrapMetadata {
    route: OpenAiRealtimeRoute,
    model: String,
    websocket_url: String,
}

impl OpenAiRealtimeBootstrapMetadata {
    pub fn official(
        route: OpenAiRealtimeRoute,
        model: impl Into<String>,
    ) -> Result<Self, OpenAiRealtimeConfigError> {
        let model = model.into();
        validate_model(&model)?;
        let mut websocket_url = Url::parse(route.websocket_url())
            .map_err(|_| OpenAiRealtimeConfigError::InvalidOfficialMetadata)?;
        ensure_model_query(&mut websocket_url, &model)?;
        Ok(Self {
            route,
            model,
            websocket_url: websocket_url.to_string(),
        })
    }

    pub const fn route(&self) -> OpenAiRealtimeRoute {
        self.route
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn client_secrets_url(&self) -> &'static str {
        self.route.client_secrets_url()
    }

    pub fn webrtc_calls_url(&self) -> &'static str {
        self.route.webrtc_calls_url()
    }

    pub fn websocket_url(&self) -> &str {
        &self.websocket_url
    }

    pub const fn data_channel_label(&self) -> &'static str {
        OPENAI_REALTIME_WEBRTC_DATA_CHANNEL
    }
}

impl fmt::Debug for OpenAiRealtimeBootstrapMetadata {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeBootstrapMetadata")
            .field("route", &self.route)
            .field("model", &self.model)
            .field("websocket_url", &self.websocket_url)
            .field("client_secrets_url", &self.client_secrets_url())
            .field("webrtc_calls_url", &self.webrtc_calls_url())
            .field("data_channel_label", &self.data_channel_label())
            .finish()
    }
}

/// Fixed anchor required by OpenAI's client-secret expiration object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiRealtimeClientSecretExpiryAnchor {
    CreatedAt,
}

/// Client-secret lifetime requested from OpenAI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OpenAiRealtimeClientSecretExpiry {
    pub anchor: OpenAiRealtimeClientSecretExpiryAnchor,
    pub seconds: u64,
}

impl OpenAiRealtimeClientSecretExpiry {
    pub const fn created_at(seconds: u64) -> Self {
        Self {
            anchor: OpenAiRealtimeClientSecretExpiryAnchor::CreatedAt,
            seconds,
        }
    }
}

/// Serializable body for either native Realtime client-secret resource.
#[derive(Clone, PartialEq, Serialize)]
pub struct OpenAiRealtimeClientSecretRequest {
    #[serde(skip)]
    route: OpenAiRealtimeRoute,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_after: Option<OpenAiRealtimeClientSecretExpiry>,
    session: Value,
}

impl fmt::Debug for OpenAiRealtimeClientSecretRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeClientSecretRequest")
            .field("route", &self.route)
            .field("expires_after", &self.expires_after)
            .field("model", &self.session.get("model").and_then(Value::as_str))
            .field(
                "session_field_count",
                &self.session.as_object().map(serde_json::Map::len),
            )
            .finish()
    }
}

impl OpenAiRealtimeClientSecretRequest {
    pub fn conversation(model: impl Into<String>) -> Result<Self, OpenAiRealtimeConfigError> {
        let model = model.into();
        validate_model(&model)?;
        Ok(Self {
            route: OpenAiRealtimeRoute::Conversation,
            expires_after: None,
            session: json!({
                "type": "realtime",
                "model": model,
            }),
        })
    }

    pub fn translation(model: impl Into<String>) -> Result<Self, OpenAiRealtimeConfigError> {
        let model = model.into();
        validate_model(&model)?;
        Ok(Self {
            route: OpenAiRealtimeRoute::Translation,
            expires_after: None,
            session: json!({
                "model": model,
            }),
        })
    }

    pub const fn route(&self) -> OpenAiRealtimeRoute {
        self.route
    }

    pub fn endpoint_url(&self) -> &'static str {
        self.route.client_secrets_url()
    }

    pub fn session(&self) -> &Value {
        &self.session
    }

    pub fn with_expires_after_seconds(
        mut self,
        seconds: u64,
    ) -> Result<Self, OpenAiRealtimeConfigError> {
        if !(MIN_CLIENT_SECRET_TTL_SECONDS..=MAX_CLIENT_SECRET_TTL_SECONDS).contains(&seconds) {
            return Err(OpenAiRealtimeConfigError::InvalidClientSecretLifetime);
        }
        self.expires_after = Some(OpenAiRealtimeClientSecretExpiry::created_at(seconds));
        Ok(self)
    }

    /// Replace the provider-native session object while preserving its route.
    pub fn with_session(mut self, session: Value) -> Result<Self, OpenAiRealtimeConfigError> {
        validate_client_secret_session(self.route, &session)?;
        self.session = session;
        Ok(self)
    }
}

/// Decoded short-lived credential returned by a client-secret resource.
pub struct OpenAiRealtimeClientSecretResource {
    route: OpenAiRealtimeRoute,
    value: SecretString,
    expires_at: Option<i64>,
    session: Value,
}

impl OpenAiRealtimeClientSecretResource {
    pub fn decode(
        route: OpenAiRealtimeRoute,
        body: &str,
    ) -> Result<Self, OpenAiRealtimeResourceError> {
        let wire = serde_json::from_str::<ClientSecretWire>(body)?;
        validate_client_secret(&wire.value)?;
        validate_resource_session(route, &wire.session)?;
        Ok(Self {
            route,
            value: SecretString::from(wire.value),
            expires_at: wire.expires_at,
            session: wire.session,
        })
    }

    pub const fn route(&self) -> OpenAiRealtimeRoute {
        self.route
    }

    /// Explicitly expose the short-lived client secret to a trusted bootstrap layer.
    pub fn expose_value(&self) -> &str {
        self.value.expose_secret()
    }

    pub const fn expires_at_unix(&self) -> Option<i64> {
        self.expires_at
    }

    pub fn session(&self) -> &Value {
        &self.session
    }

    pub fn model(&self) -> Option<&str> {
        self.session.get("model").and_then(Value::as_str)
    }
}

impl fmt::Debug for OpenAiRealtimeClientSecretResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRealtimeClientSecretResource")
            .field("route", &self.route)
            .field("value", &"[REDACTED]")
            .field("expires_at", &self.expires_at)
            .field("session_present", &self.session.is_object())
            .finish()
    }
}

#[derive(Deserialize)]
struct ClientSecretWire {
    value: String,
    expires_at: Option<i64>,
    session: Value,
}

/// Invalid client-secret response data.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenAiRealtimeResourceError {
    #[error("OpenAI Realtime client secret is empty, malformed, or too large")]
    InvalidSecret,
    #[error("OpenAI Realtime client-secret response has no session object")]
    InvalidSession,
    #[error(transparent)]
    InvalidJson(#[from] serde_json::Error),
}

fn validate_client_secret(value: &str) -> Result<(), OpenAiRealtimeResourceError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_CLIENT_SECRET_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(OpenAiRealtimeResourceError::InvalidSecret);
    }
    Ok(())
}

fn validate_client_secret_session(
    route: OpenAiRealtimeRoute,
    session: &Value,
) -> Result<(), OpenAiRealtimeConfigError> {
    let object = session
        .as_object()
        .ok_or(OpenAiRealtimeConfigError::InvalidClientSecretSession)?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or(OpenAiRealtimeConfigError::InvalidClientSecretSession)?;
    validate_model(model)?;
    if let Some(session_type) = object.get("type") {
        let expected = match route {
            OpenAiRealtimeRoute::Conversation => "realtime",
            OpenAiRealtimeRoute::Translation => "translation",
        };
        if session_type.as_str() != Some(expected) {
            return Err(OpenAiRealtimeConfigError::InvalidClientSecretSession);
        }
    }
    Ok(())
}

fn validate_resource_session(
    route: OpenAiRealtimeRoute,
    session: &Value,
) -> Result<(), OpenAiRealtimeResourceError> {
    let object = session
        .as_object()
        .ok_or(OpenAiRealtimeResourceError::InvalidSession)?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or(OpenAiRealtimeResourceError::InvalidSession)?;
    if validate_model(model).is_err() {
        return Err(OpenAiRealtimeResourceError::InvalidSession);
    }
    if let Some(session_type) = object.get("type") {
        let expected = match route {
            OpenAiRealtimeRoute::Conversation => "realtime",
            OpenAiRealtimeRoute::Translation => "translation",
        };
        if session_type.as_str() != Some(expected) {
            return Err(OpenAiRealtimeResourceError::InvalidSession);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{Cancellation, ProviderId, ProviderOptions};

    enum MockIncoming {
        Frame(WebSocketFrame),
        Failure(Error),
        Eof,
    }

    struct MockSender {
        sent: mpsc::UnboundedSender<WebSocketFrame>,
    }

    #[async_trait]
    impl OpenAiRealtimeSocketSender for MockSender {
        async fn send(&mut self, frame: WebSocketFrame) -> Result<(), Error> {
            self.sent.send(frame).map_err(|_| {
                Error::new(ErrorKind::Transport, "mock WebSocket observer was dropped")
            })
        }

        async fn close(&mut self) -> Result<(), Error> {
            Ok(())
        }
    }

    struct MockReceiver {
        incoming: mpsc::UnboundedReceiver<MockIncoming>,
        received: Option<Arc<std::sync::atomic::AtomicUsize>>,
    }

    #[async_trait]
    impl OpenAiRealtimeSocketReceiver for MockReceiver {
        async fn receive(&mut self) -> Result<Option<WebSocketFrame>, Error> {
            let incoming = self.incoming.recv().await;
            if incoming.is_some()
                && let Some(received) = &self.received
            {
                received.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            match incoming {
                Some(MockIncoming::Frame(frame)) => Ok(Some(frame)),
                Some(MockIncoming::Failure(error)) => Err(error),
                Some(MockIncoming::Eof) | None => Ok(None),
            }
        }
    }

    struct SingleSocketConnector {
        socket: Mutex<Option<OpenAiRealtimeSocket>>,
        requests: Arc<Mutex<Vec<ConnectSnapshot>>>,
    }

    #[derive(Debug, Clone)]
    struct ConnectSnapshot {
        lineage_id: String,
        route: OpenAiRealtimeRoute,
        model: String,
        endpoint: String,
        supports_provider_resume: bool,
        request_debug: String,
    }

    #[async_trait]
    impl OpenAiRealtimeConnector for SingleSocketConnector {
        async fn connect(
            &self,
            request: OpenAiRealtimeConnectRequest,
        ) -> Result<OpenAiRealtimeSocket, Error> {
            let snapshot = ConnectSnapshot {
                lineage_id: request.lineage_id().to_string(),
                route: request.route(),
                model: request.model().to_owned(),
                endpoint: request.endpoint().expose_url().as_str().to_owned(),
                supports_provider_resume: request.supports_provider_resume(),
                request_debug: format!("{request:?}"),
            };
            self.requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(snapshot);
            self.socket
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .ok_or_else(|| Error::new(ErrorKind::Internal, "mock socket was already consumed"))
        }
    }

    struct FailingConnector {
        lineages: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl OpenAiRealtimeConnector for FailingConnector {
        async fn connect(
            &self,
            request: OpenAiRealtimeConnectRequest,
        ) -> Result<OpenAiRealtimeSocket, Error> {
            self.lineages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request.lineage_id().to_string());
            Err(Error::new(
                ErrorKind::Transport,
                "mock Realtime connection failed",
            ))
        }
    }

    fn mock_socket() -> (
        OpenAiRealtimeSocket,
        mpsc::UnboundedReceiver<WebSocketFrame>,
        mpsc::UnboundedSender<MockIncoming>,
    ) {
        let (sent, sent_frames) = mpsc::unbounded_channel();
        let (incoming_frames, incoming) = mpsc::unbounded_channel();
        (
            OpenAiRealtimeSocket::new(
                MockSender { sent },
                MockReceiver {
                    incoming,
                    received: None,
                },
            ),
            sent_frames,
            incoming_frames,
        )
    }

    fn tracked_mock_socket() -> (
        OpenAiRealtimeSocket,
        mpsc::UnboundedReceiver<WebSocketFrame>,
        mpsc::UnboundedSender<MockIncoming>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let (sent, sent_frames) = mpsc::unbounded_channel();
        let (incoming_frames, incoming) = mpsc::unbounded_channel();
        let received = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            OpenAiRealtimeSocket::new(
                MockSender { sent },
                MockReceiver {
                    incoming,
                    received: Some(received.clone()),
                },
            ),
            sent_frames,
            incoming_frames,
            received,
        )
    }

    fn local_endpoint() -> OpenAiRealtimeEndpoint {
        OpenAiRealtimeEndpoint::local_explicit("ws://127.0.0.1/realtime")
            .expect("explicit loopback endpoint")
    }

    fn connector_with_socket(
        socket: OpenAiRealtimeSocket,
    ) -> (Arc<SingleSocketConnector>, Arc<Mutex<Vec<ConnectSnapshot>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(SingleSocketConnector {
                socket: Mutex::new(Some(socket)),
                requests: requests.clone(),
            }),
            requests,
        )
    }

    #[tokio::test]
    async fn realtime_actor_supports_concurrent_io_and_idempotent_close() {
        let (socket, mut sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let config = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector);
        let session = config.connect(CallOptions::default()).await.unwrap();

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.created","session":{"id":"sess_1"}}"#.to_owned(),
            )))
            .unwrap();
        let created = session.receive().await.unwrap();
        assert!(matches!(
            created,
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiRealtimeServerEvent::SessionCreated { .. },
                ..
            })
        ));

        let send = session.send(OpenAiRealtimeClientEvent::ResponseCreate {
            event_id: Some("event_1".to_owned()),
            response: None,
        });
        let receive = session.receive();
        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"response.created","response":{"id":"resp_1"}}"#.to_owned(),
            )))
            .unwrap();
        let (sent_result, received) = tokio::join!(send, receive);
        sent_result.unwrap();
        assert!(matches!(
            received.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiRealtimeServerEvent::ResponseCreated { .. },
                ..
            })
        ));
        let outbound = sent.recv().await.unwrap();
        assert!(matches!(
            outbound,
            WebSocketFrame::Text(text)
                if serde_json::from_str::<Value>(&text).unwrap()["type"] == "response.create"
        ));

        let (first, second) = tokio::join!(
            session.close(SessionCloseRequest::new()),
            session.close(SessionCloseRequest::new()),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first, second);
        assert!(matches!(first, SessionTerminal::Closed(_)));
        assert_eq!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(first.clone())
        );
        assert_eq!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(first)
        );
    }

    #[tokio::test]
    async fn cancellation_terminates_a_session_while_inbound_delivery_is_backpressured() {
        let (socket, _sent, incoming, received) = tracked_mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let cancellation = Cancellation::new();
        let session = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_incoming_queue_capacity(1)
        .with_connector(connector)
        .connect(CallOptions::default().with_cancellation(cancellation.clone()))
        .await
        .unwrap();

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.created","session":{"id":"sess_backpressure"}}"#.to_owned(),
            )))
            .unwrap();
        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"response.created","response":{"id":"resp_backpressure"}}"#.to_owned(),
            )))
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while received.load(std::sync::atomic::Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
            tokio::task::yield_now().await;
        })
        .await
        .unwrap();

        cancellation.cancel();
        let terminal = tokio::time::timeout(
            Duration::from_secs(1),
            session.close(SessionCloseRequest::new()),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(terminal, SessionTerminal::Cancelled { .. }));

        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiRealtimeServerEvent::SessionCreated { .. },
                ..
            })
        ));
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiRealtimeServerEvent::ResponseCreated { .. },
                ..
            })
        ));
        assert_eq!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(terminal)
        );
    }

    #[tokio::test]
    async fn configured_session_deadline_emits_expired_terminal() {
        let (socket, _sent, _incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let session = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_session_timeout(Duration::from_millis(10))
        .with_connector(connector)
        .connect(CallOptions::default())
        .await
        .unwrap();

        let terminal = tokio::time::timeout(Duration::from_secs(1), session.receive())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            terminal,
            SessionIncoming::Terminal(SessionTerminal::Expired { .. })
        ));
    }

    #[tokio::test]
    async fn realtime_rejects_language_model_provider_options_before_connecting() {
        let (socket, _sent, _incoming) = mock_socket();
        let (connector, requests) = connector_with_socket(socket);
        let options = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"future_language_option": true}),
        )
        .unwrap();
        let error = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector)
        .connect(CallOptions::default().with_provider_options(options))
        .await
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(
            requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );
    }

    #[tokio::test]
    async fn translation_close_waits_for_typed_session_closed() {
        let (socket, mut sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let config = OpenAiTranslationConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_TRANSLATION_MODEL,
            local_endpoint(),
        )
        .with_connector(connector);
        let session = config.connect(CallOptions::default()).await.unwrap();

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.created","session":{"id":"sess_t1"}}"#.to_owned(),
            )))
            .unwrap();
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiTranslationServerEvent::SessionCreated { .. },
                ..
            })
        ));

        let closing_session = session.clone();
        let close = tokio::spawn(async move {
            closing_session
                .close(SessionCloseRequest::new())
                .await
                .unwrap()
        });
        let close_frame = sent.recv().await.unwrap();
        assert!(matches!(
            close_frame,
            WebSocketFrame::Text(text)
                if serde_json::from_str::<Value>(&text).unwrap()["type"] == "session.close"
        ));

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.output_audio.delta","delta":"AAE="}"#.to_owned(),
            )))
            .unwrap();
        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.output_transcript.delta","delta":"bonjour"}"#.to_owned(),
            )))
            .unwrap();
        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.closed","session":{"id":"sess_t1"}}"#.to_owned(),
            )))
            .unwrap();

        let terminal = close.await.unwrap();
        assert!(matches!(
            &terminal,
            SessionTerminal::Closed(SessionCloseMetadata {
                origin: SessionCloseOrigin::Local,
                ..
            })
        ));
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiTranslationServerEvent::OutputAudioDelta(_),
                ..
            })
        ));
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiTranslationServerEvent::OutputTranscriptDelta { .. },
                ..
            })
        ));
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiTranslationServerEvent::SessionClosed { .. },
                ..
            })
        ));
        assert_eq!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(terminal.clone())
        );
        assert_eq!(
            session.close(SessionCloseRequest::new()).await.unwrap(),
            terminal
        );
    }

    #[tokio::test]
    async fn dropping_a_close_future_does_not_duplicate_actor_waiters_or_close_frames() {
        let (socket, mut sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let session = OpenAiTranslationConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_TRANSLATION_MODEL,
            local_endpoint(),
        )
        .with_connector(connector)
        .connect(CallOptions::default())
        .await
        .unwrap();

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.created","session":{"id":"sess_close_waiter"}}"#.to_owned(),
            )))
            .unwrap();
        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Event(DecodedRealtimeEvent {
                event: OpenAiTranslationServerEvent::SessionCreated { .. },
                ..
            })
        ));

        let abandoned_session = session.clone();
        let abandoned =
            tokio::spawn(async move { abandoned_session.close(SessionCloseRequest::new()).await });
        let first_close_frame = sent.recv().await.unwrap();
        assert!(matches!(
            first_close_frame,
            WebSocketFrame::Text(text)
                if serde_json::from_str::<Value>(&text).unwrap()["type"] == "session.close"
        ));
        abandoned.abort();
        let _ = abandoned.await;

        let active_session = session.clone();
        let active = tokio::spawn(async move {
            active_session
                .close(SessionCloseRequest::new())
                .await
                .unwrap()
        });
        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Text(
                r#"{"type":"session.closed","session":{"id":"sess_close_waiter"}}"#.to_owned(),
            )))
            .unwrap();

        assert!(matches!(active.await.unwrap(), SessionTerminal::Closed(_)));
        assert!(sent.try_recv().is_err());
    }

    #[tokio::test]
    async fn binary_protocol_failure_is_cached_as_one_terminal() {
        let (socket, _sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let session = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector)
        .connect(CallOptions::default())
        .await
        .unwrap();

        incoming
            .send(MockIncoming::Frame(WebSocketFrame::Binary(
                vec![0_u8, 1].into(),
            )))
            .unwrap();
        let terminal = match session.receive().await.unwrap() {
            SessionIncoming::Terminal(terminal) => terminal,
            event => panic!("unexpected event: {event:?}"),
        };
        assert!(matches!(
            terminal,
            SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Protocol,
                ..
            })
        ));
        assert_eq!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(terminal.clone())
        );
        assert_eq!(
            session
                .send(OpenAiRealtimeClientEvent::ResponseCancel {
                    event_id: None,
                    response_id: None,
                })
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[tokio::test]
    async fn failed_connect_attempts_always_allocate_new_lineages() {
        let lineages = Arc::new(Mutex::new(Vec::new()));
        let connector = Arc::new(FailingConnector {
            lineages: lineages.clone(),
        });
        let config = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector);

        assert!(config.connect(CallOptions::default()).await.is_err());
        assert!(config.connect(CallOptions::default()).await.is_err());
        let lineages = lineages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(lineages.len(), 2);
        assert_ne!(lineages[0], lineages[1]);
    }

    #[tokio::test]
    async fn official_routes_use_current_models_and_redact_credentials() {
        let (realtime_socket, _sent, _incoming) = mock_socket();
        let (realtime_connector, realtime_requests) = connector_with_socket(realtime_socket);
        let realtime =
            OpenAiRealtimeConfig::official(OpenAiCredential::api_key("canary-official-secret"))
                .with_connector(realtime_connector);
        assert!(!format!("{realtime:?}").contains("canary-official-secret"));
        let session = realtime.connect(CallOptions::default()).await.unwrap();
        drop(session);

        let snapshot = realtime_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .clone();
        assert_eq!(snapshot.route, OpenAiRealtimeRoute::Conversation);
        assert_eq!(snapshot.model, OPENAI_REALTIME_MODEL);
        assert_eq!(
            snapshot.endpoint,
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
        );
        assert!(!snapshot.supports_provider_resume);
        assert!(snapshot.lineage_id.starts_with("openai-realtime:"));
        assert!(!snapshot.request_debug.contains("canary-official-secret"));

        let (translation_socket, _sent, _incoming) = mock_socket();
        let (translation_connector, translation_requests) =
            connector_with_socket(translation_socket);
        let translation = OpenAiTranslationConfig::official(OpenAiCredential::api_key(
            "canary-translation-secret",
        ))
        .with_connector(translation_connector);
        let session = translation.connect(CallOptions::default()).await.unwrap();
        drop(session);
        let snapshot = translation_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .clone();
        assert_eq!(snapshot.route, OpenAiRealtimeRoute::Translation);
        assert_eq!(snapshot.model, OPENAI_REALTIME_TRANSLATION_MODEL);
        assert_eq!(
            snapshot.endpoint,
            "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
        );
        assert!(!snapshot.request_debug.contains("canary-translation-secret"));

        assert!(matches!(
            OpenAiTranslationConfig::official(OpenAiCredential::unauthenticated()).validate(),
            Err(OpenAiRealtimeConfigError::OfficialEndpointRequiresAuthentication)
        ));
    }

    #[test]
    fn bootstrap_resources_are_route_specific_and_secret_safe() {
        let realtime = OpenAiRealtimeBootstrapMetadata::official(
            OpenAiRealtimeRoute::Conversation,
            OPENAI_REALTIME_MODEL,
        )
        .unwrap();
        assert_eq!(
            realtime.websocket_url(),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
        );
        assert_eq!(
            realtime.client_secrets_url(),
            OPENAI_REALTIME_CLIENT_SECRETS_URL
        );
        assert_eq!(
            realtime.webrtc_calls_url(),
            OPENAI_REALTIME_WEBRTC_CALLS_URL
        );

        let request =
            OpenAiRealtimeClientSecretRequest::translation(OPENAI_REALTIME_TRANSLATION_MODEL)
                .unwrap()
                .with_expires_after_seconds(600)
                .unwrap();
        assert_eq!(
            request.endpoint_url(),
            OPENAI_REALTIME_TRANSLATION_CLIENT_SECRETS_URL
        );
        let encoded = serde_json::to_value(&request).unwrap();
        assert_eq!(encoded["expires_after"]["anchor"], "created_at");
        assert_eq!(
            encoded["session"]["model"],
            OPENAI_REALTIME_TRANSLATION_MODEL
        );

        let resource = OpenAiRealtimeClientSecretResource::decode(
            OpenAiRealtimeRoute::Translation,
            r#"{
                "value":"canary-ephemeral-secret",
                "expires_at":1756310470,
                "session":{"type":"translation","model":"gpt-realtime-translate"}
            }"#,
        )
        .unwrap();
        assert_eq!(resource.expose_value(), "canary-ephemeral-secret");
        assert_eq!(resource.model(), Some(OPENAI_REALTIME_TRANSLATION_MODEL));
        assert!(!format!("{resource:?}").contains("canary-ephemeral-secret"));
    }

    #[test]
    fn client_secret_request_validates_expiry_and_redacts_native_session() {
        for seconds in [MIN_CLIENT_SECRET_TTL_SECONDS, MAX_CLIENT_SECRET_TTL_SECONDS] {
            OpenAiRealtimeClientSecretRequest::conversation(OPENAI_REALTIME_MODEL)
                .unwrap()
                .with_expires_after_seconds(seconds)
                .unwrap();
        }

        for seconds in [
            MIN_CLIENT_SECRET_TTL_SECONDS - 1,
            MAX_CLIENT_SECRET_TTL_SECONDS + 1,
        ] {
            assert!(matches!(
                OpenAiRealtimeClientSecretRequest::conversation(OPENAI_REALTIME_MODEL)
                    .unwrap()
                    .with_expires_after_seconds(seconds),
                Err(OpenAiRealtimeConfigError::InvalidClientSecretLifetime)
            ));
        }

        let request = OpenAiRealtimeClientSecretRequest::conversation(OPENAI_REALTIME_MODEL)
            .unwrap()
            .with_session(json!({
                "type": "realtime",
                "model": OPENAI_REALTIME_MODEL,
                "instructions": "canary-native-session-secret",
            }))
            .unwrap();
        let debug = format!("{request:?}");
        assert!(debug.contains(OPENAI_REALTIME_MODEL));
        assert!(!debug.contains("canary-native-session-secret"));
        assert!(!debug.contains("instructions"));
    }

    #[tokio::test]
    async fn socket_failure_becomes_terminal_instead_of_receive_error() {
        let (socket, _sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let session = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector)
        .connect(CallOptions::default())
        .await
        .unwrap();
        incoming
            .send(MockIncoming::Failure(Error::new(
                ErrorKind::Transport,
                "mock established transport failed",
            )))
            .unwrap();

        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::Transport,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn eof_without_close_is_not_reported_as_clean_shutdown() {
        let (socket, _sent, incoming) = mock_socket();
        let (connector, _) = connector_with_socket(socket);
        let session = OpenAiRealtimeConfig::new(
            OpenAiCredential::unauthenticated(),
            OPENAI_REALTIME_MODEL,
            local_endpoint(),
        )
        .with_connector(connector)
        .connect(CallOptions::default())
        .await
        .unwrap();
        incoming.send(MockIncoming::Eof).unwrap();

        assert!(matches!(
            session.receive().await.unwrap(),
            SessionIncoming::Terminal(SessionTerminal::Failed(SessionFailure {
                kind: ErrorKind::UnexpectedEof,
                ..
            }))
        ));
    }
}
