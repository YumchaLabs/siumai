//! Experimental lifecycle vocabulary for provider-owned bidirectional sessions.
//!
//! The core contract deliberately does not define a universal event envelope.
//! Providers retain their native command and event types while sharing only
//! session identity, transport, and terminal lifecycle semantics.

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{Error, ErrorKind, PublicDiagnosticText};

const MAX_SESSION_LINEAGE_ID_BYTES: usize = 512;

/// A stable identity shared by reconnects or transport replacements of one session.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct SessionLineageId(String);

impl SessionLineageId {
    /// Create a provider-owned lineage identifier without changing its spelling.
    pub fn new(value: impl Into<String>) -> Result<Self, SessionLineageIdError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SessionLineageIdError::Empty);
        }
        if value != value.trim() {
            return Err(SessionLineageIdError::SurroundingWhitespace);
        }
        if value.len() > MAX_SESSION_LINEAGE_ID_BYTES {
            return Err(SessionLineageIdError::TooLong {
                maximum: MAX_SESSION_LINEAGE_ID_BYTES,
            });
        }
        if value.chars().any(char::is_control) {
            return Err(SessionLineageIdError::ControlCharacter);
        }
        Ok(Self(value))
    }

    /// Return the provider-owned identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SessionLineageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("SessionLineageId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for SessionLineageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<&str> for SessionLineageId {
    type Error = SessionLineageIdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for SessionLineageId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Why a session lineage identifier is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SessionLineageIdError {
    #[error("session lineage identifier must not be empty")]
    Empty,
    #[error("session lineage identifier must not contain surrounding whitespace")]
    SurroundingWhitespace,
    #[error("session lineage identifier must not exceed {maximum} bytes")]
    TooLong { maximum: usize },
    #[error("session lineage identifier must not contain control characters")]
    ControlCharacter,
}

/// The transport carrying a provider-owned session protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionTransportKind {
    WebSocket,
    WebRtc,
    HttpDuplex,
}

/// Which peer initiated a clean session close.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionCloseOrigin {
    Local,
    Remote,
}

/// Sanitized metadata associated with a clean session close.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseMetadata {
    pub origin: SessionCloseOrigin,
    /// A transport-specific numeric close code, when the transport has one.
    pub transport_code: Option<u32>,
    /// A bounded reason approved for default logs and serialization.
    pub reason: Option<PublicDiagnosticText>,
}

impl SessionCloseMetadata {
    pub fn local() -> Self {
        Self {
            origin: SessionCloseOrigin::Local,
            transport_code: None,
            reason: None,
        }
    }

    pub fn remote() -> Self {
        Self {
            origin: SessionCloseOrigin::Remote,
            transport_code: None,
            reason: None,
        }
    }

    pub fn with_transport_code(mut self, transport_code: u32) -> Self {
        self.transport_code = Some(transport_code);
        self
    }

    pub fn with_reason(mut self, reason: PublicDiagnosticText) -> Self {
        self.reason = Some(reason);
        self
    }
}

/// A caller-requested graceful close.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseRequest {
    /// A transport-specific numeric close code, when supported.
    pub transport_code: Option<u32>,
    /// A bounded reason safe for transport and diagnostic surfaces.
    pub reason: Option<PublicDiagnosticText>,
}

impl SessionCloseRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_transport_code(mut self, transport_code: u32) -> Self {
        self.transport_code = Some(transport_code);
        self
    }

    pub fn with_reason(mut self, reason: PublicDiagnosticText) -> Self {
        self.reason = Some(reason);
        self
    }
}

/// A sanitized failure suitable for session lifecycle events.
///
/// Provider payloads and underlying error sources must remain in the provider's
/// private diagnostics rather than being copied into this serializable value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionFailure {
    pub kind: ErrorKind,
    pub message: PublicDiagnosticText,
    pub retryable: Option<bool>,
}

impl SessionFailure {
    pub fn new(kind: ErrorKind, message: PublicDiagnosticText) -> Self {
        Self {
            kind,
            message,
            retryable: None,
        }
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }
}

/// The final lifecycle state observed for a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionTerminal {
    Closed(SessionCloseMetadata),
    Cancelled {
        reason: Option<PublicDiagnosticText>,
    },
    Expired {
        reason: Option<PublicDiagnosticText>,
    },
    Failed(SessionFailure),
}

/// One provider-owned inbound event or the session's terminal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionIncoming<Event> {
    Event(Event),
    Terminal(SessionTerminal),
}

/// A provider-owned typed bidirectional session.
///
/// `Outbound` and `Inbound` are native provider/protocol types. All methods use
/// shared references so callers can keep one send future and one receive future
/// in flight concurrently. Implementations own any internal serialization and
/// must converge repeated terminal observations on the same terminal state.
#[async_trait]
pub trait ProviderSession<Outbound, Inbound>: Send + Sync
where
    Outbound: Send + 'static,
    Inbound: Send + 'static,
{
    /// Return the identity that remains stable across transport replacement.
    fn lineage_id(&self) -> &SessionLineageId;

    /// Return the active transport class without exposing its implementation.
    fn transport_kind(&self) -> SessionTransportKind;

    /// Send one provider-native command.
    ///
    /// Sending after a terminal state must fail rather than reopen the session.
    async fn send(&self, event: Outbound) -> Result<(), Error>;

    /// Receive one provider-native event or the authoritative terminal state.
    ///
    /// Established transport and provider failures belong in
    /// [`SessionTerminal::Failed`]. `Err` is reserved for failures to perform
    /// the receive operation itself, such as invalid local use.
    async fn receive(&self) -> Result<SessionIncoming<Inbound>, Error>;

    /// Request a graceful local close and return the authoritative terminal.
    ///
    /// Implementations must make this operation idempotent once terminal.
    async fn close(&self, request: SessionCloseRequest) -> Result<SessionTerminal, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_id_preserves_provider_spelling() {
        let id = SessionLineageId::new("session_ABC:turn-7").expect("valid lineage ID");

        assert_eq!(id.as_str(), "session_ABC:turn-7");
    }

    #[test]
    fn lineage_id_rejects_unsafe_values() {
        assert_eq!(
            SessionLineageId::new(" ").expect_err("blank ID must fail"),
            SessionLineageIdError::Empty
        );
        assert_eq!(
            SessionLineageId::new(" session").expect_err("whitespace must fail"),
            SessionLineageIdError::SurroundingWhitespace
        );
        assert_eq!(
            SessionLineageId::new("session\nsecret").expect_err("control byte must fail"),
            SessionLineageIdError::ControlCharacter
        );
    }

    #[test]
    fn lineage_id_deserialization_revalidates_input() {
        let error = serde_json::from_str::<SessionLineageId>("\"bad\\nvalue\"")
            .expect_err("deserialization must validate IDs");

        assert!(error.to_string().contains("control characters"));
    }

    #[test]
    fn terminal_failure_serializes_only_sanitized_fields() {
        let terminal = SessionTerminal::Failed(
            SessionFailure::new(
                ErrorKind::Provider,
                PublicDiagnosticText::new("provider rejected the session")
                    .expect("safe diagnostic"),
            )
            .with_retryable(false),
        );

        let json = serde_json::to_value(&terminal).expect("terminal must serialize");

        assert_eq!(json["Failed"]["kind"], "Provider");
        assert_eq!(json["Failed"]["retryable"], false);
        assert_eq!(json["Failed"]["message"], "provider rejected the session");
    }

    #[allow(dead_code)]
    fn provider_session_is_object_safe_and_concurrent(
        session: &dyn ProviderSession<String, String>,
    ) {
        let send = session.send(String::from("command"));
        let receive = session.receive();
        let close = session.close(SessionCloseRequest::new());
        let _concurrent_futures = (send, receive, close);
    }
}
