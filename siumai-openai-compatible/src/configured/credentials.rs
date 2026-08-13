use std::fmt;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use http::HeaderValue;
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{Error, ErrorKind};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, CredentialRevision, RequestBuildError,
};
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const DEFAULT_EXPIRY_SKEW: Duration = Duration::from_secs(30);
const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a dynamic source is being consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CredentialRequest {
    MissingOrExpired,
    AfterUnauthorized,
}

/// A bearer credential with optional proactive-refresh time.
#[derive(Clone)]
pub struct BearerCredential {
    secret: SecretString,
    expires_at: Option<Instant>,
}

impl BearerCredential {
    pub fn new(secret: impl Into<String>) -> Self {
        Self {
            secret: SecretString::from(secret.into()),
            expires_at: None,
        }
    }

    pub fn with_expiry(mut self, expires_at: Instant) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    fn validate(&self) -> Result<(), CredentialSourceError> {
        validate_secret(self.secret.expose_secret())?;
        if self
            .expires_at
            .is_some_and(|expiry| expiry <= Instant::now())
        {
            return Err(CredentialSourceError::expired());
        }
        Ok(())
    }

    fn is_usable(&self, skew: Duration) -> bool {
        self.expires_at
            .and_then(|expiry| Instant::now().checked_add(skew).map(|now| expiry > now))
            .unwrap_or(self.expires_at.is_none())
    }
}

impl fmt::Debug for BearerCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BearerCredential")
            .field("secret", &"[REDACTED]")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Sanitized failure returned by a dynamic credential source.
#[derive(Clone)]
pub struct CredentialSourceError {
    kind: CredentialSourceErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CredentialSourceErrorKind {
    Unavailable,
    Invalid,
    Expired,
    TimedOut,
    Cancelled,
}

impl CredentialSourceError {
    pub fn unavailable() -> Self {
        Self {
            kind: CredentialSourceErrorKind::Unavailable,
        }
    }

    fn invalid() -> Self {
        Self {
            kind: CredentialSourceErrorKind::Invalid,
        }
    }

    fn expired() -> Self {
        Self {
            kind: CredentialSourceErrorKind::Expired,
        }
    }

    fn timed_out() -> Self {
        Self {
            kind: CredentialSourceErrorKind::TimedOut,
        }
    }

    fn cancelled() -> Self {
        Self {
            kind: CredentialSourceErrorKind::Cancelled,
        }
    }
}

impl fmt::Debug for CredentialSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialSourceError")
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Display for CredentialSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            CredentialSourceErrorKind::Unavailable => "credential source is unavailable",
            CredentialSourceErrorKind::Invalid => "credential source returned invalid material",
            CredentialSourceErrorKind::Expired => "credential source returned expired material",
            CredentialSourceErrorKind::TimedOut => "credential source timed out",
            CredentialSourceErrorKind::Cancelled => "credential source was cancelled",
        })
    }
}

impl std::error::Error for CredentialSourceError {}

#[async_trait]
pub trait DynamicCredentialSource: Send + Sync {
    async fn load(
        &self,
        request: CredentialRequest,
    ) -> Result<BearerCredential, CredentialSourceError>;
}

/// Explicit authentication mode for one configured provider.
#[derive(Clone)]
pub enum OpenAiCompatibleCredential {
    Static(BearerCredential),
    Dynamic(Arc<dyn DynamicCredentialSource>),
    Unauthenticated,
}

impl OpenAiCompatibleCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::Static(BearerCredential::new(value))
    }

    pub fn dynamic(source: Arc<dyn DynamicCredentialSource>) -> Self {
        Self::Dynamic(source)
    }

    pub fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    #[doc(hidden)]
    pub fn validate_static(&self) -> Result<(), CredentialSourceError> {
        match self {
            Self::Static(credential) => credential.validate(),
            Self::Dynamic(_) | Self::Unauthenticated => Ok(()),
        }
    }

    #[doc(hidden)]
    pub fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::Static(credential) => Arc::new(BearerAuth::Static(credential)),
            Self::Dynamic(source) => {
                Arc::new(BearerAuth::Dynamic(DynamicCredentialManager::new(source)))
            }
            Self::Unauthenticated => Arc::new(siumai_transport::NoAuth),
        }
    }
}

impl fmt::Debug for OpenAiCompatibleCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OpenAiCompatibleCredential")
            .field(&match self {
                Self::Static(_) => "Static([REDACTED])",
                Self::Dynamic(_) => "Dynamic",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

enum BearerAuth {
    Static(BearerCredential),
    Dynamic(DynamicCredentialManager),
}

#[async_trait]
impl AuthApplier for BearerAuth {
    fn supports_refresh(&self) -> bool {
        matches!(self, Self::Dynamic(_))
    }

    async fn apply(
        &self,
        _context: AuthContext<'_>,
        refresh: AuthRefresh,
    ) -> Result<CredentialPatch, Error> {
        let (credential, revision) = match self {
            Self::Static(credential) => (credential.clone(), None),
            Self::Dynamic(manager) => {
                let versioned = manager.load(refresh).await.map_err(credential_error)?;
                (versioned.credential, Some(versioned.revision))
            }
        };
        let value = HeaderValue::from_str(&format!("Bearer {}", credential.secret.expose_secret()))
            .map_err(|source| {
                Error::new(
                    ErrorKind::Authentication,
                    "credential could not be encoded as an authorization header",
                )
                .with_source(source)
            })?;
        let patch = CredentialPatch::new()
            .try_insert(http::header::AUTHORIZATION, value)
            .map_err(request_build_error)?;
        Ok(match revision {
            Some(revision) => patch.with_revision(revision),
            None => patch,
        })
    }
}

fn credential_error(source: CredentialSourceError) -> Error {
    Error::new(
        ErrorKind::Authentication,
        "dynamic credential source failed",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "credential header conflicts with the transport contract",
    )
    .with_source(source)
}

fn validate_secret(value: &str) -> Result<(), CredentialSourceError> {
    if value.is_empty() || value.len() > MAX_CREDENTIAL_BYTES || value.chars().any(char::is_control)
    {
        Err(CredentialSourceError::invalid())
    } else {
        Ok(())
    }
}

#[derive(Clone)]
struct DynamicCredentialManager {
    inner: Arc<DynamicCredentialInner>,
}

struct DynamicCredentialInner {
    source: Arc<dyn DynamicCredentialSource>,
    state: Mutex<CredentialState>,
    shutdown: CancellationToken,
    expiry_skew: Duration,
    load_timeout: Duration,
}

impl Drop for DynamicCredentialInner {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[derive(Default)]
struct CredentialState {
    cached: Option<VersionedCredential>,
    generation: u64,
    flight: Option<CredentialFlight>,
}

#[derive(Clone)]
struct VersionedCredential {
    credential: BearerCredential,
    revision: CredentialRevision,
}

#[derive(Clone)]
struct CredentialFlight {
    generation: u64,
    receiver: watch::Receiver<Option<Result<VersionedCredential, CredentialSourceError>>>,
}

impl DynamicCredentialManager {
    fn new(source: Arc<dyn DynamicCredentialSource>) -> Self {
        Self {
            inner: Arc::new(DynamicCredentialInner {
                source,
                state: Mutex::new(CredentialState::default()),
                shutdown: CancellationToken::new(),
                expiry_skew: DEFAULT_EXPIRY_SKEW,
                load_timeout: DEFAULT_LOAD_TIMEOUT,
            }),
        }
    }

    async fn load(
        &self,
        refresh: AuthRefresh,
    ) -> Result<VersionedCredential, CredentialSourceError> {
        let rejected_revision = match refresh {
            AuthRefresh::Current => None,
            AuthRefresh::AfterUnauthorized { rejected_revision } => Some(rejected_revision),
        };
        let mut state = self.inner.state.lock().await;
        if let Some(cached) = &state.cached
            && cached.credential.is_usable(self.inner.expiry_skew)
            && rejected_revision.is_none_or(|rejected| rejected != cached.revision)
        {
            return Ok(cached.clone());
        }

        let mut receiver = if let Some(flight) = &state.flight {
            flight.receiver.clone()
        } else {
            state.generation = state.generation.wrapping_add(1);
            let generation = state.generation;
            let (sender, receiver) = watch::channel(None);
            state.flight = Some(CredentialFlight {
                generation,
                receiver: receiver.clone(),
            });
            spawn_credential_load(
                Arc::downgrade(&self.inner),
                self.inner.source.clone(),
                self.inner.shutdown.clone(),
                self.inner.load_timeout,
                generation,
                if rejected_revision.is_some() {
                    CredentialRequest::AfterUnauthorized
                } else {
                    CredentialRequest::MissingOrExpired
                },
                sender,
            );
            receiver
        };
        drop(state);

        loop {
            if let Some(result) = receiver.borrow().clone() {
                return result;
            }
            receiver
                .changed()
                .await
                .map_err(|_| CredentialSourceError::cancelled())?;
        }
    }
}

fn spawn_credential_load(
    inner: Weak<DynamicCredentialInner>,
    source: Arc<dyn DynamicCredentialSource>,
    shutdown: CancellationToken,
    timeout: Duration,
    generation: u64,
    request: CredentialRequest,
    sender: watch::Sender<Option<Result<VersionedCredential, CredentialSourceError>>>,
) {
    tokio::spawn(async move {
        let result = tokio::select! {
            biased;
            _ = shutdown.cancelled() => Err(CredentialSourceError::cancelled()),
            result = tokio::time::timeout(timeout, source.load(request)) => {
                match result {
                    Ok(Ok(credential)) => credential.validate().map(|_| credential),
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err(CredentialSourceError::timed_out()),
                }
            }
        };

        let result = result.map(|credential| VersionedCredential {
            credential,
            revision: CredentialRevision::new(generation),
        });

        if let Some(inner) = inner.upgrade() {
            let mut state = inner.state.lock().await;
            if state
                .flight
                .as_ref()
                .is_some_and(|flight| flight.generation == generation)
            {
                if let Ok(credential) = &result {
                    state.cached = Some(credential.clone());
                }
                state.flight = None;
            }
        }
        let _ = sender.send(Some(result));
    });
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::Notify;

    use super::*;

    struct BlockingSource {
        calls: AtomicUsize,
        started: Notify,
        release: Notify,
    }

    #[async_trait]
    impl DynamicCredentialSource for BlockingSource {
        async fn load(
            &self,
            _request: CredentialRequest,
        ) -> Result<BearerCredential, CredentialSourceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            Ok(BearerCredential::new("canary-secret"))
        }
    }

    #[tokio::test]
    async fn concurrent_waiters_share_refresh_and_cancelled_waiter_does_not_cancel_it() {
        let source = Arc::new(BlockingSource {
            calls: AtomicUsize::new(0),
            started: Notify::new(),
            release: Notify::new(),
        });
        let manager = DynamicCredentialManager::new(source.clone());
        let first = tokio::spawn({
            let manager = manager.clone();
            async move { manager.load(AuthRefresh::Current).await }
        });
        source.started.notified().await;

        let mut peers = Vec::new();
        for _ in 0..31 {
            let manager = manager.clone();
            peers.push(tokio::spawn(async move {
                manager.load(AuthRefresh::Current).await
            }));
        }
        first.abort();
        source.release.notify_waiters();

        for peer in peers {
            assert_eq!(
                peer.await
                    .unwrap()
                    .unwrap()
                    .credential
                    .secret
                    .expose_secret(),
                "canary-secret"
            );
        }
        assert_eq!(source.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejected_old_revision_reuses_a_newer_cached_credential() {
        struct RotatingSource(AtomicUsize);

        #[async_trait]
        impl DynamicCredentialSource for RotatingSource {
            async fn load(
                &self,
                _request: CredentialRequest,
            ) -> Result<BearerCredential, CredentialSourceError> {
                let version = self.0.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(BearerCredential::new(format!("secret-{version}")))
            }
        }

        let source = Arc::new(RotatingSource(AtomicUsize::new(0)));
        let manager = DynamicCredentialManager::new(source.clone());
        let first = manager.load(AuthRefresh::Current).await.unwrap();
        let second = manager
            .load(AuthRefresh::AfterUnauthorized {
                rejected_revision: first.revision,
            })
            .await
            .unwrap();
        let reused = manager
            .load(AuthRefresh::AfterUnauthorized {
                rejected_revision: first.revision,
            })
            .await
            .unwrap();

        assert_eq!(second.credential.secret.expose_secret(), "secret-2");
        assert_eq!(reused.credential.secret.expose_secret(), "secret-2");
        assert_eq!(source.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn versioned_credential_debug_does_not_exist_on_public_surfaces() {
        let patch = CredentialPatch::new().with_revision(CredentialRevision::new(7));
        assert!(!format!("{patch:?}").contains("secret"));
    }

    #[test]
    fn credential_debug_is_redacted_and_static_validation_is_synchronous() {
        let credential = OpenAiCompatibleCredential::api_key("canary-secret");
        assert!(!format!("{credential:?}").contains("canary-secret"));
        assert!(
            OpenAiCompatibleCredential::api_key("")
                .validate_static()
                .is_err()
        );
    }
}
