//! One bounded HTTP execution path for configured providers.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime};

use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use http::header::{HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};
use http::{Method, StatusCode};
use reqwest::dns::{Addrs, Name, Resolve as ReqwestResolve, Resolving};
use reqwest::redirect;
use siumai_core::{
    CallOptions, Cancellation, Error, ErrorKind, ResponseDiagnostics, RetryIntent,
    SensitiveResponse,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::auth::{AuthApplier, AuthContext, AuthRefresh, NoAuth};
use crate::endpoint::{EndpointConfig, Resolver as ProviderResolver, SystemResolver};
use crate::{
    ReplaySafety, RequestBuildError, RequestPlan, RetryPolicy, TransportConfigError,
    TransportLimits,
};

const ERROR_BODY_CAPTURE_BYTES: usize = 64 * 1024;
const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Sanitized reason for consuming another attempt from the logical-call budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryReason {
    Transport,
    Unauthorized,
    RateLimited,
    ServerUnavailable,
}

/// Read-only, payload-free transport observation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransportEvent {
    AttemptStarted {
        method: Method,
        attempt: u8,
        maximum_attempts: u8,
    },
    ResponseReceived {
        status: StatusCode,
        attempt: u8,
    },
    RetryScheduled {
        reason: RetryReason,
        completed_attempts: u8,
        delay: Duration,
    },
    Completed {
        status: StatusCode,
        attempts: u8,
    },
}

/// Observer that cannot inspect or mutate URLs, headers, or bodies.
pub trait TransportObserver: Send + Sync {
    fn observe(&self, event: &TransportEvent);
}

#[derive(Debug, Default)]
struct NoopObserver;

impl TransportObserver for NoopObserver {
    fn observe(&self, _event: &TransportEvent) {}
}

/// Builder for one provider-runtime-level transport.
pub struct ProviderTransportBuilder {
    endpoint: EndpointConfig,
    resolver: Arc<dyn ProviderResolver>,
    auth: Arc<dyn AuthApplier>,
    observer: Arc<dyn TransportObserver>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Duration,
    call_timeout: Duration,
    read_timeout: Duration,
}

impl ProviderTransportBuilder {
    pub fn new(endpoint: EndpointConfig) -> Self {
        Self {
            endpoint,
            resolver: Arc::new(SystemResolver),
            auth: Arc::new(NoAuth),
            observer: Arc::new(NoopObserver),
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: Duration::from_secs(10),
            call_timeout: DEFAULT_CALL_TIMEOUT,
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
    }

    pub fn with_resolver(mut self, resolver: Arc<dyn ProviderResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    pub fn with_auth(mut self, auth: Arc<dyn AuthApplier>) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_observer(mut self, observer: Arc<dyn TransportObserver>) -> Self {
        self.observer = observer;
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = timeout;
        self
    }

    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
        self
    }

    /// Validate static settings without performing DNS, I/O, or credential work.
    pub fn build(self) -> Result<ProviderTransport, TransportConfigError> {
        self.limits.validate()?;
        for (name, timeout) in [
            ("connect_timeout", self.connect_timeout),
            ("call_timeout", self.call_timeout),
            ("read_timeout", self.read_timeout),
        ] {
            if timeout.is_zero() {
                return Err(TransportConfigError::ZeroTimeout { name });
            }
            if Instant::now().checked_add(timeout).is_none() {
                return Err(TransportConfigError::TimeoutTooLarge { name });
            }
        }
        let admission_capacity = self
            .limits
            .max_in_flight_requests
            .checked_add(self.limits.max_queued_requests)
            .ok_or(TransportConfigError::CapacityOverflow)?;
        let client = build_guarded_client(
            &self.endpoint,
            self.resolver,
            self.connect_timeout,
            self.read_timeout,
            &self.limits,
        )
        .map_err(|_| TransportConfigError::ClientBuild)?;
        Ok(ProviderTransport {
            inner: Arc::new(TransportInner {
                endpoint: self.endpoint,
                auth: self.auth,
                observer: self.observer,
                limits: self.limits.clone(),
                retry_policy: self.retry_policy,
                call_timeout: self.call_timeout,
                client,
                admission: Arc::new(Semaphore::new(admission_capacity)),
                in_flight: Arc::new(Semaphore::new(self.limits.max_in_flight_requests)),
            }),
        })
    }
}

/// Cloneable provider transport. Clones share DNS proof, pool, and admission limits.
#[derive(Clone)]
pub struct ProviderTransport {
    inner: Arc<TransportInner>,
}

impl ProviderTransport {
    pub fn builder(endpoint: EndpointConfig) -> ProviderTransportBuilder {
        ProviderTransportBuilder::new(endpoint)
    }

    pub fn endpoint(&self) -> &EndpointConfig {
        &self.inner.endpoint
    }

    pub fn limits(&self) -> &TransportLimits {
        &self.inner.limits
    }

    /// Execute and buffer one response within the configured decompressed-byte limit.
    pub async fn execute(
        &self,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<TransportResponse, Error> {
        let pending = self.begin(plan, options).await?;
        let PendingResponse {
            response,
            attempts,
            controls,
            permits: _permits,
        } = pending;
        let status = response.status();
        let headers = ResponseHeaders::checked(response.headers().clone(), &self.inner.limits)
            .map_err(|error| response_limit_error(status, response.headers(), Vec::new(), error))?;
        if let Some(length) = response.content_length()
            && length > self.inner.limits.max_response_bytes as u64
        {
            return Err(response_limit_error(
                status,
                response.headers(),
                Vec::new(),
                "response Content-Length exceeds the configured limit",
            ));
        }

        let mut body = BytesMut::new();
        let mut stream = response.bytes_stream();
        loop {
            let next =
                run_controlled(stream.next(), &controls.cancellation, controls.deadline).await?;
            let Some(chunk) = next else {
                break;
            };
            let chunk = chunk.map_err(transport_source_error)?;
            if body.len().saturating_add(chunk.len()) > self.inner.limits.max_response_bytes {
                return Err(response_limit_error(
                    status,
                    headers.expose(),
                    bounded_body_prefix(&body, Some(&chunk), self.inner.limits.max_response_bytes),
                    "response body exceeds the configured limit",
                ));
            }
            body.extend_from_slice(&chunk);
        }
        self.inner
            .observer
            .observe(&TransportEvent::Completed { status, attempts });
        Ok(TransportResponse {
            status,
            headers,
            body: body.freeze(),
            attempts,
        })
    }

    /// Establish a response byte stream. Once returned, no transport replay occurs.
    pub async fn execute_stream(
        &self,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<TransportStreamResponse, Error> {
        let pending = self.begin(plan, options).await?;
        let status = pending.response.status();
        let headers =
            ResponseHeaders::checked(pending.response.headers().clone(), &self.inner.limits)
                .map_err(|error| {
                    response_limit_error(status, pending.response.headers(), Vec::new(), error)
                })?;
        if let Some(length) = pending.response.content_length()
            && length > self.inner.limits.max_response_bytes as u64
        {
            return Err(response_limit_error(
                status,
                pending.response.headers(),
                Vec::new(),
                "response Content-Length exceeds the configured limit",
            ));
        }
        self.inner.observer.observe(&TransportEvent::Completed {
            status,
            attempts: pending.attempts,
        });
        let body = TransportByteStream::new(
            pending.response.bytes_stream(),
            pending.controls,
            pending.permits,
            self.inner.limits.max_response_bytes,
            status,
            headers.clone(),
        );
        Ok(TransportStreamResponse {
            status,
            headers,
            body,
            attempts: pending.attempts,
        })
    }

    async fn begin(
        &self,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<PendingResponse, Error> {
        let controls = CallControls {
            deadline: effective_deadline(options.deadline(), self.inner.call_timeout),
            cancellation: options.cancellation().child(),
        };
        let permits = self.acquire(&controls).await?;
        let prepared = plan
            .prepare(&self.inner.limits)
            .map_err(request_build_error)?;
        let url = self
            .inner
            .endpoint
            .request_url(plan.target())
            .map_err(endpoint_runtime_error)?;
        let retry_allowed =
            options.retry() != RetryIntent::Never && plan.replay_safety().permits_replay();
        let idempotency = match plan.replay_safety() {
            ReplaySafety::IdempotencyKey(header) => Some((
                header.name().clone(),
                HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                    .expect("UUIDs are valid header values"),
            )),
            ReplaySafety::Never | ReplaySafety::SemanticallyIdempotent => None,
        };
        let maximum_attempts = if retry_allowed {
            self.inner.retry_policy.max_attempts()
        } else {
            1
        };
        let mut attempts = 0_u8;
        let mut refresh = AuthRefresh::Current;
        let mut refreshed_once = false;

        loop {
            attempts = attempts.saturating_add(1);
            self.inner
                .observer
                .observe(&TransportEvent::AttemptStarted {
                    method: plan.method().clone(),
                    attempt: attempts,
                    maximum_attempts,
                });
            let attempt = self
                .send_attempt(AttemptRequest {
                    plan: &plan,
                    base_headers: &prepared.headers,
                    body: &prepared.body,
                    url: url.clone(),
                    idempotency: idempotency.as_ref(),
                    refresh,
                    controls: &controls,
                })
                .await;
            let response = match attempt {
                Ok(response) => response,
                Err(failure) => {
                    if attempts >= maximum_attempts || !failure.is_retryable() {
                        return Err(failure.into_error());
                    }
                    self.backoff(RetryReason::Transport, attempts, None, &controls)
                        .await?;
                    refresh = AuthRefresh::Current;
                    continue;
                }
            };

            if let Some(remote) = response.remote_addr()
                && self.inner.endpoint.validate_remote(remote).is_err()
            {
                return Err(Error::new(
                    ErrorKind::Transport,
                    "connected peer does not match the validated endpoint",
                ));
            }
            ResponseHeaders::validate(response.headers(), &self.inner.limits).map_err(|error| {
                response_limit_error(response.status(), response.headers(), Vec::new(), error)
            })?;
            self.inner
                .observer
                .observe(&TransportEvent::ResponseReceived {
                    status: response.status(),
                    attempt: attempts,
                });

            let retry_reason = retry_reason(response.status());
            let can_refresh = response.status() == StatusCode::UNAUTHORIZED
                && self.inner.auth.supports_refresh()
                && !refreshed_once;
            let should_retry = attempts < maximum_attempts
                && retry_reason.is_some()
                && (response.status() != StatusCode::UNAUTHORIZED || can_refresh);
            if !should_retry {
                return Ok(PendingResponse {
                    response,
                    attempts,
                    controls,
                    permits,
                });
            }

            let reason = retry_reason.expect("retry reason was checked");
            let retry_after = retry_after(response.headers());
            drop(response);
            self.backoff(reason, attempts, retry_after, &controls)
                .await?;
            if can_refresh {
                refreshed_once = true;
                refresh = AuthRefresh::AfterUnauthorized;
            } else {
                refresh = AuthRefresh::Current;
            }
        }
    }

    async fn send_attempt(
        &self,
        request: AttemptRequest<'_>,
    ) -> Result<reqwest::Response, AttemptFailure> {
        let AttemptRequest {
            plan,
            base_headers,
            body,
            mut url,
            idempotency,
            refresh,
            controls,
        } = request;
        let mut headers = base_headers.clone();
        if let Some((name, value)) = idempotency
            && headers.insert(name, value.clone()).is_some()
        {
            return Err(AttemptFailure::Fatal(request_build_error(
                RequestBuildError::IdempotencyHeaderConflict,
            )));
        }
        let patch = run_controlled(
            self.inner.auth.apply(
                AuthContext::new(
                    self.inner.endpoint.audience(),
                    plan.method(),
                    &url,
                    &headers,
                    body,
                ),
                refresh,
            ),
            &controls.cancellation,
            controls.deadline,
        )
        .await
        .map_err(AttemptFailure::Fatal)?
        .map_err(AttemptFailure::Fatal)?;
        let (credential_headers, credential_query) = patch.into_parts();
        for (name, value) in credential_headers.iter() {
            if headers.contains_key(name) {
                return Err(AttemptFailure::Fatal(request_build_error(
                    RequestBuildError::ProtectedHeader,
                )));
            }
            headers.insert(name.clone(), value.clone());
        }
        if headers.len() > self.inner.limits.max_header_count
            || headers
                .values()
                .any(|value| value.as_bytes().len() > self.inner.limits.max_header_value_bytes)
        {
            return Err(AttemptFailure::Fatal(request_build_error(
                RequestBuildError::TooManyHeaders,
            )));
        }
        if !credential_query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (name, value) in credential_query {
                pairs.append_pair(&name, &value);
            }
        }
        if !self.inner.endpoint.audience().matches(&url) {
            return Err(AttemptFailure::Fatal(endpoint_runtime_error(
                crate::EndpointError::AudienceMismatch,
            )));
        }

        let request = self
            .inner
            .client
            .request(plan.method().clone(), url)
            .headers(headers)
            .body(body.clone());
        run_controlled(request.send(), &controls.cancellation, controls.deadline)
            .await
            .map_err(AttemptFailure::Fatal)?
            .map_err(|error| AttemptFailure::Replayable(transport_source_error(error)))
    }

    async fn acquire(&self, controls: &CallControls) -> Result<CallPermits, Error> {
        let admission = self
            .inner
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorKind::Transport, "transport request queue is full"))?;
        let in_flight = run_controlled(
            self.inner.in_flight.clone().acquire_owned(),
            &controls.cancellation,
            controls.deadline,
        )
        .await?
        .map_err(|_| Error::new(ErrorKind::Internal, "transport admission control is closed"))?;
        Ok(CallPermits {
            _admission: admission,
            _in_flight: in_flight,
        })
    }

    async fn backoff(
        &self,
        reason: RetryReason,
        completed_attempts: u8,
        retry_after: Option<Duration>,
        controls: &CallControls,
    ) -> Result<(), Error> {
        let delay = if let Some(retry_after) = retry_after {
            retry_after.min(self.inner.retry_policy.max_backoff())
        } else {
            let upper = self.inner.retry_policy.backoff_for(completed_attempts);
            if upper.is_zero() || !self.inner.retry_policy.uses_jitter() {
                upper
            } else {
                let upper_millis = upper.as_millis().min(u128::from(u64::MAX)) as u64;
                Duration::from_millis(rand::random_range(0..=upper_millis))
            }
        };
        self.inner
            .observer
            .observe(&TransportEvent::RetryScheduled {
                reason,
                completed_attempts,
                delay,
            });
        if !delay.is_zero() {
            run_controlled(
                tokio::time::sleep(delay),
                &controls.cancellation,
                controls.deadline,
            )
            .await?;
        }
        Ok(())
    }
}

impl fmt::Debug for ProviderTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderTransport")
            .field("endpoint", &self.inner.endpoint)
            .field("limits", &self.inner.limits)
            .field("retry_policy", &self.inner.retry_policy)
            .finish()
    }
}

struct TransportInner {
    endpoint: EndpointConfig,
    auth: Arc<dyn AuthApplier>,
    observer: Arc<dyn TransportObserver>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    call_timeout: Duration,
    client: reqwest::Client,
    admission: Arc<Semaphore>,
    in_flight: Arc<Semaphore>,
}

struct GuardedDnsResolver {
    endpoint: EndpointConfig,
    resolver: Arc<dyn ProviderResolver>,
}

impl ReqwestResolve for GuardedDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let endpoint = self.endpoint.clone();
        let resolver = self.resolver.clone();
        let requested_host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = endpoint
                .resolve_for_connector(&requested_host, resolver.as_ref())
                .await
                .map_err(|error| {
                    Box::new(error) as Box<dyn std::error::Error + Send + Sync + 'static>
                })?;
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

pub(crate) fn build_guarded_client(
    endpoint: &EndpointConfig,
    resolver: Arc<dyn ProviderResolver>,
    connect_timeout: Duration,
    read_timeout: Duration,
    limits: &TransportLimits,
) -> Result<reqwest::Client, reqwest::Error> {
    let maximum_header_bytes = limits
        .max_header_count
        .saturating_mul(limits.max_header_value_bytes)
        .min(u32::MAX as usize) as u32;
    reqwest::Client::builder()
        .redirect(redirect::Policy::none())
        .referer(false)
        .no_proxy()
        .retry(reqwest::retry::never())
        .dns_resolver(GuardedDnsResolver {
            endpoint: endpoint.clone(),
            resolver,
        })
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .http2_max_header_list_size(maximum_header_bytes)
        .pool_max_idle_per_host(limits.max_connections)
        .tcp_nodelay(true)
        .user_agent(concat!("siumai-transport/", env!("CARGO_PKG_VERSION")))
        .build()
}

struct PendingResponse {
    response: reqwest::Response,
    attempts: u8,
    controls: CallControls,
    permits: CallPermits,
}

struct AttemptRequest<'a> {
    plan: &'a RequestPlan,
    base_headers: &'a HeaderMap,
    body: &'a Bytes,
    url: reqwest::Url,
    idempotency: Option<&'a (HeaderName, HeaderValue)>,
    refresh: AuthRefresh,
    controls: &'a CallControls,
}

enum AttemptFailure {
    Replayable(Error),
    Fatal(Error),
}

impl AttemptFailure {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::Replayable(_))
    }

    fn into_error(self) -> Error {
        match self {
            Self::Replayable(error) | Self::Fatal(error) => error,
        }
    }
}

struct CallControls {
    deadline: Option<Instant>,
    cancellation: Cancellation,
}

struct CallPermits {
    _admission: OwnedSemaphorePermit,
    _in_flight: OwnedSemaphorePermit,
}

/// Response headers with redacted `Debug` and explicit access.
#[derive(Clone)]
pub struct ResponseHeaders(HeaderMap);

impl ResponseHeaders {
    fn checked(headers: HeaderMap, limits: &TransportLimits) -> Result<Self, &'static str> {
        Self::validate(&headers, limits)?;
        Ok(Self(headers))
    }

    pub(crate) fn validate(
        headers: &HeaderMap,
        limits: &TransportLimits,
    ) -> Result<(), &'static str> {
        if headers.len() > limits.max_header_count {
            return Err("response has too many headers");
        }
        if headers
            .values()
            .any(|value| value.as_bytes().len() > limits.max_header_value_bytes)
        {
            return Err("response header exceeds the configured limit");
        }
        Ok(())
    }

    /// Explicit access for protocol and provider error decoding.
    pub fn expose(&self) -> &HeaderMap {
        &self.0
    }

    pub fn get(&self, name: &HeaderName) -> Option<&HeaderValue> {
        self.0.get(name)
    }
}

impl fmt::Debug for ResponseHeaders {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponseHeaders")
            .field("count", &self.0.len())
            .field("contents", &"[REDACTED]")
            .finish()
    }
}

/// Fully buffered transport response. Body access is explicit and `Debug` is redacted.
pub struct TransportResponse {
    status: StatusCode,
    headers: ResponseHeaders,
    body: Bytes,
    attempts: u8,
}

impl TransportResponse {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &ResponseHeaders {
        &self.headers
    }

    pub fn body(&self) -> &Bytes {
        &self.body
    }

    pub fn attempts(&self) -> u8 {
        self.attempts
    }

    pub fn into_parts(self) -> (StatusCode, ResponseHeaders, Bytes) {
        (self.status, self.headers, self.body)
    }
}

impl fmt::Debug for TransportResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransportResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body_bytes", &self.body.len())
            .field("body", &"[REDACTED]")
            .field("attempts", &self.attempts)
            .finish()
    }
}

/// Established response byte stream with bounded consumption and drop cancellation.
type RawByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub struct TransportByteStream {
    inner: Option<RawByteStream>,
    cancellation: Cancellation,
    cancellation_wait: Pin<Box<dyn Future<Output = ()> + Send>>,
    deadline_wait: Option<Pin<Box<tokio::time::Sleep>>>,
    permits: Option<CallPermits>,
    maximum_bytes: usize,
    consumed_bytes: usize,
    status: StatusCode,
    headers: ResponseHeaders,
    terminated: bool,
}

impl TransportByteStream {
    fn new<S>(
        stream: S,
        controls: CallControls,
        permits: CallPermits,
        maximum_bytes: usize,
        status: StatusCode,
        headers: ResponseHeaders,
    ) -> Self
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let cancellation = controls.cancellation;
        let cancellation_wait = {
            let cancellation = cancellation.clone();
            Box::pin(async move { cancellation.cancelled().await })
                as Pin<Box<dyn Future<Output = ()> + Send>>
        };
        let deadline_wait = controls
            .deadline
            .map(|deadline| Box::pin(tokio::time::sleep_until(deadline.into())));
        Self {
            inner: Some(Box::pin(stream)),
            cancellation,
            cancellation_wait,
            deadline_wait,
            permits: Some(permits),
            maximum_bytes,
            consumed_bytes: 0,
            status,
            headers,
            terminated: false,
        }
    }

    fn terminate(&mut self) {
        self.terminated = true;
        self.inner = None;
        self.permits = None;
    }
}

impl fmt::Debug for TransportByteStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransportByteStream")
            .field("consumed_bytes", &self.consumed_bytes)
            .field("maximum_bytes", &self.maximum_bytes)
            .field("terminated", &self.terminated)
            .finish()
    }
}

impl Stream for TransportByteStream {
    type Item = Result<Bytes, Error>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminated {
            return Poll::Ready(None);
        }
        if this.cancellation_wait.as_mut().poll(context).is_ready() {
            this.terminate();
            return Poll::Ready(Some(Err(Error::cancelled("response stream was cancelled"))));
        }
        if this
            .deadline_wait
            .as_mut()
            .is_some_and(|deadline| deadline.as_mut().poll(context).is_ready())
        {
            this.terminate();
            return Poll::Ready(Some(Err(Error::new(
                ErrorKind::Timeout,
                "response stream deadline elapsed",
            ))));
        }
        let Some(inner) = this.inner.as_mut() else {
            this.terminate();
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                this.terminate();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                this.terminate();
                Poll::Ready(Some(Err(transport_source_error(error))))
            }
            Poll::Ready(Some(Ok(chunk))) => {
                this.consumed_bytes = this.consumed_bytes.saturating_add(chunk.len());
                if this.consumed_bytes > this.maximum_bytes {
                    let error = response_limit_error(
                        this.status,
                        this.headers.expose(),
                        Vec::new(),
                        "response body exceeds the configured limit",
                    );
                    this.terminate();
                    Poll::Ready(Some(Err(error)))
                } else {
                    Poll::Ready(Some(Ok(chunk)))
                }
            }
        }
    }
}

impl Drop for TransportByteStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.terminate();
    }
}

/// Metadata and body for an established streaming response.
pub struct TransportStreamResponse {
    status: StatusCode,
    headers: ResponseHeaders,
    body: TransportByteStream,
    attempts: u8,
}

impl TransportStreamResponse {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &ResponseHeaders {
        &self.headers
    }

    pub fn attempts(&self) -> u8 {
        self.attempts
    }

    pub fn into_body(self) -> TransportByteStream {
        self.body
    }

    pub fn into_parts(self) -> (StatusCode, ResponseHeaders, TransportByteStream) {
        (self.status, self.headers, self.body)
    }
}

impl fmt::Debug for TransportStreamResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransportStreamResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body", &self.body)
            .field("attempts", &self.attempts)
            .finish()
    }
}

pub(crate) async fn run_controlled<F>(
    future: F,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<F::Output, Error>
where
    F: Future,
{
    tokio::pin!(future);
    match deadline {
        Some(deadline) => {
            let sleep = tokio::time::sleep_until(deadline.into());
            tokio::pin!(sleep);
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(Error::cancelled("transport call was cancelled")),
                _ = &mut sleep => Err(Error::new(ErrorKind::Timeout, "transport deadline elapsed")),
                output = &mut future => Ok(output),
            }
        }
        None => {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(Error::cancelled("transport call was cancelled")),
                output = &mut future => Ok(output),
            }
        }
    }
}

fn retry_reason(status: StatusCode) -> Option<RetryReason> {
    match status {
        StatusCode::UNAUTHORIZED => Some(RetryReason::Unauthorized),
        StatusCode::TOO_MANY_REQUESTS => Some(RetryReason::RateLimited),
        StatusCode::REQUEST_TIMEOUT
        | StatusCode::TOO_EARLY
        | StatusCode::INTERNAL_SERVER_ERROR
        | StatusCode::BAD_GATEWAY
        | StatusCode::SERVICE_UNAVAILABLE
        | StatusCode::GATEWAY_TIMEOUT => Some(RetryReason::ServerUnavailable),
        _ => None,
    }
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    Some(
        retry_at
            .duration_since(SystemTime::now())
            .unwrap_or_default(),
    )
}

pub(crate) fn effective_deadline(
    explicit: Option<Instant>,
    default_timeout: Duration,
) -> Option<Instant> {
    let default = Instant::now().checked_add(default_timeout);
    match (explicit, default) {
        (Some(explicit), Some(default)) => Some(explicit.min(default)),
        (Some(explicit), None) => Some(explicit),
        (None, default) => default,
    }
}

fn request_build_error(error: RequestBuildError) -> Error {
    Error::new(ErrorKind::InvalidInput, "provider request plan is invalid").with_source(error)
}

fn endpoint_runtime_error(error: crate::EndpointError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "provider endpoint failed security validation",
    )
    .with_source(error)
}

pub(crate) fn transport_source_error(error: reqwest::Error) -> Error {
    let kind = if error.is_timeout() {
        ErrorKind::Timeout
    } else {
        ErrorKind::Transport
    };
    Error::new(kind, "provider transport request failed").with_source(error)
}

pub(crate) fn response_limit_error(
    status: StatusCode,
    headers: &HeaderMap,
    partial_body: Vec<u8>,
    _detail: &'static str,
) -> Error {
    let diagnostics = ResponseDiagnostics::default()
        .with_status(status.as_u16())
        .with_body_truncated(true);
    let raw_headers = headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_owned()))
        })
        .collect::<BTreeMap<_, _>>();
    Error::new(
        ErrorKind::ResponseLimit,
        "provider response exceeded a transport limit",
    )
    .with_diagnostics(diagnostics)
    .with_sensitive_response(SensitiveResponse::new(raw_headers, partial_body))
}

pub(crate) fn bounded_body_prefix(
    current: &[u8],
    overflow: Option<&[u8]>,
    configured_maximum: usize,
) -> Vec<u8> {
    let overflow_length = overflow.map_or(0, <[u8]>::len);
    let maximum = ERROR_BODY_CAPTURE_BYTES.min(configured_maximum);
    let mut prefix = Vec::with_capacity(maximum.min(current.len().saturating_add(overflow_length)));
    let current_bytes = current.len().min(maximum);
    prefix.extend_from_slice(&current[..current_bytes]);
    if let Some(overflow) = overflow {
        let remaining = maximum.saturating_sub(prefix.len());
        prefix.extend_from_slice(&overflow[..overflow.len().min(remaining)]);
    }
    prefix
}

#[cfg(test)]
mod retry_after_tests {
    use super::*;

    #[test]
    fn retry_after_accepts_seconds_and_http_dates() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("3"));
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(3)));

        let future = SystemTime::now() + Duration::from_secs(30);
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&httpdate::fmt_http_date(future)).unwrap(),
        );
        let parsed = retry_after(&headers).unwrap();
        assert!(parsed <= Duration::from_secs(30));
        assert!(parsed >= Duration::from_secs(28));
    }

    #[test]
    fn response_header_debug_redacts_names_and_values() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-canary-header-name"),
            HeaderValue::from_static("canary-header-value"),
        );
        let headers = ResponseHeaders::checked(headers, &TransportLimits::default()).unwrap();
        assert!(!format!("{headers:?}").contains("canary"));
    }
}
