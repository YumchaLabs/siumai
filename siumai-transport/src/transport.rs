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
    CallOptions, Cancellation, Error, ErrorKind, PublicDiagnosticText, ResponseDiagnostics,
    SensitiveResponse,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::auth::{AuthApplier, AuthContext, AuthRefresh, NoAuth, append_credential_query};
use crate::endpoint::{EndpointConfig, Resolver as ProviderResolver, SystemResolver};
use crate::settings::ProviderHttpTransportSettings;
use crate::{
    EndpointError, HttpTransportRoute, ReplaySafety, RequestBuildError, RequestPlan,
    TransportConfigError, TransportLimits,
};

const ERROR_BODY_CAPTURE_BYTES: usize = 64 * 1024;

/// Opaque correlation token for one provider HTTP transport execution.
///
/// The token carries no provider, model, account, route, URL, or payload
/// identity. It remains stable across retries within one execution.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransportCallId(uuid::Uuid);

impl TransportCallId {
    fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl fmt::Debug for TransportCallId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("TransportCallId")
            .field(&self.0)
            .finish()
    }
}

/// Sanitized reason for consuming another attempt from the logical-call budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryReason {
    Transport,
    Unauthorized,
    RateLimited,
    ServerUnavailable,
}

/// Structural authority that limits whether another attempt may occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryLimit {
    ReplaySafety,
    ProviderPolicy,
    CallerCap,
    ServerDelayPolicy,
    CallDeadline,
}

/// Payload-free result of the transport-owned attempt loop.
///
/// A buffered call ends when its response is ready to return. A streaming call
/// ends when its byte stream is established; later body or protocol events are
/// intentionally outside this observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttemptLoopOutcome {
    ResponseReturned { status: StatusCode },
    StreamEstablished { status: StatusCode },
    Failed { kind: ErrorKind },
    Cancelled,
    TimedOut,
}

/// Read-only, payload-free transport observation.
///
/// Events expose only bounded structural retry and lifecycle data. They never
/// contain URLs, headers, credentials, bodies, prompts, provider error bodies,
/// or provider/model/account identity.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransportEvent {
    AttemptBudgetResolved {
        call_id: TransportCallId,
        provider_maximum_attempts: u8,
        caller_maximum_attempts: Option<u8>,
        effective_maximum_attempts: u8,
        limiting_authority: RetryLimit,
    },
    AttemptStarted {
        call_id: TransportCallId,
        method: Method,
        attempt: u8,
        maximum_attempts: u8,
    },
    ResponseHeadReceived {
        call_id: TransportCallId,
        status: StatusCode,
        attempt: u8,
        retry_reason: Option<RetryReason>,
        server_retry_after: Option<Duration>,
    },
    RetryScheduled {
        call_id: TransportCallId,
        reason: RetryReason,
        completed_attempts: u8,
        delay: Duration,
    },
    RetryDeclined {
        call_id: TransportCallId,
        reason: RetryReason,
        completed_attempts: u8,
        delay: Option<Duration>,
        limiting_authority: RetryLimit,
    },
    AttemptLoopFinished {
        call_id: TransportCallId,
        attempts: u8,
        outcome: AttemptLoopOutcome,
    },
}

impl TransportEvent {
    /// Return the payload-free correlation token for this execution.
    pub fn call_id(&self) -> TransportCallId {
        match self {
            Self::AttemptBudgetResolved { call_id, .. }
            | Self::AttemptStarted { call_id, .. }
            | Self::ResponseHeadReceived { call_id, .. }
            | Self::RetryScheduled { call_id, .. }
            | Self::RetryDeclined { call_id, .. }
            | Self::AttemptLoopFinished { call_id, .. } => *call_id,
        }
    }
}

/// Observer that cannot inspect or mutate URLs, headers, or bodies.
///
/// Observation runs synchronously on the transport execution task. Implementors
/// should perform bounded work and hand off any blocking export asynchronously.
pub trait TransportObserver: Send + Sync {
    fn observe(&self, event: &TransportEvent);
}

/// Provider-owned classification for statuses outside the standard HTTP retry set.
/// The transport retains replay proof, attempt budgeting, backoff, and deadlines.
pub trait RetryClassifier: Send + Sync {
    fn classify_status(&self, status: StatusCode) -> Option<RetryReason>;
}

#[derive(Debug, Default)]
struct NoAdditionalRetryClassifier;

impl RetryClassifier for NoAdditionalRetryClassifier {
    fn classify_status(&self, _status: StatusCode) -> Option<RetryReason> {
        None
    }
}

/// Builder for one provider-runtime-level transport.
pub struct ProviderTransportBuilder {
    endpoint: EndpointConfig,
    resolver: Arc<dyn ProviderResolver>,
    auth: Arc<dyn AuthApplier>,
    retry_classifier: Arc<dyn RetryClassifier>,
    http_transport_settings: ProviderHttpTransportSettings,
}

impl ProviderTransportBuilder {
    pub fn new(endpoint: EndpointConfig) -> Self {
        Self {
            endpoint,
            resolver: Arc::new(SystemResolver),
            auth: Arc::new(NoAuth),
            retry_classifier: Arc::new(NoAdditionalRetryClassifier),
            http_transport_settings: ProviderHttpTransportSettings::default(),
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

    pub fn with_retry_classifier(mut self, classifier: Arc<dyn RetryClassifier>) -> Self {
        self.retry_classifier = classifier;
        self
    }

    /// Apply the complete provider stateless-HTTP infrastructure settings.
    ///
    /// Endpoint, authentication, DNS resolution, provider retry
    /// classification, and per-request replay proof remain separate inputs.
    pub fn with_http_transport_settings(mut self, settings: ProviderHttpTransportSettings) -> Self {
        self.http_transport_settings = settings;
        self
    }

    /// Validate static settings without performing DNS, I/O, or credential work.
    pub fn build(self) -> Result<ProviderTransport, TransportConfigError> {
        self.http_transport_settings.validate()?;
        self.http_transport_settings
            .route()
            .validate_for_endpoint(&self.endpoint)?;
        let limits = self.http_transport_settings.limits();
        let admission_capacity = limits
            .max_in_flight_requests
            .checked_add(limits.max_queued_requests)
            .ok_or(TransportConfigError::CapacityOverflow)?;
        let maximum_in_flight = limits.max_in_flight_requests;
        let client = HttpRouteReqwestClient::build(
            self.http_transport_settings.route(),
            guarded_client_builder(
                self.http_transport_settings.connect_timeout(),
                self.http_transport_settings.read_timeout(),
                limits,
            ),
            Some(&self.endpoint),
            self.resolver,
        )?;
        Ok(ProviderTransport {
            inner: Arc::new(TransportInner {
                endpoint: self.endpoint,
                auth: self.auth,
                retry_classifier: self.retry_classifier,
                http_transport_settings: self.http_transport_settings,
                client,
                admission: Arc::new(Semaphore::new(admission_capacity)),
                in_flight: Arc::new(Semaphore::new(maximum_in_flight)),
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
        self.inner.http_transport_settings.limits()
    }

    /// Execute and buffer one response within the configured decompressed-byte limit.
    pub async fn execute(
        &self,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<TransportResponse, Error> {
        let call_id = TransportCallId::new();
        let pending = self.begin(call_id, plan, options).await?;
        let PendingResponse {
            response,
            attempts,
            controls,
            permits: _permits,
        } = pending;
        let result = async {
            let limits = self.inner.http_transport_settings.limits();
            let status = response.status();
            let headers =
                ResponseHeaders::checked(response.headers().clone(), limits).map_err(|error| {
                    response_limit_error(status, response.headers(), Vec::new(), error)
                })?;
            if let Some(length) = response.content_length()
                && length > limits.max_response_bytes as u64
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
                let next = run_controlled(stream.next(), &controls.cancellation, controls.deadline)
                    .await?;
                let Some(chunk) = next else {
                    break;
                };
                let chunk = chunk.map_err(transport_source_error)?;
                if body.len().saturating_add(chunk.len()) > limits.max_response_bytes {
                    return Err(response_limit_error(
                        status,
                        headers.expose(),
                        bounded_body_prefix(&body, Some(&chunk), limits.max_response_bytes),
                        "response body exceeds the configured limit",
                    ));
                }
                body.extend_from_slice(&chunk);
            }
            Ok(TransportResponse {
                status,
                headers,
                body: body.freeze(),
                attempts,
            })
        }
        .await;

        match result {
            Ok(response) => {
                self.finish_attempt_loop(
                    call_id,
                    attempts,
                    AttemptLoopOutcome::ResponseReturned {
                        status: response.status(),
                    },
                );
                Ok(response)
            }
            Err(error) => Err(self.finish_attempt_loop_error(call_id, attempts, error)),
        }
    }

    /// Establish a response byte stream. Once returned, no transport replay occurs.
    pub async fn execute_stream(
        &self,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<TransportStreamResponse, Error> {
        let call_id = TransportCallId::new();
        let pending = self.begin(call_id, plan, options).await?;
        let attempts = pending.attempts;
        let result = (|| {
            let limits = self.inner.http_transport_settings.limits();
            let status = pending.response.status();
            let headers = ResponseHeaders::checked(pending.response.headers().clone(), limits)
                .map_err(|error| {
                    response_limit_error(status, pending.response.headers(), Vec::new(), error)
                })?;
            if let Some(length) = pending.response.content_length()
                && length > limits.max_response_bytes as u64
            {
                return Err(response_limit_error(
                    status,
                    pending.response.headers(),
                    Vec::new(),
                    "response Content-Length exceeds the configured limit",
                ));
            }
            let body = TransportByteStream::new(
                pending.response.bytes_stream(),
                pending.controls,
                pending.permits,
                limits.max_response_bytes,
                status,
                headers.clone(),
            );
            Ok(TransportStreamResponse {
                status,
                headers,
                body,
                attempts,
            })
        })();

        match result {
            Ok(response) => {
                self.finish_attempt_loop(
                    call_id,
                    attempts,
                    AttemptLoopOutcome::StreamEstablished {
                        status: response.status(),
                    },
                );
                Ok(response)
            }
            Err(error) => Err(self.finish_attempt_loop_error(call_id, attempts, error)),
        }
    }

    async fn begin(
        &self,
        call_id: TransportCallId,
        plan: RequestPlan,
        options: CallOptions,
    ) -> Result<PendingResponse, Error> {
        let controls = CallControls {
            deadline: effective_deadline(
                options.deadline(),
                self.inner.http_transport_settings.call_timeout(),
            ),
            cancellation: options.cancellation().child(),
        };
        let provider_maximum_attempts = self
            .inner
            .http_transport_settings
            .retry_policy()
            .max_attempts();
        let caller_maximum_attempts = options.retry().maximum_attempts();
        let (maximum_attempts, budget_limit) = if !plan.replay_safety().permits_replay() {
            (1, RetryLimit::ReplaySafety)
        } else {
            match caller_maximum_attempts {
                Some(maximum) if maximum < provider_maximum_attempts => {
                    (maximum, RetryLimit::CallerCap)
                }
                _ => (provider_maximum_attempts, RetryLimit::ProviderPolicy),
            }
        };
        self.observe(TransportEvent::AttemptBudgetResolved {
            call_id,
            provider_maximum_attempts,
            caller_maximum_attempts,
            effective_maximum_attempts: maximum_attempts,
            limiting_authority: budget_limit,
        });
        let permits = self
            .acquire(&controls)
            .await
            .map_err(|error| self.finish_attempt_loop_error(call_id, 0, error))?;
        let prepared = plan
            .prepare(self.inner.http_transport_settings.limits())
            .map_err(request_build_error)
            .map_err(|error| self.finish_attempt_loop_error(call_id, 0, error))?;
        let url = self
            .inner
            .endpoint
            .request_url(plan.target())
            .map_err(endpoint_runtime_error)
            .map_err(|error| self.finish_attempt_loop_error(call_id, 0, error))?;
        let idempotency = match plan.replay_safety() {
            ReplaySafety::IdempotencyKey(header) => Some((
                header.name().clone(),
                HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                    .expect("UUIDs are valid header values"),
            )),
            ReplaySafety::Never | ReplaySafety::SemanticallyIdempotent => None,
        };
        let mut attempts = 0_u8;
        let mut refresh = AuthRefresh::Current;
        let mut refreshed_once = false;

        loop {
            attempts = attempts.saturating_add(1);
            self.observe(TransportEvent::AttemptStarted {
                call_id,
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
            let (response, credential_revision) = match attempt {
                Ok(response) => response,
                Err(failure) => {
                    let retryable = failure.is_retryable();
                    if attempts >= maximum_attempts || !retryable {
                        if retryable {
                            self.observe_retry_declined(
                                call_id,
                                RetryReason::Transport,
                                attempts,
                                None,
                                budget_limit,
                            );
                        }
                        let error = failure.into_error();
                        return Err(self.finish_attempt_loop_error(call_id, attempts, error));
                    }
                    let delay = match self.retry_delay(attempts, None, &controls) {
                        RetryDelay::Schedule(delay) => delay,
                        RetryDelay::Decline { .. } => {
                            return Err(self.finish_attempt_loop_error(
                                call_id,
                                attempts,
                                Error::new(
                                    ErrorKind::Internal,
                                    "local retry delay was declined unexpectedly",
                                ),
                            ));
                        }
                        RetryDelay::DeadlineExceeded => {
                            return Err(self.finish_attempt_loop_error(
                                call_id,
                                attempts,
                                Error::new(
                                    ErrorKind::Timeout,
                                    "provider retry delay exceeds the call deadline",
                                ),
                            ));
                        }
                    };
                    if let Err(error) = self
                        .wait_before_retry(
                            call_id,
                            RetryReason::Transport,
                            attempts,
                            delay,
                            &controls,
                        )
                        .await
                    {
                        return Err(self.finish_attempt_loop_error(call_id, attempts, error));
                    }
                    refresh = AuthRefresh::Current;
                    continue;
                }
            };

            if self.inner.client.validate_response_peer(&response).is_err() {
                return Err(self.finish_attempt_loop_error(
                    call_id,
                    attempts,
                    Error::new(
                        ErrorKind::Transport,
                        "connected peer does not match the validated endpoint",
                    ),
                ));
            }
            if let Err(error) = ResponseHeaders::validate(
                response.headers(),
                self.inner.http_transport_settings.limits(),
            ) {
                let error =
                    response_limit_error(response.status(), response.headers(), Vec::new(), error);
                return Err(self.finish_attempt_loop_error(call_id, attempts, error));
            }
            let can_refresh = response.status() == StatusCode::UNAUTHORIZED
                && self.inner.auth.supports_refresh()
                && !refreshed_once
                && credential_revision.is_some();
            let retry_reason = standard_retry_reason(response.status()).or_else(|| {
                self.inner
                    .retry_classifier
                    .classify_status(response.status())
            });
            let retry_reason = match retry_reason {
                Some(RetryReason::Unauthorized) if !can_refresh => None,
                reason => reason,
            };
            let server_retry_after = retry_reason.and_then(|_| retry_after(response.headers()));
            self.observe(TransportEvent::ResponseHeadReceived {
                call_id,
                status: response.status(),
                attempt: attempts,
                retry_reason,
                server_retry_after,
            });

            let Some(reason) = retry_reason else {
                return Ok(PendingResponse {
                    response,
                    attempts,
                    controls,
                    permits,
                });
            };
            if attempts >= maximum_attempts {
                self.observe_retry_declined(
                    call_id,
                    reason,
                    attempts,
                    server_retry_after,
                    budget_limit,
                );
                return Ok(PendingResponse {
                    response,
                    attempts,
                    controls,
                    permits,
                });
            }
            let delay = match self.retry_delay(attempts, server_retry_after, &controls) {
                RetryDelay::Schedule(delay) => delay,
                RetryDelay::Decline {
                    delay,
                    limiting_authority,
                } => {
                    self.observe_retry_declined(
                        call_id,
                        reason,
                        attempts,
                        Some(delay),
                        limiting_authority,
                    );
                    return Ok(PendingResponse {
                        response,
                        attempts,
                        controls,
                        permits,
                    });
                }
                RetryDelay::DeadlineExceeded => {
                    return Err(self.finish_attempt_loop_error(
                        call_id,
                        attempts,
                        Error::new(
                            ErrorKind::Timeout,
                            "provider retry delay exceeds the call deadline",
                        ),
                    ));
                }
            };
            drop(response);
            if let Err(error) = self
                .wait_before_retry(call_id, reason, attempts, delay, &controls)
                .await
            {
                return Err(self.finish_attempt_loop_error(call_id, attempts, error));
            }
            if can_refresh {
                let Some(rejected_revision) = credential_revision else {
                    return Err(self.finish_attempt_loop_error(
                        call_id,
                        attempts,
                        Error::new(
                            ErrorKind::Internal,
                            "credential refresh revision was unavailable",
                        ),
                    ));
                };
                refreshed_once = true;
                refresh = AuthRefresh::AfterUnauthorized { rejected_revision };
            } else {
                refresh = AuthRefresh::Current;
            }
        }
    }

    async fn send_attempt(
        &self,
        request: AttemptRequest<'_>,
    ) -> Result<(reqwest::Response, Option<crate::CredentialRevision>), AttemptFailure> {
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
        let (credential_headers, credential_query, credential_revision) = patch.into_parts();
        for (name, value) in credential_headers.iter() {
            if headers.contains_key(name) {
                return Err(AttemptFailure::Fatal(request_build_error(
                    RequestBuildError::ProtectedHeader,
                )));
            }
            headers.insert(name.clone(), value.clone());
        }
        let limits = self.inner.http_transport_settings.limits();
        if headers.len() > limits.max_header_count
            || headers
                .values()
                .any(|value| value.as_bytes().len() > limits.max_header_value_bytes)
        {
            return Err(AttemptFailure::Fatal(request_build_error(
                RequestBuildError::TooManyHeaders,
            )));
        }
        append_credential_query(&mut url, credential_query)
            .map_err(request_build_error)
            .map_err(AttemptFailure::Fatal)?;
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
        let response = run_controlled(request.send(), &controls.cancellation, controls.deadline)
            .await
            .map_err(AttemptFailure::Fatal)?
            .map_err(|error| AttemptFailure::Replayable(transport_source_error(error)))?;
        Ok((response, credential_revision))
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

    fn retry_delay(
        &self,
        completed_attempts: u8,
        retry_after: Option<Duration>,
        controls: &CallControls,
    ) -> RetryDelay {
        let delay = if let Some(retry_after) = retry_after {
            retry_after
        } else {
            let retry_policy = self.inner.http_transport_settings.retry_policy();
            let upper = retry_policy.backoff_for(completed_attempts);
            if upper.is_zero() || !retry_policy.uses_jitter() {
                upper
            } else {
                let upper_millis = upper.as_millis().min(u128::from(u64::MAX)) as u64;
                Duration::from_millis(rand::random_range(0..=upper_millis))
            }
        };
        if retry_after.is_some()
            && delay
                > self
                    .inner
                    .http_transport_settings
                    .retry_policy()
                    .max_server_delay()
        {
            return RetryDelay::Decline {
                delay,
                limiting_authority: RetryLimit::ServerDelayPolicy,
            };
        }
        let retry_at = Instant::now().checked_add(delay);
        if let Some(deadline) = controls.deadline {
            if retry_at.is_none_or(|retry_at| retry_at >= deadline) {
                return if retry_after.is_some() {
                    RetryDelay::Decline {
                        delay,
                        limiting_authority: RetryLimit::CallDeadline,
                    }
                } else {
                    RetryDelay::DeadlineExceeded
                };
            }
        } else if retry_at.is_none() {
            return if retry_after.is_some() {
                RetryDelay::Decline {
                    delay,
                    limiting_authority: RetryLimit::ServerDelayPolicy,
                }
            } else {
                RetryDelay::DeadlineExceeded
            };
        }
        RetryDelay::Schedule(delay)
    }

    async fn wait_before_retry(
        &self,
        call_id: TransportCallId,
        reason: RetryReason,
        completed_attempts: u8,
        delay: Duration,
        controls: &CallControls,
    ) -> Result<(), Error> {
        self.observe(TransportEvent::RetryScheduled {
            call_id,
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

    fn observe_retry_declined(
        &self,
        call_id: TransportCallId,
        reason: RetryReason,
        completed_attempts: u8,
        delay: Option<Duration>,
        limiting_authority: RetryLimit,
    ) {
        self.observe(TransportEvent::RetryDeclined {
            call_id,
            reason,
            completed_attempts,
            delay,
            limiting_authority,
        });
    }

    fn observe(&self, event: TransportEvent) {
        self.inner
            .http_transport_settings
            .observer()
            .observe(&event);
    }

    fn finish_attempt_loop(
        &self,
        call_id: TransportCallId,
        attempts: u8,
        outcome: AttemptLoopOutcome,
    ) {
        self.observe(TransportEvent::AttemptLoopFinished {
            call_id,
            attempts,
            outcome,
        });
    }

    fn finish_attempt_loop_error(
        &self,
        call_id: TransportCallId,
        attempts: u8,
        error: Error,
    ) -> Error {
        let outcome = match error.kind() {
            ErrorKind::Cancelled => AttemptLoopOutcome::Cancelled,
            ErrorKind::Timeout => AttemptLoopOutcome::TimedOut,
            kind => AttemptLoopOutcome::Failed { kind },
        };
        self.finish_attempt_loop(call_id, attempts, outcome);
        error
    }
}

enum RetryDelay {
    Schedule(Duration),
    Decline {
        delay: Duration,
        limiting_authority: RetryLimit,
    },
    DeadlineExceeded,
}

impl fmt::Debug for ProviderTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderTransport")
            .field("endpoint", &self.inner.endpoint)
            .field(
                "http_transport_settings",
                &self.inner.http_transport_settings,
            )
            .finish()
    }
}

struct TransportInner {
    endpoint: EndpointConfig,
    auth: Arc<dyn AuthApplier>,
    retry_classifier: Arc<dyn RetryClassifier>,
    http_transport_settings: ProviderHttpTransportSettings,
    client: HttpRouteReqwestClient,
    admission: Arc<Semaphore>,
    in_flight: Arc<Semaphore>,
}

struct GuardedDnsResolver {
    endpoint: EndpointConfig,
    resolver: Arc<dyn ProviderResolver>,
}

/// Opaque route-aware reqwest client shared by provider HTTP and MCP.
///
/// This is a workspace integration seam, not a general client-injection API.
/// It keeps proxy endpoint interpretation, credential application, DNS
/// validation, peer validation, environment-proxy disabling, redirects, and
/// reqwest retry policy inside `siumai-transport`.
#[doc(hidden)]
#[derive(Clone)]
pub struct HttpRouteReqwestClient {
    client: reqwest::Client,
    network_endpoint: Option<EndpointConfig>,
}

impl HttpRouteReqwestClient {
    /// Build one route-aware client from caller-owned lifecycle settings.
    ///
    /// `direct_endpoint` is supplied by provider transport so Direct keeps its
    /// existing DNS and peer guard. MCP passes `None` so Direct keeps its
    /// existing origin-resolution behavior; a trusted CONNECT route always
    /// guards the proxy endpoint instead. The caller still owns validation of
    /// the logical request destination; this adapter owns only the selected
    /// network route and peer.
    #[doc(hidden)]
    pub fn build(
        route: &HttpTransportRoute,
        builder: reqwest::ClientBuilder,
        direct_endpoint: Option<&EndpointConfig>,
        resolver: Arc<dyn ProviderResolver>,
    ) -> Result<Self, TransportConfigError> {
        let network_endpoint = route
            .proxy()
            .map(|proxy| proxy.endpoint().clone())
            .or_else(|| direct_endpoint.cloned());
        let mut builder = builder
            .redirect(redirect::Policy::none())
            .referer(false)
            .no_proxy()
            .retry(reqwest::retry::never());
        if let Some(endpoint) = &network_endpoint {
            builder = builder.dns_resolver(GuardedDnsResolver {
                endpoint: endpoint.clone(),
                resolver,
            });
        }
        if let HttpTransportRoute::TrustedConnect { proxy, credential } = route {
            let proxy_config = reqwest::Proxy::https(proxy.url().clone())
                .map_err(|_| TransportConfigError::ClientBuild)?;
            let proxy_config = match credential {
                Some(credential) => credential.apply_to(proxy_config),
                None => proxy_config,
            };
            builder = builder.proxy(proxy_config);
        }
        let client = builder
            .build()
            .map_err(|_| TransportConfigError::ClientBuild)?;
        Ok(Self {
            client,
            network_endpoint,
        })
    }

    /// Start one request without exposing the configured reqwest client.
    #[doc(hidden)]
    pub fn request(&self, method: Method, url: impl reqwest::IntoUrl) -> reqwest::RequestBuilder {
        self.client.request(method, url)
    }

    /// Validate the connected network peer selected for this route.
    #[doc(hidden)]
    pub fn validate_response_peer(
        &self,
        response: &reqwest::Response,
    ) -> Result<(), EndpointError> {
        match (&self.network_endpoint, response.remote_addr()) {
            (Some(endpoint), Some(remote)) => endpoint.validate_remote(remote),
            (Some(_) | None, None) | (None, Some(_)) => Ok(()),
        }
    }
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
    guarded_client_builder(connect_timeout, read_timeout, limits)
        .dns_resolver(GuardedDnsResolver {
            endpoint: endpoint.clone(),
            resolver,
        })
        .build()
}

fn guarded_client_builder(
    connect_timeout: Duration,
    read_timeout: Duration,
    limits: &TransportLimits,
) -> reqwest::ClientBuilder {
    let maximum_header_bytes = limits
        .max_header_count
        .saturating_mul(limits.max_header_value_bytes)
        .min(u32::MAX as usize) as u32;
    reqwest::Client::builder()
        .redirect(redirect::Policy::none())
        .referer(false)
        .no_proxy()
        .retry(reqwest::retry::never())
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .http2_max_header_list_size(maximum_header_bytes)
        .pool_max_idle_per_host(limits.max_connections)
        .tcp_nodelay(true)
        .user_agent(concat!("siumai-transport/", env!("CARGO_PKG_VERSION")))
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

    /// Build the bounded diagnostic context that may accompany an in-band stream failure.
    pub fn diagnostics(&self) -> ResponseDiagnostics {
        let request_id = self
            .0
            .get("x-request-id")
            .or_else(|| self.0.get("request-id"))
            .and_then(public_response_identifier);
        let mut diagnostics = ResponseDiagnostics::default();
        if let Some(request_id) = request_id {
            diagnostics = diagnostics.with_request_id(request_id);
        }
        if let Some(retry_after) = retry_after(&self.0) {
            diagnostics = diagnostics.with_retry_after(retry_after);
        }
        diagnostics
    }
}

fn public_response_identifier(value: &HeaderValue) -> Option<PublicDiagnosticText> {
    let value = value.to_str().ok()?;
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return None;
    }
    PublicDiagnosticText::new(value.to_owned()).ok()
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
    if cancellation.is_cancelled() {
        return Err(Error::cancelled("transport call was cancelled"));
    }
    if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
        return Err(Error::new(ErrorKind::Timeout, "transport deadline elapsed"));
    }
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

fn standard_retry_reason(status: StatusCode) -> Option<RetryReason> {
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
        headers.insert(RETRY_AFTER, HeaderValue::from_static("3"));
        headers.insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_static("request-1"),
        );
        headers.insert(
            HeaderName::from_static("x-ratelimit-api-key"),
            HeaderValue::from_static("sentinel-secret"),
        );
        let headers = ResponseHeaders::checked(headers, &TransportLimits::default()).unwrap();
        assert!(!format!("{headers:?}").contains("canary"));
        let diagnostics = headers.diagnostics();
        assert_eq!(diagnostics.retry_after(), Some(Duration::from_secs(3)));
        assert_eq!(diagnostics.request_id(), Some("request-1"));
        assert!(!format!("{diagnostics:?}").contains("sentinel-secret"));
        assert!(
            !serde_json::to_string(&diagnostics)
                .unwrap()
                .contains("sentinel-secret")
        );
    }
}

#[cfg(test)]
mod call_control_tests {
    use super::*;

    #[tokio::test]
    async fn expired_deadline_wins_over_an_immediately_ready_future() {
        let error = run_controlled(async { 42_u8 }, &Cancellation::new(), Some(Instant::now()))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn pre_cancelled_call_wins_when_deadline_is_also_expired() {
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let error = run_controlled(async { 42_u8 }, &cancellation, Some(Instant::now()))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Cancelled);
    }
}

#[cfg(test)]
mod trusted_connect_tls_tests {
    use super::*;
    use std::net::SocketAddr;

    use async_trait::async_trait;
    use base64::Engine as _;
    use http::header::AUTHORIZATION;
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::ServerConfig;
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

    use crate::{
        CredentialPatch, EndpointError, HttpTransportRoute, ProxyBasicCredential, ProxyEndpoint,
        Resolver,
    };

    // Offline `.example.test` certificate fixtures; these are not credentials.
    const TEST_CA_DER: &str = "MIIBpzCCAU2gAwIBAgIUbTYlr1376Yr/+ZGcs/7LTVRAn9owCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMCExHzAdBgNVBAMMFlNpdW1haSBPZmZsaW5lIFRlc3QgQ0EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATS9UwJf41omyTuG43+oHO6FXHq3G1Msddvu66IHfd3+PDaPqSiY8XDX+Ey5d3EAgKWxzILQUfOPT0QCt9RacK6o2MwYTAdBgNVHQ4EFgQUEpo8dhQlLwIWZeaCSQlAUGUGuu0wHwYDVR0jBBgwFoAUEpo8dhQlLwIWZeaCSQlAUGUGuu0wDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwCgYIKoZIzj0EAwIDSAAwRQIhAO8iTQUm3hTksiBftChN8ziP91dYm3pH5iYJyqa4e5pOAiBbPFSbx5UM0wSdVkmEYs7+B2Dw7gZBCoB1OwphmBlrLg==";
    const PROXY_CERT_DER: &str = "MIIB1zCCAXygAwIBAgIUeauQEUJf7Pl280FQrQf3FSHAFGYwCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMB0xGzAZBgNVBAMMEnByb3h5LmV4YW1wbGUudGVzdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABJOBb+14giMbsCKDztTFG65FCL39GGl9sIl8C/I8AkR5aqjzEB1U9XtFjTHUeabjyodO5JTv3ZB/MhF+EQV6K/WjgZUwgZIwHQYDVR0RBBYwFIIScHJveHkuZXhhbXBsZS50ZXN0MAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMBMB0GA1UdDgQWBBRFpAwYyf389PxKkdIFuETkW59wIzAfBgNVHSMEGDAWgBQSmjx2FCUvAhZl5oJJCUBQZQa67TAKBggqhkjOPQQDAgNJADBGAiEArYfyumOCRVrLAJosZ9O0ecknTdbOe3aCRbcthT7s58sCIQDuq995l+HAm+PoJsx0DhVeKpjPDDkJLuxT8/eOXLEMcg==";
    const PROXY_KEY_DER: &str = "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgYlViovjr5cdWq8MDvDUlYGhAPFNKp5+ExoX0wpYHyCqhRANCAASTgW/teIIjG7Aig87UxRuuRQi9/RhpfbCJfAvyPAJEeWqo8xAdVPV7RY0x1Hmm48qHTuSU792QfzIRfhEFeiv1";
    const PROVIDER_CERT_DER: &str = "MIIB3TCCAYKgAwIBAgIUeauQEUJf7Pl280FQrQf3FSHAFGcwCgYIKoZIzj0EAwIwITEfMB0GA1UEAwwWU2l1bWFpIE9mZmxpbmUgVGVzdCBDQTAeFw0yNjA4MTcwNTQ5MzZaFw0zNjA4MTQwNTQ5MzZaMCAxHjAcBgNVBAMMFXByb3ZpZGVyLmV4YW1wbGUudGVzdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABNrtKG7bQ83W+iw+kj39Wuwp5mE01ezAJbR+9NZeCTvg2YYMSmXnXwDniIviSyJwK3dh8MfSbUIx2GEs81eobQWjgZgwgZUwIAYDVR0RBBkwF4IVcHJvdmlkZXIuZXhhbXBsZS50ZXN0MAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMBMB0GA1UdDgQWBBQ2L3md9NSIZIaY4RUoxlP0eziwaTAfBgNVHSMEGDAWgBQSmjx2FCUvAhZl5oJJCUBQZQa67TAKBggqhkjOPQQDAgNJADBGAiEAiU7zCZp7WT9f1w2OJ6z84Cj4Lo3yASg0RbqGucZTzrkCIQC9Q+REa8I+UT2EVXZl5JobOhaFxYvDrZmyKGZxBEdtiQ==";
    const PROVIDER_KEY_DER: &str = "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgw91aPqnWKZ/kNPOkZC8uPc/RncGLJZDegIMETxlFlTihRANCAATa7Shu20PN1vosPpI9/VrsKeZhNNXswCW0fvTWXgk74NmGDEpl518A54iL4ksicCt3YfDH0m1CMdhhLPNXqG0F";

    struct FixtureResolver(SocketAddr);

    #[async_trait]
    impl Resolver for FixtureResolver {
        async fn resolve(&self, _host: &str, _port: u16) -> Result<Vec<SocketAddr>, EndpointError> {
            Ok(vec![self.0])
        }
    }

    struct ProviderBearerAuth;

    #[async_trait]
    impl AuthApplier for ProviderBearerAuth {
        async fn apply(
            &self,
            _context: AuthContext<'_>,
            _refresh: AuthRefresh,
        ) -> Result<CredentialPatch, Error> {
            CredentialPatch::new()
                .try_insert(
                    AUTHORIZATION,
                    HeaderValue::from_static("Bearer provider-secret"),
                )
                .and_then(|patch| patch.try_insert_query("key", "provider-query-secret"))
                .map_err(|_| {
                    Error::new(
                        ErrorKind::Authentication,
                        "provider credential construction failed",
                    )
                })
        }
    }

    #[tokio::test]
    async fn trusted_connect_preserves_nested_tls_and_credential_phases() {
        let (proxy_address, server) = spawn_nested_tls_proxy(b"ok").await;
        let proxy = ProxyEndpoint::local_explicit(format!(
            "https://proxy.example.test:{}",
            proxy_address.port()
        ))
        .unwrap();
        let route = HttpTransportRoute::trusted_connect(proxy)
            .with_basic_auth(ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap())
            .unwrap();
        let transport = test_transport(
            proxy_address,
            route,
            TransportLimits::default(),
            Arc::new(ProviderBearerAuth),
        );
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            transport.execute(
                RequestPlan::new(Method::GET, crate::RequestTarget::new("models").unwrap()),
                CallOptions::default(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.body(), &Bytes::from_static(b"ok"));

        let (connect, provider_request) = server.await.unwrap();
        assert!(connect.starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n"));
        assert_eq!(
            header(&connect, "proxy-authorization"),
            Some("Basic cHJveHktdXNlcjpwcm94eS1zZWNyZXQ=")
        );
        assert!(header(&connect, "authorization").is_none());
        assert!(!connect.contains("provider-query-secret"));

        assert!(
            provider_request.starts_with("GET /v1/models?key=provider-query-secret HTTP/1.1\r\n")
        );
        assert_eq!(
            header(&provider_request, "authorization"),
            Some("Bearer provider-secret")
        );
        assert!(header(&provider_request, "proxy-authorization").is_none());
    }

    #[tokio::test]
    async fn trusted_connect_preserves_provider_response_bounds() {
        let (proxy_address, server) = spawn_nested_tls_proxy(b"provider-response-secret").await;
        let proxy = ProxyEndpoint::local_explicit(format!(
            "https://proxy.example.test:{}",
            proxy_address.port()
        ))
        .unwrap();
        let route = HttpTransportRoute::trusted_connect(proxy);
        let transport = test_transport(
            proxy_address,
            route,
            TransportLimits {
                max_response_bytes: 4,
                ..TransportLimits::default()
            },
            Arc::new(NoAuth),
        );
        let error = transport
            .execute(
                RequestPlan::new(Method::GET, crate::RequestTarget::new("models").unwrap()),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        for surface in [format!("{error:?}"), error.to_string()] {
            assert!(!surface.contains("provider-response-secret"));
            assert!(!surface.contains("provider.example.test"));
            assert!(!surface.contains("proxy.example.test"));
        }

        let (connect, provider_request) = server.await.unwrap();
        assert!(connect.starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n"));
        assert!(provider_request.starts_with("GET /v1/models HTTP/1.1\r\n"));
    }

    #[tokio::test]
    async fn trusted_connect_preserves_the_shared_admission_bound() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = listener.local_addr().unwrap();
        let (connect_sender, connect_receiver) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let connect = read_http_head(&mut socket).await;
            let _ = connect_sender.send(connect);
            let mut byte = [0_u8; 1];
            while socket.read(&mut byte).await.unwrap_or(0) != 0 {}
        });
        let proxy = ProxyEndpoint::local_explicit(format!(
            "http://proxy.example.test:{}",
            proxy_address.port()
        ))
        .unwrap();
        let limits = TransportLimits {
            max_connections: 1,
            max_in_flight_requests: 1,
            max_queued_requests: 1,
            ..TransportLimits::default()
        };
        let transport = ProviderTransport::builder(
            EndpointConfig::public_custom("https://provider.example.test/v1").unwrap(),
        )
        .with_resolver(Arc::new(FixtureResolver(proxy_address)))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default()
                .with_limits(limits)
                .unwrap()
                .with_route(HttpTransportRoute::trusted_connect(proxy))
                .unwrap(),
        )
        .build()
        .unwrap();

        let first_cancellation = Cancellation::new();
        let first = {
            let transport = transport.clone();
            let cancellation = first_cancellation.clone();
            tokio::spawn(async move {
                transport
                    .execute(
                        RequestPlan::new(Method::GET, crate::RequestTarget::new("first").unwrap()),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            })
        };
        let connect = tokio::time::timeout(Duration::from_secs(1), connect_receiver)
            .await
            .unwrap()
            .unwrap();
        assert!(connect.starts_with("CONNECT provider.example.test:443 HTTP/1.1\r\n"));

        let second_cancellation = Cancellation::new();
        let second = {
            let transport = transport.clone();
            let cancellation = second_cancellation.clone();
            tokio::spawn(async move {
                transport
                    .execute(
                        RequestPlan::new(Method::GET, crate::RequestTarget::new("second").unwrap()),
                        CallOptions::default().with_cancellation(cancellation),
                    )
                    .await
            })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while transport.inner.admission.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the queued call must consume the only queue permit");

        let error = transport
            .execute(
                RequestPlan::new(Method::GET, crate::RequestTarget::new("third").unwrap()),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert_eq!(error.message(), "transport request queue is full");
        for surface in [format!("{error:?}"), error.to_string()] {
            assert!(!surface.contains("provider.example.test"));
            assert!(!surface.contains("proxy.example.test"));
        }

        second_cancellation.cancel();
        assert_eq!(
            second.await.unwrap().unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        first_cancellation.cancel();
        assert_eq!(
            first.await.unwrap().unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap();
    }

    async fn spawn_nested_tls_proxy(
        provider_body: &'static [u8],
    ) -> (SocketAddr, tokio::task::JoinHandle<(String, String)>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = listener.local_addr().unwrap();
        let proxy_acceptor = tls_acceptor(PROXY_CERT_DER, PROXY_KEY_DER);
        let provider_acceptor = tls_acceptor(PROVIDER_CERT_DER, PROVIDER_KEY_DER);
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut proxy_tls = proxy_acceptor.accept(socket).await.unwrap();
            let connect = read_http_head(&mut proxy_tls).await;
            proxy_tls
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
                .unwrap();

            let mut provider_tls = provider_acceptor.accept(proxy_tls).await.unwrap();
            let provider_request = read_http_head(&mut provider_tls).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                provider_body.len()
            );
            provider_tls.write_all(response.as_bytes()).await.unwrap();
            provider_tls.write_all(provider_body).await.unwrap();
            (connect, provider_request)
        });
        (proxy_address, server)
    }

    fn test_transport(
        proxy_address: SocketAddr,
        route: HttpTransportRoute,
        limits: TransportLimits,
        auth: Arc<dyn AuthApplier>,
    ) -> ProviderTransport {
        let settings = ProviderHttpTransportSettings::default()
            .with_limits(limits.clone())
            .unwrap()
            .with_route(route)
            .unwrap();
        let builder =
            guarded_client_builder(Duration::from_secs(2), Duration::from_secs(2), &limits)
                .tls_certs_only([reqwest::Certificate::from_der(&decode(TEST_CA_DER)).unwrap()]);
        let client = HttpRouteReqwestClient::build(
            settings.route(),
            builder,
            Some(&EndpointConfig::public_custom("https://provider.example.test/v1").unwrap()),
            Arc::new(FixtureResolver(proxy_address)),
        )
        .unwrap();
        ProviderTransport {
            inner: Arc::new(TransportInner {
                endpoint: EndpointConfig::public_custom("https://provider.example.test/v1")
                    .unwrap(),
                auth,
                retry_classifier: Arc::new(NoAdditionalRetryClassifier),
                http_transport_settings: settings,
                client,
                admission: Arc::new(Semaphore::new(
                    limits.max_in_flight_requests + limits.max_queued_requests,
                )),
                in_flight: Arc::new(Semaphore::new(limits.max_in_flight_requests)),
            }),
        }
    }

    fn tls_acceptor(certificate: &str, private_key: &str) -> TlsAcceptor {
        let certificate = CertificateDer::from(decode(certificate));
        let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(decode(private_key)));
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate], private_key)
            .unwrap();
        TlsAcceptor::from(Arc::new(config))
    }

    fn decode(value: &str) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(value)
            .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(value))
            .unwrap()
    }

    async fn read_http_head<S>(stream: &mut S) -> String
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let mut bytes = Vec::new();
        loop {
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                return String::from_utf8(bytes[..end + 4].to_vec()).unwrap();
            }
            assert!(
                bytes.len() <= 16 * 1024,
                "fixture HTTP head exceeded its bound"
            );
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read > 0, "fixture connection ended before the HTTP head");
            bytes.extend_from_slice(&chunk[..read]);
        }
    }

    fn header<'a>(head: &'a str, expected: &str) -> Option<&'a str> {
        head.lines().skip(1).find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case(expected).then(|| value.trim())
        })
    }
}
