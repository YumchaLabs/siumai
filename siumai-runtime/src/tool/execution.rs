use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{ExecutionOwner, ToolBindingIdentity, ToolCall, ToolResult, ToolSpec};
use thiserror::Error;

use crate::approval::{ApprovalClaimField, VerifiedApproval};

use super::binding::ToolBinding;
use super::binding::canonical_arguments_digest;

/// Whether a tool can change state outside the runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolEffect {
    ReadOnly,
    #[default]
    SideEffecting,
}

/// Per-binding concurrency contract used by the step engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolConcurrency {
    #[default]
    Sequential,
    /// The host explicitly declares this binding safe for concurrent calls.
    SafeParallel { max_in_flight: NonZeroUsize },
}

/// Whether a local call needs host approval before dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApprovalPolicy {
    NotRequired,
    #[default]
    Required,
}

/// What is known about an external effect when execution does not complete normally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EffectCertainty {
    #[default]
    NotDispatched,
    KnownNotApplied,
    Applied,
    Indeterminate,
}

/// Binding-owned recovery contract for an interrupted execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RecoveryPolicy {
    #[default]
    NeverReplay,
    RetryWhenKnownNotApplied,
    /// Indeterminate recovery additionally requires a stable idempotency key.
    ReplayWithStableIdempotencyKey,
}

impl RecoveryPolicy {
    /// Return whether the runtime may retry under the supplied evidence.
    ///
    /// `Applied` is never retried. An indeterminate execution is retried only
    /// when the binding selected the idempotent policy and the engine has a
    /// stable key bound to the frozen request.
    pub fn permits_retry(
        self,
        certainty: EffectCertainty,
        has_stable_idempotency_key: bool,
    ) -> bool {
        match (self, certainty) {
            (_, EffectCertainty::Applied) => false,
            (Self::NeverReplay, _) => false,
            (
                Self::RetryWhenKnownNotApplied | Self::ReplayWithStableIdempotencyKey,
                EffectCertainty::NotDispatched | EffectCertainty::KnownNotApplied,
            ) => true,
            (Self::ReplayWithStableIdempotencyKey, EffectCertainty::Indeterminate) => {
                has_stable_idempotency_key
            }
            _ => false,
        }
    }
}

/// Stable binding-owned key reused for every attempt of one logical tool call.
///
/// The runtime freezes this value before approval or dispatch. Executors can
/// forward it to an external service that provides idempotent request keys.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolIdempotencyKey(String);

impl ToolIdempotencyKey {
    pub fn new(value: impl Into<String>) -> Result<Self, ToolIdempotencyKeyError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ToolIdempotencyKeyError::Empty);
        }
        if value.len() > 512 {
            return Err(ToolIdempotencyKeyError::TooLong {
                actual: value.len(),
                maximum: 512,
            });
        }
        if value.chars().any(char::is_control) {
            return Err(ToolIdempotencyKeyError::ControlCharacter);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ToolIdempotencyKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("ToolIdempotencyKey")
            .field(&"<redacted>")
            .finish()
    }
}

/// Invalid stable idempotency key returned by a trusted binding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolIdempotencyKeyError {
    #[error("tool idempotency key must not be empty")]
    Empty,
    #[error("tool idempotency key is {actual} bytes; maximum is {maximum}")]
    TooLong { actual: usize, maximum: usize },
    #[error("tool idempotency key must not contain control characters")]
    ControlCharacter,
}

/// Binding-owned synchronous seam for deriving a stable logical-call key.
///
/// Implementations must return the same key whenever the same frozen call is
/// restored. Changes to the derivation contract require a binding revision
/// change so snapshots and approvals receive a new binding fingerprint.
pub trait ToolIdempotencyKeyProvider: Send + Sync + 'static {
    fn stable_key(&self, call: &ToolCall) -> Result<ToolIdempotencyKey, ToolIdempotencyKeyError>;
}

/// One-based attempt number frozen into the executor request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolExecutionAttempt(u32);

impl ToolExecutionAttempt {
    pub const INITIAL: Self = Self(1);

    pub fn get(self) -> u32 {
        self.0
    }

    pub(crate) fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// A rejected tool argument payload.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct ToolArgumentError {
    message: String,
}

impl ToolArgumentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Always-on validation seam for untrusted model arguments.
pub trait ToolArgumentValidator: Send + Sync + 'static {
    fn validate(&self, arguments: &Value) -> Result<(), ToolArgumentError>;
}

/// Boxed future returned by an object-safe [`ToolExecutor`].
pub type ToolExecutionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<siumai_core::ToolOutcome, ToolExecutionError>> + Send + 'a>>;

/// Trusted host implementation for one local tool binding.
pub trait ToolExecutor: Send + Sync + 'static {
    fn execute<'a>(&'a self, request: &'a ToolExecutionRequest) -> ToolExecutionFuture<'a>;
}

/// A frozen local call bound to the exact executor selected by its [`ToolSet`](super::ToolSet).
///
/// The call, binding, recovery contract, stable idempotency key, and attempt
/// are immutable after resolution. This public value deliberately has no
/// execution method: local code can run only after the runtime constructs a
/// single-use authorization permit.
///
/// ```compile_fail
/// use siumai_runtime::tool::ToolExecutionRequest;
///
/// async fn bypass_approval(request: ToolExecutionRequest) {
///     let _ = request.execute().await;
/// }
/// ```
#[derive(Clone)]
pub struct ToolExecutionRequest {
    binding: ToolBinding,
    call: ToolCall,
    idempotency_key: Option<ToolIdempotencyKey>,
    recovery_policy: RecoveryPolicy,
    attempt: ToolExecutionAttempt,
}

impl std::fmt::Debug for ToolExecutionRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolExecutionRequest")
            .field("binding", &self.binding.identity())
            .field("call_id", &self.call.id())
            .field("tool", &self.call.name())
            .field("owner", &self.call.owner())
            .field("arguments", &"<redacted>")
            .field("has_idempotency_key", &self.idempotency_key.is_some())
            .field("recovery_policy", &self.recovery_policy)
            .field("attempt", &self.attempt)
            .finish()
    }
}

impl ToolExecutionRequest {
    pub(crate) fn local(binding: ToolBinding, call: ToolCall) -> Result<Self, ToolExecutionError> {
        debug_assert!(matches!(call.owner(), ExecutionOwner::Local));
        debug_assert_eq!(binding.name(), call.name());
        let idempotency_key = binding.stable_idempotency_key(&call).map_err(|source| {
            ToolExecutionError::InvalidIdempotencyKey {
                call_id: call.id().to_owned(),
                tool: call.name().to_owned(),
                message: source.to_string(),
            }
        })?;
        let recovery_policy = binding.recovery_policy();
        Ok(Self {
            binding,
            call,
            idempotency_key,
            recovery_policy,
            attempt: ToolExecutionAttempt::INITIAL,
        })
    }

    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn call_id(&self) -> &str {
        self.call.id()
    }

    pub fn name(&self) -> &str {
        self.call.name()
    }

    pub fn arguments(&self) -> &Value {
        self.call.arguments()
    }

    pub fn owner(&self) -> &ExecutionOwner {
        self.call.owner()
    }

    pub fn binding_identity(&self) -> &ToolBindingIdentity {
        self.binding.identity()
    }

    pub fn canonical_arguments_digest(&self) -> String {
        canonical_arguments_digest(self.arguments())
    }

    pub fn spec(&self) -> &ToolSpec {
        self.binding.spec()
    }

    pub fn effect(&self) -> ToolEffect {
        self.binding.effect()
    }

    pub fn concurrency(&self) -> ToolConcurrency {
        self.binding.concurrency()
    }

    pub fn approval_policy(&self) -> ApprovalPolicy {
        self.binding.approval_policy()
    }

    pub fn recovery_policy(&self) -> RecoveryPolicy {
        self.recovery_policy
    }

    pub fn idempotency_key(&self) -> Option<&ToolIdempotencyKey> {
        self.idempotency_key.as_ref()
    }

    pub fn attempt(&self) -> ToolExecutionAttempt {
        self.attempt
    }

    /// Return whether this exact frozen request may be retried under the
    /// supplied effect evidence.
    pub fn permits_retry(&self, certainty: EffectCertainty) -> bool {
        self.recovery_policy
            .permits_retry(certainty, self.idempotency_key.is_some())
    }

    /// Validate arguments before the engine records `Prepared` or requests approval.
    ///
    /// The request already owns the exact frozen binding, so a successful
    /// validation can be persisted together with its binding identity.
    pub fn validate(&self) -> Result<(), ToolExecutionError> {
        self.binding.validate(self.arguments()).map_err(|source| {
            ToolExecutionError::InvalidArguments {
                tool: self.name().to_string(),
                message: source.message,
            }
        })
    }

    /// Validate and dispatch the exact frozen binding.
    ///
    /// Executor errors retain their effect certainty. In particular, an
    /// indeterminate side effect is never converted into a normal tool outcome.
    pub(crate) async fn execute(&self) -> Result<ToolResult, ToolExecutionError> {
        self.validate()?;

        let outcome = self.binding.execute(self).await?;
        Ok(ToolResult {
            call_id: self.call.id().to_owned(),
            name: self.call.name().to_owned(),
            outcome,
        })
    }

    /// Authorize a binding that explicitly opted out of approval.
    pub(crate) fn authorize_not_required(
        self,
    ) -> Result<AuthorizedToolCall, ToolAuthorizationError> {
        if self.approval_policy() != ApprovalPolicy::NotRequired {
            return Err(ToolAuthorizationError::ApprovalRequired {
                call_id: self.call.id().to_owned(),
                tool: self.call.name().to_owned(),
            });
        }
        Ok(AuthorizedToolCall::new(
            self,
            AuthorizationEvidence::NotRequired,
        ))
    }

    /// Authorize a call through an explicit trusted host auto-approve policy.
    pub(crate) fn authorize_host_auto_approved(self) -> AuthorizedToolCall {
        AuthorizedToolCall::new(self, AuthorizationEvidence::HostAutoApproved)
    }

    /// Bind a consumed external approval to this exact frozen request.
    pub(crate) fn authorize_verified(
        self,
        approval: VerifiedApproval,
    ) -> Result<AuthorizedToolCall, ToolAuthorizationError> {
        let claims = approval.claims();
        require_approval_field(
            claims.execution_owner() == self.owner(),
            ApprovalClaimField::ExecutionOwner,
        )?;
        require_approval_field(
            claims.binding_identity() == self.binding_identity(),
            ApprovalClaimField::BindingIdentity,
        )?;
        require_approval_field(
            claims.tool_call_id() == self.call_id(),
            ApprovalClaimField::ToolCallId,
        )?;
        require_approval_field(
            claims.canonical_arguments_digest() == self.canonical_arguments_digest(),
            ApprovalClaimField::CanonicalArgumentsDigest,
        )?;
        Ok(AuthorizedToolCall::new(
            self,
            AuthorizationEvidence::Verified(Box::new(approval)),
        ))
    }

    /// Create the next attempt without changing the frozen binding, call, key,
    /// or recovery contract.
    pub(crate) fn next_attempt(
        &self,
        certainty: EffectCertainty,
    ) -> Result<Self, ToolAuthorizationError> {
        if !self.permits_retry(certainty) {
            return Err(ToolAuthorizationError::RecoveryNotPermitted {
                call_id: self.call.id().to_owned(),
                tool: self.call.name().to_owned(),
                certainty,
            });
        }
        let attempt =
            self.attempt
                .next()
                .ok_or_else(|| ToolAuthorizationError::AttemptOverflow {
                    call_id: self.call.id().to_owned(),
                    tool: self.call.name().to_owned(),
                })?;
        let mut next = self.clone();
        next.attempt = attempt;
        Ok(next)
    }
}

fn require_approval_field(
    matches: bool,
    field: ApprovalClaimField,
) -> Result<(), ToolAuthorizationError> {
    if matches {
        Ok(())
    } else {
        Err(ToolAuthorizationError::VerifiedApprovalMismatch { field })
    }
}

enum AuthorizationEvidence {
    NotRequired,
    HostAutoApproved,
    Verified(Box<VerifiedApproval>),
}

impl AuthorizationEvidence {
    /// Consume the single-use proof exactly when dispatch crosses the trusted
    /// executor boundary. Verified claims remain owned by the permit until
    /// this point and cannot be reused to construct another permit.
    fn consume(self) {
        if let Self::Verified(approval) = self {
            let _ = (*approval).into_claims();
        }
    }
}

impl std::fmt::Debug for AuthorizationEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NotRequired => "NotRequired",
            Self::HostAutoApproved => "HostAutoApproved",
            Self::Verified(_) => "Verified",
        })
    }
}

/// Single-use proof that the exact frozen request may be dispatched.
///
/// This type intentionally does not implement `Clone`. Construction and
/// dispatch remain crate-private so public callers cannot bypass `ToolLoop`.
pub(crate) struct AuthorizedToolCall {
    request: ToolExecutionRequest,
    evidence: AuthorizationEvidence,
}

impl std::fmt::Debug for AuthorizedToolCall {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AuthorizedToolCall")
            .field("request", &self.request)
            .field("evidence", &self.evidence)
            .finish()
    }
}

impl AuthorizedToolCall {
    fn new(request: ToolExecutionRequest, evidence: AuthorizationEvidence) -> Self {
        Self { request, evidence }
    }

    pub(crate) fn request(&self) -> &ToolExecutionRequest {
        &self.request
    }

    /// Consume the permit and dispatch the already-frozen executor directly.
    pub(crate) async fn dispatch(self) -> Result<ToolResult, ToolExecutionError> {
        let Self { request, evidence } = self;
        evidence.consume();
        request.execute().await
    }
}

/// Failure to authorize or recover a frozen local tool call.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub(crate) enum ToolAuthorizationError {
    #[error("tool `{tool}` requires approval before dispatch")]
    ApprovalRequired { call_id: String, tool: String },
    #[error("verified approval does not match frozen field {field:?}")]
    VerifiedApprovalMismatch { field: ApprovalClaimField },
    #[error("tool `{tool}` cannot recover from effect certainty {certainty:?}")]
    RecoveryNotPermitted {
        call_id: String,
        tool: String,
        certainty: EffectCertainty,
    },
    #[error("tool `{tool}` execution attempt overflowed")]
    AttemptOverflow { call_id: String, tool: String },
}

/// Failure to resolve, validate, or dispatch a local tool call.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolExecutionError {
    #[error("no local binding exists for tool `{tool}`")]
    UnknownLocalTool { call_id: String, tool: String },
    #[error("the frozen binding identity for tool `{tool}` no longer matches the trusted catalog")]
    BindingIdentityMismatch {
        call_id: String,
        tool: String,
        expected: Box<ToolBindingIdentity>,
        actual: Box<ToolBindingIdentity>,
    },
    #[error("invalid arguments for tool `{tool}`: {message}")]
    InvalidArguments { tool: String, message: String },
    #[error("tool `{tool}` produced an invalid stable idempotency key: {message}")]
    InvalidIdempotencyKey {
        call_id: String,
        tool: String,
        message: String,
    },
    #[error("tool `{tool}` executor failed: {message}")]
    ExecutorFailed {
        tool: String,
        message: String,
        retryable: bool,
        certainty: EffectCertainty,
    },
}

impl ToolExecutionError {
    pub fn executor_failed(
        tool: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
        certainty: EffectCertainty,
    ) -> Self {
        Self::ExecutorFailed {
            tool: tool.into(),
            message: message.into(),
            retryable,
            certainty,
        }
    }

    /// Effect evidence available to the step engine after this failure.
    pub fn effect_certainty(&self) -> EffectCertainty {
        match self {
            Self::ExecutorFailed { certainty, .. } => *certainty,
            _ => EffectCertainty::NotDispatched,
        }
    }
}
