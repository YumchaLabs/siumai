use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    ExecutionOwner, ProviderId, ToolBindingIdentity, ToolCall, ToolResult, ToolSpec,
};
use thiserror::Error;

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
/// The call and binding are immutable after resolution. Executing the request
/// validates arguments immediately before dispatch and never performs another
/// lookup by tool name.
#[derive(Clone)]
pub struct ToolExecutionRequest {
    binding: ToolBinding,
    call: ToolCall,
}

impl std::fmt::Debug for ToolExecutionRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolExecutionRequest")
            .field("binding", &self.binding.identity())
            .field("call", &self.call)
            .finish()
    }
}

impl ToolExecutionRequest {
    pub(crate) fn local(binding: ToolBinding, call: ToolCall) -> Self {
        debug_assert!(matches!(&call.owner, ExecutionOwner::Local));
        debug_assert_eq!(binding.name(), call.name.as_str());
        Self { binding, call }
    }

    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    pub fn call_id(&self) -> &str {
        &self.call.id
    }

    pub fn name(&self) -> &str {
        &self.call.name
    }

    pub fn arguments(&self) -> &Value {
        &self.call.arguments
    }

    pub fn owner(&self) -> &ExecutionOwner {
        &self.call.owner
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
        self.binding.recovery_policy()
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
    pub async fn execute(&self) -> Result<ToolResult, ToolExecutionError> {
        self.validate()?;

        let outcome = self.binding.execute(self).await?;
        Ok(ToolResult {
            call_id: self.call.id.clone(),
            name: self.call.name.clone(),
            outcome,
        })
    }
}

/// Failure to resolve, validate, or dispatch a local tool call.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolExecutionError {
    #[error("provider-owned tool `{tool}` cannot resolve to a local binding")]
    ProviderOwnedCall {
        call_id: String,
        tool: String,
        provider: ProviderId,
    },
    #[error("no local binding exists for tool `{tool}`")]
    UnknownLocalTool { call_id: String, tool: String },
    #[error("the frozen binding identity for tool `{tool}` no longer matches the trusted catalog")]
    BindingIdentityMismatch {
        call_id: String,
        tool: String,
        expected: Box<ToolBindingIdentity>,
        actual: Box<ToolBindingIdentity>,
    },
    #[error("the tool call uses an unsupported execution owner")]
    UnsupportedExecutionOwner { call_id: String, tool: String },
    #[error("invalid arguments for tool `{tool}`: {message}")]
    InvalidArguments { tool: String, message: String },
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
