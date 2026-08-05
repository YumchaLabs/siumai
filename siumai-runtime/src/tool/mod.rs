//! Explicit local tool bindings and execution contracts.
//!
//! Model-visible definitions and outcomes remain owned by `siumai-core`. This
//! module binds those portable values to trusted host executors without making
//! ordinary model calls capable of executing local code.

mod approval;
mod binding;
mod execution;

pub use approval::{
    ApprovalDecider, ApprovalDecision, ApprovalDecisionError, ApprovalDecisionFuture,
    ApprovalDenial, ApprovalPolicyFingerprint, ApprovalRequest, ExternalApprovalDecider,
};
pub use binding::{
    CatalogFingerprint, ToolBinding, ToolBindingConfigError, ToolSet, ToolSetBuildError,
    ToolSetBuilder, canonical_arguments_digest,
};
pub(crate) use execution::AuthorizedToolCall;
pub use execution::{
    ApprovalPolicy, EffectCertainty, RecoveryPolicy, ToolArgumentError, ToolArgumentValidator,
    ToolConcurrency, ToolEffect, ToolExecutionAttempt, ToolExecutionError, ToolExecutionFuture,
    ToolExecutionRequest, ToolExecutor, ToolIdempotencyKey, ToolIdempotencyKeyError,
    ToolIdempotencyKeyProvider,
};
