//! Provider-neutral high-level execution for Siumai language models.
//!
//! Plain [`generate`] and [`stream`] perform exactly one model call. Explicit
//! multi-step tool execution is owned by the runtime's tool-loop APIs.

#![deny(unsafe_code)]

pub mod approval;
#[cfg(feature = "json-schema")]
pub mod json_schema;
pub mod snapshot;
pub mod tool;

mod agent;
mod budget;
mod call;
mod durable;
mod engine;
mod history;
mod options;
mod output;
mod provider_deferred;
mod run;
mod selection;
mod single_step;
mod structured_run;
mod tool_loop;
mod usage;

pub use agent::{Agent, AgentConfigError, AgentInput};
pub use budget::{BudgetError, BudgetKind, BudgetLedger, RunBudget, RunBudgetBuilder, RunTimeouts};
pub use call::{generate, stream};
pub use durable::{
    DurableApproval, DurableResume, DurableRun, DurableRunError, DurableToolLoop,
    IndeterminateRecoveryPolicy,
};
pub use history::{
    HistoryProjectionError, ProjectedHistory, ProjectionLocation, ProjectionLoss,
    ProjectionLossReason, ProjectionPolicy, ProjectionScope, ProjectionSeverity, project_history,
};
#[cfg(feature = "json-schema")]
pub use json_schema::{
    JsonSchemaCompilationError, JsonSchemaError, JsonSchemaValidationError, JsonSchemaValidator,
    JsonSchemaViolation, validate_json,
};
pub use options::{ModelTarget, Runtime, RuntimeBuilder, RuntimeConfigError, StepOptions};
pub use output::{
    OutputDescriptor, OutputDescriptorError, OutputSchemaValidator, RepairPolicy,
    SchemaValidationError, StructuredOutputAttemptKind, StructuredOutputError,
    StructuredOutputFailureKind, StructuredOutputRepair, StructuredOutputResult,
};
pub use provider_deferred::ProviderDeferredObservation;
pub use run::{
    IndeterminateEffect, ModelTransitionOutcome, ModelTransitionRecord, RunEvent, RunReport,
    RunStopReason, RunStream, RunTerminal, RunTimeoutKind, StepRecord, SuspensionReason,
};
pub use selection::{
    StepModelContext, StepModelSelector, StepModelSelectorIdentity, VersionedStepModelSelector,
};
pub use structured_run::{
    StructuredOutputRunError, StructuredOutputRunResult, StructuredOutputRunner,
};
pub use tool_loop::{ToolLoop, ToolOutcomeAction, ToolOutcomePolicy};
