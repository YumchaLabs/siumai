//! Orchestrator for multi-step tool calling.
//!
//! This module provides a flexible orchestration system for multi-step tool calling with LLMs.
//! It supports:
//! - Flexible stop conditions (step count, specific tool calls, custom conditions)
//! - Dynamic step preparation (modify tools, messages, etc. before each step)
//! - Tool approval workflows
//! - Streaming and non-streaming execution
//! - Full telemetry integration
//!
//! # Example
//!
//! ```rust,ignore
//! use siumai::orchestrator::{generate, step_count_is, OrchestratorOptions};
//!
//! let (response, steps) = generate(
//!     &model,
//!     messages,
//!     Some(tools),
//!     Some(&resolver),
//!     vec![step_count_is(10)],
//!     OrchestratorOptions::default(),
//! ).await?;
//! ```

// Public modules
pub mod agent;
pub mod prepare_step;
pub mod stop_condition;
pub mod types;

// Private modules
mod generate;
mod stream;

// Test modules
#[cfg(test)]
mod tests;

// Re-export public types
pub use agent::ToolLoopAgent;
pub use prepare_step::{PrepareStepContext, PrepareStepFn, PrepareStepResult, ToolChoice};
pub use stop_condition::{
    StopCondition, all_of, any_of, custom_condition, has_no_tool_calls, has_text_response,
    has_tool_call, has_tool_result, step_count_is,
};
pub use types::{
    AgentResult, OrchestratorOptions, OrchestratorStreamOptions, StepResult, ToolApproval,
    ToolExecutionResult, ToolResolver,
};

// Re-export main functions
pub use generate::generate;
pub use stream::{StreamOrchestration, generate_stream, generate_stream_owned};
