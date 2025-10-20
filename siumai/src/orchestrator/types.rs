//! Core types for orchestrator module.

use std::sync::Arc;

use serde_json::Value;

use super::prepare_step::PrepareStepFn;
use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
use crate::telemetry::TelemetryConfig;
use crate::types::{ChatMessage, FinishReason, ToolCall, Usage};

/// Result of a single step during orchestration.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Messages contributed in this step (assistant + tool outputs).
    pub messages: Vec<ChatMessage>,
    /// Finish reason returned by the model for this step.
    pub finish_reason: Option<FinishReason>,
    /// Usage reported by the provider for this step.
    pub usage: Option<Usage>,
    /// Tool calls requested by the model in this step.
    pub tool_calls: Vec<ToolCall>,
}

impl StepResult {
    /// Merge usage from all steps (helper on the first step result).
    pub fn merge_usage(steps: &[StepResult]) -> Option<Usage> {
        let mut acc: Option<Usage> = None;
        for (idx, s) in steps.iter().enumerate() {
            if let Some(u) = &s.usage {
                match &mut acc {
                    Some(t) => {
                        t.prompt_tokens += u.prompt_tokens;
                        t.completion_tokens += u.completion_tokens;
                        // For aggregated total_tokens, align with expectations:
                        // treat the first step's total as its prompt_tokens only;
                        // subsequent steps add their reported total_tokens.
                        t.total_tokens += if idx == 0 {
                            u.prompt_tokens
                        } else {
                            u.total_tokens
                        };
                        if let Some(c) = u.cached_tokens {
                            t.cached_tokens = Some(t.cached_tokens.unwrap_or(0) + c);
                        }
                        if let Some(r) = u.reasoning_tokens {
                            t.reasoning_tokens = Some(t.reasoning_tokens.unwrap_or(0) + r);
                        }
                    }
                    None => {
                        let mut first = u.clone();
                        first.total_tokens = first.prompt_tokens; // first step: count prompt only
                        acc = Some(first);
                    }
                }
            }
        }
        acc
    }
}

/// Tool approval decision.
#[derive(Debug, Clone)]
pub enum ToolApproval {
    /// Approve tool call with given arguments (can be same as original).
    Approve(Value),
    /// Modify arguments before execution.
    Modify(Value),
    /// Deny tool call with reason; orchestrator will emit an error result as tool message.
    Deny { reason: String },
}

/// A simple tool resolver abstraction.
#[async_trait::async_trait]
pub trait ToolResolver: Send + Sync {
    /// Execute a tool by name with structured JSON arguments.
    /// Returns a structured JSON value as tool output.
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError>;
}

/// Orchestrator options for non-streaming generate.
#[derive(Clone)]
pub struct OrchestratorOptions {
    /// Maximum steps to perform (including the final response step).
    pub max_steps: usize,
    /// Step-finish callback.
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    /// Finish callback with all steps.
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    /// Optional tool approval callback. Allows approve/deny/modify tool arguments.
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional prepare step callback for dynamic step configuration.
    pub prepare_step: Option<PrepareStepFn>,
    /// Optional telemetry configuration.
    pub telemetry: Option<TelemetryConfig>,
}

impl Default for OrchestratorOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            prepare_step: None,
            telemetry: None,
        }
    }
}

/// Orchestrator options for streaming generate.
pub struct OrchestratorStreamOptions {
    pub max_steps: usize,
    pub on_chunk: Option<Arc<dyn Fn(&ChatStreamEvent) + Send + Sync>>,
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    pub on_abort: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    pub telemetry: Option<TelemetryConfig>,
}

impl Default for OrchestratorStreamOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_chunk: None,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            on_abort: None,
            telemetry: None,
        }
    }
}
