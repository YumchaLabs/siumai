//! Telemetry Events
//!
//! Structured events for LLM operations, compatible with Langfuse and Helicone.

use crate::types::{ChatMessage, ChatResponse, FinishReason, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Main telemetry event enum
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryEvent {
    /// Span start event (for hierarchical tracing)
    SpanStart(SpanEvent),
    /// Span end event
    SpanEnd(SpanEvent),
    /// LLM generation event
    Generation(GenerationEvent),
    /// Tool execution event
    ToolExecution(ToolExecutionEvent),
    /// Orchestrator multi-step event
    Orchestrator(OrchestratorEvent),
}

/// Span event for hierarchical tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Unique span ID
    pub span_id: String,
    /// Parent span ID (if nested)
    pub parent_span_id: Option<String>,
    /// Trace ID (for grouping related spans)
    pub trace_id: String,
    /// Span name (e.g., "ai.generateText", "ai.orchestrator.step")
    pub name: String,
    /// Span start time
    pub start_time: SystemTime,
    /// Span end time (only for SpanEnd events)
    pub end_time: Option<SystemTime>,
    /// Span duration (only for SpanEnd events)
    pub duration: Option<Duration>,
    /// Span attributes
    pub attributes: HashMap<String, String>,
    /// Span status
    pub status: SpanStatus,
    /// Error message (if status is Error)
    pub error: Option<String>,
}

/// Span status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SpanStatus {
    /// Span is in progress
    InProgress,
    /// Span completed successfully
    Ok,
    /// Span completed with error
    Error,
}

/// LLM generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationEvent {
    /// Event ID
    pub id: String,
    /// Trace ID
    pub trace_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: String,
    /// Model ID (e.g., "gpt-4", "claude-3-opus")
    pub model: String,
    /// Input messages (if record_inputs is true)
    pub input: Option<Vec<ChatMessage>>,
    /// Output response (if record_outputs is true)
    pub output: Option<ChatResponse>,
    /// Usage information
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Generation duration
    pub duration: Option<Duration>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Error message (if generation failed)
    pub error: Option<String>,
}

/// Tool execution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionEvent {
    /// Event ID
    pub id: String,
    /// Trace ID
    pub trace_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Tool call identifier (if available)
    pub tool_call_id: Option<String>,
    /// Tool name (function name)
    pub tool_name: Option<String>,
    /// Tool arguments (parsed JSON if available)
    pub arguments: Option<serde_json::Value>,
    /// Tool execution result (if record_outputs is true)
    pub result: Option<String>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Error message (if execution failed)
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Orchestrator multi-step event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorEvent {
    /// Event ID
    pub id: String,
    /// Trace ID
    pub trace_id: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Total number of steps
    pub total_steps: usize,
    /// Current step number
    pub current_step: usize,
    /// Step type
    pub step_type: OrchestratorStepType,
    /// Aggregated usage across all steps
    pub total_usage: Option<Usage>,
    /// Total duration
    pub total_duration: Option<Duration>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Orchestrator step type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrchestratorStepType {
    /// Initial generation
    Generation,
    /// Tool execution
    ToolExecution,
    /// Final completion
    Completion,
}

impl SpanEvent {
    /// Create a new span start event
    pub fn start(
        span_id: String,
        parent_span_id: Option<String>,
        trace_id: String,
        name: String,
    ) -> Self {
        Self {
            span_id,
            parent_span_id,
            trace_id,
            name,
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            attributes: HashMap::new(),
            status: SpanStatus::InProgress,
            error: None,
        }
    }

    /// End the span successfully
    pub fn end_ok(mut self) -> Self {
        let now = SystemTime::now();
        self.end_time = Some(now);
        self.duration = now.duration_since(self.start_time).ok();
        self.status = SpanStatus::Ok;
        self
    }

    /// End the span with error
    pub fn end_error(mut self, error: String) -> Self {
        let now = SystemTime::now();
        self.end_time = Some(now);
        self.duration = now.duration_since(self.start_time).ok();
        self.status = SpanStatus::Error;
        self.error = Some(error);
        self
    }

    /// Add an attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

impl GenerationEvent {
    /// Create a new generation event
    pub fn new(id: String, trace_id: String, provider: String, model: String) -> Self {
        Self {
            id,
            trace_id,
            parent_span_id: None,
            timestamp: SystemTime::now(),
            provider,
            model,
            input: None,
            output: None,
            usage: None,
            finish_reason: None,
            duration: None,
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Set input messages
    pub fn with_input(mut self, input: Vec<ChatMessage>) -> Self {
        self.input = Some(input);
        self
    }

    /// Set output response
    pub fn with_output(mut self, output: ChatResponse) -> Self {
        self.output = Some(output);
        self
    }

    /// Set usage information
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set error
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl ToolExecutionEvent {
    /// Create a new tool execution event
    pub fn new(
        id: String,
        trace_id: String,
        tool_call_id: Option<String>,
        tool_name: Option<String>,
        arguments: Option<serde_json::Value>,
    ) -> Self {
        Self {
            id,
            trace_id,
            parent_span_id: None,
            timestamp: SystemTime::now(),
            tool_call_id,
            tool_name,
            arguments,
            result: None,
            duration: None,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Set result
    pub fn with_result(mut self, result: String) -> Self {
        self.result = Some(result);
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set error
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }
}
