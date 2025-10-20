//! LLM Interaction Tracing
//!
//! This module provides tracing capabilities for LLM interactions.

use super::events::{
    ChatEvent, ChatExchange, LlmEvent, LlmInteractionType, StreamEvent, StreamEventType,
    TokenUsage, ToolEvent, TracingEvent,
};
use super::{SpanId, TraceId};
use crate::types::{ChatMessage, ChatResponse, Tool, ToolCall};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn};

/// Parameters for recording a chat exchange
pub struct ExchangeParams {
    pub input: Vec<ChatMessage>,
    pub output: Option<ChatResponse>,
    pub duration: Duration,
    pub streaming: bool,
    pub tools_used: Vec<String>,
    pub token_usage: Option<TokenUsage>,
    pub error: Option<String>,
}

/// LLM interaction tracer
#[derive(Debug, Clone)]
pub struct LlmTracer {
    /// Provider name
    provider: String,
    /// Model name
    model: Option<String>,
    /// Whether to include message content
    include_content: bool,
    /// Maximum content length to log
    max_content_length: usize,
}

impl LlmTracer {
    /// Create a new LLM tracer
    pub fn new(provider: String, model: Option<String>) -> Self {
        Self {
            provider,
            model,
            include_content: true,
            max_content_length: 1000,
        }
    }

    /// Configure content inclusion
    pub fn with_content_settings(mut self, include: bool, max_length: usize) -> Self {
        self.include_content = include;
        self.max_content_length = max_length;
        self
    }

    /// Start tracing a chat interaction
    pub fn start_chat(
        &self,
        trace_id: TraceId,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        parameters: HashMap<String, Value>,
    ) -> ChatContext {
        let span_id = SpanId::new();
        let start_time = Instant::now();

        info!(
            trace_id = %trace_id,
            span_id = %span_id,
            provider = %self.provider,
            model = ?self.model,
            message_count = messages.len(),
            tool_count = tools.map(|t| t.len()).unwrap_or(0),
            "LLM chat interaction started"
        );

        if self.include_content {
            debug!(
                trace_id = %trace_id,
                span_id = %span_id,
                messages = ?self.truncate_messages(messages),
                tools = ?tools,
                parameters = ?parameters,
                "LLM chat details"
            );
        }

        ChatContext {
            trace_id,
            span_id,
            start_time,
            provider: self.provider.clone(),
            model: self.model.clone(),
            input_messages: messages.to_vec(),
            tools: tools.map(|t| t.to_vec()),
            parameters,
            response: None,
            error: None,
        }
    }

    /// End tracing a chat interaction
    pub fn end_chat(
        &self,
        mut context: ChatContext,
        result: Result<ChatResponse, crate::error::LlmError>,
    ) -> ChatContext {
        let duration = context.start_time.elapsed();

        match result {
            Ok(response) => {
                context.response = Some(response.clone());

                info!(
                    trace_id = %context.trace_id,
                    span_id = %context.span_id,
                    duration_ms = duration.as_millis(),
                    "LLM chat interaction completed successfully"
                );

                if self.include_content {
                    debug!(
                        trace_id = %context.trace_id,
                        span_id = %context.span_id,
                        response = ?self.truncate_response(&response),
                        "LLM chat response"
                    );
                }
            }
            Err(error) => {
                context.error = Some(error.to_string());

                warn!(
                    trace_id = %context.trace_id,
                    span_id = %context.span_id,
                    error = %error,
                    duration_ms = duration.as_millis(),
                    "LLM chat interaction failed"
                );
            }
        }

        context
    }

    /// Create a tracing event from chat context
    pub fn create_chat_event(&self, context: &ChatContext) -> TracingEvent {
        let duration = context.start_time.elapsed();

        TracingEvent::Llm(LlmEvent {
            timestamp: SystemTime::now(),
            provider: context.provider.clone(),
            model: context.model.clone().unwrap_or_default(),
            interaction_type: LlmInteractionType::Chat,
            input_messages: if self.include_content {
                self.truncate_messages(&context.input_messages)
            } else {
                vec![]
            },
            tools: context.tools.clone(),
            response: if self.include_content {
                context
                    .response
                    .as_ref()
                    .map(|r| self.truncate_response(r).clone())
            } else {
                None
            },
            duration: Some(duration),
            token_usage: context
                .response
                .as_ref()
                .and_then(|r| self.extract_token_usage(r)),
            parameters: context.parameters.clone(),
            error: context.error.clone(),
        })
    }

    /// Truncate messages for logging
    fn truncate_messages(&self, messages: &[ChatMessage]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|msg| self.truncate_message(msg))
            .collect()
    }

    /// Truncate a single message
    fn truncate_message(&self, message: &ChatMessage) -> ChatMessage {
        // This would need to be implemented based on your ChatMessage structure
        // For now, returning a clone
        message.clone()
    }

    /// Truncate response for logging
    fn truncate_response(&self, response: &ChatResponse) -> ChatResponse {
        // This would need to be implemented based on your ChatResponse structure
        // For now, returning a clone
        response.clone()
    }

    /// Extract token usage from response
    fn extract_token_usage(&self, _response: &ChatResponse) -> Option<TokenUsage> {
        // This would need to be implemented based on your ChatResponse structure
        None
    }
}

/// Chat interaction context
#[derive(Debug, Clone)]
pub struct ChatContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub start_time: Instant,
    pub provider: String,
    pub model: Option<String>,
    pub input_messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Tool>>,
    pub parameters: HashMap<String, Value>,
    pub response: Option<ChatResponse>,
    pub error: Option<String>,
}

/// Stream tracer for monitoring streaming responses
#[derive(Debug, Clone)]
pub struct StreamTracer {
    /// Trace ID
    trace_id: TraceId,
    /// Span ID
    span_id: SpanId,
    /// Cumulative data size
    cumulative_size: usize,
    /// Stream position
    position: u64,
}

impl StreamTracer {
    /// Create a new stream tracer
    pub fn new(trace_id: TraceId) -> Self {
        Self {
            trace_id,
            span_id: SpanId::new(),
            cumulative_size: 0,
            position: 0,
        }
    }

    /// Record a stream event
    pub fn record_event(
        &mut self,
        event_type: StreamEventType,
        data: String,
        is_final: bool,
    ) -> TracingEvent {
        let chunk_size = data.len();
        self.cumulative_size += chunk_size;
        self.position += 1;

        debug!(
            trace_id = %self.trace_id,
            span_id = %self.span_id,
            event_type = ?event_type,
            chunk_size = chunk_size,
            cumulative_size = self.cumulative_size,
            position = self.position,
            is_final = is_final,
            "Stream event recorded"
        );

        TracingEvent::Stream(StreamEvent {
            timestamp: SystemTime::now(),
            event_type,
            data,
            chunk_size,
            cumulative_size: self.cumulative_size,
            position: self.position,
            is_final,
        })
    }
}

/// Tool call tracer
#[derive(Debug, Clone)]
pub struct ToolTracer {
    /// Trace ID
    trace_id: TraceId,
    /// Include tool parameters and results
    include_details: bool,
}

impl ToolTracer {
    /// Create a new tool tracer
    pub fn new(trace_id: TraceId, include_details: bool) -> Self {
        Self {
            trace_id,
            include_details,
        }
    }

    /// Record a tool call
    pub fn record_tool_call(
        &self,
        tool_call: &ToolCall,
        result: Option<String>,
        duration: Option<Duration>,
        error: Option<String>,
    ) -> TracingEvent {
        let span_id = SpanId::new();

        info!(
            trace_id = %self.trace_id,
            span_id = %span_id,
            tool_name = ?tool_call.function.as_ref().map(|f| &f.name),
            duration_ms = duration.map(|d| d.as_millis()),
            success = error.is_none(),
            "Tool call recorded"
        );

        let parameters = if self.include_details {
            tool_call
                .function
                .as_ref()
                .and_then(|f| serde_json::from_str(&f.arguments).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        TracingEvent::Tool(ToolEvent {
            timestamp: SystemTime::now(),
            tool_call: tool_call.clone(),
            result: if self.include_details { result } else { None },
            duration,
            error,
            parameters,
        })
    }
}

/// Chat tracer for high-level chat session tracking
#[derive(Debug, Clone)]
pub struct ChatTracer {
    /// Session ID
    session_id: String,
    /// Include detailed information
    include_details: bool,
}

impl ChatTracer {
    /// Create a new chat tracer
    pub fn new(session_id: String, include_details: bool) -> Self {
        Self {
            session_id,
            include_details,
        }
    }

    /// Record a complete chat exchange
    pub fn record_exchange(&self, params: ExchangeParams) -> TracingEvent {
        info!(
            session_id = %self.session_id,
            duration_ms = params.duration.as_millis(),
            streaming = params.streaming,
            tools_count = params.tools_used.len(),
            success = params.error.is_none(),
            "Chat exchange recorded"
        );

        TracingEvent::Chat(ChatEvent {
            timestamp: SystemTime::now(),
            session_id: self.session_id.clone(),
            exchange: ChatExchange {
                input: if self.include_details {
                    params.input
                } else {
                    vec![]
                },
                output: if self.include_details {
                    params.output
                } else {
                    None
                },
                streaming: params.streaming,
                tool_calls_count: params.tools_used.len() as u32,
            },
            duration: params.duration,
            token_usage: params.token_usage,
            tools_used: params.tools_used,
            error: params.error,
        })
    }
}
