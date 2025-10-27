//! Tracing Events
//!
//! This module defines the various types of events that can be traced.

use crate::types::{ChatMessage, ChatResponse, Tool};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Main tracing event enum
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "data")]
pub enum TracingEvent {
    /// HTTP request/response event
    Http(HttpEvent),
    /// LLM interaction event
    Llm(LlmEvent),
    /// Performance monitoring event
    Performance(PerformanceEvent),
    /// Error event
    Error(ErrorEvent),
    /// Streaming event
    Stream(StreamEvent),
    /// Tool call event
    Tool(ToolEvent),
    /// Chat interaction event
    Chat(ChatEvent),
}

/// HTTP request/response event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Request information
    pub request: HttpRequestInfo,
    /// Response information (if available)
    pub response: Option<HttpResponseInfo>,
    /// Request duration
    pub duration: Option<Duration>,
    /// Error information (if request failed)
    pub error: Option<String>,
    /// Network timing breakdown
    pub timing: Option<NetworkTiming>,
}

/// HTTP request information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequestInfo {
    /// HTTP method
    pub method: String,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body (if included in config)
    pub body: Option<String>,
    /// Request body size in bytes
    pub body_size: u64,
    /// Content type
    pub content_type: Option<String>,
}

/// HTTP response information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponseInfo {
    /// HTTP status code
    pub status_code: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body (if included in config)
    pub body: Option<String>,
    /// Response body size in bytes
    pub body_size: u64,
    /// Content type
    pub content_type: Option<String>,
}

/// Network timing breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTiming {
    /// DNS resolution time
    pub dns_lookup: Option<Duration>,
    /// TCP connection time
    pub tcp_connect: Option<Duration>,
    /// TLS handshake time
    pub tls_handshake: Option<Duration>,
    /// Time to first byte
    pub time_to_first_byte: Option<Duration>,
    /// Content download time
    pub content_download: Option<Duration>,
}

/// LLM interaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Interaction type
    pub interaction_type: LlmInteractionType,
    /// Input messages
    pub input_messages: Vec<ChatMessage>,
    /// Tools available
    pub tools: Option<Vec<Tool>>,
    /// Response (if available)
    pub response: Option<ChatResponse>,
    /// Duration
    pub duration: Option<Duration>,
    /// Token usage
    pub token_usage: Option<TokenUsage>,
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Error (if any)
    pub error: Option<String>,
}

/// Types of LLM interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmInteractionType {
    /// Regular chat completion
    Chat,
    /// Streaming chat completion
    ChatStream,
    /// Text embedding
    Embedding,
    /// Audio transcription
    AudioTranscription,
    /// Audio generation
    AudioGeneration,
    /// Image analysis
    ImageAnalysis,
    /// Image generation
    ImageGeneration,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens
    pub prompt_tokens: u32,
    /// Output tokens
    pub completion_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
    /// Cached tokens (if supported)
    pub cached_tokens: Option<u32>,
    /// Reasoning tokens (for thinking models)
    pub reasoning_tokens: Option<u32>,
}

/// Performance monitoring event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Performance metric type
    pub metric_type: PerformanceMetricType,
    /// Metric value
    pub value: f64,
    /// Metric unit
    pub unit: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetricType {
    /// Request latency
    Latency,
    /// Throughput (requests per second)
    Throughput,
    /// Memory usage
    MemoryUsage,
    /// CPU usage
    CpuUsage,
    /// Connection pool size
    ConnectionPoolSize,
    /// Cache hit rate
    CacheHitRate,
    /// Token generation rate
    TokenGenerationRate,
}

/// Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Error details (as string for serialization)
    pub error: String,
    /// Error context
    pub context: HashMap<String, String>,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Recovery actions taken
    pub recovery_actions: Vec<String>,
    /// Whether the error was retried
    pub retried: bool,
    /// Retry attempt number
    pub retry_attempt: Option<u32>,
}

/// Streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Stream event type
    pub event_type: StreamEventType,
    /// Event data
    pub data: String,
    /// Chunk size in bytes
    pub chunk_size: usize,
    /// Cumulative data size
    pub cumulative_size: usize,
    /// Stream position
    pub position: u64,
    /// Whether this is the final chunk
    pub is_final: bool,
}

/// Types of streaming events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEventType {
    /// Text delta
    TextDelta,
    /// Tool call delta
    ToolCallDelta,
    /// Thinking delta (for reasoning models)
    ThinkingDelta,
    /// Stream start
    StreamStart,
    /// Stream end
    StreamEnd,
    /// Stream error
    StreamError,
}

/// Tool call event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Tool call identifier (if available)
    pub tool_call_id: Option<String>,
    /// Tool name (function name)
    pub tool_name: Option<String>,
    /// Tool execution result
    pub result: Option<String>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Error (if tool call failed)
    pub error: Option<String>,
    /// Tool parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Chat interaction event (high-level)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Chat session ID
    pub session_id: String,
    /// Message exchange
    pub exchange: ChatExchange,
    /// Total duration
    pub duration: Duration,
    /// Token usage
    pub token_usage: Option<TokenUsage>,
    /// Tools used
    pub tools_used: Vec<String>,
    /// Error (if any)
    pub error: Option<String>,
}

/// A complete chat exchange (request + response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatExchange {
    /// Input messages
    pub input: Vec<ChatMessage>,
    /// Output response
    pub output: Option<ChatResponse>,
    /// Whether streaming was used
    pub streaming: bool,
    /// Number of tool calls made
    pub tool_calls_count: u32,
}
