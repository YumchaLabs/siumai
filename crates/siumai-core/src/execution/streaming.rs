//! Core streaming event traits and types (provider-agnostic)

use crate::error::LlmError;
use crate::types::FinishReasonCore;
use eventsource_stream::Event;
use serde::{Deserialize, Serialize};

/// Minimal chat streaming events for core standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEventCore {
    ContentDelta {
        delta: String,
        index: Option<usize>,
    },
    ToolCallDelta {
        id: Option<String>,
        function_name: Option<String>,
        arguments_delta: Option<String>,
        index: Option<usize>,
    },
    ThinkingDelta {
        delta: String,
    },
    UsageUpdate {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    StreamStart {},
    StreamEnd {
        finish_reason: Option<FinishReasonCore>,
    },
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
    Error {
        error: String,
    },
}

/// Convert SSE events into core chat streaming events
pub trait ChatStreamEventConverterCore: Send + Sync {
    fn provider_id(&self) -> &str;
    fn convert_event(&self, event: Event) -> Vec<Result<ChatStreamEventCore, LlmError>>;
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEventCore, LlmError>> {
        None
    }
}
