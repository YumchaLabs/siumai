//! Chat executor traits
#![allow(clippy::items_after_test_module)]
//!
//! Defines the abstraction that will drive chat operations using transformers
//! and HTTP. For now this is an interface stub for the refactor.

use crate::error::LlmError;
use crate::streaming::ChatStream;
// Telemetry event emission is handled via execution::telemetry helpers
use crate::types::{ChatRequest, ChatResponse};

mod builder;
mod helpers;
mod http;

pub use self::builder::ChatExecutorBuilder;
pub use self::http::HttpChatExecutor;

#[async_trait::async_trait]
pub trait ChatExecutor: Send + Sync {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError>;
    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError>;
}

#[cfg(test)]
mod tests;

// HttpChatExecutor is implemented in the `http` submodule and re-exported above.
