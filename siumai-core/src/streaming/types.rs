//! Core Streaming Types
//!
//! Defines the main types used for streaming responses from LLM providers.

use futures::Stream;
use std::pin::Pin;

use crate::error::LlmError;

// Re-export ChatStreamEvent from types module to avoid duplication
pub use crate::types::ChatStreamEvent;

/// Chat Stream - Main interface for streaming responses
///
/// This is a pinned, boxed stream that yields `ChatStreamEvent` items.
/// All providers implement streaming by returning this type.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>;

/// Chat stream with first-class cancellation handle
///
/// This struct wraps a `ChatStream` with a cancellation handle,
/// allowing the stream to be cancelled at any time.
///
/// # Example
/// ```rust,no_run
/// # use siumai::prelude::*;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Siumai::builder().openai().api_key("key").model("gpt-4").build().await?;
/// let handle = client.chat_stream_with_cancel(vec![user!("Hello")], None).await?;
///
/// // Use the stream
/// // ...
///
/// // Cancel if needed
/// handle.cancel.cancel();
/// # Ok(())
/// # }
/// ```
pub struct ChatStreamHandle {
    /// The underlying chat stream
    pub stream: ChatStream,
    /// Handle to cancel the stream
    pub cancel: crate::utils::cancel::CancelHandle,
}
