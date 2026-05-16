//! Core data type definitions (re-export).
//!
//! The canonical spec-level types live in `siumai-spec`.

use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

pub use siumai_spec::types::*;

/// A cloneable runtime cancellation handle for request-scoped abort semantics.
#[derive(Clone, Debug, Default)]
pub struct CancelHandle {
    token: CancellationToken,
}

impl CancelHandle {
    /// Create a new cancellation handle.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Future that resolves when cancellation is requested.
    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    /// Clone the underlying cancellation token for integrations that need it directly.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

/// Core runtime request controls use the core-owned cancellation handle.
pub type RequestOptions = siumai_spec::types::RequestOptions<CancelHandle>;

/// Core runtime V4 call controls use the core-owned cancellation handle.
pub type LanguageModelV4CallOptions = siumai_spec::types::LanguageModelV4CallOptions<CancelHandle>;

/// Deprecated combined call settings with core-owned cancellation semantics.
#[allow(deprecated)]
pub type CallSettings = siumai_spec::types::CallSettings<CancelHandle>;

/// Runtime audio stream for streaming TTS/STT providers.
pub type AudioStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<AudioStreamEvent, crate::error::LlmError>> + Send + Sync>,
>;
