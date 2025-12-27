//! Streaming Module
//!
//! Unified streaming functionality for all LLM providers.
//! This module consolidates all streaming-related code including:
//! - Chat stream types and events
//! - SSE (Server-Sent Events) handling
//! - Stream factories and converters
//! - UTF-8 safe processing
//! - Stream processors for accumulating content

// Core streaming types
pub mod adapters;
mod builder;
mod converters;
mod events;
mod factory;
mod json_repair;
mod processor;
mod sse;
mod state_tracker;
mod telemetry_wrapper;
mod types;

// Re-exports
pub use adapters::*;
pub use builder::*;
pub use converters::*;
pub use events::*;
pub use factory::*;
#[doc(hidden)]
pub use json_repair::parse_json_with_repair;
pub use processor::*;
pub use sse::*;
pub use state_tracker::*;
pub use telemetry_wrapper::*;
pub use types::*;
