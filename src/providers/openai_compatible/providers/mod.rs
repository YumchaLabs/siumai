//! OpenAI-Compatible Provider Models and Adapters
//!
//! This module contains model constants and adapters for various OpenAI-compatible providers.

pub mod models;
pub mod siliconflow;

pub use models::*;
// Re-export new adapter system
pub use siliconflow::{SiliconFlowAdapter, SiliconFlowBuilder};
