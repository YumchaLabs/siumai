//! OpenAI-Compatible Provider Models and Adapters
//!
//! This module contains model constants and adapters for various OpenAI-compatible providers.

pub mod deepseek;
pub mod models;
pub mod openrouter;
pub mod siliconflow;

pub use models::*;
// Re-export new adapter system
pub use deepseek::{DeepSeekAdapter, DeepSeekBuilder};
pub use openrouter::{OpenRouterAdapter, OpenRouterBuilder};
pub use siliconflow::{SiliconFlowAdapter, SiliconFlowBuilder};
