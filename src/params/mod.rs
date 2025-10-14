//! Parameter Management Module
//!
//! Handles parameter mapping, validation, and provider-specific configurations.
//! This module provides a comprehensive parameter system that supports:
//! - Cross-provider parameter mapping
//! - Enhanced validation with detailed error reporting
//! - Parameter optimization for specific providers
//! - Compatibility checking between providers

pub mod anthropic;
pub mod common;
pub mod gemini;
pub mod mapper;
pub mod ollama;
pub mod openai;
pub mod validator;

// Re-export main types and traits
pub use anthropic::*;
pub use common::*;
pub use gemini::*;
// pub use mapper::*; // mapper types no longer re-exported; use Transformers instead
pub use openai::*;
pub use validator::*;

// Re-export for backward compatibility (mappers removed from public surface)
