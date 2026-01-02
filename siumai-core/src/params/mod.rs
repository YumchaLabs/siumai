//! Parameter Management Module
//!
//! This module handles parameter mapping, validation, and provider-specific configurations.
//!
//! ## Purpose
//!
//! The `params` module provides utilities for:
//! - **Parameter validation** - Validate parameters before sending to providers
//! - **Cross-provider mapping** - Map parameters between different provider formats
//! - **Provider-specific parameters** - Define provider-specific parameter structures
//!
//! ## Module Organization
//!
//! - **`common`** - Common parameter validation utilities
//! - **`validator`** - Parameter validation logic
//! - **`mapper`** - Cross-provider parameter mapping (internal use)
//!
//! Provider-specific parameter structs are provider-owned and live in provider crates.
//! The `siumai-core::params` module only keeps provider-agnostic validation utilities
//! and legacy OpenAI params until they are fully migrated.
//!
//! ## Relationship with `types` Module
//!
//! - **`types/`** - Contains data structures for requests/responses (what users send/receive)
//! - **`params/`** - Contains parameter validation and mapping utilities (how we process them)
//!
//! ### Example:
//! - `types::CommonParams` - The actual parameter structure
//! - `params::ParameterValidator` - Validates `CommonParams` values
//!
//! ## Usage Guidelines
//!
//! ### For Application Developers
//!
//! Most users don't need to use this module directly. Parameters are typically set through builders:
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//!
//! let client = LlmBuilder::new()
//!     .openai()
//!     .temperature(0.7)  // Automatically validated
//!     .max_tokens(1000)
//!     .build()
//!     .await?;
//! ```
//!
//! ### For Library Developers
//!
//! When adding provider-specific parameters:
//!
//! 1. Define the parameter structure in `params/<provider>.rs`
//! 2. Implement validation logic
//! 3. Add mapping logic in `mapper.rs` if needed
//!
//! ## Provider-Specific Parameters
//!
//! Each provider has unique parameters beyond the common ones:
//!
//! - **OpenAI**: `frequency_penalty`, `presence_penalty`, `logit_bias`, etc.
//! - **Anthropic**: `thinking_budget`, `cache_control`, etc.
//! - **Gemini**: `safety_settings`, `thinking_config`, etc.
//! - **Ollama**: `keep_alive`, `raw`, `format`, etc.

pub mod common;
pub mod mapper;
pub mod validator;

// Re-export main types and traits
pub use common::*;
// pub use mapper::*; // mapper types no longer re-exported; use Transformers instead
pub use validator::*;

// Re-export for backward compatibility (mappers removed from public surface)
