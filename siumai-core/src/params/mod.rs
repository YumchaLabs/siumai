//! Parameter Management Module
//!
//! This module handles provider-agnostic parameter validation and common mapping helpers.
//!
//! ## Purpose
//!
//! The `params` module provides utilities for:
//! - **Parameter validation** - Validate parameters before sending to providers
//! - **Common mapping helpers** - Convert shared parameters into provider-neutral shapes
//!
//! ## Module Organization
//!
//! - **`common`** - Common parameter validation utilities
//! - **`validator`** - Parameter validation logic
//! - **`mapper`** - Legacy internal mapping helpers
//!
//! Provider-specific parameter structs are provider-owned and live in provider crates.
//! The `siumai-core::params` module only keeps provider-agnostic validation utilities
//! and legacy shared parameter mapping helpers until they are fully migrated.
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
//! Most users don't need to use this module directly. Set shared parameters through family request
//! types, registry-level build options, or provider config structs:
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//!
//! let request = ChatRequest::new(vec![user!("Hello")])
//!     .with_temperature(0.7)
//!     .with_max_tokens(1000);
//! ```
//!
//! ### For Library Developers
//!
//! When adding provider-specific parameters:
//!
//! 1. Define typed provider options in the provider crate.
//! 2. Add request/response extension traits under the provider crate.
//! 3. Re-export the stable provider-owned surface through `siumai::provider_ext`.
//!
//! ## Provider-Specific Parameters
//!
//! Each provider has unique parameters beyond the common ones. Those options belong in provider
//! crates and extension traits, for example:
//!
//! - generation penalty controls
//! - reasoning or cache controls
//! - safety settings
//! - local-runtime controls

pub mod common;
pub mod mapper;
pub mod validator;

// Re-export main types and traits
pub use common::*;
pub use validator::*;
