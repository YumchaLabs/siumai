//! Protocol standards and compatibility helpers.
//!
//! This module contains protocol-level building blocks that are shared across
//! multiple provider crates.
#![deny(unsafe_code)]

pub mod tool_name_mapping;

pub use tool_name_mapping::{ToolNameMapping, create_tool_name_mapping};
