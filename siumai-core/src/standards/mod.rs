//! Protocol standards and compatibility helpers.
//!
//! This module contains protocol-level building blocks that are shared across
//! multiple provider crates (e.g. OpenAI-compatible adapter helpers).
#![deny(unsafe_code)]

pub mod openai;
pub mod tool_name_mapping;
