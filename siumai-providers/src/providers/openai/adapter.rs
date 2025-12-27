//! OpenAI standard adapter (re-export)
//!
//! The canonical OpenAI-compatible adapter implementation lives in
//! `standards::openai::compat::adapter`. This module re-exports it to preserve
//! existing import paths under `providers::openai`.

pub use crate::standards::openai::compat::adapter::OpenAiStandardAdapter;
