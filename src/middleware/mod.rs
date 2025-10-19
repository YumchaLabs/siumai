//! Middleware module (model-level)
//!
//! This module defines model-level middleware abstractions. They operate above
//! HTTP interceptors and below high-level orchestrators. The primary initial
//! goal is to support parameter transformation prior to provider-specific
//! request mapping. Wrapping generate/stream calls can be added progressively.

pub mod language_model;
pub mod samples;
