//! Server adapter utilities
//!
//! This module provides utilities for integrating siumai with web frameworks.

/// Axum-specific server adapters
#[cfg(feature = "server")]
pub mod axum;
