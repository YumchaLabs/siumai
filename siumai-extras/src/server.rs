//! Server adapter utilities
//!
//! This module provides utilities for integrating siumai with web frameworks.

/// Axum-specific server adapters
pub mod axum;

/// Tool-loop gateway helpers (execute tools in-process while keeping one downstream stream open).
pub mod tool_loop;
