//! Anthropic provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need Anthropic-specific behaviors beyond the unified surface.

pub mod hosted_tools;
pub mod structured_output;
pub mod thinking;
pub mod tools;
