//! Unified `Siumai` interface (facade shim).
//!
//! The canonical implementation lives in `siumai-registry` so `siumai` can remain a thin facade.

pub use siumai_registry::provider::*;
