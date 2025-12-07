//! Provider-defined tools (re-exported from the core `siumai` crate).
//!
//! This module simply re-exports `siumai::hosted_tools` so that orchestrator
//! and workflow code in `siumai-extras` can use hosted tools without
//! depending on the core crate's module path directly.

pub use siumai::hosted_tools::*;
