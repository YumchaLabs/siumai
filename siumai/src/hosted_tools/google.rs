//! Google Provider-Defined Tools
//!
//! Factory functions for creating Google/Gemini-specific provider-defined tools.
//! These tools are executed by Google's servers.

use crate::types::{ProviderDefinedTool, Tool};

/// Create a code execution tool
///
/// This tool allows the model to execute Python code in a sandboxed environment.
pub fn code_execution() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.code_execution",
        "code_execution",
    ))
}
