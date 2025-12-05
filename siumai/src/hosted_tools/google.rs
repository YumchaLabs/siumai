//! Google Provider-Defined Tools
//!
//! This module provides factory functions for creating Google/Gemini-specific provider-defined tools.
//! These tools are executed by Google's servers.
//!
//! # Examples
//!
//! ```rust
//! use siumai::hosted_tools::google;
//!
//! // Create a code execution tool
//! let code_exec = google::code_execution();
//! ```

use crate::types::{ProviderDefinedTool, Tool};

/// Create a code execution tool
///
/// This tool allows the model to execute Python code in a sandboxed environment.
///
/// # Example
///
/// ```rust
/// use siumai::hosted_tools::google;
///
/// let tool = google::code_execution();
/// ```
pub fn code_execution() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.code_execution",
        "code_execution",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_execution() {
        let tool = code_execution();
        match tool {
            Tool::ProviderDefined(pt) => {
                assert_eq!(pt.id, "google.code_execution");
                assert_eq!(pt.name, "code_execution");
                assert_eq!(pt.provider(), Some("google"));
            }
            _ => panic!("Expected ProviderDefined variant"),
        }
    }
}
