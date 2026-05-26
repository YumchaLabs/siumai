//! Tool contract validation helpers.

use thiserror::Error;

/// Validation error for portable tool names and provider-tool ids.
///
/// Siumai keeps legacy constructors infallible for compatibility. Provider request shapers and
/// callers that need an early failure mode should use the fallible constructors or
/// `validate_contract()` before projecting tools to provider requests.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ToolNameValidationError {
    /// Tool name does not satisfy the portable lookup-key contract.
    #[error("invalid tool name `{name}`: {reason}")]
    InvalidToolName {
        /// Rejected tool name.
        name: String,
        /// Static validation reason.
        reason: &'static str,
    },

    /// Provider tool id does not satisfy the `<provider>.<tool>` contract.
    #[error("invalid provider tool id `{id}`: {reason}")]
    InvalidProviderToolId {
        /// Rejected provider tool id.
        id: String,
        /// Static validation reason.
        reason: &'static str,
    },
}

/// Validate a portable tool name.
///
/// This is the shared provider-agnostic contract for Siumai tool lookup keys: names must be
/// non-empty, unpadded, and free of whitespace or control characters. Individual providers may
/// impose stricter rules when lowering a request.
pub fn validate_tool_name(name: &str) -> Result<(), ToolNameValidationError> {
    if name.is_empty() {
        return Err(invalid_tool_name(name, "must not be empty"));
    }

    if name.trim() != name {
        return Err(invalid_tool_name(
            name,
            "must not contain leading or trailing whitespace",
        ));
    }

    if name.chars().any(char::is_control) {
        return Err(invalid_tool_name(
            name,
            "must not contain control characters",
        ));
    }

    if name.chars().any(char::is_whitespace) {
        return Err(invalid_tool_name(name, "must not contain whitespace"));
    }

    Ok(())
}

/// Validate an AI SDK-style provider tool id.
///
/// Provider tools use `<provider>.<tool>` ids. Dots inside the tool segment are allowed so
/// provider-owned versioning schemes can stay lossless.
pub fn validate_provider_tool_id(id: &str) -> Result<(), ToolNameValidationError> {
    if id.is_empty() {
        return Err(invalid_provider_tool_id(id, "must not be empty"));
    }

    if id.trim() != id {
        return Err(invalid_provider_tool_id(
            id,
            "must not contain leading or trailing whitespace",
        ));
    }

    if id.chars().any(char::is_control) {
        return Err(invalid_provider_tool_id(
            id,
            "must not contain control characters",
        ));
    }

    if id.chars().any(char::is_whitespace) {
        return Err(invalid_provider_tool_id(id, "must not contain whitespace"));
    }

    let Some((provider, tool)) = id.split_once('.') else {
        return Err(invalid_provider_tool_id(
            id,
            "must follow `<provider>.<tool>` format",
        ));
    };

    if provider.is_empty() {
        return Err(invalid_provider_tool_id(
            id,
            "provider segment must not be empty",
        ));
    }

    if tool.is_empty() {
        return Err(invalid_provider_tool_id(
            id,
            "tool segment must not be empty",
        ));
    }

    if id.split('.').any(str::is_empty) {
        return Err(invalid_provider_tool_id(id, "segments must not be empty"));
    }

    Ok(())
}

fn invalid_tool_name(name: &str, reason: &'static str) -> ToolNameValidationError {
    ToolNameValidationError::InvalidToolName {
        name: name.to_string(),
        reason,
    }
}

fn invalid_provider_tool_id(id: &str, reason: &'static str) -> ToolNameValidationError {
    ToolNameValidationError::InvalidProviderToolId {
        id: id.to_string(),
        reason,
    }
}
