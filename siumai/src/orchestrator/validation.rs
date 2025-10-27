//! Shared validation helpers for the orchestrator.
//!
//! Note: JSON Schema validation has been moved to `siumai-extras`.
//! These helpers are stubs that log guidance for users who need schema checks.

use serde_json::Value;

/// Validate tool arguments against a JSON schema.
///
/// Currently a no-op in core. If you need strict schema validation,
/// use `siumai-extras::schema::validate_json` in your application.
pub(crate) fn validate_args_with_schema(_schema: &Value, _instance: &Value) -> Result<(), String> {
    tracing::debug!(
        "Schema validation is no longer built-in. Use siumai-extras::schema::validate_json for validation."
    );
    Ok(())
}
