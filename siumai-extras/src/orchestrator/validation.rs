//! Shared validation helpers for the orchestrator.
//!
//! JSON Schema validation is optional and only active when the `schema`
//! feature is enabled on `siumai-extras`. Without that feature, these
//! helpers are no-ops.

use serde_json::Value;

/// Validate tool arguments against a JSON schema.
///
/// When the `schema` feature is enabled, this uses `siumai-extras::schema`
/// for strict JSON Schema validation. Otherwise it returns `Ok(())`.
pub(crate) fn validate_args_with_schema(schema: &Value, instance: &Value) -> Result<(), String> {
    #[cfg(feature = "schema")]
    {
        if let Err(e) = crate::schema::validate_json(schema, instance) {
            return Err(e.to_string());
        }
    }

    // Always mark inputs as used to avoid unused-variable warnings when the
    // `schema` feature is disabled.
    let _ = (schema, instance);

    // When schema feature is disabled, treat validation as a no-op.
    Ok(())
}
