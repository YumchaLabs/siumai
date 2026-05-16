//! AI SDK-style provider options utility helpers.

use crate::error::LlmError;
use crate::types::{ProviderOptionsMap, Schema, ValidationResult};

/// Parse provider-specific options using a runtime schema validator.
///
/// This mirrors AI SDK `parseProviderOptions`: when the requested provider key is missing or
/// `null`, `Ok(None)` is returned; otherwise the provider value is validated against `schema`.
pub fn parse_provider_options<T>(
    provider: &str,
    provider_options: Option<&ProviderOptionsMap>,
    schema: &Schema<T>,
) -> Result<Option<T>, LlmError> {
    let Some(value) = provider_options.and_then(|options| options.get(provider)) else {
        return Ok(None);
    };

    if value.is_null() {
        return Ok(None);
    }

    match schema.validate(value) {
        Some(ValidationResult::Success { value }) => Ok(Some(value)),
        Some(ValidationResult::Failure { error }) => Err(LlmError::InvalidParameter(format!(
            "invalid {provider} provider options: {error}"
        ))),
        None => Err(LlmError::InvalidInput(format!(
            "Schema for {provider} provider options does not define a runtime validator"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::json_schema_with_validator;

    #[test]
    fn parse_provider_options_returns_none_for_missing_or_null_provider() {
        let schema = json_schema_with_validator(serde_json::json!({}), |_| {
            ValidationResult::success("unused".to_string())
        });
        let mut options = ProviderOptionsMap::new();
        options.insert("provider-a", serde_json::Value::Null);

        assert!(
            parse_provider_options("provider-b", Some(&options), &schema)
                .expect("missing provider")
                .is_none()
        );
        assert!(
            parse_provider_options("provider-a", Some(&options), &schema)
                .expect("null provider")
                .is_none()
        );
    }

    #[test]
    fn parse_provider_options_validates_provider_value() {
        let schema =
            json_schema_with_validator(
                serde_json::json!({ "type": "object" }),
                |value| match value.get("mode").and_then(serde_json::Value::as_str) {
                    Some(mode) => ValidationResult::success(mode.to_string()),
                    None => ValidationResult::failure(LlmError::InvalidInput(
                        "missing mode".to_string(),
                    )),
                },
            );
        let mut options = ProviderOptionsMap::new();
        options.insert("Provider-A", serde_json::json!({ "mode": "strict" }));

        assert_eq!(
            parse_provider_options("provider-a", Some(&options), &schema)
                .expect("valid provider options"),
            Some("strict".to_string())
        );

        options.insert("provider-a", serde_json::json!({ "mode": 42 }));
        assert!(matches!(
            parse_provider_options("provider-a", Some(&options), &schema),
            Err(LlmError::InvalidParameter(_))
        ));
    }
}
