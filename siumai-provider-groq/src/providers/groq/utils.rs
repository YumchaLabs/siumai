//! `Groq` Utility Functions
//!
//! Utility functions for the Groq provider.

use crate::error::LlmError;
use chrono::TimeZone;

fn get_any<'a>(params: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|key| params.get(*key))
}

/// Extract Groq response metadata fields that AI SDK exposes on the stable response metadata lane.
pub fn extract_groq_response_metadata(
    raw: &serde_json::Value,
) -> Option<crate::types::ProviderMetadataMap> {
    let mut meta = std::collections::HashMap::new();

    if let Some(id) = raw
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        meta.insert("id".to_string(), serde_json::Value::String(id.to_string()));
    }

    if let Some(model_id) = raw
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        meta.insert(
            "modelId".to_string(),
            serde_json::Value::String(model_id.to_string()),
        );
    }

    let created = raw
        .get("created")
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)))
        .filter(|value| *value > 0)
        .and_then(|value| chrono::Utc.timestamp_opt(value, 0).single());
    if let Some(timestamp) = created {
        meta.insert(
            "timestamp".to_string(),
            serde_json::Value::String(timestamp.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        );
    }

    if meta.is_empty() {
        None
    } else {
        Some(crate::types::provider_metadata::provider_metadata_from_object("groq", meta))
    }
}

/// Validate Groq-specific parameters
pub fn validate_groq_params(params: &serde_json::Value) -> Result<(), LlmError> {
    // Validate frequency_penalty
    if let Some(freq_penalty) = params.get("frequency_penalty")
        && let Some(value) = freq_penalty.as_f64()
        && !(-2.0..=2.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "frequency_penalty must be between -2.0 and 2.0".to_string(),
        ));
    }

    // Validate presence_penalty
    if let Some(pres_penalty) = params.get("presence_penalty")
        && let Some(value) = pres_penalty.as_f64()
        && !(-2.0..=2.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "presence_penalty must be between -2.0 and 2.0".to_string(),
        ));
    }

    // Validate temperature (relaxed validation - only check for negative values)
    if let Some(temperature) = params.get("temperature")
        && let Some(value) = temperature.as_f64()
        && value < 0.0
    {
        return Err(LlmError::InvalidParameter(
            "temperature cannot be negative".to_string(),
        ));
    }

    // Validate top_p
    if let Some(top_p) = params.get("top_p")
        && let Some(value) = top_p.as_f64()
        && !(0.0..=1.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "top_p must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Validate n (number of choices)
    if let Some(n) = params.get("n")
        && let Some(value) = n.as_u64()
        && value != 1
    {
        return Err(LlmError::InvalidParameter(
            "Groq only supports n=1".to_string(),
        ));
    }

    // Validate service_tier
    if let Some(service_tier) = get_any(params, &["service_tier", "serviceTier"])
        && let Some(value) = service_tier.as_str()
        && !["auto", "on_demand", "performance", "flex"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "service_tier must be one of: auto, on_demand, performance, flex".to_string(),
        ));
    }

    // Validate reasoning_effort (for qwen3 models)
    if let Some(reasoning_effort) = get_any(params, &["reasoning_effort", "reasoningEffort"])
        && let Some(value) = reasoning_effort.as_str()
        && !["none", "default", "low", "medium", "high"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "reasoning_effort must be one of: none, default, low, medium, high".to_string(),
        ));
    }

    // Validate reasoning_format
    if let Some(reasoning_format) = get_any(params, &["reasoning_format", "reasoningFormat"])
        && let Some(value) = reasoning_format.as_str()
        && !["hidden", "raw", "parsed"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "reasoning_format must be one of: hidden, raw, parsed".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_response_metadata_extractor_uses_ai_sdk_field_names() {
        let raw = serde_json::json!({
            "id": "chatcmpl-groq-test",
            "created": 1_741_392_000,
            "model": "llama-3.3-70b-versatile"
        });

        let metadata = extract_groq_response_metadata(&raw).expect("metadata");
        let groq = metadata.get("groq").expect("groq namespace");

        assert_eq!(
            groq.get("id"),
            Some(&serde_json::json!("chatcmpl-groq-test"))
        );
        assert_eq!(
            groq.get("modelId"),
            Some(&serde_json::json!("llama-3.3-70b-versatile"))
        );
        assert_eq!(
            groq.get("timestamp").and_then(|value| value.as_str()),
            Some("2025-03-08T00:00:00Z")
        );
    }

    #[test]
    fn test_validate_groq_params() {
        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "serviceTier": "performance",
            "reasoningEffort": "high",
            "reasoningFormat": "parsed"
        });
        assert!(validate_groq_params(&valid_params).is_ok());

        // High temperature (now allowed with relaxed validation)
        let high_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(validate_groq_params(&high_temp).is_ok());

        // Negative temperature (still invalid)
        let invalid_temp = serde_json::json!({
            "temperature": -1.0
        });
        assert!(validate_groq_params(&invalid_temp).is_err());

        // Invalid service_tier
        let invalid_tier = serde_json::json!({
            "service_tier": "invalid"
        });
        assert!(validate_groq_params(&invalid_tier).is_err());

        // Invalid reasoning_effort
        let invalid_reasoning_effort = serde_json::json!({
            "reasoning_effort": "turbo"
        });
        assert!(validate_groq_params(&invalid_reasoning_effort).is_err());
    }
}
