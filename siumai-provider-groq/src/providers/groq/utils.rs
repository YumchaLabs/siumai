//! `Groq` Utility Functions
//!
//! Utility functions for the Groq provider.

use crate::error::LlmError;

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
    if let Some(service_tier) = params.get("service_tier")
        && let Some(value) = service_tier.as_str()
        && !["auto", "on_demand", "flex"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "service_tier must be one of: auto, on_demand, flex".to_string(),
        ));
    }

    // Validate reasoning_effort (for qwen3 models)
    if let Some(reasoning_effort) = params.get("reasoning_effort")
        && let Some(value) = reasoning_effort.as_str()
        && !["none", "default"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "reasoning_effort must be one of: none, default".to_string(),
        ));
    }

    // Validate reasoning_format
    if let Some(reasoning_format) = params.get("reasoning_format")
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
    fn test_validate_groq_params() {
        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "service_tier": "auto"
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
    }
}
