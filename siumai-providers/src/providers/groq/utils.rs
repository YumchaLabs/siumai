//! `Groq` Utility Functions
//!
//! Utility functions for the Groq provider.

use reqwest::header::HeaderMap;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::types::{ChatMessage, FinishReason};

/// Build HTTP headers for Groq API requests
pub fn build_headers(
    api_key: &str,
    custom_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::groq(api_key, custom_headers)
}

/// Convert internal messages to Groq API format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<serde_json::Value>, LlmError> {
    // Start from the canonical OpenAI(-compatible) wire format conversion and then apply
    // Groq-specific constraints.
    let openai_messages = crate::standards::openai::utils::convert_messages(messages)?;

    let mut out = Vec::with_capacity(openai_messages.len());
    for msg in openai_messages {
        let mut v = serde_json::to_value(msg).map_err(|e| {
            LlmError::ParseError(format!("Serialize OpenAI message to JSON failed: {e}"))
        })?;

        // Groq does not accept "developer"; treat it as "system".
        if v.get("role").and_then(|r| r.as_str()) == Some("developer") {
            v["role"] = serde_json::json!("system");
        }

        // Groq does not support audio/file message parts; normalize to text placeholders.
        if let Some(content) = v.get_mut("content") && let Some(parts) = content.as_array_mut() {
            for p in parts {
                let ty = p.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match ty {
                    "input_audio" => {
                        *p = serde_json::json!({"type":"text","text":"[Audio not supported]"})
                    }
                    "file" => *p = serde_json::json!({"type":"text","text":"[File not supported]"}),
                    _ => {}
                }
            }
        }

        out.push(v);
    }

    Ok(out)
}

/// Parse finish reason from Groq API response
pub fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    crate::standards::openai::utils::parse_finish_reason(reason)
        .unwrap_or(FinishReason::Other("unknown".to_string()))
}

/// Extract error message from Groq API error response
pub fn extract_error_message(error_text: &str) -> String {
    // Try to parse as JSON error response
    if let Ok(error_response) = serde_json::from_str::<super::types::GroqErrorResponse>(error_text)
    {
        return error_response.error.message;
    }

    // Fallback to raw error text
    error_text.to_string()
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
    use crate::types::ChatMessage;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, USER_AGENT};

    #[test]
    fn test_build_headers() {
        let custom_headers = HashMap::new();
        let headers = build_headers("test-api-key", &custom_headers).unwrap();

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-api-key");
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert!(headers.get(USER_AGENT).is_some());
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant").build(),
            ChatMessage::user("Hello, world!").build(),
        ];

        let groq_messages = convert_messages(&messages).unwrap();
        assert_eq!(groq_messages.len(), 2);
        assert_eq!(groq_messages[0]["role"], "system");
        assert_eq!(groq_messages[1]["role"], "user");
    }

    #[test]
    fn test_parse_finish_reason() {
        assert_eq!(parse_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(parse_finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            parse_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_finish_reason(Some("unknown")),
            FinishReason::Other("unknown".to_string())
        );
        assert_eq!(
            parse_finish_reason(None),
            FinishReason::Other("unknown".to_string())
        );
    }

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
