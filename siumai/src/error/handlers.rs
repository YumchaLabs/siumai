//! API Error Handling Utilities
//!
//! Common utilities for handling API response errors across all providers.

use super::types::LlmError;
use reqwest::Response;
use serde_json::Value;

/// Generic API error handler
pub struct ApiErrorHandler;

impl ApiErrorHandler {
    /// Handle a failed HTTP response and convert it to an appropriate LlmError
    pub async fn handle_response_error(response: Response, provider_name: &str) -> LlmError {
        let status_code = response.status().as_u16();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        // Try to parse as JSON for structured error information
        let error_details = serde_json::from_str::<Value>(&error_text).ok();

        // Map common HTTP status codes to appropriate error types
        match status_code {
            400 => LlmError::InvalidInput(format!("{provider_name} API error: {error_text}")),
            401 => LlmError::AuthenticationError(format!(
                "Authentication failed for {provider_name}: {error_text}"
            )),
            403 => LlmError::AuthenticationError(format!(
                "Access forbidden for {provider_name}: {error_text}"
            )),
            404 => LlmError::NotFound(format!("{provider_name} resource not found: {error_text}")),
            413 => LlmError::InvalidInput("Request payload too large".to_string()),
            415 => LlmError::InvalidInput("Unsupported media type".to_string()),
            429 => LlmError::RateLimitError(format!(
                "Rate limit exceeded for {provider_name}: {error_text}"
            )),
            500..=599 => LlmError::ApiError {
                code: status_code,
                message: format!("{provider_name} server error: {error_text}"),
                details: error_details,
            },
            _ => LlmError::ApiError {
                code: status_code,
                message: format!("{provider_name} API error {status_code}: {error_text}"),
                details: error_details,
            },
        }
    }

    /// Handle OpenAI-specific error responses
    pub async fn handle_openai_error(response: Response) -> LlmError {
        let status_code = response.status().as_u16();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        // OpenAI-specific error handling
        match status_code {
            404 => LlmError::NotFound(format!("OpenAI resource not found: {error_text}")),
            413 => LlmError::InvalidInput("File too large".to_string()),
            415 => LlmError::InvalidInput("Unsupported file type".to_string()),
            _ => {
                // Try to parse as JSON for structured error information
                let error_details = serde_json::from_str::<Value>(&error_text).ok();

                match status_code {
                    400 => LlmError::InvalidInput(format!("OpenAI API error: {error_text}")),
                    401 => LlmError::AuthenticationError(format!(
                        "Authentication failed for OpenAI: {error_text}"
                    )),
                    403 => LlmError::AuthenticationError(format!(
                        "Access forbidden for OpenAI: {error_text}"
                    )),
                    429 => LlmError::RateLimitError(format!(
                        "Rate limit exceeded for OpenAI: {error_text}"
                    )),
                    500..=599 => LlmError::ApiError {
                        code: status_code,
                        message: format!("OpenAI server error: {error_text}"),
                        details: error_details,
                    },
                    _ => LlmError::ApiError {
                        code: status_code,
                        message: format!("OpenAI API error {status_code}: {error_text}"),
                        details: error_details,
                    },
                }
            }
        }
    }

    /// Handle Anthropic-specific error responses with structured error parsing
    pub async fn handle_anthropic_error(response: Response) -> LlmError {
        let status_code = response.status().as_u16();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        // Try to parse Anthropic's structured error format
        if let Ok(error_json) = serde_json::from_str::<Value>(&error_text)
            && let Some(error_obj) = error_json.get("error")
        {
            let error_type = error_obj
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");
            let error_message = error_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");

            return Self::map_anthropic_error_type(
                status_code,
                error_type,
                error_message,
                error_json.clone(),
            );
        }

        // Fallback to generic error handling
        let error_details = serde_json::from_str::<Value>(&error_text).ok();
        LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error: {error_text}"),
            details: error_details,
        }
    }

    /// Map Anthropic error types to appropriate LlmError variants
    fn map_anthropic_error_type(
        status_code: u16,
        error_type: &str,
        error_message: &str,
        error_details: Value,
    ) -> LlmError {
        match error_type {
            "authentication_error" => LlmError::AuthenticationError(error_message.to_string()),
            "permission_error" => {
                LlmError::AuthenticationError(format!("Permission denied: {error_message}"))
            }
            "not_found_error" => LlmError::NotFound(error_message.to_string()),
            "rate_limit_error" => LlmError::RateLimitError(error_message.to_string()),
            "api_error" => LlmError::ApiError {
                code: status_code,
                message: format!("Anthropic API error: {error_message}"),
                details: Some(error_details),
            },
            "overloaded_error" => LlmError::ApiError {
                code: 503,
                message: format!("Anthropic service overloaded: {error_message}"),
                details: Some(error_details),
            },
            "invalid_request_error" => LlmError::InvalidInput(error_message.to_string()),
            _ => LlmError::ApiError {
                code: status_code,
                message: format!("Anthropic error ({error_type}): {error_message}"),
                details: Some(error_details),
            },
        }
    }

    /// Handle Gemini-specific error responses
    pub async fn handle_gemini_error(response: Response) -> LlmError {
        Self::handle_response_error(response, "Gemini").await
    }

    /// Handle Groq-specific error responses
    pub async fn handle_groq_error(response: Response) -> LlmError {
        Self::handle_response_error(response, "Groq").await
    }

    /// Handle xAI-specific error responses
    pub async fn handle_xai_error(response: Response) -> LlmError {
        Self::handle_response_error(response, "xAI").await
    }

    /// Handle Ollama-specific error responses
    pub async fn handle_ollama_error(response: Response) -> LlmError {
        Self::handle_response_error(response, "Ollama").await
    }
}

/// Provider-specific error handlers
pub struct ProviderErrorHandlers;

impl ProviderErrorHandlers {
    /// Handle error for OpenAI
    pub async fn openai(response: Response) -> LlmError {
        ApiErrorHandler::handle_openai_error(response).await
    }

    /// Handle error for Anthropic
    pub async fn anthropic(response: Response) -> LlmError {
        ApiErrorHandler::handle_anthropic_error(response).await
    }

    /// Handle error for Gemini
    pub async fn gemini(response: Response) -> LlmError {
        ApiErrorHandler::handle_gemini_error(response).await
    }

    /// Handle error for Groq
    pub async fn groq(response: Response) -> LlmError {
        ApiErrorHandler::handle_groq_error(response).await
    }

    /// Handle error for xAI
    pub async fn xai(response: Response) -> LlmError {
        ApiErrorHandler::handle_xai_error(response).await
    }

    /// Handle error for Ollama
    pub async fn ollama(response: Response) -> LlmError {
        ApiErrorHandler::handle_ollama_error(response).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_error_mapping() {
        let error_details = serde_json::json!({
            "type": "authentication_error",
            "message": "Invalid API key"
        });

        let error = ApiErrorHandler::map_anthropic_error_type(
            401,
            "authentication_error",
            "Invalid API key",
            error_details,
        );

        match error {
            LlmError::AuthenticationError(msg) => assert_eq!(msg, "Invalid API key"),
            _ => panic!("Expected AuthenticationError"),
        }
    }

    #[test]
    fn test_rate_limit_error_mapping() {
        let error_details = serde_json::json!({
            "type": "rate_limit_error",
            "message": "Rate limit exceeded"
        });

        let error = ApiErrorHandler::map_anthropic_error_type(
            429,
            "rate_limit_error",
            "Rate limit exceeded",
            error_details,
        );

        match error {
            LlmError::RateLimitError(msg) => assert_eq!(msg, "Rate limit exceeded"),
            _ => panic!("Expected RateLimitError"),
        }
    }
}
