//! Conversions to LlmError (reqwest/json)

use super::types::LlmError;

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        Self::HttpError(err.to_string())
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err.to_string())
    }
}
