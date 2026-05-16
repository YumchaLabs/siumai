use crate::types::{HttpRequestInfo, ResponseMetadata};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::JSONValue;

/// AI SDK V4 request metadata.
pub type LanguageModelV4RequestMetadata = LanguageModelRequestMetadata;

/// AI SDK V4 response metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelV4ResponseMetadata {
    /// Response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Response start timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    /// Model id used for the response.
    #[serde(
        rename = "modelId",
        alias = "model_id",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub model_id: Option<String>,
}

impl From<&ResponseMetadata> for LanguageModelV4ResponseMetadata {
    fn from(value: &ResponseMetadata) -> Self {
        Self {
            id: value.id.clone(),
            timestamp: value.created,
            model_id: value.model.clone(),
        }
    }
}

impl From<ResponseMetadata> for LanguageModelV4ResponseMetadata {
    fn from(value: ResponseMetadata) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 non-streaming response metadata with transport details.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelV4GenerateResponseMetadata {
    /// Response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Response start timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    /// Model id used for the response.
    #[serde(
        rename = "modelId",
        alias = "model_id",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub model_id: Option<String>,
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Response HTTP body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl LanguageModelV4GenerateResponseMetadata {
    /// Attach a response body.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

impl From<&ResponseMetadata> for LanguageModelV4GenerateResponseMetadata {
    fn from(value: &ResponseMetadata) -> Self {
        Self {
            id: value.id.clone(),
            timestamp: value.created,
            model_id: value.model.clone(),
            headers: value.headers.clone(),
            body: value.body.clone(),
        }
    }
}

impl From<ResponseMetadata> for LanguageModelV4GenerateResponseMetadata {
    fn from(value: ResponseMetadata) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 streaming response metadata with transport details.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelV4StreamResponseMetadata {
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

/// AI SDK-style request metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelRequestMetadata {
    /// Serialized request body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl From<HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: HttpRequestInfo) -> Self {
        Self {
            body: value.body.map(string_body_to_json_value),
        }
    }
}

impl From<&HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: &HttpRequestInfo) -> Self {
        Self {
            body: value.body.clone().map(string_body_to_json_value),
        }
    }
}

/// AI SDK-style response metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelResponseMetadata {
    /// Response identifier.
    pub id: String,
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: ResponseMetadata) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &ResponseMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.id.clone().ok_or("missing response id")?,
            timestamp: value.created.ok_or("missing response timestamp")?,
            model_id: value.model.clone().ok_or("missing response model id")?,
            headers: value.headers.clone(),
        })
    }
}

fn string_body_to_json_value(body: String) -> JSONValue {
    serde_json::from_str(&body).unwrap_or(JSONValue::String(body))
}
