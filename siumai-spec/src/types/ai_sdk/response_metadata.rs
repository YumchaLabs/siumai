use crate::types::HttpResponseInfo;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{JSONValue, VideoModelProviderMetadata};

/// AI SDK-style response metadata for image-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
        })
    }
}

/// AI SDK-style response metadata for video-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VideoModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Provider-specific metadata for this call when available.
    #[serde(rename = "providerMetadata", skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<VideoModelProviderMetadata>,
}

impl VideoModelResponseMetadata {
    /// Attach provider metadata when it is not empty.
    pub fn with_provider_metadata(mut self, provider_metadata: VideoModelProviderMetadata) -> Self {
        if !provider_metadata.is_empty() {
            self.provider_metadata = Some(provider_metadata);
        }
        self
    }
}

impl TryFrom<HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            provider_metadata: None,
        })
    }
}

/// AI SDK-style response metadata for speech-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl TryFrom<HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            body: value.body.clone(),
        })
    }
}

/// AI SDK-style response metadata for transcription-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl TryFrom<HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            body: value.body.clone(),
        })
    }
}
