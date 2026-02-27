use serde::{Deserialize, Serialize};

/// Media source - unified way to represent media data across providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MediaSource {
    /// URL (http, https, gs, data URLs, etc.)
    Url { url: String },
    /// Base64-encoded data
    Base64 { data: String },
    /// Binary data (will be base64-encoded when needed)
    #[serde(skip)]
    Binary { data: Vec<u8> },
}

impl MediaSource {
    /// Create from URL string
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Create from base64 string
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64 { data: data.into() }
    }

    /// Create from binary data
    pub fn binary(data: Vec<u8>) -> Self {
        Self::Binary { data }
    }

    /// Get as URL if available
    pub fn as_url(&self) -> Option<&str> {
        match self {
            Self::Url { url } => Some(url),
            _ => None,
        }
    }

    /// Get as base64 if available, or convert binary to base64
    pub fn as_base64(&self) -> Option<String> {
        match self {
            Self::Base64 { data } => Some(data.clone()),
            Self::Binary { data } => Some(base64_encode(data)),
            _ => None,
        }
    }

    /// Check if this is a URL
    pub fn is_url(&self) -> bool {
        matches!(self, Self::Url { .. })
    }

    /// Check if this is base64 data
    pub fn is_base64(&self) -> bool {
        matches!(self, Self::Base64 { .. })
    }

    /// Check if this is binary data
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }
}

/// Image detail level (for providers that support it)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

impl From<&str> for ImageDetail {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "low" => ImageDetail::Low,
            "high" => ImageDetail::High,
            _ => ImageDetail::Auto,
        }
    }
}

// Helper function for base64 encoding
fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    STANDARD.encode(data)
}
