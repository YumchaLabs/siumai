use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

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

/// Provider reference map aligned with AI SDK v4 file references.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ProviderReference(pub BTreeMap<String, String>);

impl ProviderReference {
    /// Create an empty provider reference map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a provider reference map with a single provider entry.
    pub fn single(provider_id: impl Into<String>, reference: impl Into<String>) -> Self {
        let mut map = BTreeMap::new();
        map.insert(provider_id.into(), reference.into());
        Self(map)
    }

    /// Insert or replace a provider reference.
    pub fn insert(
        &mut self,
        provider_id: impl Into<String>,
        reference: impl Into<String>,
    ) -> Option<String> {
        self.0.insert(provider_id.into(), reference.into())
    }

    /// Resolve the provider-specific reference.
    pub fn get(&self, provider_id: &str) -> Option<&str> {
        self.0.get(provider_id).map(String::as_str)
    }

    /// Check whether the provider reference map is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return the available provider ids in deterministic order.
    pub fn available_providers(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }

    /// Resolve the first matching provider reference from a preference list.
    pub fn preferred_value<'a>(&'a self, provider_ids: &[&str]) -> Option<&'a str> {
        provider_ids
            .iter()
            .find_map(|provider_id| self.get(provider_id))
    }
}

impl From<BTreeMap<String, String>> for ProviderReference {
    fn from(value: BTreeMap<String, String>) -> Self {
        Self(value)
    }
}

impl From<HashMap<String, String>> for ProviderReference {
    fn from(value: HashMap<String, String>) -> Self {
        Self(value.into_iter().collect())
    }
}

impl<const N: usize> From<[(&str, &str); N]> for ProviderReference {
    fn from(value: [(&str, &str); N]) -> Self {
        Self(
            value
                .into_iter()
                .map(|(provider, reference)| (provider.to_string(), reference.to_string()))
                .collect(),
        )
    }
}

/// Source union for image/file prompt parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FilePartSource {
    /// URL/base64/binary-backed media source.
    Media(MediaSource),
    /// Provider-managed file reference.
    ProviderReference {
        /// Provider-specific references keyed by provider id.
        #[serde(rename = "providerReference", alias = "provider_reference")]
        provider_reference: ProviderReference,
    },
}

impl FilePartSource {
    /// Create from URL string.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Media(MediaSource::url(url))
    }

    /// Create from base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Media(MediaSource::base64(data))
    }

    /// Create from binary data.
    pub fn binary(data: Vec<u8>) -> Self {
        Self::Media(MediaSource::binary(data))
    }

    /// Create from provider references.
    pub fn provider_reference(provider_reference: impl Into<ProviderReference>) -> Self {
        Self::ProviderReference {
            provider_reference: provider_reference.into(),
        }
    }

    /// Create from a single provider reference entry.
    pub fn single_provider_reference(
        provider_id: impl Into<String>,
        reference: impl Into<String>,
    ) -> Self {
        Self::provider_reference(ProviderReference::single(provider_id, reference))
    }

    /// Get the media source when this part is media-backed.
    pub fn as_media_source(&self) -> Option<&MediaSource> {
        match self {
            Self::Media(source) => Some(source),
            Self::ProviderReference { .. } => None,
        }
    }

    /// Get the provider reference map when this part is provider-backed.
    pub fn as_provider_reference(&self) -> Option<&ProviderReference> {
        match self {
            Self::ProviderReference { provider_reference } => Some(provider_reference),
            Self::Media(_) => None,
        }
    }

    /// Get as URL if available.
    pub fn as_url(&self) -> Option<&str> {
        self.as_media_source().and_then(MediaSource::as_url)
    }

    /// Get as base64 if available.
    pub fn as_base64(&self) -> Option<String> {
        self.as_media_source().and_then(MediaSource::as_base64)
    }

    /// Check if this is a URL source.
    pub fn is_url(&self) -> bool {
        self.as_media_source().is_some_and(MediaSource::is_url)
    }

    /// Check if this is a base64 source.
    pub fn is_base64(&self) -> bool {
        self.as_media_source().is_some_and(MediaSource::is_base64)
    }

    /// Check if this is a binary source.
    pub fn is_binary(&self) -> bool {
        self.as_media_source().is_some_and(MediaSource::is_binary)
    }

    /// Check if this is a provider reference source.
    pub fn is_provider_reference(&self) -> bool {
        matches!(self, Self::ProviderReference { .. })
    }

    /// Approximate content length for memory estimation.
    pub fn content_length(&self) -> usize {
        match self {
            Self::Media(MediaSource::Url { url }) => url.len(),
            Self::Media(MediaSource::Base64 { data }) => data.len(),
            Self::Media(MediaSource::Binary { data }) => data.len(),
            Self::ProviderReference { provider_reference } => provider_reference
                .0
                .iter()
                .map(|(provider, reference)| provider.len() + reference.len())
                .sum(),
        }
    }
}

impl From<MediaSource> for FilePartSource {
    fn from(value: MediaSource) -> Self {
        Self::Media(value)
    }
}

impl From<ProviderReference> for FilePartSource {
    fn from(value: ProviderReference) -> Self {
        Self::provider_reference(value)
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

#[cfg(test)]
mod tests {
    use super::{FilePartSource, ProviderReference};

    #[test]
    fn file_part_source_serializes_provider_reference_shape() {
        let source = FilePartSource::provider_reference(ProviderReference::from([
            ("openai", "file-123"),
            ("anthropic", "file-456"),
        ]));

        let value = serde_json::to_value(&source).expect("serialize provider reference");
        assert_eq!(
            value,
            serde_json::json!({
                "providerReference": {
                    "anthropic": "file-456",
                    "openai": "file-123"
                }
            })
        );
    }

    #[test]
    fn file_part_source_accepts_provider_reference_alias() {
        let value = serde_json::json!({
            "provider_reference": {
                "openai": "file-123"
            }
        });

        let source = serde_json::from_value::<FilePartSource>(value)
            .expect("deserialize provider reference");
        let provider_reference = source
            .as_provider_reference()
            .expect("provider reference source");
        assert_eq!(provider_reference.get("openai"), Some("file-123"));
    }
}
