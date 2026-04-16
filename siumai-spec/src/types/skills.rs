//! Skill upload types aligned with AI SDK `uploadSkill`

use super::{HttpConfig, ProviderMetadataMap, ProviderOptionsMap, ProviderReference, Warning};

/// Provider-id keyed metadata map used by skill upload results.
pub type SkillProviderMetadata = ProviderMetadataMap;

/// File content accepted by skill uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillFileContent {
    /// Raw file bytes.
    Bytes(Vec<u8>),
    /// Base64-encoded file bytes.
    Base64(String),
}

impl SkillFileContent {
    /// Create content from raw bytes.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self::Bytes(data)
    }

    /// Create content from base64.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }
}

impl From<Vec<u8>> for SkillFileContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for SkillFileContent {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// One uploaded skill file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillUploadFile {
    /// File path relative to the skill root.
    pub path: String,
    /// File content.
    pub content: SkillFileContent,
}

impl SkillUploadFile {
    /// Create a skill file from a path and content.
    pub fn new(path: impl Into<String>, content: impl Into<SkillFileContent>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }

    /// Create a skill file from raw bytes.
    pub fn bytes(path: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(path, SkillFileContent::Bytes(data))
    }

    /// Create a skill file from base64.
    pub fn base64(path: impl Into<String>, data: impl Into<String>) -> Self {
        Self::new(path, SkillFileContent::Base64(data.into()))
    }
}

/// Request payload for provider-owned skill uploads.
#[derive(Debug, Clone, Default)]
pub struct SkillUploadRequest {
    /// Files that make up the skill.
    pub files: Vec<SkillUploadFile>,
    /// Optional human-readable title.
    pub display_title: Option<String>,
    /// Optional provider-specific options (`providerOptions`).
    pub provider_options: ProviderOptionsMap,
    /// Optional per-request HTTP overrides.
    pub http_config: Option<HttpConfig>,
}

impl SkillUploadRequest {
    /// Create a request with the provided skill files.
    pub fn new(files: Vec<SkillUploadFile>) -> Self {
        Self {
            files,
            ..Default::default()
        }
    }

    /// Set the display title.
    pub fn with_display_title(mut self, display_title: impl Into<String>) -> Self {
        self.display_title = Some(display_title.into());
        self
    }

    /// Replace the provider options map.
    pub fn with_provider_options(mut self, provider_options: ProviderOptionsMap) -> Self {
        self.provider_options = provider_options;
        self
    }

    /// Insert one provider option entry.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_options.insert(provider_id, value);
        self
    }

    /// Set per-request HTTP configuration.
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }

    /// Add one per-request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut http_config = self.http_config.take().unwrap_or_else(HttpConfig::empty);
        http_config.headers.insert(key.into(), value.into());
        self.http_config = Some(http_config);
        self
    }
}

/// Result returned by provider-owned skill uploads.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillUploadResult {
    /// Provider-owned skill reference in stable AI SDK-style shape.
    pub provider_reference: ProviderReference,
    /// Optional human-readable title.
    pub display_title: Option<String>,
    /// Optional canonical skill name.
    pub name: Option<String>,
    /// Optional skill description.
    pub description: Option<String>,
    /// Optional latest version id.
    pub latest_version: Option<String>,
    /// Provider-owned metadata under the provider id root.
    pub provider_metadata: Option<SkillProviderMetadata>,
    /// Non-fatal warnings emitted while uploading.
    pub warnings: Vec<Warning>,
}
