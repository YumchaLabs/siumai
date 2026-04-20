use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reference image entry for Google Vertex video generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleVertexReferenceImage {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "bytesBase64Encoded",
        alias = "bytes_base64_encoded"
    )]
    pub bytes_base64_encoded: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "gcsUri",
        alias = "gcs_uri"
    )]
    pub gcs_uri: Option<String>,
}

impl GoogleVertexReferenceImage {
    /// Create an empty reference image entry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set inline base64-encoded image bytes.
    pub fn with_bytes_base64_encoded(mut self, value: impl Into<String>) -> Self {
        self.bytes_base64_encoded = Some(value.into());
        self
    }

    /// Set a GCS URI reference.
    pub fn with_gcs_uri(mut self, value: impl Into<String>) -> Self {
        self.gcs_uri = Some(value.into());
        self
    }
}

/// Provider-owned video-model options for Google Vertex video requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoogleVertexVideoModelOptions {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollIntervalMs",
        alias = "poll_interval_ms"
    )]
    pub poll_interval_ms: Option<u64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "pollTimeoutMs",
        alias = "poll_timeout_ms"
    )]
    pub poll_timeout_ms: Option<u64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "personGeneration",
        alias = "person_generation"
    )]
    pub person_generation: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "negativePrompt",
        alias = "negative_prompt"
    )]
    pub negative_prompt: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "generateAudio",
        alias = "generate_audio"
    )]
    pub generate_audio: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "gcsOutputDirectory",
        alias = "gcs_output_directory"
    )]
    pub gcs_output_directory: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "referenceImages",
        alias = "reference_images"
    )]
    pub reference_images: Option<Vec<GoogleVertexReferenceImage>>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl GoogleVertexVideoModelOptions {
    /// Create empty Google Vertex video options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set polling interval in milliseconds.
    pub const fn with_poll_interval_ms(mut self, value: u64) -> Self {
        self.poll_interval_ms = Some(value);
        self
    }

    /// Set polling timeout in milliseconds.
    pub const fn with_poll_timeout_ms(mut self, value: u64) -> Self {
        self.poll_timeout_ms = Some(value);
        self
    }

    /// Set person-generation policy.
    pub fn with_person_generation(mut self, value: impl Into<String>) -> Self {
        self.person_generation = Some(value.into());
        self
    }

    /// Set negative prompt.
    pub fn with_negative_prompt(mut self, value: impl Into<String>) -> Self {
        self.negative_prompt = Some(value.into());
        self
    }

    /// Toggle audio generation.
    pub const fn with_generate_audio(mut self, value: bool) -> Self {
        self.generate_audio = Some(value);
        self
    }

    /// Set GCS output directory.
    pub fn with_gcs_output_directory(mut self, value: impl Into<String>) -> Self {
        self.gcs_output_directory = Some(value.into());
        self
    }

    /// Set reference images.
    pub fn with_reference_images(mut self, value: Vec<GoogleVertexReferenceImage>) -> Self {
        self.reference_images = Some(value);
        self
    }

    /// Add one extra provider-owned field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// Deprecated AI SDK compatibility alias for Google Vertex video-model options.
#[deprecated(note = "Use `GoogleVertexVideoModelOptions` instead.")]
pub type GoogleVertexVideoProviderOptions = GoogleVertexVideoModelOptions;

/// AI SDK-style alias for Google Vertex video model ids.
pub type GoogleVertexVideoModelId = String;
