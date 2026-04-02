//! Image generation and processing types

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{HttpConfig, HttpResponseInfo, ProviderOptionsMap, Warning};

/// Image generation request
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the image
    pub prompt: String,
    /// Negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Image size (e.g., "1024x1024")
    pub size: Option<String>,
    /// Number of images to generate
    pub count: u32,
    /// Model to use for generation
    pub model: Option<String>,
    /// Quality setting
    pub quality: Option<String>,
    /// Style setting
    pub style: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of inference steps
    pub steps: Option<u32>,
    /// Guidance scale
    pub guidance_scale: Option<f32>,
    /// Whether to enhance the prompt
    pub enhance_prompt: Option<bool>,
    /// Response format (url or `b64_json`)
    pub response_format: Option<String>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

impl ImageGenerationRequest {
    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set per-request HTTP config (headers, proxy, timeout, etc.).
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }
}

/// Image edit request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ImageEditFileData {
    /// Base64-encoded file data.
    Base64(String),
    /// Binary file data.
    Binary(Vec<u8>),
}

impl ImageEditFileData {
    /// Create file data from binary bytes.
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Self::Binary(data.into())
    }

    /// Create file data from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    /// Convert the file data to a base64 string.
    pub fn as_base64(&self) -> String {
        match self {
            Self::Base64(data) => data.clone(),
            Self::Binary(data) => base64::engine::general_purpose::STANDARD.encode(data),
        }
    }

    /// Convert the file data to bytes, decoding base64 when necessary.
    pub fn as_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        match self {
            Self::Base64(data) => base64::engine::general_purpose::STANDARD.decode(data),
            Self::Binary(data) => Ok(data.clone()),
        }
    }
}

/// Image edit input aligned with AI SDK image model files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageEditInput {
    /// File-like image input backed by binary or base64 data.
    File {
        /// File payload.
        data: ImageEditFileData,
        /// Optional media type such as `image/png`.
        #[serde(
            rename = "mediaType",
            alias = "media_type",
            skip_serializing_if = "Option::is_none"
        )]
        media_type: Option<String>,
    },
    /// URL-like image input.
    Url {
        /// URL or data URL.
        url: String,
    },
}

impl ImageEditInput {
    /// Create a file input from binary bytes.
    pub fn file(data: impl Into<Vec<u8>>) -> Self {
        Self::File {
            data: ImageEditFileData::binary(data),
            media_type: None,
        }
    }

    /// Create a file input from binary bytes and media type.
    pub fn file_with_media_type(data: impl Into<Vec<u8>>, media_type: impl Into<String>) -> Self {
        Self::file(data).with_media_type(media_type)
    }

    /// Create a file input from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::File {
            data: ImageEditFileData::base64(data),
            media_type: None,
        }
    }

    /// Create a file input from a base64 string and media type.
    pub fn base64_with_media_type(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::base64(data).with_media_type(media_type)
    }

    /// Create a URL input.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Set the media type for file inputs.
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        if let Self::File {
            media_type: current,
            ..
        } = &mut self
        {
            *current = Some(media_type.into());
        }
        self
    }

    /// Return the media type if available.
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::File { media_type, .. } => media_type.as_deref(),
            Self::Url { .. } => None,
        }
    }

    /// Return the URL when this input is URL-backed.
    pub fn as_url(&self) -> Option<&str> {
        match self {
            Self::Url { url } => Some(url.as_str()),
            Self::File { .. } => None,
        }
    }

    /// Return the file payload when this input is file-backed.
    pub fn file_data(&self) -> Option<&ImageEditFileData> {
        match self {
            Self::File { data, .. } => Some(data),
            Self::Url { .. } => None,
        }
    }

    /// Check whether the input is file-backed.
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Check whether the input is URL-backed.
    pub fn is_url(&self) -> bool {
        matches!(self, Self::Url { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEditRequest {
    /// Input images for image editing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageEditInput>,
    /// Mask image input (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<ImageEditInput>,
    /// Text prompt for editing
    pub prompt: String,
    /// Model to use for editing (provider-specific; optional).
    ///
    /// This is required by some providers (e.g., Vertex AI Imagen) where the model
    /// is part of the request URL.
    pub model: Option<String>,
    /// Number of images to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

impl ImageEditRequest {
    /// Replace the input images with a single image.
    pub fn with_image(mut self, image: ImageEditInput) -> Self {
        self.images = vec![image];
        self
    }

    /// Replace the input images with a new image list.
    pub fn with_images<I>(mut self, images: I) -> Self
    where
        I: IntoIterator<Item = ImageEditInput>,
    {
        self.images = images.into_iter().collect();
        self
    }

    /// Append an image input.
    pub fn push_image(mut self, image: ImageEditInput) -> Self {
        self.images.push(image);
        self
    }

    /// Set the mask input.
    pub fn with_mask(mut self, mask: ImageEditInput) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set per-request HTTP config (headers, proxy, timeout, etc.).
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }
}

/// Image variation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageVariationRequest {
    /// Original image data
    pub image: Vec<u8>,
    /// Model to use for variations (provider-specific; optional).
    ///
    /// This is required by some providers where the model is part of the request URL.
    pub model: Option<String>,
    /// Number of variations to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

impl ImageVariationRequest {
    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set per-request HTTP config (headers, proxy, timeout, etc.).
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }
}

/// Image generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    /// Generated images
    pub images: Vec<GeneratedImage>,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Warnings from the provider (e.g., unsupported settings).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Warning>>,
    /// HTTP response envelope (timestamp, model id, headers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<HttpResponseInfo>,
}

/// A single generated image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Image URL (if `response_format` is "url")
    pub url: Option<String>,
    /// Base64 encoded image data (if `response_format` is "`b64_json`")
    pub b64_json: Option<String>,
    /// Image format
    pub format: Option<String>,
    /// Image dimensions
    pub width: Option<u32>,
    /// Image height
    pub height: Option<u32>,
    /// Revised prompt (if prompt was enhanced)
    pub revised_prompt: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

// Keep for backward compatibility
#[deprecated(
    since = "0.11.0-beta.5",
    note = "Deprecated with VisionCapability; use ImageGenerationRequest/ImageGenerationResponse and multimodal Chat messages instead."
)]
pub type ImageGenRequest = ();
#[deprecated(
    since = "0.11.0-beta.5",
    note = "Deprecated with VisionCapability; use ImageGenerationRequest/ImageGenerationResponse and multimodal Chat messages instead."
)]
pub type ImageResponse = ();
#[deprecated(
    since = "0.11.0-beta.5",
    note = "Deprecated with VisionCapability; use multimodal Chat messages instead."
)]
pub type VisionRequest = ();
#[deprecated(
    since = "0.11.0-beta.5",
    note = "Deprecated with VisionCapability; use multimodal Chat messages instead."
)]
pub type VisionResponse = ();
