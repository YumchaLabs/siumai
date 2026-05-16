//! Image generation and processing types

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    DataContent, HttpConfig, HttpResponseInfo, InvalidDataContentError, ProviderOptionsMap, Warning,
};

/// Image generation request
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the image
    pub prompt: String,
    /// Negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Image size (e.g., "1024x1024")
    pub size: Option<String>,
    /// Aspect ratio (e.g., "16:9")
    #[serde(
        default,
        rename = "aspectRatio",
        alias = "aspect_ratio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
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
    /// Set the aspect ratio for the request.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the random seed for the request.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

/// High-level image prompt shape aligned with AI SDK `GenerateImagePrompt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum GenerateImagePrompt {
    /// Text-only image prompt.
    Text(String),
    /// Image edit/inpainting prompt with optional text and mask.
    Images {
        /// Input images for edit-style generation.
        images: Vec<DataContent>,
        /// Optional text description for the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Optional mask for inpainting-style generation.
        #[serde(skip_serializing_if = "Option::is_none")]
        mask: Option<DataContent>,
    },
}

impl GenerateImagePrompt {
    /// Create a text-only prompt.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create an image prompt without text.
    pub fn images(images: impl Into<Vec<DataContent>>) -> Self {
        Self::Images {
            images: images.into(),
            text: None,
            mask: None,
        }
    }

    /// Create an image prompt with accompanying text.
    pub fn images_with_text(images: impl Into<Vec<DataContent>>, text: impl Into<String>) -> Self {
        Self::Images {
            images: images.into(),
            text: Some(text.into()),
            mask: None,
        }
    }

    /// Attach or replace the optional mask for image prompts.
    pub fn with_mask(mut self, mask: impl Into<DataContent>) -> Self {
        if let Self::Images { mask: current, .. } = &mut self {
            *current = Some(mask.into());
        }
        self
    }

    /// Return the text portion of the prompt when present.
    pub fn text_part(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Images { text, .. } => text.as_deref(),
        }
    }

    /// Return the image inputs when present.
    pub fn image_parts(&self) -> Option<&[DataContent]> {
        match self {
            Self::Text(_) => None,
            Self::Images { images, .. } => Some(images.as_slice()),
        }
    }

    /// Return the mask input when present.
    pub fn mask_part(&self) -> Option<&DataContent> {
        match self {
            Self::Text(_) => None,
            Self::Images { mask, .. } => mask.as_ref(),
        }
    }
}

impl From<String> for GenerateImagePrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for GenerateImagePrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<DataContent>> for GenerateImagePrompt {
    fn from(value: Vec<DataContent>) -> Self {
        Self::images(value)
    }
}

impl<const N: usize> From<[DataContent; N]> for GenerateImagePrompt {
    fn from(value: [DataContent; N]) -> Self {
        Self::images(value)
    }
}

/// AI SDK-style unified image request surface.
///
/// This request bridges generation, editing, and variation creation through one
/// stable shape that mirrors the upstream `generateImage()` call options more
/// closely than the older split request family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateImageRequest {
    /// Optional text prompt describing the target image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Input files/URLs for edit or variation-style requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ImageEditInput>,
    /// Optional mask image for inpainting-style requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<ImageEditInput>,
    /// Negative prompt (what to avoid).
    pub negative_prompt: Option<String>,
    /// Image size (e.g., `1024x1024`).
    pub size: Option<String>,
    /// Aspect ratio (e.g., `16:9`).
    #[serde(
        default,
        rename = "aspectRatio",
        alias = "aspect_ratio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
    /// Number of images to generate.
    pub count: u32,
    /// Optional provider-specific per-request model override.
    pub model: Option<String>,
    /// Quality setting.
    pub quality: Option<String>,
    /// Style setting.
    pub style: Option<String>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Number of inference steps.
    pub steps: Option<u32>,
    /// Guidance scale.
    pub guidance_scale: Option<f32>,
    /// Whether to enhance the prompt.
    pub enhance_prompt: Option<bool>,
    /// Response format (url or `b64_json`).
    pub response_format: Option<String>,
    /// Additional provider-specific parameters.
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,
    /// Per-request HTTP configuration (headers, timeout, etc.).
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

impl Default for GenerateImageRequest {
    fn default() -> Self {
        Self {
            prompt: None,
            files: Vec::new(),
            mask: None,
            negative_prompt: None,
            size: None,
            aspect_ratio: None,
            count: 1,
            model: None,
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: None,
            extra_params: HashMap::new(),
            provider_options_map: ProviderOptionsMap::default(),
            http_config: None,
        }
    }
}

impl GenerateImageRequest {
    /// Create a new unified image request with a text prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self::default().with_prompt(prompt)
    }

    /// Set or replace the prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Replace the full input-file list.
    pub fn with_files<I>(mut self, files: I) -> Self
    where
        I: IntoIterator<Item = ImageEditInput>,
    {
        self.files = files.into_iter().collect();
        self
    }

    /// Replace the full input-file list with a single file.
    pub fn with_file(mut self, file: ImageEditInput) -> Self {
        self.files = vec![file];
        self
    }

    /// Append one input file.
    pub fn push_file(mut self, file: ImageEditInput) -> Self {
        self.files.push(file);
        self
    }

    /// Set the optional mask input.
    pub fn with_mask(mut self, mask: ImageEditInput) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Set the aspect ratio for the request.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the random seed for the request.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

fn normalized_generate_image_prompt(prompt: String) -> Option<String> {
    (!prompt.trim().is_empty()).then_some(prompt)
}

fn is_url_like_prompt_content(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://") || value.starts_with("data:")
}

fn image_edit_input_from_prompt_content(content: DataContent) -> ImageEditInput {
    match content {
        DataContent::Base64(value) => {
            if is_url_like_prompt_content(&value) {
                ImageEditInput::url(value)
            } else {
                ImageEditInput::base64(value)
            }
        }
        DataContent::Binary(value) => ImageEditInput::file(value),
    }
}

impl From<GenerateImagePrompt> for GenerateImageRequest {
    fn from(prompt: GenerateImagePrompt) -> Self {
        match prompt {
            GenerateImagePrompt::Text(text) => Self::new(text),
            GenerateImagePrompt::Images { images, text, mask } => Self {
                prompt: text,
                files: images
                    .into_iter()
                    .map(image_edit_input_from_prompt_content)
                    .collect(),
                mask: mask.map(image_edit_input_from_prompt_content),
                ..Self::default()
            },
        }
    }
}

impl From<ImageGenerationRequest> for GenerateImageRequest {
    fn from(request: ImageGenerationRequest) -> Self {
        Self {
            prompt: normalized_generate_image_prompt(request.prompt),
            files: Vec::new(),
            mask: None,
            negative_prompt: request.negative_prompt,
            size: request.size,
            aspect_ratio: request.aspect_ratio,
            count: request.count.max(1),
            model: request.model,
            quality: request.quality,
            style: request.style,
            seed: request.seed,
            steps: request.steps,
            guidance_scale: request.guidance_scale,
            enhance_prompt: request.enhance_prompt,
            response_format: request.response_format,
            extra_params: request.extra_params,
            provider_options_map: request.provider_options_map,
            http_config: request.http_config,
        }
    }
}

impl From<ImageEditRequest> for GenerateImageRequest {
    fn from(request: ImageEditRequest) -> Self {
        Self {
            prompt: normalized_generate_image_prompt(request.prompt),
            files: request.images,
            mask: request.mask,
            negative_prompt: None,
            size: request.size,
            aspect_ratio: request.aspect_ratio,
            count: request.count.unwrap_or(1).max(1),
            model: request.model,
            quality: None,
            style: None,
            seed: request.seed,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: request.response_format,
            extra_params: request.extra_params,
            provider_options_map: request.provider_options_map,
            http_config: request.http_config,
        }
    }
}

impl From<ImageVariationRequest> for GenerateImageRequest {
    fn from(request: ImageVariationRequest) -> Self {
        Self {
            prompt: None,
            files: vec![request.image],
            mask: None,
            negative_prompt: None,
            size: request.size,
            aspect_ratio: request.aspect_ratio,
            count: request.count.unwrap_or(1).max(1),
            model: request.model,
            quality: None,
            style: None,
            seed: request.seed,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: request.response_format,
            extra_params: request.extra_params,
            provider_options_map: request.provider_options_map,
            http_config: request.http_config,
        }
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
    pub fn as_bytes(&self) -> Result<Vec<u8>, InvalidDataContentError> {
        match self {
            Self::Base64(data) => base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|source| InvalidDataContentError::invalid_base64(data.clone(), source)),
            Self::Binary(data) => Ok(data.clone()),
        }
    }
}

impl From<DataContent> for ImageEditFileData {
    fn from(value: DataContent) -> Self {
        match value {
            DataContent::Base64(data) => Self::Base64(data),
            DataContent::Binary(data) => Self::Binary(data),
        }
    }
}

impl From<ImageEditFileData> for DataContent {
    fn from(value: ImageEditFileData) -> Self {
        match value {
            ImageEditFileData::Base64(data) => Self::Base64(data),
            ImageEditFileData::Binary(data) => Self::Binary(data),
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
        /// Optional provider-specific metadata for this file input.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options_map: ProviderOptionsMap,
    },
    /// URL-like image input.
    Url {
        /// URL or data URL.
        url: String,
        /// Optional provider-specific metadata for this URL input.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options_map: ProviderOptionsMap,
    },
}

impl ImageEditInput {
    /// Create a file input from binary bytes.
    pub fn file(data: impl Into<Vec<u8>>) -> Self {
        Self::File {
            data: ImageEditFileData::binary(data),
            media_type: None,
            provider_options_map: ProviderOptionsMap::default(),
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
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create a file input from a base64 string and media type.
    pub fn base64_with_media_type(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::base64(data).with_media_type(media_type)
    }

    /// Create a file input from shared AI SDK-style data content.
    pub fn from_data_content(data: DataContent) -> Self {
        Self::File {
            data: ImageEditFileData::from(data),
            media_type: None,
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create a URL input.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            provider_options_map: ProviderOptionsMap::default(),
        }
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

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        match &mut self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => {
                *provider_options_map = map;
            }
        }
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        match &mut self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => {
                provider_options_map.insert(provider_id, options);
            }
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
            Self::Url { url, .. } => Some(url.as_str()),
            Self::File { .. } => None,
        }
    }

    /// Return the provider options map attached to this input.
    pub fn provider_options_map(&self) -> &ProviderOptionsMap {
        match self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => provider_options_map,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Aspect ratio (e.g., "16:9")
    #[serde(
        default,
        rename = "aspectRatio",
        alias = "aspect_ratio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
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

    /// Set the aspect ratio for the request.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the random seed for the request.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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
    /// Original image input aligned with AI SDK image model files.
    pub image: ImageEditInput,
    /// Model to use for variations (provider-specific; optional).
    ///
    /// This is required by some providers where the model is part of the request URL.
    pub model: Option<String>,
    /// Number of variations to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Aspect ratio (e.g., "16:9")
    #[serde(
        default,
        rename = "aspectRatio",
        alias = "aspect_ratio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
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

impl Default for ImageVariationRequest {
    fn default() -> Self {
        Self {
            image: ImageEditInput::file(Vec::<u8>::new()),
            model: None,
            count: None,
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: HashMap::new(),
            provider_options_map: ProviderOptionsMap::default(),
            http_config: None,
        }
    }
}

impl ImageVariationRequest {
    /// Set the source image input.
    pub fn with_image(mut self, image: ImageEditInput) -> Self {
        self.image = image;
        self
    }

    /// Set the aspect ratio for the request.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the random seed for the request.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_edit_input_helpers_and_provider_options() {
        let file = ImageEditInput::file(vec![1, 2, 3])
            .with_media_type("image/png")
            .with_provider_option("openaiCompatible", serde_json::json!({ "detail": "high" }));

        assert!(file.is_file());
        assert!(!file.is_url());
        assert_eq!(file.media_type(), Some("image/png"));
        assert_eq!(
            file.file_data()
                .expect("file data")
                .as_bytes()
                .expect("bytes"),
            vec![1, 2, 3]
        );
        assert_eq!(
            file.provider_options_map()
                .get("openaicompatible")
                .and_then(|value| value.get("detail"))
                .and_then(|value| value.as_str()),
            Some("high")
        );

        let url = ImageEditInput::url("https://example.com/input.png")
            .with_provider_option("google", serde_json::json!({ "mimeHint": "image/png" }));

        assert!(url.is_url());
        assert!(!url.is_file());
        assert_eq!(url.as_url(), Some("https://example.com/input.png"));
        assert!(url.file_data().is_none());
        assert_eq!(
            url.provider_options_map()
                .get("google")
                .and_then(|value| value.get("mimeHint"))
                .and_then(|value| value.as_str()),
            Some("image/png")
        );
    }

    #[test]
    fn test_image_edit_input_provider_options_serde_roundtrip() {
        let value = serde_json::to_value(
            ImageEditInput::file(vec![1, 2, 3])
                .with_provider_option("openaiCompatible", serde_json::json!({ "detail": "low" })),
        )
        .expect("serialize image input");

        assert_eq!(
            value.get("type").and_then(|value| value.as_str()),
            Some("file")
        );
        assert!(value.get("providerOptions").is_some());
        assert!(
            value
                .get("providerOptions")
                .and_then(|value| value.get("openaiCompatible"))
                .is_some()
        );
        assert!(
            value
                .get("providerOptions")
                .and_then(|value| value.get("openaicompatible"))
                .is_none()
        );

        let input: ImageEditInput = serde_json::from_value(serde_json::json!({
            "type": "url",
            "url": "https://example.com/input.png",
            "providerOptions": {
                "OpenAICompatible": {
                    "detail": "high"
                }
            }
        }))
        .expect("deserialize image input");

        assert_eq!(input.as_url(), Some("https://example.com/input.png"));
        assert_eq!(
            input
                .provider_options_map()
                .get("openaicompatible")
                .and_then(|value| value.get("detail"))
                .and_then(|value| value.as_str()),
            Some("high")
        );
    }

    #[test]
    fn test_image_edit_input_accepts_shared_data_content() {
        let input = ImageEditInput::from_data_content(DataContent::base64("AQID"))
            .with_media_type("image/png");

        assert_eq!(input.media_type(), Some("image/png"));
        assert_eq!(
            input
                .file_data()
                .expect("file data")
                .as_bytes()
                .expect("bytes"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn test_generate_image_prompt_matches_ai_sdk_shape() {
        let text_prompt = GenerateImagePrompt::text("draw a tea house");
        let text_json = serde_json::to_value(&text_prompt).expect("serialize text prompt");
        assert_eq!(text_json, serde_json::json!("draw a tea house"));

        let image_prompt = GenerateImagePrompt::images_with_text(
            vec![DataContent::base64("AQID")],
            "make it watercolor",
        )
        .with_mask(DataContent::base64("BAUG"));
        let image_json = serde_json::to_value(&image_prompt).expect("serialize image prompt");
        assert_eq!(
            image_json,
            serde_json::json!({
                "images": ["AQID"],
                "text": "make it watercolor",
                "mask": "BAUG"
            })
        );

        let decoded: GenerateImagePrompt =
            serde_json::from_value(image_json).expect("deserialize image prompt");
        assert_eq!(decoded.text_part(), Some("make it watercolor"));
        assert_eq!(
            decoded
                .image_parts()
                .and_then(|images| images.first())
                .map(DataContent::as_base64),
            Some("AQID".to_string())
        );
        assert_eq!(
            decoded.mask_part().map(DataContent::as_base64),
            Some("BAUG".to_string())
        );
    }

    #[test]
    fn test_generate_image_prompt_converts_to_unified_request() {
        let request = GenerateImageRequest::from(
            GenerateImagePrompt::images_with_text(
                [
                    DataContent::base64("https://example.com/input.png"),
                    DataContent::binary([1_u8, 2, 3]),
                ],
                "edit",
            )
            .with_mask("data:image/png;base64,BAUG"),
        );

        assert_eq!(request.prompt.as_deref(), Some("edit"));
        assert_eq!(request.files.len(), 2);
        assert_eq!(
            request.files.first().and_then(ImageEditInput::as_url),
            Some("https://example.com/input.png")
        );
        assert_eq!(
            request
                .files
                .get(1)
                .and_then(ImageEditInput::file_data)
                .and_then(|data| data.as_bytes().ok()),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            request.mask.as_ref().and_then(ImageEditInput::as_url),
            Some("data:image/png;base64,BAUG")
        );
    }
}
