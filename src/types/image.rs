//! Image generation and processing types

use std::collections::HashMap;

/// Image generation request
#[derive(Debug, Clone, Default)]
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
}

/// Image edit request
#[derive(Debug, Clone)]
pub struct ImageEditRequest {
    /// Original image data
    pub image: Vec<u8>,
    /// Mask image data (optional)
    pub mask: Option<Vec<u8>>,
    /// Text prompt for editing
    pub prompt: String,
    /// Number of images to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Image variation request
#[derive(Debug, Clone)]
pub struct ImageVariationRequest {
    /// Original image data
    pub image: Vec<u8>,
    /// Number of variations to generate
    pub count: Option<u32>,
    /// Image size
    pub size: Option<String>,
    /// Response format
    pub response_format: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Image generation response
#[derive(Debug, Clone)]
pub struct ImageGenerationResponse {
    /// Generated images
    pub images: Vec<GeneratedImage>,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A single generated image
#[derive(Debug, Clone)]
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
pub type ImageGenRequest = ();
pub type ImageResponse = ();
pub type VisionRequest = ();
pub type VisionResponse = ();
