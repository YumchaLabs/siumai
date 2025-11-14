//! Image generation and processing types (subset)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::common::HttpConfig;

/// Image generation request
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub size: Option<String>,
    pub count: u32,
    pub model: Option<String>,
    pub quality: Option<String>,
    pub style: Option<String>,
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub enhance_prompt: Option<bool>,
    pub response_format: Option<String>,
    pub extra_params: HashMap<String, serde_json::Value>,
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

/// Image edit request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEditRequest {
    pub image: Vec<u8>,
    pub mask: Option<Vec<u8>>,
    pub prompt: String,
    pub count: Option<u32>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub extra_params: HashMap<String, serde_json::Value>,
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

/// Image variation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageVariationRequest {
    pub image: Vec<u8>,
    pub count: Option<u32>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub extra_params: HashMap<String, serde_json::Value>,
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
}

/// Image generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A single generated image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub url: Option<String>,
    pub b64_json: Option<String>,
    pub format: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub revised_prompt: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}
