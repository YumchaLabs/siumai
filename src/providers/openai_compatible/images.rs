//! SiliconFlow Image Generation Implementation
//!
//! This module provides SiliconFlow-specific image generation functionality
//! that adapts OpenAI's image generation interface to SiliconFlow's API.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::ImageGenerationCapability;
use crate::types::{GeneratedImage, ImageGenerationRequest, ImageGenerationResponse};

/// SiliconFlow-specific image generation request
#[derive(Debug, Clone, Serialize)]
pub struct SiliconFlowImageRequest {
    /// Model to use for generation
    pub model: String,
    /// Text prompt describing the image
    pub prompt: String,
    /// Negative prompt (what to avoid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Image size in "widthxheight" format
    pub image_size: String,
    /// Number of images to generate
    pub batch_size: u32,
    /// Random seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Number of inference steps
    pub num_inference_steps: u32,
    /// Guidance scale for prompt adherence
    pub guidance_scale: f64,
    /// Base64 encoded image for image-to-image generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// SiliconFlow image generation response
#[derive(Debug, Clone, Deserialize)]
pub struct SiliconFlowImageResponse {
    /// Generated images
    pub images: Vec<SiliconFlowImage>,
    /// Timing information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<SiliconFlowTimings>,
    /// Seed used for generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// SiliconFlow image object
#[derive(Debug, Clone, Deserialize)]
pub struct SiliconFlowImage {
    /// URL of the generated image
    pub url: String,
}

/// SiliconFlow timing information
#[derive(Debug, Clone, Deserialize)]
pub struct SiliconFlowTimings {
    /// Inference time in seconds
    pub inference: f64,
}

/// SiliconFlow image generation capability implementation
#[derive(Debug, Clone)]
pub struct SiliconFlowImages {
    /// API key for authentication
    api_key: String,
    /// Base URL for the API
    base_url: String,
    /// HTTP client
    http_client: reqwest::Client,
}

impl SiliconFlowImages {
    /// Create a new SiliconFlow images client
    pub fn new(api_key: String, base_url: String, http_client: reqwest::Client) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
        }
    }

    /// Get default model for image generation
    pub fn default_model() -> String {
        "Kwai-Kolors/Kolors".to_string()
    }

    /// Convert ImageGenerationRequest to SiliconFlowImageRequest
    fn convert_request(&self, request: ImageGenerationRequest) -> SiliconFlowImageRequest {
        let model = request.model.unwrap_or_else(|| Self::default_model());

        let image_size = request.size.unwrap_or_else(|| "1024x1024".to_string());

        SiliconFlowImageRequest {
            model,
            prompt: request.prompt,
            negative_prompt: request.negative_prompt,
            image_size,
            batch_size: if request.count > 0 { request.count } else { 1 },
            seed: request.seed,
            num_inference_steps: 20, // Default value
            guidance_scale: 7.5,     // Default value
            image: None,
        }
    }

    /// Convert SiliconFlowImageResponse to ImageGenerationResponse
    fn convert_response(&self, response: SiliconFlowImageResponse) -> ImageGenerationResponse {
        let images = response
            .images
            .into_iter()
            .map(|img| GeneratedImage {
                url: Some(img.url),
                b64_json: None,
                format: Some("url".to_string()),
                width: None,
                height: None,
                revised_prompt: None,
                metadata: HashMap::new(),
            })
            .collect();

        let mut metadata = HashMap::new();
        if let Some(timings) = response.timings {
            metadata.insert(
                "inference_time".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(timings.inference).unwrap()),
            );
        }
        if let Some(seed) = response.seed {
            metadata.insert(
                "seed".to_string(),
                serde_json::Value::Number(serde_json::Number::from(seed)),
            );
        }

        ImageGenerationResponse { images, metadata }
    }

    /// Make HTTP request to SiliconFlow API
    async fn make_request(
        &self,
        request: SiliconFlowImageRequest,
    ) -> Result<SiliconFlowImageResponse, LlmError> {
        let url = format!("{}/images/generations", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("SiliconFlow Images API error {status}: {error_text}"),
                details: None,
            });
        }

        let siliconflow_response: SiliconFlowImageResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        Ok(siliconflow_response)
    }
}

#[async_trait]
impl ImageGenerationCapability for SiliconFlowImages {
    /// Generate images from text prompts
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let siliconflow_request = self.convert_request(request);
        let siliconflow_response = self.make_request(siliconflow_request).await?;
        Ok(self.convert_response(siliconflow_response))
    }

    /// Get supported image sizes for SiliconFlow
    fn get_supported_sizes(&self) -> Vec<String> {
        vec![
            "1024x1024".to_string(),
            "960x1280".to_string(),
            "768x1024".to_string(),
            "720x1440".to_string(),
            "720x1280".to_string(),
        ]
    }

    /// Get supported response formats for SiliconFlow
    fn get_supported_formats(&self) -> Vec<String> {
        vec!["url".to_string()]
    }

    /// SiliconFlow doesn't support image editing
    fn supports_image_editing(&self) -> bool {
        false
    }

    /// SiliconFlow doesn't support image variations
    fn supports_image_variations(&self) -> bool {
        false
    }
}
