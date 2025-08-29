//! `OpenAI` Image Generation Implementation
//!
//! This module provides the `OpenAI` implementation of the `ImageGenerationCapability` trait,
//! including DALL-E image generation, editing, and variations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::ImageGenerationCapability;
use crate::types::{
    GeneratedImage, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageVariationRequest,
};

use super::config::OpenAiConfig;

/// `OpenAI` image generation API request structure
#[derive(Debug, Clone, Serialize)]
struct OpenAiImageRequest {
    /// Text prompt describing the image
    prompt: String,
    /// Negative prompt (what to avoid) - for SiliconFlow
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    /// Model to use (dall-e-2 or dall-e-3)
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Number of images to generate (1-10 for dall-e-2, 1 for dall-e-3)
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    /// Image size
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    /// Quality (dall-e-3 only)
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    /// Style (dall-e-3 only)
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
    /// Response format (url or `b64_json`)
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// `OpenAI` image generation API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiImageResponse {
    /// Creation timestamp
    created: u64,
    /// Generated images
    data: Vec<OpenAiImageData>,
}

/// Individual image data
#[derive(Debug, Clone, Deserialize)]
struct OpenAiImageData {
    /// Image URL (if `response_format` is "url")
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    /// Base64 encoded image (if `response_format` is "`b64_json`")
    #[serde(skip_serializing_if = "Option::is_none")]
    b64_json: Option<String>,
    /// Revised prompt (dall-e-3 only)
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
}

/// `OpenAI` image generation capability implementation.
///
/// This struct provides the OpenAI-specific implementation of image generation
/// using the DALL-E models.
///
/// # Supported Models
/// - dall-e-2: Can generate 1-10 images, sizes: 256x256, 512x512, 1024x1024
/// - dall-e-3: Can generate 1 image, sizes: 1024x1024, 1792x1024, 1024x1792
///
/// # API Reference
/// <https://platform.openai.com/docs/api-reference/images/create>
#[derive(Debug, Clone)]
pub struct OpenAiImages {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiImages {
    /// Create a new `OpenAI` images instance.
    ///
    /// # Arguments
    /// * `config` - `OpenAI` configuration
    /// * `http_client` - HTTP client for making requests
    pub const fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Check if this is a SiliconFlow instance
    fn is_siliconflow(&self) -> bool {
        self.config.base_url.contains("siliconflow.cn")
    }

    /// Get the default image generation model.
    fn default_model(&self) -> String {
        if self.is_siliconflow() {
            "Kwai-Kolors/Kolors".to_string()
        } else {
            "dall-e-3".to_string()
        }
    }

    /// Get supported image generation models.
    fn get_supported_models(&self) -> Vec<String> {
        if self.is_siliconflow() {
            vec![
                "Kwai-Kolors/Kolors".to_string(),
                "black-forest-labs/FLUX.1-schnell".to_string(),
                "stabilityai/stable-diffusion-3.5-large".to_string(),
            ]
        } else {
            vec![
                "dall-e-2".to_string(),
                "dall-e-3".to_string(),
                "gpt-image-1".to_string(), // New model
            ]
        }
    }

    /// Convert OpenAI request to SiliconFlow format if needed
    fn convert_request_for_provider(&self, request: &OpenAiImageRequest) -> serde_json::Value {
        if self.is_siliconflow() {
            // Convert to SiliconFlow format
            let mut siliconflow_request = serde_json::json!({
                "model": request.model.as_ref().unwrap_or(&self.default_model()),
                "prompt": request.prompt,
                "image_size": request.size.as_ref().unwrap_or(&"1024x1024".to_string()),
                "batch_size": request.n.unwrap_or(1),
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            });

            // Add optional parameters
            if let Some(negative_prompt) = &request.negative_prompt {
                siliconflow_request["negative_prompt"] =
                    serde_json::Value::String(negative_prompt.clone());
            }

            siliconflow_request
        } else {
            // Use original OpenAI format
            serde_json::to_value(request).unwrap_or_default()
        }
    }

    /// Make an image generation API request.
    async fn make_request(
        &self,
        request: OpenAiImageRequest,
    ) -> Result<OpenAiImageResponse, LlmError> {
        let url = format!("{}/images/generations", self.config.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        // Convert request format based on provider
        let request_body = self.convert_request_for_provider(&request);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body)
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
                message: format!("OpenAI Images API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiImageResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        Ok(openai_response)
    }

    /// Convert `OpenAI` response to our standard format.
    fn convert_response(&self, openai_response: OpenAiImageResponse) -> ImageGenerationResponse {
        let images: Vec<GeneratedImage> = openai_response
            .data
            .into_iter()
            .map(|img| GeneratedImage {
                url: img.url,
                b64_json: img.b64_json,
                format: None, // OpenAI doesn't provide format info in response
                width: None,  // OpenAI doesn't provide dimensions in response
                height: None, // OpenAI doesn't provide dimensions in response
                revised_prompt: img.revised_prompt,
                metadata: HashMap::new(),
            })
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert(
            "created".to_string(),
            serde_json::Value::Number(openai_response.created.into()),
        );

        ImageGenerationResponse { images, metadata }
    }

    /// Get supported image sizes for the given model.
    fn get_supported_sizes(&self, model: &str) -> Vec<String> {
        match model {
            "dall-e-2" => vec![
                "256x256".to_string(),
                "512x512".to_string(),
                "1024x1024".to_string(),
            ],
            "dall-e-3" => vec![
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
            ],
            "gpt-image-1" => vec![
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
                "2048x2048".to_string(), // Higher resolution support
            ],
            _ => vec!["1024x1024".to_string()], // Default fallback
        }
    }

    /// Validate request parameters.
    fn validate_request(&self, request: &ImageGenerationRequest) -> Result<(), LlmError> {
        let model = request.model.as_deref().unwrap_or("dall-e-3");

        // Validate model is supported
        if !self.get_supported_models().contains(&model.to_string()) {
            return Err(LlmError::InvalidInput(format!(
                "Unsupported model: {}. Supported models: {:?}",
                model,
                self.get_supported_models()
            )));
        }

        // Validate count based on model
        match model {
            "dall-e-2" => {
                if request.count > 10 {
                    return Err(LlmError::InvalidInput(
                        "DALL-E 2 can generate at most 10 images".to_string(),
                    ));
                }
            }
            "dall-e-3" => {
                if request.count > 1 {
                    return Err(LlmError::InvalidInput(
                        "DALL-E 3 can generate only 1 image at a time".to_string(),
                    ));
                }
            }
            "gpt-image-1" => {
                if request.count > 4 {
                    return Err(LlmError::InvalidInput(
                        "GPT-Image-1 can generate at most 4 images".to_string(),
                    ));
                }
            }
            _ => {
                // This should not happen due to model validation above
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported model: {model}"
                )));
            }
        }

        // Validate size
        if let Some(size) = &request.size {
            let supported_sizes = self.get_supported_sizes(model);
            if !supported_sizes.contains(size) {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported size '{size}' for model '{model}'. Supported sizes: {supported_sizes:?}"
                )));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiImages {
    /// Generate images from text prompts.
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // Validate request
        self.validate_request(&request)?;

        // Use model from request or default
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.default_model());

        let openai_request = OpenAiImageRequest {
            prompt: request.prompt,
            negative_prompt: request.negative_prompt,
            model: Some(model),
            n: if request.count > 0 {
                Some(request.count)
            } else {
                Some(1)
            },
            size: request.size,
            quality: request.quality,
            style: request.style,
            response_format: Some("url".to_string()), // Default to URL
            user: None,                               // Could be added to request if needed
        };

        let openai_response = self.make_request(openai_request).await?;
        Ok(self.convert_response(openai_response))
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String> {
        if self.is_siliconflow() {
            // SiliconFlow supported sizes
            vec![
                "1024x1024".to_string(),
                "960x1280".to_string(),
                "768x1024".to_string(),
                "720x1440".to_string(),
                "720x1280".to_string(),
            ]
        } else {
            // OpenAI supported sizes
            vec![
                "256x256".to_string(),
                "512x512".to_string(),
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
                "2048x2048".to_string(), // New size for gpt-image-1
            ]
        }
    }

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String> {
        if self.is_siliconflow() {
            // SiliconFlow only supports URL format
            vec!["url".to_string()]
        } else {
            // OpenAI supports both formats
            vec!["url".to_string(), "b64_json".to_string()]
        }
    }

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        !self.is_siliconflow() // SiliconFlow doesn't support image editing
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        !self.is_siliconflow() // SiliconFlow doesn't support image variations
    }

    /// Edit an existing image based on a prompt.
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // OpenAI image editing API request
        let url = format!("{}/images/edits", self.config.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        // Create multipart form
        let mut form = reqwest::multipart::Form::new().text("prompt", request.prompt);

        // Add image file
        let part = reqwest::multipart::Part::bytes(request.image)
            .file_name("image.png")
            .mime_str("image/png")?;
        form = form.part("image", part);

        // Add mask if provided
        if let Some(mask_data) = request.mask {
            let part = reqwest::multipart::Part::bytes(mask_data)
                .file_name("mask.png")
                .mime_str("image/png")?;
            form = form.part("mask", part);
        }

        // Add optional parameters
        if let Some(size) = request.size {
            form = form.text("size", size);
        }
        if let Some(count) = request.count
            && count > 0
        {
            form = form.text("n", count.to_string());
        }
        if let Some(response_format) = request.response_format {
            form = form.text("response_format", response_format);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .multipart(form)
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
                message: format!("OpenAI Images API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiImageResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        Ok(self.convert_response(openai_response))
    }

    /// Create variations of an existing image.
    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // OpenAI image variations API request
        let url = format!("{}/images/variations", self.config.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        // Create multipart form
        let mut form = reqwest::multipart::Form::new();

        // Add image file
        let part = reqwest::multipart::Part::bytes(request.image)
            .file_name("image.png")
            .mime_str("image/png")?;
        form = form.part("image", part);

        // Add optional parameters
        if let Some(size) = request.size {
            form = form.text("size", size);
        }
        if let Some(count) = request.count
            && count > 0
        {
            form = form.text("n", count.to_string());
        }
        if let Some(response_format) = request.response_format {
            form = form.text("response_format", response_format);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .multipart(form)
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
                message: format!("OpenAI Images API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiImageResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        Ok(self.convert_response(openai_response))
    }
}
