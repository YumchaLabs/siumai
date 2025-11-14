//! OpenAI Image Generation API Standard
//!
//! Implements the OpenAI Image Generation API format, which is supported
//! by many providers for image generation, editing, and variation tasks.

use siumai_core::error::LlmError;
use siumai_core::execution::image::{
    ImageHttpBody, ImageRequestTransformer, ImageResponseTransformer,
};
use siumai_core::types::image::{
    GeneratedImage, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageVariationRequest,
};
use std::sync::Arc;

/// OpenAI Image API Standard
#[derive(Clone)]
pub struct OpenAiImageStandard {
    /// Optional adapter for provider-specific differences
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
}

impl OpenAiImageStandard {
    /// Create a new standard OpenAI Image implementation
    pub fn new() -> Self {
        Self { adapter: None }
    }

    /// Create with a provider-specific adapter
    pub fn with_adapter(adapter: Arc<dyn OpenAiImageAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    /// Create transformers for image requests
    pub fn create_transformers(&self, provider_id: &str) -> ImageTransformers {
        let request_tx = Arc::new(OpenAiImageRequestTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        let response_tx = Arc::new(OpenAiImageResponseTransformer {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        });

        ImageTransformers {
            request: request_tx,
            response: response_tx,
        }
    }
}

impl Default for OpenAiImageStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in OpenAI Image API
pub trait OpenAiImageAdapter: Send + Sync {
    /// Transform image generation request JSON before sending
    fn transform_generation_request(
        &self,
        _req: &ImageGenerationRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform image edit request before sending
    fn transform_edit_request(
        &self,
        _req: &ImageEditRequest,
        _form: &mut reqwest::multipart::Form,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform image variation request before sending
    fn transform_variation_request(
        &self,
        _req: &ImageVariationRequest,
        _form: &mut reqwest::multipart::Form,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform response JSON after receiving
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Get provider-specific endpoint path for image generation
    fn generation_endpoint(&self) -> &str {
        "/images/generations"
    }
    /// Get provider-specific endpoint path for image editing
    fn edit_endpoint(&self) -> &str {
        "/images/edits"
    }
    /// Get provider-specific endpoint path for image variations
    fn variation_endpoint(&self) -> &str {
        "/images/variations"
    }

    /// Get provider-specific headers override (optional)
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Image transformers bundle
pub struct ImageTransformers {
    pub request: Arc<dyn ImageRequestTransformer>,
    pub response: Arc<dyn ImageResponseTransformer>,
}

/// Request transformer for OpenAI Image API
struct OpenAiImageRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
}

impl ImageRequestTransformer for OpenAiImageRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Build base OpenAI image generation request
        let mut body = serde_json::json!({
            "prompt": request.prompt,
        });

        // Add optional fields
        if let Some(n) = Some(request.count).filter(|c| *c > 0) {
            body["n"] = serde_json::json!(n);
        }
        if let Some(size) = &request.size {
            body["size"] = serde_json::json!(size);
        }
        if let Some(model) = &request.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(quality) = &request.quality {
            body["quality"] = serde_json::json!(quality);
        }
        if let Some(style) = &request.style {
            body["style"] = serde_json::json!(style);
        }
        if let Some(response_format) = &request.response_format {
            body["response_format"] = serde_json::json!(response_format);
        }

        // Add extra params
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &request.extra_params {
                obj.insert(k.clone(), v.clone());
            }
        }

        // Apply adapter transformation
        if let Some(adapter) = &self.adapter {
            adapter.transform_generation_request(request, &mut body)?;
        }

        Ok(body)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        use reqwest::multipart::{Form, Part};

        // Build multipart form for OpenAI Images Edit
        let mut form = Form::new().text("prompt", req.prompt.clone());

        let image_mime = siumai_core::utils::mime::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid MIME type: {}", e)))?;
        form = form.part("image", image_part);

        if let Some(mask) = &req.mask {
            let mask_mime = siumai_core::utils::mime::guess_mime(Some(mask), None);
            let mask_part = Part::bytes(mask.clone())
                .file_name("mask")
                .mime_str(&mask_mime)
                .map_err(|e| LlmError::InvalidParameter(format!("Invalid MIME type: {}", e)))?;
            form = form.part("mask", mask_part);
        }

        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(response_format) = &req.response_format {
            form = form.text("response_format", response_format.clone());
        }

        // Apply adapter transformation
        if let Some(adapter) = &self.adapter {
            adapter.transform_edit_request(req, &mut form)?;
        }

        Ok(ImageHttpBody::Multipart(form))
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        use reqwest::multipart::{Form, Part};

        // Build multipart form for OpenAI Images Variation
        let mut form = Form::new();

        let image_mime = siumai_core::utils::mime::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid MIME type: {}", e)))?;
        form = form.part("image", image_part);

        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(response_format) = &req.response_format {
            form = form.text("response_format", response_format.clone());
        }

        // Apply adapter transformation
        if let Some(adapter) = &self.adapter {
            adapter.transform_variation_request(req, &mut form)?;
        }

        Ok(ImageHttpBody::Multipart(form))
    }
}

/// Response transformer for OpenAI Image API
struct OpenAiImageResponseTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
}

impl ImageResponseTransformer for OpenAiImageResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut resp = raw.clone();
        // Apply adapter transformation first
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }

        // Parse OpenAI image response format
        let data = resp.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            LlmError::ParseError("Missing or invalid 'data' field in image response".to_string())
        })?;

        let images = data
            .iter()
            .map(|img| {
                let url = img.get("url").and_then(|u| u.as_str()).map(String::from);
                let b64_json = img
                    .get("b64_json")
                    .and_then(|b| b.as_str())
                    .map(String::from);
                let revised_prompt = img
                    .get("revised_prompt")
                    .and_then(|p| p.as_str())
                    .map(String::from);

                GeneratedImage {
                    url,
                    b64_json,
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt,
                    metadata: Default::default(),
                }
            })
            .collect::<Vec<_>>();

        Ok(ImageGenerationResponse {
            images,
            metadata: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_request_transformer() {
        let standard = OpenAiImageStandard::new();
        let transformers = standard.create_transformers("test-provider");

        let request = ImageGenerationRequest {
            prompt: "A beautiful sunset".to_string(),
            count: 2,
            size: Some("1024x1024".to_string()),
            model: Some("dall-e-3".to_string()),
            quality: Some("hd".to_string()),
            ..Default::default()
        };

        let result = transformers.request.transform_image(&request).unwrap();
        assert_eq!(result["prompt"], "A beautiful sunset");
        assert_eq!(result["n"], 2);
        assert_eq!(result["size"], "1024x1024");
        assert_eq!(result["model"], "dall-e-3");
        assert_eq!(result["quality"], "hd");
    }

    #[test]
    fn test_image_response_transformer() {
        let standard = OpenAiImageStandard::new();
        let transformers = standard.create_transformers("test-provider");

        let response_json = serde_json::json!({
            "data": [
                { "url": "https://example.com/image1.png", "revised_prompt": "A beautiful sunset over the ocean" },
                { "b64_json": "base64encodeddata" }
            ]
        });

        let response = transformers
            .response
            .transform_image_response(&response_json)
            .unwrap();
        assert_eq!(response.images.len(), 2);
        assert_eq!(
            response.images[0].url,
            Some("https://example.com/image1.png".to_string())
        );
        assert_eq!(
            response.images[0].revised_prompt,
            Some("A beautiful sunset over the ocean".to_string())
        );
        assert_eq!(
            response.images[1].b64_json,
            Some("base64encodeddata".to_string())
        );
    }
}
