//! OpenAI Image Generation API Standard
//!
//! This module implements the OpenAI Image Generation API format, which is supported
//! by many providers for image generation, editing, and variation tasks.
//!
//! ## Supported Providers
//!
//! - OpenAI (native)
//! - OpenAI-compatible providers that support image generation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use siumai::standards::openai::image::OpenAiImageStandard;
//!
//! // Standard OpenAI implementation
//! let standard = OpenAiImageStandard::new();
//!
//! // With provider-specific adapter
//! let standard = OpenAiImageStandard::with_adapter(
//!     Arc::new(MyCustomAdapter)
//! );
//! ```

use crate::error::LlmError;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use crate::{core::ProviderContext, core::ProviderSpec};
use std::sync::Arc;

/// OpenAI Image API Standard
///
/// Represents the OpenAI Image Generation API format.
/// Can be used by any provider that implements OpenAI-compatible image API.
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

    /// Create a ProviderSpec wrapper for this standard.
    pub fn create_spec(&self, provider_id: &'static str) -> OpenAiImageSpec {
        OpenAiImageSpec {
            provider_id,
            adapter: self.adapter.clone(),
        }
    }
}

impl Default for OpenAiImageStandard {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter trait for provider-specific differences in OpenAI Image API
///
/// Implement this trait to handle provider-specific variations of the OpenAI Image API.
pub trait OpenAiImageAdapter: Send + Sync {
    /// Transform image generation request JSON before sending
    ///
    /// This is called after the standard OpenAI request transformation.
    /// Use this to add provider-specific fields or modify existing ones.
    fn transform_generation_request(
        &self,
        _req: &ImageGenerationRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform image edit request before sending
    ///
    /// This is called after the standard OpenAI multipart form construction.
    /// Use this to add provider-specific form fields.
    fn transform_edit_request(
        &self,
        _req: &ImageEditRequest,
        _form: &mut reqwest::multipart::Form,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform image variation request before sending
    ///
    /// This is called after the standard OpenAI multipart form construction.
    /// Use this to add provider-specific form fields.
    fn transform_variation_request(
        &self,
        _req: &ImageVariationRequest,
        _form: &mut reqwest::multipart::Form,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Transform response JSON after receiving
    ///
    /// This is called before the standard OpenAI response transformation.
    /// Use this to normalize provider-specific response fields.
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    /// Get provider-specific endpoint path for image generation
    ///
    /// Default is "/images/generations" (standard OpenAI)
    fn generation_endpoint(&self) -> &str {
        "/images/generations"
    }

    /// Get provider-specific endpoint path for image editing
    ///
    /// Default is "/images/edits" (standard OpenAI)
    fn edit_endpoint(&self) -> &str {
        "/images/edits"
    }

    /// Get provider-specific endpoint path for image variations
    ///
    /// Default is "/images/variations" (standard OpenAI)
    fn variation_endpoint(&self) -> &str {
        "/images/variations"
    }

    /// Get provider-specific headers
    ///
    /// Default is standard OpenAI headers (Authorization: Bearer <token>)
    fn build_headers(
        &self,
        _api_key: &str,
        _base_headers: &mut reqwest::header::HeaderMap,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// ProviderSpec implementation for OpenAI Image Standard.
pub struct OpenAiImageSpec {
    provider_id: &'static str,
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
}

impl ProviderSpec for OpenAiImageSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers =
            crate::standards::openai::headers::build_openai_compatible_json_headers(ctx)?;

        if let Some(adapter) = &self.adapter {
            adapter.build_headers(ctx.api_key.as_deref().unwrap_or(""), &mut headers)?;
        }

        Ok(headers)
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let endpoint = self
            .adapter
            .as_ref()
            .map(|a| a.generation_endpoint())
            .unwrap_or("/images/generations");
        crate::utils::url::join_url(&ctx.base_url, endpoint)
    }

    fn image_edit_url(&self, _req: &ImageEditRequest, ctx: &ProviderContext) -> String {
        let endpoint = self
            .adapter
            .as_ref()
            .map(|a| a.edit_endpoint())
            .unwrap_or("/images/edits");
        crate::utils::url::join_url(&ctx.base_url, endpoint)
    }

    fn image_variation_url(&self, _req: &ImageVariationRequest, ctx: &ProviderContext) -> String {
        let endpoint = self
            .adapter
            .as_ref()
            .map(|a| a.variation_endpoint())
            .unwrap_or("/images/variations");
        crate::utils::url::join_url(&ctx.base_url, endpoint)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> crate::core::ImageTransformers {
        let standard = OpenAiImageStandard {
            adapter: self.adapter.clone(),
        };
        let t = standard.create_transformers(&ctx.provider_id);
        crate::core::ImageTransformers {
            request: t.request,
            response: t.response,
        }
    }
}

/// Image transformers bundle
pub struct ImageTransformers {
    pub request: Arc<dyn RequestTransformer>,
    pub response: Arc<dyn ResponseTransformer>,
}

/// Request transformer for OpenAI Image API
struct OpenAiImageRequestTransformer {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiImageAdapter>>,
}

impl RequestTransformer for OpenAiImageRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Chat is not supported by image transformer".to_string(),
        ))
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

        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid MIME type: {}", e)))?;
        form = form.part("image", image_part);

        if let Some(mask) = &req.mask {
            let mask_mime = crate::utils::guess_mime(Some(mask), None);
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

        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
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

impl ResponseTransformer for OpenAiImageResponseTransformer {
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

                crate::types::GeneratedImage {
                    url,
                    b64_json,
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt,
                    metadata: std::collections::HashMap::new(),
                }
            })
            .collect();

        Ok(ImageGenerationResponse {
            images,
            metadata: std::collections::HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_image_standard_new() {
        let standard = OpenAiImageStandard::new();
        assert!(standard.adapter.is_none());
    }

    #[test]
    fn test_openai_image_standard_with_adapter() {
        struct TestAdapter;
        impl OpenAiImageAdapter for TestAdapter {}

        let standard = OpenAiImageStandard::with_adapter(Arc::new(TestAdapter));
        assert!(standard.adapter.is_some());
    }

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

        let result = transformers.request.transform_image(&request);
        assert!(result.is_ok());

        let body = result.unwrap();
        assert_eq!(body["prompt"], "A beautiful sunset");
        assert_eq!(body["n"], 2);
        assert_eq!(body["size"], "1024x1024");
        assert_eq!(body["model"], "dall-e-3");
        assert_eq!(body["quality"], "hd");
    }

    #[test]
    fn test_image_response_transformer() {
        let standard = OpenAiImageStandard::new();
        let transformers = standard.create_transformers("test-provider");

        let response_json = serde_json::json!({
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean"
                },
                {
                    "b64_json": "base64encodeddata",
                }
            ]
        });

        let result = transformers
            .response
            .transform_image_response(&response_json);
        assert!(result.is_ok());

        let response = result.unwrap();
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

    #[test]
    fn test_image_edit_request_transformer() {
        let standard = OpenAiImageStandard::new();
        let transformers = standard.create_transformers("test-provider");

        let request = ImageEditRequest {
            image: vec![1, 2, 3, 4],
            mask: Some(vec![5, 6, 7, 8]),
            prompt: "Add a rainbow".to_string(),
            count: Some(1),
            size: Some("512x512".to_string()),
            response_format: Some("url".to_string()),
            extra_params: std::collections::HashMap::new(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let result = transformers.request.transform_image_edit(&request);
        assert!(result.is_ok());

        // Verify it returns multipart form
        match result.unwrap() {
            ImageHttpBody::Multipart(_) => {
                // Success - it's a multipart form
            }
            _ => panic!("Expected multipart form"),
        }
    }

    #[test]
    fn test_image_variation_request_transformer() {
        let standard = OpenAiImageStandard::new();
        let transformers = standard.create_transformers("test-provider");

        let request = ImageVariationRequest {
            image: vec![1, 2, 3, 4],
            count: Some(2),
            size: Some("1024x1024".to_string()),
            response_format: Some("b64_json".to_string()),
            extra_params: std::collections::HashMap::new(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let result = transformers.request.transform_image_variation(&request);
        assert!(result.is_ok());

        // Verify it returns multipart form
        match result.unwrap() {
            ImageHttpBody::Multipart(_) => {
                // Success - it's a multipart form
            }
            _ => panic!("Expected multipart form"),
        }
    }

    #[test]
    fn test_adapter_transform_generation_request() {
        struct CustomAdapter;
        impl OpenAiImageAdapter for CustomAdapter {
            fn transform_generation_request(
                &self,
                _req: &ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                body["custom_field"] = serde_json::json!("custom_value");
                Ok(())
            }
        }

        let standard = OpenAiImageStandard::with_adapter(Arc::new(CustomAdapter));
        let transformers = standard.create_transformers("test-provider");

        let request = ImageGenerationRequest {
            prompt: "Test prompt".to_string(),
            ..Default::default()
        };

        let result = transformers.request.transform_image(&request);
        assert!(result.is_ok());

        let body = result.unwrap();
        assert_eq!(body["custom_field"], "custom_value");
    }

    #[test]
    fn test_adapter_transform_response() {
        struct CustomAdapter;
        impl OpenAiImageAdapter for CustomAdapter {
            fn transform_response(&self, resp: &mut serde_json::Value) -> Result<(), LlmError> {
                // Normalize provider-specific response format
                if let Some(images) = resp.get_mut("images") {
                    *resp = serde_json::json!({ "data": images });
                }
                Ok(())
            }
        }

        let standard = OpenAiImageStandard::with_adapter(Arc::new(CustomAdapter));
        let transformers = standard.create_transformers("test-provider");

        let response_json = serde_json::json!({
            "images": [
                { "url": "https://example.com/image.png" }
            ]
        });

        let result = transformers
            .response
            .transform_image_response(&response_json);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.images.len(), 1);
    }
}
