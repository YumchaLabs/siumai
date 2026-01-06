//! Vertex AI Imagen standard (Google publisher).
//!
//! This module provides a minimal request/response mapping for Vertex AI's Imagen
//! models using the `:predict` endpoint. It is intentionally scoped to image
//! generation/editing and does not implement chat/embeddings.

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::execution::transformers::response::ResponseTransformer;
use crate::standards::gemini::headers::build_gemini_headers;
use crate::types::{ChatRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest};
use base64::Engine;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

fn normalize_vertex_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }
    trimmed.to_string()
}

fn looks_like_vertex_base_url(base_url: &str) -> bool {
    base_url.contains("aiplatform.googleapis.com")
}

fn guess_mime(bytes: &[u8]) -> &'static str {
    infer::get(bytes)
        .map(|t| t.mime_type())
        .unwrap_or("image/png")
}

fn bytes_to_inline_image(bytes: &[u8]) -> serde_json::Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    serde_json::json!({
        "bytesBase64Encoded": b64,
        "mimeType": guess_mime(bytes),
    })
}

fn size_to_aspect_ratio(size: &str) -> Option<String> {
    let (w, h) = size.split_once('x')?;
    let w: u32 = w.trim().parse().ok()?;
    let h: u32 = h.trim().parse().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }
    let g = gcd(w, h);
    Some(format!("{}:{}", w / g, h / g))
}

fn extract_vertex_imagen_options(
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Value> {
    let v = map.get("gemini")?;
    let obj = v.as_object()?;

    // Accept both Vercel-style and snake_case keys.
    obj.get("vertexImagen")
        .cloned()
        .or_else(|| obj.get("vertex_imagen").cloned())
}

fn merge_object_skipping(
    into: &mut serde_json::Map<String, serde_json::Value>,
    value: &serde_json::Value,
    skip: &[&str],
) {
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            if skip.iter().any(|s| *s == k) {
                continue;
            }
            into.insert(k.clone(), v.clone());
        }
    }
}

fn vertex_imagen_reference_images(
    extra_params: &HashMap<String, serde_json::Value>,
    provider_opts: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    if let Some(v) = extra_params.get("referenceImages") {
        return Some(v.clone());
    }
    if let Some(v) = extra_params.get("reference_images") {
        return Some(v.clone());
    }
    if let Some(opts) = provider_opts
        && let Some(obj) = opts.as_object()
    {
        if let Some(v) = obj.get("referenceImages") {
            return Some(v.clone());
        }
        if let Some(v) = obj.get("reference_images") {
            return Some(v.clone());
        }
    }
    None
}

fn vertex_imagen_negative_prompt(
    req_negative_prompt: Option<&String>,
    extra_params: &HashMap<String, serde_json::Value>,
    provider_opts: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(v) = req_negative_prompt {
        return Some(v.clone());
    }
    if let Some(v) = extra_params.get("negativePrompt").and_then(|v| v.as_str()) {
        return Some(v.to_string());
    }
    if let Some(v) = extra_params.get("negative_prompt").and_then(|v| v.as_str()) {
        return Some(v.to_string());
    }
    if let Some(opts) = provider_opts
        && let Some(obj) = opts.as_object()
    {
        if let Some(v) = obj.get("negativePrompt").and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
        if let Some(v) = obj.get("negative_prompt").and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
    }
    None
}

#[derive(Clone)]
pub struct VertexImagenRequestTransformer {
    provider_id: &'static str,
}

impl VertexImagenRequestTransformer {
    pub fn new(provider_id: &'static str) -> Self {
        Self { provider_id }
    }
}

impl RequestTransformer for VertexImagenRequestTransformer {
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn transform_chat(&self, _req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex Imagen transformer does not support chat".to_string(),
        ))
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        let provider_opts = extract_vertex_imagen_options(&req.provider_options_map);

        let mut instance = serde_json::Map::new();
        instance.insert("prompt".to_string(), serde_json::json!(req.prompt));
        if let Some(neg) = vertex_imagen_negative_prompt(
            req.negative_prompt.as_ref(),
            &req.extra_params,
            provider_opts.as_ref(),
        ) {
            instance.insert("negativePrompt".to_string(), serde_json::json!(neg));
        }
        if let Some(reference_images) =
            vertex_imagen_reference_images(&req.extra_params, provider_opts.as_ref())
        {
            instance.insert("referenceImages".to_string(), reference_images);
        }

        let mut parameters = serde_json::Map::new();
        if req.count > 0 {
            parameters.insert("sampleCount".to_string(), serde_json::json!(req.count));
        }
        if let Some(seed) = req.seed {
            parameters.insert("seed".to_string(), serde_json::json!(seed));
        }
        if let Some(size) = &req.size
            && let Some(ar) = size_to_aspect_ratio(size)
        {
            parameters.insert("aspectRatio".to_string(), serde_json::json!(ar));
        }

        // Merge provider options (vertexImagen) and extra_params as loose parameters.
        if let Some(opts) = &provider_opts {
            merge_object_skipping(
                &mut parameters,
                opts,
                &[
                    "referenceImages",
                    "reference_images",
                    "negativePrompt",
                    "negative_prompt",
                ],
            );
        }
        for (k, v) in &req.extra_params {
            // Avoid clobbering instance-only keys we already handled.
            if matches!(
                k.as_str(),
                "referenceImages" | "reference_images" | "negativePrompt" | "negative_prompt"
            ) {
                continue;
            }
            parameters.insert(k.clone(), v.clone());
        }

        Ok(serde_json::json!({
            "instances": [serde_json::Value::Object(instance)],
            "parameters": serde_json::Value::Object(parameters),
        }))
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        let provider_opts = extract_vertex_imagen_options(&req.provider_options_map);

        let mut instance = serde_json::Map::new();
        instance.insert("prompt".to_string(), serde_json::json!(req.prompt));
        instance.insert("image".to_string(), bytes_to_inline_image(&req.image));
        if let Some(mask) = &req.mask {
            instance.insert("mask".to_string(), bytes_to_inline_image(mask));
        }
        if let Some(reference_images) =
            vertex_imagen_reference_images(&req.extra_params, provider_opts.as_ref())
        {
            instance.insert("referenceImages".to_string(), reference_images);
        }
        if let Some(neg) =
            vertex_imagen_negative_prompt(None, &req.extra_params, provider_opts.as_ref())
        {
            instance.insert("negativePrompt".to_string(), serde_json::json!(neg));
        }

        let mut parameters = serde_json::Map::new();
        if let Some(n) = req.count {
            parameters.insert("sampleCount".to_string(), serde_json::json!(n));
        }
        if let Some(size) = &req.size
            && let Some(ar) = size_to_aspect_ratio(size)
        {
            parameters.insert("aspectRatio".to_string(), serde_json::json!(ar));
        }

        if let Some(opts) = &provider_opts {
            merge_object_skipping(
                &mut parameters,
                opts,
                &[
                    "referenceImages",
                    "reference_images",
                    "negativePrompt",
                    "negative_prompt",
                ],
            );
        }
        for (k, v) in &req.extra_params {
            if matches!(
                k.as_str(),
                "referenceImages" | "reference_images" | "negativePrompt" | "negative_prompt"
            ) {
                continue;
            }
            parameters.insert(k.clone(), v.clone());
        }

        Ok(ImageHttpBody::Json(serde_json::json!({
            "instances": [serde_json::Value::Object(instance)],
            "parameters": serde_json::Value::Object(parameters),
        })))
    }

    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex Imagen does not implement image variations in this SDK".to_string(),
        ))
    }
}

#[derive(Clone)]
pub struct VertexImagenResponseTransformer {
    provider_id: &'static str,
}

impl VertexImagenResponseTransformer {
    pub fn new(provider_id: &'static str) -> Self {
        Self { provider_id }
    }
}

impl ResponseTransformer for VertexImagenResponseTransformer {
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        let mut images = Vec::new();
        if let Some(predictions) = raw.get("predictions").and_then(|v| v.as_array()) {
            for p in predictions {
                let obj = p.as_object().cloned().unwrap_or_default();

                let bytes = obj
                    .get("bytesBase64Encoded")
                    .cloned()
                    .or_else(|| obj.get("bytes_base64_encoded").cloned())
                    .or_else(|| {
                        obj.get("image")
                            .and_then(|v| v.get("bytesBase64Encoded"))
                            .cloned()
                    });
                let mime = obj
                    .get("mimeType")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        obj.get("image")
                            .and_then(|v| v.get("mimeType"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });

                let mut meta = HashMap::new();
                for (k, v) in obj {
                    if k == "bytesBase64Encoded"
                        || k == "bytes_base64_encoded"
                        || k == "image"
                        || k == "mimeType"
                        || k == "mime_type"
                    {
                        continue;
                    }
                    meta.insert(k, v);
                }

                images.push(crate::types::GeneratedImage {
                    url: None,
                    b64_json: bytes.and_then(|v| v.as_str().map(|s| s.to_string())),
                    format: mime,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: meta,
                });
            }
        }

        let mut metadata = HashMap::new();
        for k in ["deployedModelId", "model", "modelVersionId"] {
            if let Some(v) = raw.get(k) {
                metadata.insert(k.to_string(), v.clone());
            }
        }

        Ok(crate::types::ImageGenerationResponse { images, metadata })
    }
}

#[derive(Clone)]
pub struct VertexImagenStandard;

impl VertexImagenStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> VertexImagenSpec {
        VertexImagenSpec { provider_id }
    }

    pub fn create_transformers(&self, provider_id: &'static str) -> ImageTransformers {
        ImageTransformers {
            request: Arc::new(VertexImagenRequestTransformer::new(provider_id)),
            response: Arc::new(VertexImagenResponseTransformer::new(provider_id)),
        }
    }
}

impl Default for VertexImagenStandard {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VertexImagenSpec {
    provider_id: &'static str,
}

impl ProviderSpec for VertexImagenSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        build_gemini_headers(api_key, &ctx.http_extra_headers)
    }

    fn image_url(&self, req: &ImageGenerationRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn image_edit_url(&self, req: &ImageEditRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn image_variation_url(&self, req: &ImageVariationRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        VertexImagenStandard::new().create_transformers(self.provider_id)
    }
}

/// Heuristic used by the Gemini provider to route image requests to Imagen.
pub fn is_vertex_imagen_model(model: &str, base_url: &str) -> bool {
    if !looks_like_vertex_base_url(base_url) {
        return false;
    }
    let m = normalize_vertex_model_id(model).to_lowercase();
    m.starts_with("imagen")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_normalization_accepts_resource_style() {
        assert_eq!(
            normalize_vertex_model_id("publishers/google/models/imagen-3.0-generate-001"),
            "imagen-3.0-generate-001"
        );
        assert_eq!(
            normalize_vertex_model_id(
                "projects/x/locations/y/publishers/google/models/imagen-3.0-edit-001"
            ),
            "imagen-3.0-edit-001"
        );
    }

    #[test]
    fn url_is_predict() {
        let spec = VertexImagenStandard::new().create_spec("gemini");
        let ctx = ProviderContext::new(
            "gemini",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google".to_string(),
            Some("".to_string()),
            Default::default(),
        );
        let req = ImageGenerationRequest {
            model: Some("imagen-3.0-generate-001".to_string()),
            ..Default::default()
        };
        assert_eq!(
            spec.image_url(&req, &ctx),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/imagen-3.0-generate-001:predict"
        );
    }

    #[test]
    fn transformer_builds_instances_and_parameters() {
        let tx = VertexImagenRequestTransformer::new("gemini");
        let mut req = ImageGenerationRequest::default();
        req.prompt = "a cat".into();
        req.count = 2;
        req.size = Some("1024x1024".into());
        req.negative_prompt = Some("blurry".into());
        req.extra_params.insert("seed".into(), serde_json::json!(7));
        let body = tx.transform_image(&req).unwrap();
        assert_eq!(body["instances"][0]["prompt"], serde_json::json!("a cat"));
        assert_eq!(
            body["instances"][0]["negativePrompt"],
            serde_json::json!("blurry")
        );
        assert_eq!(body["parameters"]["sampleCount"], serde_json::json!(2));
        assert_eq!(body["parameters"]["aspectRatio"], serde_json::json!("1:1"));
        assert_eq!(body["parameters"]["seed"], serde_json::json!(7));
    }

    #[test]
    fn edit_supports_mask_and_reference_images_passthrough() {
        let tx = VertexImagenRequestTransformer::new("gemini");
        let mut req = ImageEditRequest {
            image: vec![137, 80, 78, 71],
            mask: Some(vec![137, 80, 78, 71]),
            prompt: "edit".into(),
            model: Some("imagen-3.0-edit-001".into()),
            count: Some(1),
            size: Some("1024x1024".into()),
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };
        req.extra_params.insert(
            "referenceImages".into(),
            serde_json::json!([{"referenceType":"SUBJECT"}]),
        );

        let ImageHttpBody::Json(body) = tx.transform_image_edit(&req).unwrap() else {
            panic!("expected json body");
        };

        assert!(body["instances"][0].get("image").is_some());
        assert!(body["instances"][0].get("mask").is_some());
        assert_eq!(
            body["instances"][0]["referenceImages"],
            serde_json::json!([{"referenceType":"SUBJECT"}])
        );
    }

    #[test]
    fn response_extracts_base64_images() {
        let tx = VertexImagenResponseTransformer::new("gemini");
        let raw = serde_json::json!({
            "predictions": [
                {"bytesBase64Encoded": "AAA", "mimeType":"image/png"},
                {"image": {"bytesBase64Encoded": "BBB", "mimeType":"image/jpeg"}}
            ],
            "deployedModelId": "d"
        });
        let out = tx.transform_image_response(&raw).unwrap();
        assert_eq!(out.images.len(), 2);
        assert_eq!(out.images[0].b64_json.as_deref(), Some("AAA"));
        assert_eq!(out.images[1].b64_json.as_deref(), Some("BBB"));
        assert_eq!(out.metadata.get("deployedModelId").unwrap(), "d");
    }
}
