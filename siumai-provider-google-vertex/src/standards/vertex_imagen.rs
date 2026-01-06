//! Vertex AI Imagen standard (Google publisher).
//!
//! This module provides a minimal request/response mapping for Vertex AI's Imagen
//! models using the `:predict` endpoint. It is intentionally scoped to image
//! generation/editing and does not implement chat/embeddings.

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::Warning;
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

fn build_vertex_headers(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_custom_headers(custom_headers)?;
    Ok(builder.build())
}

fn bytes_to_inline_image(bytes: &[u8]) -> serde_json::Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    serde_json::json!({
        "bytesBase64Encoded": b64,
    })
}

fn extract_vertex_imagen_options(
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Value> {
    // Vercel-aligned key: providerOptions["vertex"]
    if let Some(v) = map.get("vertex") {
        return Some(v.clone());
    }

    // Backward compatibility: providerOptions["gemini"]["vertex"] (legacy nesting).
    let v = map.get("gemini")?;
    let obj = v.as_object()?;
    obj.get("vertex")
        .cloned()
        .or_else(|| obj.get("vertexImagen").cloned())
        .or_else(|| obj.get("vertex_imagen").cloned())
}

const VERTEX_IMAGEN_PROVIDER_OPTIONS_ALLOWLIST: &[&str] = &[
    "negativePrompt",
    "personGeneration",
    "safetySetting",
    "addWatermark",
    "storageUri",
    "sampleImageSize",
];

fn merge_object_allowlist_skipping(
    into: &mut serde_json::Map<String, serde_json::Value>,
    value: &serde_json::Value,
    allowlist: &[&str],
    skip: &[&str],
) {
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            if skip.iter().any(|s| *s == k) {
                continue;
            }
            if !allowlist.iter().any(|a| *a == k) {
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

fn vertex_imagen_edit_options(
    provider_opts: Option<&serde_json::Value>,
) -> Option<&serde_json::Value> {
    let opts = provider_opts?;
    let obj = opts.as_object()?;
    obj.get("edit")
}

fn vertex_imagen_aspect_ratio(
    extra_params: &HashMap<String, serde_json::Value>,
    provider_opts: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(v) = extra_params.get("aspectRatio").and_then(|v| v.as_str()) {
        return Some(v.to_string());
    }
    if let Some(v) = extra_params.get("aspect_ratio").and_then(|v| v.as_str()) {
        return Some(v.to_string());
    }
    if let Some(opts) = provider_opts
        && let Some(obj) = opts.as_object()
    {
        if let Some(v) = obj.get("aspectRatio").and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
        if let Some(v) = obj.get("aspect_ratio").and_then(|v| v.as_str()) {
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
        if let Some(ar) = vertex_imagen_aspect_ratio(&req.extra_params, provider_opts.as_ref()) {
            parameters.insert("aspectRatio".to_string(), serde_json::json!(ar));
        }
        if let Some(neg) = vertex_imagen_negative_prompt(
            req.negative_prompt.as_ref(),
            &req.extra_params,
            provider_opts.as_ref(),
        ) {
            parameters.insert("negativePrompt".to_string(), serde_json::json!(neg));
        }

        if let Some(opts) = &provider_opts {
            merge_object_allowlist_skipping(
                &mut parameters,
                opts,
                VERTEX_IMAGEN_PROVIDER_OPTIONS_ALLOWLIST,
                &[
                    "edit",
                    "referenceImages",
                    "reference_images",
                    "negativePrompt",
                    "negative_prompt",
                    "aspectRatio",
                    "aspect_ratio",
                ],
            );
        }
        for (k, v) in &req.extra_params {
            if matches!(
                k.as_str(),
                "edit"
                    | "referenceImages"
                    | "reference_images"
                    | "negativePrompt"
                    | "negative_prompt"
                    | "aspectRatio"
                    | "aspect_ratio"
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

        let mut reference_images = Vec::new();
        reference_images.push(serde_json::json!({
            "referenceId": 1,
            "referenceType": "REFERENCE_TYPE_RAW",
            "referenceImage": bytes_to_inline_image(&req.image),
        }));

        if let Some(mask) = &req.mask {
            let mut mask_obj = serde_json::json!({
                "referenceId": 2,
                "referenceType": "REFERENCE_TYPE_MASK",
                "referenceImage": bytes_to_inline_image(mask),
                "maskImageConfig": { "maskMode": "MASK_MODE_USER_PROVIDED" }
            });

            if let Some(edit) = vertex_imagen_edit_options(provider_opts.as_ref())
                && let Some(edit_obj) = edit.as_object()
            {
                if let Some(mode) = edit_obj.get("maskMode").and_then(|v| v.as_str()) {
                    mask_obj["maskImageConfig"]["maskMode"] = serde_json::json!(mode);
                }
                if let Some(dilation) = edit_obj.get("maskDilation").and_then(|v| v.as_f64()) {
                    mask_obj["maskImageConfig"]["dilation"] = serde_json::json!(dilation);
                }
            }

            reference_images.push(mask_obj);
        }

        if let Some(extra_ref) =
            vertex_imagen_reference_images(&req.extra_params, provider_opts.as_ref())
        {
            if let Some(arr) = extra_ref.as_array() {
                reference_images.extend(arr.iter().cloned());
            } else {
                reference_images.push(extra_ref);
            }
        }

        if !reference_images.is_empty() {
            instance.insert(
                "referenceImages".to_string(),
                serde_json::Value::Array(reference_images),
            );
        }

        let mut parameters = serde_json::Map::new();
        if let Some(n) = req.count {
            parameters.insert("sampleCount".to_string(), serde_json::json!(n));
        }
        if let Some(ar) = vertex_imagen_aspect_ratio(&req.extra_params, provider_opts.as_ref()) {
            parameters.insert("aspectRatio".to_string(), serde_json::json!(ar));
        }

        if req.mask.is_some() {
            parameters.insert(
                "editMode".to_string(),
                serde_json::json!("EDIT_MODE_INPAINT_INSERTION"),
            );
        }

        if let Some(edit) = vertex_imagen_edit_options(provider_opts.as_ref())
            && let Some(edit_obj) = edit.as_object()
        {
            if let Some(mode) = edit_obj.get("mode").and_then(|v| v.as_str()) {
                parameters.insert("editMode".to_string(), serde_json::json!(mode));
            }
            if let Some(base_steps) = edit_obj.get("baseSteps").and_then(|v| v.as_i64()) {
                parameters
                    .entry("editConfig".to_string())
                    .or_insert_with(|| serde_json::json!({}))
                    .as_object_mut()
                    .expect("object inserted above")
                    .insert("baseSteps".to_string(), serde_json::json!(base_steps));
            }
        }

        if let Some(neg) =
            vertex_imagen_negative_prompt(None, &req.extra_params, provider_opts.as_ref())
        {
            parameters.insert("negativePrompt".to_string(), serde_json::json!(neg));
        }

        if let Some(opts) = &provider_opts {
            merge_object_allowlist_skipping(
                &mut parameters,
                opts,
                VERTEX_IMAGEN_PROVIDER_OPTIONS_ALLOWLIST,
                &[
                    "edit",
                    "referenceImages",
                    "reference_images",
                    "negativePrompt",
                    "negative_prompt",
                    "aspectRatio",
                    "aspect_ratio",
                ],
            );
        }
        for (k, v) in &req.extra_params {
            if matches!(
                k.as_str(),
                "edit"
                    | "referenceImages"
                    | "reference_images"
                    | "negativePrompt"
                    | "negative_prompt"
                    | "aspectRatio"
                    | "aspect_ratio"
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
        let preds = raw
            .get("predictions")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut images = Vec::with_capacity(preds.len());
        for pred in preds {
            if let Some(obj) = pred.as_object() {
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
                        obj.get("mime_type")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .or_else(|| {
                        obj.get("image")
                            .and_then(|v| v.get("mimeType"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });
                let revised_prompt = obj
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let mut meta = HashMap::new();
                for (k, v) in obj {
                    if k == "bytesBase64Encoded"
                        || k == "bytes_base64_encoded"
                        || k == "image"
                        || k == "mimeType"
                        || k == "mime_type"
                        || k == "prompt"
                    {
                        continue;
                    }
                    meta.insert(k.clone(), v.clone());
                }

                images.push(crate::types::GeneratedImage {
                    url: None,
                    b64_json: bytes.and_then(|v| v.as_str().map(|s| s.to_string())),
                    format: mime,
                    width: None,
                    height: None,
                    revised_prompt,
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

        Ok(crate::types::ImageGenerationResponse {
            images,
            metadata,
            warnings: None,
            response: None,
        })
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
        build_vertex_headers(&ctx.http_extra_headers)
    }

    fn image_url(&self, req: &ImageGenerationRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn image_warnings(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        if req.size.is_some() {
            return Some(vec![Warning::unsupported_setting(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead."),
            )]);
        }
        None
    }

    fn image_edit_url(&self, req: &ImageEditRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn image_edit_warnings(
        &self,
        req: &ImageEditRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        if req.size.is_some() {
            return Some(vec![Warning::unsupported_setting(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead."),
            )]);
        }
        None
    }

    fn image_variation_url(&self, req: &ImageVariationRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        format!("{}/models/{}:predict", base, model)
    }

    fn image_variation_warnings(
        &self,
        req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        if req.size.is_some() {
            return Some(vec![Warning::unsupported_setting(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead."),
            )]);
        }
        None
    }

    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        VertexImagenStandard::new().create_transformers(self.provider_id)
    }
}

/// Heuristic used by clients/registry to detect Imagen models.
pub fn is_vertex_imagen_model(model: &str, base_url: &str) -> bool {
    if !looks_like_vertex_base_url(base_url) {
        return false;
    }
    let m = normalize_vertex_model_id(model).to_lowercase();
    m.starts_with("imagen")
}
