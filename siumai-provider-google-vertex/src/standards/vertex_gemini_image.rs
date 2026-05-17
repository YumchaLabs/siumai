//! Vertex AI Gemini image standard (`generateContent` via Vertex endpoints).
//!
//! This module owns the Vertex-specific image runtime for Gemini image models:
//! - `gemini-2.5-flash-image`
//! - `gemini-3-pro-image-preview`
//! - `gemini-3.1-flash-image-preview`
//!
//! It intentionally keeps Imagen on the separate `vertex_imagen` standard.

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{
    ChatMessage, ChatRequest, CommonParams, ContentPart, FilePartSource, ImageEditFileData,
    ImageEditInput, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest, Warning,
};
use reqwest::header::HeaderMap;
use siumai_protocol_gemini::standards::gemini::transformers::{
    GeminiRequestTransformer, GeminiResponseTransformer,
};
use siumai_protocol_gemini::standards::gemini::types::GeminiConfig;
use std::collections::HashMap;
use std::sync::Arc;

const VERTEX_USER_AGENT: &str = concat!("siumai/google-vertex/", env!("CARGO_PKG_VERSION"));
const SIZE_WARNING_DETAILS: &str =
    "This model does not support the `size` option. Use `aspectRatio` instead.";

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

fn has_auth_header(headers: &HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization"))
}

fn append_api_key_query(url: String, api_key: &str) -> String {
    let key = urlencoding::encode(api_key);
    if url.contains('?') {
        format!("{url}&key={key}")
    } else {
        format!("{url}?key={key}")
    }
}

fn build_vertex_headers(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_user_agent(VERTEX_USER_AGENT)?
        .with_custom_headers(custom_headers)?;
    Ok(builder.build())
}

fn size_warning() -> Warning {
    Warning::unsupported("size", Some(SIZE_WARNING_DETAILS))
}

fn warn_for_size(size: &Option<String>) -> Option<Vec<Warning>> {
    size.as_ref().map(|_| vec![size_warning()])
}

fn prompt_from_variation_extra_params(
    extra_params: &HashMap<String, serde_json::Value>,
) -> Option<String> {
    extra_params
        .get("prompt")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn merged_vertex_provider_options(
    provider_options_map: &crate::types::ProviderOptionsMap,
    aspect_ratio: Option<&str>,
) -> crate::types::ProviderOptionsMap {
    let mut merged = provider_options_map.clone();

    let mut vertex_options = provider_options_map
        .get("vertex")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();

    vertex_options.insert(
        "responseModalities".to_string(),
        serde_json::json!(["IMAGE"]),
    );

    if let Some(aspect_ratio) = aspect_ratio {
        let mut image_config = vertex_options
            .get("imageConfig")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();
        image_config.insert(
            "aspectRatio".to_string(),
            serde_json::Value::String(aspect_ratio.to_string()),
        );
        vertex_options.insert(
            "imageConfig".to_string(),
            serde_json::Value::Object(image_config),
        );
    }

    merged.insert("vertex", serde_json::Value::Object(vertex_options));
    merged
}

fn validate_gemini_image_count(count: Option<u32>) -> Result<(), LlmError> {
    if count.is_some_and(|count| count > 1) {
        return Err(LlmError::InvalidParameter(
            "Gemini image models do not support generating a set number of images per call. Use n=1 or omit the n parameter.".to_string(),
        ));
    }

    Ok(())
}

fn request_text_part(text: impl Into<String>) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options: Default::default(),
        provider_metadata: None,
    }
}

fn image_input_media_type(input: &ImageEditInput) -> String {
    if let Some(media_type) = input.media_type() {
        return media_type.to_string();
    }

    if let Some(file_data) = input.file_data()
        && let Ok(bytes) = file_data.as_bytes()
    {
        return crate::utils::guess_mime(Some(bytes.as_slice()), None);
    }

    "image/jpeg".to_string()
}

fn request_file_part(
    source: FilePartSource,
    media_type: impl Into<String>,
    provider_options: crate::types::ProviderOptionsMap,
) -> ContentPart {
    ContentPart::File {
        source,
        media_type: media_type.into(),
        filename: None,
        provider_options,
        provider_metadata: None,
    }
}

fn image_input_part(input: &ImageEditInput) -> Result<ContentPart, LlmError> {
    let provider_options = input.provider_options_map().clone();
    match input {
        ImageEditInput::Url { url, .. } => Ok(request_file_part(
            FilePartSource::url(url.clone()),
            "image/*",
            provider_options,
        )),
        ImageEditInput::File { data, .. } => {
            let source = match data {
                ImageEditFileData::Base64(data) => FilePartSource::base64(data.clone()),
                ImageEditFileData::Binary(data) => FilePartSource::binary(data.clone()),
            };

            Ok(request_file_part(
                source,
                image_input_media_type(input),
                provider_options,
            ))
        }
    }
}

fn build_image_chat_request(
    config: &GeminiConfig,
    prompt: Option<&str>,
    files: &[ImageEditInput],
    seed: Option<u64>,
    aspect_ratio: Option<&str>,
    provider_options_map: &crate::types::ProviderOptionsMap,
) -> Result<ChatRequest, LlmError> {
    let mut parts = Vec::new();

    if let Some(prompt) = prompt
        && !prompt.is_empty()
    {
        parts.push(request_text_part(prompt));
    }

    for file in files {
        parts.push(image_input_part(file)?);
    }

    if parts.is_empty() {
        return Err(LlmError::InvalidParameter(
            "Gemini image requests require a prompt or at least one input image".to_string(),
        ));
    }

    let mut request = ChatRequest::new(vec![
        ChatMessage::user("").with_content_parts(parts).build(),
    ]);
    request.common_params = CommonParams {
        model: config.model.clone(),
        seed,
        ..Default::default()
    };
    request.provider_options_map =
        merged_vertex_provider_options(provider_options_map, aspect_ratio);
    Ok(request)
}

#[derive(Clone)]
pub struct VertexGeminiImageRequestTransformer {
    provider_id: &'static str,
    config: GeminiConfig,
}

impl VertexGeminiImageRequestTransformer {
    pub fn new(provider_id: &'static str, config: GeminiConfig) -> Self {
        Self {
            provider_id,
            config,
        }
    }

    fn gemini_request_transformer(&self) -> GeminiRequestTransformer {
        GeminiRequestTransformer {
            config: self.config.clone(),
        }
    }
}

impl RequestTransformer for VertexGeminiImageRequestTransformer {
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn transform_chat(&self, _req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Vertex Gemini image transformer does not support chat".to_string(),
        ))
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        validate_gemini_image_count((req.count > 0).then_some(req.count))?;

        let synthetic = build_image_chat_request(
            &self.config,
            Some(req.prompt.as_str()),
            &[],
            req.seed,
            req.aspect_ratio.as_deref(),
            &req.provider_options_map,
        )?;

        self.gemini_request_transformer().transform_chat(&synthetic)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        if req.mask.is_some() {
            return Err(LlmError::InvalidParameter(
                "Gemini image models do not support mask-based image editing.".to_string(),
            ));
        }
        validate_gemini_image_count(req.count)?;

        let synthetic = build_image_chat_request(
            &self.config,
            Some(req.prompt.as_str()),
            &req.images,
            req.seed,
            req.aspect_ratio.as_deref(),
            &req.provider_options_map,
        )?;

        Ok(ImageHttpBody::Json(
            self.gemini_request_transformer()
                .transform_chat(&synthetic)?,
        ))
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        validate_gemini_image_count(req.count)?;

        let files = vec![req.image.clone()];
        let prompt = prompt_from_variation_extra_params(&req.extra_params);
        let synthetic = build_image_chat_request(
            &self.config,
            prompt.as_deref(),
            &files,
            req.seed,
            req.aspect_ratio.as_deref(),
            &req.provider_options_map,
        )?;

        Ok(ImageHttpBody::Json(
            self.gemini_request_transformer()
                .transform_chat(&synthetic)?,
        ))
    }
}

#[derive(Clone)]
pub struct VertexGeminiImageResponseTransformer {
    provider_id: &'static str,
    config: GeminiConfig,
}

impl VertexGeminiImageResponseTransformer {
    pub fn new(provider_id: &'static str, config: GeminiConfig) -> Self {
        Self {
            provider_id,
            config,
        }
    }
}

impl ResponseTransformer for VertexGeminiImageResponseTransformer {
    fn provider_id(&self) -> &str {
        self.provider_id
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        GeminiResponseTransformer {
            config: self.config.clone(),
        }
        .transform_image_response(raw)
    }
}

#[derive(Clone, Default)]
pub struct VertexGeminiImageStandard {
    gemini_config: Option<GeminiConfig>,
}

impl VertexGeminiImageStandard {
    pub fn new() -> Self {
        Self {
            gemini_config: None,
        }
    }

    pub fn with_gemini_config(mut self, gemini_config: GeminiConfig) -> Self {
        self.gemini_config = Some(gemini_config);
        self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> VertexGeminiImageSpec {
        VertexGeminiImageSpec {
            provider_id,
            gemini_config: self.gemini_config.clone(),
        }
    }
}

pub struct VertexGeminiImageSpec {
    provider_id: &'static str,
    gemini_config: Option<GeminiConfig>,
}

impl VertexGeminiImageSpec {
    fn config_for_request(&self, req: &ImageGenerationRequest) -> GeminiConfig {
        let mut config = self.gemini_config.clone().unwrap_or_default();
        let model = normalize_vertex_model_id(req.model.as_deref().unwrap_or(""));
        if !model.is_empty() {
            config.model = model.clone();
            config.common_params.model = model;
        }
        if config.provider_metadata_key.is_none() {
            config.provider_metadata_key = Some("vertex".to_string());
        }
        config
    }

    fn image_request_url(&self, model: &str, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let url = format!("{base}/models/{model}:generateContent");
        if let Some(key) = ctx.api_key.as_deref()
            && !key.is_empty()
            && !has_auth_header(&ctx.http_extra_headers)
        {
            append_api_key_query(url, key)
        } else {
            url
        }
    }
}

impl ProviderSpec for VertexGeminiImageSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_vertex_headers(&ctx.http_extra_headers)
    }

    fn try_image_url(
        &self,
        req: &ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(self.image_request_url(
            &normalize_vertex_model_id(req.model.as_deref().unwrap_or("")),
            ctx,
        ))
    }

    fn try_image_edit_url(
        &self,
        req: &ImageEditRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(self.image_request_url(
            &normalize_vertex_model_id(req.model.as_deref().unwrap_or("")),
            ctx,
        ))
    }

    fn try_image_variation_url(
        &self,
        req: &ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(self.image_request_url(
            &normalize_vertex_model_id(req.model.as_deref().unwrap_or("")),
            ctx,
        ))
    }

    fn image_warnings(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        warn_for_size(&req.size)
    }

    fn image_edit_warnings(
        &self,
        req: &ImageEditRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        warn_for_size(&req.size)
    }

    fn materialize_image_edit_urls(&self, _req: &ImageEditRequest, _ctx: &ProviderContext) -> bool {
        false
    }

    fn image_variation_warnings(
        &self,
        req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        warn_for_size(&req.size)
    }

    fn materialize_image_variation_urls(
        &self,
        _req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> bool {
        false
    }

    fn choose_image_transformers(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        let config = self.config_for_request(req);
        ImageTransformers {
            request: Arc::new(VertexGeminiImageRequestTransformer::new(
                self.provider_id,
                config.clone(),
            )),
            response: Arc::new(VertexGeminiImageResponseTransformer::new(
                self.provider_id,
                config,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageContent;
    use serde_json::json;

    #[test]
    fn vertex_gemini_image_request_content_construction_is_centralized() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/standards/vertex_gemini_image.rs"
        ));
        let helper_start = source
            .find("fn request_text_part")
            .expect("request content adapter helpers should exist");
        let helper_end = source
            .find("fn image_input_part")
            .expect("request content adapter helpers should precede image input conversion");
        assert!(helper_start < helper_end);

        let mut provider_metadata_writes = Vec::new();
        let mut direct_text_constructors = Vec::new();
        let mut offset = 0usize;
        for (index, line) in source.lines().enumerate() {
            let line_start = offset;
            offset += line.len() + 1;
            if line.trim() == "provider_metadata: None,"
                && !(helper_start..helper_end).contains(&line_start)
            {
                provider_metadata_writes.push(index + 1);
            }
            if line.contains("ContentPart::text(")
                && !line.contains("line.contains(")
                && !(helper_start..helper_end).contains(&line_start)
            {
                direct_text_constructors.push(index + 1);
            }
        }

        assert!(
            provider_metadata_writes.is_empty(),
            "Vertex Gemini image request conversion should only write legacy provider_metadata inside request content adapters: {provider_metadata_writes:?}"
        );
        assert!(
            direct_text_constructors.is_empty(),
            "Vertex Gemini image request conversion should route text prompts through request_text_part: {direct_text_constructors:?}"
        );
    }

    #[test]
    fn image_input_part_maps_provider_options_without_provider_metadata() {
        let input = ImageEditInput::url("https://example.com/input.png")
            .with_provider_option("vertex", json!({ "asset": "input" }));

        let part = image_input_part(&input).expect("image input part");
        let ContentPart::File {
            media_type,
            provider_options,
            provider_metadata,
            ..
        } = part
        else {
            panic!("expected file part");
        };

        assert_eq!(media_type, "image/*");
        assert_eq!(
            provider_options
                .get_object("vertex")
                .and_then(|vertex| vertex.get("asset")),
            Some(&json!("input"))
        );
        assert!(provider_metadata.is_none());
    }

    #[test]
    fn build_image_chat_request_routes_prompt_through_request_text_adapter() {
        let config = GeminiConfig::new("test-key").with_model("gemini-2.5-flash-image".to_string());
        let request = build_image_chat_request(
            &config,
            Some("Draw a blue square"),
            &[],
            None,
            None,
            &Default::default(),
        )
        .expect("image chat request");

        let MessageContent::MultiModal(parts) = &request.messages[0].content else {
            panic!("expected multimodal request content");
        };
        let ContentPart::Text {
            text,
            provider_options,
            provider_metadata,
        } = &parts[0]
        else {
            panic!("expected prompt text part");
        };

        assert_eq!(text, "Draw a blue square");
        assert!(provider_options.is_empty());
        assert!(provider_metadata.is_none());
    }
}
