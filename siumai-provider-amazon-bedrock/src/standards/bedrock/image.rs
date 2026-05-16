//! Amazon Bedrock Image Standard (AI SDK-aligned).

use crate::core::{ImageTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{
    GeneratedImage, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, Warning,
};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

const NOVA_CANVAS_MODEL_ID: &str = "amazon.nova-canvas-v1:0";

pub fn bedrock_image_max_images_per_call(model_id: &str) -> u32 {
    match model_id.trim() {
        NOVA_CANVAS_MODEL_ID => 5,
        _ => 1,
    }
}

#[derive(Clone, Default)]
pub struct BedrockImageStandard;

impl BedrockImageStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> ImageTransformers {
        ImageTransformers {
            request: Arc::new(BedrockImageRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(BedrockImageResponseTransformer {
                provider_id: provider_id.to_string(),
            }),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> BedrockImageSpec {
        BedrockImageSpec { provider_id }
    }
}

pub struct BedrockImageSpec {
    provider_id: &'static str,
}

impl ProviderSpec for BedrockImageSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::bedrock::headers::build_bedrock_json_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::bedrock::errors::classify_bedrock_http_error(
            self.provider_id,
            status,
            body_text,
            headers,
        )
    }

    fn try_image_url(
        &self,
        req: &ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(build_bedrock_invoke_url(
            ctx.base_url.as_str(),
            req.model.as_deref().unwrap_or(""),
        ))
    }

    fn image_warnings(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        generation_request_warnings(req)
    }

    fn try_image_edit_url(
        &self,
        req: &ImageEditRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(build_bedrock_invoke_url(
            ctx.base_url.as_str(),
            req.model.as_deref().unwrap_or(""),
        ))
    }

    fn image_edit_warnings(
        &self,
        req: &ImageEditRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        edit_request_warnings(req)
    }

    fn try_image_variation_url(
        &self,
        req: &ImageVariationRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        Ok(build_bedrock_invoke_url(
            ctx.base_url.as_str(),
            req.model.as_deref().unwrap_or(""),
        ))
    }

    fn image_variation_warnings(
        &self,
        req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        variation_request_warnings(req)
    }

    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        BedrockImageStandard::new().create_transformers(&ctx.provider_id)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct BedrockImageProviderOptions {
    quality: Option<String>,
    #[serde(rename = "cfgScale", alias = "cfg_scale")]
    cfg_scale: Option<f32>,
    #[serde(rename = "negativeText", alias = "negative_text")]
    negative_text: Option<String>,
    style: Option<String>,
    #[serde(rename = "maskPrompt", alias = "mask_prompt")]
    mask_prompt: Option<String>,
    #[serde(rename = "taskType", alias = "task_type")]
    task_type: Option<String>,
    #[serde(rename = "outPaintingMode", alias = "out_painting_mode")]
    out_painting_mode: Option<String>,
    #[serde(rename = "similarityStrength", alias = "similarity_strength")]
    similarity_strength: Option<f32>,
}

#[derive(Debug, Clone, Default)]
struct ResolvedBedrockImageOptions {
    quality: Option<String>,
    cfg_scale: Option<f32>,
    negative_text: Option<String>,
    style: Option<String>,
    mask_prompt: Option<String>,
    task_type: Option<String>,
    out_painting_mode: Option<String>,
    similarity_strength: Option<f32>,
}

struct BedrockImageRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

struct BedrockImageResponseTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BedrockImageResponse {
    #[serde(default)]
    images: Option<Vec<String>>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    progress: Option<serde_json::Value>,
    #[serde(default)]
    details: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    preview: Option<serde_json::Value>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

impl ResolvedBedrockImageOptions {
    fn from_generation_request(req: &ImageGenerationRequest) -> Result<Self, LlmError> {
        let provider_options = parse_bedrock_image_provider_options(&req.provider_options_map)?;
        Ok(Self {
            quality: provider_options
                .quality
                .or_else(|| req.quality.clone())
                .or(extra_string(&req.extra_params, "quality", "quality")?),
            cfg_scale: provider_options
                .cfg_scale
                .or(req.guidance_scale)
                .or(extra_f32(
                    &req.extra_params,
                    "guidance_scale",
                    "guidanceScale",
                )?),
            negative_text: provider_options
                .negative_text
                .or_else(|| req.negative_prompt.clone())
                .or(extra_string(
                    &req.extra_params,
                    "negative_prompt",
                    "negativePrompt",
                )?),
            style: provider_options
                .style
                .or_else(|| req.style.clone())
                .or(extra_string(&req.extra_params, "style", "style")?),
            mask_prompt: provider_options.mask_prompt.or(extra_string(
                &req.extra_params,
                "mask_prompt",
                "maskPrompt",
            )?),
            task_type: provider_options.task_type.or(extra_string(
                &req.extra_params,
                "task_type",
                "taskType",
            )?),
            out_painting_mode: provider_options.out_painting_mode.or(extra_string(
                &req.extra_params,
                "out_painting_mode",
                "outPaintingMode",
            )?),
            similarity_strength: provider_options.similarity_strength.or(extra_f32(
                &req.extra_params,
                "similarity_strength",
                "similarityStrength",
            )?),
        })
    }

    fn from_edit_request(req: &ImageEditRequest) -> Result<Self, LlmError> {
        let provider_options = parse_bedrock_image_provider_options(&req.provider_options_map)?;
        Ok(Self {
            quality: provider_options.quality.or(extra_string(
                &req.extra_params,
                "quality",
                "quality",
            )?),
            cfg_scale: provider_options.cfg_scale.or(extra_f32(
                &req.extra_params,
                "guidance_scale",
                "guidanceScale",
            )?),
            negative_text: provider_options.negative_text.or(extra_string(
                &req.extra_params,
                "negative_prompt",
                "negativePrompt",
            )?),
            style: provider_options
                .style
                .or(extra_string(&req.extra_params, "style", "style")?),
            mask_prompt: provider_options.mask_prompt.or(extra_string(
                &req.extra_params,
                "mask_prompt",
                "maskPrompt",
            )?),
            task_type: provider_options.task_type.or(extra_string(
                &req.extra_params,
                "task_type",
                "taskType",
            )?),
            out_painting_mode: provider_options.out_painting_mode.or(extra_string(
                &req.extra_params,
                "out_painting_mode",
                "outPaintingMode",
            )?),
            similarity_strength: provider_options.similarity_strength.or(extra_f32(
                &req.extra_params,
                "similarity_strength",
                "similarityStrength",
            )?),
        })
    }

    fn from_variation_request(req: &ImageVariationRequest) -> Result<Self, LlmError> {
        let provider_options = parse_bedrock_image_provider_options(&req.provider_options_map)?;
        Ok(Self {
            quality: provider_options.quality.or(extra_string(
                &req.extra_params,
                "quality",
                "quality",
            )?),
            cfg_scale: provider_options.cfg_scale.or(extra_f32(
                &req.extra_params,
                "guidance_scale",
                "guidanceScale",
            )?),
            negative_text: provider_options.negative_text.or(extra_string(
                &req.extra_params,
                "negative_prompt",
                "negativePrompt",
            )?),
            style: provider_options
                .style
                .or(extra_string(&req.extra_params, "style", "style")?),
            mask_prompt: provider_options.mask_prompt.or(extra_string(
                &req.extra_params,
                "mask_prompt",
                "maskPrompt",
            )?),
            task_type: provider_options.task_type.or(extra_string(
                &req.extra_params,
                "task_type",
                "taskType",
            )?),
            out_painting_mode: provider_options.out_painting_mode.or(extra_string(
                &req.extra_params,
                "out_painting_mode",
                "outPaintingMode",
            )?),
            similarity_strength: provider_options.similarity_strength.or(extra_f32(
                &req.extra_params,
                "similarity_strength",
                "similarityStrength",
            )?),
        })
    }
}

impl RequestTransformer for BedrockImageRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement chat transformer on the Bedrock image standard",
            self.provider_id
        )))
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        let options = ResolvedBedrockImageOptions::from_generation_request(req)?;
        build_text_image_body(req, &options)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        let options = ResolvedBedrockImageOptions::from_edit_request(req)?;
        Ok(ImageHttpBody::Json(build_edit_image_body(req, &options)?))
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        let options = ResolvedBedrockImageOptions::from_variation_request(req)?;
        Ok(ImageHttpBody::Json(build_variation_image_body(
            req, &options,
        )?))
    }
}

impl ResponseTransformer for BedrockImageResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut response: BedrockImageResponse =
            serde_json::from_value(raw.clone()).map_err(|error| {
                LlmError::ParseError(format!("Failed to parse Bedrock image response: {error}"))
            })?;

        if response.status.as_deref() == Some("Request Moderated") {
            let reasons = response
                .details
                .as_ref()
                .and_then(|details| details.get("Moderation Reasons"))
                .and_then(|value| value.as_array())
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .filter(|reasons| !reasons.is_empty())
                .unwrap_or_else(|| vec!["Unknown".to_string()]);

            return Err(LlmError::ApiError {
                code: 400,
                message: format!(
                    "Amazon Bedrock request was moderated: {}",
                    reasons.join(", ")
                ),
                details: Some(raw.clone()),
            });
        }

        let images = response.images.take().ok_or_else(|| {
            LlmError::ParseError(
                "Amazon Bedrock image response did not include `images`".to_string(),
            )
        })?;
        if images.is_empty() {
            return Err(LlmError::ParseError(
                "Amazon Bedrock image response included an empty `images` array".to_string(),
            ));
        }

        Ok(ImageGenerationResponse {
            images: images
                .into_iter()
                .map(|image| GeneratedImage {
                    url: None,
                    b64_json: Some(image),
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: HashMap::new(),
                })
                .collect(),
            metadata: response.into_metadata(),
            warnings: None,
            response: Some(crate::types::HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: None,
                headers: HashMap::new(),
                body: None,
            }),
        })
    }
}

impl BedrockImageResponse {
    fn into_metadata(self) -> HashMap<String, serde_json::Value> {
        let mut metadata = self.extra_fields;

        if let Some(id) = self.id {
            metadata.insert("id".to_string(), serde_json::json!(id));
        }
        if let Some(status) = self.status {
            metadata.insert("status".to_string(), serde_json::json!(status));
        }
        if let Some(result) = self.result {
            metadata.insert("result".to_string(), result);
        }
        if let Some(progress) = self.progress {
            metadata.insert("progress".to_string(), progress);
        }
        if let Some(preview) = self.preview {
            metadata.insert("preview".to_string(), preview);
        }
        if let Some(details) = self.details {
            metadata.insert(
                "details".to_string(),
                serde_json::Value::Object(details.into_iter().collect()),
            );
        }

        metadata
    }
}

fn build_bedrock_invoke_url(base_url: &str, model_id: &str) -> String {
    crate::utils::url::join_url(
        base_url,
        &format!("/model/{}/invoke", urlencoding::encode(model_id)),
    )
}

fn parse_bedrock_image_provider_options(
    map: &crate::types::ProviderOptionsMap,
) -> Result<BedrockImageProviderOptions, LlmError> {
    let Some(value) = map.get("bedrock") else {
        return Ok(BedrockImageProviderOptions::default());
    };

    serde_json::from_value(value.clone()).map_err(|error| {
        LlmError::InvalidParameter(format!(
            "providerOptions.bedrock is invalid for Amazon Bedrock image requests: {error}"
        ))
    })
}

fn extra_string(
    extra_params: &HashMap<String, serde_json::Value>,
    primary: &str,
    alias: &str,
) -> Result<Option<String>, LlmError> {
    let value = extra_params
        .get(primary)
        .or_else(|| extra_params.get(alias));
    let Some(value) = value else {
        return Ok(None);
    };

    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| {
            LlmError::InvalidParameter(format!(
                "Amazon Bedrock image extra param `{primary}` must be a string when provided"
            ))
        })
}

fn extra_f32(
    extra_params: &HashMap<String, serde_json::Value>,
    primary: &str,
    alias: &str,
) -> Result<Option<f32>, LlmError> {
    let value = extra_params
        .get(primary)
        .or_else(|| extra_params.get(alias));
    let Some(value) = value else {
        return Ok(None);
    };

    value
        .as_f64()
        .map(|value| Some(value as f32))
        .ok_or_else(|| {
            LlmError::InvalidParameter(format!(
                "Amazon Bedrock image extra param `{primary}` must be a number when provided"
            ))
        })
}

fn parse_size(size: Option<&str>) -> Result<(Option<u32>, Option<u32>), LlmError> {
    let Some(size) = size.map(str::trim).filter(|size| !size.is_empty()) else {
        return Ok((None, None));
    };

    let (width, height) = size.split_once('x').ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "Amazon Bedrock image size must be `WIDTHxHEIGHT`, got `{size}`"
        ))
    })?;

    let width = width.parse::<u32>().map_err(|error| {
        LlmError::InvalidParameter(format!(
            "Amazon Bedrock image size width is invalid in `{size}`: {error}"
        ))
    })?;
    let height = height.parse::<u32>().map_err(|error| {
        LlmError::InvalidParameter(format!(
            "Amazon Bedrock image size height is invalid in `{size}`: {error}"
        ))
    })?;

    Ok((Some(width), Some(height)))
}

fn image_generation_config(
    size: Option<&str>,
    seed: Option<u64>,
    count: u32,
    options: &ResolvedBedrockImageOptions,
) -> Result<serde_json::Map<String, serde_json::Value>, LlmError> {
    let (width, height) = parse_size(size)?;
    let mut config = serde_json::Map::new();

    if let Some(width) = width {
        config.insert("width".to_string(), serde_json::json!(width));
    }
    if let Some(height) = height {
        config.insert("height".to_string(), serde_json::json!(height));
    }
    if let Some(seed) = seed {
        config.insert("seed".to_string(), serde_json::json!(seed));
    }
    if count > 0 {
        config.insert("numberOfImages".to_string(), serde_json::json!(count));
    }
    if let Some(quality) = options.quality.as_ref() {
        config.insert("quality".to_string(), serde_json::json!(quality));
    }
    if let Some(cfg_scale) = options.cfg_scale {
        config.insert("cfgScale".to_string(), serde_json::json!(cfg_scale));
    }

    Ok(config)
}

fn image_input_to_base64(input: &ImageEditInput, label: &str) -> Result<String, LlmError> {
    match input {
        ImageEditInput::Url { .. } => Err(LlmError::InvalidParameter(format!(
            "Amazon Bedrock image editing does not support URL-backed `{label}` inputs; provide the image bytes directly"
        ))),
        ImageEditInput::File { data, .. } => Ok(data.as_base64()),
    }
}

fn build_text_image_body(
    req: &ImageGenerationRequest,
    options: &ResolvedBedrockImageOptions,
) -> Result<serde_json::Value, LlmError> {
    let mut text_to_image_params = serde_json::Map::new();
    text_to_image_params.insert("text".to_string(), serde_json::json!(req.prompt));
    if let Some(negative_text) = options.negative_text.as_ref() {
        text_to_image_params.insert("negativeText".to_string(), serde_json::json!(negative_text));
    }
    if let Some(style) = options.style.as_ref() {
        text_to_image_params.insert("style".to_string(), serde_json::json!(style));
    }

    Ok(serde_json::json!({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": serde_json::Value::Object(text_to_image_params),
        "imageGenerationConfig": serde_json::Value::Object(image_generation_config(
            req.size.as_deref(),
            req.seed,
            req.count.max(1),
            options,
        )?),
    }))
}

fn build_edit_image_body(
    req: &ImageEditRequest,
    options: &ResolvedBedrockImageOptions,
) -> Result<serde_json::Value, LlmError> {
    if req.images.is_empty() {
        return Err(LlmError::InvalidParameter(
            "Amazon Bedrock image editing requires at least one source image".to_string(),
        ));
    }

    let task_type = options.task_type.as_deref().unwrap_or_else(|| {
        if req.mask.is_some() || options.mask_prompt.is_some() {
            "INPAINTING"
        } else {
            "IMAGE_VARIATION"
        }
    });

    build_file_backed_image_body(
        task_type,
        &req.images,
        req.mask.as_ref(),
        (!req.prompt.trim().is_empty()).then_some(req.prompt.as_str()),
        req.size.as_deref(),
        req.seed,
        req.count.unwrap_or(1).max(1),
        options,
    )
}

fn build_variation_image_body(
    req: &ImageVariationRequest,
    options: &ResolvedBedrockImageOptions,
) -> Result<serde_json::Value, LlmError> {
    build_file_backed_image_body(
        options.task_type.as_deref().unwrap_or("IMAGE_VARIATION"),
        std::slice::from_ref(&req.image),
        None,
        None,
        req.size.as_deref(),
        req.seed,
        req.count.unwrap_or(1).max(1),
        options,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_file_backed_image_body(
    task_type: &str,
    images: &[ImageEditInput],
    mask: Option<&ImageEditInput>,
    prompt: Option<&str>,
    size: Option<&str>,
    seed: Option<u64>,
    count: u32,
    options: &ResolvedBedrockImageOptions,
) -> Result<serde_json::Value, LlmError> {
    let image_generation_config = image_generation_config(size, seed, count, options)?;
    let source_image = image_input_to_base64(&images[0], "image")?;

    let body = match task_type {
        "INPAINTING" => {
            let mut params = serde_json::Map::new();
            params.insert("image".to_string(), serde_json::json!(source_image));
            if let Some(prompt) = prompt {
                params.insert("text".to_string(), serde_json::json!(prompt));
            }
            if let Some(negative_text) = options.negative_text.as_ref() {
                params.insert("negativeText".to_string(), serde_json::json!(negative_text));
            }
            if let Some(mask) = mask {
                params.insert(
                    "maskImage".to_string(),
                    serde_json::json!(image_input_to_base64(mask, "mask")?),
                );
            } else if let Some(mask_prompt) = options.mask_prompt.as_ref() {
                params.insert("maskPrompt".to_string(), serde_json::json!(mask_prompt));
            }

            serde_json::json!({
                "taskType": "INPAINTING",
                "inPaintingParams": serde_json::Value::Object(params),
                "imageGenerationConfig": serde_json::Value::Object(image_generation_config),
            })
        }
        "OUTPAINTING" => {
            let mut params = serde_json::Map::new();
            params.insert("image".to_string(), serde_json::json!(source_image));
            if let Some(prompt) = prompt {
                params.insert("text".to_string(), serde_json::json!(prompt));
            }
            if let Some(negative_text) = options.negative_text.as_ref() {
                params.insert("negativeText".to_string(), serde_json::json!(negative_text));
            }
            if let Some(out_painting_mode) = options.out_painting_mode.as_ref() {
                params.insert(
                    "outPaintingMode".to_string(),
                    serde_json::json!(out_painting_mode),
                );
            }
            if let Some(mask) = mask {
                params.insert(
                    "maskImage".to_string(),
                    serde_json::json!(image_input_to_base64(mask, "mask")?),
                );
            } else if let Some(mask_prompt) = options.mask_prompt.as_ref() {
                params.insert("maskPrompt".to_string(), serde_json::json!(mask_prompt));
            }

            serde_json::json!({
                "taskType": "OUTPAINTING",
                "outPaintingParams": serde_json::Value::Object(params),
                "imageGenerationConfig": serde_json::Value::Object(image_generation_config),
            })
        }
        "BACKGROUND_REMOVAL" => serde_json::json!({
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {
                "image": source_image
            }
        }),
        "IMAGE_VARIATION" => {
            let images = images
                .iter()
                .map(|image| image_input_to_base64(image, "image"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut params = serde_json::Map::new();
            params.insert("images".to_string(), serde_json::json!(images));
            if let Some(prompt) = prompt {
                params.insert("text".to_string(), serde_json::json!(prompt));
            }
            if let Some(negative_text) = options.negative_text.as_ref() {
                params.insert("negativeText".to_string(), serde_json::json!(negative_text));
            }
            if let Some(similarity_strength) = options.similarity_strength {
                params.insert(
                    "similarityStrength".to_string(),
                    serde_json::json!(similarity_strength),
                );
            }

            serde_json::json!({
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": serde_json::Value::Object(params),
                "imageGenerationConfig": serde_json::Value::Object(image_generation_config),
            })
        }
        other => {
            return Err(LlmError::InvalidParameter(format!(
                "Unsupported Amazon Bedrock image taskType `{other}`"
            )));
        }
    };

    Ok(body)
}

fn push_response_format_warning(warnings: &mut Vec<Warning>, response_format: Option<&str>) {
    if response_format.is_some_and(|format| !format.eq_ignore_ascii_case("b64_json")) {
        warnings.push(Warning::unsupported_setting(
            "response_format",
            Some(
                "Amazon Bedrock image responses on this path always return base64 image payloads.",
            ),
        ));
    }
}

fn push_generation_only_extra_warnings(
    warnings: &mut Vec<Warning>,
    extra_params: &HashMap<String, serde_json::Value>,
) {
    if extra_params.contains_key("steps") {
        warnings.push(Warning::unsupported_setting(
            "steps",
            Some("Amazon Bedrock does not expose the generic `steps` option on this image path."),
        ));
    }
    if extra_params.contains_key("enhance_prompt") || extra_params.contains_key("enhancePrompt") {
        warnings.push(Warning::unsupported_setting(
            "enhance_prompt",
            Some(
                "Amazon Bedrock does not expose the generic `enhance_prompt` option on this image path.",
            ),
        ));
    }
}

fn generation_request_warnings(req: &ImageGenerationRequest) -> Option<Vec<Warning>> {
    let mut warnings = Vec::new();

    if req.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported_setting(
            "aspectRatio",
            Some("This model does not support aspect ratio. Use `size` instead."),
        ));
    }
    push_response_format_warning(&mut warnings, req.response_format.as_deref());
    if req.steps.is_some() {
        warnings.push(Warning::unsupported_setting(
            "steps",
            Some("Amazon Bedrock does not expose the generic `steps` option on this image path."),
        ));
    }
    if req.enhance_prompt.is_some() {
        warnings.push(Warning::unsupported_setting(
            "enhance_prompt",
            Some(
                "Amazon Bedrock does not expose the generic `enhance_prompt` option on this image path.",
            ),
        ));
    }

    (!warnings.is_empty()).then_some(warnings)
}

fn edit_request_warnings(req: &ImageEditRequest) -> Option<Vec<Warning>> {
    let mut warnings = Vec::new();

    if req.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported_setting(
            "aspectRatio",
            Some("This model does not support aspect ratio. Use `size` instead."),
        ));
    }
    push_response_format_warning(&mut warnings, req.response_format.as_deref());
    push_generation_only_extra_warnings(&mut warnings, &req.extra_params);

    (!warnings.is_empty()).then_some(warnings)
}

fn variation_request_warnings(req: &ImageVariationRequest) -> Option<Vec<Warning>> {
    let mut warnings = Vec::new();

    if req.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported_setting(
            "aspectRatio",
            Some("This model does not support aspect ratio. Use `size` instead."),
        ));
    }
    push_response_format_warning(&mut warnings, req.response_format.as_deref());
    push_generation_only_extra_warnings(&mut warnings, &req.extra_params);

    (!warnings.is_empty()).then_some(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bedrock_provider_options(value: serde_json::Value) -> crate::types::ProviderOptionsMap {
        let mut map = crate::types::ProviderOptionsMap::new();
        map.insert("bedrock", value);
        map
    }

    #[test]
    fn text_image_request_uses_sdk_shaped_bedrock_fields() {
        let transformer = BedrockImageRequestTransformer {
            provider_id: "bedrock".to_string(),
        };
        let request = ImageGenerationRequest {
            prompt: "a tiny silver robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            count: 2,
            quality: Some("draft".to_string()),
            style: Some("pixel".to_string()),
            guidance_scale: Some(5.5),
            provider_options_map: bedrock_provider_options(serde_json::json!({
                "quality": "premium",
                "cfgScale": 7.25,
                "negativeText": "washed out",
                "style": "photographic"
            })),
            ..Default::default()
        };

        let body = transformer
            .transform_image(&request)
            .expect("transform body");
        assert_eq!(
            body,
            serde_json::json!({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": "a tiny silver robot",
                    "negativeText": "washed out",
                    "style": "photographic"
                },
                "imageGenerationConfig": {
                    "width": 1024,
                    "height": 1024,
                    "numberOfImages": 2,
                    "quality": "premium",
                    "cfgScale": 7.25
                }
            })
        );
    }

    #[test]
    fn image_edit_request_defaults_to_inpainting_with_mask_image() {
        let transformer = BedrockImageRequestTransformer {
            provider_id: "bedrock".to_string(),
        };
        let request = ImageEditRequest {
            images: vec![ImageEditInput::file(vec![1, 2, 3])],
            mask: Some(ImageEditInput::file(vec![4, 5, 6])),
            prompt: "restore the sky".to_string(),
            count: Some(2),
            size: Some("768x512".to_string()),
            seed: Some(42),
            provider_options_map: bedrock_provider_options(serde_json::json!({
                "negativeText": "artifact",
                "cfgScale": 6.5
            })),
            extra_params: HashMap::from([("quality".to_string(), serde_json::json!("premium"))]),
            model: Some(NOVA_CANVAS_MODEL_ID.to_string()),
            response_format: None,
            aspect_ratio: None,
            http_config: None,
        };

        let body = transformer
            .transform_image_edit(&request)
            .expect("transform image edit");
        let ImageHttpBody::Json(body) = body else {
            panic!("expected json body");
        };

        assert_eq!(
            body,
            serde_json::json!({
                "taskType": "INPAINTING",
                "inPaintingParams": {
                    "image": "AQID",
                    "text": "restore the sky",
                    "negativeText": "artifact",
                    "maskImage": "BAUG"
                },
                "imageGenerationConfig": {
                    "width": 768,
                    "height": 512,
                    "seed": 42,
                    "numberOfImages": 2,
                    "quality": "premium",
                    "cfgScale": 6.5
                }
            })
        );
    }

    #[test]
    fn image_variation_request_forwards_similarity_strength() {
        let transformer = BedrockImageRequestTransformer {
            provider_id: "bedrock".to_string(),
        };
        let request = ImageVariationRequest {
            image: ImageEditInput::file(vec![1, 2, 3]),
            count: Some(3),
            size: Some("640x640".to_string()),
            seed: Some(11),
            provider_options_map: bedrock_provider_options(serde_json::json!({
                "similarityStrength": 0.35
            })),
            model: Some(NOVA_CANVAS_MODEL_ID.to_string()),
            response_format: None,
            aspect_ratio: None,
            extra_params: HashMap::new(),
            http_config: None,
        };

        let body = transformer
            .transform_image_variation(&request)
            .expect("transform image variation");
        let ImageHttpBody::Json(body) = body else {
            panic!("expected json body");
        };

        assert_eq!(body["taskType"], serde_json::json!("IMAGE_VARIATION"));
        assert_eq!(
            body["imageVariationParams"]["images"],
            serde_json::json!(["AQID"])
        );
        let similarity_strength = body["imageVariationParams"]["similarityStrength"]
            .as_f64()
            .expect("similarityStrength should be numeric");
        assert!(
            (similarity_strength - 0.35).abs() < 1e-6,
            "unexpected similarityStrength: {similarity_strength}"
        );
        assert_eq!(
            body["imageGenerationConfig"],
            serde_json::json!({
                "width": 640,
                "height": 640,
                "seed": 11,
                "numberOfImages": 3
            })
        );
    }

    #[test]
    fn response_transformer_reads_base64_images_and_metadata() {
        let transformer = BedrockImageResponseTransformer {
            provider_id: "bedrock".to_string(),
        };
        let response = transformer
            .transform_image_response(&serde_json::json!({
                "images": ["aGVsbG8="],
                "id": "job-1",
                "status": "Completed",
                "details": {
                    "seed": 42
                },
                "preview": {
                    "type": "thumbnail"
                }
            }))
            .expect("transform response");

        assert_eq!(response.images.len(), 1);
        assert_eq!(response.images[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert_eq!(
            response.metadata.get("id"),
            Some(&serde_json::json!("job-1"))
        );
        assert_eq!(
            response.metadata.get("status"),
            Some(&serde_json::json!("Completed"))
        );
        assert_eq!(
            response.metadata.get("details"),
            Some(&serde_json::json!({ "seed": 42 }))
        );
        let response_info = response.response.expect("response metadata");
        assert!(response_info.model_id.is_none());
        assert!(response_info.headers.is_empty());
        assert!(response_info.body.is_none());
    }

    #[test]
    fn response_transformer_raises_moderation_error() {
        let transformer = BedrockImageResponseTransformer {
            provider_id: "bedrock".to_string(),
        };
        let err = transformer
            .transform_image_response(&serde_json::json!({
                "status": "Request Moderated",
                "details": {
                    "Moderation Reasons": ["violence"]
                }
            }))
            .expect_err("moderated response should error");

        match err {
            LlmError::ApiError { message, .. } => {
                assert!(message.contains("violence"));
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[test]
    fn spec_warns_for_aspect_ratio_and_non_b64_response_format() {
        let spec = BedrockImageStandard::new().create_spec("bedrock");
        let ctx = ProviderContext::new(
            "bedrock",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );
        let request = ImageGenerationRequest {
            prompt: "robot".to_string(),
            aspect_ratio: Some("16:9".to_string()),
            response_format: Some("url".to_string()),
            ..Default::default()
        };

        let warnings = spec
            .image_warnings(&request, &ctx)
            .expect("warnings should exist");

        assert_eq!(
            warnings,
            vec![
                Warning::unsupported_setting(
                    "aspectRatio",
                    Some("This model does not support aspect ratio. Use `size` instead.")
                ),
                Warning::unsupported_setting(
                    "response_format",
                    Some(
                        "Amazon Bedrock image responses on this path always return base64 image payloads."
                    )
                )
            ]
        );
    }
}
