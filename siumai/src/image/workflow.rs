use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::image::{ImageModel, ImageModelV4};
use siumai_core::traits::ImageExtras;
use siumai_core::types::{
    GenerateImageRequest, HttpConfig, HttpResponseInfo, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, RequestOptions, Warning,
};
use std::collections::HashMap;
use std::time::Duration;
/// Options for image-family helper calls.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Maximum number of images to generate in a single provider call.
    ///
    /// When omitted, the helper falls back to the model/provider default if one
    /// is exposed, and finally to `1`.
    pub max_images_per_call: Option<u32>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `ImageGenerationRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `ImageGenerationRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnifiedImageDispatchKind {
    Generate,
    Edit,
    Variation,
}

fn merge_http_config(
    mut http_config: Option<HttpConfig>,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> Option<HttpConfig> {
    if timeout.is_none() && headers.is_empty() {
        return http_config;
    }

    let mut http = http_config.take().unwrap_or_else(HttpConfig::empty);
    if let Some(t) = timeout {
        http.timeout = Some(t);
    }
    if !headers.is_empty() {
        http.headers.extend(headers);
    }
    Some(http)
}

pub(super) fn apply_generation_call_options(
    mut request: ImageGenerationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageGenerationRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

pub(super) fn apply_edit_call_options(
    mut request: ImageEditRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageEditRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

pub(super) fn apply_variation_call_options(
    mut request: ImageVariationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageVariationRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

pub(super) fn apply_unified_call_options(
    mut request: GenerateImageRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> GenerateImageRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

pub(super) fn normalize_generation_count(count: u32) -> u32 {
    count.max(1)
}

pub(super) fn normalize_optional_generation_count(count: Option<u32>) -> u32 {
    count.unwrap_or(1).max(1)
}

pub(super) fn resolve_effective_max_images_per_call(
    explicit: Option<u32>,
    model_default: Option<u32>,
) -> Result<u32, LlmError> {
    let limit = explicit.or(model_default).unwrap_or(1);
    if limit == 0 {
        return Err(LlmError::InvalidParameter(
            "GenerateOptions.max_images_per_call must be greater than 0".to_string(),
        ));
    }
    Ok(limit)
}

pub(super) fn split_call_image_counts(total_images: u32, max_images_per_call: u32) -> Vec<u32> {
    let total_images = normalize_generation_count(total_images);
    let mut remaining = total_images;
    let mut counts = Vec::new();
    while remaining > 0 {
        let current = remaining.min(max_images_per_call);
        counts.push(current);
        remaining -= current;
    }
    counts
}

pub(super) async fn generate_single<M: ImageModel + ?Sized>(
    model: &M,
    request: ImageGenerationRequest,
    retry: Option<RetryOptions>,
) -> Result<ImageGenerationResponse, LlmError> {
    if let Some(retry) = retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.generate(req).await }
            },
            retry,
        )
        .await
    } else {
        model.generate(request).await
    }
}

pub(super) async fn edit_single<M: ImageModel + ImageExtras + ?Sized>(
    model: &M,
    request: ImageEditRequest,
    retry: Option<RetryOptions>,
) -> Result<ImageGenerationResponse, LlmError> {
    if let Some(retry) = retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.edit_image(req).await }
            },
            retry,
        )
        .await
    } else {
        model.edit_image(request).await
    }
}

pub(super) async fn variation_single<M: ImageModel + ImageExtras + ?Sized>(
    model: &M,
    request: ImageVariationRequest,
    retry: Option<RetryOptions>,
) -> Result<ImageGenerationResponse, LlmError> {
    if let Some(retry) = retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.create_variation(req).await }
            },
            retry,
        )
        .await
    } else {
        model.create_variation(request).await
    }
}

fn serialize_http_response_info(response: HttpResponseInfo) -> serde_json::Value {
    serde_json::to_value(response).unwrap_or_else(|_| serde_json::Value::Null)
}

fn serialize_response_metadata(metadata: HashMap<String, serde_json::Value>) -> serde_json::Value {
    serde_json::to_value(metadata).unwrap_or_else(|_| serde_json::Value::Null)
}

fn merge_batched_image_responses(
    results: Vec<ImageGenerationResponse>,
    call_image_counts: Vec<u32>,
) -> ImageGenerationResponse {
    let mut results = results.into_iter();
    let Some(first) = results.next() else {
        return ImageGenerationResponse {
            images: Vec::new(),
            metadata: HashMap::new(),
            warnings: None,
            response: None,
        };
    };

    if call_image_counts.len() <= 1 {
        return first;
    }

    let mut all_results = Vec::with_capacity(call_image_counts.len());
    all_results.push(first);
    all_results.extend(results);

    let mut images = Vec::new();
    let mut warnings = Vec::new();
    let mut responses = Vec::new();
    let mut metadata_entries = Vec::new();
    let mut top_level_response = None;

    for result in all_results {
        images.extend(result.images);
        if let Some(result_warnings) = result.warnings {
            warnings.extend(result_warnings);
        }
        if let Some(response) = result.response {
            if top_level_response.is_none() {
                top_level_response = Some(response.clone());
            }
            responses.push(serialize_http_response_info(response));
        }
        metadata_entries.push(serialize_response_metadata(result.metadata));
    }

    warnings.push(Warning::compatibility(
        "batched_image_calls",
        Some(
            "Per-call metadata and response envelopes are preserved under `metadata._siumai` because the stable Rust image response still exposes a single `response` field.",
        ),
    ));

    let mut metadata = HashMap::new();
    metadata.insert(
        "_siumai".to_string(),
        serde_json::json!({
            "batched_call_count": call_image_counts.len(),
            "call_image_counts": call_image_counts,
            "responses": responses,
            "metadata": metadata_entries,
        }),
    );

    ImageGenerationResponse {
        images,
        metadata,
        warnings: Some(warnings),
        response: top_level_response,
    }
}

pub(super) fn ensure_images_generated(
    results: Vec<ImageGenerationResponse>,
    call_image_counts: Vec<u32>,
) -> Result<ImageGenerationResponse, LlmError> {
    let responses = results
        .iter()
        .filter_map(|result| result.response.clone())
        .collect::<Vec<_>>();

    if results.iter().all(|result| result.images.is_empty()) {
        return Err(LlmError::NoImageGenerated { responses });
    }

    Ok(merge_batched_image_responses(results, call_image_counts))
}

fn has_prompt(prompt: Option<&str>) -> bool {
    prompt.is_some_and(|value| !value.trim().is_empty())
}

fn classify_generate_image_request(request: &GenerateImageRequest) -> UnifiedImageDispatchKind {
    if request.files.is_empty() && request.mask.is_none() {
        UnifiedImageDispatchKind::Generate
    } else if request.mask.is_some()
        || request.files.len() != 1
        || has_prompt(request.prompt.as_deref())
    {
        UnifiedImageDispatchKind::Edit
    } else {
        UnifiedImageDispatchKind::Variation
    }
}

fn preserve_generation_only_fields_as_extra_params(
    extra_params: &mut HashMap<String, serde_json::Value>,
    negative_prompt: Option<String>,
    quality: Option<String>,
    style: Option<String>,
    steps: Option<u32>,
    guidance_scale: Option<f32>,
    enhance_prompt: Option<bool>,
) {
    if let Some(value) = negative_prompt {
        extra_params
            .entry("negative_prompt".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
    if let Some(value) = quality {
        extra_params
            .entry("quality".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
    if let Some(value) = style {
        extra_params
            .entry("style".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
    if let Some(value) = steps {
        extra_params
            .entry("steps".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
    if let Some(value) = guidance_scale {
        extra_params
            .entry("guidance_scale".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
    if let Some(value) = enhance_prompt {
        extra_params
            .entry("enhance_prompt".to_string())
            .or_insert_with(|| serde_json::json!(value));
    }
}

fn into_generation_request(request: GenerateImageRequest) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: request.prompt.unwrap_or_default(),
        negative_prompt: request.negative_prompt,
        size: request.size,
        aspect_ratio: request.aspect_ratio,
        count: request.count.max(1),
        model: request.model,
        quality: request.quality,
        style: request.style,
        seed: request.seed,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        enhance_prompt: request.enhance_prompt,
        response_format: request.response_format,
        extra_params: request.extra_params,
        provider_options_map: request.provider_options_map,
        http_config: request.http_config,
    }
}

fn into_edit_request(request: GenerateImageRequest) -> ImageEditRequest {
    let GenerateImageRequest {
        prompt,
        files,
        mask,
        negative_prompt,
        size,
        aspect_ratio,
        count,
        model,
        quality,
        style,
        seed,
        steps,
        guidance_scale,
        enhance_prompt,
        response_format,
        mut extra_params,
        provider_options_map,
        http_config,
    } = request;

    preserve_generation_only_fields_as_extra_params(
        &mut extra_params,
        negative_prompt,
        quality,
        style,
        steps,
        guidance_scale,
        enhance_prompt,
    );

    ImageEditRequest {
        images: files,
        mask,
        prompt: prompt.unwrap_or_default(),
        model,
        count: Some(count.max(1)),
        size,
        aspect_ratio,
        seed,
        response_format,
        extra_params,
        provider_options_map,
        http_config,
    }
}

fn into_variation_request(
    request: GenerateImageRequest,
) -> Result<ImageVariationRequest, LlmError> {
    let GenerateImageRequest {
        prompt: _,
        files,
        mask,
        negative_prompt,
        size,
        aspect_ratio,
        count,
        model,
        quality,
        style,
        seed,
        steps,
        guidance_scale,
        enhance_prompt,
        response_format,
        mut extra_params,
        provider_options_map,
        http_config,
    } = request;

    if mask.is_some() {
        return Err(LlmError::InvalidParameter(
            "Unified image variation dispatch does not accept a mask input".to_string(),
        ));
    }

    let mut files_iter = files.into_iter();
    let image = files_iter.next().ok_or_else(|| {
        LlmError::InvalidParameter(
            "Unified image variation dispatch requires exactly one input file".to_string(),
        )
    })?;
    if files_iter.next().is_some() {
        return Err(LlmError::InvalidParameter(
            "Unified image variation dispatch requires exactly one input file".to_string(),
        ));
    }

    preserve_generation_only_fields_as_extra_params(
        &mut extra_params,
        negative_prompt,
        quality,
        style,
        steps,
        guidance_scale,
        enhance_prompt,
    );

    Ok(ImageVariationRequest {
        image,
        model,
        count: Some(count.max(1)),
        size,
        aspect_ratio,
        seed,
        response_format,
        extra_params,
        provider_options_map,
        http_config,
    })
}

pub(super) async fn dispatch_generate_image<M: ImageModelV4 + ImageExtras + ?Sized>(
    model: &M,
    request: GenerateImageRequest,
) -> Result<ImageGenerationResponse, LlmError> {
    match classify_generate_image_request(&request) {
        UnifiedImageDispatchKind::Generate => {
            model.generate(into_generation_request(request)).await
        }
        UnifiedImageDispatchKind::Edit => model.edit_image(into_edit_request(request)).await,
        UnifiedImageDispatchKind::Variation => {
            let edit_fallback_request = into_edit_request(request.clone());
            let variation_request = into_variation_request(request)?;
            match model.create_variation(variation_request).await {
                Err(LlmError::UnsupportedOperation(_)) => {
                    model.edit_image(edit_fallback_request).await
                }
                other => other,
            }
        }
    }
}
