//! Image generation model family APIs.
//!
//! This is the recommended Rust-first surface for image generation:
//! - `generate` for direct generation
//! - `edit` for direct edit/inpainting requests
//! - `variation` for direct variation requests
//! - `generate_image` for the AI SDK-style unified request surface

use crate::request_options::{EffectiveRequestOptions, run_with_abort};
use crate::retry_api::{RetryOptions, retry_with};
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::future::try_join_all;
use siumai_core::error::LlmError;
use siumai_core::traits::ImageExtras;
use siumai_core::types::{
    HttpConfig, HttpResponseInfo, JSONValue, RequestOptions, Warning, merge_provider_metadata,
    provider_metadata_from_object,
};
use siumai_core::utils::{
    download_url,
    mime::{guess_mime_from_bytes, guess_mime_from_path_or_url},
};
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::image::{ImageModel, ImageModelV4};
pub use siumai_core::types::{
    GenerateImagePrompt, GenerateImageRequest, GenerateImageResult, GeneratedFile, GeneratedImage,
    ImageEditInput, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageModelProviderMetadata, ImageModelResponseMetadata, ImageModelUsage, ImageVariationRequest,
};

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

fn apply_generation_call_options(
    mut request: ImageGenerationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageGenerationRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

fn apply_edit_call_options(
    mut request: ImageEditRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageEditRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

fn apply_variation_call_options(
    mut request: ImageVariationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageVariationRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

fn apply_unified_call_options(
    mut request: GenerateImageRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> GenerateImageRequest {
    request.http_config = merge_http_config(request.http_config.take(), timeout, headers);
    request
}

fn normalize_generation_count(count: u32) -> u32 {
    count.max(1)
}

fn normalize_optional_generation_count(count: Option<u32>) -> u32 {
    count.unwrap_or(1).max(1)
}

fn resolve_effective_max_images_per_call(
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

fn split_call_image_counts(total_images: u32, max_images_per_call: u32) -> Vec<u32> {
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

async fn generate_single<M: ImageModel + ?Sized>(
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

async fn edit_single<M: ImageModel + ImageExtras + ?Sized>(
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

async fn variation_single<M: ImageModel + ImageExtras + ?Sized>(
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

fn ensure_images_generated(
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

async fn dispatch_generate_image<M: ImageModelV4 + ImageExtras + ?Sized>(
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

/// Generate images directly through the generation request family.
pub async fn generate<M: ImageModel + ?Sized>(
    model: &M,
    request: ImageGenerationRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    let GenerateOptions {
        retry,
        max_images_per_call,
        timeout,
        headers,
        request_options,
    } = options;
    let effective = EffectiveRequestOptions::from_parts(request_options, retry, timeout, headers);
    let mut request =
        apply_generation_call_options(request, effective.timeout(), effective.headers());
    request.count = normalize_generation_count(request.count);

    let max_images_per_call =
        resolve_effective_max_images_per_call(max_images_per_call, model.max_images_per_call())?;
    let call_image_counts = split_call_image_counts(request.count, max_images_per_call);
    let retry = effective.retry();
    let abort_signal = effective.abort_signal();
    let results = run_with_abort(
        abort_signal,
        try_join_all(call_image_counts.iter().copied().map(|count| {
            let mut req = request.clone();
            req.count = count;
            generate_single(model, req, retry.clone())
        })),
    )
    .await?;

    ensure_images_generated(results, call_image_counts)
}

/// Edit or inpaint images through the provider-owned image-extras lane.
pub async fn edit<M: ImageModel + ImageExtras + ?Sized>(
    model: &M,
    request: ImageEditRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    let GenerateOptions {
        retry,
        max_images_per_call,
        timeout,
        headers,
        request_options,
    } = options;
    let effective = EffectiveRequestOptions::from_parts(request_options, retry, timeout, headers);
    let mut request = apply_edit_call_options(request, effective.timeout(), effective.headers());
    let total_images = normalize_optional_generation_count(request.count);
    request.count = Some(total_images);

    let max_images_per_call =
        resolve_effective_max_images_per_call(max_images_per_call, model.max_images_per_call())?;
    let call_image_counts = split_call_image_counts(total_images, max_images_per_call);
    let retry = effective.retry();
    let abort_signal = effective.abort_signal();
    let results = run_with_abort(
        abort_signal,
        try_join_all(call_image_counts.iter().copied().map(|count| {
            let mut req = request.clone();
            req.count = Some(count);
            edit_single(model, req, retry.clone())
        })),
    )
    .await?;

    ensure_images_generated(results, call_image_counts)
}

/// Create image variations through the provider-owned image-extras lane.
pub async fn variation<M: ImageModel + ImageExtras + ?Sized>(
    model: &M,
    request: ImageVariationRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    let GenerateOptions {
        retry,
        max_images_per_call,
        timeout,
        headers,
        request_options,
    } = options;
    let effective = EffectiveRequestOptions::from_parts(request_options, retry, timeout, headers);
    let mut request =
        apply_variation_call_options(request, effective.timeout(), effective.headers());
    let total_images = normalize_optional_generation_count(request.count);
    request.count = Some(total_images);

    let max_images_per_call =
        resolve_effective_max_images_per_call(max_images_per_call, model.max_images_per_call())?;
    let call_image_counts = split_call_image_counts(total_images, max_images_per_call);
    let retry = effective.retry();
    let abort_signal = effective.abort_signal();
    let results = run_with_abort(
        abort_signal,
        try_join_all(call_image_counts.iter().copied().map(|count| {
            let mut req = request.clone();
            req.count = Some(count);
            variation_single(model, req, retry.clone())
        })),
    )
    .await?;

    ensure_images_generated(results, call_image_counts)
}

/// AI SDK-style unified image helper.
///
/// This bridges one stable request shape onto the current generation/edit/
/// variation execution lanes without forcing provider runtimes into one generic
/// transport path.
pub async fn generate_image<M: ImageModelV4 + ImageExtras + ?Sized>(
    model: &M,
    request: GenerateImageRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    let GenerateOptions {
        retry,
        max_images_per_call,
        timeout,
        headers,
        request_options,
    } = options;
    let effective = EffectiveRequestOptions::from_parts(request_options, retry, timeout, headers);
    let mut request = apply_unified_call_options(request, effective.timeout(), effective.headers());
    request.count = normalize_generation_count(request.count);

    let max_images_per_call =
        resolve_effective_max_images_per_call(max_images_per_call, model.max_images_per_call())?;
    let call_image_counts = split_call_image_counts(request.count, max_images_per_call);
    let retry = effective.retry();
    let abort_signal = effective.abort_signal();
    let results = run_with_abort(
        abort_signal,
        try_join_all(call_image_counts.iter().copied().map(|count| {
            let mut req = request.clone();
            req.count = count;
            let retry = retry.clone();
            async move {
                if let Some(retry) = retry {
                    retry_with(
                        || {
                            let retried_request = req.clone();
                            async move { dispatch_generate_image(model, retried_request).await }
                        },
                        retry,
                    )
                    .await
                } else {
                    dispatch_generate_image(model, req).await
                }
            }
        })),
    )
    .await?;

    ensure_images_generated(results, call_image_counts)
}

/// Deprecated AI SDK-style alias for `generate_image`.
#[deprecated(note = "Use generate_image instead.")]
pub async fn experimental_generate_image<M: ImageModelV4 + ImageExtras + ?Sized>(
    model: &M,
    request: GenerateImageRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    generate_image(model, request, options).await
}

fn normalize_image_media_type(value: &str) -> Option<String> {
    let media_type = value.split(';').next()?.trim();
    if media_type.is_empty() {
        return None;
    }
    Some(match media_type.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg".to_string(),
        "png" => "image/png".to_string(),
        "webp" => "image/webp".to_string(),
        "gif" => "image/gif".to_string(),
        "bmp" => "image/bmp".to_string(),
        "svg" => "image/svg+xml".to_string(),
        value if value.starts_with("image/") => value.to_string(),
        value if !value.contains('/') => format!("image/{value}"),
        value => value.to_string(),
    })
}

fn image_metadata_media_type(metadata: &HashMap<String, JSONValue>) -> Option<String> {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("media_type"))
        .or_else(|| metadata.get("mimeType"))
        .or_else(|| metadata.get("mime_type"))
        .and_then(JSONValue::as_str)
        .and_then(normalize_image_media_type)
}

fn generated_image_media_type(
    image: &GeneratedImage,
    bytes: Option<&[u8]>,
    downloaded_media_type: Option<&str>,
) -> String {
    image_metadata_media_type(&image.metadata)
        .or_else(|| downloaded_media_type.and_then(normalize_image_media_type))
        .or_else(|| bytes.and_then(guess_mime_from_bytes))
        .or_else(|| image.format.as_deref().and_then(normalize_image_media_type))
        .or_else(|| image.url.as_deref().and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "image/png".to_string())
}

async fn generated_image_to_file(image: &GeneratedImage) -> Result<GeneratedFile, LlmError> {
    if let Some(base64) = image.b64_json.as_deref() {
        if base64.starts_with("data:") {
            let downloaded = download_url(base64)
                .await
                .map_err(|error| LlmError::HttpError(error.message))?;
            let media_type = generated_image_media_type(
                image,
                Some(downloaded.data.as_slice()),
                downloaded.media_type.as_deref(),
            );
            return Ok(GeneratedFile::from_bytes(downloaded.data, media_type));
        }

        let bytes = STANDARD.decode(base64).map_err(|error| {
            LlmError::InvalidInput(format!("Invalid generated image base64 data: {error}"))
        })?;
        let media_type = generated_image_media_type(image, Some(bytes.as_slice()), None);
        return Ok(GeneratedFile::from_base64(base64.to_string(), media_type));
    }

    let Some(url) = image.url.as_deref() else {
        return Err(LlmError::ParseError(
            "Generated image was missing both `b64_json` and `url`".to_string(),
        ));
    };

    let downloaded = download_url(url)
        .await
        .map_err(|error| LlmError::HttpError(error.message))?;
    let media_type = generated_image_media_type(
        image,
        Some(downloaded.data.as_slice()),
        downloaded.media_type.as_deref(),
    );
    Ok(GeneratedFile::from_bytes(downloaded.data, media_type))
}

fn merge_image_provider_metadata(
    target: &mut ImageModelProviderMetadata,
    source: ImageModelProviderMetadata,
) {
    for (provider_id, source_value) in source {
        match (target.get_mut(&provider_id), source_value) {
            (Some(JSONValue::Object(target_obj)), JSONValue::Object(mut source_obj)) => {
                if let Some(JSONValue::Array(mut source_images)) = source_obj.remove("images") {
                    match target_obj.get_mut("images") {
                        Some(JSONValue::Array(target_images)) => {
                            target_images.append(&mut source_images);
                        }
                        _ => {
                            target_obj
                                .insert("images".to_string(), JSONValue::Array(source_images));
                        }
                    }
                }
                target_obj.extend(source_obj);
            }
            (_, value) => {
                target.insert(provider_id, value);
            }
        }
    }
}

fn provider_metadata_from_image_metadata_entry(
    provider_id: &str,
    mut metadata: HashMap<String, JSONValue>,
) -> ImageModelProviderMetadata {
    let mut provider_metadata = ImageModelProviderMetadata::default();

    if let Some(value) = metadata.remove(provider_id) {
        merge_provider_metadata(
            &mut provider_metadata,
            HashMap::from([(provider_id.to_string(), value)]),
        );
    }

    if !metadata.is_empty() {
        merge_provider_metadata(
            &mut provider_metadata,
            provider_metadata_from_object(provider_id.to_string(), metadata),
        );
    }

    provider_metadata
}

fn provider_metadata_from_image_metadata(
    provider_id: &str,
    metadata: &HashMap<String, JSONValue>,
) -> ImageModelProviderMetadata {
    let mut provider_metadata = ImageModelProviderMetadata::default();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("metadata"))
        .and_then(JSONValue::as_array)
    {
        for entry in entries {
            if let Ok(entry_metadata) =
                serde_json::from_value::<HashMap<String, JSONValue>>(entry.clone())
            {
                merge_image_provider_metadata(
                    &mut provider_metadata,
                    provider_metadata_from_image_metadata_entry(provider_id, entry_metadata),
                );
            }
        }
    }

    merge_image_provider_metadata(
        &mut provider_metadata,
        provider_metadata_from_image_metadata_entry(provider_id, metadata.clone()),
    );

    provider_metadata
}

fn u32_metadata_value(value: Option<&JSONValue>) -> Option<u32> {
    value
        .and_then(JSONValue::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn image_usage_from_metadata_entry(metadata: &HashMap<String, JSONValue>) -> ImageModelUsage {
    let usage = metadata
        .get("usage")
        .and_then(JSONValue::as_object)
        .or_else(|| {
            metadata.values().find_map(|value| {
                value
                    .as_object()
                    .and_then(|object| object.get("usage"))
                    .and_then(JSONValue::as_object)
            })
        });

    let Some(usage) = usage else {
        return ImageModelUsage::default();
    };

    ImageModelUsage::new(
        u32_metadata_value(
            usage
                .get("inputTokens")
                .or_else(|| usage.get("input_tokens")),
        ),
        u32_metadata_value(
            usage
                .get("outputTokens")
                .or_else(|| usage.get("output_tokens")),
        ),
        u32_metadata_value(
            usage
                .get("totalTokens")
                .or_else(|| usage.get("total_tokens")),
        ),
    )
}

fn image_usage_from_metadata(metadata: &HashMap<String, JSONValue>) -> ImageModelUsage {
    let mut usage = ImageModelUsage::default();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("metadata"))
        .and_then(JSONValue::as_array)
    {
        for entry in entries {
            if let Ok(entry_metadata) =
                serde_json::from_value::<HashMap<String, JSONValue>>(entry.clone())
            {
                usage.merge(&image_usage_from_metadata_entry(&entry_metadata));
            }
        }
    }

    usage.merge(&image_usage_from_metadata_entry(metadata));
    usage
}

fn image_response_metadata_from_response(
    metadata: &HashMap<String, JSONValue>,
    response: Option<&HttpResponseInfo>,
) -> Vec<ImageModelResponseMetadata> {
    let mut responses = Vec::new();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("responses"))
        .and_then(JSONValue::as_array)
    {
        responses.extend(entries.iter().filter_map(|entry| {
            serde_json::from_value::<HttpResponseInfo>(entry.clone())
                .ok()
                .and_then(|response| ImageModelResponseMetadata::try_from(response).ok())
        }));
    }

    if responses.is_empty()
        && let Some(response) = response
        && let Ok(metadata) = ImageModelResponseMetadata::try_from(response)
    {
        responses.push(metadata);
    }

    responses
}

async fn project_generate_image_response(
    provider_id: &str,
    response: ImageGenerationResponse,
) -> Result<GenerateImageResult, LlmError> {
    let ImageGenerationResponse {
        images,
        metadata,
        warnings,
        response,
    } = response;

    let mut files = Vec::with_capacity(images.len());
    for image in &images {
        files.push(generated_image_to_file(image).await?);
    }

    let Some(result) = GenerateImageResult::from_images(files) else {
        return Err(LlmError::NoImageGenerated {
            responses: response.into_iter().collect(),
        });
    };

    Ok(result
        .with_warnings(warnings.unwrap_or_default())
        .with_responses(image_response_metadata_from_response(
            &metadata,
            response.as_ref(),
        ))
        .with_provider_metadata(provider_metadata_from_image_metadata(
            provider_id,
            &metadata,
        ))
        .with_usage(image_usage_from_metadata(&metadata)))
}

/// Generate images and project the response into an AI SDK-style `GenerateImageResult`.
///
/// Use `image::generate_image` when you need the raw Rust-first
/// `ImageGenerationResponse` with provider-returned URLs preserved.
pub async fn generate_image_result<M: ImageModelV4 + ImageExtras + ?Sized>(
    model: &M,
    request: GenerateImageRequest,
    options: GenerateOptions,
) -> Result<GenerateImageResult, LlmError> {
    let response = generate_image(model, request, options).await?;
    project_generate_image_response(model.provider_id(), response).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::traits::{ImageGenerationCapability, ModelMetadata};
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeImageModel {
        routes: Mutex<Vec<&'static str>>,
        generation_requests: Mutex<Vec<ImageGenerationRequest>>,
        edit_requests: Mutex<Vec<ImageEditRequest>>,
        variation_requests: Mutex<Vec<ImageVariationRequest>>,
        supports_variation: bool,
        max_images_per_call: Option<u32>,
        forced_image_count: Option<u32>,
        base64_outputs: bool,
    }

    impl FakeImageModel {
        fn recorded_routes(&self) -> Vec<&'static str> {
            self.routes.lock().unwrap().clone()
        }

        fn push_route(&self, route: &'static str) {
            self.routes.lock().unwrap().push(route);
        }

        fn record_generation_request(&self, request: &ImageGenerationRequest) -> usize {
            let mut requests = self.generation_requests.lock().unwrap();
            let index = requests.len();
            requests.push(request.clone());
            index
        }

        fn recorded_generation_requests(&self) -> Vec<ImageGenerationRequest> {
            self.generation_requests.lock().unwrap().clone()
        }

        fn record_edit_request(&self, request: &ImageEditRequest) -> usize {
            let mut requests = self.edit_requests.lock().unwrap();
            let index = requests.len();
            requests.push(request.clone());
            index
        }

        fn recorded_edit_requests(&self) -> Vec<ImageEditRequest> {
            self.edit_requests.lock().unwrap().clone()
        }

        fn record_variation_request(&self, request: &ImageVariationRequest) -> usize {
            let mut requests = self.variation_requests.lock().unwrap();
            let index = requests.len();
            requests.push(request.clone());
            index
        }

        fn recorded_variation_requests(&self) -> Vec<ImageVariationRequest> {
            self.variation_requests.lock().unwrap().clone()
        }

        fn build_response(
            route: &'static str,
            call_index: usize,
            image_count: u32,
            forced_image_count: Option<u32>,
            base64_outputs: bool,
        ) -> ImageGenerationResponse {
            let image_count = forced_image_count.unwrap_or(image_count);
            let mut metadata = HashMap::from([
                ("route".to_string(), serde_json::json!(route)),
                ("call_index".to_string(), serde_json::json!(call_index)),
            ]);
            if base64_outputs {
                metadata.insert(
                    "fake".to_string(),
                    serde_json::json!({
                        "images": (0..image_count)
                            .map(|image_index| serde_json::json!({
                                "route": route,
                                "index": image_index,
                            }))
                            .collect::<Vec<_>>()
                    }),
                );
                metadata.insert(
                    "usage".to_string(),
                    serde_json::json!({
                        "inputTokens": call_index as u32 + 1,
                        "outputTokens": image_count,
                        "totalTokens": call_index as u32 + image_count + 1,
                    }),
                );
            }

            ImageGenerationResponse {
                images: (0..image_count)
                    .map(|image_index| {
                        let image_bytes = format!("{route}-{call_index}-{image_index}");
                        siumai_core::types::GeneratedImage {
                            url: (!base64_outputs).then(|| {
                                format!(
                                    "https://example.com/{route}-{call_index}-{image_index}.png"
                                )
                            }),
                            b64_json: base64_outputs
                                .then(|| STANDARD.encode(image_bytes.as_bytes())),
                            format: Some("png".to_string()),
                            width: None,
                            height: None,
                            revised_prompt: None,
                            metadata: HashMap::new(),
                        }
                    })
                    .collect(),
                metadata,
                warnings: Some(vec![Warning::other(format!("{route}-{call_index}"))]),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some(format!("{route}-{call_index}")),
                    headers: HashMap::from([("x-test-call".to_string(), call_index.to_string())]),
                    body: None,
                }),
            }
        }
    }

    impl ModelMetadata for FakeImageModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-image"
        }
    }

    #[async_trait::async_trait]
    impl ImageGenerationCapability for FakeImageModel {
        async fn generate_images(
            &self,
            request: ImageGenerationRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            self.push_route("generate");
            let call_index = self.record_generation_request(&request);
            Ok(Self::build_response(
                "generate",
                call_index,
                request.count,
                self.forced_image_count,
                self.base64_outputs,
            ))
        }

        fn max_images_per_call(&self) -> Option<u32> {
            self.max_images_per_call
        }
    }

    #[async_trait::async_trait]
    impl ImageExtras for FakeImageModel {
        async fn edit_image(
            &self,
            request: ImageEditRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            self.push_route("edit");
            let call_index = self.record_edit_request(&request);
            Ok(Self::build_response(
                "edit",
                call_index,
                normalize_optional_generation_count(request.count),
                self.forced_image_count,
                self.base64_outputs,
            ))
        }

        async fn create_variation(
            &self,
            request: ImageVariationRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            self.push_route("variation");
            let call_index = self.record_variation_request(&request);
            if self.supports_variation {
                Ok(Self::build_response(
                    "variation",
                    call_index,
                    normalize_optional_generation_count(request.count),
                    self.forced_image_count,
                    self.base64_outputs,
                ))
            } else {
                Err(LlmError::UnsupportedOperation(
                    "variation not supported".to_string(),
                ))
            }
        }
    }

    #[tokio::test]
    async fn unified_helper_routes_text_only_requests_to_generation() {
        let model = FakeImageModel::default();

        let _ = generate_image(
            &model,
            GenerateImageRequest::new("draw a robot"),
            Default::default(),
        )
        .await
        .unwrap();

        assert_eq!(model.recorded_routes(), vec!["generate"]);
    }

    #[tokio::test]
    async fn unified_helper_routes_prompt_plus_file_requests_to_edit() {
        let model = FakeImageModel::default();

        let _ = generate_image(
            &model,
            GenerateImageRequest::new("edit this robot")
                .with_file(ImageEditInput::url("https://example.com/input.png")),
            Default::default(),
        )
        .await
        .unwrap();

        assert_eq!(model.recorded_routes(), vec!["edit"]);
    }

    #[tokio::test]
    async fn unified_helper_routes_single_file_without_prompt_to_variation() {
        let model = FakeImageModel {
            supports_variation: true,
            ..Default::default()
        };

        let _ = generate_image(
            &model,
            GenerateImageRequest::default()
                .with_file(ImageEditInput::url("https://example.com/input.png")),
            Default::default(),
        )
        .await
        .unwrap();

        assert_eq!(model.recorded_routes(), vec!["variation"]);
    }

    #[tokio::test]
    async fn unified_helper_falls_back_to_edit_when_variation_is_unsupported() {
        let model = FakeImageModel::default();

        let _ = generate_image(
            &model,
            GenerateImageRequest::default()
                .with_file(ImageEditInput::url("https://example.com/input.png")),
            Default::default(),
        )
        .await
        .unwrap();

        assert_eq!(model.recorded_routes(), vec!["variation", "edit"]);
    }

    #[tokio::test]
    async fn unified_helper_preserves_generation_only_fields_on_variation_dispatch() {
        let model = FakeImageModel {
            supports_variation: true,
            ..Default::default()
        };

        let mut request = GenerateImageRequest::default()
            .with_file(ImageEditInput::url("https://example.com/input.png"))
            .with_seed(7)
            .with_aspect_ratio("1:1")
            .with_provider_option("openai", serde_json::json!({ "quality": "hd" }))
            .with_http_config(HttpConfig::empty());
        request.negative_prompt = Some("blurry".to_string());
        request.quality = Some("high".to_string());
        request.style = Some("vivid".to_string());
        request.steps = Some(28);
        request.guidance_scale = Some(6.5);
        request.enhance_prompt = Some(true);
        request
            .extra_params
            .insert("existing".to_string(), serde_json::json!("keep"));

        let _ = generate_image(&model, request, GenerateOptions::default())
            .await
            .unwrap();

        let variation_requests = model.recorded_variation_requests();
        assert_eq!(variation_requests.len(), 1);
        let request = &variation_requests[0];
        assert_eq!(request.count, Some(1));
        assert_eq!(request.aspect_ratio.as_deref(), Some("1:1"));
        assert_eq!(request.seed, Some(7));
        assert_eq!(
            request.provider_options_map.get("openai"),
            Some(&serde_json::json!({ "quality": "hd" }))
        );
        assert_eq!(
            request.extra_params.get("negative_prompt"),
            Some(&serde_json::json!("blurry"))
        );
        assert_eq!(
            request.extra_params.get("quality"),
            Some(&serde_json::json!("high"))
        );
        assert_eq!(
            request.extra_params.get("style"),
            Some(&serde_json::json!("vivid"))
        );
        assert_eq!(
            request.extra_params.get("steps"),
            Some(&serde_json::json!(28))
        );
        assert_eq!(
            request.extra_params.get("guidance_scale"),
            Some(&serde_json::json!(6.5))
        );
        assert_eq!(
            request.extra_params.get("enhance_prompt"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            request.extra_params.get("existing"),
            Some(&serde_json::json!("keep"))
        );
    }

    #[tokio::test]
    async fn unified_helper_downshifts_generation_only_fields_to_edit_extra_params_on_fallback() {
        let model = FakeImageModel::default();

        let mut request = GenerateImageRequest::default()
            .with_file(ImageEditInput::url("https://example.com/input.png"))
            .with_seed(42)
            .with_aspect_ratio("16:9")
            .with_provider_option("vertex", serde_json::json!({ "sampleCount": 2 }))
            .with_http_config(HttpConfig::empty());
        request.negative_prompt = Some("blurry".to_string());
        request.quality = Some("hd".to_string());
        request.style = Some("natural".to_string());
        request.steps = Some(32);
        request.guidance_scale = Some(7.25);
        request.enhance_prompt = Some(false);
        request
            .extra_params
            .insert("existing".to_string(), serde_json::json!("keep"));

        let _ = generate_image(&model, request, GenerateOptions::default())
            .await
            .unwrap();

        assert_eq!(model.recorded_routes(), vec!["variation", "edit"]);
        let edit_requests = model.recorded_edit_requests();
        assert_eq!(edit_requests.len(), 1);
        let request = &edit_requests[0];
        assert_eq!(request.count, Some(1));
        assert_eq!(request.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(request.seed, Some(42));
        assert_eq!(
            request.provider_options_map.get("vertex"),
            Some(&serde_json::json!({ "sampleCount": 2 }))
        );
        assert_eq!(
            request.extra_params.get("negative_prompt"),
            Some(&serde_json::json!("blurry"))
        );
        assert_eq!(
            request.extra_params.get("quality"),
            Some(&serde_json::json!("hd"))
        );
        assert_eq!(
            request.extra_params.get("style"),
            Some(&serde_json::json!("natural"))
        );
        assert_eq!(
            request.extra_params.get("steps"),
            Some(&serde_json::json!(32))
        );
        assert_eq!(
            request.extra_params.get("guidance_scale"),
            Some(&serde_json::json!(7.25))
        );
        assert_eq!(
            request.extra_params.get("enhance_prompt"),
            Some(&serde_json::json!(false))
        );
        assert_eq!(
            request.extra_params.get("existing"),
            Some(&serde_json::json!("keep"))
        );
    }

    #[tokio::test]
    async fn generate_batches_requests_using_explicit_max_images_per_call() {
        let model = FakeImageModel::default();

        let response = generate(
            &model,
            ImageGenerationRequest {
                prompt: "batch".to_string(),
                count: 5,
                ..Default::default()
            },
            GenerateOptions {
                max_images_per_call: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let requests = model.recorded_generation_requests();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.count)
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(response.images.len(), 5);
        assert_eq!(
            response
                .response
                .as_ref()
                .and_then(|response| response.model_id.as_deref()),
            Some("generate-0")
        );
        assert!(
            response
                .warnings
                .as_ref()
                .unwrap()
                .iter()
                .any(|warning| matches!(
                    warning,
                    Warning::Compatibility { feature, .. } if feature == "batched_image_calls"
                ))
        );
        let batch_metadata = response.metadata.get("_siumai").expect("batch metadata");
        assert_eq!(
            batch_metadata.get("batched_call_count"),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            batch_metadata.get("call_image_counts"),
            Some(&serde_json::json!([2, 2, 1]))
        );
        assert_eq!(
            batch_metadata.get("metadata"),
            Some(&serde_json::json!([
                { "call_index": 0, "route": "generate" },
                { "call_index": 1, "route": "generate" },
                { "call_index": 2, "route": "generate" }
            ]))
        );
        let responses = batch_metadata
            .get("responses")
            .and_then(|value| value.as_array())
            .expect("response metadata array");
        assert_eq!(responses.len(), 3);
        assert_eq!(
            responses[0].get("modelId").and_then(|value| value.as_str()),
            Some("generate-0")
        );
        assert_eq!(
            responses[1].get("modelId").and_then(|value| value.as_str()),
            Some("generate-1")
        );
        assert_eq!(
            responses[2].get("modelId").and_then(|value| value.as_str()),
            Some("generate-2")
        );
    }

    #[tokio::test]
    async fn generate_returns_no_image_generated_error_when_all_calls_are_empty() {
        let model = FakeImageModel {
            forced_image_count: Some(0),
            ..Default::default()
        };

        let err = generate(
            &model,
            ImageGenerationRequest {
                prompt: "empty".to_string(),
                count: 2,
                ..Default::default()
            },
            GenerateOptions {
                max_images_per_call: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

        match err {
            LlmError::NoImageGenerated { responses } => {
                assert_eq!(responses.len(), 2);
                assert_eq!(responses[0].model_id.as_deref(), Some("generate-0"));
                assert_eq!(responses[1].model_id.as_deref(), Some("generate-1"));
            }
            other => panic!("expected NoImageGenerated error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unified_generate_image_returns_no_image_generated_error_when_empty() {
        let model = FakeImageModel {
            forced_image_count: Some(0),
            ..Default::default()
        };

        let err = generate_image(
            &model,
            GenerateImageRequest::new("empty image result"),
            Default::default(),
        )
        .await
        .unwrap_err();

        match err {
            LlmError::NoImageGenerated { responses } => {
                assert_eq!(responses.len(), 1);
                assert_eq!(responses[0].model_id.as_deref(), Some("generate-0"));
            }
            other => panic!("expected NoImageGenerated error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_image_result_projects_ai_sdk_result_envelope() {
        let model = FakeImageModel {
            base64_outputs: true,
            max_images_per_call: Some(2),
            ..Default::default()
        };
        let mut request = GenerateImageRequest::new("batch result");
        request.count = 3;

        let result = generate_image_result(&model, request, Default::default())
            .await
            .expect("project image result");

        assert_eq!(result.images.len(), 3);
        assert_eq!(result.image.base64, STANDARD.encode("generate-0-0"));
        assert_eq!(result.image.media_type, "image/png");
        assert_eq!(
            result
                .images
                .iter()
                .map(|image| image.base64.as_str())
                .collect::<Vec<_>>(),
            vec![
                STANDARD.encode("generate-0-0"),
                STANDARD.encode("generate-0-1"),
                STANDARD.encode("generate-1-0"),
            ]
        );
        assert_eq!(result.responses.len(), 2);
        assert_eq!(result.responses[0].model_id, "generate-0");
        assert_eq!(
            result.responses[0]
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-test-call")),
            Some(&"0".to_string())
        );
        assert!(result.warnings.iter().any(|warning| {
            matches!(
                warning,
                Warning::Compatibility { feature, .. } if feature == "batched_image_calls"
            )
        }));
        assert_eq!(result.usage.input_tokens, Some(3));
        assert_eq!(result.usage.output_tokens, Some(3));
        assert_eq!(result.usage.total_tokens, Some(6));

        let fake_metadata = result
            .provider_metadata
            .get("fake")
            .and_then(JSONValue::as_object)
            .expect("fake provider metadata");
        assert_eq!(
            fake_metadata
                .get("images")
                .and_then(JSONValue::as_array)
                .map(Vec::len),
            Some(3)
        );
        assert!(fake_metadata.get("_siumai").is_some());
    }

    #[tokio::test]
    async fn generate_image_result_materializes_data_url_images() {
        let response = ImageGenerationResponse {
            images: vec![GeneratedImage {
                url: Some("data:image/png;base64,aGVsbG8=".to_string()),
                b64_json: None,
                format: None,
                width: None,
                height: None,
                revised_prompt: None,
                metadata: HashMap::new(),
            }],
            metadata: HashMap::new(),
            warnings: None,
            response: None,
        };

        let result = project_generate_image_response("fake", response)
            .await
            .expect("project data url result");

        assert_eq!(result.image.base64, "aGVsbG8=");
        assert_eq!(result.image.media_type, "image/png");
        assert!(result.responses.is_empty());
    }

    #[tokio::test]
    async fn generate_uses_model_default_max_images_per_call_when_option_is_missing() {
        let model = FakeImageModel {
            max_images_per_call: Some(3),
            ..Default::default()
        };

        let _ = generate(
            &model,
            ImageGenerationRequest {
                prompt: "batch".to_string(),
                count: 5,
                ..Default::default()
            },
            GenerateOptions::default(),
        )
        .await
        .unwrap();

        let requests = model.recorded_generation_requests();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.count)
                .collect::<Vec<_>>(),
            vec![3, 2]
        );
    }

    #[tokio::test]
    async fn edit_batches_requests_using_model_default_max_images_per_call() {
        let model = FakeImageModel {
            max_images_per_call: Some(2),
            ..Default::default()
        };

        let response = edit(
            &model,
            ImageEditRequest {
                prompt: "edit".to_string(),
                images: vec![ImageEditInput::url("https://example.com/input.png")],
                count: Some(5),
                ..Default::default()
            },
            GenerateOptions::default(),
        )
        .await
        .unwrap();

        let requests = model.recorded_edit_requests();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.count.unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(response.images.len(), 5);
    }

    #[tokio::test]
    async fn unified_helper_batches_variation_fallback_per_chunk() {
        let model = FakeImageModel {
            max_images_per_call: Some(2),
            ..Default::default()
        };

        let response = generate_image(
            &model,
            {
                let mut request = GenerateImageRequest::default()
                    .with_file(ImageEditInput::url("https://example.com/input.png"))
                    .with_seed(9)
                    .with_aspect_ratio("1:1")
                    .with_provider_option("vertex", serde_json::json!({ "sampleCount": 2 }))
                    .with_http_config(HttpConfig::empty());
                request.count = 3;
                request
            },
            GenerateOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            model.recorded_routes(),
            vec!["variation", "edit", "variation", "edit"]
        );
        let variation_requests = model.recorded_variation_requests();
        assert_eq!(
            variation_requests
                .iter()
                .map(|request| request.count.unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        let edit_requests = model.recorded_edit_requests();
        assert_eq!(
            edit_requests
                .iter()
                .map(|request| request.count.unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(response.images.len(), 3);
        assert_eq!(
            response
                .metadata
                .get("_siumai")
                .and_then(|value| value.get("call_image_counts")),
            Some(&serde_json::json!([2, 1]))
        );
    }
}
