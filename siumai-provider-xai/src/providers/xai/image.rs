use super::XaiClient;
use super::http::{build_http_execution_config, headers_to_map};
use crate::error::LlmError;
use crate::provider_options::{XaiImageOptions, XaiImageQuality, XaiImageResolution};
use crate::types::{
    GeneratedImage, HttpResponseInfo, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, Warning,
};
use crate::utils::mime::guess_mime_from_bytes;
use base64::Engine;
use std::collections::HashMap;

fn parse_xai_image_options(
    map: &crate::types::ProviderOptionsMap,
) -> Result<Option<XaiImageOptions>, LlmError> {
    let Some(value) = map.get("xai") else {
        return Ok(None);
    };

    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|err| {
            LlmError::InvalidParameter(format!(
                "Invalid xAI image options in providerOptions.xai: {err}"
            ))
        })
}

fn push_warning(warnings: &mut Vec<Warning>, feature: &str, details: &'static str) {
    warnings.push(Warning::unsupported(feature, Some(details)));
}

fn object_string(
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
    value.as_str().map(|v| Some(v.to_string())).ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "xAI image extra param '{primary}' must be a string when provided"
        ))
    })
}

fn object_bool(
    extra_params: &HashMap<String, serde_json::Value>,
    primary: &str,
    alias: &str,
) -> Result<Option<bool>, LlmError> {
    let value = extra_params
        .get(primary)
        .or_else(|| extra_params.get(alias));
    let Some(value) = value else {
        return Ok(None);
    };
    value.as_bool().map(Some).ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "xAI image extra param '{primary}' must be a boolean when provided"
        ))
    })
}

fn image_data_url(bytes: &[u8]) -> String {
    let mime = guess_mime_from_bytes(bytes).unwrap_or_else(|| "image/png".to_string());
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{encoded}")
}

fn image_edit_input_url(input: &ImageEditInput) -> Result<String, LlmError> {
    match input {
        ImageEditInput::Url { url } => Ok(url.clone()),
        ImageEditInput::File { data, media_type } => {
            let bytes = data.as_bytes().map_err(|err| {
                LlmError::InvalidParameter(format!("Invalid xAI image edit file data: {err}"))
            })?;
            if let Some(media_type) = media_type {
                let encoded = data.as_base64();
                Ok(format!("data:{media_type};base64,{encoded}"))
            } else {
                Ok(image_data_url(&bytes))
            }
        }
    }
}

fn resolve_model(request_model: Option<&str>, fallback_model: &str) -> String {
    match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ => fallback_model.to_string(),
    }
}

fn resolve_image_options(
    extra_params: &HashMap<String, serde_json::Value>,
    provider_options: Option<XaiImageOptions>,
) -> Result<XaiImageOptions, LlmError> {
    let mut options = provider_options.unwrap_or_default();

    if options.aspect_ratio.is_none() {
        options.aspect_ratio = object_string(extra_params, "aspect_ratio", "aspectRatio")?;
    }
    if options.output_format.is_none() {
        options.output_format = object_string(extra_params, "output_format", "outputFormat")?;
    }
    if options.sync_mode.is_none() {
        options.sync_mode = object_bool(extra_params, "sync_mode", "syncMode")?;
    }
    if options.resolution.is_none()
        && let Some(value) = object_string(extra_params, "resolution", "resolution")?
    {
        options.resolution = Some(XaiImageResolution::from(value));
    }
    if options.quality.is_none()
        && let Some(value) = object_string(extra_params, "quality", "quality")?
    {
        options.quality = Some(XaiImageQuality::from(value));
    }
    if options.user.is_none() {
        options.user = object_string(extra_params, "user", "user")?;
    }

    Ok(options)
}

fn build_generation_body(
    client: &XaiClient,
    request: &ImageGenerationRequest,
) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
    let provider_options = parse_xai_image_options(&request.provider_options_map)?;
    let xai_options = resolve_image_options(&request.extra_params, provider_options)?;
    let mut warnings = Vec::new();

    if request.size.is_some() {
        push_warning(
            &mut warnings,
            "size",
            "xAI image models do not support the generic `size` option. Use `providerOptions.xai.aspectRatio` instead.",
        );
    }
    if request.seed.is_some() {
        push_warning(
            &mut warnings,
            "seed",
            "xAI image models do not support the generic `seed` option.",
        );
    }
    if request.negative_prompt.is_some() {
        push_warning(
            &mut warnings,
            "negative_prompt",
            "xAI image models do not support `negative_prompt` on the provider-owned path.",
        );
    }
    if request.style.is_some() {
        push_warning(
            &mut warnings,
            "style",
            "xAI image models do not support the generic `style` option.",
        );
    }
    if request.steps.is_some() {
        push_warning(
            &mut warnings,
            "steps",
            "xAI image models do not expose `steps` on the provider-owned path.",
        );
    }
    if request.guidance_scale.is_some() {
        push_warning(
            &mut warnings,
            "guidance_scale",
            "xAI image models do not expose `guidance_scale` on the provider-owned path.",
        );
    }
    if request.enhance_prompt.is_some() {
        push_warning(
            &mut warnings,
            "enhance_prompt",
            "xAI image models do not expose `enhance_prompt` on the provider-owned path.",
        );
    }
    if request
        .response_format
        .as_deref()
        .is_some_and(|value| value != "b64_json")
    {
        push_warning(
            &mut warnings,
            "response_format",
            "xAI image models on the provider-owned path always request `b64_json` responses.",
        );
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::json!(resolve_model(
            request.model.as_deref(),
            client.inner().model(),
        )),
    );
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert("n".to_string(), serde_json::json!(request.count.max(1)));
    body.insert("response_format".to_string(), serde_json::json!("b64_json"));

    if let Some(aspect_ratio) = xai_options.aspect_ratio {
        body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some(output_format) = xai_options.output_format {
        body.insert(
            "output_format".to_string(),
            serde_json::json!(output_format),
        );
    }
    if let Some(sync_mode) = xai_options.sync_mode {
        body.insert("sync_mode".to_string(), serde_json::json!(sync_mode));
    }
    if let Some(resolution) = xai_options.resolution {
        body.insert(
            "resolution".to_string(),
            serde_json::json!(resolution.as_str()),
        );
    }
    if let Some(quality) = xai_options.quality {
        body.insert("quality".to_string(), serde_json::json!(quality.as_str()));
    }
    if let Some(user) = xai_options.user {
        body.insert("user".to_string(), serde_json::json!(user));
    }
    for (key, value) in xai_options.extra_fields {
        body.entry(key).or_insert(value);
    }

    Ok((serde_json::Value::Object(body), warnings))
}

fn build_edit_body(
    client: &XaiClient,
    request: &ImageEditRequest,
) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
    if request.images.is_empty() {
        return Err(LlmError::InvalidParameter(
            "xAI image editing requires at least one source image".to_string(),
        ));
    }

    let provider_options = parse_xai_image_options(&request.provider_options_map)?;
    let xai_options = resolve_image_options(&request.extra_params, provider_options)?;
    let mut warnings = Vec::new();

    if request.mask.is_some() {
        push_warning(
            &mut warnings,
            "mask",
            "xAI image editing does not support `mask` on the provider-owned path.",
        );
    }
    if request.size.is_some() {
        push_warning(
            &mut warnings,
            "size",
            "xAI image editing does not support the generic `size` option. Use `providerOptions.xai.aspectRatio` instead.",
        );
    }
    if request
        .response_format
        .as_deref()
        .is_some_and(|value| value != "b64_json")
    {
        push_warning(
            &mut warnings,
            "response_format",
            "xAI image editing on the provider-owned path always requests `b64_json` responses.",
        );
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::json!(resolve_model(
            request.model.as_deref(),
            client.inner().model(),
        )),
    );
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert(
        "n".to_string(),
        serde_json::json!(request.count.unwrap_or(1).max(1)),
    );
    body.insert("response_format".to_string(), serde_json::json!("b64_json"));

    if let Some(aspect_ratio) = xai_options.aspect_ratio {
        body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some(output_format) = xai_options.output_format {
        body.insert(
            "output_format".to_string(),
            serde_json::json!(output_format),
        );
    }
    if let Some(sync_mode) = xai_options.sync_mode {
        body.insert("sync_mode".to_string(), serde_json::json!(sync_mode));
    }
    if let Some(resolution) = xai_options.resolution {
        body.insert(
            "resolution".to_string(),
            serde_json::json!(resolution.as_str()),
        );
    }
    if let Some(quality) = xai_options.quality {
        body.insert("quality".to_string(), serde_json::json!(quality.as_str()));
    }
    if let Some(user) = xai_options.user {
        body.insert("user".to_string(), serde_json::json!(user));
    }
    for (key, value) in xai_options.extra_fields {
        body.entry(key).or_insert(value);
    }

    let image_inputs = request
        .images
        .iter()
        .map(image_edit_input_url)
        .collect::<Result<Vec<_>, _>>()?;

    if image_inputs.len() == 1 {
        body.insert(
            "image".to_string(),
            serde_json::json!({
                "url": image_inputs[0],
                "type": "image_url"
            }),
        );
    } else {
        body.insert(
            "images".to_string(),
            serde_json::Value::Array(
                image_inputs
                    .into_iter()
                    .map(|url| {
                        serde_json::json!({
                            "url": url,
                            "type": "image_url"
                        })
                    })
                    .collect(),
            ),
        );
    }

    Ok((serde_json::Value::Object(body), warnings))
}

#[derive(Debug, serde::Deserialize)]
struct XaiImageResponse {
    data: Vec<XaiImageData>,
    #[serde(default)]
    usage: Option<XaiImageUsage>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiImageData {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiImageUsage {
    #[serde(default)]
    cost_in_usd_ticks: Option<serde_json::Value>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

async fn download_as_base64(client: &XaiClient, url: &str) -> Result<String, LlmError> {
    let config = build_http_execution_config(client);
    let result =
        crate::execution::executors::common::execute_get_binary(&config, url, None).await?;
    Ok(base64::engine::general_purpose::STANDARD.encode(result.bytes))
}

async fn parse_image_response(
    client: &XaiClient,
    json: serde_json::Value,
    model: String,
    output_format: Option<String>,
    warnings: Vec<Warning>,
    headers: reqwest::header::HeaderMap,
) -> Result<ImageGenerationResponse, LlmError> {
    let response: XaiImageResponse = serde_json::from_value(json).map_err(|err| {
        LlmError::ParseError(format!("Failed to parse xAI image response: {err}"))
    })?;

    let mut images = Vec::with_capacity(response.data.len());
    for item in response.data {
        let b64_json = match (item.b64_json, item.url) {
            (Some(b64), _) => Some(b64),
            (None, Some(url)) => Some(download_as_base64(client, &url).await?),
            (None, None) => {
                return Err(LlmError::ParseError(
                    "xAI image response item was missing both `b64_json` and `url`".to_string(),
                ));
            }
        };

        images.push(GeneratedImage {
            url: None,
            b64_json,
            format: output_format.clone(),
            width: None,
            height: None,
            revised_prompt: item.revised_prompt,
            metadata: item.extra_fields,
        });
    }

    let mut metadata = response.extra_fields;
    if let Some(usage) = response.usage {
        let mut usage_meta = usage.extra_fields;
        if let Some(cost) = usage.cost_in_usd_ticks {
            usage_meta.insert("cost_in_usd_ticks".to_string(), cost);
        }
        if !usage_meta.is_empty() {
            metadata.insert(
                "usage".to_string(),
                serde_json::Value::Object(usage_meta.into_iter().collect()),
            );
        }
    }

    Ok(ImageGenerationResponse {
        images,
        metadata,
        warnings: (!warnings.is_empty()).then_some(warnings),
        response: Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: Some(model),
            headers: headers_to_map(&headers),
        }),
    })
}

pub(super) async fn generate_images(
    client: &XaiClient,
    mut request: ImageGenerationRequest,
) -> Result<ImageGenerationResponse, LlmError> {
    client.merge_default_provider_options_map_non_chat(&mut request.provider_options_map);
    let (body, warnings) = build_generation_body(client, &request)?;
    let model = resolve_model(request.model.as_deref(), client.inner().model());
    let output_format = resolve_image_options(
        &request.extra_params,
        parse_xai_image_options(&request.provider_options_map)?,
    )?
    .output_format;
    let config = build_http_execution_config(client);
    let url = format!(
        "{}/images/generations",
        client.base_url().trim_end_matches('/')
    );
    let result = crate::execution::executors::common::execute_json_request(
        &config,
        &url,
        crate::execution::executors::common::HttpBody::Json(body),
        request.http_config.as_ref(),
        false,
    )
    .await?;

    parse_image_response(
        client,
        result.json,
        model,
        output_format,
        warnings,
        result.headers,
    )
    .await
}

pub(super) async fn edit_image(
    client: &XaiClient,
    mut request: ImageEditRequest,
) -> Result<ImageGenerationResponse, LlmError> {
    client.merge_default_provider_options_map_non_chat(&mut request.provider_options_map);
    let provider_options = parse_xai_image_options(&request.provider_options_map)?;
    let output_format =
        resolve_image_options(&request.extra_params, provider_options.clone())?.output_format;
    let (body, warnings) = build_edit_body(client, &request)?;
    let model = resolve_model(request.model.as_deref(), client.inner().model());
    let config = build_http_execution_config(client);
    let url = format!("{}/images/edits", client.base_url().trim_end_matches('/'));
    let result = crate::execution::executors::common::execute_json_request(
        &config,
        &url,
        crate::execution::executors::common::HttpBody::Json(body),
        request.http_config.as_ref(),
        false,
    )
    .await?;

    parse_image_response(
        client,
        result.json,
        model,
        output_format,
        warnings,
        result.headers,
    )
    .await
}

pub(super) async fn create_variation(
    _client: &XaiClient,
    _request: ImageVariationRequest,
) -> Result<ImageGenerationResponse, LlmError> {
    Err(LlmError::UnsupportedOperation(
        "xAI does not expose image variations on the provider-owned path".to_string(),
    ))
}
