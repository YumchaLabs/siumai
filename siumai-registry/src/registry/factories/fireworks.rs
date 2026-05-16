//! Fireworks unified provider factory.
//!
//! AI SDK exposes Fireworks as one provider surface:
//! - OpenAI-compatible chat/completion/embedding/transcription
//! - provider-owned image generation/edit routes

use super::*;
use crate::core::{ProviderContext, ProviderSpec};
use crate::execution::executors::common::{
    HttpBody, execute_bytes_request, execute_get_binary, execute_json_request,
};
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{ImageExtras, ImageGenerationCapability, ModelMetadata, ProviderCapabilities};
use crate::types::{
    GeneratedImage, HttpConfig, HttpResponseInfo, ImageEditInput, ImageEditRequest,
    ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest, Warning,
};
use base64::Engine;
use reqwest::header::{CONTENT_TYPE, HeaderMap};
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::embedding::EmbeddingModel as FamilyEmbeddingModel;
use siumai_core::image::ImageModel as FamilyImageModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_provider_openai_compatible::providers::openai_compatible::fireworks as fireworks_models;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

const DEFAULT_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";
const DEFAULT_TEXT_MODEL: &str = fireworks_models::CHAT;
const DEFAULT_IMAGE_MODEL: &str = fireworks_models::IMAGE;
const DEFAULT_EDIT_MODEL: &str = fireworks_models::image::FLUX_KONTEXT_PRO;
const DEFAULT_POLL_INTERVAL_MILLIS: u64 = 500;
const DEFAULT_POLL_TIMEOUT_MILLIS: u64 = 120_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FireworksImageRouteKind {
    Workflows,
    WorkflowsAsync,
    ImageGeneration,
}

#[derive(Clone, Copy, Debug)]
struct FireworksImageBackendConfig {
    route_kind: FireworksImageRouteKind,
    supports_size: bool,
    supports_editing: bool,
}

impl FireworksImageBackendConfig {
    const fn workflows() -> Self {
        Self {
            route_kind: FireworksImageRouteKind::Workflows,
            supports_size: false,
            supports_editing: false,
        }
    }

    const fn workflows_async() -> Self {
        Self {
            route_kind: FireworksImageRouteKind::WorkflowsAsync,
            supports_size: false,
            supports_editing: true,
        }
    }

    const fn image_generation() -> Self {
        Self {
            route_kind: FireworksImageRouteKind::ImageGeneration,
            supports_size: true,
            supports_editing: false,
        }
    }
}

fn fireworks_capabilities() -> ProviderCapabilities {
    crate::native_provider_metadata::native_providers_metadata()
        .into_iter()
        .find(|meta| meta.id == ids::FIREWORKS)
        .map(|meta| meta.capabilities)
        .unwrap_or_else(|| {
            ProviderCapabilities::new()
                .with_chat()
                .with_completion()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_image_generation()
                .with_transcription()
        })
}

fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    if let Some(api_key) = &ctx.api_key {
        return Ok(api_key.clone());
    }

    std::env::var("FIREWORKS_API_KEY").map_err(|_| {
        LlmError::ConfigurationError(
            "Missing FIREWORKS_API_KEY or explicit api_key in BuildContext".to_string(),
        )
    })
}

fn resolve_root_base_url(ctx: &BuildContext) -> String {
    crate::utils::builder_helpers::resolve_base_url(ctx.base_url.clone(), DEFAULT_BASE_URL)
}

fn known_backend_config_for_model(model_id: &str) -> Option<FireworksImageBackendConfig> {
    match model_id {
        fireworks_models::image::FLUX_1_DEV_FP8 | fireworks_models::image::FLUX_1_SCHNELL_FP8 => {
            Some(FireworksImageBackendConfig::workflows())
        }
        fireworks_models::image::FLUX_KONTEXT_PRO | fireworks_models::image::FLUX_KONTEXT_MAX => {
            Some(FireworksImageBackendConfig::workflows_async())
        }
        fireworks_models::image::PLAYGROUND_V2_5_1024PX_AESTHETIC
        | fireworks_models::image::JAPANESE_STABLE_DIFFUSION_XL
        | fireworks_models::image::PLAYGROUND_V2_1024PX_AESTHETIC
        | fireworks_models::image::SSD_1B
        | fireworks_models::image::STABLE_DIFFUSION_XL_1024_V1_0 => {
            Some(FireworksImageBackendConfig::image_generation())
        }
        _ => None,
    }
}

fn backend_config_for_model(model_id: &str) -> FireworksImageBackendConfig {
    known_backend_config_for_model(model_id).unwrap_or_else(FireworksImageBackendConfig::workflows)
}

fn default_image_model() -> &'static str {
    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_image_model(
        ids::FIREWORKS,
    )
    .unwrap_or(DEFAULT_IMAGE_MODEL)
}

fn default_text_model() -> &'static str {
    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_chat_model(
        ids::FIREWORKS,
    )
    .unwrap_or(DEFAULT_TEXT_MODEL)
}

fn resolve_generation_model(request_model: Option<&str>, current_model: &str) -> String {
    match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ if known_backend_config_for_model(current_model).is_some() => current_model.to_string(),
        _ if current_model.trim().is_empty() || current_model == default_text_model() => {
            default_image_model().to_string()
        }
        _ => default_image_model().to_string(),
    }
}

fn resolve_edit_model(
    request_model: Option<&str>,
    current_model: &str,
) -> Result<String, LlmError> {
    let model = match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ if backend_config_for_model(current_model).supports_editing => current_model.to_string(),
        _ => DEFAULT_EDIT_MODEL.to_string(),
    };

    if backend_config_for_model(&model).supports_editing {
        Ok(model)
    } else {
        Err(LlmError::UnsupportedOperation(format!(
            "Fireworks image editing is only supported on Kontext models; got `{model}`"
        )))
    }
}

async fn build_text_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<
    siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    LlmError,
> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let common_params =
        crate::utils::builder_helpers::resolve_common_params(ctx.common_params.clone(), model_id);

    crate::registry::factory::build_openai_compatible_typed_client(
        ids::FIREWORKS.to_string(),
        resolve_api_key(ctx)?,
        Some(resolve_root_base_url(ctx)),
        http_client,
        common_params,
        ctx.reasoning_enabled,
        ctx.reasoning_budget,
        http_config,
        None,
        None,
        ctx.tracing_config.clone(),
        ctx.retry_options.clone(),
        ctx.http_interceptors.clone(),
        ctx.model_middlewares.clone(),
        ctx.http_transport.clone(),
    )
    .await
}

#[derive(Clone, Copy, Default)]
struct FireworksImageSpec;

impl ProviderSpec for FireworksImageSpec {
    fn id(&self) -> &'static str {
        ids::FIREWORKS
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        siumai_protocol_openai::standards::openai::headers::build_openai_compatible_json_headers(
            ctx,
        )
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        if let Ok(json) = serde_json::from_str::<Value>(body_text)
            && let Some(message) = json.get("error").and_then(Value::as_str)
        {
            return Some(LlmError::ApiError {
                code: status,
                message: message.to_string(),
                details: Some(json),
            });
        }

        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            ids::FIREWORKS,
            status,
            body_text,
        )
    }
}

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            Some((key.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect()
}

fn provider_options_object(
    map: &crate::types::ProviderOptionsMap,
) -> Result<Option<&Map<String, Value>>, LlmError> {
    let Some(value) = map.get(ids::FIREWORKS) else {
        return Ok(None);
    };

    value
        .as_object()
        .ok_or_else(|| {
            LlmError::InvalidParameter(
                "providerOptions.fireworks must be a JSON object when provided".to_string(),
            )
        })
        .map(Some)
}

fn merge_object_fields(body: &mut Map<String, Value>, fields: Option<&Map<String, Value>>) {
    let Some(fields) = fields else {
        return;
    };

    for (key, value) in fields {
        body.insert(key.clone(), value.clone());
    }
}

fn split_size(size: Option<&str>) -> Result<Option<(&str, &str)>, LlmError> {
    let Some(size) = size else {
        return Ok(None);
    };

    size.split_once('x')
        .ok_or_else(|| {
            LlmError::InvalidParameter(format!(
                "Invalid Fireworks image size `{size}`; expected WIDTHxHEIGHT"
            ))
        })
        .map(Some)
}

fn warnings_for_generation_request(
    request: &ImageGenerationRequest,
    backend: FireworksImageBackendConfig,
) -> Vec<Warning> {
    let mut warnings = Vec::new();

    if !backend.supports_size && request.size.is_some() {
        warnings.push(Warning::unsupported(
            "size",
            Some("This model does not support the `size` option. Use `aspectRatio` instead."),
        ));
    }

    if backend.supports_size && request.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported(
            "aspectRatio",
            Some("This model does not support the `aspectRatio` option."),
        ));
    }

    warnings
}

fn image_input_to_wire_value(input: &ImageEditInput) -> Result<String, LlmError> {
    match input {
        ImageEditInput::Url { url, .. } => Ok(url.clone()),
        ImageEditInput::File {
            data, media_type, ..
        } => {
            let media_type = media_type
                .clone()
                .unwrap_or_else(|| "image/png".to_string());
            Ok(format!("data:{media_type};base64,{}", data.as_base64()))
        }
    }
}

fn build_generation_body(
    request: &ImageGenerationRequest,
    backend: FireworksImageBackendConfig,
) -> Result<(Value, Vec<Warning>), LlmError> {
    let provider_options = provider_options_object(&request.provider_options_map)?;
    let mut body = Map::new();
    let warnings = warnings_for_generation_request(request, backend);

    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert(
        "samples".to_string(),
        serde_json::json!(request.count.max(1)),
    );
    if let Some(aspect_ratio) = request.aspect_ratio.as_ref() {
        body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some((width, height)) = split_size(request.size.as_deref())? {
        body.insert("width".to_string(), serde_json::json!(width));
        body.insert("height".to_string(), serde_json::json!(height));
    }
    if let Some(seed) = request.seed {
        body.insert("seed".to_string(), serde_json::json!(seed));
    }

    for (key, value) in &request.extra_params {
        body.insert(key.clone(), value.clone());
    }
    merge_object_fields(&mut body, provider_options);

    Ok((Value::Object(body), warnings))
}

fn build_edit_body(
    request: &ImageEditRequest,
    backend: FireworksImageBackendConfig,
) -> Result<(Value, Vec<Warning>), LlmError> {
    if request.images.is_empty() {
        return Err(LlmError::InvalidParameter(
            "Fireworks image edits require at least one input image".to_string(),
        ));
    }

    let provider_options = provider_options_object(&request.provider_options_map)?;
    let mut body = Map::new();
    let mut warnings = Vec::new();

    if !backend.supports_size && request.size.is_some() {
        warnings.push(Warning::unsupported(
            "size",
            Some("This model does not support the `size` option. Use `aspectRatio` instead."),
        ));
    }

    if backend.supports_size && request.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported(
            "aspectRatio",
            Some("This model does not support the `aspectRatio` option."),
        ));
    }

    if request.images.len() > 1 {
        warnings.push(Warning::other(
            "Fireworks only supports a single input image. Additional images are ignored.",
        ));
    }

    if request.mask.is_some() {
        warnings.push(Warning::unsupported(
            "mask",
            Some(
                "Fireworks Kontext models do not support explicit masks. Use the prompt to describe the areas to edit.",
            ),
        ));
    }

    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert(
        "samples".to_string(),
        serde_json::json!(request.count.unwrap_or(1).max(1)),
    );
    body.insert(
        "input_image".to_string(),
        serde_json::json!(image_input_to_wire_value(&request.images[0])?),
    );
    if let Some(aspect_ratio) = request.aspect_ratio.as_ref() {
        body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some((width, height)) = split_size(request.size.as_deref())? {
        body.insert("width".to_string(), serde_json::json!(width));
        body.insert("height".to_string(), serde_json::json!(height));
    }
    if let Some(seed) = request.seed {
        body.insert("seed".to_string(), serde_json::json!(seed));
    }

    for (key, value) in &request.extra_params {
        body.insert(key.clone(), value.clone());
    }
    merge_object_fields(&mut body, provider_options);

    Ok((Value::Object(body), warnings))
}

fn media_type_to_format(media_type: &str) -> Option<String> {
    media_type
        .split(';')
        .next()
        .and_then(|value| value.split('/').nth(1))
        .map(ToString::to_string)
}

fn generated_image_from_bytes(bytes: Vec<u8>, media_type: Option<&str>) -> GeneratedImage {
    GeneratedImage {
        url: None,
        b64_json: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
        format: media_type.and_then(media_type_to_format),
        width: None,
        height: None,
        revised_prompt: None,
        metadata: HashMap::new(),
    }
}

fn parse_data_url(url: &str) -> Result<(Vec<u8>, Option<String>), LlmError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(LlmError::InvalidParameter(
            "Expected a Fireworks data URL".to_string(),
        ));
    };
    let Some((meta, data)) = payload.split_once(',') else {
        return Err(LlmError::InvalidParameter(
            "Invalid Fireworks data URL".to_string(),
        ));
    };
    let Some(media_type) = meta.strip_suffix(";base64") else {
        return Err(LlmError::InvalidParameter(
            "Fireworks data URLs must use base64 encoding".to_string(),
        ));
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|err| {
            LlmError::InvalidParameter(format!(
                "Invalid base64 payload in Fireworks data URL: {err}"
            ))
        })?;
    let media_type = (!media_type.is_empty()).then_some(media_type.to_string());
    Ok((bytes, media_type))
}

#[derive(Debug, Deserialize)]
struct FireworksAsyncSubmitResponse {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct FireworksAsyncPollResult {
    sample: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FireworksAsyncPollResponse {
    status: String,
    result: Option<FireworksAsyncPollResult>,
}

#[derive(Clone)]
struct FireworksImageClient {
    model_id: String,
    root_base_url: String,
    wiring: HttpExecutionWiring,
}

impl FireworksImageClient {
    fn from_text_client(
        text_client: &siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    ) -> Self {
        let root_base_url = text_client.provider_context().base_url;
        let mut wiring = HttpExecutionWiring::new(
            ids::FIREWORKS,
            text_client.http_client(),
            text_client.provider_context(),
        )
        .with_interceptors(text_client.http_interceptors())
        .with_retry_options(text_client.retry_options());
        if let Some(transport) = text_client.http_transport() {
            wiring = wiring.with_transport(transport);
        }

        Self {
            model_id: text_client.model().to_string(),
            root_base_url,
            wiring,
        }
    }

    fn execution_config(&self) -> crate::execution::executors::common::HttpExecutionConfig {
        self.wiring.config(Arc::new(FireworksImageSpec))
    }

    fn image_url(&self, model: &str) -> String {
        match backend_config_for_model(model).route_kind {
            FireworksImageRouteKind::ImageGeneration => {
                format!(
                    "{}/image_generation/{}",
                    self.root_base_url.trim_end_matches('/'),
                    model
                )
            }
            FireworksImageRouteKind::WorkflowsAsync => {
                format!(
                    "{}/workflows/{}",
                    self.root_base_url.trim_end_matches('/'),
                    model
                )
            }
            FireworksImageRouteKind::Workflows => {
                format!(
                    "{}/workflows/{}/text_to_image",
                    self.root_base_url.trim_end_matches('/'),
                    model
                )
            }
        }
    }

    fn poll_url(&self, model: &str) -> String {
        format!(
            "{}/workflows/{}/get_result",
            self.root_base_url.trim_end_matches('/'),
            model
        )
    }

    fn poll_interval(&self) -> Duration {
        Duration::from_millis(DEFAULT_POLL_INTERVAL_MILLIS)
    }

    fn poll_timeout(&self) -> Duration {
        Duration::from_millis(DEFAULT_POLL_TIMEOUT_MILLIS)
    }

    async fn download_polled_image(
        &self,
        url: &str,
        per_request_http_config: Option<&HttpConfig>,
    ) -> Result<(GeneratedImage, Option<HashMap<String, String>>), LlmError> {
        if url.starts_with("data:") {
            let (bytes, media_type) = parse_data_url(url)?;
            return Ok((
                generated_image_from_bytes(bytes, media_type.as_deref()),
                None,
            ));
        }

        let execution = self.execution_config();
        match execute_get_binary(&execution, url, per_request_http_config).await {
            Ok(result) => Ok((
                generated_image_from_bytes(
                    result.bytes,
                    result
                        .headers
                        .get(CONTENT_TYPE)
                        .and_then(|value| value.to_str().ok()),
                ),
                Some(headers_to_map(&result.headers)),
            )),
            Err(LlmError::UnsupportedOperation(_)) => {
                let headers = FireworksImageSpec.build_headers(&self.wiring.provider_context)?;
                let headers = if let Some(req_http) = per_request_http_config {
                    FireworksImageSpec.merge_request_headers(headers, &req_http.headers)
                } else {
                    headers
                };
                let mut request = self.wiring.http_client.get(url).headers(headers);
                if let Some(req_http) = per_request_http_config
                    && let Some(timeout) = req_http.timeout
                {
                    request = request.timeout(timeout);
                }
                let response = request.send().await.map_err(|err| {
                    LlmError::HttpError(format!(
                        "Failed to download Fireworks polled image result: {err}"
                    ))
                })?;
                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(LlmError::ApiError {
                        code: status.as_u16(),
                        message: format!(
                            "Failed to download Fireworks polled image result from {url}"
                        ),
                        details: Some(serde_json::json!({ "url": url, "body": body })),
                    });
                }

                let response_headers = response.headers().clone();
                let media_type = response_headers
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .map(ToString::to_string);
                let bytes = response.bytes().await.map_err(|err| {
                    LlmError::HttpError(format!(
                        "Failed to read Fireworks polled image result bytes: {err}"
                    ))
                })?;

                Ok((
                    generated_image_from_bytes(bytes.to_vec(), media_type.as_deref()),
                    Some(headers_to_map(&response_headers)),
                ))
            }
            Err(err) => Err(err),
        }
    }

    async fn generate_or_edit_with_body(
        &self,
        model: &str,
        body: Value,
        warnings: Vec<Warning>,
        per_request_http_config: Option<&HttpConfig>,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let config = backend_config_for_model(model);
        match config.route_kind {
            FireworksImageRouteKind::WorkflowsAsync => {
                self.generate_or_edit_async(model, body, warnings, per_request_http_config)
                    .await
            }
            FireworksImageRouteKind::Workflows | FireworksImageRouteKind::ImageGeneration => {
                let result = execute_bytes_request(
                    &self.execution_config(),
                    &self.image_url(model),
                    HttpBody::Json(body),
                    per_request_http_config,
                )
                .await?;

                Ok(ImageGenerationResponse {
                    images: vec![generated_image_from_bytes(
                        result.bytes,
                        result
                            .headers
                            .get(CONTENT_TYPE)
                            .and_then(|value| value.to_str().ok()),
                    )],
                    metadata: HashMap::new(),
                    warnings: (!warnings.is_empty()).then_some(warnings),
                    response: Some(HttpResponseInfo {
                        timestamp: chrono::Utc::now(),
                        model_id: Some(model.to_string()),
                        headers: headers_to_map(&result.headers),
                        body: None,
                    }),
                })
            }
        }
    }

    async fn generate_or_edit_async(
        &self,
        model: &str,
        body: Value,
        warnings: Vec<Warning>,
        per_request_http_config: Option<&HttpConfig>,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let submit_result = execute_json_request(
            &self.execution_config(),
            &self.image_url(model),
            HttpBody::Json(body),
            per_request_http_config,
            false,
        )
        .await?;
        let submit: FireworksAsyncSubmitResponse = serde_json::from_value(submit_result.json)
            .map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse Fireworks async image submit response: {err}"
                ))
            })?;

        let deadline = Instant::now() + self.poll_timeout();
        loop {
            if Instant::now() >= deadline {
                return Err(LlmError::TimeoutError(format!(
                    "Fireworks image generation timed out after {}ms",
                    self.poll_timeout().as_millis()
                )));
            }

            let poll_result = execute_json_request(
                &self.execution_config(),
                &self.poll_url(model),
                HttpBody::Json(serde_json::json!({ "id": submit.request_id })),
                per_request_http_config,
                false,
            )
            .await?;
            let poll: FireworksAsyncPollResponse = serde_json::from_value(poll_result.json)
                .map_err(|err| {
                    LlmError::ParseError(format!(
                        "Failed to parse Fireworks async image poll response: {err}"
                    ))
                })?;

            match poll.status.as_str() {
                "Ready" => {
                    let Some(sample) = poll.result.and_then(|result| result.sample) else {
                        return Err(LlmError::ParseError(
                            "Fireworks poll response is Ready but missing result.sample"
                                .to_string(),
                        ));
                    };
                    let (image, response_headers) = self
                        .download_polled_image(&sample, per_request_http_config)
                        .await?;
                    return Ok(ImageGenerationResponse {
                        images: vec![image],
                        metadata: HashMap::new(),
                        warnings: (!warnings.is_empty()).then_some(warnings),
                        response: Some(HttpResponseInfo {
                            timestamp: chrono::Utc::now(),
                            model_id: Some(model.to_string()),
                            headers: response_headers
                                .unwrap_or_else(|| headers_to_map(&poll_result.headers)),
                            body: None,
                        }),
                    });
                }
                "Error" | "Failed" => {
                    return Err(LlmError::ApiError {
                        code: 500,
                        message: format!(
                            "Fireworks image generation failed with status: {}",
                            poll.status
                        ),
                        details: None,
                    });
                }
                _ => sleep(self.poll_interval()).await,
            }
        }
    }
}

impl std::fmt::Debug for FireworksImageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FireworksImageClient")
            .field("provider_id", &ids::FIREWORKS)
            .field("model_id", &self.model_id)
            .field("root_base_url", &self.root_base_url)
            .finish()
    }
}

impl ModelMetadata for FireworksImageClient {
    fn provider_id(&self) -> &str {
        ids::FIREWORKS
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl crate::client::LlmClient for FireworksImageClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::FIREWORKS)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![resolve_generation_model(None, &self.model_id)]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::client::LlmClient> {
        Box::new(self.clone())
    }
}

#[async_trait::async_trait]
impl ImageGenerationCapability for FireworksImageClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_generation_model(request.model.as_deref(), &self.model_id);
        let config = backend_config_for_model(&model);
        let (body, warnings) = build_generation_body(&request, config)?;
        self.generate_or_edit_with_body(&model, body, warnings, request.http_config.as_ref())
            .await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        Some(1)
    }
}

#[async_trait::async_trait]
impl ImageExtras for FireworksImageClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_edit_model(request.model.as_deref(), &self.model_id)?;
        let config = backend_config_for_model(&model);
        let (body, warnings) = build_edit_body(&request, config)?;
        self.generate_or_edit_with_body(&model, body, warnings, request.http_config.as_ref())
            .await
    }

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Fireworks does not currently expose image variations on the provider-owned path"
                .to_string(),
        ))
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }
}

#[derive(Clone)]
struct FireworksUnifiedClient {
    text_client:
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    image_client: FireworksImageClient,
}

impl std::fmt::Debug for FireworksUnifiedClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FireworksUnifiedClient")
            .field("provider_id", &ids::FIREWORKS)
            .field("text_model", &self.text_client.model_id())
            .field("image_model", &self.image_client.model_id())
            .finish()
    }
}

impl LlmClient for FireworksUnifiedClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::FIREWORKS)
    }

    fn supported_models(&self) -> Vec<String> {
        let mut models = self.text_client.supported_models();
        for model in self.image_client.supported_models() {
            if !models.iter().any(|existing| existing == &model) {
                models.push(model);
            }
        }
        models
    }

    fn capabilities(&self) -> ProviderCapabilities {
        fireworks_capabilities()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn crate::traits::ChatCapability> {
        self.text_client.as_chat_capability()
    }

    fn as_completion_capability(&self) -> Option<&dyn crate::traits::CompletionCapability> {
        self.text_client.as_completion_capability()
    }

    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        self.text_client.as_embedding_capability()
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        self.text_client.as_embedding_extensions()
    }

    fn as_audio_capability(&self) -> Option<&dyn crate::traits::AudioCapability> {
        self.text_client.as_audio_capability()
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        self.text_client.as_speech_capability()
    }

    fn as_speech_extras(&self) -> Option<&dyn crate::traits::SpeechExtras> {
        self.text_client.as_speech_extras()
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        self.text_client.as_transcription_capability()
    }

    fn as_transcription_extras(&self) -> Option<&dyn crate::traits::TranscriptionExtras> {
        self.text_client.as_transcription_extras()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(&self.image_client)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(&self.image_client)
    }
}

/// Fireworks provider factory.
#[cfg(feature = "openai")]
pub struct FireworksProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for FireworksProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        fireworks_capabilities()
    }

    async fn compat_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.compat_language_client_with_ctx(model_id, &ctx).await
    }

    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        let image_client = FireworksImageClient::from_text_client(&text_client);
        Ok(Arc::new(FireworksUnifiedClient {
            text_client,
            image_client,
        }))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn compat_completion_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn compat_image_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(FireworksImageClient::from_text_client(
            &text_client,
        )))
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(FireworksImageClient::from_text_client(
            &text_client,
        )))
    }

    async fn compat_transcription_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn transcription_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyTranscriptionModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::FIREWORKS)
    }
}
