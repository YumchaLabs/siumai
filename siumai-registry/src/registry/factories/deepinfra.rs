//! DeepInfra unified provider factory.
//!
//! AI SDK exposes DeepInfra as a first-class provider with:
//! - OpenAI-compatible chat/completion/embedding
//! - provider-owned image generation/edit routes

use super::*;
use crate::core::{ProviderContext, ProviderSpec};
use crate::execution::executors::common::{
    HttpBody, execute_json_request, execute_multipart_request,
};
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{ImageExtras, ImageGenerationCapability, ModelMetadata, ProviderCapabilities};
use crate::types::{
    GeneratedImage, HttpResponseInfo, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, Warning,
};
use base64::Engine;
use reqwest::header::{CONTENT_TYPE, HeaderMap};
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::embedding::EmbeddingModel as FamilyEmbeddingModel;
use siumai_core::image::ImageModel as FamilyImageModel;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_ROOT_BASE_URL: &str = "https://api.deepinfra.com/v1";
const DEFAULT_TEXT_MODEL: &str = "meta-llama/Llama-3.3-70B-Instruct";
const DEFAULT_IMAGE_MODEL: &str = "black-forest-labs/FLUX-1-schnell";

fn deepinfra_capabilities() -> ProviderCapabilities {
    crate::native_provider_metadata::native_providers_metadata()
        .into_iter()
        .find(|meta| meta.id == ids::DEEPINFRA)
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
        })
}

fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    if let Some(api_key) = &ctx.api_key {
        return Ok(api_key.clone());
    }

    std::env::var("DEEPINFRA_API_KEY").map_err(|_| {
        LlmError::ConfigurationError(
            "Missing DEEPINFRA_API_KEY or explicit api_key in BuildContext".to_string(),
        )
    })
}

fn normalize_root_base_url(base_url: &str) -> String {
    siumai_protocol_openai::standards::openai::compat::base_url::deepinfra_root_base_url(base_url)
}

fn resolve_root_base_url(ctx: &BuildContext) -> String {
    normalize_root_base_url(&crate::utils::builder_helpers::resolve_base_url(
        ctx.base_url.clone(),
        DEFAULT_ROOT_BASE_URL,
    ))
}

fn text_base_url(root_base_url: &str) -> String {
    siumai_protocol_openai::standards::openai::compat::base_url::deepinfra_text_base_url(
        root_base_url,
    )
}

fn inference_base_url(root_base_url: &str) -> String {
    format!("{}/inference", root_base_url.trim_end_matches('/'))
}

fn resolve_image_fallback_model(model_id: &str) -> String {
    let image_default =
        siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_image_model(
            ids::DEEPINFRA,
        )
        .unwrap_or(DEFAULT_IMAGE_MODEL);
    let chat_default =
        siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_chat_model(
            ids::DEEPINFRA,
        )
        .unwrap_or(DEFAULT_TEXT_MODEL);

    if model_id.trim().is_empty() || model_id == chat_default {
        image_default.to_string()
    } else {
        model_id.to_string()
    }
}

fn resolve_image_model(request_model: Option<&str>, fallback_model: &str) -> String {
    match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ => resolve_image_fallback_model(fallback_model),
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
    let root_base_url = resolve_root_base_url(ctx);

    crate::registry::factory::build_openai_compatible_typed_client(
        ids::DEEPINFRA.to_string(),
        resolve_api_key(ctx)?,
        Some(text_base_url(&root_base_url)),
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
struct DeepInfraImageSpec;

impl ProviderSpec for DeepInfraImageSpec {
    fn id(&self) -> &'static str {
        ids::DEEPINFRA
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
            && let Some(message) = json
                .get("detail")
                .and_then(|detail| detail.get("error"))
                .and_then(Value::as_str)
        {
            return Some(LlmError::ApiError {
                code: status,
                message: message.to_string(),
                details: Some(json),
            });
        }

        siumai_protocol_openai::standards::openai::errors::classify_openai_compatible_http_error(
            ids::DEEPINFRA,
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
    let Some(value) = map.get(ids::DEEPINFRA) else {
        return Ok(None);
    };

    value
        .as_object()
        .ok_or_else(|| {
            LlmError::InvalidParameter(
                "providerOptions.deepinfra must be a JSON object when provided".to_string(),
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
                "Invalid DeepInfra image size `{size}`; expected WIDTHxHEIGHT"
            ))
        })
        .map(Some)
}

fn build_generation_body(
    request: &ImageGenerationRequest,
) -> Result<(Value, Vec<Warning>), LlmError> {
    let provider_options = provider_options_object(&request.provider_options_map)?;
    let mut body = Map::new();
    let warnings = Vec::new();

    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert(
        "num_images".to_string(),
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

fn multipart_scalar_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn media_type_extension(media_type: &str) -> &'static str {
    match media_type {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    }
}

fn parse_data_url(url: &str) -> Result<(Vec<u8>, Option<String>), LlmError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(LlmError::InvalidParameter(
            "Expected a data URL for DeepInfra image input".to_string(),
        ));
    };
    let Some((meta, data)) = payload.split_once(',') else {
        return Err(LlmError::InvalidParameter(
            "Invalid DeepInfra image data URL".to_string(),
        ));
    };

    let Some(meta) = meta.strip_suffix(";base64") else {
        return Err(LlmError::InvalidParameter(
            "DeepInfra image data URLs must use base64 encoding".to_string(),
        ));
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|err| {
            LlmError::InvalidParameter(format!(
                "Invalid base64 payload in DeepInfra image data URL: {err}"
            ))
        })?;
    let media_type = (!meta.is_empty()).then_some(meta.to_string());
    Ok((bytes, media_type))
}

async fn load_image_part_bytes(
    http_client: &reqwest::Client,
    input: &ImageEditInput,
) -> Result<(Vec<u8>, String), LlmError> {
    match input {
        ImageEditInput::File {
            data, media_type, ..
        } => {
            let bytes = data.as_bytes().map_err(|err| {
                LlmError::InvalidParameter(format!("Invalid DeepInfra image input data: {err}"))
            })?;
            Ok((
                bytes,
                media_type
                    .clone()
                    .unwrap_or_else(|| "image/png".to_string()),
            ))
        }
        ImageEditInput::Url { url, .. } => {
            if url.starts_with("data:") {
                let (bytes, media_type) = parse_data_url(url)?;
                return Ok((bytes, media_type.unwrap_or_else(|| "image/png".to_string())));
            }

            let response = http_client.get(url).send().await.map_err(|err| {
                LlmError::HttpError(format!("Failed to download image input: {err}"))
            })?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(LlmError::ApiError {
                    code: status.as_u16(),
                    message: format!("Failed to download DeepInfra image input from {url}"),
                    details: Some(serde_json::json!({ "url": url, "body": body })),
                });
            }

            let media_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .filter(|value| !value.is_empty())
                .unwrap_or("image/png")
                .to_string();
            let bytes = response.bytes().await.map_err(|err| {
                LlmError::HttpError(format!("Failed to read image input bytes: {err}"))
            })?;
            Ok((bytes.to_vec(), media_type))
        }
    }
}

#[derive(Clone)]
struct PreparedImagePart {
    field_name: &'static str,
    index: usize,
    bytes: Vec<u8>,
    media_type: String,
}

impl PreparedImagePart {
    fn into_part(self) -> Result<reqwest::multipart::Part, LlmError> {
        let file_name = format!(
            "{}-{}.{}",
            self.field_name,
            self.index,
            media_type_extension(&self.media_type)
        );
        reqwest::multipart::Part::bytes(self.bytes)
            .file_name(file_name)
            .mime_str(&self.media_type)
            .map_err(|err| {
                LlmError::InvalidParameter(format!(
                    "Invalid DeepInfra multipart media type `{}`: {err}",
                    self.media_type
                ))
            })
    }
}

async fn prepare_image_part(
    http_client: &reqwest::Client,
    input: &ImageEditInput,
    field_name: &str,
    index: usize,
) -> Result<PreparedImagePart, LlmError> {
    let (bytes, media_type) = load_image_part_bytes(http_client, input).await?;
    Ok(PreparedImagePart {
        field_name: match field_name {
            "image" => "image",
            "mask" => "mask",
            _ => "image",
        },
        index,
        bytes,
        media_type,
    })
}

fn data_url_payload(value: &str) -> Option<(Option<String>, String)> {
    let payload = value.strip_prefix("data:")?;
    let (meta, data) = payload.split_once(',')?;
    let media_type = meta
        .strip_suffix(";base64")
        .map(|value| (!value.is_empty()).then_some(value.to_string()))?;
    Some((media_type, data.to_string()))
}

fn generated_image_from_string(value: String) -> GeneratedImage {
    if let Some((media_type, b64_json)) = data_url_payload(&value) {
        return GeneratedImage {
            url: None,
            b64_json: Some(b64_json),
            format: media_type
                .as_deref()
                .and_then(|media_type| media_type.split('/').nth(1))
                .map(ToString::to_string),
            width: None,
            height: None,
            revised_prompt: None,
            metadata: HashMap::new(),
        };
    }

    if value.starts_with("http://") || value.starts_with("https://") {
        return GeneratedImage {
            url: Some(value),
            b64_json: None,
            format: None,
            width: None,
            height: None,
            revised_prompt: None,
            metadata: HashMap::new(),
        };
    }

    GeneratedImage {
        url: None,
        b64_json: Some(value),
        format: None,
        width: None,
        height: None,
        revised_prompt: None,
        metadata: HashMap::new(),
    }
}

#[derive(Debug, Deserialize)]
struct DeepInfraGenerationResponse {
    images: Vec<String>,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct DeepInfraEditImageData {
    b64_json: String,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct DeepInfraEditResponse {
    data: Vec<DeepInfraEditImageData>,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[derive(Clone)]
struct DeepInfraImageClient {
    model_id: String,
    root_base_url: String,
    wiring: HttpExecutionWiring,
}

impl DeepInfraImageClient {
    fn from_text_client(
        text_client: &siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    ) -> Self {
        let root_base_url = normalize_root_base_url(&text_client.provider_context().base_url);
        let mut wiring = HttpExecutionWiring::new(
            ids::DEEPINFRA,
            text_client.http_client(),
            text_client.provider_context(),
        )
        .with_interceptors(text_client.http_interceptors())
        .with_retry_options(text_client.retry_options());
        wiring.provider_context.base_url = root_base_url.clone();
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
        self.wiring.config(Arc::new(DeepInfraImageSpec))
    }

    fn generation_url(&self, model: &str) -> String {
        format!("{}/{}", inference_base_url(&self.root_base_url), model)
    }

    fn edit_url(&self) -> String {
        format!(
            "{}/openai/images/edits",
            self.root_base_url.trim_end_matches('/')
        )
    }
}

impl std::fmt::Debug for DeepInfraImageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepInfraImageClient")
            .field("provider_id", &ids::DEEPINFRA)
            .field("model_id", &self.model_id)
            .field("root_base_url", &self.root_base_url)
            .finish()
    }
}

impl ModelMetadata for DeepInfraImageClient {
    fn provider_id(&self) -> &str {
        ids::DEEPINFRA
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl crate::client::LlmClient for DeepInfraImageClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::DEEPINFRA)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![resolve_image_fallback_model(&self.model_id)]
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
impl ImageGenerationCapability for DeepInfraImageClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_image_model(request.model.as_deref(), &self.model_id);
        let (body, warnings) = build_generation_body(&request)?;
        let result = execute_json_request(
            &self.execution_config(),
            &self.generation_url(&model),
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;
        let response: DeepInfraGenerationResponse =
            serde_json::from_value(result.json).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse DeepInfra image generation response: {err}"
                ))
            })?;

        Ok(ImageGenerationResponse {
            images: response
                .images
                .into_iter()
                .map(generated_image_from_string)
                .collect(),
            metadata: response.extra_fields,
            warnings: (!warnings.is_empty()).then_some(warnings),
            response: Some(HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: Some(model),
                headers: headers_to_map(&result.headers),
                body: None,
            }),
        })
    }

    fn max_images_per_call(&self) -> Option<u32> {
        Some(1)
    }
}

#[async_trait::async_trait]
impl ImageExtras for DeepInfraImageClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if request.images.is_empty() {
            return Err(LlmError::InvalidParameter(
                "DeepInfra image edits require at least one input image".to_string(),
            ));
        }

        let model = resolve_image_model(request.model.as_deref(), &self.model_id);
        let provider_options = provider_options_object(&request.provider_options_map)?
            .cloned()
            .unwrap_or_default();
        let execution_config = self.execution_config();
        let http_client = self.wiring.http_client.clone();
        let mut prepared_images = Vec::with_capacity(request.images.len());
        for (index, image) in request.images.iter().enumerate() {
            prepared_images.push(prepare_image_part(&http_client, image, "image", index).await?);
        }
        let prepared_mask = if let Some(mask) = request.mask.as_ref() {
            Some(prepare_image_part(&http_client, mask, "mask", 0).await?)
        } else {
            None
        };

        let request_clone = request.clone();
        let model_clone = model.clone();
        let prepared_images_clone = prepared_images.clone();
        let prepared_mask_clone = prepared_mask.clone();
        let build_form = move || {
            let provider_options = provider_options.clone();
            let request = request_clone.clone();
            let model = model_clone.clone();
            let prepared_images = prepared_images_clone.clone();
            let prepared_mask = prepared_mask_clone.clone();

            let mut form = reqwest::multipart::Form::new()
                .text("model", model)
                .text("prompt", request.prompt.clone())
                .text("n", request.count.unwrap_or(1).max(1).to_string());
            if let Some(size) = request.size.clone() {
                form = form.text("size", size);
            }
            if let Some(aspect_ratio) = request.aspect_ratio.clone() {
                form = form.text("aspect_ratio", aspect_ratio);
            }
            if let Some(seed) = request.seed {
                form = form.text("seed", seed.to_string());
            }

            for image in prepared_images {
                form = form.part("image", image.into_part()?);
            }

            if let Some(mask) = prepared_mask {
                form = form.part("mask", mask.into_part()?);
            }

            for (key, value) in &request.extra_params {
                form = form.text(key.clone(), multipart_scalar_string(value));
            }
            for (key, value) in &provider_options {
                form = form.text(key.clone(), multipart_scalar_string(value));
            }

            Ok::<_, LlmError>(form)
        };

        let result = execute_multipart_request(
            &execution_config,
            &self.edit_url(),
            build_form,
            request.http_config.as_ref(),
        )
        .await?;
        let response: DeepInfraEditResponse =
            serde_json::from_value(result.json).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse DeepInfra image edit response: {err}"
                ))
            })?;

        Ok(ImageGenerationResponse {
            images: response
                .data
                .into_iter()
                .map(|item| GeneratedImage {
                    url: None,
                    b64_json: Some(item.b64_json),
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: item.extra_fields,
                })
                .collect(),
            metadata: response.extra_fields,
            warnings: None,
            response: Some(HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: Some(model),
                headers: headers_to_map(&result.headers),
                body: None,
            }),
        })
    }

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepInfra does not currently expose image variations on the provider-owned path"
                .to_string(),
        ))
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string(), "url".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }
}

#[derive(Clone)]
struct DeepInfraCompatCompositeClient {
    text_client:
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    image_client: DeepInfraImageClient,
}

impl std::fmt::Debug for DeepInfraCompatCompositeClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepInfraCompatCompositeClient")
            .field("provider_id", &ids::DEEPINFRA)
            .field("text_model", &self.text_client.model_id())
            .field("image_model", &self.image_client.model_id())
            .finish()
    }
}

impl crate::client::LlmClient for DeepInfraCompatCompositeClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::DEEPINFRA)
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
        deepinfra_capabilities()
    }

    fn as_chat_capability(&self) -> Option<&dyn crate::traits::ChatCapability> {
        self.text_client.as_chat_capability()
    }

    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        self.text_client.as_embedding_capability()
    }

    fn as_completion_capability(&self) -> Option<&dyn crate::traits::CompletionCapability> {
        self.text_client.as_completion_capability()
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        self.text_client.as_embedding_extensions()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(&self.image_client)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(&self.image_client)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::client::LlmClient> {
        Box::new(self.clone())
    }
}

/// DeepInfra provider factory.
#[cfg(feature = "deepinfra")]
pub struct DeepInfraProviderFactory;

#[cfg(feature = "deepinfra")]
#[async_trait::async_trait]
impl ProviderFactory for DeepInfraProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        deepinfra_capabilities()
    }

    async fn compat_language_client(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn crate::client::LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.compat_language_client_with_ctx(model_id, &ctx).await
    }

    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn crate::client::LlmClient>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        let image_client = DeepInfraImageClient::from_text_client(&text_client);
        Ok(Arc::new(DeepInfraCompatCompositeClient {
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
    ) -> Result<Arc<dyn crate::client::LlmClient>, LlmError> {
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
    ) -> Result<Arc<dyn crate::client::LlmClient>, LlmError> {
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
    ) -> Result<Arc<dyn crate::client::LlmClient>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(DeepInfraImageClient::from_text_client(
            &text_client,
        )))
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let text_client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(DeepInfraImageClient::from_text_client(
            &text_client,
        )))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::DEEPINFRA)
    }
}
