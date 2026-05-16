//! TogetherAI provider factory.
//!
//! AI SDK exposes TogetherAI as a single provider surface:
//! - OpenAI-compatible chat/completion/embedding/audio families
//! - provider-owned image + rerank
//!
//! Siumai keeps the provider-owned rerank client in `siumai-provider-togetherai`, reuses the
//! shared OpenAI-compatible runtime for text/audio, and owns the image family here under the
//! canonical `togetherai` provider id.

use super::*;
use crate::core::{ProviderContext, ProviderSpec};
use crate::execution::executors::common::{HttpBody, execute_json_request};
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider::ids;
use crate::registry::factories::OpenAICompatibleProviderFactory;
use crate::text::LanguageModel as FamilyLanguageModel;
use crate::traits::{ImageExtras, ImageGenerationCapability, ModelMetadata, ProviderCapabilities};
use crate::types::{
    GeneratedImage, HttpResponseInfo, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, Warning,
};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::embedding::EmbeddingModel as FamilyEmbeddingModel;
use siumai_core::image::ImageModel as FamilyImageModel;
use siumai_core::rerank::RerankingModel as FamilyRerankingModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_BASE_URL: &str = "https://api.together.xyz/v1";
const DEFAULT_TEXT_MODEL: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
const DEFAULT_IMAGE_MODEL: &str = "black-forest-labs/FLUX.1-schnell";

#[cfg(feature = "togetherai")]
fn togetherai_capabilities() -> ProviderCapabilities {
    OpenAICompatibleProviderFactory::new(ids::TOGETHERAI.to_string())
        .capabilities()
        .with_rerank()
}

#[cfg(feature = "togetherai")]
fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    crate::utils::builder_helpers::get_api_key_with_envs(
        ctx.api_key.clone(),
        ids::TOGETHERAI,
        Some("TOGETHER_API_KEY"),
        &["TOGETHER_AI_API_KEY".to_string()],
    )
    .map_err(|_| {
        LlmError::ConfigurationError(
            "Missing TOGETHER_API_KEY, TOGETHER_AI_API_KEY, or explicit api_key in BuildContext"
                .to_string(),
        )
    })
}

#[cfg(feature = "togetherai")]
fn resolve_root_base_url(ctx: &BuildContext) -> String {
    crate::utils::builder_helpers::resolve_base_url(ctx.base_url.clone(), DEFAULT_BASE_URL)
}

#[cfg(feature = "togetherai")]
fn default_text_model() -> &'static str {
    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_chat_model(
        ids::TOGETHERAI,
    )
    .unwrap_or(DEFAULT_TEXT_MODEL)
}

#[cfg(feature = "togetherai")]
fn default_image_model() -> &'static str {
    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_image_model(
        ids::TOGETHERAI,
    )
    .unwrap_or(DEFAULT_IMAGE_MODEL)
}

#[cfg(feature = "togetherai")]
fn resolve_image_model(request_model: Option<&str>, current_model: &str) -> String {
    match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ if current_model.trim().is_empty() || current_model == default_text_model() => {
            default_image_model().to_string()
        }
        _ => current_model.to_string(),
    }
}

#[cfg(feature = "togetherai")]
fn provider_options_object(
    map: &crate::types::ProviderOptionsMap,
) -> Result<Option<Map<String, Value>>, LlmError> {
    let mut merged = Map::new();
    let mut found = false;

    for provider_id in ["together", ids::TOGETHERAI] {
        let Some(value) = map.get(provider_id) else {
            continue;
        };

        let object = value.as_object().ok_or_else(|| {
            LlmError::InvalidParameter(format!(
                "providerOptions.{provider_id} must be a JSON object when provided"
            ))
        })?;

        for (key, value) in object {
            merged.insert(key.clone(), value.clone());
        }
        found = true;
    }

    Ok(found.then_some(merged))
}

#[cfg(feature = "togetherai")]
fn merge_object_fields(body: &mut Map<String, Value>, fields: Option<Map<String, Value>>) {
    let Some(fields) = fields else {
        return;
    };

    for (key, value) in fields {
        body.insert(key, value);
    }
}

#[cfg(feature = "togetherai")]
fn split_size(size: Option<&str>) -> Result<Option<(u32, u32)>, LlmError> {
    let Some(size) = size else {
        return Ok(None);
    };

    let (width, height) = size.split_once('x').ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "Invalid TogetherAI image size `{size}`; expected WIDTHxHEIGHT"
        ))
    })?;

    let width = width.parse::<u32>().map_err(|err| {
        LlmError::InvalidParameter(format!(
            "Invalid TogetherAI image width `{width}` in size `{size}`: {err}"
        ))
    })?;
    let height = height.parse::<u32>().map_err(|err| {
        LlmError::InvalidParameter(format!(
            "Invalid TogetherAI image height `{height}` in size `{size}`: {err}"
        ))
    })?;

    Ok(Some((width, height)))
}

#[cfg(feature = "togetherai")]
fn image_input_to_wire_value(input: &ImageEditInput) -> String {
    match input {
        ImageEditInput::Url { url, .. } => url.clone(),
        ImageEditInput::File {
            data, media_type, ..
        } => {
            let media_type = media_type
                .clone()
                .unwrap_or_else(|| "image/png".to_string());
            format!("data:{media_type};base64,{}", data.as_base64())
        }
    }
}

#[cfg(feature = "togetherai")]
fn togetherai_aspect_ratio_warning() -> Warning {
    Warning::unsupported(
        "aspectRatio",
        Some("This model does not support the `aspectRatio` option. Use `size` instead."),
    )
}

#[cfg(feature = "togetherai")]
fn build_generation_body(
    request: &ImageGenerationRequest,
    model: &str,
) -> Result<(Value, Vec<Warning>), LlmError> {
    let mut body = Map::new();
    let provider_options = provider_options_object(&request.provider_options_map)?;
    let mut warnings = Vec::new();

    if request.aspect_ratio.is_some() {
        warnings.push(togetherai_aspect_ratio_warning());
    }

    body.insert("model".to_string(), serde_json::json!(model));
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    if let Some(seed) = request.seed {
        body.insert("seed".to_string(), serde_json::json!(seed));
    }
    if request.count > 1 {
        body.insert("n".to_string(), serde_json::json!(request.count));
    }
    if let Some((width, height)) = split_size(request.size.as_deref())? {
        body.insert("width".to_string(), serde_json::json!(width));
        body.insert("height".to_string(), serde_json::json!(height));
    }
    body.insert("response_format".to_string(), serde_json::json!("base64"));

    for (key, value) in &request.extra_params {
        body.insert(key.clone(), value.clone());
    }
    merge_object_fields(&mut body, provider_options);

    Ok((Value::Object(body), warnings))
}

#[cfg(feature = "togetherai")]
fn build_edit_body(
    request: &ImageEditRequest,
    model: &str,
) -> Result<(Value, Vec<Warning>), LlmError> {
    if request.images.is_empty() {
        return Err(LlmError::InvalidParameter(
            "TogetherAI image edits require at least one input image".to_string(),
        ));
    }

    if request.mask.is_some() {
        return Err(LlmError::UnsupportedOperation(
            "Together AI does not support mask-based image editing. Use FLUX Kontext models with a reference image and descriptive prompt instead.".to_string(),
        ));
    }

    let mut body = Map::new();
    let provider_options = provider_options_object(&request.provider_options_map)?;
    let mut warnings = Vec::new();

    if request.aspect_ratio.is_some() {
        warnings.push(togetherai_aspect_ratio_warning());
    }
    if request.images.len() > 1 {
        warnings.push(Warning::other(
            "Together AI only supports a single input image. Additional images are ignored.",
        ));
    }

    body.insert("model".to_string(), serde_json::json!(model));
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));
    body.insert(
        "image_url".to_string(),
        serde_json::json!(image_input_to_wire_value(&request.images[0])),
    );
    if let Some(seed) = request.seed {
        body.insert("seed".to_string(), serde_json::json!(seed));
    }
    if let Some(count) = request.count.filter(|count| *count > 1) {
        body.insert("n".to_string(), serde_json::json!(count));
    }
    if let Some((width, height)) = split_size(request.size.as_deref())? {
        body.insert("width".to_string(), serde_json::json!(width));
        body.insert("height".to_string(), serde_json::json!(height));
    }
    body.insert("response_format".to_string(), serde_json::json!("base64"));

    for (key, value) in &request.extra_params {
        body.insert(key.clone(), value.clone());
    }
    merge_object_fields(&mut body, provider_options);

    Ok((Value::Object(body), warnings))
}

#[cfg(feature = "togetherai")]
#[derive(Clone, Copy, Default)]
struct TogetherAiImageSpec;

#[cfg(feature = "togetherai")]
impl ProviderSpec for TogetherAiImageSpec {
    fn id(&self) -> &'static str {
        ids::TOGETHERAI
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        siumai_protocol_openai::standards::openai::headers::build_openai_compatible_json_headers(
            ctx,
        )
    }
}

#[cfg(feature = "togetherai")]
fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            Some((key.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect()
}

#[cfg(feature = "togetherai")]
#[derive(Debug, Deserialize)]
struct TogetherAiImageResponseItem {
    b64_json: String,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[cfg(feature = "togetherai")]
#[derive(Debug, Deserialize)]
struct TogetherAiImageResponse {
    data: Vec<TogetherAiImageResponseItem>,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[cfg(feature = "togetherai")]
fn generated_image_from_together_item(item: TogetherAiImageResponseItem) -> GeneratedImage {
    GeneratedImage {
        url: None,
        b64_json: Some(item.b64_json),
        format: None,
        width: None,
        height: None,
        revised_prompt: None,
        metadata: item.extra_fields,
    }
}

#[cfg(feature = "togetherai")]
fn build_native_rerank_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<siumai_provider_togetherai::providers::togetherai::TogetherAiClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut cfg = siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::new(
        resolve_api_key(ctx)?,
    )
    .with_base_url(crate::utils::builder_helpers::resolve_base_url(
        ctx.base_url.clone(),
        siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::DEFAULT_BASE_URL,
    ))
    .with_model(model_id)
    .with_http_config(http_config)
    .with_http_interceptors(ctx.http_interceptors.clone());

    if let Some(http_transport) = ctx.http_transport.clone() {
        cfg = cfg.with_http_transport(http_transport);
    }

    let mut client =
        siumai_provider_togetherai::providers::togetherai::TogetherAiClient::with_http_client(
            cfg,
            http_client,
        )?;

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(client)
}

#[cfg(feature = "togetherai")]
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
        ids::TOGETHERAI.to_string(),
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

#[cfg(feature = "togetherai")]
#[derive(Clone)]
struct TogetherAiImageClient {
    model_id: String,
    root_base_url: String,
    wiring: HttpExecutionWiring,
}

#[cfg(feature = "togetherai")]
impl TogetherAiImageClient {
    fn from_text_client(
        text_client: &siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    ) -> Self {
        let root_base_url = text_client.provider_context().base_url;
        let mut wiring = HttpExecutionWiring::new(
            ids::TOGETHERAI,
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
        self.wiring.config(Arc::new(TogetherAiImageSpec))
    }

    fn generation_url(&self) -> String {
        format!(
            "{}/images/generations",
            self.root_base_url.trim_end_matches('/')
        )
    }
}

#[cfg(feature = "togetherai")]
impl std::fmt::Debug for TogetherAiImageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TogetherAiImageClient")
            .field("provider_id", &ids::TOGETHERAI)
            .field("model_id", &self.model_id)
            .field("root_base_url", &self.root_base_url)
            .finish()
    }
}

#[cfg(feature = "togetherai")]
impl ModelMetadata for TogetherAiImageClient {
    fn provider_id(&self) -> &str {
        ids::TOGETHERAI
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(feature = "togetherai")]
impl LlmClient for TogetherAiImageClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::TOGETHERAI)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![resolve_image_model(None, &self.model_id)]
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

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageGenerationCapability for TogetherAiImageClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_image_model(request.model.as_deref(), &self.model_id);
        let (body, warnings) = build_generation_body(&request, &model)?;
        let result = execute_json_request(
            &self.execution_config(),
            &self.generation_url(),
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        let response: TogetherAiImageResponse =
            serde_json::from_value(result.json).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse TogetherAI image generation response: {err}"
                ))
            })?;

        Ok(ImageGenerationResponse {
            images: response
                .data
                .into_iter()
                .map(generated_image_from_together_item)
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

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageExtras for TogetherAiImageClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_image_model(request.model.as_deref(), &self.model_id);
        let (body, warnings) = build_edit_body(&request, &model)?;
        let result = execute_json_request(
            &self.execution_config(),
            &self.generation_url(),
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        let response: TogetherAiImageResponse =
            serde_json::from_value(result.json).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse TogetherAI image edit response: {err}"
                ))
            })?;

        Ok(ImageGenerationResponse {
            images: response
                .data
                .into_iter()
                .map(generated_image_from_together_item)
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

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "TogetherAI does not support image variations".to_string(),
        ))
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }
}

#[cfg(feature = "togetherai")]
#[derive(Clone)]
struct TogetherAiCompatCompositeClient {
    text_client:
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    image_client: TogetherAiImageClient,
    rerank_client: siumai_provider_togetherai::providers::togetherai::TogetherAiClient,
}

#[cfg(feature = "togetherai")]
impl std::fmt::Debug for TogetherAiCompatCompositeClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TogetherAiCompatCompositeClient")
            .field("provider_id", &ids::TOGETHERAI)
            .field("text_model", &self.text_client.model_id())
            .field("image_model", &self.image_client.model_id())
            .field("rerank_model", &self.rerank_client.model_id())
            .finish()
    }
}

#[cfg(feature = "togetherai")]
impl LlmClient for TogetherAiCompatCompositeClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::TOGETHERAI)
    }

    fn supported_models(&self) -> Vec<String> {
        let mut models = self.text_client.supported_models();
        for model in self.image_client.supported_models() {
            if !models.iter().any(|existing| existing == &model) {
                models.push(model);
            }
        }
        for model in self.rerank_client.supported_models() {
            if !models.iter().any(|existing| existing == &model) {
                models.push(model);
            }
        }
        models
    }

    fn capabilities(&self) -> ProviderCapabilities {
        togetherai_capabilities()
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

    fn as_embedding_capability(&self) -> Option<&dyn crate::traits::EmbeddingCapability> {
        self.text_client.as_embedding_capability()
    }

    fn as_completion_capability(&self) -> Option<&dyn crate::traits::CompletionCapability> {
        self.text_client.as_completion_capability()
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

    fn as_image_generation_capability(
        &self,
    ) -> Option<&dyn crate::traits::ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn crate::traits::ImageExtras> {
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        self.text_client.as_file_management_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        self.text_client.as_model_listing_capability()
    }

    fn as_rerank_capability(&self) -> Option<&dyn crate::traits::RerankCapability> {
        self.rerank_client.as_rerank_capability()
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageGenerationCapability for TogetherAiCompatCompositeClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.generate_images(request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        ImageGenerationCapability::max_images_per_call(&self.image_client)
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ImageExtras for TogetherAiCompatCompositeClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.edit_image(request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.image_client.create_variation(request).await
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.image_client.get_supported_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.image_client.supports_image_editing()
    }
}

/// TogetherAI provider factory.
#[cfg(feature = "togetherai")]
pub struct TogetherAiProviderFactory;

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ProviderFactory for TogetherAiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        togetherai_capabilities()
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
        let image_client = TogetherAiImageClient::from_text_client(&text_client);
        let rerank_client = build_native_rerank_client_with_ctx(model_id, ctx)?;
        Ok(Arc::new(TogetherAiCompatCompositeClient {
            text_client,
            image_client,
            rerank_client,
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
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(TogetherAiImageClient::from_text_client(&client)))
    }

    async fn image_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyImageModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(TogetherAiImageClient::from_text_client(&client)))
    }

    async fn compat_speech_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn speech_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilySpeechModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
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

    async fn compat_reranking_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(build_native_rerank_client_with_ctx(
            model_id, ctx,
        )?))
    }

    async fn reranking_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyRerankingModel>, LlmError> {
        Ok(Arc::new(build_native_rerank_client_with_ctx(
            model_id, ctx,
        )?))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::TOGETHERAI)
    }
}
