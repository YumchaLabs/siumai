//! TogetherAI image model runtime.
//!
//! TogetherAI text execution reuses the shared OpenAI-compatible runtime, but the image endpoint is
//! a provider-owned surface in `@ai-sdk/togetherai`. This module keeps the image request/response
//! mapping inside the TogetherAI provider crate instead of the registry factory.

use super::config::TogetherAiConfig;
use crate::core::{ProviderContext, ProviderSpec};
use crate::core_compat::client::LlmClient;
use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_json_request};
use crate::execution::wiring::HttpExecutionWiring;
use crate::traits::{ImageExtras, ImageGenerationCapability, ModelMetadata, ProviderCapabilities};
use crate::types::{
    GeneratedImage, HttpResponseInfo, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest, Warning,
};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Default)]
struct TogetherAiImageSpec;

impl ProviderSpec for TogetherAiImageSpec {
    fn id(&self) -> &'static str {
        "togetherai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_image_generation()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::togetherai::build_togetherai_json_headers(ctx)
    }
}

/// Provider-owned TogetherAI image client.
#[derive(Clone)]
pub struct TogetherAiImageClient {
    config: TogetherAiConfig,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
}

impl std::fmt::Debug for TogetherAiImageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TogetherAiImageClient")
            .field("provider_id", &"togetherai")
            .field("model_id", &self.config.common_params.model)
            .field("base_url", &self.config.base_url)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl TogetherAiImageClient {
    /// Build a TogetherAI image client from provider-owned config.
    pub fn from_config(config: TogetherAiConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Build a TogetherAI image client from provider-owned config and explicit HTTP client.
    pub fn with_http_client(
        config: TogetherAiConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    /// Set retry options.
    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    fn provider_context(&self) -> ProviderContext {
        ProviderContext::new(
            "togetherai",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
    }

    fn execution_config(&self) -> crate::execution::executors::common::HttpExecutionConfig {
        let mut wiring = HttpExecutionWiring::new(
            "togetherai",
            self.http_client.clone(),
            self.provider_context(),
        )
        .with_interceptors(self.config.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());
        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }
        wiring.config(Arc::new(TogetherAiImageSpec))
    }

    fn generation_url(&self) -> String {
        format!(
            "{}/images/generations",
            self.config.base_url.trim_end_matches('/')
        )
    }

    fn model_id(&self) -> &str {
        &self.config.common_params.model
    }
}

impl ModelMetadata for TogetherAiImageClient {
    fn provider_id(&self) -> &str {
        "togetherai"
    }

    fn model_id(&self) -> &str {
        self.model_id()
    }
}

impl LlmClient for TogetherAiImageClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("togetherai")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![resolve_image_model(None, self.model_id())]
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

#[async_trait]
impl ImageGenerationCapability for TogetherAiImageClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_image_model(request.model.as_deref(), self.model_id());
        let (body, warnings) = build_generation_body(&request, &model)?;
        let result = execute_json_request(
            &self.execution_config(),
            &self.generation_url(),
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        image_response_from_raw(result.json, result.headers, model, warnings, "generation")
    }

    fn max_images_per_call(&self) -> Option<u32> {
        Some(1)
    }
}

#[async_trait]
impl ImageExtras for TogetherAiImageClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let model = resolve_image_model(request.model.as_deref(), self.model_id());
        let (body, warnings) = build_edit_body(&request, &model)?;
        let result = execute_json_request(
            &self.execution_config(),
            &self.generation_url(),
            HttpBody::Json(body),
            request.http_config.as_ref(),
            false,
        )
        .await?;

        image_response_from_raw(result.json, result.headers, model, warnings, "edit")
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

fn image_response_from_raw(
    raw: Value,
    headers: HeaderMap,
    model: String,
    warnings: Vec<Warning>,
    operation: &str,
) -> Result<ImageGenerationResponse, LlmError> {
    let response: TogetherAiImageResponse = serde_json::from_value(raw).map_err(|err| {
        LlmError::ParseError(format!(
            "Failed to parse TogetherAI image {operation} response: {err}"
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
            headers: headers_to_map(&headers),
            body: None,
        }),
    })
}

fn resolve_image_model(request_model: Option<&str>, current_model: &str) -> String {
    match request_model {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        _ if current_model.trim().is_empty()
            || current_model == TogetherAiConfig::DEFAULT_MODEL =>
        {
            TogetherAiConfig::DEFAULT_IMAGE_MODEL.to_string()
        }
        _ => current_model.to_string(),
    }
}

fn provider_options_object(
    map: &crate::types::ProviderOptionsMap,
) -> Result<Option<Map<String, Value>>, LlmError> {
    let mut merged = Map::new();
    let mut found = false;

    for provider_id in ["together", "togetherai"] {
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

fn merge_object_fields(body: &mut Map<String, Value>, fields: Option<Map<String, Value>>) {
    let Some(fields) = fields else {
        return;
    };

    for (key, value) in fields {
        body.insert(key, value);
    }
}

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

fn togetherai_aspect_ratio_warning() -> Warning {
    Warning::unsupported(
        "aspectRatio",
        Some("This model does not support the `aspectRatio` option. Use `size` instead."),
    )
}

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

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            Some((key.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect()
}

#[derive(Debug, Deserialize)]
struct TogetherAiImageResponseItem {
    b64_json: String,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct TogetherAiImageResponse {
    data: Vec<TogetherAiImageResponseItem>,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::TogetherAiImageOptions;
    use std::collections::BTreeMap;

    fn image_response() -> Value {
        serde_json::json!({
            "data": [
                {
                    "b64_json": "aGVsbG8=",
                    "revised_prompt": "a tiny robot"
                }
            ],
            "provider": "togetherai"
        })
    }

    #[test]
    fn generation_body_maps_togetherai_image_options_and_size() {
        let request = ImageGenerationRequest {
            prompt: "a tiny robot".to_string(),
            size: Some("1024x768".to_string()),
            aspect_ratio: Some("4:3".to_string()),
            response_format: Some("url".to_string()),
            provider_options_map: crate::types::ProviderOptionsMap(BTreeMap::from([(
                "togetherai".to_string(),
                serde_json::to_value(
                    TogetherAiImageOptions::new()
                        .with_steps(28)
                        .with_guidance(3.5)
                        .with_negative_prompt("blurry"),
                )
                .expect("serialize options"),
            )])),
            ..Default::default()
        };

        let (body, warnings) =
            build_generation_body(&request, "black-forest-labs/FLUX.1-schnell").expect("body");

        assert_eq!(body["prompt"], serde_json::json!("a tiny robot"));
        assert_eq!(body["width"], serde_json::json!(1024));
        assert_eq!(body["height"], serde_json::json!(768));
        assert_eq!(body["response_format"], serde_json::json!("base64"));
        assert_eq!(body["steps"], serde_json::json!(28));
        assert_eq!(body["guidance"], serde_json::json!(3.5));
        assert_eq!(body["negative_prompt"], serde_json::json!("blurry"));
        assert_eq!(warnings.len(), 1);
        assert!(body.get("size").is_none());
        assert!(body.get("aspect_ratio").is_none());
    }

    #[test]
    fn edit_body_rejects_masks_before_http_execution() {
        let request = ImageEditRequest {
            images: vec![ImageEditInput::url("https://example.com/input.png")],
            mask: Some(ImageEditInput::file_with_media_type(
                vec![255, 255, 255, 0],
                "image/png",
            )),
            prompt: "edit with mask".to_string(),
            ..Default::default()
        };

        let error = build_edit_body(&request, "black-forest-labs/FLUX.1-schnell")
            .expect_err("mask edit should be rejected");

        assert!(matches!(error, LlmError::UnsupportedOperation(_)));
    }

    #[test]
    fn image_response_preserves_extra_fields_as_metadata() {
        let response = image_response_from_raw(
            image_response(),
            HeaderMap::new(),
            "black-forest-labs/FLUX.1-schnell".to_string(),
            Vec::new(),
            "generation",
        )
        .expect("response");

        assert_eq!(response.images[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert_eq!(
            response.images[0].metadata["revised_prompt"],
            serde_json::json!("a tiny robot")
        );
        assert_eq!(
            response.metadata["provider"],
            serde_json::json!("togetherai")
        );
    }
}
