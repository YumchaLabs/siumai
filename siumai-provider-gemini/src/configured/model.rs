use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageArtifact, ImageLimits, ImageModel,
    ImageRequest, ImageResponse, MediaData, Model, ModelAdvisory, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ModelPolicy, ModelPolicyDecision, ProviderOptionContext,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, ResourceKind, ResponseDiagnostics, ResponseMetadata, SensitiveResponse,
    SupportState, Usage, Warning, WarningKind,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders,
    TransportResponse,
};

use super::models::{GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_PRO_IMAGE};
use super::options::{GoogleImageAspectRatio, GoogleImageOptions, GoogleImageSize};
use super::provider::ProviderRuntime;

const MAX_IMAGES_PER_CALL: u32 = 1;
const DEFAULT_MEDIA_TYPE: &str = "image/png";
const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Lightweight Gemini image handle sharing one configured Google runtime.
#[derive(Clone)]
pub struct GoogleImageModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GoogleImageModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor =
            ModelDescriptor::from_scope(runtime.scope.clone(), model, ModelFamily::Image);
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(&self) -> Result<(ModelPolicyDecision, Vec<Warning>), Error> {
        let decision = self
            .runtime
            .policy
            .evaluate(&siumai_core::ModelPolicyContext::new(
                self.runtime.scope.clone(),
                self.model_id().clone(),
                ModelOperation::GenerateImage,
            ));
        if let SupportState::Unsupported { .. } = decision.state() {
            return Err(self.contextualize(Error::new(
                ErrorKind::Unsupported,
                "model policy rejected the Google Interactions image operation",
            )));
        }
        let warnings = decision.advisories().iter().map(advisory_warning).collect();
        Ok((decision, warnings))
    }

    fn options(&self, call: &CallOptions) -> Result<GoogleImageOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                ProviderOptionContext::new(
                    self.provider_id(),
                    ModelFamily::Image,
                    self.descriptor.scope().api_mode(),
                ),
                &GoogleImageOptionMerger {
                    defaults: self.runtime.default_options.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &ImageRequest,
        options: GoogleImageOptions,
    ) -> Result<RequestPlan, Error> {
        if request.size().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Google Interactions image models use typed image-size tiers instead of portable pixel dimensions",
            ));
        }
        validate_model_options(self.model_id(), &options)?;
        let body = InteractionRequest {
            model: self.model_id().as_str(),
            input: request.prompt(),
            response_format: [InteractionImageFormat {
                kind: "image",
                mime_type: requested_media_type(request.format())?,
                aspect_ratio: options.aspect_ratio.map(GoogleImageAspectRatio::as_wire),
                image_size: options.image_size.map(GoogleImageSize::as_wire),
            }],
        };
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new("interactions").map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::GenerateImage),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for GoogleImageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GoogleImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GoogleImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for GoogleImageModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(MAX_IMAGES_PER_CALL),
        }
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let (_, warnings) = self.policy()?;
        let provider_options = self
            .options(&options)
            .map_err(|error| self.contextualize(error))?;
        let plan = self
            .plan(&request, provider_options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        decode_response(self.model_id(), &request, response, warnings)
            .map_err(|error| self.contextualize(error))
    }
}

#[derive(Serialize)]
struct InteractionRequest<'a> {
    model: &'a str,
    input: &'a str,
    response_format: [InteractionImageFormat<'a>; 1],
}

#[derive(Serialize)]
struct InteractionImageFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<&'static str>,
}

#[derive(Deserialize)]
struct InteractionResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    status: String,
    #[serde(default)]
    steps: Vec<InteractionStep>,
    #[serde(default)]
    usage: Option<InteractionUsage>,
    #[serde(default)]
    service_tier: Option<String>,
}

#[derive(Deserialize)]
struct InteractionStep {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    content: Vec<InteractionContent>,
}

#[derive(Deserialize)]
struct InteractionContent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    uri: Option<String>,
}

#[derive(Deserialize)]
struct InteractionUsage {
    #[serde(default)]
    total_input_tokens: Option<u64>,
    #[serde(default)]
    total_output_tokens: Option<u64>,
    #[serde(default)]
    total_thought_tokens: Option<u64>,
    #[serde(default)]
    total_cached_tokens: Option<u64>,
    #[serde(default)]
    total_tool_use_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_by_modality: Option<Value>,
    #[serde(default)]
    output_tokens_by_modality: Option<Value>,
    #[serde(default)]
    cached_tokens_by_modality: Option<Value>,
    #[serde(default)]
    tool_use_tokens_by_modality: Option<Value>,
}

fn decode_response(
    requested_model: &ModelId,
    request: &ImageRequest,
    response: TransportResponse,
    mut warnings: Vec<Warning>,
) -> Result<ImageResponse, Error> {
    let (_, headers, body) = response.into_parts();
    let request_id = response_request_id(&headers);
    let decoded: InteractionResponse = serde_json::from_slice(&body).map_err(|source| {
        let truncated = body.len() > ERROR_CAPTURE_BYTES;
        let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
        Error::new(
            ErrorKind::Protocol,
            "Google returned malformed Interactions JSON",
        )
        .with_source(source)
        .with_diagnostics(ResponseDiagnostics::default().with_body_truncated(truncated))
        .with_sensitive_response(SensitiveResponse::new(BTreeMap::new(), captured))
    })?;
    validate_terminal_status(&decoded.status)?;

    let mut images = Vec::new();
    for content in decoded
        .steps
        .into_iter()
        .filter(|step| step.kind == "model_output")
        .flat_map(|step| step.content)
        .filter(|content| content.kind == "image")
    {
        images.push(decode_image(content)?);
    }

    let mut provider = BTreeMap::new();
    if images.len() > 1 {
        provider.insert(
            "google.image_block_count".to_string(),
            Value::from(images.len() as u64),
        );
        warnings.push(Warning::provider(
            "multiple_image_outputs",
            "Google returned multiple image blocks; the final block was selected",
        ));
        let final_image = images.pop().expect("image collection is non-empty");
        images.clear();
        images.push(final_image);
    }
    provider.insert("google.status".to_string(), Value::from(decoded.status));
    if let Some(service_tier) = decoded.service_tier {
        provider.insert("google.service_tier".to_string(), Value::from(service_tier));
    }
    let model = match decoded.model {
        Some(model) => Some(ModelId::new(model).map_err(|source| {
            Error::protocol_violation("Google returned an invalid model identifier")
                .with_source(source)
        })?),
        None => Some(requested_model.clone()),
    };
    let response = ImageResponse {
        images,
        metadata: ResponseMetadata {
            response_id: decoded.id,
            request_id,
            model,
        },
        usage: decode_usage(decoded.usage),
        warnings,
        provider,
    };
    response.validate(request)?;
    Ok(response)
}

fn decode_image(content: InteractionContent) -> Result<ImageArtifact, Error> {
    let media_type = content
        .mime_type
        .unwrap_or_else(|| DEFAULT_MEDIA_TYPE.to_string());
    if !media_type.starts_with("image/") {
        return Err(Error::protocol_violation(
            "Google returned a non-image media type in an image block",
        ));
    }
    let data = match (content.data, content.uri) {
        (Some(encoded), _) => MediaData::Bytes(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|source| {
                    Error::protocol_violation(
                        "Google returned invalid base64 data in an image block",
                    )
                    .with_source(source)
                })?
                .into(),
        ),
        (None, Some(uri)) if !uri.trim().is_empty() => MediaData::Url(uri),
        _ => {
            return Err(Error::protocol_violation(
                "Google returned an image block without data or URI",
            ));
        }
    };
    Ok(ImageArtifact {
        media_type,
        data,
        revised_prompt: None,
    })
}

fn decode_usage(wire: Option<InteractionUsage>) -> Usage {
    let Some(wire) = wire else {
        return Usage::default();
    };
    let mut usage = Usage::default()
        .with_input_tokens(wire.total_input_tokens)
        .with_output_tokens(wire.total_output_tokens)
        .with_total_tokens(wire.total_tokens)
        .with_reasoning_tokens(wire.total_thought_tokens)
        .with_cache_read_tokens(wire.total_cached_tokens)
        .with_orchestration_tokens(wire.total_tool_use_tokens);
    for (name, value) in [
        (
            "google.input_tokens_by_modality",
            wire.input_tokens_by_modality,
        ),
        (
            "google.output_tokens_by_modality",
            wire.output_tokens_by_modality,
        ),
        (
            "google.cached_tokens_by_modality",
            wire.cached_tokens_by_modality,
        ),
        (
            "google.tool_use_tokens_by_modality",
            wire.tool_use_tokens_by_modality,
        ),
    ] {
        if let Some(value) = value {
            usage = usage.with_provider_value(name, value);
        }
    }
    usage
}

fn validate_terminal_status(status: &str) -> Result<(), Error> {
    match status {
        "completed" => Ok(()),
        "cancelled" => Err(Error::cancelled("Google cancelled the image interaction")),
        "incomplete" => Err(Error::partial_result(ResourceKind::ImageOutputs, 1, 0)),
        "failed" => Err(Error::new(
            ErrorKind::Provider,
            "Google reported a failed image interaction",
        )),
        "in_progress" | "requires_action" => Err(Error::protocol_violation(
            "Google returned a non-terminal image interaction",
        )),
        _ => Err(Error::protocol_violation(
            "Google returned an unknown interaction status",
        )),
    }
}

fn validate_model_options(model: &ModelId, options: &GoogleImageOptions) -> Result<(), Error> {
    let model = model.as_str();
    if model == GEMINI_3_1_FLASH_LITE_IMAGE {
        if options
            .image_size
            .is_some_and(|size| size != GoogleImageSize::OneK)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3.1 Flash Lite Image supports only the 1K image-size tier",
            ));
        }
        if options
            .aspect_ratio
            .is_some_and(GoogleImageAspectRatio::is_extended)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3.1 Flash Lite Image does not support extended aspect ratios",
            ));
        }
    }
    if model == GEMINI_3_PRO_IMAGE {
        if options.image_size == Some(GoogleImageSize::Pixels512) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3 Pro Image does not support the 512 image-size tier",
            ));
        }
        if options
            .aspect_ratio
            .is_some_and(GoogleImageAspectRatio::is_extended)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3 Pro Image does not support extended aspect ratios",
            ));
        }
    }
    if model == GEMINI_3_1_FLASH_IMAGE {
        return Ok(());
    }
    Ok(())
}

fn requested_media_type(format: Option<&str>) -> Result<Option<&'static str>, Error> {
    match format.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("png" | "image/png") => Ok(Some("image/png")),
        Some("jpg" | "jpeg" | "image/jpeg") => Ok(Some("image/jpeg")),
        Some(_) => Err(Error::new(
            ErrorKind::Unsupported,
            "Google Interactions image models support PNG or JPEG output",
        )),
    }
}

struct GoogleImageOptionMerger {
    defaults: GoogleImageOptions,
}

impl ProviderOptionMerger for GoogleImageOptionMerger {
    type Output = GoogleImageOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options).map(|_| ())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            let value = decode_options(options)?;
            if value.aspect_ratio.is_some() {
                merged.aspect_ratio = value.aspect_ratio;
            }
            if value.image_size.is_some() {
                merged.image_size = value.image_size;
            }
        }
        Ok(merged)
    }
}

fn decode_options(options: &ProviderOptions) -> Result<GoogleImageOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "google".to_string(),
            reason: "options do not match the Google Interactions image schema".to_string(),
        }
    })
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the current Google image advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "Google image model is deprecated",
        ),
        ModelAdvisory::Retired { .. } => {
            Warning::new(WarningKind::RetiredModel, "Google image model is retired")
        }
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "Google image model ID is a rolling alias",
        ),
        _ => Warning::provider("model_advisory", "Google returned a model advisory"),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Google Interactions image generation",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Google Interactions request violates the transport contract",
    )
    .with_source(source)
}

fn provider_status_error(response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::InvalidInput,
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        _ => ErrorKind::Provider,
    };
    let raw_headers = response_headers(&headers);
    let truncated = body.len() > ERROR_CAPTURE_BYTES;
    let captured = body[..body.len().min(ERROR_CAPTURE_BYTES)].to_vec();
    Error::new(kind, "Google rejected the Interactions image request")
        .with_diagnostics(
            ResponseDiagnostics::default()
                .with_status(status.as_u16())
                .with_body_truncated(truncated),
        )
        .with_sensitive_response(SensitiveResponse::new(raw_headers, captured))
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "x-goog-request-id"]
        .into_iter()
        .find_map(|name| {
            headers
                .get(&HeaderName::from_static(name))
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned)
        })
}

fn response_headers(headers: &ResponseHeaders) -> BTreeMap<String, String> {
    headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use base64::Engine as _;
    use siumai_core::{
        ApiStability, Cancellation, ErrorDetail, ImageModel, Model, ModelId, ProviderOptions,
        ResourceKind, UsageValue, VerifiedFidelity,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GoogleCredential, GoogleImageProvider};

    const MODEL: &str = GEMINI_3_1_FLASH_IMAGE;

    fn provider(base_url: String) -> GoogleImageProvider {
        GoogleImageProvider::builder(GoogleCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .build()
            .unwrap()
    }

    fn success_body() -> String {
        let data = base64::engine::general_purpose::STANDARD.encode("image-0");
        serde_json::json!({
            "id": "interaction-1",
            "model": MODEL,
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "image",
                    "data": data,
                    "mime_type": "image/png"
                }]
            }],
            "usage": {
                "total_input_tokens": 7,
                "total_output_tokens": 11,
                "total_thought_tokens": 3,
                "total_cached_tokens": 2,
                "total_tokens": 18,
                "output_tokens_by_modality": [{"modality": "IMAGE", "tokens": 11}]
            },
            "service_tier": "default"
        })
        .to_string()
    }

    #[tokio::test]
    async fn invalid_configuration_and_portable_mismatches_fail_before_transport() {
        assert!(matches!(
            GoogleImageProvider::builder(GoogleCredential::api_key("bad\nkey")).build(),
            Err(crate::GoogleImageConfigError::InvalidCredential)
        ));
        let provider = provider("http://127.0.0.1:9/v1beta".to_string());
        let model = provider.image(MODEL).unwrap();

        let count_error = model
            .generate_image(
                ImageRequest::new("mountains")
                    .unwrap()
                    .with_count(2)
                    .unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(count_error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            count_error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::ImageOutputs,
                actual: 2,
                maximum: 1,
            })
        );

        for request in [
            ImageRequest::new("mountains")
                .unwrap()
                .with_size(1024, 1024)
                .unwrap(),
            ImageRequest::new("mountains")
                .unwrap()
                .with_format("webp")
                .unwrap(),
        ] {
            assert_eq!(
                model
                    .generate_image(request, CallOptions::default())
                    .await
                    .unwrap_err()
                    .kind(),
                ErrorKind::Unsupported
            );
        }

        let future = provider.image("future-google-image-model").unwrap();
        assert_eq!(future.descriptor().api_mode(), Some("interactions-image"));
        assert_eq!(
            future
                .plan(
                    &ImageRequest::new("mountains").unwrap(),
                    GoogleImageOptions::default(),
                )
                .unwrap()
                .target()
                .as_str(),
            "interactions"
        );
    }

    #[tokio::test]
    async fn direct_and_registration_paths_share_the_interactions_contract() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1beta/interactions")
            .match_header("x-goog-api-key", "test-key")
            .match_header("accept", "application/json")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"gemini-3.1-flash-image\""#.to_string()),
                mockito::Matcher::Regex(r#"\"input\":\"mountains\""#.to_string()),
                mockito::Matcher::Regex(r#"\"type\":\"image\""#.to_string()),
                mockito::Matcher::Regex(r#"\"mime_type\":\"image/jpeg\""#.to_string()),
                mockito::Matcher::Regex(r#"\"aspect_ratio\":\"16:9\""#.to_string()),
                mockito::Matcher::Regex(r#"\"image_size\":\"2K\""#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "request-1")
            .with_body(success_body())
            .expect(2)
            .create_async()
            .await;
        let provider = provider(format!("{}/v1beta", server.url()));
        let request = ImageRequest::new("mountains")
            .unwrap()
            .with_format("jpeg")
            .unwrap();
        let typed = ProviderOptions::typed(
            &GoogleImageOptions::new()
                .with_aspect_ratio(GoogleImageAspectRatio::LandscapeSixteenNine)
                .with_image_size(GoogleImageSize::TwoK),
        )
        .unwrap();
        let call = CallOptions::default().with_provider_options(typed);

        let direct = provider.image(MODEL).unwrap();
        let direct_response = direct
            .generate_image(request.clone(), call.clone())
            .await
            .unwrap();
        let erased = provider
            .registration()
            .image_model(ModelId::new(MODEL).unwrap())
            .unwrap();
        let erased_response = erased.generate_image(request, call).await.unwrap();

        assert_eq!(direct.descriptor(), erased.descriptor());
        assert_eq!(direct_response, erased_response);
        assert_eq!(
            direct_response.metadata.request_id.as_deref(),
            Some("request-1")
        );
        assert_eq!(direct_response.usage.input_tokens, UsageValue::Known(7));
        assert_eq!(direct_response.usage.reasoning_tokens, UsageValue::Known(3));
        assert!(matches!(
            &direct_response.images[0].data,
            MediaData::Bytes(bytes) if bytes.as_ref() == b"image-0"
        ));
        mock.assert_async().await;
    }

    #[test]
    fn profiles_distinguish_official_evidence_from_custom_compatibility() {
        let current = super::super::GoogleImageProfile::current().unwrap();
        let claims = current.provider_profile().verified_claims().unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].fidelity(), VerifiedFidelity::Native);
        assert_eq!(claims[0].stability(), ApiStability::Experimental);
        assert_eq!(
            current.provider_profile().catalog().unwrap().iter().count(),
            3
        );

        let custom = super::super::GoogleImageProfile::custom().unwrap();
        assert!(custom.provider_profile().verified_claims().is_none());
        assert_eq!(custom.provider_profile().generic_claims().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn uri_output_and_unknown_usage_remain_representable() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1beta/interactions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "id": "interaction-uri",
                    "status": "completed",
                    "steps": [{
                        "type": "model_output",
                        "content": [{
                            "type": "image",
                            "uri": "https://example.invalid/image.png",
                            "mime_type": "image/png"
                        }]
                    }]
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;
        let response = provider(format!("{}/v1beta", server.url()))
            .image(MODEL)
            .unwrap()
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();
        assert!(matches!(
            &response.images[0].data,
            MediaData::Url(uri) if uri == "https://example.invalid/image.png"
        ));
        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn model_specific_options_and_cancellation_are_checked_locally() {
        let provider = provider("http://127.0.0.1:9/v1beta".to_string());
        let lite = provider.image(GEMINI_3_1_FLASH_LITE_IMAGE).unwrap();
        let options = ProviderOptions::typed(
            &GoogleImageOptions::new().with_image_size(GoogleImageSize::FourK),
        )
        .unwrap();
        assert_eq!(
            lite.generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default().with_provider_options(options),
            )
            .await
            .unwrap_err()
            .kind(),
            ErrorKind::Unsupported
        );

        let cancellation = Cancellation::new();
        cancellation.cancel();
        assert_eq!(
            provider
                .image(MODEL)
                .unwrap()
                .generate_image(
                    ImageRequest::new("mountains").unwrap(),
                    CallOptions::default().with_cancellation(cancellation),
                )
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Cancelled
        );
    }

    #[tokio::test]
    async fn provider_errors_are_sanitized_and_never_replayed() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1beta/interactions")
            .with_status(500)
            .with_header("x-request-id", "canary-header-secret")
            .with_body("canary-body-secret")
            .expect(1)
            .create_async()
            .await;
        let error = provider(format!("{}/v1beta", server.url()))
            .image(MODEL)
            .unwrap()
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Provider);
        assert_eq!(
            error.diagnostics().and_then(ResponseDiagnostics::status),
            Some(500)
        );
        let debug = format!("{error:?}");
        assert!(!debug.contains("canary-header-secret"));
        assert!(!debug.contains("canary-body-secret"));
        mock.assert_async().await;
    }

    #[test]
    fn default_timeout_builder_remains_network_free() {
        let provider = GoogleImageProvider::builder(GoogleCredential::unauthenticated())
            .with_call_timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        let claim = &provider
            .profile()
            .provider_profile()
            .verified_claims()
            .unwrap()[0];
        assert_eq!(claim.scope().api_mode().as_str(), "interactions-image");
    }
}
