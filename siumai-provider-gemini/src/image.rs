use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageLimits, ImageModel, ImageRequest,
    ImageResponse, Model, ModelDescriptor, ModelFamily, ModelId, ModelOperation,
    ProviderOptionError, ProviderOptionSelection, ProviderOptions, TypedProviderOptions,
};
use siumai_protocol_gemini::interactions::{
    ImageAspectRatio as ProtocolImageAspectRatio, ImageMimeType as ProtocolImageMimeType,
    ImageResponseFormat, ImageSize as ProtocolImageSize, STABLE_V1_CREATE_TARGET,
    decode_image_response, encode_image_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use crate::http::{response_error, response_request_id};
use crate::models::{GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_PRO_IMAGE};
use crate::options::{GeminiImageAspectRatio, GeminiImageOptions, GeminiImageSize};
use crate::provider::ProviderRuntime;

const MAX_IMAGES_PER_CALL: u32 = 1;

/// Lightweight Gemini image handle sharing one configured Google runtime.
#[derive(Clone)]
pub struct GeminiImageModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GeminiImageModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.image_scope.clone(),
            model,
            ModelFamily::Image,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<GeminiImageOptions, Error> {
        let selection = call.provider_options_for(self).map_err(option_error)?;
        merge_options(&self.runtime.image_defaults, &selection).map_err(option_error)
    }

    fn plan(
        &self,
        request: &ImageRequest,
        options: GeminiImageOptions,
    ) -> Result<RequestPlan, Error> {
        if request.size().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini Interactions image models use typed image-size tiers instead of portable pixel dimensions",
            ));
        }
        validate_model_options(self.model_id(), &options)?;
        let mut response_format = ImageResponseFormat::new();
        if let Some(mime_type) = requested_media_type(request.format())? {
            response_format = response_format.with_mime_type(mime_type);
        }
        if let Some(aspect_ratio) = options.aspect_ratio {
            response_format =
                response_format.with_aspect_ratio(protocol_aspect_ratio(aspect_ratio));
        }
        if let Some(image_size) = options.image_size {
            response_format = response_format.with_image_size(protocol_image_size(image_size));
        }
        let body = encode_image_request(self.model_id(), request.prompt(), &response_format)?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(STABLE_V1_CREATE_TARGET).map_err(request_build_error)?,
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

impl std::fmt::Debug for GeminiImageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeminiImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for GeminiImageModel {
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
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the Interactions image request",
            )));
        }
        let (_, headers, body) = response.into_parts();
        let request_id = response_request_id(&headers);
        let decoded =
            decode_image_response(&body, self.model_id(), request_id.as_deref(), Vec::new())
                .map_err(|error| self.contextualize(error))?;
        decoded
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(decoded)
    }
}

fn validate_model_options(model: &ModelId, options: &GeminiImageOptions) -> Result<(), Error> {
    let model = model.as_str();
    if model == GEMINI_3_1_FLASH_LITE_IMAGE {
        if options
            .image_size
            .is_some_and(|size| size != GeminiImageSize::OneK)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3.1 Flash Lite Image supports only the 1K image-size tier",
            ));
        }
        if options
            .aspect_ratio
            .is_some_and(GeminiImageAspectRatio::is_extended)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3.1 Flash Lite Image does not support extended aspect ratios",
            ));
        }
    }
    if model == GEMINI_3_PRO_IMAGE {
        if options.image_size == Some(GeminiImageSize::Pixels512) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini 3 Pro Image does not support the 512 image-size tier",
            ));
        }
        if options
            .aspect_ratio
            .is_some_and(GeminiImageAspectRatio::is_extended)
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

fn requested_media_type(format: Option<&str>) -> Result<Option<ProtocolImageMimeType>, Error> {
    match format.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("jpg" | "jpeg" | "image/jpeg") => Ok(Some(ProtocolImageMimeType::Jpeg)),
        Some(_) => Err(Error::new(
            ErrorKind::Unsupported,
            "stable-v1 Gemini Interactions accepts only explicit JPEG image output",
        )),
    }
}

fn protocol_aspect_ratio(value: GeminiImageAspectRatio) -> ProtocolImageAspectRatio {
    match value {
        GeminiImageAspectRatio::Square => ProtocolImageAspectRatio::Square,
        GeminiImageAspectRatio::PortraitOneFour => ProtocolImageAspectRatio::PortraitOneFour,
        GeminiImageAspectRatio::PortraitOneEight => ProtocolImageAspectRatio::PortraitOneEight,
        GeminiImageAspectRatio::PortraitTwoThree => ProtocolImageAspectRatio::PortraitTwoThree,
        GeminiImageAspectRatio::LandscapeThreeTwo => ProtocolImageAspectRatio::LandscapeThreeTwo,
        GeminiImageAspectRatio::PortraitThreeFour => ProtocolImageAspectRatio::PortraitThreeFour,
        GeminiImageAspectRatio::LandscapeFourOne => ProtocolImageAspectRatio::LandscapeFourOne,
        GeminiImageAspectRatio::LandscapeFourThree => ProtocolImageAspectRatio::LandscapeFourThree,
        GeminiImageAspectRatio::PortraitFourFive => ProtocolImageAspectRatio::PortraitFourFive,
        GeminiImageAspectRatio::LandscapeFiveFour => ProtocolImageAspectRatio::LandscapeFiveFour,
        GeminiImageAspectRatio::LandscapeEightOne => ProtocolImageAspectRatio::LandscapeEightOne,
        GeminiImageAspectRatio::PortraitNineSixteen => {
            ProtocolImageAspectRatio::PortraitNineSixteen
        }
        GeminiImageAspectRatio::LandscapeSixteenNine => {
            ProtocolImageAspectRatio::LandscapeSixteenNine
        }
        GeminiImageAspectRatio::Ultrawide => ProtocolImageAspectRatio::Ultrawide,
    }
}

fn protocol_image_size(value: GeminiImageSize) -> ProtocolImageSize {
    match value {
        GeminiImageSize::Pixels512 => ProtocolImageSize::Pixels512,
        GeminiImageSize::OneK => ProtocolImageSize::OneK,
        GeminiImageSize::TwoK => ProtocolImageSize::TwoK,
        GeminiImageSize::FourK => ProtocolImageSize::FourK,
    }
}

fn merge_options(
    defaults: &GeminiImageOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<GeminiImageOptions, ProviderOptionError> {
    let mut merged = defaults.clone();
    for options in selection.typed() {
        let value = decode_options(options)?;
        value.validate()?;
        if value.aspect_ratio.is_some() {
            merged.aspect_ratio = value.aspect_ratio;
        }
        if value.image_size.is_some() {
            merged.image_size = value.image_size;
        }
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "Gemini Interactions image generation only accepts typed provider options"
                .to_string(),
        });
    }
    merged.validate()?;
    Ok(merged)
}

fn decode_options(options: &ProviderOptions) -> Result<GeminiImageOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "google".to_string(),
            reason: "options do not match the Gemini Interactions image schema".to_string(),
        }
    })
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Gemini Interactions image generation",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini Interactions request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use base64::Engine as _;
    use siumai_core::{
        ApiStability, Cancellation, ErrorDetail, ImageModel, MediaData, Model, ModelId,
        ReplayDomain, ReplayDomainId, ResourceKind, ResponseDiagnostics, UsageValue,
        VerifiedFidelity,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GeminiCredential, GeminiProvider};

    const MODEL: &str = GEMINI_3_1_FLASH_IMAGE;

    fn provider(base_url: String) -> GeminiProvider {
        GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-test-endpoint").unwrap(),
            ))
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
            GeminiProvider::builder(GeminiCredential::api_key("bad\nkey")).build(),
            Err(crate::GeminiConfigError::InvalidCredential)
        ));
        let provider = provider("http://127.0.0.1:9".to_string());
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
        assert_eq!(future.descriptor().api_mode(), Some("interactions"));
        assert_eq!(
            future
                .plan(
                    &ImageRequest::new("mountains").unwrap(),
                    GeminiImageOptions::default(),
                )
                .unwrap()
                .target()
                .as_str(),
            "v1/interactions"
        );
    }

    #[tokio::test]
    async fn direct_and_registration_paths_share_the_interactions_contract() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/interactions")
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
        let provider = provider(server.url());
        let request = ImageRequest::new("mountains")
            .unwrap()
            .with_format("jpeg")
            .unwrap();
        let direct = provider.image(MODEL).unwrap();
        let typed = GeminiImageOptions::new()
            .with_aspect_ratio(GeminiImageAspectRatio::LandscapeSixteenNine)
            .with_image_size(GeminiImageSize::TwoK);
        let call = CallOptions::default()
            .with_provider_options_for(&direct, &typed)
            .unwrap();
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
        let current = crate::GeminiProfile::current(ReplayDomain::official(
            ReplayDomainId::new("google-gemini-api").unwrap(),
        ))
        .unwrap();
        let claims = current.provider_profile().verified_claims().unwrap();
        assert_eq!(claims.len(), 5);
        assert!(
            claims
                .iter()
                .all(|claim| claim.fidelity() == VerifiedFidelity::Native)
        );
        assert!(claims.iter().any(|claim| {
            claim.scope().api_mode().as_str() == "interactions-speech"
                && claim.stability() == ApiStability::Experimental
        }));
        assert!(
            claims
                .iter()
                .filter(|claim| claim.scope().api_mode().as_str() != "interactions-speech")
                .all(|claim| claim.stability() == ApiStability::Stable)
        );
        assert_eq!(
            current.provider_profile().catalog().unwrap().iter().count(),
            14
        );

        let custom = crate::GeminiProfile::custom(ReplayDomain::custom(
            ReplayDomainId::new("gemini-test-endpoint").unwrap(),
        ))
        .unwrap();
        assert!(custom.provider_profile().verified_claims().is_none());
        assert_eq!(custom.provider_profile().generic_claims().unwrap().len(), 5);
    }

    #[tokio::test]
    async fn uri_output_and_unknown_usage_remain_representable() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/interactions")
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
        let response = provider(server.url())
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
        let provider = provider("http://127.0.0.1:9".to_string());
        let lite = provider.image(GEMINI_3_1_FLASH_LITE_IMAGE).unwrap();
        let options = GeminiImageOptions::new().with_image_size(GeminiImageSize::FourK);
        let call = CallOptions::default()
            .with_provider_options_for(&lite, &options)
            .unwrap();
        assert_eq!(
            lite.generate_image(ImageRequest::new("mountains").unwrap(), call,)
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
            .mock("POST", "/v1/interactions")
            .with_status(500)
            .with_header("x-request-id", "canary-header-secret")
            .with_body("canary-body-secret")
            .expect(1)
            .create_async()
            .await;
        let error = provider(server.url())
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
        let provider = GeminiProvider::builder(GeminiCredential::unauthenticated())
            .with_call_timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        let claim = &provider
            .profile()
            .provider_profile()
            .verified_claims()
            .unwrap()[0];
        assert_eq!(claim.scope().api_mode().as_str(), "interactions");
    }
}
