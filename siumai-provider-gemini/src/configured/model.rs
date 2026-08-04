use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use http::header::{ACCEPT, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageArtifact, ImageLimits, ImageModel,
    ImageRequest, ImageResponse, MediaData, Model, ModelAdvisory, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ModelPolicy, ModelPolicyDecision, ProviderOptionError,
    ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin, ProviderOptions,
    ResponseDiagnostics, ResponseMetadata, SensitiveResponse, SupportState, Usage, Warning,
    WarningKind,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, ResponseHeaders,
    TransportResponse,
};

use super::options::{GoogleImagenAspectRatio, GoogleImagenOptions, GoogleImagenPersonGeneration};
use super::provider::ProviderRuntime;

const MAX_IMAGES_PER_CALL: u32 = 4;
const DEFAULT_MEDIA_TYPE: &str = "image/png";
const ERROR_CAPTURE_BYTES: usize = 64 * 1024;

/// Lightweight Imagen model handle sharing one configured Google runtime.
#[derive(Clone)]
pub struct GoogleImagenModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GoogleImagenModel {
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
            .evaluate(&siumai_core::ModelPolicyContext {
                scope: self.runtime.scope.clone(),
                model: self.model_id().clone(),
                family: ModelFamily::Image,
                operation: ModelOperation::GenerateImage,
            });
        if let SupportState::Unsupported { .. } = decision.state() {
            return Err(self.contextualize(Error::new(
                ErrorKind::Unsupported,
                "model policy rejected the Imagen predict operation",
            )));
        }
        let warnings = decision.advisories().iter().map(advisory_warning).collect();
        Ok((decision, warnings))
    }

    fn options(&self, call: &CallOptions) -> Result<GoogleImagenOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                self.provider_id(),
                &GoogleImagenOptionMerger {
                    defaults: self.runtime.default_options.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &ImageRequest,
        options: GoogleImagenOptions,
    ) -> Result<RequestPlan, Error> {
        if request.size().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Imagen predict does not support the portable image size option",
            ));
        }
        if request
            .format()
            .is_some_and(|format| !format.eq_ignore_ascii_case("png"))
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Imagen predict returns PNG and cannot honor the requested image format",
            ));
        }
        let body = ImagenPredictRequest {
            instances: [ImagenInstance {
                prompt: request.prompt(),
            }],
            parameters: ImagenParameters {
                sample_count: request.count(),
                aspect_ratio: options
                    .aspect_ratio
                    .unwrap_or(GoogleImagenAspectRatio::Square)
                    .as_wire(),
                person_generation: options
                    .person_generation
                    .map(GoogleImagenPersonGeneration::as_wire),
            },
        };
        let target = format!(
            "models/{}:predict",
            urlencoding::encode(self.model_id().as_str())
        );
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(target).map_err(request_build_error)?,
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

impl std::fmt::Debug for GoogleImagenModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GoogleImagenModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GoogleImagenModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for GoogleImagenModel {
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
#[serde(rename_all = "camelCase")]
struct ImagenPredictRequest<'a> {
    instances: [ImagenInstance<'a>; 1],
    parameters: ImagenParameters<'a>,
}

#[derive(Serialize)]
struct ImagenInstance<'a> {
    prompt: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImagenParameters<'a> {
    sample_count: u32,
    aspect_ratio: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    person_generation: Option<&'a str>,
}

#[derive(Deserialize)]
struct ImagenPredictResponse {
    predictions: Vec<ImagenPrediction>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImagenPrediction {
    #[serde(default)]
    bytes_base64_encoded: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

fn decode_response(
    model: &ModelId,
    request: &ImageRequest,
    response: TransportResponse,
    warnings: Vec<Warning>,
) -> Result<ImageResponse, Error> {
    let request_id = response
        .headers()
        .get(&http::header::HeaderName::from_static("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let decoded: ImagenPredictResponse =
        serde_json::from_slice(response.body()).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned an invalid Imagen predict response",
            )
            .with_source(source)
        })?;
    let mut images = Vec::with_capacity(decoded.predictions.len());
    let mut filtered_outputs = 0_u64;
    for prediction in decoded.predictions {
        let Some(encoded) = prediction.bytes_base64_encoded else {
            filtered_outputs = filtered_outputs.saturating_add(1);
            continue;
        };
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|source| {
                Error::protocol_violation("Imagen response contains invalid base64 image data")
                    .with_source(source)
            })?;
        let media_type = prediction
            .mime_type
            .unwrap_or_else(|| DEFAULT_MEDIA_TYPE.to_string());
        if !media_type.starts_with("image/") {
            return Err(Error::protocol_violation(
                "Imagen response contains a non-image media type",
            ));
        }
        images.push(ImageArtifact {
            media_type,
            data: MediaData::Bytes(data.into()),
            revised_prompt: prediction.prompt,
        });
    }
    let mut provider = BTreeMap::new();
    if filtered_outputs > 0 {
        provider.insert(
            "google.filtered_outputs".to_string(),
            Value::from(filtered_outputs),
        );
    }
    let response = ImageResponse {
        images,
        metadata: ResponseMetadata {
            response_id: None,
            request_id,
            model: Some(model.clone()),
        },
        usage: Usage::default(),
        warnings,
        provider,
    };
    response.validate(request)?;
    Ok(response)
}

struct GoogleImagenOptionMerger {
    defaults: GoogleImagenOptions,
}

impl ProviderOptionMerger for GoogleImagenOptionMerger {
    type Output = GoogleImagenOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        serde_json::from_value::<GoogleImagenOptions>(Value::Object(options.value().clone()))
            .map(|_| ())
            .map_err(|_| ProviderOptionError::Rejected {
                path: "google".to_string(),
                reason: "options do not match the Imagen predict schema".to_string(),
            })
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            let value = serde_json::from_value::<GoogleImagenOptions>(Value::Object(
                options.value().clone(),
            ))
            .map_err(|_| ProviderOptionError::Rejected {
                path: "google".to_string(),
                reason: "options do not match the Imagen predict schema".to_string(),
            })?;
            if value.aspect_ratio.is_some() {
                merged.aspect_ratio = value.aspect_ratio;
            }
            if value.person_generation.is_some() {
                merged.person_generation = value.person_generation;
            }
        }
        Ok(merged)
    }
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the Google Imagen advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "Google Imagen model is deprecated",
        ),
        ModelAdvisory::Retired { .. } => {
            Warning::new(WarningKind::RetiredModel, "Google Imagen model is retired")
        }
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "Google Imagen model ID is a rolling alias",
        ),
        _ => Warning::provider("model_advisory", "Google returned a model advisory"),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Imagen predict",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Imagen request violates the transport contract",
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
    Error::new(kind, "provider rejected the Imagen predict request")
        .with_diagnostics(
            ResponseDiagnostics::default()
                .with_status(status.as_u16())
                .with_body_truncated(truncated),
        )
        .with_sensitive_response(SensitiveResponse::new(raw_headers, captured))
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
    use std::time::{Duration, Instant};

    use base64::Engine as _;
    use siumai_core::{
        Cancellation, ErrorDetail, ImageModel, Model, ModelId, ProviderOptions, ResourceKind,
        UsageValue,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GoogleCredential, GoogleImagenProvider};

    const MODEL: &str = "imagen-4.0-generate-001";

    fn provider(base_url: String) -> GoogleImagenProvider {
        GoogleImagenProvider::builder(GoogleCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .build()
            .unwrap()
    }

    fn success_body(count: usize) -> String {
        let predictions = (0..count)
            .map(|index| {
                let data =
                    base64::engine::general_purpose::STANDARD.encode(format!("image-{index}"));
                serde_json::json!({
                    "bytesBase64Encoded": data,
                    "mimeType": "image/png"
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "predictions": predictions }).to_string()
    }

    #[tokio::test]
    async fn invalid_request_and_count_limit_fail_before_transport() {
        assert_eq!(
            ImageRequest::new("  ").unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
        assert!(matches!(
            GoogleImagenProvider::builder(GoogleCredential::api_key("bad\nkey")).build(),
            Err(crate::GoogleImagenConfigError::InvalidCredential)
        ));

        let provider = provider("http://127.0.0.1:9/v1beta".to_string());
        let model = provider.imagen(MODEL).unwrap();
        let request = ImageRequest::new("mountains")
            .unwrap()
            .with_count(5)
            .unwrap();
        let error = model
            .generate_image(request, CallOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::ImageOutputs,
                actual: 5,
                maximum: 4,
            })
        );

        let future = provider.imagen("future:model:revision").unwrap();
        assert_eq!(future.descriptor().api_mode(), Some("imagen-predict"));
        let plan = future
            .plan(
                &ImageRequest::new("mountains").unwrap(),
                GoogleImagenOptions::default(),
            )
            .unwrap();
        assert_eq!(
            plan.target().as_str(),
            "models/future%3Amodel%3Arevision:predict"
        );
    }

    #[tokio::test]
    async fn direct_and_registration_paths_share_one_authenticated_predict_contract() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", format!("/v1beta/models/{MODEL}:predict").as_str())
            .match_header("x-goog-api-key", "test-key")
            .match_header("accept", "application/json")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"prompt\":\"mountains\""#.to_string()),
                mockito::Matcher::Regex(r#"\"sampleCount\":2"#.to_string()),
                mockito::Matcher::Regex(r#"\"aspectRatio\":\"16:9\""#.to_string()),
                mockito::Matcher::Regex(r#"\"personGeneration\":\"allow_adult\""#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "request-1")
            .with_body(success_body(2))
            .expect(2)
            .create_async()
            .await;
        let provider = provider(format!("{}/v1beta", server.url()));
        let request = ImageRequest::new("mountains")
            .unwrap()
            .with_count(2)
            .unwrap();
        let typed = ProviderOptions::typed(&GoogleImagenOptions {
            aspect_ratio: Some(GoogleImagenAspectRatio::LandscapeSixteenNine),
            person_generation: Some(GoogleImagenPersonGeneration::AllowAdult),
        })
        .unwrap();
        let call = CallOptions::default().with_provider_options(typed);

        let direct = provider.imagen(MODEL).unwrap();
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
        assert_eq!(direct_response.usage.input_tokens, UsageValue::Unknown);
        assert!(matches!(
            &direct_response.images[0].data,
            MediaData::Bytes(bytes) if bytes.as_ref() == b"image-0"
        ));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn empty_and_invalid_predictions_are_typed_protocol_failures() {
        let mut server = mockito::Server::new_async().await;
        let empty = server
            .mock("POST", format!("/v1beta/models/{MODEL}:predict").as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"predictions":[]}"#)
            .expect(1)
            .create_async()
            .await;
        let configured = provider(format!("{}/v1beta", server.url()));
        let error = configured
            .imagen(MODEL)
            .unwrap()
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PartialResult);
        empty.assert_async().await;

        let mut invalid_server = mockito::Server::new_async().await;
        let invalid = invalid_server
            .mock("POST", format!("/v1beta/models/{MODEL}:predict").as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"predictions":[{"bytesBase64Encoded":"%%%"}]}"#)
            .expect(1)
            .create_async()
            .await;
        let error = provider(format!("{}/v1beta", invalid_server.url()))
            .imagen(MODEL)
            .unwrap()
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
        invalid.assert_async().await;
    }

    #[tokio::test]
    async fn returned_image_bytes_are_owned_and_cross_task_boundaries() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", format!("/v1beta/models/{MODEL}:predict").as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(success_body(1))
            .expect(1)
            .create_async()
            .await;
        let response = provider(format!("{}/v1beta", server.url()))
            .imagen(MODEL)
            .unwrap()
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default(),
            )
            .await
            .unwrap();
        let bytes = tokio::spawn(async move {
            let MediaData::Bytes(bytes) = &response.images[0].data else {
                panic!("Imagen predict must return owned bytes");
            };
            bytes.clone()
        })
        .await
        .unwrap();
        assert_eq!(bytes.as_ref(), b"image-0");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn cancellation_and_deadline_reach_transport_controls() {
        let model = provider("http://127.0.0.1:9/v1beta".to_string())
            .imagen(MODEL)
            .unwrap();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let error = model
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default().with_cancellation(cancellation),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);

        let error = model
            .generate_image(
                ImageRequest::new("mountains").unwrap(),
                CallOptions::default().with_deadline(Instant::now() - Duration::from_millis(1)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn provider_errors_are_sanitized_and_never_replayed() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", format!("/v1beta/models/{MODEL}:predict").as_str())
            .with_status(500)
            .with_header("x-request-id", "canary-header-secret")
            .with_body("canary-body-secret")
            .expect(1)
            .create_async()
            .await;
        let error = provider(format!("{}/v1beta", server.url()))
            .imagen(MODEL)
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
}
