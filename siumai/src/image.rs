//! Image generation model family APIs.
//!
//! This is the recommended Rust-first surface for image generation:
//! - `generate` for direct generation
//! - `edit` for direct edit/inpainting requests
//! - `variation` for direct variation requests
//! - `generate_image` for the AI SDK-style unified request surface

mod projection;
mod workflow;

use crate::image::projection::project_generate_image_response;
use crate::image::workflow::{
    apply_edit_call_options, apply_generation_call_options, apply_unified_call_options,
    apply_variation_call_options, dispatch_generate_image, edit_single, ensure_images_generated,
    generate_single, normalize_generation_count, normalize_optional_generation_count,
    resolve_effective_max_images_per_call, split_call_image_counts, variation_single,
};
use crate::request_options::{EffectiveRequestOptions, run_with_abort};
use crate::retry_api::retry_with;
#[cfg(test)]
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::future::try_join_all;
use siumai_core::error::LlmError;
use siumai_core::traits::ImageExtras;
#[cfg(test)]
use siumai_core::types::{HttpConfig, HttpResponseInfo, JSONValue, Warning};
#[cfg(test)]
use std::collections::HashMap;

pub use siumai_core::image::{ImageModel, ImageModelV4};
pub use siumai_core::types::{
    GenerateImagePrompt, GenerateImageRequest, GenerateImageResult, GeneratedFile, GeneratedImage,
    ImageEditInput, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageModelProviderMetadata, ImageModelResponseMetadata, ImageModelUsage, ImageVariationRequest,
};

pub use self::workflow::GenerateOptions;
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
