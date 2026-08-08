use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamDecoder, Model, ModelAdvisory, ModelDescriptor, ModelFamily,
    ModelId, ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision,
    ProviderOptionContext, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, SupportState, TypedProviderOptions, Warning,
    WarningKind,
};
use siumai_protocol_gemini::generate_content::{
    DecodedGenerateContent, GenerateContentLanguageConfig, GenerateContentStreamDecoder,
    GenerateContentThinkingConfig, LEGACY_STABLE_V1_GENERATE_CONTENT_TARGET,
    LEGACY_STABLE_V1_STREAM_GENERATE_CONTENT_TARGET, decode_language_response,
    encode_language_request,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
};

use crate::http::{response_diagnostics, response_error, stream_response_error};
use crate::language::{decode_sse_stream, with_response_context};
use crate::models::is_current_interactions;
use crate::profile::{GENERATE_CONTENT_API_MODE_ID, GENERATE_CONTENT_PROTOCOL_ID, PROVIDER_ID};
use crate::provider::ProviderRuntime;

/// Provider API mode identifier for the stable-v1 Generate Content compatibility surface.
pub const GEMINI_GENERATE_CONTENT_API_MODE_ID: &str = GENERATE_CONTENT_API_MODE_ID;

/// Thinking level accepted by the Generate Content API.
pub use siumai_protocol_gemini::generate_content::GenerateContentThinkingLevel as GeminiGenerateContentThinkingLevel;

/// Service tier accepted by the Generate Content API.
pub use siumai_protocol_gemini::generate_content::GenerateContentServiceTier as GeminiGenerateContentServiceTier;

/// Provider-owned thinking controls for Generate Content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiGenerateContentThinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<GeminiGenerateContentThinkingLevel>,
}

impl GeminiGenerateContentThinking {
    pub const fn new() -> Self {
        Self {
            include_thoughts: None,
            budget: None,
            level: None,
        }
    }

    pub const fn with_include_thoughts(mut self, include_thoughts: bool) -> Self {
        self.include_thoughts = Some(include_thoughts);
        self
    }

    pub const fn with_budget(mut self, budget: i32) -> Self {
        self.budget = Some(budget);
        self
    }

    pub const fn with_level(mut self, level: GeminiGenerateContentThinkingLevel) -> Self {
        self.level = Some(level);
        self
    }
}

/// Provider-owned options for the explicit Generate Content language mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiGenerateContentOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<GeminiGenerateContentServiceTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<GeminiGenerateContentThinking>,
}

impl GeminiGenerateContentOptions {
    pub const fn new() -> Self {
        Self {
            service_tier: None,
            store: None,
            top_k: None,
            thinking: None,
        }
    }

    pub const fn with_service_tier(mut self, tier: GeminiGenerateContentServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub const fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    pub const fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_thinking(mut self, thinking: GeminiGenerateContentThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }
}

impl TypedProviderOptions for GeminiGenerateContentOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(GENERATE_CONTENT_API_MODE_ID);
}

/// Lightweight handle for the explicit stable-v1 Generate Content language mode.
#[derive(Clone)]
pub struct GeminiGenerateContentModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GeminiGenerateContentModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.generate_content_scope.clone(),
            model,
            ModelFamily::Language,
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(&self, operation: ModelOperation) -> Result<Vec<Warning>, Error> {
        let decision = self
            .runtime
            .generate_content_policy
            .evaluate(&ModelPolicyContext::new(
                self.runtime.generate_content_scope.clone(),
                self.model_id().clone(),
                operation,
            ));
        if matches!(decision.state(), SupportState::Unsupported { .. }) {
            return Err(self.contextualize(
                operation,
                Error::new(
                    ErrorKind::Unsupported,
                    "model policy rejected the Gemini Generate Content language operation",
                ),
            ));
        }
        Ok(decision.advisories().iter().map(advisory_warning).collect())
    }

    fn options(&self, call: &CallOptions) -> Result<GeminiGenerateContentOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                ProviderOptionContext::new(
                    self.provider_id(),
                    ModelFamily::Language,
                    self.descriptor.scope().api_mode(),
                ),
                &GeminiGenerateContentOptionMerger {
                    defaults: self.runtime.generate_content_defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        options: &GeminiGenerateContentOptions,
        stream: bool,
    ) -> Result<RequestPlan, Error> {
        let config = protocol_config(options);
        let body =
            encode_language_request(request, self.model_id(), self.descriptor.scope(), &config)?;
        let target = generate_content_target(self.model_id(), stream)?;
        let accept = if stream {
            HeaderValue::from_static("text/event-stream")
        } else {
            HeaderValue::from_static("application/json")
        };
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, accept)
            .map_err(request_build_error)?;
        RequestPlan::new(Method::POST, target)
            .with_headers(headers)
            .with_body(RequestBody::json(&body).map_err(request_build_error)?)
            .with_replay_safety(ReplaySafety::Never)
            .map_err(request_build_error)
    }

    /// Execute a direct Generate Content call while retaining its native JSON response.
    pub async fn generate_native(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<DecodedGenerateContent, Error> {
        let operation = ModelOperation::Generate;
        let warnings = self.policy(operation)?;
        let provider_options = self
            .options(&options)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, &provider_options, false)
            .map_err(|error| self.contextualize(operation, error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                response_error(response, "Gemini rejected the Generate Content request"),
            ));
        }
        let (_, headers, body) = response.into_parts();
        decode_language_response(&body, self.descriptor.scope(), self.model_id())
            .map(|decoded| {
                decoded.map_canonical(|canonical| {
                    with_response_context(canonical, &headers, &warnings)
                })
            })
            .map_err(|error| self.contextualize(operation, error))
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(operation),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for GeminiGenerateContentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiGenerateContentModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiGenerateContentModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for GeminiGenerateContentModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        self.generate_native(request, options).await?.into_result()
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let operation = ModelOperation::Stream;
        let warnings = self.policy(operation)?;
        let provider_options = self
            .options(&options)
            .map_err(|error| self.contextualize(operation, error))?;
        let plan = self
            .plan(&request, &provider_options, true)
            .map_err(|error| self.contextualize(operation, error))?;
        let cancellation = options.cancellation().clone();
        let response = self
            .runtime
            .transport
            .execute_stream(plan, options)
            .await
            .map_err(|error| self.contextualize(operation, error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(
                operation,
                stream_response_error(response, "Gemini rejected the Generate Content stream")
                    .await,
            ));
        }
        let status = response.status();
        let headers = response.headers().clone();
        let diagnostics = response_diagnostics(status, &headers);
        let body = response.into_body();
        let mut decoder = GenerateContentStreamDecoder::new(
            self.descriptor.scope().clone(),
            self.model_id().clone(),
        );
        decoder.set_response_diagnostics(diagnostics);
        let context = ErrorContext {
            operation: Some(operation),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        };
        Ok(decode_sse_stream(
            cancellation,
            body,
            self.runtime.limits.clone(),
            decoder,
            headers,
            warnings,
            context,
        ))
    }
}

impl ModelPolicy for GeminiGenerateContentPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope().provider_id().as_str() == PROVIDER_ID
            && context.scope().protocol().map(|value| value.as_str())
                == Some(GENERATE_CONTENT_PROTOCOL_ID)
            && context.scope().api_mode().map(|value| value.as_str())
                == Some(GENERATE_CONTENT_API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(
                siumai_core::UnsupportedReason::ApiModeMismatch,
            );
        }
        if !matches!(
            context.operation(),
            ModelOperation::Generate | ModelOperation::Stream
        ) {
            return ModelPolicyDecision::unsupported(
                siumai_core::UnsupportedReason::OperationNotImplemented,
            );
        }
        if self.verified_endpoint && is_current_interactions(context.model().as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

pub(crate) struct GeminiGenerateContentPolicy {
    pub(crate) verified_endpoint: bool,
}

fn generate_content_target(model: &ModelId, stream: bool) -> Result<RequestTarget, Error> {
    let encoded = urlencoding::encode(model.as_str());
    let template = if stream {
        LEGACY_STABLE_V1_STREAM_GENERATE_CONTENT_TARGET
    } else {
        LEGACY_STABLE_V1_GENERATE_CONTENT_TARGET
    };
    RequestTarget::new(template.replace("{model}", encoded.as_ref())).map_err(request_build_error)
}

fn protocol_config(options: &GeminiGenerateContentOptions) -> GenerateContentLanguageConfig {
    let mut config = GenerateContentLanguageConfig::new();
    if let Some(tier) = options.service_tier {
        config = config.with_service_tier(tier);
    }
    if let Some(store) = options.store {
        config = config.with_store(store);
    }
    if let Some(top_k) = options.top_k {
        config = config.with_top_k(top_k);
    }
    if let Some(thinking) = options.thinking {
        let mut value = GenerateContentThinkingConfig::new();
        if let Some(include) = thinking.include_thoughts {
            value = value.with_include_thoughts(include);
        }
        if let Some(budget) = thinking.budget {
            value = value.with_budget(budget);
        }
        if let Some(level) = thinking.level {
            value = value.with_level(level);
        }
        config = config.with_thinking(value);
    }
    config
}

struct GeminiGenerateContentOptionMerger {
    defaults: GeminiGenerateContentOptions,
}

impl ProviderOptionMerger for GeminiGenerateContentOptionMerger {
    type Output = GeminiGenerateContentOptions;

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
            if value.service_tier.is_some() {
                merged.service_tier = value.service_tier;
            }
            if value.store.is_some() {
                merged.store = value.store;
            }
            if value.top_k.is_some() {
                merged.top_k = value.top_k;
            }
            if value.thinking.is_some() {
                merged.thinking = value.thinking;
            }
        }
        Ok(merged)
    }
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<GeminiGenerateContentOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "google".to_string(),
            reason: "options do not match the Gemini Generate Content language schema".to_string(),
        }
    })
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the current Gemini Generate Content advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "Gemini Generate Content model is deprecated",
        ),
        ModelAdvisory::Retired { .. } => Warning::new(
            WarningKind::RetiredModel,
            "Gemini Generate Content model is retired",
        ),
        ModelAdvisory::RollingAlias => Warning::new(
            WarningKind::RollingModelAlias,
            "Gemini model ID is a rolling alias",
        ),
        _ => Warning::provider("model_advisory", "Gemini returned a model advisory"),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Gemini Generate Content language generation",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini Generate Content language request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use siumai_core::{
        CallOptions, ContentPart, LanguageModel, LanguageStreamEvent, Message, ProviderOptions,
        ReplayDomain, ReplayDomainId, StreamTerminal,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GeminiCredential, GeminiProvider};

    fn provider(base_url: String) -> GeminiProvider {
        GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-generate-content-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn explicit_mode_uses_stable_v1_direct_and_stream_targets() {
        let mut server = mockito::Server::new_async().await;
        let direct = server
            .mock("POST", "/v1/models/gemini-3.6-flash:generateContent")
            .match_header("x-goog-api-key", "test-key")
            .match_header("accept", "application/json")
            .match_body(mockito::Matcher::Regex(
                "\\\"serviceTier\\\":\\\"priority\\\"".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "responseId": "generate-direct",
                    "modelVersion": "gemini-3.6-flash",
                    "candidates": [{
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "hello"}]},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 2,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 3
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
        let stream = server
            .mock("POST", "/v1/models/gemini-3.6-flash:streamGenerateContent")
            .match_query("alt=sse")
            .match_header("accept", "text/event-stream")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(
                [
                    serde_json::json!({
                        "responseId": "generate-stream",
                        "modelVersion": "gemini-3.6-flash",
                        "candidates": [{
                            "index": 0,
                            "content": {"role": "model", "parts": [{"text": "hello"}]},
                            "finishReason": "STOP"
                        }]
                    }),
                    serde_json::json!({
                        "responseId": "generate-stream",
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 3
                        }
                    }),
                ]
                .into_iter()
                .map(|value| format!("data: {value}\n\n"))
                .collect::<String>(),
            )
            .create_async()
            .await;
        let provider = provider(server.url());
        let model = provider.generate_content("gemini-3.6-flash").unwrap();
        assert_eq!(model.descriptor().api_mode(), Some("generate-content"));
        assert_eq!(
            provider
                .language("gemini-3.6-flash")
                .unwrap()
                .descriptor()
                .api_mode(),
            Some("interactions")
        );
        let options = ProviderOptions::typed(
            &GeminiGenerateContentOptions::new()
                .with_service_tier(GeminiGenerateContentServiceTier::Priority),
        )
        .unwrap();
        let call = CallOptions::default().with_provider_options(options);
        let request = LanguageRequest::new(vec![Message::user("hello")]);

        let native = model
            .generate_native(request.clone(), call.clone())
            .await
            .unwrap();
        assert_eq!(native.native()["responseId"], "generate-direct");

        let events = model
            .stream(request, call)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let terminals = events
            .iter()
            .filter_map(LanguageStreamEvent::terminal)
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1);
        let StreamTerminal::Completed { response } = terminals[0] else {
            panic!("expected a completed Generate Content stream");
        };
        assert!(matches!(
            &response.content()[0],
            ContentPart::Text { text } if text == "hello"
        ));
        direct.assert_async().await;
        stream.assert_async().await;
    }
}
