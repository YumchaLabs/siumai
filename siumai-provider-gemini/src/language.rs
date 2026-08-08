use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use http::Method;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use serde_json::Value;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamDecoder, LanguageStreamEvent, Model, ModelAdvisory,
    ModelDescriptor, ModelFamily, ModelId, ModelOperation, ModelPolicy, ProviderOptionContext,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, StreamTerminal, SupportState, Warning, WarningKind,
};
use siumai_protocol_gemini::interactions::{
    DecodedInteraction, InteractionLanguageConfig, InteractionStorage, InteractionThinkingLevel,
    InteractionThinkingSummaries, InteractionsStreamDecoder, STABLE_V1_LANGUAGE_TARGET,
    decode_language_response, encode_language_request,
};
use siumai_transport::framing::SseDecoder;
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, TransportByteStream,
};

use crate::http::{
    response_diagnostics, response_error, response_request_id, sse_error, stream_response_error,
};
use crate::options::{
    GeminiInteractionStorage, GeminiInteractionsOptions, GeminiThinkingLevel,
    GeminiThinkingSummaries,
};
use crate::provider::ProviderRuntime;

/// Lightweight stable-v1 Gemini Interactions language handle.
#[derive(Clone)]
pub struct GeminiLanguageModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GeminiLanguageModel {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>, model: ModelId) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            runtime.interactions_scope.clone(),
            model,
            ModelFamily::Language,
        );
        Self {
            runtime,
            descriptor,
        }
    }

    fn policy(&self, operation: ModelOperation) -> Result<Vec<Warning>, Error> {
        let decision =
            self.runtime
                .interactions_policy
                .evaluate(&siumai_core::ModelPolicyContext::new(
                    self.runtime.interactions_scope.clone(),
                    self.model_id().clone(),
                    operation,
                ));
        if let SupportState::Unsupported { .. } = decision.state() {
            return Err(self.contextualize(
                operation,
                Error::new(
                    ErrorKind::Unsupported,
                    "model policy rejected the Gemini Interactions language operation",
                ),
            ));
        }
        Ok(decision.advisories().iter().map(advisory_warning).collect())
    }

    fn options(&self, call: &CallOptions) -> Result<GeminiInteractionsOptions, Error> {
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
                &GeminiInteractionsOptionMerger {
                    defaults: self.runtime.interactions_defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &LanguageRequest,
        options: &GeminiInteractionsOptions,
        stream: bool,
    ) -> Result<RequestPlan, Error> {
        let config = protocol_config(options);
        let body = encode_language_request(
            request,
            self.model_id(),
            self.descriptor.scope(),
            &config,
            stream,
        )?;
        let accept = if stream {
            HeaderValue::from_static("text/event-stream")
        } else {
            HeaderValue::from_static("application/json")
        };
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, accept)
            .map_err(request_build_error)?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(STABLE_V1_LANGUAGE_TARGET).map_err(request_build_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_build_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_build_error)
    }

    fn contextualize(&self, operation: ModelOperation, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(operation),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }

    /// Execute one direct Interactions call while retaining the complete native resource.
    pub async fn generate_native(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<DecodedInteraction, Error> {
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
                response_error(
                    response,
                    "Gemini rejected the Interactions language request",
                ),
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
}

impl std::fmt::Debug for GeminiLanguageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeminiLanguageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for GeminiLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        let operation = ModelOperation::Generate;
        self.generate_native(request, options)
            .await?
            .into_result()
            .map_err(|error| self.contextualize(operation, error))
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
                stream_response_error(response, "Gemini rejected the Interactions language stream")
                    .await,
            ));
        }
        let status = response.status();
        let headers = response.headers().clone();
        let diagnostics = response_diagnostics(status, &headers);
        let body = response.into_body();
        let mut decoder = InteractionsStreamDecoder::new(
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

pub(crate) fn decode_sse_stream<D>(
    cancellation: siumai_core::Cancellation,
    body: TransportByteStream,
    limits: siumai_transport::TransportLimits,
    mut protocol: D,
    headers: siumai_transport::ResponseHeaders,
    warnings: Vec<Warning>,
    context: ErrorContext,
) -> LanguageStream
where
    D: LanguageStreamDecoder<ProtocolFrame = str> + Send + 'static,
{
    established_stream(cancellation, move |_| {
        async_stream::try_stream! {
            let mut body = body;
            let mut framing = SseDecoder::new(&limits);
            while let Some(chunk) = body.next().await {
                let chunk = chunk.map_err(|error| error.with_context(context.clone()))?;
                let frames = framing
                    .push(&chunk)
                    .map_err(|source| sse_error(source).with_context(context.clone()))?;
                for frame in frames {
                    let events = protocol
                        .decode(frame.data())
                        .map_err(|error| error.with_context(context.clone()))?;
                    for mut event in events {
                        contextualize_terminal_error(&mut event, &context);
                        attach_response_context(&mut event, &headers, &warnings);
                        let terminal = event.terminal().is_some();
                        yield event;
                        if terminal {
                            return;
                        }
                    }
                }
            }
            framing
                .finish()
                .map_err(|source| sse_error(source).with_context(context.clone()))?;
            let events = protocol
                .finish()
                .map_err(|error| error.with_context(context.clone()))?;
            for mut event in events {
                contextualize_terminal_error(&mut event, &context);
                attach_response_context(&mut event, &headers, &warnings);
                let terminal = event.terminal().is_some();
                yield event;
                if terminal {
                    return;
                }
            }
        }
    })
}

pub(crate) fn with_response_context(
    response: LanguageResponse,
    headers: &siumai_transport::ResponseHeaders,
    warnings: &[Warning],
) -> LanguageResponse {
    let mut provider = response.provider_metadata().clone();
    if let Some(request_id) = response_request_id(headers) {
        provider.insert("google.request_id".to_string(), Value::String(request_id));
    }
    if let Some(service_tier) = response_header(headers, "x-gemini-service-tier") {
        provider.insert(
            "google.service_tier".to_string(),
            Value::String(service_tier),
        );
    }
    let mut combined_warnings = response.warnings().to_vec();
    combined_warnings.extend_from_slice(warnings);
    response
        .with_provider_metadata(provider)
        .with_warnings(combined_warnings)
}

fn attach_response_context(
    event: &mut LanguageStreamEvent,
    headers: &siumai_transport::ResponseHeaders,
    warnings: &[Warning],
) {
    let response = match event {
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })
        | LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            response: Some(response),
            ..
        })
        | LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            response: Some(response),
            ..
        }) => response,
        _ => return,
    };
    **response = with_response_context(response.as_ref().clone(), headers, warnings);
}

fn contextualize_terminal_error(event: &mut LanguageStreamEvent, context: &ErrorContext) {
    let LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. }) = event else {
        return;
    };
    let original = std::mem::replace(
        error,
        Error::new(
            ErrorKind::Internal,
            "stream error context replacement failed",
        ),
    );
    *error = original.with_context(context.clone());
}

fn response_header(
    headers: &siumai_transport::ResponseHeaders,
    name: &'static str,
) -> Option<String> {
    headers
        .get(&HeaderName::from_static(name))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty() && value.len() <= 4 * 1024)
        .map(ToOwned::to_owned)
}

fn protocol_config(options: &GeminiInteractionsOptions) -> InteractionLanguageConfig {
    let mut config =
        InteractionLanguageConfig::new().with_storage(match options.storage.unwrap_or_default() {
            GeminiInteractionStorage::Disabled => InteractionStorage::Disabled,
            GeminiInteractionStorage::Enabled => InteractionStorage::Enabled,
        });
    if let Some(level) = options.thinking_level {
        config = config.with_thinking_level(match level {
            GeminiThinkingLevel::Minimal => InteractionThinkingLevel::Minimal,
            GeminiThinkingLevel::Low => InteractionThinkingLevel::Low,
            GeminiThinkingLevel::Medium => InteractionThinkingLevel::Medium,
            GeminiThinkingLevel::High => InteractionThinkingLevel::High,
        });
    }
    if let Some(summaries) = options.thinking_summaries {
        config = config.with_thinking_summaries(match summaries {
            GeminiThinkingSummaries::Auto => InteractionThinkingSummaries::Auto,
            GeminiThinkingSummaries::None => InteractionThinkingSummaries::None,
        });
    }
    config
}

struct GeminiInteractionsOptionMerger {
    defaults: GeminiInteractionsOptions,
}

impl ProviderOptionMerger for GeminiInteractionsOptionMerger {
    type Output = GeminiInteractionsOptions;

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
            if value.storage.is_some() {
                merged.storage = value.storage;
            }
            if value.thinking_level.is_some() {
                merged.thinking_level = value.thinking_level;
            }
            if value.thinking_summaries.is_some() {
                merged.thinking_summaries = value.thinking_summaries;
            }
        }
        Ok(merged)
    }
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<GeminiInteractionsOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "google".to_string(),
            reason: "options do not match the Gemini Interactions language schema".to_string(),
        }
    })
}

fn advisory_warning(advisory: &ModelAdvisory) -> Warning {
    match advisory {
        ModelAdvisory::UnknownModel => Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the current Gemini Interactions advisory catalog",
        ),
        ModelAdvisory::Deprecated { .. } => Warning::new(
            WarningKind::DeprecatedModel,
            "Gemini Interactions model is deprecated",
        ),
        ModelAdvisory::Retired { .. } => Warning::new(
            WarningKind::RetiredModel,
            "Gemini Interactions model is retired",
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
        "provider options are invalid for Gemini Interactions language generation",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini Interactions language request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use siumai_core::{
        CallOptions, ContentPart, LanguageModel, Message, ProviderOptions, ReplayDomain,
        ReplayDomainId, ToolSpec,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GeminiCredential, GeminiProvider};

    fn provider(base_url: String) -> GeminiProvider {
        GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-language-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn direct_and_stream_paths_share_the_interactions_contract() {
        let mut server = mockito::Server::new_async().await;
        let direct = server
            .mock("POST", "/v1/interactions")
            .match_header("x-goog-api-key", "test-key")
            .match_header("accept", "application/json")
            .match_body(mockito::Matcher::Regex("\\\"store\\\":false".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "id": "interaction-direct",
                    "status": "completed",
                    "model": "gemini-3.6-flash",
                    "steps": [{
                        "type": "model_output",
                        "content": [{"type": "text", "text": "hello"}]
                    }],
                    "usage": {"total_input_tokens": 2, "total_output_tokens": 1}
                })
                .to_string(),
            )
            .expect(2)
            .create_async()
            .await;
        let stream = server
            .mock("POST", "/v1/interactions")
            .match_header("accept", "text/event-stream")
            .match_body(mockito::Matcher::Regex("\\\"stream\\\":true".to_string()))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(
                [
                    serde_json::json!({
                        "event_type": "interaction.created",
                        "interaction": {
                            "id": "interaction-stream",
                            "status": "in_progress",
                            "model": "gemini-3.6-flash"
                        }
                    }),
                    serde_json::json!({
                        "event_type": "step.start",
                        "index": 0,
                        "step": {"type": "model_output"}
                    }),
                    serde_json::json!({
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "text", "text": "hello"}
                    }),
                    serde_json::json!({"event_type": "step.stop", "index": 0}),
                    serde_json::json!({
                        "event_type": "interaction.completed",
                        "interaction": {
                            "id": "interaction-stream",
                            "status": "completed",
                            "model": "gemini-3.6-flash",
                            "usage": {"total_input_tokens": 2, "total_output_tokens": 1}
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
        let model = provider.language("gemini-3.6-flash").unwrap();
        let options = ProviderOptions::typed(
            &GeminiInteractionsOptions::new()
                .with_thinking_summaries(GeminiThinkingSummaries::Auto),
        )
        .unwrap();
        let call = CallOptions::default().with_provider_options(options);
        let request = LanguageRequest::new(vec![Message::user("hello")]);

        let native = model
            .generate_native(request.clone(), call.clone())
            .await
            .unwrap();
        assert_eq!(native.native()["id"], "interaction-direct");
        let response = model.generate(request.clone(), call.clone()).await.unwrap();
        assert!(matches!(
            &response.content()[0],
            ContentPart::Text { text } if text == "hello"
        ));

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
            panic!("expected a completed Gemini stream");
        };
        assert!(matches!(
            &response.content()[0],
            ContentPart::Text { text } if text == "hello"
        ));
        direct.assert_async().await;
        stream.assert_async().await;
    }

    #[test]
    fn typed_options_and_tool_schema_stay_provider_owned() {
        let options = ProviderOptions::typed(
            &GeminiInteractionsOptions::new()
                .with_storage(GeminiInteractionStorage::Enabled)
                .with_thinking_level(GeminiThinkingLevel::High),
        )
        .unwrap();
        assert_eq!(options.namespace().as_str(), "google");
        assert!(ToolSpec::new("lookup", None, serde_json::json!({"type": "object"})).is_ok());
    }
}
