//! OpenAI portable text embedding model.

use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId, ModelOperation,
    ProviderOptionContext, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, ProviderScope, TypedProviderOptions, Warning,
    WarningKind,
};
use siumai_protocol_openai::embedding::{
    API_MODE_ID, EmbeddingConfig, TARGET, decode_embedding_response, encode_embedding_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use super::http_error::{request_build_error, response_error};
use super::provider::OpenAiRuntime;

/// Current small OpenAI text embedding model hint.
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// Current large OpenAI text embedding model hint.
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// Legacy OpenAI text embedding model hint.
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

const MAX_INPUTS_PER_CALL: usize = 2_048;
const MAX_INPUT_TOKENS: u64 = 8_192;
const MAX_USER_BYTES: usize = 2_048;

/// Provider-owned controls for OpenAI text embeddings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiEmbeddingOptions {
    /// Stable end-user identifier forwarded to OpenAI abuse monitoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAiEmbeddingOptions {
    pub const fn new() -> Self {
        Self { user: None }
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Result<Self, ProviderOptionError> {
        self.user = Some(user.into());
        self.validate()?;
        Ok(self)
    }
}

impl TypedProviderOptions for OpenAiEmbeddingOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Embedding;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(user) = self.user.as_deref()
            && (user.trim().is_empty()
                || user.len() > MAX_USER_BYTES
                || user.chars().any(char::is_control))
        {
            return Err(ProviderOptionError::Rejected {
                path: "user".to_string(),
                reason:
                    "must be non-empty, contain no control characters, and be at most 2048 bytes"
                        .to_string(),
            });
        }
        Ok(())
    }
}

/// Lightweight OpenAI text embedding handle.
#[derive(Clone)]
pub struct OpenAiEmbeddingModel {
    runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
    defaults: OpenAiEmbeddingOptions,
}

impl OpenAiEmbeddingModel {
    pub(crate) fn new(
        runtime: Arc<OpenAiRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
        defaults: OpenAiEmbeddingOptions,
    ) -> Self {
        let instance_id = runtime.instance_id.clone();
        Self {
            runtime,
            descriptor: ModelDescriptor::from_scope(
                scope,
                model,
                ModelFamily::Embedding,
                instance_id,
            ),
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<OpenAiEmbeddingOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                ProviderOptionContext::new(
                    self.provider_id(),
                    ModelFamily::Embedding,
                    self.descriptor.scope().api_mode(),
                ),
                &OpenAiEmbeddingOptionMerger {
                    defaults: self.defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &EmbeddingRequest,
        options: &OpenAiEmbeddingOptions,
    ) -> Result<RequestPlan, Error> {
        validate_dimensions(self.model_id(), request)?;
        let mut config = EmbeddingConfig::new();
        if let Some(user) = options.user.as_deref() {
            config = config.with_user(user);
        }
        let body = encode_embedding_request(request, self.model_id(), &config)?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(|source| {
                request_build_error(
                    "OpenAI embedding request violates the transport contract",
                    source,
                )
            })?;
        RequestPlan::new(
            Method::POST,
            RequestTarget::new(TARGET).map_err(|source| {
                request_build_error(
                    "OpenAI embedding request violates the transport contract",
                    source,
                )
            })?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(|source| {
            request_build_error(
                "OpenAI embedding request violates the transport contract",
                source,
            )
        })?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(|source| {
            request_build_error(
                "OpenAI embedding request violates the transport contract",
                source,
            )
        })
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Embed),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl std::fmt::Debug for OpenAiEmbeddingModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiEmbeddingModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiEmbeddingModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for OpenAiEmbeddingModel {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(MAX_INPUTS_PER_CALL),
            max_input_tokens: Some(MAX_INPUT_TOKENS),
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        call: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let options = self
            .options(&call)
            .map_err(|error| self.contextualize(error))?;
        let plan = self
            .plan(&request, &options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                "OpenAI rejected the embedding request",
                response,
            )));
        }
        let (_, headers, body) = response.into_parts();
        let mut decoded = decode_embedding_response(&body, &request, self.model_id())
            .map_err(|error| self.contextualize(error))?;
        decoded.metadata.request_id = response_request_id(&headers);
        if !is_verified_model(self.descriptor.scope(), self.model_id()) {
            decoded.warnings.push(unknown_model_warning());
        }
        Ok(decoded)
    }
}

struct OpenAiEmbeddingOptionMerger {
    defaults: OpenAiEmbeddingOptions,
}

impl ProviderOptionMerger for OpenAiEmbeddingOptionMerger {
    type Output = OpenAiEmbeddingOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options).and_then(|options| options.validate())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            let options = decode_options(options)?;
            if options.user.is_some() {
                merged.user = options.user;
            }
        }
        merged.validate()?;
        Ok(merged)
    }
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<OpenAiEmbeddingOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "openai".to_string(),
            reason: "options do not match the OpenAI embedding schema".to_string(),
        }
    })
}

fn is_verified_model(scope: &ProviderScope, model: &ModelId) -> bool {
    scope.platform().map(|value| value.as_str()) == Some("openai-api")
        && matches!(
            model.as_str(),
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_3_LARGE | TEXT_EMBEDDING_ADA_002
        )
}

fn validate_dimensions(model: &ModelId, request: &EmbeddingRequest) -> Result<(), Error> {
    if request.dimensions().is_some()
        && !matches!(
            model.as_str(),
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_3_LARGE
        )
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI embedding dimensions are verified only for text-embedding-3 models",
        ));
    }
    Ok(())
}

fn unknown_model_warning() -> Warning {
    Warning::new(
        WarningKind::UnknownModel,
        "model support is not verified for OpenAI embeddings",
    )
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for OpenAI embeddings",
    )
    .with_source(source)
}

fn response_request_id(headers: &siumai_transport::ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 256
                    && value.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
                    })
            })
            .map(str::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_reject_unbounded_or_empty_user_ids() {
        assert!(OpenAiEmbeddingOptions::new().with_user("user-1").is_ok());
        assert!(OpenAiEmbeddingOptions::new().with_user("\n").is_err());
        assert!(
            OpenAiEmbeddingOptions::new()
                .with_user("x".repeat(MAX_USER_BYTES + 1))
                .is_err()
        );
    }

    #[test]
    fn dimensions_fail_closed_for_legacy_and_unknown_models() {
        let request = EmbeddingRequest::new(["hello"])
            .unwrap()
            .with_dimensions(256)
            .unwrap();

        assert!(
            validate_dimensions(&ModelId::new(TEXT_EMBEDDING_3_SMALL).unwrap(), &request).is_ok()
        );
        assert!(
            validate_dimensions(&ModelId::new(TEXT_EMBEDDING_ADA_002).unwrap(), &request).is_err()
        );
        assert!(
            validate_dimensions(&ModelId::new("future-embedding-model").unwrap(), &request)
                .is_err()
        );
    }
}
