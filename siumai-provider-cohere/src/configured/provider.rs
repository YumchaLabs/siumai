use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;
use siumai_core::{
    ApiModeId, EmbeddingModel, EmbeddingModelProvider, InvalidId, ModelFamily, ModelId,
    ModelLookupError, ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision,
    ProtocolId, Provider, ProviderId, ProviderRegistration, ProviderScope, RerankModel,
    RerankModelProvider, UnsupportedReason,
};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, ProviderTransport, ReplaySafety, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error;

use super::auth::{CohereBearerAuth, validate_api_key};
use super::model::{CohereEmbeddingModel, CohereRerankModel};

const COHERE_ORIGIN: &str = "https://api.cohere.com";
const COHERE_V2_BASE_URL: &str = "https://api.cohere.com/v2";

const KNOWN_EMBEDDING_MODELS: &[&str] = &[
    "embed-v4.0",
    "embed-english-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-light-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
];

const KNOWN_RERANK_MODELS: &[&str] = &[
    "rerank-v3.5",
    "rerank-english-v3.0",
    "rerank-multilingual-v3.0",
];

/// A synchronously configured Cohere v2 provider.
#[derive(Clone)]
pub struct CohereProvider {
    pub(crate) runtime: Arc<CohereRuntime>,
}

impl CohereProvider {
    /// Start configuring Cohere with a static API key.
    pub fn builder(api_key: impl Into<String>) -> CohereProviderBuilder {
        CohereProviderBuilder::new(api_key)
    }

    /// Create a lightweight embedding model from a textual open model ID.
    pub fn embedding(
        &self,
        model: impl Into<String>,
    ) -> Result<CohereEmbeddingModel, ModelLookupError> {
        let model = parse_model_id(model)?;
        Ok(self.create_embedding_model(model))
    }

    /// Construct the canonical embedding family handle.
    pub fn embedding_model(
        &self,
        model: ModelId,
    ) -> Result<CohereEmbeddingModel, ModelLookupError> {
        Ok(self.create_embedding_model(model))
    }

    /// Create a lightweight rerank model from a textual open model ID.
    pub fn reranker(
        &self,
        model: impl Into<String>,
    ) -> Result<CohereRerankModel, ModelLookupError> {
        let model = parse_model_id(model)?;
        Ok(self.create_rerank_model(model))
    }

    /// Construct the canonical rerank family handle.
    pub fn rerank_model(&self, model: ModelId) -> Result<CohereRerankModel, ModelLookupError> {
        Ok(self.create_rerank_model(model))
    }

    /// Capture narrow factories backed by this provider's shared runtime.
    pub fn registration(&self) -> ProviderRegistration {
        let embedding_provider = self.clone();
        let rerank_provider = self.clone();
        ProviderRegistration::from_scope(self.runtime.scope.clone(), self.runtime.policy.clone())
            .with_embedding(Arc::new(move |model| {
                Ok(Arc::new(embedding_provider.create_embedding_model(model))
                    as Arc<dyn EmbeddingModel>)
            }))
            .with_rerank(Arc::new(move |model| {
                Ok(Arc::new(rerank_provider.create_rerank_model(model)) as Arc<dyn RerankModel>)
            }))
    }

    fn create_embedding_model(&self, model: ModelId) -> CohereEmbeddingModel {
        CohereEmbeddingModel::new(self.runtime.clone(), model)
    }

    fn create_rerank_model(&self, model: ModelId) -> CohereRerankModel {
        CohereRerankModel::new(self.runtime.clone(), model)
    }
}

impl Provider for CohereProvider {
    fn scope(&self) -> &ProviderScope {
        &self.runtime.scope
    }
}

impl EmbeddingModelProvider for CohereProvider {
    type Model = CohereEmbeddingModel;

    fn embedding_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_embedding_model(model))
    }
}

impl RerankModelProvider for CohereProvider {
    type Model = CohereRerankModel;

    fn rerank_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_rerank_model(model))
    }
}

impl fmt::Debug for CohereProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereProvider")
            .field("scope", &self.runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

/// Builder for one immutable Cohere provider runtime.
pub struct CohereProviderBuilder {
    api_key: SecretString,
    endpoint: Option<EndpointConfig>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
}

impl CohereProviderBuilder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            endpoint: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
        }
    }

    /// Replace the official endpoint with an explicitly validated endpoint.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = Some(timeout);
        self
    }

    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = Some(timeout);
        self
    }

    /// Validate static settings and build one shared transport runtime.
    pub fn build(self) -> Result<CohereProvider, CohereConfigError> {
        if !validate_api_key(&self.api_key) {
            return Err(CohereConfigError::InvalidApiKey);
        }
        let provider_id = ProviderId::new("cohere")?;
        let scope = Arc::new(
            ProviderScope::new(provider_id)
                .with_protocol(ProtocolId::new("cohere-native")?)
                .with_api_mode(ApiModeId::new("v2")?),
        );
        let endpoint = match self.endpoint {
            Some(endpoint) => endpoint,
            None => default_endpoint()?,
        };
        let mut transport = ProviderTransport::builder(endpoint)
            .with_auth(Arc::new(CohereBearerAuth::new(self.api_key)))
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            transport = transport.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            transport = transport.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            transport = transport.with_read_timeout(timeout);
        }
        let transport = transport.build()?;
        Ok(CohereProvider {
            runtime: Arc::new(CohereRuntime {
                scope,
                transport,
                policy: Arc::new(CohereModelPolicy),
                replay_safety: ReplaySafety::Never,
            }),
        })
    }
}

impl fmt::Debug for CohereProviderBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereProviderBuilder")
            .field("api_key", &"[REDACTED]")
            .field("has_custom_endpoint", &self.endpoint.is_some())
            .field("limits", &self.limits)
            .field("retry_policy", &self.retry_policy)
            .field("connect_timeout", &self.connect_timeout)
            .field("call_timeout", &self.call_timeout)
            .field("read_timeout", &self.read_timeout)
            .finish()
    }
}

pub(crate) struct CohereRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<CohereModelPolicy>,
    pub(crate) replay_safety: ReplaySafety,
}

impl fmt::Debug for CohereRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("replay_safety", &self.replay_safety)
            .finish()
    }
}

pub(crate) struct CohereModelPolicy;

impl ModelPolicy for CohereModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let operation_matches = matches!(
            (context.family, context.operation),
            (ModelFamily::Embedding, ModelOperation::Embed)
                | (ModelFamily::Rerank, ModelOperation::Rerank)
        );
        if !operation_matches {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }

        let known = match context.family {
            ModelFamily::Embedding => KNOWN_EMBEDDING_MODELS.contains(&context.model.as_str()),
            ModelFamily::Rerank => KNOWN_RERANK_MODELS.contains(&context.model.as_str()),
            _ => false,
        };
        if known {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

fn parse_model_id(model: impl Into<String>) -> Result<ModelId, ModelLookupError> {
    ModelId::new(model.into())
        .map_err(|error| ModelLookupError::InvalidReference(error.to_string()))
}

fn default_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(COHERE_V2_BASE_URL, OfficialOrigin::new(COHERE_ORIGIN)?)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CohereConfigError {
    #[error("invalid Cohere provider identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("Cohere API key is empty or contains invalid bytes")]
    InvalidApiKey,
    #[error("invalid Cohere endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Cohere transport settings: {0}")]
    Transport(#[from] TransportConfigError),
}
