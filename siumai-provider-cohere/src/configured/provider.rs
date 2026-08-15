use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;
use siumai_core::{
    ApiStability, EmbeddingModel, EmbeddingModelProvider, InvalidId, ModelId, ModelLookupError,
    NativeSupportScope, NativeSurfaceId, NativeSurfaceKind, NativeVerificationEvidence,
    OfficialSource, Provider, ProviderInstanceId, ProviderRegistration, ProviderRegistrationError,
    ProviderScope, ProviderSupportManifest, RerankModel, RerankModelProvider, SupportManifestError,
    UpstreamLifecycle, UpstreamMaturity, UpstreamSupportStatus, VerificationDate, VerifiedFidelity,
    VerifiedNativeSupportClaim,
};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, ProviderTransport, ReplaySafety, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error;

use super::auth::{CohereBearerAuth, validate_api_key};
use super::model::{CohereEmbeddingModel, CohereRerankModel};
use super::profile::{CohereProfile, CohereProfileError};
use super::transcription::CohereTranscriptions;

const COHERE_ORIGIN: &str = "https://api.cohere.com";
const COHERE_V2_BASE_URL: &str = "https://api.cohere.com/v2";
const TRANSCRIPTION_SOURCE: &str = "https://docs.cohere.com/reference/create-audio-transcription";

/// A synchronously configured Cohere v2 provider.
#[derive(Clone)]
pub struct CohereProvider {
    pub(crate) runtime: Arc<CohereRuntime>,
    profile: CohereProfile,
    registration: ProviderRegistration,
    support_manifest: Arc<ProviderSupportManifest>,
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

    /// Access Cohere's model-less v2 audio transcription operation.
    pub fn transcriptions(&self) -> CohereTranscriptions {
        CohereTranscriptions::new(self.runtime.clone())
    }

    /// Capture narrow factories backed by this provider's shared runtime.
    pub fn registration(&self) -> ProviderRegistration {
        self.registration.clone()
    }

    fn build_registration(
        runtime: Arc<CohereRuntime>,
    ) -> Result<ProviderRegistration, ProviderRegistrationError> {
        let embedding_runtime = runtime.clone();
        let rerank_runtime = runtime.clone();
        ProviderRegistration::from_embedding(
            runtime.scope.clone(),
            Arc::new(move |model| {
                Ok(
                    Arc::new(CohereEmbeddingModel::new(embedding_runtime.clone(), model))
                        as Arc<dyn EmbeddingModel>,
                )
            }),
        )
        .bind_rerank(
            runtime.scope.clone(),
            Arc::new(move |model| {
                Ok(
                    Arc::new(CohereRerankModel::new(rerank_runtime.clone(), model))
                        as Arc<dyn RerankModel>,
                )
            }),
        )
    }

    pub fn profile(&self) -> &CohereProfile {
        &self.profile
    }

    /// Return evidence-backed portable and provider-native support claims.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        self.support_manifest.as_ref()
    }

    fn create_embedding_model(&self, model: ModelId) -> CohereEmbeddingModel {
        CohereEmbeddingModel::new(self.runtime.clone(), model)
    }

    fn create_rerank_model(&self, model: ModelId) -> CohereRerankModel {
        CohereRerankModel::new(self.runtime.clone(), model)
    }
}

impl Provider for CohereProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
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
    provider_selected_endpoint: bool,
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
            provider_selected_endpoint: true,
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
        self.provider_selected_endpoint = false;
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
        let endpoint = match self.endpoint {
            Some(endpoint) => endpoint,
            None => default_endpoint()?,
        };
        let verified_endpoint = self.provider_selected_endpoint;
        let profile = if verified_endpoint {
            CohereProfile::current()?
        } else {
            CohereProfile::custom()?
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            profile.scope().provider_id().clone(),
            [profile.provider_profile().clone()],
            if verified_endpoint {
                vec![transcription_support_claim()?]
            } else {
                Vec::new()
            },
        )?);
        let scope = profile.scope();
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
        let runtime = Arc::new(CohereRuntime {
            scope: scope.clone(),
            instance_id: ProviderInstanceId::new(),
            transport,
            replay_safety: ReplaySafety::Never,
        });
        let registration = CohereProvider::build_registration(runtime.clone())?;
        Ok(CohereProvider {
            runtime,
            profile,
            registration,
            support_manifest,
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
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
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

fn parse_model_id(model: impl Into<String>) -> Result<ModelId, ModelLookupError> {
    ModelId::new(model.into()).map_err(ModelLookupError::from)
}

fn default_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(COHERE_V2_BASE_URL, OfficialOrigin::new(COHERE_ORIGIN)?)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CohereConfigError {
    #[error("invalid Cohere provider identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Cohere support profile: {0}")]
    Profile(#[from] CohereProfileError),
    #[error("Cohere API key is empty or contains invalid bytes")]
    InvalidApiKey,
    #[error("invalid Cohere endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Cohere transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Cohere default registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("invalid Cohere support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Cohere support evidence: {0}")]
    SupportEvidence(#[from] siumai_core::ProfileError),
    #[error("invalid Cohere support verification date")]
    SupportVerificationDate,
}

fn transcription_support_claim() -> Result<VerifiedNativeSupportClaim, CohereConfigError> {
    let verified_at = chrono::NaiveDate::from_ymd_opt(2026, 8, 15)
        .ok_or(CohereConfigError::SupportVerificationDate)?;
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            siumai_core::ProviderId::new(super::profile::PROVIDER_ID)?,
            siumai_core::PlatformId::new(super::profile::PLATFORM_ID)?,
            NativeSurfaceKind::Resource,
            NativeSurfaceId::new("audio-transcriptions")?,
        ),
        VerifiedFidelity::Native,
        ApiStability::Stable,
        NativeVerificationEvidence::new(
            OfficialSource::new(TRANSCRIPTION_SOURCE)?,
            VerificationDate::new(verified_at),
        )
        .with_upstream(UpstreamLifecycle::new(
            Some(UpstreamMaturity::Stable),
            Some(UpstreamSupportStatus::Active),
            None,
        )),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::ModelFamily;

    #[test]
    fn default_registration_exposes_both_native_families() {
        let provider = CohereProvider::builder("test-key")
            .with_endpoint(
                EndpointConfig::local_explicit("http://127.0.0.1:9/v2").expect("local endpoint"),
            )
            .build()
            .expect("provider");
        let registration = provider.registration();

        assert!(registration.supports_family(ModelFamily::Embedding));
        assert!(registration.supports_family(ModelFamily::Rerank));
        assert_eq!(
            registration.scope(ModelFamily::Embedding),
            registration.scope(ModelFamily::Rerank)
        );
        assert_eq!(
            registration
                .embedding_model(ModelId::new("future-embed").expect("model ID"))
                .expect("embedding model")
                .descriptor()
                .family(),
            ModelFamily::Embedding
        );
        assert_eq!(
            registration
                .rerank_model(ModelId::new("future-rerank").expect("model ID"))
                .expect("rerank model")
                .descriptor()
                .family(),
            ModelFamily::Rerank
        );
    }
}
