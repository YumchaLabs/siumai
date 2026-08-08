use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use http::header::{HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{
    Error as CoreError, ErrorKind, ImageModel, ImageModelProvider, InvalidId, LanguageModel,
    LanguageModelProvider, ModelId, ModelLookupError, ModelOperation, ModelPolicy,
    ModelPolicyContext, ModelPolicyDecision, Provider, ProviderOptionError, ProviderOptions,
    ProviderRegistration, ProviderScope, ReplayDomain, ReplayDomainId, UnsupportedReason,
};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, EndpointConfig, EndpointError,
    OfficialOrigin, ProviderTransport, RetryPolicy, TransportConfigError, TransportLimits,
};
use thiserror::Error;

use crate::image::GeminiImageModel;
use crate::language::GeminiLanguageModel;
use crate::models::{is_current_image, is_current_interactions};
use crate::options::{GeminiImageOptions, GeminiInteractionsOptions};
use crate::profile::{API_MODE_ID, GeminiProfile, GeminiProfileError, PROTOCOL_ID, PROVIDER_ID};

const OFFICIAL_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const OFFICIAL_ORIGIN: &str = "https://generativelanguage.googleapis.com";
const OFFICIAL_REPLAY_DOMAIN: &str = "google-gemini-api";
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

/// Authentication used by one configured Gemini runtime.
#[derive(Clone)]
#[non_exhaustive]
pub enum GeminiCredential {
    ApiKey(SecretString),
    Unauthenticated,
}

impl GeminiCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    /// Explicitly disable authentication for a trusted local test double or gateway.
    pub fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    fn validate(&self) -> Result<(), GeminiConfigError> {
        let Self::ApiKey(value) = self else {
            return Ok(());
        };
        let value = value.expose_secret();
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > MAX_CREDENTIAL_BYTES
            || HeaderValue::from_str(value).is_err()
        {
            return Err(GeminiConfigError::InvalidCredential);
        }
        Ok(())
    }

    fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::ApiKey(value) => Arc::new(GeminiApiKeyAuth { value }),
            Self::Unauthenticated => Arc::new(siumai_transport::NoAuth),
        }
    }
}

impl fmt::Debug for GeminiCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GeminiCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

struct GeminiApiKeyAuth {
    value: SecretString,
}

#[async_trait]
impl AuthApplier for GeminiApiKeyAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, CoreError> {
        let value = HeaderValue::from_str(self.value.expose_secret()).map_err(|source| {
            CoreError::new(
                ErrorKind::Authentication,
                "Gemini API key could not be encoded as a request header",
            )
            .with_source(source)
        })?;
        CredentialPatch::new()
            .try_insert(HeaderName::from_static("x-goog-api-key"), value)
            .map_err(|source| {
                CoreError::new(
                    ErrorKind::Configuration,
                    "Gemini authentication conflicts with the transport contract",
                )
                .with_source(source)
            })
    }
}

/// A long-lived, model-independent provider for the Google Gemini product surface.
#[derive(Clone)]
pub struct GeminiProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
    profile: GeminiProfile,
}

impl GeminiProvider {
    pub fn builder(credential: GeminiCredential) -> GeminiProviderBuilder {
        GeminiProviderBuilder::new(credential)
    }

    pub fn image(&self, model: impl Into<String>) -> Result<GeminiImageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_image_model(model))
    }

    /// Create the primary stable-v1 Interactions language model handle.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<GeminiLanguageModel, ModelLookupError> {
        self.interactions(model)
    }

    /// Create an explicit stable-v1 Interactions language model handle.
    pub fn interactions(
        &self,
        model: impl Into<String>,
    ) -> Result<GeminiLanguageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_language_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let language_provider = self.clone();
        let image_provider = self.clone();
        ProviderRegistration::from_language(
            self.runtime.interactions_scope.clone(),
            self.runtime.interactions_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(language_provider.create_language_model(model))
                    as Arc<dyn LanguageModel>)
            }),
        )
        .merge(ProviderRegistration::from_image(
            self.runtime.image_scope.clone(),
            self.runtime.image_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(image_provider.create_image_model(model)) as Arc<dyn ImageModel>)
            }),
        ))
        .expect("Gemini family registrations share one canonical provider")
    }

    pub fn profile(&self) -> &GeminiProfile {
        &self.profile
    }

    fn create_image_model(&self, model: ModelId) -> GeminiImageModel {
        GeminiImageModel::new(self.runtime.clone(), model)
    }

    fn create_language_model(&self, model: ModelId) -> GeminiLanguageModel {
        GeminiLanguageModel::new(self.runtime.clone(), model)
    }
}

impl Provider for GeminiProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.interactions_scope.provider_id()
    }
}

impl LanguageModelProvider for GeminiProvider {
    type Model = GeminiLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_language_model(model))
    }
}

impl ImageModelProvider for GeminiProvider {
    type Model = GeminiImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl fmt::Debug for GeminiProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiProvider")
            .field("interactions_scope", &self.runtime.interactions_scope)
            .field("image_scope", &self.runtime.image_scope)
            .field("transport", &"shared")
            .finish()
    }
}

/// Builder for one immutable Gemini provider runtime.
pub struct GeminiProviderBuilder {
    credential: GeminiCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    interactions_defaults: GeminiInteractionsOptions,
    image_defaults: GeminiImageOptions,
}

impl GeminiProviderBuilder {
    fn new(credential: GeminiCredential) -> Self {
        let endpoint = OfficialOrigin::new(OFFICIAL_ORIGIN)
            .and_then(|origin| EndpointConfig::official(OFFICIAL_BASE_URL, origin));
        Self {
            credential,
            endpoint,
            provider_selected_endpoint: true,
            replay_domain: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            interactions_defaults: GeminiInteractionsOptions::default(),
            image_defaults: GeminiImageOptions::default(),
        }
    }

    /// Replace the provider-owned endpoint with a caller-controlled endpoint.
    ///
    /// Ownership remains caller-controlled even if `endpoint` uses an official transport policy.
    /// Call [`Self::with_replay_domain`] with an explicit custom audience before building.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self.provider_selected_endpoint = false;
        self
    }

    /// Replace the default endpoint with a caller-selected public endpoint.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::public_custom(base_url);
        self.provider_selected_endpoint = false;
        self
    }

    /// Bind provider-native replay data to a caller-declared, non-secret endpoint audience.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Self {
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

    pub fn with_image_defaults(mut self, defaults: GeminiImageOptions) -> Self {
        self.image_defaults = defaults;
        self
    }

    pub fn with_interactions_defaults(mut self, defaults: GeminiInteractionsOptions) -> Self {
        self.interactions_defaults = defaults;
        self
    }

    /// Validate static settings and create exactly one shared transport runtime.
    pub fn build(self) -> Result<GeminiProvider, GeminiConfigError> {
        self.credential.validate()?;
        ProviderOptions::typed(&self.interactions_defaults)?;
        ProviderOptions::typed(&self.image_defaults)?;
        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let replay_domain = match (self.replay_domain, verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN)?),
            (None, false) => return Err(GeminiConfigError::CustomEndpointRequiresReplayDomain),
        };
        if replay_domain.audience().is_official() != verified_endpoint {
            return Err(GeminiConfigError::ReplayAudienceMismatch);
        }
        let profile = if verified_endpoint {
            GeminiProfile::current(replay_domain)?
        } else {
            GeminiProfile::custom(replay_domain)?
        };
        let limits = self.limits.clone();
        let mut transport = ProviderTransport::builder(endpoint)
            .with_auth(self.credential.into_auth())
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
        Ok(GeminiProvider {
            runtime: Arc::new(ProviderRuntime {
                interactions_scope: profile.interactions_scope(),
                image_scope: profile.image_scope(),
                transport: transport.build()?,
                limits,
                interactions_policy: Arc::new(GeminiInteractionsPolicy { verified_endpoint }),
                image_policy: Arc::new(GeminiImagePolicy { verified_endpoint }),
                interactions_defaults: self.interactions_defaults,
                image_defaults: self.image_defaults,
            }),
            profile,
        })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GeminiConfigError {
    #[error("Gemini API key must be non-empty and at most 16 KiB")]
    InvalidCredential,
    #[error("invalid Gemini profile: {0}")]
    Profile(#[from] GeminiProfileError),
    #[error("invalid Gemini provider options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("invalid Gemini endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Gemini transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Gemini identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("a caller-controlled Gemini endpoint requires an explicit custom replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Gemini endpoint ownership")]
    ReplayAudienceMismatch,
}

pub(crate) struct ProviderRuntime {
    pub(crate) interactions_scope: Arc<ProviderScope>,
    pub(crate) image_scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) limits: TransportLimits,
    pub(crate) interactions_policy: Arc<GeminiInteractionsPolicy>,
    pub(crate) image_policy: Arc<GeminiImagePolicy>,
    pub(crate) interactions_defaults: GeminiInteractionsOptions,
    pub(crate) image_defaults: GeminiImageOptions,
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("interactions_scope", &self.interactions_scope)
            .field("image_scope", &self.image_scope)
            .field("transport", &"shared")
            .field("limits", &self.limits)
            .field("interactions_defaults", &self.interactions_defaults)
            .field("image_defaults", &self.image_defaults)
            .finish()
    }
}

pub(crate) struct GeminiImagePolicy {
    verified_endpoint: bool,
}

pub(crate) struct GeminiInteractionsPolicy {
    verified_endpoint: bool,
}

impl ModelPolicy for GeminiInteractionsPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope().provider_id().as_str() == PROVIDER_ID
            && context.scope().protocol().map(|value| value.as_str()) == Some(PROTOCOL_ID)
            && context.scope().api_mode().map(|value| value.as_str()) == Some(API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if !matches!(
            context.operation(),
            ModelOperation::Generate | ModelOperation::Stream
        ) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if self.verified_endpoint && is_current_interactions(context.model().as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

impl ModelPolicy for GeminiImagePolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope().provider_id().as_str() == PROVIDER_ID
            && context.scope().protocol().map(|value| value.as_str()) == Some(PROTOCOL_ID)
            && context.scope().api_mode().map(|value| value.as_str()) == Some(API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if context.operation() != ModelOperation::GenerateImage {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if self.verified_endpoint && is_current_image(context.model().as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::{Provider, ReplayAudience};

    use super::*;

    fn caller_declared_official_endpoint() -> EndpointConfig {
        let origin = OfficialOrigin::new("https://relay.example").unwrap();
        EndpointConfig::official("https://relay.example", origin).unwrap()
    }

    #[test]
    fn credential_debug_is_redacted() {
        let debug = format!("{:?}", GeminiCredential::api_key("canary-secret"));
        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn caller_endpoint_cannot_forge_official_evidence_or_replay_audience() {
        assert!(matches!(
            GeminiProvider::builder(GeminiCredential::unauthenticated())
                .with_endpoint(caller_declared_official_endpoint())
                .build(),
            Err(GeminiConfigError::CustomEndpointRequiresReplayDomain)
        ));
        assert!(matches!(
            GeminiProvider::builder(GeminiCredential::unauthenticated())
                .with_endpoint(caller_declared_official_endpoint())
                .with_replay_domain(ReplayDomain::official(
                    ReplayDomainId::new("forged-official").unwrap(),
                ))
                .build(),
            Err(GeminiConfigError::ReplayAudienceMismatch)
        ));

        let provider = GeminiProvider::builder(GeminiCredential::unauthenticated())
            .with_endpoint(caller_declared_official_endpoint())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("caller-relay").unwrap(),
            ))
            .build()
            .unwrap();
        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert!(
            provider
                .profile()
                .provider_profile()
                .verified_claims()
                .is_none()
        );
        assert_eq!(
            provider
                .runtime
                .image_scope
                .replay_domain()
                .unwrap()
                .audience(),
            &ReplayAudience::Custom(ReplayDomainId::new("caller-relay").unwrap())
        );
    }
}
