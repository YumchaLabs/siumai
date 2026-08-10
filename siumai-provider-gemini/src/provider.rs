use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use http::header::{HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{
    ApiStability, EmbeddingModel, EmbeddingModelProvider, Error as CoreError, ErrorKind,
    ImageModel, ImageModelProvider, InvalidId, LanguageModel, LanguageModelProvider, ModelId,
    ModelLookupError, ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision,
    NativeSupportScope, NativeSurfaceId, NativeSurfaceKind, NativeVerificationEvidence,
    OfficialSource, PlatformId, Provider, ProviderInstanceId, ProviderOptionError, ProviderOptions,
    ProviderRegistration, ProviderScope, ProviderSupportManifest, ReplayDomain, ReplayDomainId,
    SpeechModel, SpeechModelProvider, SupportManifestError, UnsupportedReason, UpstreamLifecycle,
    UpstreamMaturity, UpstreamSupportStatus, VerificationDate, VerifiedFidelity,
    VerifiedNativeSupportClaim,
};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, EndpointConfig, EndpointError,
    OfficialOrigin, ProviderTransport, RetryPolicy, TransportConfigError, TransportLimits,
};
use thiserror::Error;

use crate::embedding::{
    GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2, GEMINI_EMBEDDING_API_MODE_ID, GeminiEmbeddingModel,
    GeminiEmbeddingOptions,
};
use crate::files::GeminiFiles;
use crate::generate_content::{
    GeminiGenerateContentModel, GeminiGenerateContentOptions, GeminiGenerateContentPolicy,
};
use crate::image::GeminiImageModel;
use crate::language::GeminiLanguageModel;
use crate::models::{is_current_image, is_current_interactions};
use crate::options::{GeminiImageOptions, GeminiInteractionsOptions};
use crate::profile::{
    API_MODE_ID, EMBEDDING_PROTOCOL_ID, FILES_SOURCE, GeminiProfile, GeminiProfileError,
    PLATFORM_ID, PROTOCOL_ID, PROVIDER_ID, VEO_SOURCE,
};
use crate::speech::{GeminiSpeechModel, GeminiSpeechOptions, GeminiSpeechPolicy};
use crate::veo::GeminiVeo;

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
    support_manifest: Arc<ProviderSupportManifest>,
}

impl GeminiProvider {
    pub fn builder(credential: GeminiCredential) -> GeminiProviderBuilder {
        GeminiProviderBuilder::new(credential)
    }

    pub fn image(&self, model: impl Into<String>) -> Result<GeminiImageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_image_model(model))
    }

    /// Create a stable-v1 synchronous text embedding model handle.
    pub fn embedding(
        &self,
        model: impl Into<String>,
    ) -> Result<GeminiEmbeddingModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_embedding_model(model))
    }

    /// Create a current v1beta Interactions buffered TTS model handle.
    pub fn speech(&self, model: impl Into<String>) -> Result<GeminiSpeechModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_speech_model(model))
    }

    /// Access stable-v1 File metadata lifecycle operations.
    pub fn files(&self) -> GeminiFiles {
        GeminiFiles::new(self.runtime.clone())
    }

    /// Access typed Veo long-running submit and operation-status APIs.
    pub fn veo(&self) -> GeminiVeo {
        GeminiVeo::new(self.runtime.clone())
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

    /// Create the explicit stable-v1 Generate Content language model handle.
    ///
    /// This is a secondary compatibility mode. [`Self::language`] remains the primary
    /// Interactions API path.
    pub fn generate_content(
        &self,
        model: impl Into<String>,
    ) -> Result<GeminiGenerateContentModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_generate_content_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let language_provider = self.clone();
        let embedding_provider = self.clone();
        let image_provider = self.clone();
        let speech_provider = self.clone();
        ProviderRegistration::from_language(
            self.runtime.interactions_scope.clone(),
            self.runtime.interactions_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(language_provider.create_language_model(model))
                    as Arc<dyn LanguageModel>)
            }),
        )
        .merge(ProviderRegistration::from_embedding(
            self.runtime.embedding_scope.clone(),
            self.runtime.embedding_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(embedding_provider.create_embedding_model(model))
                    as Arc<dyn EmbeddingModel>)
            }),
        ))
        .expect("Gemini language and embedding registrations share one provider")
        .merge(ProviderRegistration::from_image(
            self.runtime.image_scope.clone(),
            self.runtime.image_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(image_provider.create_image_model(model)) as Arc<dyn ImageModel>)
            }),
        ))
        .expect("Gemini image registration shares one provider")
        .merge(ProviderRegistration::from_speech(
            self.runtime.speech_scope.clone(),
            self.runtime.speech_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(speech_provider.create_speech_model(model)) as Arc<dyn SpeechModel>)
            }),
        ))
        .expect("Gemini speech registration shares one provider")
    }

    /// Build a separate registration for the explicit Generate Content language mode.
    ///
    /// A Registry route selects one language API mode at a time, so this registration is
    /// intentionally separate from [`Self::registration`].
    pub fn generate_content_registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_language(
            self.runtime.generate_content_scope.clone(),
            self.runtime.generate_content_policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(provider.create_generate_content_model(model))
                    as Arc<dyn LanguageModel>)
            }),
        )
    }

    pub fn profile(&self) -> &GeminiProfile {
        &self.profile
    }

    /// Return evidence-backed portable and native support claims for this configured endpoint.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        self.support_manifest.as_ref()
    }

    fn create_image_model(&self, model: ModelId) -> GeminiImageModel {
        GeminiImageModel::new(self.runtime.clone(), model)
    }

    fn create_embedding_model(&self, model: ModelId) -> GeminiEmbeddingModel {
        GeminiEmbeddingModel::new(
            self.runtime.clone(),
            self.runtime.embedding_scope.clone(),
            model,
            self.runtime.embedding_defaults.clone(),
        )
    }

    fn create_speech_model(&self, model: ModelId) -> GeminiSpeechModel {
        GeminiSpeechModel::new(self.runtime.clone(), model)
    }

    fn create_language_model(&self, model: ModelId) -> GeminiLanguageModel {
        GeminiLanguageModel::new(self.runtime.clone(), model)
    }

    fn create_generate_content_model(&self, model: ModelId) -> GeminiGenerateContentModel {
        GeminiGenerateContentModel::new(self.runtime.clone(), model)
    }
}

impl Provider for GeminiProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for GeminiProvider {
    type Model = GeminiLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_language_model(model))
    }
}

impl EmbeddingModelProvider for GeminiProvider {
    type Model = GeminiEmbeddingModel;

    fn embedding_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_embedding_model(model))
    }
}

impl ImageModelProvider for GeminiProvider {
    type Model = GeminiImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl SpeechModelProvider for GeminiProvider {
    type Model = GeminiSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }
}

impl fmt::Debug for GeminiProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiProvider")
            .field("interactions_scope", &self.runtime.interactions_scope)
            .field("embedding_scope", &self.runtime.embedding_scope)
            .field("image_scope", &self.runtime.image_scope)
            .field("speech_scope", &self.runtime.speech_scope)
            .field("veo_scope", &self.runtime.veo_scope)
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
    embedding_defaults: GeminiEmbeddingOptions,
    image_defaults: GeminiImageOptions,
    speech_defaults: GeminiSpeechOptions,
    generate_content_defaults: GeminiGenerateContentOptions,
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
            embedding_defaults: GeminiEmbeddingOptions::default(),
            image_defaults: GeminiImageOptions::default(),
            speech_defaults: GeminiSpeechOptions::default(),
            generate_content_defaults: GeminiGenerateContentOptions::default(),
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

    pub fn with_embedding_defaults(mut self, defaults: GeminiEmbeddingOptions) -> Self {
        self.embedding_defaults = defaults;
        self
    }

    pub fn with_speech_defaults(mut self, defaults: GeminiSpeechOptions) -> Self {
        self.speech_defaults = defaults;
        self
    }

    pub fn with_generate_content_defaults(
        mut self,
        defaults: GeminiGenerateContentOptions,
    ) -> Self {
        self.generate_content_defaults = defaults;
        self
    }

    /// Validate static settings and create exactly one shared transport runtime.
    pub fn build(self) -> Result<GeminiProvider, GeminiConfigError> {
        self.credential.validate()?;
        ProviderOptions::typed(&self.interactions_defaults)?;
        ProviderOptions::typed(&self.embedding_defaults)?;
        ProviderOptions::typed(&self.image_defaults)?;
        ProviderOptions::typed(&self.speech_defaults)?;
        ProviderOptions::typed(&self.generate_content_defaults)?;
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
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [profile.provider_profile().clone()],
            if verified_endpoint {
                native_support_claims()?
            } else {
                Vec::new()
            },
        )?);
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
                instance_id: ProviderInstanceId::new(),
                interactions_scope: profile.interactions_scope(),
                embedding_scope: profile.embedding_scope(),
                image_scope: profile.image_scope(),
                speech_scope: profile.speech_scope(),
                veo_scope: profile.veo_scope(),
                generate_content_scope: profile.generate_content_scope(),
                transport: transport.build()?,
                limits,
                interactions_policy: Arc::new(GeminiInteractionsPolicy { verified_endpoint }),
                embedding_policy: Arc::new(GeminiEmbeddingPolicy { verified_endpoint }),
                image_policy: Arc::new(GeminiImagePolicy { verified_endpoint }),
                speech_policy: Arc::new(GeminiSpeechPolicy { verified_endpoint }),
                generate_content_policy: Arc::new(GeminiGenerateContentPolicy {
                    verified_endpoint,
                }),
                interactions_defaults: self.interactions_defaults,
                embedding_defaults: self.embedding_defaults,
                image_defaults: self.image_defaults,
                speech_defaults: self.speech_defaults,
                generate_content_defaults: self.generate_content_defaults,
            }),
            profile,
            support_manifest,
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
    #[error("invalid Gemini support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Gemini support evidence: {0}")]
    SupportEvidence(#[from] siumai_core::ProfileError),
    #[error("invalid Gemini support verification date")]
    SupportVerificationDate,
}

pub(crate) struct ProviderRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) interactions_scope: Arc<ProviderScope>,
    pub(crate) embedding_scope: Arc<ProviderScope>,
    pub(crate) image_scope: Arc<ProviderScope>,
    pub(crate) speech_scope: Arc<ProviderScope>,
    pub(crate) veo_scope: Arc<ProviderScope>,
    pub(crate) generate_content_scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) limits: TransportLimits,
    pub(crate) interactions_policy: Arc<GeminiInteractionsPolicy>,
    pub(crate) embedding_policy: Arc<GeminiEmbeddingPolicy>,
    pub(crate) image_policy: Arc<GeminiImagePolicy>,
    pub(crate) speech_policy: Arc<GeminiSpeechPolicy>,
    pub(crate) generate_content_policy: Arc<GeminiGenerateContentPolicy>,
    pub(crate) interactions_defaults: GeminiInteractionsOptions,
    pub(crate) embedding_defaults: GeminiEmbeddingOptions,
    pub(crate) image_defaults: GeminiImageOptions,
    pub(crate) speech_defaults: GeminiSpeechOptions,
    pub(crate) generate_content_defaults: GeminiGenerateContentOptions,
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("interactions_scope", &self.interactions_scope)
            .field("embedding_scope", &self.embedding_scope)
            .field("image_scope", &self.image_scope)
            .field("speech_scope", &self.speech_scope)
            .field("veo_scope", &self.veo_scope)
            .field("generate_content_scope", &self.generate_content_scope)
            .field("transport", &"shared")
            .field("limits", &self.limits)
            .field("interactions_defaults", &self.interactions_defaults)
            .field("embedding_defaults", &self.embedding_defaults)
            .field("image_defaults", &self.image_defaults)
            .field("speech_defaults", &self.speech_defaults)
            .field("generate_content_defaults", &self.generate_content_defaults)
            .finish()
    }
}

pub(crate) struct GeminiImagePolicy {
    verified_endpoint: bool,
}

pub(crate) struct GeminiEmbeddingPolicy {
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

impl ModelPolicy for GeminiEmbeddingPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope().provider_id().as_str() == PROVIDER_ID
            && context.scope().protocol().map(|value| value.as_str())
                == Some(EMBEDDING_PROTOCOL_ID)
            && context.scope().api_mode().map(|value| value.as_str())
                == Some(GEMINI_EMBEDDING_API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if context.operation() != ModelOperation::Embed {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if self.verified_endpoint
            && matches!(
                context.model().as_str(),
                GEMINI_EMBEDDING_2 | GEMINI_EMBEDDING_001
            )
        {
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

fn native_support_claims() -> Result<Vec<VerifiedNativeSupportClaim>, GeminiConfigError> {
    let provider = siumai_core::ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let verified_at = VerificationDate::new(
        NaiveDate::from_ymd_opt(2026, 8, 8).ok_or(GeminiConfigError::SupportVerificationDate)?,
    );
    [
        (
            "files-metadata",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            FILES_SOURCE,
            UpstreamLifecycle::new(
                Some(UpstreamMaturity::Stable),
                Some(UpstreamSupportStatus::Active),
                None,
            ),
        ),
        (
            "veo-predict-long-running",
            NativeSurfaceKind::Job,
            ApiStability::Experimental,
            VEO_SOURCE,
            UpstreamLifecycle::new(
                Some(UpstreamMaturity::Preview),
                Some(UpstreamSupportStatus::Active),
                Some("preview".to_string()),
            ),
        ),
    ]
    .into_iter()
    .map(|(surface, kind, stability, source, upstream)| {
        Ok(VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider.clone(),
                platform.clone(),
                kind,
                NativeSurfaceId::new(surface)?,
            ),
            VerifiedFidelity::Native,
            stability,
            NativeVerificationEvidence::new(OfficialSource::new(source)?, verified_at)
                .with_upstream(upstream),
        ))
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use siumai_core::{
        EmbeddingModel, EmbeddingRequest, Model as _, Provider, ReplayAudience, SpeechModel,
        SpeechRequest, UsageValue,
    };

    use crate::GEMINI_3_1_FLASH_TTS_PREVIEW;

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
    fn configured_provider_models_share_one_instance_capability() {
        let first = GeminiProvider::builder(GeminiCredential::api_key("first-key"))
            .build()
            .unwrap();
        let second = GeminiProvider::builder(GeminiCredential::api_key("second-key"))
            .build()
            .unwrap();

        let language = first.language("future-language").unwrap();
        let image = first.image("future-image").unwrap();
        let embedding = first.embedding("future-embedding").unwrap();
        let other = second.language("future-language").unwrap();

        assert_eq!(
            language.descriptor().instance_id(),
            image.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().instance_id(),
            embedding.descriptor().instance_id()
        );
        assert_ne!(
            language.descriptor().instance_id(),
            other.descriptor().instance_id()
        );
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
        assert!(provider.support_manifest().native_claims().is_empty());
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

    #[tokio::test]
    async fn portable_embedding_and_speech_use_the_shared_provider_runtime() {
        let mut server = mockito::Server::new_async().await;
        let embedding = server
            .mock("POST", "/v1/models/gemini-embedding-001:embedContent")
            .match_header("x-goog-api-key", "test-key")
            .match_body(mockito::Matcher::Regex(
                "\\\"autoTruncate\\\":false".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "embedding": {"values": [0.1, 0.2]},
                    "usageMetadata": {"promptTokenCount": 2}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let audio = base64::engine::general_purpose::STANDARD.encode([0_u8, 1, 2, 3]);
        let speech = server
            .mock("POST", "/v1beta/interactions")
            .match_header("x-goog-api-key", "test-key")
            .match_body(mockito::Matcher::Regex(
                "\\\"voice\\\":\\\"Kore\\\"".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "id": "interaction-tts",
                    "status": "completed",
                    "model": GEMINI_3_1_FLASH_TTS_PREVIEW,
                    "steps": [{
                        "type": "model_output",
                        "content": [{
                            "type": "audio",
                            "mime_type": "audio/L16",
                            "data": audio
                        }]
                    }],
                    "usage": {
                        "total_input_tokens": 2,
                        "total_output_tokens": 1,
                        "total_tokens": 3
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
        let provider = GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(server.url()).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-family-test").unwrap(),
            ))
            .build()
            .unwrap();

        let embedded = provider
            .embedding(GEMINI_EMBEDDING_001)
            .unwrap()
            .embed(
                EmbeddingRequest::single("hello").unwrap(),
                siumai_core::CallOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(embedded.embeddings, vec![vec![0.1_f32, 0.2_f32]]);
        assert_eq!(embedded.usage.input_tokens, UsageValue::Known(2));

        let spoken = provider
            .speech(GEMINI_3_1_FLASH_TTS_PREVIEW)
            .unwrap()
            .synthesize(
                SpeechRequest::new("hello")
                    .unwrap()
                    .with_voice("Kore")
                    .unwrap(),
                siumai_core::CallOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(spoken.audio.as_ref(), [0_u8, 1, 2, 3]);
        assert_eq!(spoken.media_type, "audio/pcm");

        embedding.assert_async().await;
        speech.assert_async().await;
    }
}
