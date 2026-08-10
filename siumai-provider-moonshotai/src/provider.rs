//! Long-lived Moonshot AI provider and synchronous Kimi model construction.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    ApiStability, CallOptions, Error, InvalidId, LanguageModel, LanguageModelProvider,
    LanguageRequest, LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelId,
    ModelLookupError, NativeSupportScope, NativeSurfaceId, NativeSurfaceKind,
    NativeVerificationEvidence, OfficialSource, ProfileError, Provider, ProviderInstanceId,
    ProviderOptionError, ProviderRegistration, ProviderSupportManifest, ReplayDomain,
    ReplayDomainId, SupportManifestError, TypedProviderOptions, VerificationDate, VerifiedFidelity,
    VerifiedNativeSupportClaim,
};
use siumai_openai_compatible::{
    CredentialSourceError, DynamicCredentialSource, OpenAiCompatibleApiMode,
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProvider,
};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, ProviderTransport, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::files::{KimiFiles, MoonshotNativeRuntime};
use crate::language::{DEFAULT_BASE_URL, PROVIDER_ID, profile};
use crate::options::KimiLanguageOptions;

const OFFICIAL_ORIGIN: &str = "https://api.moonshot.ai";
const OFFICIAL_REPLAY_DOMAIN: &str = "moonshotai-kimi-public-api";
const FILES_SOURCE: &str = "https://platform.kimi.ai/docs/api/files";
const FILES_VERIFIED_ON: &str = "2026-08-08";

/// Moonshot AI credential source.
#[derive(Clone)]
pub struct MoonshotCredential(OpenAiCompatibleCredential);

impl MoonshotCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self(OpenAiCompatibleCredential::api_key(value))
    }

    pub fn dynamic(source: Arc<dyn DynamicCredentialSource>) -> Self {
        Self(OpenAiCompatibleCredential::dynamic(source))
    }

    pub fn unauthenticated() -> Self {
        Self(OpenAiCompatibleCredential::unauthenticated())
    }
}

impl fmt::Debug for MoonshotCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MoonshotCredential")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// Long-lived, model-independent Moonshot AI provider for the Kimi product surface.
#[derive(Clone)]
pub struct MoonshotProvider {
    language: OpenAiCompatibleProvider,
    native: Arc<MoonshotNativeRuntime>,
    registration: ProviderRegistration,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl MoonshotProvider {
    pub fn builder(credential: MoonshotCredential) -> MoonshotProviderBuilder {
        MoonshotProviderBuilder::new(credential)
    }

    pub fn from_api_key(api_key: impl Into<String>) -> Result<Self, MoonshotConfigError> {
        Self::builder(MoonshotCredential::api_key(api_key)).build()
    }

    /// Create a lightweight Kimi Chat Completions model from an open textual model ID.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<MoonshotLanguageModel, ModelLookupError> {
        self.chat_completions(model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<MoonshotLanguageModel, ModelLookupError> {
        self.language
            .chat_completions(model)
            .map(|inner| MoonshotLanguageModel { inner })
    }

    /// Registration for the Kimi Chat Completions family surface.
    pub fn registration(&self) -> ProviderRegistration {
        self.registration.clone()
    }

    /// Provider-owned Kimi Files lifecycle.
    pub fn files(&self) -> KimiFiles {
        KimiFiles::new(self.native.clone())
    }

    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        &self.support_manifest
    }

    /// Inspect the exact support evidence for this configured endpoint.
    ///
    /// The official endpoint exposes verified Kimi evidence. Caller-controlled endpoints retain
    /// the Moonshot AI provider identity but expose only generic compatibility evidence.
    pub fn profile(&self) -> &siumai_core::ProviderProfile {
        self.language.profile().provider_profile()
    }
}

impl Provider for MoonshotProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for MoonshotProvider {
    type Model = MoonshotLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.language(model.to_string())
    }
}

impl fmt::Debug for MoonshotProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoonshotProvider")
            .field("provider_id", &PROVIDER_ID)
            .finish()
    }
}

/// Builder for one immutable Moonshot AI provider runtime.
pub struct MoonshotProviderBuilder {
    credential: MoonshotCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    language_defaults: KimiLanguageOptions,
}

impl MoonshotProviderBuilder {
    fn new(credential: MoonshotCredential) -> Self {
        Self {
            credential,
            endpoint: official_endpoint(),
            provider_selected_endpoint: true,
            replay_domain: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            language_defaults: KimiLanguageOptions::default(),
        }
    }

    /// Replace the provider-owned endpoint with a caller-controlled endpoint.
    ///
    /// Ownership remains caller-controlled even when `endpoint` uses an official transport
    /// policy. A matching custom replay domain is required before [`Self::build`].
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self.provider_selected_endpoint = false;
        self
    }

    /// Replace the default endpoint with a caller-selected public endpoint.
    ///
    /// Call [`Self::with_replay_domain`] with an explicit custom audience before building.
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

    pub fn with_language_defaults(mut self, defaults: KimiLanguageOptions) -> Self {
        self.language_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<MoonshotProvider, MoonshotConfigError> {
        self.language_defaults.validate()?;
        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let replay_domain = match (self.replay_domain, verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN)?),
            (None, false) => return Err(MoonshotConfigError::CustomEndpointRequiresReplayDomain),
        };
        if replay_domain.audience().is_official() != verified_endpoint {
            return Err(MoonshotConfigError::ReplayAudienceMismatch);
        }

        self.credential.0.validate_static()?;
        let auth = self.credential.0.into_auth();
        let profile = profile(endpoint.clone(), replay_domain, verified_endpoint)?;
        let native_claims = if verified_endpoint {
            vec![files_support_claim()?]
        } else {
            Vec::new()
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [profile.provider_profile().clone()],
            native_claims,
        )?);
        let instance_id = ProviderInstanceId::new();
        let mut builder = OpenAiCompatibleProvider::builder_with_auth(profile, auth.clone())
            .with_provider_instance(instance_id)
            .with_limits(self.limits.clone())
            .with_retry_policy(self.retry_policy);
        let mut native_builder = ProviderTransport::builder(endpoint)
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            builder = builder.with_connect_timeout(timeout);
            native_builder = native_builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            builder = builder.with_call_timeout(timeout);
            native_builder = native_builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            builder = builder.with_read_timeout(timeout);
            native_builder = native_builder.with_read_timeout(timeout);
        }
        for (name, value) in option_map(&self.language_defaults)? {
            builder =
                builder.with_default_option(OpenAiCompatibleApiMode::ChatCompletions, name, value);
        }

        let language = builder.build()?;
        let native = Arc::new(MoonshotNativeRuntime::new(native_builder.build()?));
        let registration = language
            .chat_completions_registration()
            .ok_or(MoonshotConfigError::MissingChatMode)?;
        Ok(MoonshotProvider {
            language,
            native,
            registration,
            support_manifest,
        })
    }
}

/// Concrete Kimi language model backed by one shared Moonshot AI provider runtime.
#[derive(Clone)]
pub struct MoonshotLanguageModel {
    inner: OpenAiCompatibleLanguageModel,
}

impl Model for MoonshotLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
}

#[async_trait]
impl LanguageModel for MoonshotLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        self.inner.generate(request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.inner.stream(request, options).await
    }
}

impl fmt::Debug for MoonshotLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoonshotLanguageModel")
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn files_support_claim() -> Result<VerifiedNativeSupportClaim, MoonshotConfigError> {
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            siumai_core::PlatformId::new("kimi-public-api")?,
            NativeSurfaceKind::Resource,
            NativeSurfaceId::new("files-basic-lifecycle")?,
        ),
        VerifiedFidelity::Native,
        ApiStability::Stable,
        NativeVerificationEvidence::new(
            OfficialSource::new(FILES_SOURCE)?,
            VerificationDate::new(NaiveDate::parse_from_str(FILES_VERIFIED_ON, "%Y-%m-%d")?),
        ),
    ))
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, MoonshotConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(MoonshotConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum MoonshotConfigError {
    #[error("invalid Moonshot AI identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Moonshot AI endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Moonshot AI compatible profile or runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid Moonshot AI credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("invalid Moonshot AI native transport: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Moonshot AI support evidence: {0}")]
    SupportEvidence(#[from] ProfileError),
    #[error("invalid Moonshot AI support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("invalid Moonshot AI support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Kimi default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("Kimi default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("Kimi default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("the Moonshot AI profile omitted its required Chat Completions mode")]
    MissingChatMode,
    #[error("a caller-controlled Moonshot AI endpoint requires an explicit custom replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Moonshot AI endpoint ownership")]
    ReplayAudienceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ApiModeId, ModelFamily, Provider};
    use siumai_transport::EndpointPolicy;

    #[test]
    fn credentials_are_redacted() {
        let debug = format!("{:?}", MoonshotCredential::api_key("canary-secret"));
        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn provider_owned_endpoint_exposes_dated_kimi_evidence() {
        let provider = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
            .build()
            .expect("official provider");
        let claims = provider
            .profile()
            .verified_claims()
            .expect("verified claims");

        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].evidence().source().as_str(),
            crate::OFFICIAL_SOURCE
        );
        assert!(
            provider
                .language("kimi-k4-future")
                .expect("future model remains open")
                .descriptor()
                .model()
                .as_str()
                .starts_with("kimi-")
        );
    }

    #[test]
    fn caller_endpoint_requires_custom_replay_even_when_marked_official() {
        let endpoint = EndpointConfig::official(
            "https://relay.example.test/v1",
            OfficialOrigin::new("https://relay.example.test").expect("origin"),
        )
        .expect("endpoint");
        assert!(matches!(endpoint.policy(), EndpointPolicy::Official(_)));

        let missing = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .expect_err("caller endpoint needs an explicit replay domain");
        assert!(matches!(
            missing,
            MoonshotConfigError::CustomEndpointRequiresReplayDomain
        ));

        let official_audience = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("forged-official").expect("domain"),
            ))
            .build()
            .expect_err("caller endpoint cannot inherit official replay");
        assert!(matches!(
            official_audience,
            MoonshotConfigError::ReplayAudienceMismatch
        ));

        let provider = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("tenant-relay").expect("domain"),
            ))
            .build()
            .expect("custom provider");
        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert!(provider.profile().verified_claims().is_none());
        assert!(provider.profile().generic_claims().is_some());
    }

    #[test]
    fn future_model_ids_remain_callable_without_network_lookup() {
        let provider = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-endpoint").expect("domain"),
            ))
            .build()
            .expect("provider");
        let model = provider.language("kimi-k4-future").expect("future model");

        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert_eq!(model.descriptor().model().as_str(), "kimi-k4-future");
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Language)
                .map(ApiModeId::as_str),
            Some("chat-completions")
        );
    }
}
