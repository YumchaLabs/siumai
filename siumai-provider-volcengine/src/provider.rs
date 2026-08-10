//! Long-lived Volcengine provider and synchronous ARK model construction.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, Error, GenericSupportClaim, ImageModel,
    ImageModelProvider, InvalidId, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelCatalog, ModelDescriptor, ModelFamily, ModelId,
    ModelLookupError, NativeSupportScope, NativeSurfaceId, NativeSurfaceKind,
    NativeVerificationEvidence, OfficialSource, PlatformId, ProfileError, ProfileId,
    ProtocolContractId, ProtocolId, Provider, ProviderId, ProviderInstanceId, ProviderOptionError,
    ProviderOptions, ProviderProfile, ProviderRegistration, ProviderRegistrationError,
    ProviderScope, ProviderSupportManifest, ReplayDomain, ReplayDomainId, SupportManifestError,
    SupportScope, TypedProviderOptions, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedNativeSupportClaim, VerifiedSupportClaim,
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

use crate::image::{ARK_IMAGE_API_MODE, ArkImageModel, ArkImageOptions, ArkImages};
use crate::language::{
    DEFAULT_BASE_URL, PLATFORM_ID, PROVIDER_ID, VolcengineProfileError, profile,
};
use crate::native::{ArkNativeRuntime, SharedArkNativeRuntime};
use crate::options::{ArkChatOptions, ArkResponsesOptions};
use crate::video::ArkVideoTasks;

const OFFICIAL_ORIGIN: &str = "https://ark.cn-beijing.volces.com";
const OFFICIAL_REPLAY_DOMAIN: &str = "volcengine-ark-cn-beijing";
const ARK_IMAGE_PROTOCOL: &str = "ark-images";
const IMAGE_SOURCE: &str = "https://api.volcengine.com/api-docs/view?action=ImageGenerations&serviceCode=ark&version=2024-01-01";
const VIDEO_SOURCE: &str = "https://api.volcengine.com/api-docs/view?action=CreateContentsGenerationsTasks&serviceCode=ark&version=2024-01-01";
const MEDIA_VERIFIED_ON: &str = "2026-08-08";

/// Public Volcengine ARK language API selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VolcengineLanguageApi {
    ChatCompletions,
    #[default]
    Responses,
}

impl From<VolcengineLanguageApi> for OpenAiCompatibleApiMode {
    fn from(value: VolcengineLanguageApi) -> Self {
        match value {
            VolcengineLanguageApi::ChatCompletions => Self::ChatCompletions,
            VolcengineLanguageApi::Responses => Self::Responses,
        }
    }
}

/// Volcengine credential source.
#[derive(Clone)]
pub struct VolcengineCredential(OpenAiCompatibleCredential);

impl VolcengineCredential {
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

impl fmt::Debug for VolcengineCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("VolcengineCredential")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// Long-lived, model-independent Volcengine provider for ARK language APIs.
#[derive(Clone)]
pub struct VolcengineProvider {
    language: OpenAiCompatibleProvider,
    native: SharedArkNativeRuntime,
    image_scope: Arc<ProviderScope>,
    image_defaults: ArkImageOptions,
    chat_registration: ProviderRegistration,
    responses_registration: ProviderRegistration,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl VolcengineProvider {
    pub fn builder(credential: VolcengineCredential) -> VolcengineProviderBuilder {
        VolcengineProviderBuilder::new(credential)
    }

    pub fn from_api_key(api_key: impl Into<String>) -> Result<Self, VolcengineConfigError> {
        Self::builder(VolcengineCredential::api_key(api_key)).build()
    }

    /// Create a lightweight model in the provider's recommended Responses mode.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<VolcengineLanguageModel, ModelLookupError> {
        self.responses(model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<VolcengineLanguageModel, ModelLookupError> {
        self.language_for(VolcengineLanguageApi::ChatCompletions, model)
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<VolcengineLanguageModel, ModelLookupError> {
        self.language_for(VolcengineLanguageApi::Responses, model)
    }

    pub fn language_for(
        &self,
        api: VolcengineLanguageApi,
        model: impl Into<String>,
    ) -> Result<VolcengineLanguageModel, ModelLookupError> {
        self.language
            .language_for(api.into(), model)
            .map(|inner| VolcengineLanguageModel { inner, api })
    }

    /// Create a lightweight provider-neutral image handle over ARK Images.
    pub fn image(&self, model: impl Into<String>) -> Result<ArkImageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_image_model(model))
    }

    /// Provider-owned ARK image-generation surface.
    pub fn images(&self) -> ArkImages {
        ArkImages::new(self.native.clone())
    }

    /// Provider-owned ARK asynchronous video task lifecycle.
    pub fn video_tasks(&self) -> ArkVideoTasks {
        ArkVideoTasks::new(self.native.clone())
    }

    /// Registration for the recommended Responses mode.
    pub fn registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn registration_for(&self, api: VolcengineLanguageApi) -> ProviderRegistration {
        match api {
            VolcengineLanguageApi::ChatCompletions => self.chat_completions_registration(),
            VolcengineLanguageApi::Responses => self.responses_registration(),
        }
    }

    /// Inspect support evidence for this configured endpoint.
    ///
    /// The provider-owned default endpoint exposes verified ARK evidence. Caller-controlled
    /// endpoints retain the Volcengine identity but expose generic compatibility claims only.
    pub fn profile(&self) -> &siumai_core::ProviderProfile {
        self.language.profile().provider_profile()
    }

    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        &self.support_manifest
    }

    fn create_image_model(&self, model: ModelId) -> ArkImageModel {
        ArkImageModel::new(
            self.native.clone(),
            ModelDescriptor::from_scope(
                self.image_scope.clone(),
                model,
                ModelFamily::Image,
                self.native.instance_id.clone(),
            ),
            self.image_defaults.clone(),
        )
    }
}

impl Provider for VolcengineProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl ImageModelProvider for VolcengineProvider {
    type Model = ArkImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl LanguageModelProvider for VolcengineProvider {
    type Model = VolcengineLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.responses(model.to_string())
    }
}

impl fmt::Debug for VolcengineProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VolcengineProvider")
            .field("provider_id", &PROVIDER_ID)
            .finish()
    }
}

/// Builder for one immutable Volcengine provider runtime.
pub struct VolcengineProviderBuilder {
    credential: VolcengineCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: ArkChatOptions,
    responses_defaults: ArkResponsesOptions,
    image_defaults: ArkImageOptions,
}

impl VolcengineProviderBuilder {
    fn new(credential: VolcengineCredential) -> Self {
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
            chat_defaults: ArkChatOptions::default(),
            responses_defaults: ArkResponsesOptions::default(),
            image_defaults: ArkImageOptions::default(),
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

    /// Bind provider-native replay data to a caller-declared non-secret endpoint audience.
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

    pub fn with_chat_defaults(mut self, defaults: ArkChatOptions) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: ArkResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn with_image_defaults(mut self, defaults: ArkImageOptions) -> Self {
        self.image_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<VolcengineProvider, VolcengineConfigError> {
        self.chat_defaults.validate()?;
        self.responses_defaults.validate()?;
        ProviderOptions::typed(&self.image_defaults)?;
        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let replay_domain = match (self.replay_domain, verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN)?),
            (None, false) => {
                return Err(VolcengineConfigError::CustomEndpointRequiresReplayDomain);
            }
        };
        if replay_domain.audience().is_official() != verified_endpoint {
            return Err(VolcengineConfigError::ReplayAudienceMismatch);
        }

        self.credential.0.validate_static()?;
        let auth = self.credential.0.into_auth();
        let profile = profile(endpoint.clone(), replay_domain.clone(), verified_endpoint)?;
        let image_scope = Arc::new(image_provider_scope(replay_domain.clone())?);
        let image_profile = image_support_profile(verified_endpoint)?;
        let native_claims = if verified_endpoint {
            native_media_claims()?
        } else {
            Vec::new()
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            ProviderId::new(PROVIDER_ID)?,
            [profile.provider_profile().clone(), image_profile],
            native_claims,
        )?);
        let instance_id = ProviderInstanceId::new();
        let mut builder = OpenAiCompatibleProvider::builder_with_auth(profile, auth.clone())
            .with_provider_instance(instance_id.clone())
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
        for (name, value) in option_map(&self.chat_defaults)? {
            builder =
                builder.with_default_option(OpenAiCompatibleApiMode::ChatCompletions, name, value);
        }
        for (name, value) in option_map(&self.responses_defaults)? {
            builder = builder.with_default_option(OpenAiCompatibleApiMode::Responses, name, value);
        }

        let language = builder.build()?;
        let native = Arc::new(ArkNativeRuntime::new(instance_id, native_builder.build()?));
        let image_registration = ProviderRegistration::from_image(
            image_scope.clone(),
            Arc::new({
                let native = native.clone();
                let image_scope = image_scope.clone();
                let image_defaults = self.image_defaults.clone();
                move |model| {
                    Ok(Arc::new(ArkImageModel::new(
                        native.clone(),
                        ModelDescriptor::from_scope(
                            image_scope.clone(),
                            model,
                            ModelFamily::Image,
                            native.instance_id.clone(),
                        ),
                        image_defaults.clone(),
                    )) as Arc<dyn ImageModel>)
                }
            }),
        );
        let chat_registration = language
            .chat_completions_registration()
            .ok_or(VolcengineConfigError::MissingChatMode)?
            .merge(image_registration.clone())?;
        let responses_registration = language
            .responses_registration()
            .ok_or(VolcengineConfigError::MissingResponsesMode)?
            .merge(image_registration)?;
        Ok(VolcengineProvider {
            language,
            native,
            image_scope,
            image_defaults: self.image_defaults,
            chat_registration,
            responses_registration,
            support_manifest,
        })
    }
}

/// Concrete ARK language model backed by one shared Volcengine provider runtime.
#[derive(Clone)]
pub struct VolcengineLanguageModel {
    inner: OpenAiCompatibleLanguageModel,
    api: VolcengineLanguageApi,
}

impl VolcengineLanguageModel {
    pub const fn api(&self) -> VolcengineLanguageApi {
        self.api
    }
}

impl Model for VolcengineLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
}

#[async_trait]
impl LanguageModel for VolcengineLanguageModel {
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

impl fmt::Debug for VolcengineLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VolcengineLanguageModel")
            .field("api", &self.api())
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn image_provider_scope(replay_domain: ReplayDomain) -> Result<ProviderScope, InvalidId> {
    Ok(ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
        .with_platform(PlatformId::new(PLATFORM_ID)?)
        .with_protocol(ProtocolId::new(ARK_IMAGE_PROTOCOL)?)
        .with_api_mode(ApiModeId::new(ARK_IMAGE_API_MODE)?)
        .with_replay_domain(replay_domain))
}

fn image_support_profile(
    verified_endpoint: bool,
) -> Result<ProviderProfile, VolcengineConfigError> {
    let scope = SupportScope::new(
        ProviderId::new(PROVIDER_ID)?,
        PlatformId::new(PLATFORM_ID)?,
        ModelFamily::Image,
        ProtocolId::new(ARK_IMAGE_PROTOCOL)?,
        ApiModeId::new(ARK_IMAGE_API_MODE)?,
    );
    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            ProfileId::new("volcengine-ark-images-custom")?,
            GenericSupportClaim::new(scope, ApiStability::Experimental),
        ));
    }
    Ok(ProviderProfile::verified(
        ProfileId::new("volcengine-ark-images")?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            VerificationEvidence::new(
                OfficialSource::new(IMAGE_SOURCE)?,
                media_verification_date(),
                ProtocolContractId::new("ark-image-generations-2026-08")?,
            ),
        )],
        ModelCatalog::default(),
    )?)
}

fn native_media_claims() -> Result<Vec<VerifiedNativeSupportClaim>, VolcengineConfigError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    Ok(vec![
        VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider.clone(),
                platform.clone(),
                NativeSurfaceKind::Resource,
                NativeSurfaceId::new("image-generation")?,
            ),
            VerifiedFidelity::Native,
            ApiStability::Stable,
            NativeVerificationEvidence::new(
                OfficialSource::new(IMAGE_SOURCE)?,
                media_verification_date(),
            ),
        ),
        VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider,
                platform,
                NativeSurfaceKind::Job,
                NativeSurfaceId::new("video-generation-tasks")?,
            ),
            VerifiedFidelity::Native,
            ApiStability::Stable,
            NativeVerificationEvidence::new(
                OfficialSource::new(VIDEO_SOURCE)?,
                media_verification_date(),
            ),
        ),
    ])
}

fn media_verification_date() -> VerificationDate {
    VerificationDate::new(
        NaiveDate::parse_from_str(MEDIA_VERIFIED_ON, "%Y-%m-%d")
            .expect("ARK media verification date is valid"),
    )
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, VolcengineConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(VolcengineConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum VolcengineConfigError {
    #[error("invalid Volcengine identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Volcengine endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Volcengine profile: {0}")]
    Profile(#[from] VolcengineProfileError),
    #[error("invalid Volcengine credential or compatible runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid Volcengine credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("invalid Volcengine native transport: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Volcengine support profile: {0}")]
    SupportProfile(#[from] ProfileError),
    #[error("invalid Volcengine support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Volcengine registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("invalid ARK default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("ARK default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("ARK default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("the Volcengine profile omitted its required Chat Completions mode")]
    MissingChatMode,
    #[error("the Volcengine profile omitted its required Responses mode")]
    MissingResponsesMode,
    #[error("a caller-controlled Volcengine endpoint requires an explicit custom replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Volcengine endpoint ownership")]
    ReplayAudienceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ApiModeId, ModelFamily, Provider};
    use siumai_transport::EndpointPolicy;

    #[test]
    fn credentials_are_redacted() {
        let debug = format!("{:?}", VolcengineCredential::api_key("canary-secret"));
        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn configured_provider_image_models_share_one_instance_capability() {
        let first = VolcengineProvider::builder(VolcengineCredential::api_key("first-key"))
            .build()
            .unwrap();
        let second = VolcengineProvider::builder(VolcengineCredential::api_key("second-key"))
            .build()
            .unwrap();

        let first_model = first.image("future-image").unwrap();
        let same_provider = first.image("another-image").unwrap();
        let other_provider = second.image("future-image").unwrap();

        assert_eq!(
            first_model.descriptor().instance_id(),
            same_provider.descriptor().instance_id()
        );
        assert_ne!(
            first_model.descriptor().instance_id(),
            other_provider.descriptor().instance_id()
        );
    }

    #[test]
    fn caller_endpoint_requires_custom_replay_even_when_marked_official() {
        let endpoint = EndpointConfig::official(
            "https://relay.example.test/api/v3",
            OfficialOrigin::new("https://relay.example.test").expect("origin"),
        )
        .expect("endpoint");
        assert!(matches!(endpoint.policy(), EndpointPolicy::Official(_)));

        let missing = VolcengineProvider::builder(VolcengineCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .expect_err("caller endpoint needs an explicit replay domain");
        assert!(matches!(
            missing,
            VolcengineConfigError::CustomEndpointRequiresReplayDomain
        ));

        let official_audience =
            VolcengineProvider::builder(VolcengineCredential::unauthenticated())
                .with_endpoint(endpoint.clone())
                .with_replay_domain(ReplayDomain::official(
                    ReplayDomainId::new("forged-official").expect("domain"),
                ))
                .build()
                .expect_err("caller endpoint cannot inherit official replay");
        assert!(matches!(
            official_audience,
            VolcengineConfigError::ReplayAudienceMismatch
        ));

        let provider = VolcengineProvider::builder(VolcengineCredential::unauthenticated())
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
    fn official_profile_is_verified_and_future_models_remain_open() {
        let provider = VolcengineProvider::builder(VolcengineCredential::unauthenticated())
            .build()
            .expect("provider");
        let model = provider
            .language("doubao-seed-3-future")
            .expect("future model");

        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert!(provider.profile().verified_claims().is_some());
        assert_eq!(model.descriptor().model().as_str(), "doubao-seed-3-future");
        assert_eq!(model.api(), VolcengineLanguageApi::Responses);
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Language)
                .map(ApiModeId::as_str),
            Some("responses")
        );
        assert_eq!(
            provider
                .chat_completions("custom-deployment-id")
                .expect("chat model")
                .api(),
            VolcengineLanguageApi::ChatCompletions
        );
    }
}
