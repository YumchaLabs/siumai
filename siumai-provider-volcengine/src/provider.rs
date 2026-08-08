//! Long-lived Volcengine provider and synchronous ARK model construction.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, InvalidId, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelId, ModelLookupError, Provider,
    ProviderOptionError, ProviderRegistration, ReplayDomain, ReplayDomainId, TypedProviderOptions,
};
use siumai_openai_compatible::{
    DynamicCredentialSource, OpenAiCompatibleApiMode, OpenAiCompatibleConfigError,
    OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel, OpenAiCompatibleProvider,
};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, RetryPolicy, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::language::{DEFAULT_BASE_URL, PROVIDER_ID, VolcengineProfileError, profile};
use crate::options::{ArkChatOptions, ArkResponsesOptions};

const OFFICIAL_ORIGIN: &str = "https://ark.cn-beijing.volces.com";
const OFFICIAL_REPLAY_DOMAIN: &str = "volcengine-ark-cn-beijing";

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
    chat_registration: ProviderRegistration,
    responses_registration: ProviderRegistration,
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
}

impl Provider for VolcengineProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.language.provider_id()
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

    pub fn build(self) -> Result<VolcengineProvider, VolcengineConfigError> {
        self.chat_defaults.validate()?;
        self.responses_defaults.validate()?;
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

        let profile = profile(endpoint, replay_domain, verified_endpoint)?;
        let mut builder = OpenAiCompatibleProvider::builder(profile, self.credential.0)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            builder = builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            builder = builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            builder = builder.with_read_timeout(timeout);
        }
        for (name, value) in option_map(&self.chat_defaults)? {
            builder =
                builder.with_default_option(OpenAiCompatibleApiMode::ChatCompletions, name, value);
        }
        for (name, value) in option_map(&self.responses_defaults)? {
            builder = builder.with_default_option(OpenAiCompatibleApiMode::Responses, name, value);
        }

        let language = builder.build()?;
        let chat_registration = language
            .chat_completions_registration()
            .ok_or(VolcengineConfigError::MissingChatMode)?;
        let responses_registration = language
            .responses_registration()
            .ok_or(VolcengineConfigError::MissingResponsesMode)?;
        Ok(VolcengineProvider {
            language,
            chat_registration,
            responses_registration,
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
