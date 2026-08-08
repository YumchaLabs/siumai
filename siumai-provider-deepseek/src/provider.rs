use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleCredential,
    AnthropicCompatibleLanguageModel, AnthropicCompatibleProvider,
};
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

use crate::language::{
    BETA_BASE_URL, DEFAULT_BASE_URL, DeepSeekProfileError, MESSAGES_BASE_URL, PROVIDER_ID,
    beta_profile, messages_profile, profile,
};
use crate::options::{DeepSeekChatOptions, DeepSeekResponsesOptions};

const OFFICIAL_ORIGIN: &str = "https://api.deepseek.com";
const OFFICIAL_REPLAY_DOMAIN: &str = "deepseek-public-api";
const OFFICIAL_BETA_REPLAY_DOMAIN: &str = "deepseek-public-beta-api";
const OFFICIAL_MESSAGES_REPLAY_DOMAIN: &str = "deepseek-public-messages-api";

/// Public DeepSeek language API selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeepSeekLanguageApi {
    #[default]
    ChatCompletions,
    BetaChatCompletions,
    Responses,
    Messages,
}

/// DeepSeek credential source.
#[derive(Clone)]
pub struct DeepSeekCredential {
    openai: OpenAiCompatibleCredential,
    messages: Option<AnthropicCompatibleCredential>,
}

impl DeepSeekCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            openai: OpenAiCompatibleCredential::api_key(value.clone()),
            messages: Some(AnthropicCompatibleCredential::api_key(value)),
        }
    }

    /// Configure a rotating bearer source for the OpenAI-compatible endpoints.
    ///
    /// The generic dynamic source cannot expose its secret for DeepSeek's required `x-api-key`
    /// Messages header, so Messages remains unavailable unless [`Self::with_messages_api_key`] is
    /// also selected explicitly.
    pub fn dynamic(source: Arc<dyn DynamicCredentialSource>) -> Self {
        Self {
            openai: OpenAiCompatibleCredential::dynamic(source),
            messages: None,
        }
    }

    /// Add an explicit static `x-api-key` credential for the Messages endpoint.
    pub fn with_messages_api_key(mut self, value: impl Into<String>) -> Self {
        self.messages = Some(AnthropicCompatibleCredential::api_key(value));
        self
    }

    pub fn unauthenticated() -> Self {
        Self {
            openai: OpenAiCompatibleCredential::unauthenticated(),
            messages: Some(AnthropicCompatibleCredential::unauthenticated()),
        }
    }
}

impl fmt::Debug for DeepSeekCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("DeepSeekCredential")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// Long-lived configured DeepSeek provider.
#[derive(Clone)]
pub struct DeepSeekProvider {
    language: OpenAiCompatibleProvider,
    beta: OpenAiCompatibleProvider,
    messages: Option<AnthropicCompatibleProvider>,
    chat_registration: ProviderRegistration,
    beta_chat_registration: ProviderRegistration,
    responses_registration: ProviderRegistration,
    messages_registration: Option<ProviderRegistration>,
}

impl DeepSeekProvider {
    pub fn builder(credential: DeepSeekCredential) -> DeepSeekProviderBuilder {
        DeepSeekProviderBuilder::new(credential)
    }

    /// Construct the broadly supported Chat Completions model handle.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.chat_completions(model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.language_for(DeepSeekLanguageApi::ChatCompletions, model)
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.language_for(DeepSeekLanguageApi::Responses, model)
    }

    /// Construct an explicit DeepSeek beta Chat handle for strict tools and prefix completion.
    pub fn beta_chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.language_for(DeepSeekLanguageApi::BetaChatCompletions, model)
    }

    pub fn messages(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.language_for(DeepSeekLanguageApi::Messages, model)
    }

    pub fn language_for(
        &self,
        api: DeepSeekLanguageApi,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        let model = model.into();
        match api {
            DeepSeekLanguageApi::ChatCompletions => self
                .language
                .language_for(OpenAiCompatibleApiMode::ChatCompletions, model)
                .map(|inner| DeepSeekLanguageModel::openai(api, inner)),
            DeepSeekLanguageApi::BetaChatCompletions => self
                .beta
                .language_for(OpenAiCompatibleApiMode::ChatCompletions, model)
                .map(|inner| DeepSeekLanguageModel::openai(api, inner)),
            DeepSeekLanguageApi::Responses => self
                .language
                .language_for(OpenAiCompatibleApiMode::Responses, model)
                .map(|inner| DeepSeekLanguageModel::openai(api, inner)),
            DeepSeekLanguageApi::Messages => self
                .messages
                .as_ref()
                .ok_or_else(messages_unavailable)?
                .language(model)
                .map(DeepSeekLanguageModel::messages),
        }
    }

    /// Registration for the recommended Chat Completions language mode.
    pub fn registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn beta_chat_completions_registration(&self) -> ProviderRegistration {
        self.beta_chat_registration.clone()
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn messages_registration(&self) -> Option<ProviderRegistration> {
        self.messages_registration.clone()
    }

    pub fn registration_for(&self, api: DeepSeekLanguageApi) -> Option<ProviderRegistration> {
        match api {
            DeepSeekLanguageApi::ChatCompletions => Some(self.chat_completions_registration()),
            DeepSeekLanguageApi::BetaChatCompletions => {
                Some(self.beta_chat_completions_registration())
            }
            DeepSeekLanguageApi::Responses => Some(self.responses_registration()),
            DeepSeekLanguageApi::Messages => self.messages_registration(),
        }
    }

    /// Inspect the exact language support claims for this configuration.
    pub fn profile(&self) -> &siumai_core::ProviderProfile {
        self.language.profile().provider_profile()
    }

    /// Inspect the beta Chat support claim for strict tools and prefix completion.
    pub fn beta_profile(&self) -> &siumai_core::ProviderProfile {
        self.beta.profile().provider_profile()
    }

    /// Inspect the exact Messages support claim when this credential configuration exposes it.
    pub fn messages_profile(&self) -> Option<&siumai_core::ProviderProfile> {
        self.messages
            .as_ref()
            .map(|provider| provider.profile().provider_profile())
    }
}

impl Provider for DeepSeekProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.language.provider_id()
    }
}

impl LanguageModelProvider for DeepSeekProvider {
    type Model = DeepSeekLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.chat_completions(model.to_string())
    }
}

impl fmt::Debug for DeepSeekProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepSeekProvider")
            .field("provider_id", &PROVIDER_ID)
            .finish()
    }
}

/// Builder for one immutable DeepSeek provider runtime.
pub struct DeepSeekProviderBuilder {
    credential: DeepSeekCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    beta_endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_beta_endpoint: bool,
    beta_replay_domain: Option<ReplayDomain>,
    messages_endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_messages_endpoint: bool,
    messages_replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: DeepSeekChatOptions,
    responses_defaults: DeepSeekResponsesOptions,
}

impl DeepSeekProviderBuilder {
    fn new(credential: DeepSeekCredential) -> Self {
        Self {
            credential,
            endpoint: official_endpoint(),
            provider_selected_endpoint: true,
            replay_domain: None,
            beta_endpoint: official_beta_endpoint(),
            provider_selected_beta_endpoint: true,
            beta_replay_domain: None,
            messages_endpoint: official_messages_endpoint(),
            provider_selected_messages_endpoint: true,
            messages_replay_domain: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            chat_defaults: DeepSeekChatOptions::default(),
            responses_defaults: DeepSeekResponsesOptions::default(),
        }
    }

    /// Replace the default endpoint with a caller-controlled endpoint.
    ///
    /// The endpoint remains caller-controlled even when its transport policy is marked official.
    /// A matching [`ReplayDomain::custom`] is required before `build`.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self.provider_selected_endpoint = false;
        self
    }

    /// Replace the default endpoint with a caller-selected public endpoint.
    ///
    /// Call [`Self::with_replay_domain`] with an explicit custom audience before `build`.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::public_custom(base_url);
        self.provider_selected_endpoint = false;
        self
    }

    /// Replace the official DeepSeek beta endpoint used by explicit beta Chat handles.
    ///
    /// The endpoint remains caller-controlled even when its transport policy is marked official.
    /// A matching [`ReplayDomain::custom`] must be supplied through
    /// [`Self::with_beta_replay_domain`] before `build`.
    pub fn with_beta_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.beta_endpoint = Ok(endpoint);
        self.provider_selected_beta_endpoint = false;
        self
    }

    /// Replace the official beta base URL with a caller-controlled endpoint.
    pub fn with_beta_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.beta_endpoint = EndpointConfig::public_custom(base_url);
        self.provider_selected_beta_endpoint = false;
        self
    }

    /// Replace the official Anthropic-compatible Messages endpoint.
    ///
    /// A matching [`ReplayDomain::custom`] must be supplied through
    /// [`Self::with_messages_replay_domain`] before `build`.
    pub fn with_messages_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.messages_endpoint = Ok(endpoint);
        self.provider_selected_messages_endpoint = false;
        self
    }

    /// Replace the official Messages base URL with a caller-controlled endpoint.
    pub fn with_messages_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.messages_endpoint = EndpointConfig::public_custom(base_url);
        self.provider_selected_messages_endpoint = false;
        self
    }

    /// Bind provider-native history to a caller-declared non-secret replay identity.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    /// Bind beta Chat history to a caller-declared non-secret replay identity.
    pub fn with_beta_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.beta_replay_domain = Some(replay_domain);
        self
    }

    /// Bind Anthropic-compatible Messages history to a caller-declared replay identity.
    pub fn with_messages_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.messages_replay_domain = Some(replay_domain);
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

    pub fn with_chat_defaults(mut self, defaults: DeepSeekChatOptions) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: DeepSeekResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<DeepSeekProvider, DeepSeekConfigError> {
        self.chat_defaults.validate()?;
        self.responses_defaults.validate()?;
        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let beta_endpoint = self.beta_endpoint?;
        let verified_beta_endpoint = self.provider_selected_beta_endpoint;
        let messages_endpoint = self.messages_endpoint?;
        let verified_messages_endpoint = self.provider_selected_messages_endpoint;
        let replay_domain = match (self.replay_domain, verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN)?),
            (None, false) => return Err(DeepSeekConfigError::CustomEndpointRequiresReplayDomain),
        };
        if replay_domain.audience().is_official() != verified_endpoint {
            return Err(DeepSeekConfigError::ReplayAudienceMismatch);
        }
        let beta_replay_domain = match (self.beta_replay_domain, verified_beta_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => {
                ReplayDomain::official(ReplayDomainId::new(OFFICIAL_BETA_REPLAY_DOMAIN)?)
            }
            (None, false) => {
                return Err(DeepSeekConfigError::CustomBetaEndpointRequiresReplayDomain);
            }
        };
        if beta_replay_domain.audience().is_official() != verified_beta_endpoint {
            return Err(DeepSeekConfigError::BetaReplayAudienceMismatch);
        }
        let messages_replay_domain = match (self.messages_replay_domain, verified_messages_endpoint)
        {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => {
                ReplayDomain::official(ReplayDomainId::new(OFFICIAL_MESSAGES_REPLAY_DOMAIN)?)
            }
            (None, false) => {
                return Err(DeepSeekConfigError::CustomMessagesEndpointRequiresReplayDomain);
            }
        };
        if messages_replay_domain.audience().is_official() != verified_messages_endpoint {
            return Err(DeepSeekConfigError::MessagesReplayAudienceMismatch);
        }
        let DeepSeekCredential { openai, messages } = self.credential;
        let beta_credential = openai.clone();
        let profile = profile(endpoint, replay_domain, verified_endpoint)?;
        let mut builder = OpenAiCompatibleProvider::builder(profile, openai)
            .with_limits(self.limits.clone())
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
        let beta_profile = beta_profile(beta_endpoint, beta_replay_domain, verified_beta_endpoint)?;
        let mut beta_builder = OpenAiCompatibleProvider::builder(beta_profile, beta_credential)
            .with_limits(self.limits.clone())
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            beta_builder = beta_builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            beta_builder = beta_builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            beta_builder = beta_builder.with_read_timeout(timeout);
        }
        for (name, value) in option_map(&self.chat_defaults)? {
            beta_builder = beta_builder.with_default_option(
                OpenAiCompatibleApiMode::ChatCompletions,
                name,
                value,
            );
        }
        let beta = beta_builder.build()?;
        let messages = messages
            .map(|credential| {
                let profile = messages_profile(
                    messages_endpoint,
                    messages_replay_domain,
                    verified_messages_endpoint,
                )?;
                let mut builder = AnthropicCompatibleProvider::builder(profile, credential)
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
                Ok::<_, DeepSeekConfigError>(builder.build()?)
            })
            .transpose()?;
        let chat_registration = language
            .chat_completions_registration()
            .ok_or(DeepSeekConfigError::MissingChatMode)?;
        let responses_registration = language
            .responses_registration()
            .ok_or(DeepSeekConfigError::MissingResponsesMode)?;
        let beta_chat_registration = beta
            .chat_completions_registration()
            .ok_or(DeepSeekConfigError::MissingBetaChatMode)?;
        let messages_registration = messages
            .as_ref()
            .map(AnthropicCompatibleProvider::registration);
        Ok(DeepSeekProvider {
            language,
            beta,
            messages,
            chat_registration,
            beta_chat_registration,
            responses_registration,
            messages_registration,
        })
    }
}

/// Concrete DeepSeek language model backed by one provider-owned protocol runtime.
#[derive(Clone)]
pub struct DeepSeekLanguageModel {
    api: DeepSeekLanguageApi,
    inner: DeepSeekLanguageModelInner,
}

#[derive(Clone)]
enum DeepSeekLanguageModelInner {
    OpenAi(OpenAiCompatibleLanguageModel),
    Messages(AnthropicCompatibleLanguageModel),
}

impl DeepSeekLanguageModel {
    fn openai(api: DeepSeekLanguageApi, inner: OpenAiCompatibleLanguageModel) -> Self {
        Self {
            api,
            inner: DeepSeekLanguageModelInner::OpenAi(inner),
        }
    }

    fn messages(inner: AnthropicCompatibleLanguageModel) -> Self {
        Self {
            api: DeepSeekLanguageApi::Messages,
            inner: DeepSeekLanguageModelInner::Messages(inner),
        }
    }

    pub const fn api(&self) -> DeepSeekLanguageApi {
        self.api
    }
}

impl Model for DeepSeekLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        match &self.inner {
            DeepSeekLanguageModelInner::OpenAi(model) => model.descriptor(),
            DeepSeekLanguageModelInner::Messages(model) => model.descriptor(),
        }
    }
}

#[async_trait]
impl LanguageModel for DeepSeekLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        match &self.inner {
            DeepSeekLanguageModelInner::OpenAi(model) => model.generate(request, options).await,
            DeepSeekLanguageModelInner::Messages(model) => model.generate(request, options).await,
        }
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        match &self.inner {
            DeepSeekLanguageModelInner::OpenAi(model) => model.stream(request, options).await,
            DeepSeekLanguageModelInner::Messages(model) => model.stream(request, options).await,
        }
    }
}

impl fmt::Debug for DeepSeekLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepSeekLanguageModel")
            .field("api", &self.api())
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn official_beta_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(BETA_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn official_messages_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(MESSAGES_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn messages_unavailable() -> ModelLookupError {
    ModelLookupError::Construction {
        source: Error::new(
            siumai_core::ErrorKind::Unsupported,
            "this DeepSeek credential configuration does not expose Anthropic-compatible Messages",
        ),
    }
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, DeepSeekConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(DeepSeekConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum DeepSeekConfigError {
    #[error("invalid DeepSeek identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid DeepSeek endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid DeepSeek profile: {0}")]
    Profile(#[from] DeepSeekProfileError),
    #[error("invalid DeepSeek credential or compatible runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid DeepSeek Anthropic-compatible Messages runtime: {0}")]
    MessagesCompatible(#[from] AnthropicCompatibleConfigError),
    #[error("invalid DeepSeek default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("DeepSeek default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("DeepSeek default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("DeepSeek profile omitted its required Chat Completions mode")]
    MissingChatMode,
    #[error("DeepSeek beta profile omitted its required Chat Completions mode")]
    MissingBetaChatMode,
    #[error("DeepSeek profile omitted its required Responses mode")]
    MissingResponsesMode,
    #[error("a custom DeepSeek endpoint requires an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured DeepSeek endpoint ownership")]
    ReplayAudienceMismatch,
    #[error("a custom DeepSeek beta endpoint requires an explicit non-secret replay domain")]
    CustomBetaEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured DeepSeek beta endpoint ownership")]
    BetaReplayAudienceMismatch,
    #[error("a custom DeepSeek Messages endpoint requires an explicit non-secret replay domain")]
    CustomMessagesEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured DeepSeek Messages endpoint ownership")]
    MessagesReplayAudienceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ApiModeId, ModelFamily, Provider, ReplayDomain, ReplayDomainId};
    use siumai_openai_compatible::{BearerCredential, CredentialRequest, CredentialSourceError};

    struct UnavailableDynamicCredential;

    #[async_trait]
    impl DynamicCredentialSource for UnavailableDynamicCredential {
        async fn load(
            &self,
            _request: CredentialRequest,
        ) -> Result<BearerCredential, CredentialSourceError> {
            Err(CredentialSourceError::unavailable())
        }
    }

    #[test]
    fn credentials_are_redacted() {
        let debug = format!("{:?}", DeepSeekCredential::api_key("canary-secret"));
        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn generic_dynamic_credentials_disable_messages_honestly() {
        let provider = DeepSeekProvider::builder(DeepSeekCredential::dynamic(Arc::new(
            UnavailableDynamicCredential,
        )))
        .build()
        .expect("dynamic provider");

        assert!(provider.messages_registration().is_none());
        let error = provider
            .messages("deepseek-v4-flash")
            .expect_err("Messages requires an explicit x-api-key credential");
        assert!(matches!(error, ModelLookupError::Construction { .. }));
    }

    #[test]
    fn custom_provider_keeps_chat_as_the_primary_registration() {
        let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
            )
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-endpoint").expect("replay domain"),
            ))
            .build()
            .expect("provider");

        assert_eq!(provider.provider_id().as_str(), PROVIDER_ID);
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Language)
                .map(ApiModeId::as_str),
            Some("chat-completions")
        );
        assert_eq!(
            provider
                .language("future-model")
                .expect("model")
                .descriptor()
                .api_mode(),
            Some("chat-completions")
        );
        assert_eq!(
            provider
                .responses("future-model")
                .expect("model")
                .descriptor()
                .api_mode(),
            Some("responses")
        );
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Language)
                .map(ApiModeId::as_str),
            Some("chat-completions")
        );
    }
}
