use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, LanguageModel, LanguageModelProvider, LanguageRequest, LanguageResponse,
    LanguageStream, Model, ModelDescriptor, ModelId, ModelLookupError, Provider,
    ProviderOptionError, ProviderRegistration, TypedProviderOptions,
};
use siumai_openai_compatible::{
    DynamicCredentialSource, OpenAiCompatibleApiMode, OpenAiCompatibleConfigError,
    OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel, OpenAiCompatibleProvider,
};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, RetryPolicy, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::language::{DEFAULT_BASE_URL, DeepSeekProfileError, PROVIDER_ID, profile};
use crate::options::{DeepSeekChatOptions, DeepSeekResponsesOptions};

const OFFICIAL_ORIGIN: &str = "https://api.deepseek.com";

/// Public DeepSeek language API selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeepSeekLanguageApi {
    #[default]
    ChatCompletions,
    Responses,
}

impl From<DeepSeekLanguageApi> for OpenAiCompatibleApiMode {
    fn from(value: DeepSeekLanguageApi) -> Self {
        match value {
            DeepSeekLanguageApi::ChatCompletions => Self::ChatCompletions,
            DeepSeekLanguageApi::Responses => Self::Responses,
        }
    }
}

/// DeepSeek credential source.
#[derive(Clone)]
pub struct DeepSeekCredential(OpenAiCompatibleCredential);

impl DeepSeekCredential {
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
    chat_registration: ProviderRegistration,
    responses_registration: ProviderRegistration,
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

    pub fn language_for(
        &self,
        api: DeepSeekLanguageApi,
        model: impl Into<String>,
    ) -> Result<DeepSeekLanguageModel, ModelLookupError> {
        self.language
            .language_for(api.into(), model)
            .map(|inner| DeepSeekLanguageModel { inner, api })
    }

    /// Registration for the recommended Chat Completions language mode.
    pub fn registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn registration_for(&self, api: DeepSeekLanguageApi) -> ProviderRegistration {
        match api {
            DeepSeekLanguageApi::ChatCompletions => self.chat_completions_registration(),
            DeepSeekLanguageApi::Responses => self.responses_registration(),
        }
    }

    /// Inspect the exact language support claims for this configuration.
    pub fn profile(&self) -> &siumai_core::ProviderProfile {
        self.language.profile().provider_profile()
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
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            chat_defaults: DeepSeekChatOptions::default(),
            responses_defaults: DeepSeekResponsesOptions::default(),
        }
    }

    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self
    }

    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::public_custom(base_url);
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
        let profile = profile(endpoint)?;
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
            .ok_or(DeepSeekConfigError::MissingChatMode)?;
        let responses_registration = language
            .responses_registration()
            .ok_or(DeepSeekConfigError::MissingResponsesMode)?;
        Ok(DeepSeekProvider {
            language,
            chat_registration,
            responses_registration,
        })
    }
}

/// Concrete DeepSeek language model backed by one shared provider runtime.
#[derive(Clone)]
pub struct DeepSeekLanguageModel {
    inner: OpenAiCompatibleLanguageModel,
    api: DeepSeekLanguageApi,
}

impl DeepSeekLanguageModel {
    pub const fn api(&self) -> DeepSeekLanguageApi {
        self.api
    }
}

impl Model for DeepSeekLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
}

#[async_trait]
impl LanguageModel for DeepSeekLanguageModel {
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

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, DeepSeekConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(DeepSeekConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum DeepSeekConfigError {
    #[error("invalid DeepSeek endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid DeepSeek profile: {0}")]
    Profile(#[from] DeepSeekProfileError),
    #[error("invalid DeepSeek credential or compatible runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid DeepSeek default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("DeepSeek default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("DeepSeek default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("DeepSeek profile omitted its required Chat Completions mode")]
    MissingChatMode,
    #[error("DeepSeek profile omitted its required Responses mode")]
    MissingResponsesMode,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ApiModeId, ModelFamily, Provider};

    #[test]
    fn credentials_are_redacted() {
        let debug = format!("{:?}", DeepSeekCredential::api_key("canary-secret"));
        assert!(!debug.contains("canary-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn custom_provider_keeps_chat_as_the_primary_registration() {
        let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
            .with_endpoint(
                EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
            )
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
