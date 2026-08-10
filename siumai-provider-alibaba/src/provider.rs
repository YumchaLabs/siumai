use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleLanguageModel, AnthropicCompatibleProvider,
};
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, EmbeddingModel, EmbeddingModelProvider, Error, ErrorKind,
    InvalidId, LanguageModel, LanguageModelProvider, LanguageRequest, LanguageResponse,
    LanguageStream, Model, ModelDescriptor, ModelFamily, ModelId, ModelLookupError,
    NativeSupportScope, NativeSurfaceId, NativeSurfaceKind, NativeVerificationEvidence,
    OfficialSource, PlatformId, ProfileError, ProtocolId, Provider, ProviderId, ProviderInstanceId,
    ProviderRegistration, ProviderRegistrationError, ProviderScope, ProviderSupportManifest,
    ReplayDomain, ReplayDomainId, SupportManifestError, VerificationDate, VerifiedFidelity,
    VerifiedNativeSupportClaim,
};
use siumai_openai_compatible::{
    CredentialSourceError, DynamicCredentialSource, OpenAiCompatibleApiMode,
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProvider,
};
use siumai_transport::{
    AuthApplier, EndpointConfig, EndpointError, EndpointPolicy, OfficialOrigin, ProviderTransport,
    ReplaySafety, ResourceDownloader, RetryPolicy, TransportConfigError, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::embedding::{
    AlibabaEmbeddingModel, AlibabaEmbeddingOptions, AlibabaEmbeddingProfileError,
    AlibabaEmbeddingRuntime, EMBEDDING_API_MODE_ID, EMBEDDING_PROTOCOL_ID,
    LEGACY_SINGAPORE_EMBEDDING_BASE_URL, support_profile,
};
use crate::language::{
    AlibabaProfileError, LEGACY_SINGAPORE_LANGUAGE_BASE_URL, LEGACY_SINGAPORE_MESSAGES_BASE_URL,
    PLATFORM_ID, PROVIDER_ID, messages_profile, profile,
};
use crate::options::{AlibabaChatOptions, AlibabaMessagesOptions, AlibabaResponsesOptions};
use crate::video::{
    AlibabaVideoDownloadPolicy, AlibabaVideoModel, AlibabaVideoRuntime,
    LEGACY_SINGAPORE_VIDEO_BASE_URL, VIDEO_API_MODE_ID, VIDEO_PROTOCOL_ID, VIDEO_TEXT_SOURCE,
};

pub const LEGACY_SINGAPORE_ORIGIN: &str = "https://dashscope-intl.aliyuncs.com";
const SUPPORT_VERIFIED_ON: &str = "2026-08-06";
const LEGACY_SINGAPORE_LANGUAGE_REPLAY_DOMAIN: &str = "legacy-singapore-language-api";
const LEGACY_SINGAPORE_MESSAGES_REPLAY_DOMAIN: &str = "legacy-singapore-messages-api";

/// Explicit caller-owned Model Studio workspace origin.
///
/// The origin includes the workspace identifier and Alibaba Cloud deployment host, for example
/// `https://workspace-id.ap-southeast-1.maas.aliyuncs.com`. Siumai deliberately does not model
/// regions or maintain a deployment catalog; the host application obtains and selects this
/// technical address.
#[derive(Clone)]
pub struct AlibabaWorkspaceEndpoint {
    language: EndpointConfig,
    native: EndpointConfig,
    messages: EndpointConfig,
}

impl AlibabaWorkspaceEndpoint {
    /// Derive the current compatible-language and native API bases from a workspace origin.
    pub fn public_origin(origin: impl AsRef<str>) -> Result<Self, AlibabaWorkspaceEndpointError> {
        let origin = EndpointConfig::public_custom(origin)?;
        let url = origin.expose_base_url();
        if url.path() != "/" || url.query().is_some() {
            return Err(AlibabaWorkspaceEndpointError::OriginMustNotContainPathOrQuery);
        }
        let origin = url.as_str().trim_end_matches('/');
        Ok(Self {
            language: EndpointConfig::public_custom(format!("{origin}/compatible-mode/v1"))?,
            native: EndpointConfig::public_custom(format!("{origin}/api/v1"))?,
            messages: EndpointConfig::public_custom(format!("{origin}/apps/anthropic"))?,
        })
    }

    pub fn language_endpoint(&self) -> EndpointConfig {
        self.language.clone()
    }

    pub fn embedding_endpoint(&self) -> EndpointConfig {
        self.native.clone()
    }

    pub fn video_endpoint(&self) -> EndpointConfig {
        self.native.clone()
    }

    pub fn messages_endpoint(&self) -> EndpointConfig {
        self.messages.clone()
    }
}

impl fmt::Debug for AlibabaWorkspaceEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaWorkspaceEndpoint")
            .field("origin", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum AlibabaWorkspaceEndpointError {
    #[error("invalid Alibaba workspace endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("Alibaba workspace origin must not contain a path or query")]
    OriginMustNotContainPathOrQuery,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlibabaLanguageApi {
    #[default]
    Responses,
    ChatCompletions,
    Messages,
}

#[derive(Clone)]
pub struct AlibabaCredential(OpenAiCompatibleCredential);

impl AlibabaCredential {
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

impl fmt::Debug for AlibabaCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("AlibabaCredential")
            .field(&"[REDACTED]")
            .finish()
    }
}

#[derive(Clone)]
pub struct AlibabaProvider {
    language: Option<OpenAiCompatibleProvider>,
    messages: Option<AnthropicCompatibleProvider>,
    embedding: Option<Arc<AlibabaEmbeddingRuntime>>,
    video: Option<Arc<AlibabaVideoRuntime>>,
    default_registration: Option<ProviderRegistration>,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl AlibabaProvider {
    pub fn builder(credential: AlibabaCredential) -> AlibabaProviderBuilder {
        AlibabaProviderBuilder::new(credential)
    }

    /// Inspect the exact portable and provider-native support claims for this configuration.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        &self.support_manifest
    }

    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<AlibabaLanguageModel, ModelLookupError> {
        self.language_for(AlibabaLanguageApi::Responses, model)
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<AlibabaLanguageModel, ModelLookupError> {
        self.language_for(AlibabaLanguageApi::Responses, model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<AlibabaLanguageModel, ModelLookupError> {
        self.language_for(AlibabaLanguageApi::ChatCompletions, model)
    }

    pub fn messages(
        &self,
        model: impl Into<String>,
    ) -> Result<AlibabaLanguageModel, ModelLookupError> {
        self.language_for(AlibabaLanguageApi::Messages, model)
    }

    pub fn language_for(
        &self,
        api: AlibabaLanguageApi,
        model: impl Into<String>,
    ) -> Result<AlibabaLanguageModel, ModelLookupError> {
        let model = model.into();
        match api {
            AlibabaLanguageApi::Responses => self
                .language
                .as_ref()
                .ok_or_else(|| self.unsupported_family(ModelFamily::Language))?
                .language_for(OpenAiCompatibleApiMode::Responses, model)
                .map(|inner| AlibabaLanguageModel::openai(api, inner)),
            AlibabaLanguageApi::ChatCompletions => self
                .language
                .as_ref()
                .ok_or_else(|| self.unsupported_family(ModelFamily::Language))?
                .language_for(OpenAiCompatibleApiMode::ChatCompletions, model)
                .map(|inner| AlibabaLanguageModel::openai(api, inner)),
            AlibabaLanguageApi::Messages => self
                .messages
                .as_ref()
                .ok_or_else(|| self.unsupported_family(ModelFamily::Language))?
                .language(model)
                .map(AlibabaLanguageModel::messages),
        }
    }

    pub fn embedding(
        &self,
        model: impl Into<String>,
    ) -> Result<AlibabaEmbeddingModel, ModelLookupError> {
        let runtime = self
            .embedding
            .as_ref()
            .ok_or_else(|| self.unsupported_family(ModelFamily::Embedding))?;
        let model = ModelId::new(model.into())?;
        Ok(AlibabaEmbeddingModel::new(runtime.clone(), model))
    }

    /// Default registration for configured stable families.
    ///
    /// Language uses the recommended Responses mode; embedding keeps its native scope.
    /// A video-only provider has no Registry registration because video remains a provider-owned job.
    pub fn registration(&self) -> Option<ProviderRegistration> {
        self.default_registration.clone()
    }

    pub fn responses_registration(&self) -> Option<ProviderRegistration> {
        self.language
            .as_ref()
            .and_then(OpenAiCompatibleProvider::responses_registration)
    }

    pub fn chat_completions_registration(&self) -> Option<ProviderRegistration> {
        self.language
            .as_ref()
            .and_then(OpenAiCompatibleProvider::chat_completions_registration)
    }

    pub fn messages_registration(&self) -> Option<ProviderRegistration> {
        self.messages
            .as_ref()
            .map(AnthropicCompatibleProvider::registration)
    }

    pub fn registration_for(&self, api: AlibabaLanguageApi) -> Option<ProviderRegistration> {
        match api {
            AlibabaLanguageApi::Responses => self.responses_registration(),
            AlibabaLanguageApi::ChatCompletions => self.chat_completions_registration(),
            AlibabaLanguageApi::Messages => self.messages_registration(),
        }
    }

    pub fn embedding_registration(&self) -> Option<ProviderRegistration> {
        self.embedding.as_ref().cloned().map(embedding_registration)
    }

    fn create_embedding_model(
        &self,
        model: ModelId,
    ) -> Result<AlibabaEmbeddingModel, ModelLookupError> {
        self.embedding
            .as_ref()
            .cloned()
            .map(|runtime| AlibabaEmbeddingModel::new(runtime, model))
            .ok_or_else(|| self.unsupported_family(ModelFamily::Embedding))
    }

    fn unsupported_family(&self, family: ModelFamily) -> ModelLookupError {
        ModelLookupError::UnsupportedFamily {
            provider: self.support_manifest.provider_id().clone(),
            family,
        }
    }
}

impl Provider for AlibabaProvider {
    fn provider_id(&self) -> &ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for AlibabaProvider {
    type Model = AlibabaLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.language_for(AlibabaLanguageApi::Responses, model.to_string())
    }
}

impl EmbeddingModelProvider for AlibabaProvider {
    type Model = AlibabaEmbeddingModel;

    fn embedding_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.create_embedding_model(model)
    }
}

impl fmt::Debug for AlibabaProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaProvider")
            .field("provider_id", &PROVIDER_ID)
            .field("language_configured", &self.language.is_some())
            .field("messages_configured", &self.messages.is_some())
            .field("embedding_configured", &self.embedding.is_some())
            .field("video_configured", &self.video.is_some())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointSource {
    ProviderOwned,
    CallerControlled,
}

struct ConfiguredEndpoint {
    endpoint: EndpointConfig,
    source: EndpointSource,
}

impl ConfiguredEndpoint {
    fn provider_owned(endpoint: EndpointConfig) -> Self {
        Self {
            endpoint,
            source: EndpointSource::ProviderOwned,
        }
    }

    fn caller_controlled(endpoint: EndpointConfig) -> Self {
        Self {
            endpoint,
            source: EndpointSource::CallerControlled,
        }
    }
}

pub struct AlibabaProviderBuilder {
    credential: AlibabaCredential,
    language_endpoint: Option<Result<ConfiguredEndpoint, EndpointError>>,
    replay_domain: Option<ReplayDomain>,
    messages_endpoint: Option<Result<ConfiguredEndpoint, EndpointError>>,
    messages_replay_domain: Option<ReplayDomain>,
    embedding_endpoint: Option<Result<ConfiguredEndpoint, EndpointError>>,
    video_endpoint: Option<Result<ConfiguredEndpoint, EndpointError>>,
    video_download_policy: AlibabaVideoDownloadPolicy,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: AlibabaChatOptions,
    responses_defaults: AlibabaResponsesOptions,
    messages_defaults: AlibabaMessagesOptions,
    embedding_defaults: AlibabaEmbeddingOptions,
}

impl AlibabaProviderBuilder {
    fn new(credential: AlibabaCredential) -> Self {
        Self {
            credential,
            language_endpoint: None,
            replay_domain: None,
            messages_endpoint: None,
            messages_replay_domain: None,
            embedding_endpoint: None,
            video_endpoint: None,
            video_download_policy: AlibabaVideoDownloadPolicy::default(),
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            chat_defaults: AlibabaChatOptions::default(),
            responses_defaults: AlibabaResponsesOptions::default(),
            messages_defaults: AlibabaMessagesOptions::default(),
            embedding_defaults: AlibabaEmbeddingOptions::default(),
        }
    }

    /// Configure the language endpoint.
    ///
    /// Caller-selected endpoints require a matching [`ReplayDomain::custom`] before `build`.
    /// Endpoint policy and URL spelling never promote this setter to provider-owned support.
    pub fn with_language_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.language_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(endpoint)));
        self
    }

    /// Configure a caller-selected public language endpoint.
    ///
    /// Call [`Self::with_replay_domain`] with an explicit custom audience before `build`.
    pub fn with_language_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.language_endpoint = Some(
            EndpointConfig::public_custom(base_url).map(ConfiguredEndpoint::caller_controlled),
        );
        self
    }

    /// Configure a caller-selected Model Studio workspace language endpoint.
    ///
    /// Call [`Self::with_replay_domain`] with a non-secret custom workspace identity before
    /// `build`. The identity is not derived from the workspace URL.
    pub fn with_language_workspace(mut self, workspace: &AlibabaWorkspaceEndpoint) -> Self {
        self.language_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(
            workspace.language_endpoint(),
        )));
        self
    }

    /// Bind provider-native language history to a caller-declared non-secret replay identity.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    /// Opt into the historical shared Singapore language endpoint.
    pub fn with_legacy_singapore_language(mut self) -> Self {
        self.language_endpoint =
            Some(legacy_singapore_language_endpoint().map(ConfiguredEndpoint::provider_owned));
        self
    }

    /// Configure a caller-controlled Anthropic-compatible Messages endpoint.
    ///
    /// The endpoint base must end before `/v1/messages`. Caller-controlled endpoints require a
    /// matching [`ReplayDomain::custom`] through [`Self::with_messages_replay_domain`].
    pub fn with_messages_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.messages_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(endpoint)));
        self
    }

    /// Configure a caller-controlled public Messages endpoint base.
    pub fn with_messages_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.messages_endpoint = Some(
            EndpointConfig::public_custom(base_url).map(ConfiguredEndpoint::caller_controlled),
        );
        self
    }

    /// Configure the caller-controlled Messages endpoint derived from a workspace origin.
    pub fn with_messages_workspace(mut self, workspace: &AlibabaWorkspaceEndpoint) -> Self {
        self.messages_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(
            workspace.messages_endpoint(),
        )));
        self
    }

    /// Bind Messages replay state to a caller-declared non-secret audience.
    pub fn with_messages_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.messages_replay_domain = Some(replay_domain);
        self
    }

    /// Opt into the provider-owned historical Singapore Messages endpoint.
    pub fn with_legacy_singapore_messages(mut self) -> Self {
        self.messages_endpoint =
            Some(legacy_singapore_messages_endpoint().map(ConfiguredEndpoint::provider_owned));
        self
    }

    /// Configure a caller-controlled embedding endpoint.
    ///
    /// Use [`Self::with_legacy_singapore_embedding`] for provider-owned verified support.
    pub fn with_embedding_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.embedding_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(endpoint)));
        self
    }

    /// Configure a caller-controlled public embedding endpoint.
    pub fn with_embedding_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.embedding_endpoint = Some(
            EndpointConfig::public_custom(base_url).map(ConfiguredEndpoint::caller_controlled),
        );
        self
    }

    /// Configure a caller-controlled workspace embedding endpoint.
    pub fn with_embedding_workspace(mut self, workspace: &AlibabaWorkspaceEndpoint) -> Self {
        self.embedding_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(
            workspace.embedding_endpoint(),
        )));
        self
    }

    /// Opt into the historical shared Singapore native embedding endpoint.
    pub fn with_legacy_singapore_embedding(mut self) -> Self {
        self.embedding_endpoint =
            Some(legacy_singapore_embedding_endpoint().map(ConfiguredEndpoint::provider_owned));
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

    pub fn with_chat_defaults(mut self, defaults: AlibabaChatOptions) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: AlibabaResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn with_messages_defaults(mut self, defaults: AlibabaMessagesOptions) -> Self {
        self.messages_defaults = defaults;
        self
    }

    pub fn with_embedding_defaults(mut self, defaults: AlibabaEmbeddingOptions) -> Self {
        self.embedding_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<AlibabaProvider, AlibabaConfigError> {
        let language_endpoint = self.language_endpoint.transpose()?;
        let messages_endpoint = self.messages_endpoint.transpose()?;
        let embedding_endpoint = self.embedding_endpoint.transpose()?;
        let video_endpoint = self.video_endpoint.transpose()?;
        if language_endpoint.is_none()
            && messages_endpoint.is_none()
            && embedding_endpoint.is_none()
            && video_endpoint.is_none()
        {
            return Err(AlibabaConfigError::MissingEndpoint);
        }
        let language_replay_domain = language_endpoint
            .as_ref()
            .map(|configured| {
                let provider_owned = configured.source == EndpointSource::ProviderOwned;
                let replay_domain = match (self.replay_domain.clone(), provider_owned) {
                    (Some(replay_domain), _) => replay_domain,
                    (None, true) => ReplayDomain::official(ReplayDomainId::new(
                        LEGACY_SINGAPORE_LANGUAGE_REPLAY_DOMAIN,
                    )?),
                    (None, false) => {
                        return Err(AlibabaConfigError::CustomLanguageEndpointRequiresReplayDomain);
                    }
                };
                if replay_domain.audience().is_official() != provider_owned {
                    return Err(AlibabaConfigError::ReplayAudienceMismatch);
                }
                Ok(replay_domain)
            })
            .transpose()?;
        let messages_replay_domain = messages_endpoint
            .as_ref()
            .map(|configured| {
                let provider_owned = configured.source == EndpointSource::ProviderOwned;
                let replay_domain = match (self.messages_replay_domain.clone(), provider_owned) {
                    (Some(replay_domain), _) => replay_domain,
                    (None, true) => ReplayDomain::official(ReplayDomainId::new(
                        LEGACY_SINGAPORE_MESSAGES_REPLAY_DOMAIN,
                    )?),
                    (None, false) => {
                        return Err(AlibabaConfigError::CustomMessagesEndpointRequiresReplayDomain);
                    }
                };
                if replay_domain.audience().is_official() != provider_owned {
                    return Err(AlibabaConfigError::MessagesReplayAudienceMismatch);
                }
                Ok(replay_domain)
            })
            .transpose()?;
        let native_transport_settings = NativeTransportSettings {
            limits: self.limits.clone(),
            retry_policy: self.retry_policy,
            connect_timeout: self.connect_timeout,
            call_timeout: self.call_timeout,
            read_timeout: self.read_timeout,
        };
        self.credential.0.validate_static()?;
        let auth = self.credential.0.into_auth();
        let instance_id = ProviderInstanceId::new();
        let language = language_endpoint
            .zip(language_replay_domain)
            .map(|(configured, replay_domain)| {
                let verified_endpoint = configured.source == EndpointSource::ProviderOwned;
                let endpoint = configured.endpoint;
                let profile = profile(endpoint, verified_endpoint, replay_domain)?;
                let mut builder =
                    OpenAiCompatibleProvider::builder_with_auth(profile, auth.clone())
                        .with_provider_instance(instance_id.clone())
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
                    builder = builder.with_default_option(
                        OpenAiCompatibleApiMode::ChatCompletions,
                        name,
                        value,
                    );
                }
                for (name, value) in option_map(&self.responses_defaults)? {
                    builder = builder.with_default_option(
                        OpenAiCompatibleApiMode::Responses,
                        name,
                        value,
                    );
                }
                Ok::<_, AlibabaConfigError>(builder.build()?)
            })
            .transpose()?;
        let messages = messages_endpoint
            .zip(messages_replay_domain)
            .map(|(configured, replay_domain)| {
                let verified_endpoint = configured.source == EndpointSource::ProviderOwned;
                let profile =
                    messages_profile(configured.endpoint, verified_endpoint, replay_domain)?;
                let mut builder =
                    AnthropicCompatibleProvider::builder_with_auth(profile, auth.clone())
                        .with_provider_instance(instance_id.clone())
                        .with_default_options(self.messages_defaults.to_engine())
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
                Ok::<_, AlibabaConfigError>(builder.build()?)
            })
            .transpose()?;
        let embedding = embedding_endpoint
            .map(|configured| {
                let verified_endpoint = configured.source == EndpointSource::ProviderOwned;
                let endpoint = configured.endpoint;
                let scope = Arc::new(native_scope(
                    &endpoint,
                    EMBEDDING_PROTOCOL_ID,
                    EMBEDDING_API_MODE_ID,
                    verified_endpoint,
                )?);
                let profile = support_profile(&scope, verified_endpoint)?;
                let transport =
                    build_native_transport(endpoint, auth.clone(), &native_transport_settings)?;
                Ok::<_, AlibabaConfigError>((
                    Arc::new(AlibabaEmbeddingRuntime {
                        scope,
                        instance_id: instance_id.clone(),
                        transport,
                        defaults: self.embedding_defaults,
                        replay_safety: ReplaySafety::Never,
                    }),
                    profile,
                ))
            })
            .transpose()?;
        let video = video_endpoint
            .map(|configured| {
                let verified_endpoint = configured.source == EndpointSource::ProviderOwned;
                let endpoint = configured.endpoint;
                let scope = Arc::new(native_scope(
                    &endpoint,
                    VIDEO_PROTOCOL_ID,
                    VIDEO_API_MODE_ID,
                    verified_endpoint,
                )?);
                let transport =
                    build_native_transport(endpoint, auth.clone(), &native_transport_settings)?;
                let mut downloader = ResourceDownloader::builder().with_limits(self.limits.clone());
                if let Some(timeout) = self.connect_timeout {
                    downloader = downloader.with_connect_timeout(timeout);
                }
                if let Some(timeout) = self.call_timeout {
                    downloader = downloader.with_download_timeout(timeout);
                }
                if let Some(timeout) = self.read_timeout {
                    downloader = downloader.with_read_timeout(timeout);
                }
                Ok::<_, AlibabaConfigError>((
                    Arc::new(AlibabaVideoRuntime {
                        scope,
                        transport,
                        downloader: downloader.build()?,
                        download_policy: self.video_download_policy,
                    }),
                    verified_endpoint,
                ))
            })
            .transpose()?;
        let embedding_profile = embedding.as_ref().map(|(_, profile)| profile.clone());
        let embedding = embedding.map(|(runtime, _)| runtime);
        let video_claim = video
            .as_ref()
            .and_then(|(_, verified_endpoint)| verified_endpoint.then_some(()))
            .map(|()| video_support_claim())
            .transpose()?;
        let video = video.map(|(runtime, _)| runtime);
        let mut profiles = Vec::new();
        if let Some(language) = &language {
            profiles.push(language.profile().provider_profile().clone());
        }
        if let Some(messages) = &messages {
            profiles.push(messages.profile().provider_profile().clone());
        }
        if let Some(profile) = embedding_profile {
            profiles.push(profile);
        }
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            ProviderId::new(PROVIDER_ID)?,
            profiles,
            video_claim,
        )?);
        let language_registration = language
            .as_ref()
            .and_then(OpenAiCompatibleProvider::responses_registration);
        let embedding_registration = embedding.as_ref().cloned().map(embedding_registration);
        let default_registration = match (language_registration, embedding_registration) {
            (Some(language), Some(embedding)) => Some(language.merge(embedding)?),
            (Some(registration), None) | (None, Some(registration)) => Some(registration),
            (None, None) => None,
        };
        Ok(AlibabaProvider {
            language,
            messages,
            embedding,
            video,
            default_registration,
            support_manifest,
        })
    }
}

/// Experimental video-job access for [`AlibabaProvider`].
///
/// Import this trait from `siumai_provider_alibaba::experimental` to opt into the unstable media
/// job contract without widening the stable provider surface.
pub trait AlibabaVideoProviderExt {
    fn video(&self, model: impl Into<String>) -> Result<AlibabaVideoModel, ModelLookupError>;
}

impl AlibabaVideoProviderExt for AlibabaProvider {
    fn video(&self, model: impl Into<String>) -> Result<AlibabaVideoModel, ModelLookupError> {
        let runtime = self
            .video
            .as_ref()
            .ok_or_else(|| ModelLookupError::Construction {
                source: Error::new(
                    ErrorKind::Unsupported,
                    "Alibaba video endpoint is not configured",
                ),
            })?;
        let model = ModelId::new(model.into())?;
        Ok(AlibabaVideoModel::new(runtime.clone(), model))
    }
}

/// Experimental Alibaba video configuration for [`AlibabaProviderBuilder`].
pub trait AlibabaVideoProviderBuilderExt: Sized {
    /// Configure a caller-controlled video endpoint.
    fn with_video_endpoint(self, endpoint: EndpointConfig) -> Self;
    /// Configure a caller-controlled public video endpoint.
    fn with_video_base_url(self, base_url: impl AsRef<str>) -> Self;
    /// Configure a caller-controlled workspace video endpoint.
    fn with_video_workspace(self, workspace: &AlibabaWorkspaceEndpoint) -> Self;
    /// Opt into the provider-owned historical Singapore video endpoint.
    fn with_legacy_singapore_video(self) -> Self;
    fn with_video_download_policy(self, policy: AlibabaVideoDownloadPolicy) -> Self;
}

impl AlibabaVideoProviderBuilderExt for AlibabaProviderBuilder {
    fn with_video_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.video_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(endpoint)));
        self
    }

    fn with_video_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.video_endpoint = Some(
            EndpointConfig::public_custom(base_url).map(ConfiguredEndpoint::caller_controlled),
        );
        self
    }

    fn with_video_workspace(mut self, workspace: &AlibabaWorkspaceEndpoint) -> Self {
        self.video_endpoint = Some(Ok(ConfiguredEndpoint::caller_controlled(
            workspace.video_endpoint(),
        )));
        self
    }

    /// Opt into the historical shared Singapore native video endpoint.
    fn with_legacy_singapore_video(mut self) -> Self {
        self.video_endpoint =
            Some(legacy_singapore_video_endpoint().map(ConfiguredEndpoint::provider_owned));
        self
    }

    fn with_video_download_policy(mut self, policy: AlibabaVideoDownloadPolicy) -> Self {
        self.video_download_policy = policy;
        self
    }
}

#[derive(Clone)]
pub struct AlibabaLanguageModel {
    api: AlibabaLanguageApi,
    inner: AlibabaLanguageModelInner,
}

#[derive(Clone)]
enum AlibabaLanguageModelInner {
    OpenAi(OpenAiCompatibleLanguageModel),
    Messages(AnthropicCompatibleLanguageModel),
}

impl AlibabaLanguageModel {
    fn openai(api: AlibabaLanguageApi, inner: OpenAiCompatibleLanguageModel) -> Self {
        Self {
            api,
            inner: AlibabaLanguageModelInner::OpenAi(inner),
        }
    }

    fn messages(inner: AnthropicCompatibleLanguageModel) -> Self {
        Self {
            api: AlibabaLanguageApi::Messages,
            inner: AlibabaLanguageModelInner::Messages(inner),
        }
    }

    pub const fn api(&self) -> AlibabaLanguageApi {
        self.api
    }
}

impl Model for AlibabaLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        match &self.inner {
            AlibabaLanguageModelInner::OpenAi(model) => model.descriptor(),
            AlibabaLanguageModelInner::Messages(model) => model.descriptor(),
        }
    }
}

#[async_trait]
impl LanguageModel for AlibabaLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        match &self.inner {
            AlibabaLanguageModelInner::OpenAi(model) => model.generate(request, options).await,
            AlibabaLanguageModelInner::Messages(model) => model.generate(request, options).await,
        }
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        match &self.inner {
            AlibabaLanguageModelInner::OpenAi(model) => model.stream(request, options).await,
            AlibabaLanguageModelInner::Messages(model) => model.stream(request, options).await,
        }
    }
}

impl fmt::Debug for AlibabaLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaLanguageModel")
            .field("api", &self.api)
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn legacy_singapore_language_endpoint() -> Result<EndpointConfig, EndpointError> {
    legacy_singapore_endpoint(LEGACY_SINGAPORE_LANGUAGE_BASE_URL)
}

fn legacy_singapore_messages_endpoint() -> Result<EndpointConfig, EndpointError> {
    legacy_singapore_endpoint(LEGACY_SINGAPORE_MESSAGES_BASE_URL)
}

fn legacy_singapore_embedding_endpoint() -> Result<EndpointConfig, EndpointError> {
    legacy_singapore_endpoint(LEGACY_SINGAPORE_EMBEDDING_BASE_URL)
}

fn legacy_singapore_video_endpoint() -> Result<EndpointConfig, EndpointError> {
    legacy_singapore_endpoint(LEGACY_SINGAPORE_VIDEO_BASE_URL)
}

fn legacy_singapore_endpoint(base_url: &str) -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(base_url, OfficialOrigin::new(LEGACY_SINGAPORE_ORIGIN)?)
}

fn native_scope(
    endpoint: &EndpointConfig,
    protocol: &str,
    api_mode: &str,
    verified_endpoint: bool,
) -> Result<ProviderScope, InvalidId> {
    let platform = match endpoint.policy() {
        _ if verified_endpoint => "alibaba-model-studio",
        EndpointPolicy::LocalExplicit(_) => "local",
        _ => "custom-endpoint",
    };
    Ok(ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
        .with_platform(PlatformId::new(platform)?)
        .with_protocol(ProtocolId::new(protocol)?)
        .with_api_mode(ApiModeId::new(api_mode)?))
}

fn video_support_claim() -> Result<VerifiedNativeSupportClaim, AlibabaConfigError> {
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            ProviderId::new(PROVIDER_ID)?,
            PlatformId::new(PLATFORM_ID)?,
            NativeSurfaceKind::Job,
            NativeSurfaceId::new("video-tasks")?,
        ),
        VerifiedFidelity::Native,
        ApiStability::Experimental,
        NativeVerificationEvidence::new(
            OfficialSource::new(VIDEO_TEXT_SOURCE)?,
            VerificationDate::new(NaiveDate::parse_from_str(SUPPORT_VERIFIED_ON, "%Y-%m-%d")?),
        ),
    ))
}

fn embedding_registration(runtime: Arc<AlibabaEmbeddingRuntime>) -> ProviderRegistration {
    ProviderRegistration::from_embedding(
        runtime.scope.clone(),
        Arc::new(move |model| {
            Ok(Arc::new(AlibabaEmbeddingModel::new(runtime.clone(), model))
                as Arc<dyn EmbeddingModel>)
        }),
    )
}

struct NativeTransportSettings {
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
}

fn build_native_transport(
    endpoint: EndpointConfig,
    auth: Arc<dyn AuthApplier>,
    settings: &NativeTransportSettings,
) -> Result<ProviderTransport, TransportConfigError> {
    let mut builder = ProviderTransport::builder(endpoint)
        .with_auth(auth)
        .with_limits(settings.limits.clone())
        .with_retry_policy(settings.retry_policy);
    if let Some(timeout) = settings.connect_timeout {
        builder = builder.with_connect_timeout(timeout);
    }
    if let Some(timeout) = settings.call_timeout {
        builder = builder.with_call_timeout(timeout);
    }
    if let Some(timeout) = settings.read_timeout {
        builder = builder.with_read_timeout(timeout);
    }
    builder.build()
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, AlibabaConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(AlibabaConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum AlibabaConfigError {
    #[error(
        "Alibaba provider requires an explicit language, Messages, embedding, or video endpoint; configure a workspace endpoint or opt into a named legacy Singapore endpoint"
    )]
    MissingEndpoint,
    #[error("invalid Alibaba endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Alibaba language profile: {0}")]
    Profile(#[from] AlibabaProfileError),
    #[error("invalid Alibaba embedding profile: {0}")]
    EmbeddingProfile(#[from] AlibabaEmbeddingProfileError),
    #[error("invalid Alibaba identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Alibaba support evidence: {0}")]
    SupportEvidence(#[from] ProfileError),
    #[error("invalid Alibaba support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("invalid Alibaba support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Alibaba default registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("invalid Alibaba credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("invalid Alibaba provider configuration: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("a custom Alibaba language endpoint requires an explicit non-secret replay domain")]
    CustomLanguageEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Alibaba language endpoint")]
    ReplayAudienceMismatch,
    #[error("a custom Alibaba Messages endpoint requires an explicit non-secret replay domain")]
    CustomMessagesEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Alibaba Messages endpoint")]
    MessagesReplayAudienceMismatch,
    #[error("invalid Alibaba Anthropic-compatible Messages runtime: {0}")]
    MessagesCompatible(#[from] AnthropicCompatibleConfigError),
    #[error("invalid Alibaba transport configuration: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("Alibaba default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("Alibaba default options must serialize to an object")]
    InvalidDefaultsShape,
}
