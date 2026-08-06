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
    ApiStability, CallOptions, Error, InvalidId, LanguageModel, LanguageModelProvider,
    LanguageRequest, LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelId,
    ModelLookupError, NativeSupportScope, NativeSurfaceId, NativeSurfaceKind,
    NativeVerificationEvidence, OfficialSource, ProfileError, Provider, ProviderOptionError,
    ProviderRegistration, ProviderSupportManifest, SupportManifestError, TypedProviderOptions,
    VerificationDate, VerifiedFidelity, VerifiedNativeSupportClaim,
};
use siumai_openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleConfigError, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProvider,
};
use siumai_transport::{
    EndpointConfig, EndpointError, EndpointPolicy, OfficialOrigin, ProviderTransport, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::credential::{MinimaxCredential, MinimaxCredentialError};
use crate::language::{
    MESSAGES_BASE_URL, MinimaxLanguageProfileError, OFFICIAL_ORIGIN, OPENAI_BASE_URL, PROVIDER_ID,
    messages_profile, openai_profile,
};
use crate::options::{
    MinimaxChatCompletionsOptions, MinimaxMessagesOptions, MinimaxResponsesOptions,
};
use crate::resources::{
    MinimaxFiles, MinimaxImages, MinimaxMusic, MinimaxSpeech, MinimaxVideo, NativeRuntime,
};

const RESOURCE_BASE_URL: &str = "https://api.minimax.io/";
const RESOURCE_VERIFIED_ON: &str = "2026-08-06";
const FILES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/file-management-upload";
const IMAGES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/image-generation-t2i";
const VIDEO_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/video-generation-v2-create";
const MUSIC_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/music-generation";
const SPEECH_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/speech-t2a-http";
const ASYNC_SPEECH_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/speech-t2a-async-create";

/// Public MiniMax language API selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxLanguageApi {
    /// Officially recommended Anthropic Messages-compatible endpoint.
    #[default]
    Messages,
    /// OpenAI Chat Completions-compatible endpoint.
    ChatCompletions,
    /// Bounded OpenAI Responses-compatible subset.
    Responses,
}

/// Long-lived, model-independent MiniMax provider.
#[derive(Clone)]
pub struct MinimaxProvider {
    messages: AnthropicCompatibleProvider,
    openai: OpenAiCompatibleProvider,
    native: Arc<NativeRuntime>,
    messages_registration: ProviderRegistration,
    chat_registration: ProviderRegistration,
    responses_registration: ProviderRegistration,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl MinimaxProvider {
    pub fn builder(credential: MinimaxCredential) -> MinimaxProviderBuilder {
        MinimaxProviderBuilder::new(credential)
    }

    /// Construct the recommended Messages model handle.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<MinimaxLanguageModel, ModelLookupError> {
        self.messages(model)
    }

    pub fn messages(
        &self,
        model: impl Into<String>,
    ) -> Result<MinimaxLanguageModel, ModelLookupError> {
        self.messages
            .language(model)
            .map(|inner| MinimaxLanguageModel {
                api: MinimaxLanguageApi::Messages,
                inner: MinimaxLanguageModelInner::Messages(inner),
            })
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<MinimaxLanguageModel, ModelLookupError> {
        self.openai
            .chat_completions(model)
            .map(|inner| MinimaxLanguageModel {
                api: MinimaxLanguageApi::ChatCompletions,
                inner: MinimaxLanguageModelInner::OpenAi(inner),
            })
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<MinimaxLanguageModel, ModelLookupError> {
        self.openai
            .responses(model)
            .map(|inner| MinimaxLanguageModel {
                api: MinimaxLanguageApi::Responses,
                inner: MinimaxLanguageModelInner::OpenAi(inner),
            })
    }

    pub fn language_for(
        &self,
        api: MinimaxLanguageApi,
        model: impl Into<String>,
    ) -> Result<MinimaxLanguageModel, ModelLookupError> {
        let model = model.into();
        match api {
            MinimaxLanguageApi::Messages => self.messages(model),
            MinimaxLanguageApi::ChatCompletions => self.chat_completions(model),
            MinimaxLanguageApi::Responses => self.responses(model),
        }
    }

    /// Registration for the recommended Messages mode.
    pub fn registration(&self) -> ProviderRegistration {
        self.messages_registration.clone()
    }

    pub fn messages_registration(&self) -> ProviderRegistration {
        self.messages_registration.clone()
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn registration_for(&self, api: MinimaxLanguageApi) -> ProviderRegistration {
        match api {
            MinimaxLanguageApi::Messages => self.messages_registration(),
            MinimaxLanguageApi::ChatCompletions => self.chat_completions_registration(),
            MinimaxLanguageApi::Responses => self.responses_registration(),
        }
    }

    /// Inspect the exact portable and provider-native scopes configured on this provider.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        self.support_manifest.as_ref()
    }

    /// Provider-native file lifecycle operations.
    pub fn files(&self) -> MinimaxFiles {
        MinimaxFiles::new(self.native.clone())
    }

    /// Provider-native MiniMax H3 V2 task operations.
    pub fn video(&self) -> MinimaxVideo {
        MinimaxVideo::new(self.native.clone())
    }

    /// Provider-native image generation operations.
    pub fn images(&self) -> MinimaxImages {
        MinimaxImages::new(self.native.clone())
    }

    /// Provider-native non-streaming music generation operations.
    pub fn music(&self) -> MinimaxMusic {
        MinimaxMusic::new(self.native.clone())
    }

    /// Provider-native buffered and asynchronous speech operations.
    pub fn speech(&self) -> MinimaxSpeech {
        MinimaxSpeech::new(self.native.clone())
    }
}

impl Provider for MinimaxProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for MinimaxProvider {
    type Model = MinimaxLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.messages(model.to_string())
    }
}

impl fmt::Debug for MinimaxProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxProvider")
            .field("provider_id", &PROVIDER_ID)
            .field("messages_runtime", &"shared")
            .field("openai_runtime", &"shared")
            .field("resource_runtime", &"shared")
            .finish()
    }
}

/// Builder for one immutable MiniMax provider runtime.
pub struct MinimaxProviderBuilder {
    credential: MinimaxCredential,
    messages_endpoint: Result<EndpointConfig, EndpointError>,
    openai_endpoint: Result<EndpointConfig, EndpointError>,
    resource_endpoint: Result<EndpointConfig, EndpointError>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    messages_defaults: MinimaxMessagesOptions,
    chat_defaults: MinimaxChatCompletionsOptions,
    responses_defaults: MinimaxResponsesOptions,
}

impl MinimaxProviderBuilder {
    fn new(credential: MinimaxCredential) -> Self {
        Self {
            credential,
            messages_endpoint: official_endpoint(MESSAGES_BASE_URL),
            openai_endpoint: official_endpoint(OPENAI_BASE_URL),
            resource_endpoint: official_endpoint(RESOURCE_BASE_URL),
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            messages_defaults: MinimaxMessagesOptions::default(),
            chat_defaults: MinimaxChatCompletionsOptions::default(),
            responses_defaults: MinimaxResponsesOptions::default(),
        }
    }

    pub fn with_messages_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.messages_endpoint = Ok(endpoint);
        self
    }

    pub fn with_openai_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.openai_endpoint = Ok(endpoint);
        self
    }

    pub fn with_resource_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.resource_endpoint = Ok(endpoint);
        self
    }

    pub fn with_messages_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.messages_endpoint = EndpointConfig::public_custom(base_url);
        self
    }

    pub fn with_openai_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.openai_endpoint = EndpointConfig::public_custom(base_url);
        self
    }

    pub fn with_resource_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.resource_endpoint = EndpointConfig::public_custom(base_url);
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

    pub fn with_messages_defaults(mut self, defaults: MinimaxMessagesOptions) -> Self {
        self.messages_defaults = defaults;
        self
    }

    pub fn with_chat_completions_defaults(
        mut self,
        defaults: MinimaxChatCompletionsOptions,
    ) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: MinimaxResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<MinimaxProvider, MinimaxConfigError> {
        self.messages_defaults.validate()?;
        self.chat_defaults.validate()?;
        self.responses_defaults.validate()?;

        let messages_endpoint = self.messages_endpoint?;
        let openai_endpoint = self.openai_endpoint?;
        let resource_endpoint = self.resource_endpoint?;
        if self.credential.is_unauthenticated()
            && [&messages_endpoint, &openai_endpoint, &resource_endpoint]
                .into_iter()
                .any(|endpoint| matches!(endpoint.policy(), EndpointPolicy::Official(_)))
        {
            return Err(MinimaxConfigError::OfficialEndpointRequiresAuthentication);
        }
        let auth = self.credential.into_auth()?;
        let messages_profile = messages_profile(messages_endpoint)?;
        let openai_profile = openai_profile(openai_endpoint)?;
        let native_claims = if matches!(resource_endpoint.policy(), EndpointPolicy::Official(_)) {
            native_support_claims()?
        } else {
            Vec::new()
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [
                messages_profile.provider_profile().clone(),
                openai_profile.provider_profile().clone(),
            ],
            native_claims,
        )?);

        let mut messages_builder =
            AnthropicCompatibleProvider::builder_with_auth(messages_profile, auth.clone())
                .with_default_options(self.messages_defaults.to_engine())
                .with_limits(self.limits.clone())
                .with_retry_policy(self.retry_policy);

        let mut openai_builder =
            OpenAiCompatibleProvider::builder_with_auth(openai_profile, auth.clone())
                .with_limits(self.limits.clone())
                .with_retry_policy(self.retry_policy);
        let mut resource_builder = ProviderTransport::builder(resource_endpoint)
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);

        if let Some(timeout) = self.connect_timeout {
            messages_builder = messages_builder.with_connect_timeout(timeout);
            openai_builder = openai_builder.with_connect_timeout(timeout);
            resource_builder = resource_builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            messages_builder = messages_builder.with_call_timeout(timeout);
            openai_builder = openai_builder.with_call_timeout(timeout);
            resource_builder = resource_builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            messages_builder = messages_builder.with_read_timeout(timeout);
            openai_builder = openai_builder.with_read_timeout(timeout);
            resource_builder = resource_builder.with_read_timeout(timeout);
        }

        for (name, value) in option_map(&self.chat_defaults)? {
            openai_builder = openai_builder.with_default_option(
                OpenAiCompatibleApiMode::ChatCompletions,
                name,
                value,
            );
        }
        for (name, value) in option_map(&self.responses_defaults)? {
            openai_builder =
                openai_builder.with_default_option(OpenAiCompatibleApiMode::Responses, name, value);
        }

        let messages = messages_builder.build()?;
        let openai = openai_builder.build()?;
        let native = Arc::new(NativeRuntime::new(resource_builder.build()?));
        let messages_registration = messages.registration();
        let chat_registration = openai
            .chat_completions_registration()
            .ok_or(MinimaxConfigError::MissingChatMode)?;
        let responses_registration = openai
            .responses_registration()
            .ok_or(MinimaxConfigError::MissingResponsesMode)?;

        Ok(MinimaxProvider {
            messages,
            openai,
            native,
            messages_registration,
            chat_registration,
            responses_registration,
            support_manifest,
        })
    }
}

/// Concrete MiniMax language model backed by one of three typed protocol engines.
#[derive(Clone)]
pub struct MinimaxLanguageModel {
    api: MinimaxLanguageApi,
    inner: MinimaxLanguageModelInner,
}

#[derive(Clone)]
enum MinimaxLanguageModelInner {
    Messages(AnthropicCompatibleLanguageModel),
    OpenAi(OpenAiCompatibleLanguageModel),
}

impl MinimaxLanguageModel {
    pub const fn api(&self) -> MinimaxLanguageApi {
        self.api
    }
}

impl Model for MinimaxLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        match &self.inner {
            MinimaxLanguageModelInner::Messages(model) => model.descriptor(),
            MinimaxLanguageModelInner::OpenAi(model) => model.descriptor(),
        }
    }
}

#[async_trait]
impl LanguageModel for MinimaxLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        match &self.inner {
            MinimaxLanguageModelInner::Messages(model) => model.generate(request, options).await,
            MinimaxLanguageModelInner::OpenAi(model) => model.generate(request, options).await,
        }
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        match &self.inner {
            MinimaxLanguageModelInner::Messages(model) => model.stream(request, options).await,
            MinimaxLanguageModelInner::OpenAi(model) => model.stream(request, options).await,
        }
    }
}

impl fmt::Debug for MinimaxLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxLanguageModel")
            .field("api", &self.api)
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn official_endpoint(base_url: &str) -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(base_url, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn native_support_claims() -> Result<Vec<VerifiedNativeSupportClaim>, MinimaxConfigError> {
    let provider = siumai_core::ProviderId::new(PROVIDER_ID)?;
    let platform = siumai_core::PlatformId::new("minimax-api")?;
    let verified_at =
        VerificationDate::new(NaiveDate::parse_from_str(RESOURCE_VERIFIED_ON, "%Y-%m-%d")?);
    [
        (
            "files",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            FILES_SOURCE,
        ),
        (
            "images",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            IMAGES_SOURCE,
        ),
        (
            "video-tasks",
            NativeSurfaceKind::Job,
            ApiStability::Experimental,
            VIDEO_SOURCE,
        ),
        (
            "music",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            MUSIC_SOURCE,
        ),
        (
            "speech-http",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            SPEECH_SOURCE,
        ),
        (
            "speech-async-tasks",
            NativeSurfaceKind::Job,
            ApiStability::Experimental,
            ASYNC_SPEECH_SOURCE,
        ),
    ]
    .into_iter()
    .map(|(surface, kind, stability, source)| {
        Ok(VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider.clone(),
                platform.clone(),
                kind,
                NativeSurfaceId::new(surface)?,
            ),
            VerifiedFidelity::Native,
            stability,
            NativeVerificationEvidence::new(OfficialSource::new(source)?, verified_at),
        ))
    })
    .collect()
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, MinimaxConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(MinimaxConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum MinimaxConfigError {
    #[error("invalid MiniMax endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid MiniMax credential: {0}")]
    Credential(#[from] MinimaxCredentialError),
    #[error("invalid MiniMax language profile: {0}")]
    Profile(#[from] MinimaxLanguageProfileError),
    #[error("invalid MiniMax support identity: {0}")]
    SupportIdentity(#[from] InvalidId),
    #[error("invalid MiniMax support evidence: {0}")]
    SupportEvidence(#[from] ProfileError),
    #[error("invalid MiniMax support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("invalid MiniMax support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid MiniMax Messages runtime: {0}")]
    Messages(#[from] AnthropicCompatibleConfigError),
    #[error("invalid MiniMax OpenAI-compatible runtime: {0}")]
    OpenAi(#[from] OpenAiCompatibleConfigError),
    #[error("invalid MiniMax resource runtime: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid MiniMax default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("MiniMax default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("MiniMax default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("MiniMax OpenAI profile omitted Chat Completions")]
    MissingChatMode,
    #[error("MiniMax OpenAI profile omitted Responses")]
    MissingResponsesMode,
    #[error("official MiniMax endpoints require authentication")]
    OfficialEndpointRequiresAuthentication,
}
