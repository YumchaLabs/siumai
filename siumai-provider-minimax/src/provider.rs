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
    ApiModeId, ApiStability, CallOptions, Error, GenericSupportClaim, ImageModel,
    ImageModelProvider, InvalidId, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelCatalog, ModelDescriptor, ModelFamily, ModelId,
    ModelLookupError, ModelPolicy, NativeSupportScope, NativeSurfaceId, NativeSurfaceKind,
    NativeVerificationEvidence, OfficialSource, PlatformId, ProfileError, ProfileId, ProtocolId,
    Provider, ProviderId, ProviderInstanceId, ProviderOptionError, ProviderProfile,
    ProviderRegistration, ProviderRegistrationError, ProviderScope, ProviderSupportManifest,
    ReplayDomain, ReplayDomainId, SpeechModel, SpeechModelProvider, SupportManifestError,
    SupportScope, TypedProviderOptions, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedNativeSupportClaim, VerifiedSupportClaim,
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
use crate::portable::{
    MinimaxImageModel, MinimaxImagePolicy, MinimaxSpeechModel, MinimaxSpeechPolicy,
};
use crate::resources::{
    MinimaxFiles, MinimaxImages, MinimaxMusic, MinimaxResponses, MinimaxSpeech, MinimaxVideo,
    MinimaxVoices, NativeRuntime, VOICE_CLONE_API_SOURCE, VOICE_DELETE_API_SOURCE,
    VOICE_DESIGN_API_SOURCE, VOICE_LIST_API_SOURCE,
};

const RESOURCE_BASE_URL: &str = "https://api.minimax.io/";
const RESOURCE_VERIFIED_ON: &str = "2026-08-08";
const FILES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/file-management-upload";
const IMAGES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/image-generation-t2i";
const VIDEO_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/video-generation-v2-create";
const MUSIC_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/music-generation";
const SPEECH_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/speech-t2a-http";
const ASYNC_SPEECH_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/speech-t2a-async-create";
const RESPONSES_INPUT_TOKENS_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/responses-input-tokens";
const RESPONSES_INPUT_TOKENS_VERIFIED_ON: &str = "2026-08-09";
const IMAGE_PROTOCOL_ID: &str = "minimax-image";
const IMAGE_API_MODE_ID: &str = "image-generation";
const SPEECH_PROTOCOL_ID: &str = "minimax-speech";
const SPEECH_API_MODE_ID: &str = "speech-http";

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
    responses_native: Arc<NativeRuntime>,
    responses_scope: Arc<ProviderScope>,
    image_scope: Arc<ProviderScope>,
    image_policy: Arc<dyn ModelPolicy>,
    speech_scope: Arc<ProviderScope>,
    speech_policy: Arc<dyn ModelPolicy>,
    registration: ProviderRegistration,
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

    /// Provider-native Responses resource operations, including exact input-token counting.
    pub fn responses_resource(&self) -> MinimaxResponses {
        MinimaxResponses::new(self.responses_native.clone(), self.responses_scope.clone())
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

    /// Recommended composite registration: Messages language plus portable image and speech.
    pub fn registration(&self) -> ProviderRegistration {
        self.registration.clone()
    }

    /// Composite registration using Messages for language plus portable image and speech.
    pub fn messages_registration(&self) -> ProviderRegistration {
        self.messages_registration.clone()
    }

    /// Composite registration using Chat Completions for language plus portable image and speech.
    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    /// Composite registration using Responses for language plus portable image and speech.
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

    /// Create a portable image-generation model over MiniMax's native Image API.
    pub fn image_model(&self, model: ModelId) -> Result<MinimaxImageModel, ModelLookupError> {
        Ok(MinimaxImageModel::new(
            self.native.clone(),
            ModelDescriptor::from_scope(
                self.image_scope.clone(),
                model,
                ModelFamily::Image,
                self.native.instance_id.clone(),
            ),
            self.image_policy.clone(),
        ))
    }

    /// Create a portable image-generation model from an open provider model identifier.
    pub fn image(&self, model: impl Into<String>) -> Result<MinimaxImageModel, ModelLookupError> {
        self.image_model(ModelId::new(model.into())?)
    }

    /// Provider-native non-streaming music generation operations.
    pub fn music(&self) -> MinimaxMusic {
        MinimaxMusic::new(self.native.clone())
    }

    /// Provider-native buffered and asynchronous speech operations.
    pub fn speech(&self) -> MinimaxSpeech {
        MinimaxSpeech::new(self.native.clone())
    }

    /// Provider-native voice clone, design, list, and delete operations.
    pub fn voices(&self) -> MinimaxVoices {
        MinimaxVoices::new(self.native.clone())
    }

    /// Create a portable buffered speech model over MiniMax's synchronous Speech API.
    pub fn speech_model(&self, model: ModelId) -> Result<MinimaxSpeechModel, ModelLookupError> {
        Ok(MinimaxSpeechModel::new(
            self.native.clone(),
            ModelDescriptor::from_scope(
                self.speech_scope.clone(),
                model,
                ModelFamily::Speech,
                self.native.instance_id.clone(),
            ),
            self.speech_policy.clone(),
        ))
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

impl ImageModelProvider for MinimaxProvider {
    type Model = MinimaxImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        MinimaxProvider::image_model(self, model)
    }
}

impl SpeechModelProvider for MinimaxProvider {
    type Model = MinimaxSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        MinimaxProvider::speech_model(self, model)
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

enum EndpointSelection {
    ProviderOwned(EndpointConfig),
    CallerControlled(EndpointConfig),
}

impl EndpointSelection {
    fn into_parts(self) -> (EndpointConfig, bool) {
        match self {
            Self::ProviderOwned(endpoint) => (endpoint, true),
            Self::CallerControlled(endpoint) => (endpoint, false),
        }
    }
}

/// Stable official replay audience shared by MiniMax language protocols.
pub const MINIMAX_REPLAY_AUDIENCE: &str = "minimax-public-api";

/// Builder for one immutable MiniMax provider runtime.
pub struct MinimaxProviderBuilder {
    credential: MinimaxCredential,
    messages_endpoint: Result<EndpointSelection, EndpointError>,
    openai_endpoint: Result<EndpointSelection, EndpointError>,
    resource_endpoint: Result<EndpointSelection, EndpointError>,
    messages_replay_domain: Option<ReplayDomain>,
    openai_replay_domain: Option<ReplayDomain>,
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
            messages_endpoint: official_endpoint(MESSAGES_BASE_URL)
                .map(EndpointSelection::ProviderOwned),
            openai_endpoint: official_endpoint(OPENAI_BASE_URL)
                .map(EndpointSelection::ProviderOwned),
            resource_endpoint: official_endpoint(RESOURCE_BASE_URL)
                .map(EndpointSelection::ProviderOwned),
            messages_replay_domain: None,
            openai_replay_domain: None,
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

    /// Replace the provider-owned Messages endpoint with a caller-controlled endpoint.
    ///
    /// The endpoint remains caller-controlled even when its transport policy is marked official.
    /// A matching [`ReplayDomain::custom`] is required before `build`.
    pub fn with_messages_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.messages_endpoint = Ok(EndpointSelection::CallerControlled(endpoint));
        self
    }

    /// Replace the provider-owned OpenAI-compatible endpoint with a caller-controlled endpoint.
    ///
    /// The endpoint remains caller-controlled even when its transport policy is marked official.
    /// A matching [`ReplayDomain::custom`] is required before `build`.
    pub fn with_openai_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.openai_endpoint = Ok(EndpointSelection::CallerControlled(endpoint));
        self
    }

    /// Replace the provider-owned native-resource endpoint with a caller-controlled endpoint.
    ///
    /// Caller-controlled endpoints never inherit MiniMax native support claims.
    pub fn with_resource_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.resource_endpoint = Ok(EndpointSelection::CallerControlled(endpoint));
        self
    }

    pub fn with_messages_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.messages_endpoint =
            EndpointConfig::public_custom(base_url).map(EndpointSelection::CallerControlled);
        self
    }

    pub fn with_openai_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.openai_endpoint =
            EndpointConfig::public_custom(base_url).map(EndpointSelection::CallerControlled);
        self
    }

    pub fn with_resource_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.resource_endpoint =
            EndpointConfig::public_custom(base_url).map(EndpointSelection::CallerControlled);
        self
    }

    /// Bind Messages history to a non-secret audience and caller scope.
    pub fn with_messages_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.messages_replay_domain = Some(replay_domain);
        self
    }

    /// Bind OpenAI-compatible history to a non-secret audience and caller scope.
    ///
    /// Chat Completions and Responses share this endpoint audience while remaining isolated by
    /// their protocol and API-mode identities.
    pub fn with_openai_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.openai_replay_domain = Some(replay_domain);
        self
    }

    /// Bind both language endpoints to the same non-secret replay domain.
    ///
    /// Use the mode-specific methods when Messages and OpenAI-compatible traffic use different
    /// endpoint audiences.
    pub fn with_language_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.messages_replay_domain = Some(replay_domain.clone());
        self.openai_replay_domain = Some(replay_domain);
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

        let (messages_endpoint, messages_is_verified) = self.messages_endpoint?.into_parts();
        let (openai_endpoint, openai_is_verified) = self.openai_endpoint?.into_parts();
        let (resource_endpoint, resource_is_verified) = self.resource_endpoint?.into_parts();
        let messages_replay_domain = resolve_replay_domain(
            self.messages_replay_domain,
            messages_is_verified,
            MinimaxConfigError::CustomMessagesEndpointRequiresReplayDomain,
            MinimaxConfigError::MessagesReplayAudienceMismatch,
        )?;
        let openai_replay_domain = resolve_replay_domain(
            self.openai_replay_domain,
            openai_is_verified,
            MinimaxConfigError::CustomOpenAiEndpointRequiresReplayDomain,
            MinimaxConfigError::OpenAiReplayAudienceMismatch,
        )?;
        if self.credential.is_unauthenticated()
            && [&messages_endpoint, &openai_endpoint, &resource_endpoint]
                .into_iter()
                .any(|endpoint| matches!(endpoint.policy(), EndpointPolicy::Official(_)))
        {
            return Err(MinimaxConfigError::OfficialEndpointRequiresAuthentication);
        }
        let auth = self.credential.into_auth()?;
        let messages_profile = messages_profile(
            messages_endpoint,
            messages_replay_domain,
            messages_is_verified,
        )?;
        let responses_endpoint = openai_endpoint.clone();
        let responses_scope = Arc::new(
            ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
                .with_platform(PlatformId::new("minimax-api")?)
                .with_protocol(ProtocolId::new(
                    siumai_protocol_openai::responses::OPENAI_RESPONSES_PROTOCOL,
                )?)
                .with_api_mode(ApiModeId::new(
                    siumai_protocol_openai::responses::API_MODE_ID,
                )?)
                .with_replay_domain(openai_replay_domain.clone()),
        );
        let openai_profile =
            openai_profile(openai_endpoint, openai_replay_domain, openai_is_verified)?;
        let image_scope = Arc::new(media_scope(IMAGE_PROTOCOL_ID, IMAGE_API_MODE_ID)?);
        let speech_scope = Arc::new(media_scope(SPEECH_PROTOCOL_ID, SPEECH_API_MODE_ID)?);
        let image_policy: Arc<dyn ModelPolicy> = Arc::new(MinimaxImagePolicy {
            verified_endpoint: resource_is_verified,
        });
        let speech_policy: Arc<dyn ModelPolicy> = Arc::new(MinimaxSpeechPolicy {
            verified_endpoint: resource_is_verified,
        });
        let image_profile = media_support_profile(
            resource_is_verified,
            ModelFamily::Image,
            IMAGE_PROTOCOL_ID,
            IMAGE_API_MODE_ID,
            IMAGES_SOURCE,
            "minimax-images-2026-08",
        )?;
        let speech_profile = media_support_profile(
            resource_is_verified,
            ModelFamily::Speech,
            SPEECH_PROTOCOL_ID,
            SPEECH_API_MODE_ID,
            SPEECH_SOURCE,
            "minimax-speech-http-2026-08",
        )?;
        let native_claims = native_support_claims(resource_is_verified, openai_is_verified)?;
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [
                messages_profile.provider_profile().clone(),
                openai_profile.provider_profile().clone(),
                image_profile,
                speech_profile,
            ],
            native_claims,
        )?);

        let instance_id = ProviderInstanceId::new();
        let mut messages_builder =
            AnthropicCompatibleProvider::builder_with_auth(messages_profile, auth.clone())
                .with_provider_instance(instance_id.clone())
                .with_default_options(self.messages_defaults.to_engine())
                .with_limits(self.limits.clone())
                .with_retry_policy(self.retry_policy);

        let mut openai_builder =
            OpenAiCompatibleProvider::builder_with_auth(openai_profile, auth.clone())
                .with_provider_instance(instance_id.clone())
                .with_limits(self.limits.clone())
                .with_retry_policy(self.retry_policy);
        let mut responses_resource_builder = ProviderTransport::builder(responses_endpoint)
            .with_auth(auth.clone())
            .with_limits(self.limits.clone())
            .with_retry_policy(self.retry_policy);
        let mut resource_builder = ProviderTransport::builder(resource_endpoint)
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);

        if let Some(timeout) = self.connect_timeout {
            messages_builder = messages_builder.with_connect_timeout(timeout);
            openai_builder = openai_builder.with_connect_timeout(timeout);
            responses_resource_builder = responses_resource_builder.with_connect_timeout(timeout);
            resource_builder = resource_builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            messages_builder = messages_builder.with_call_timeout(timeout);
            openai_builder = openai_builder.with_call_timeout(timeout);
            responses_resource_builder = responses_resource_builder.with_call_timeout(timeout);
            resource_builder = resource_builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            messages_builder = messages_builder.with_read_timeout(timeout);
            openai_builder = openai_builder.with_read_timeout(timeout);
            responses_resource_builder = responses_resource_builder.with_read_timeout(timeout);
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
        let responses_native = Arc::new(NativeRuntime::new(
            instance_id.clone(),
            responses_resource_builder.build()?,
        ));
        let native = Arc::new(NativeRuntime::new(instance_id, resource_builder.build()?));
        let image_registration = ProviderRegistration::from_image(
            image_scope.clone(),
            image_policy.clone(),
            Arc::new({
                let native = native.clone();
                let image_scope = image_scope.clone();
                let image_policy = image_policy.clone();
                move |model| {
                    Ok(Arc::new(MinimaxImageModel::new(
                        native.clone(),
                        ModelDescriptor::from_scope(
                            image_scope.clone(),
                            model,
                            ModelFamily::Image,
                            native.instance_id.clone(),
                        ),
                        image_policy.clone(),
                    )) as Arc<dyn ImageModel>)
                }
            }),
        );
        let speech_registration = ProviderRegistration::from_speech(
            speech_scope.clone(),
            speech_policy.clone(),
            Arc::new({
                let native = native.clone();
                let speech_scope = speech_scope.clone();
                let speech_policy = speech_policy.clone();
                move |model| {
                    Ok(Arc::new(MinimaxSpeechModel::new(
                        native.clone(),
                        ModelDescriptor::from_scope(
                            speech_scope.clone(),
                            model,
                            ModelFamily::Speech,
                            native.instance_id.clone(),
                        ),
                        speech_policy.clone(),
                    )) as Arc<dyn SpeechModel>)
                }
            }),
        );
        let media_registration = image_registration.merge(speech_registration)?;
        let messages_registration = messages.registration().merge(media_registration.clone())?;
        let registration = messages_registration.clone();
        let chat_registration = openai
            .chat_completions_registration()
            .ok_or(MinimaxConfigError::MissingChatMode)?
            .merge(media_registration.clone())?;
        let responses_registration = openai
            .responses_registration()
            .ok_or(MinimaxConfigError::MissingResponsesMode)?
            .merge(media_registration)?;

        Ok(MinimaxProvider {
            messages,
            openai,
            native,
            responses_native,
            responses_scope,
            image_scope,
            image_policy,
            speech_scope,
            speech_policy,
            registration,
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

fn media_scope(protocol: &str, api_mode: &str) -> Result<ProviderScope, InvalidId> {
    Ok(ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
        .with_platform(PlatformId::new("minimax-api")?)
        .with_protocol(ProtocolId::new(protocol)?)
        .with_api_mode(ApiModeId::new(api_mode)?))
}

fn media_support_profile(
    verified_endpoint: bool,
    family: ModelFamily,
    protocol: &str,
    api_mode: &str,
    source: &str,
    contract: &str,
) -> Result<ProviderProfile, MinimaxConfigError> {
    let scope = SupportScope::new(
        ProviderId::new(PROVIDER_ID)?,
        PlatformId::new("minimax-api")?,
        family,
        ProtocolId::new(protocol)?,
        ApiModeId::new(api_mode)?,
    );
    let profile_id = ProfileId::new(format!("minimax-{api_mode}"))?;
    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(scope, ApiStability::Experimental),
        ));
    }
    Ok(ProviderProfile::verified(
        profile_id,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            VerificationEvidence::new(
                OfficialSource::new(source)?,
                resource_verification_date(),
                siumai_core::ProtocolContractId::new(contract)?,
            ),
        )],
        ModelCatalog::default(),
    )?)
}

fn resolve_replay_domain(
    configured: Option<ReplayDomain>,
    provider_selected_endpoint: bool,
    missing: MinimaxConfigError,
    mismatch: MinimaxConfigError,
) -> Result<ReplayDomain, MinimaxConfigError> {
    let replay_domain = match (configured, provider_selected_endpoint) {
        (Some(replay_domain), _) => replay_domain,
        (None, true) => ReplayDomain::official(ReplayDomainId::new(MINIMAX_REPLAY_AUDIENCE)?),
        (None, false) => return Err(missing),
    };
    let audience_matches = if provider_selected_endpoint {
        replay_domain.audience().is_official()
            && replay_domain.audience().id().as_str() == MINIMAX_REPLAY_AUDIENCE
    } else {
        !replay_domain.audience().is_official()
    };
    if !audience_matches {
        return Err(mismatch);
    }
    Ok(replay_domain)
}

fn native_support_claims(
    resource_is_verified: bool,
    responses_is_verified: bool,
) -> Result<Vec<VerifiedNativeSupportClaim>, MinimaxConfigError> {
    let provider = siumai_core::ProviderId::new(PROVIDER_ID)?;
    let platform = siumai_core::PlatformId::new("minimax-api")?;
    let resource_verified_at = resource_verification_date();
    let current_surface_verified_at = verification_date(RESPONSES_INPUT_TOKENS_VERIFIED_ON);
    let mut claims = Vec::new();
    if resource_is_verified {
        for (surface, kind, stability, source, verified_at) in [
            (
                "files",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                FILES_SOURCE,
                resource_verified_at,
            ),
            (
                "images",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                IMAGES_SOURCE,
                resource_verified_at,
            ),
            (
                "video-tasks",
                NativeSurfaceKind::Job,
                ApiStability::Experimental,
                VIDEO_SOURCE,
                resource_verified_at,
            ),
            (
                "music",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                MUSIC_SOURCE,
                resource_verified_at,
            ),
            (
                "speech-http",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                SPEECH_SOURCE,
                resource_verified_at,
            ),
            (
                "speech-async-tasks",
                NativeSurfaceKind::Job,
                ApiStability::Experimental,
                ASYNC_SPEECH_SOURCE,
                resource_verified_at,
            ),
            (
                "voice-cloning",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                VOICE_CLONE_API_SOURCE,
                current_surface_verified_at,
            ),
            (
                "voice-design",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                VOICE_DESIGN_API_SOURCE,
                current_surface_verified_at,
            ),
            (
                "voice-management",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                VOICE_LIST_API_SOURCE,
                current_surface_verified_at,
            ),
            (
                "voice-delete",
                NativeSurfaceKind::Resource,
                ApiStability::Stable,
                VOICE_DELETE_API_SOURCE,
                current_surface_verified_at,
            ),
        ] {
            claims.push(VerifiedNativeSupportClaim::new(
                NativeSupportScope::surface(
                    provider.clone(),
                    platform.clone(),
                    kind,
                    NativeSurfaceId::new(surface)?,
                ),
                VerifiedFidelity::Native,
                stability,
                NativeVerificationEvidence::new(OfficialSource::new(source)?, verified_at),
            ));
        }
    }
    if responses_is_verified {
        claims.push(VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider,
                platform,
                NativeSurfaceKind::Resource,
                NativeSurfaceId::new("responses-input-tokens")?,
            ),
            VerifiedFidelity::Native,
            ApiStability::Stable,
            NativeVerificationEvidence::new(
                OfficialSource::new(RESPONSES_INPUT_TOKENS_SOURCE)?,
                current_surface_verified_at,
            ),
        ));
    }
    Ok(claims)
}

fn resource_verification_date() -> VerificationDate {
    verification_date(RESOURCE_VERIFIED_ON)
}

fn verification_date(value: &str) -> VerificationDate {
    VerificationDate::new(
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .expect("MiniMax resource verification date is valid"),
    )
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
    #[error("invalid MiniMax provider registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
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
    #[error(
        "a caller-controlled MiniMax Messages endpoint requires an explicit custom replay domain"
    )]
    CustomMessagesEndpointRequiresReplayDomain,
    #[error(
        "a caller-controlled MiniMax OpenAI endpoint requires an explicit custom replay domain"
    )]
    CustomOpenAiEndpointRequiresReplayDomain,
    #[error("the MiniMax Messages replay audience does not match endpoint ownership")]
    MessagesReplayAudienceMismatch,
    #[error("the MiniMax OpenAI replay audience does not match endpoint ownership")]
    OpenAiReplayAudienceMismatch,
}
