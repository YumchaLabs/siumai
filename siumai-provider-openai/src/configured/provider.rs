use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use chrono::NaiveDate;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use siumai_core::{
    ApiStability, CallOptions, CatalogError, EmbeddingModel, EmbeddingModelProvider, ImageModel,
    ImageModelProvider, InvalidId, LanguageModel, LanguageModelProvider, ModelFamily, ModelId,
    ModelLookupError, NativeSupportScope, NativeSurfaceId, NativeSurfaceKind,
    NativeVerificationEvidence, OfficialSource, ProfileError, Provider, ProviderOptionContext,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, ProviderRegistration, ProviderScope, ProviderSupportManifest, ReplayDomain,
    ReplayDomainId, SpeechModel, SpeechModelProvider, SupportManifestError, TranscriptionModel,
    TranscriptionModelProvider, TypedProviderOptions, VerificationDate, VerifiedFidelity,
    VerifiedNativeSupportClaim,
};
use siumai_protocol_openai::responses::{FunctionToolEncodingOptions, ResponsesWireDialect};
use siumai_transport::{
    EndpointConfig, EndpointError, OfficialOrigin, ProviderTransport, ReplaySafety, RetryPolicy,
    TransportConfigError, TransportLimits, TransportObserver,
};
#[cfg(feature = "openai-responses-websocket")]
use siumai_transport::{WebSocketEndpoint, WebSocketTransport};
use thiserror::Error;

use super::credential::{OpenAiCredential, OpenAiCredentialError};
use super::embedding::{OpenAiEmbeddingModel, OpenAiEmbeddingOptions};
use super::image::{OpenAiImageModel, OpenAiImageOptions};
use super::mode::OpenAiApiMode;
use super::model::{OpenAiChatCompletionsModel, OpenAiResponsesModel};
use super::options::{OpenAiChatCompletionsOptions, OpenAiResponsesOptions};
use super::policy::OpenAiModelPolicy;
use super::profile::{OpenAiProfile, PROVIDER_ID};
#[cfg(feature = "openai-realtime")]
use super::realtime::{
    OPENAI_REALTIME_TRANSLATION_SOURCE_URL, OPENAI_REALTIME_WEBSOCKET_SOURCE_URL,
    OpenAiRealtimeConfig, OpenAiRealtimeConfigError, OpenAiRealtimeEndpoint,
    OpenAiTranslationConfig,
};
#[cfg(feature = "openai-realtime")]
use super::realtime_resource::OpenAiRealtimeResource;
use super::resources::{OpenAiConversations, OpenAiFiles, OpenAiVectorStores};
use super::responses_resource::OpenAiResponsesResource;
#[cfg(feature = "openai-responses-websocket")]
use super::responses_websocket::{
    OPENAI_RESPONSES_WEBSOCKET_URL, OpenAiResponsesWebSocketConfigError,
    OpenAiResponsesWebSocketRuntime,
};
use super::speech::{OpenAiSpeechModel, OpenAiSpeechOptions};
use super::transcription::{OpenAiTranscriptionModel, OpenAiTranscriptionOptions};

const OFFICIAL_ORIGIN: &str = "https://api.openai.com";
const OFFICIAL_BASE_URL: &str = "https://api.openai.com/v1";
const RESPONSES_RESOURCE_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/responses/methods/create";
const CONVERSATIONS_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/conversations/methods/create";
const FILES_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/files/methods/create";
const VECTOR_STORES_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/vector-stores/methods/create";
const SKILLS_SOURCE: &str =
    "https://developers.openai.com/api/reference/resources/skills/methods/create";
const RESPONSES_SUPPORT_VERIFIED_ON: &str = "2026-08-06";
#[cfg(feature = "openai-responses-websocket")]
const RESPONSES_WEBSOCKET_SOURCE: &str =
    "https://developers.openai.com/api/docs/guides/websocket-mode";
#[cfg(feature = "openai-responses-websocket")]
const RESPONSES_WEBSOCKET_SUPPORT_VERIFIED_ON: &str = "2026-08-09";
#[cfg(feature = "openai-realtime")]
const REALTIME_SUPPORT_VERIFIED_ON: &str = "2026-08-06";
const RESOURCE_SUPPORT_VERIFIED_ON: &str = "2026-08-08";

/// One synchronously configured OpenAI provider with portable families and native resources.
#[derive(Clone)]
pub struct OpenAiProvider {
    pub(crate) runtime: Arc<OpenAiRuntime>,
}

impl OpenAiProvider {
    pub fn builder(credential: OpenAiCredential) -> OpenAiProviderBuilder {
        OpenAiProviderBuilder::new(credential)
    }

    /// Create the recommended Responses model through the unified provider contract.
    pub fn language_model(&self, model: ModelId) -> Result<OpenAiResponsesModel, ModelLookupError> {
        Ok(self.create_responses_model(model))
    }

    /// Create a lightweight Responses model handle.
    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiResponsesModel, ModelLookupError> {
        Ok(self.create_responses_model(parse_model_id(model)?))
    }

    /// Create a lightweight Chat Completions model handle.
    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiChatCompletionsModel, ModelLookupError> {
        Ok(self.create_chat_completions_model(parse_model_id(model)?))
    }

    /// Create a lightweight portable text embedding handle.
    pub fn embedding_model(
        &self,
        model: ModelId,
    ) -> Result<OpenAiEmbeddingModel, ModelLookupError> {
        Ok(self.create_embedding_model(model))
    }

    /// Create a portable text embedding handle from an open model identifier.
    pub fn embedding(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiEmbeddingModel, ModelLookupError> {
        self.embedding_model(parse_model_id(model)?)
    }

    /// Create a lightweight portable image generation handle.
    pub fn image_model(&self, model: ModelId) -> Result<OpenAiImageModel, ModelLookupError> {
        Ok(self.create_image_model(model))
    }

    /// Create a portable image generation handle from an open model identifier.
    pub fn image(&self, model: impl Into<String>) -> Result<OpenAiImageModel, ModelLookupError> {
        self.image_model(parse_model_id(model)?)
    }

    /// Create a lightweight portable buffered speech handle.
    pub fn speech_model(&self, model: ModelId) -> Result<OpenAiSpeechModel, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }

    /// Create a portable buffered speech handle from an open model identifier.
    pub fn speech(&self, model: impl Into<String>) -> Result<OpenAiSpeechModel, ModelLookupError> {
        self.speech_model(parse_model_id(model)?)
    }

    /// Create a lightweight portable final-result transcription handle.
    pub fn transcription_model(
        &self,
        model: ModelId,
    ) -> Result<OpenAiTranscriptionModel, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }

    /// Create a portable final-result transcription handle from an open model identifier.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiTranscriptionModel, ModelLookupError> {
        self.transcription_model(parse_model_id(model)?)
    }

    /// Access stored-response, background-response, and compaction operations.
    pub fn responses_resource(&self) -> OpenAiResponsesResource {
        OpenAiResponsesResource::new(self.runtime.clone())
    }

    /// Access the provider-owned Conversations lifecycle.
    pub fn conversations(&self) -> OpenAiConversations {
        OpenAiConversations::new(self.runtime.clone())
    }

    /// Access the provider-owned Files lifecycle.
    pub fn files(&self) -> OpenAiFiles {
        OpenAiFiles::new(self.runtime.clone())
    }

    /// Access the provider-owned Vector Stores lifecycle.
    pub fn vector_stores(&self) -> OpenAiVectorStores {
        OpenAiVectorStores::new(self.runtime.clone())
    }

    /// Access provider-authenticated Realtime client-secret operations.
    #[cfg(feature = "openai-realtime")]
    pub fn realtime_resource(&self) -> OpenAiRealtimeResource {
        OpenAiRealtimeResource::new(self.runtime.clone())
    }

    /// Create an experimental native Realtime conversation configuration.
    #[cfg(feature = "openai-realtime")]
    pub fn realtime(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiRealtimeConfig, OpenAiRealtimeConfigError> {
        let endpoint = self
            .runtime
            .realtime_endpoint
            .clone()
            .ok_or(OpenAiRealtimeConfigError::ExplicitEndpointRequiredForCustomProvider)?;
        let mut config =
            OpenAiRealtimeConfig::new(self.runtime.realtime_credential.clone(), model, endpoint)
                .with_transport_limits(self.runtime.realtime_limits.clone());
        if let Some(organization) = &self.runtime.realtime_organization {
            config = config.with_organization(organization.clone());
        }
        if let Some(project) = &self.runtime.realtime_project {
            config = config.with_project(project.clone());
        }
        if let Some(timeout) = self.runtime.realtime_connect_timeout {
            config = config.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.runtime.realtime_session_timeout {
            config = config.with_session_timeout(timeout);
        }
        if let Some(timeout) = self.runtime.realtime_io_timeout {
            config = config.with_io_timeout(timeout);
        }
        Ok(config)
    }

    /// Create an experimental native Realtime Translation configuration.
    #[cfg(feature = "openai-realtime")]
    pub fn translation(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiTranslationConfig, OpenAiRealtimeConfigError> {
        let endpoint = self
            .runtime
            .translation_endpoint
            .clone()
            .ok_or(OpenAiRealtimeConfigError::ExplicitEndpointRequiredForCustomProvider)?;
        let mut config =
            OpenAiTranslationConfig::new(self.runtime.realtime_credential.clone(), model, endpoint)
                .with_transport_limits(self.runtime.realtime_limits.clone());
        if let Some(organization) = &self.runtime.realtime_organization {
            config = config.with_organization(organization.clone());
        }
        if let Some(project) = &self.runtime.realtime_project {
            config = config.with_project(project.clone());
        }
        if let Some(timeout) = self.runtime.realtime_connect_timeout {
            config = config.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.runtime.realtime_session_timeout {
            config = config.with_session_timeout(timeout);
        }
        if let Some(timeout) = self.runtime.realtime_io_timeout {
            config = config.with_io_timeout(timeout);
        }
        Ok(config)
    }

    /// Capture the recommended Responses language route and all portable family bindings.
    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        let mut registration = self.responses_registration();
        registration = registration
            .bind_embedding(
                self.runtime.family_scope_arc(ModelFamily::Embedding),
                self.runtime.policy.clone(),
                Arc::new({
                    let provider = provider.clone();
                    move |model| {
                        Ok(Arc::new(provider.create_embedding_model(model))
                            as Arc<dyn EmbeddingModel>)
                    }
                }),
            )
            .expect("OpenAI family scopes share one canonical provider identity");
        registration = registration
            .bind_image(
                self.runtime.family_scope_arc(ModelFamily::Image),
                self.runtime.policy.clone(),
                Arc::new({
                    let provider = provider.clone();
                    move |model| {
                        Ok(Arc::new(provider.create_image_model(model)) as Arc<dyn ImageModel>)
                    }
                }),
            )
            .expect("OpenAI family scopes share one canonical provider identity");
        registration = registration
            .bind_speech(
                self.runtime.family_scope_arc(ModelFamily::Speech),
                self.runtime.policy.clone(),
                Arc::new({
                    let provider = provider.clone();
                    move |model| {
                        Ok(Arc::new(provider.create_speech_model(model)) as Arc<dyn SpeechModel>)
                    }
                }),
            )
            .expect("OpenAI family scopes share one canonical provider identity");
        registration
            .bind_transcription(
                self.runtime.family_scope_arc(ModelFamily::Transcription),
                self.runtime.policy.clone(),
                Arc::new(move |model| {
                    Ok(Arc::new(provider.create_transcription_model(model))
                        as Arc<dyn TranscriptionModel>)
                }),
            )
            .expect("OpenAI family scopes share one canonical provider identity")
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.registration_for(OpenAiApiMode::Responses)
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.registration_for(OpenAiApiMode::ChatCompletions)
    }

    /// Capture a mode-bound narrow registration without inventing a provider ID.
    pub fn registration_for(&self, mode: OpenAiApiMode) -> ProviderRegistration {
        let provider = self.clone();
        let scope = self.runtime.scope_arc(mode);
        ProviderRegistration::from_language(
            scope,
            self.runtime.policy.clone(),
            Arc::new(move |model| match mode {
                OpenAiApiMode::Responses => {
                    Ok(Arc::new(provider.create_responses_model(model)) as Arc<dyn LanguageModel>)
                }
                OpenAiApiMode::ChatCompletions => {
                    Ok(Arc::new(provider.create_chat_completions_model(model))
                        as Arc<dyn LanguageModel>)
                }
            }),
        )
    }

    pub fn profile(&self) -> &OpenAiProfile {
        &self.runtime.profile
    }

    /// Inspect the exact model and provider-native scopes configured on this provider.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        self.runtime.support_manifest.as_ref()
    }

    pub const fn recommended_mode(&self) -> OpenAiApiMode {
        OpenAiApiMode::Responses
    }

    fn create_responses_model(&self, model: ModelId) -> OpenAiResponsesModel {
        OpenAiResponsesModel::new(self.runtime.clone(), model)
    }

    fn create_chat_completions_model(&self, model: ModelId) -> OpenAiChatCompletionsModel {
        OpenAiChatCompletionsModel::new(self.runtime.clone(), model)
    }

    fn create_embedding_model(&self, model: ModelId) -> OpenAiEmbeddingModel {
        OpenAiEmbeddingModel::new(
            self.runtime.clone(),
            self.runtime.family_scope_arc(ModelFamily::Embedding),
            model,
            self.runtime.embedding_options.clone(),
        )
    }

    fn create_image_model(&self, model: ModelId) -> OpenAiImageModel {
        OpenAiImageModel::new(
            self.runtime.clone(),
            self.runtime.family_scope_arc(ModelFamily::Image),
            model,
            self.runtime.image_options.clone(),
        )
    }

    fn create_speech_model(&self, model: ModelId) -> OpenAiSpeechModel {
        OpenAiSpeechModel::new(
            self.runtime.clone(),
            self.runtime.family_scope_arc(ModelFamily::Speech),
            model,
            self.runtime.speech_options.clone(),
        )
    }

    fn create_transcription_model(&self, model: ModelId) -> OpenAiTranscriptionModel {
        OpenAiTranscriptionModel::new(
            self.runtime.clone(),
            self.runtime.family_scope_arc(ModelFamily::Transcription),
            model,
            self.runtime.transcription_options.clone(),
        )
    }
}

impl Provider for OpenAiProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for OpenAiProvider {
    type Model = OpenAiResponsesModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_responses_model(model))
    }
}

impl EmbeddingModelProvider for OpenAiProvider {
    type Model = OpenAiEmbeddingModel;

    fn embedding_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_embedding_model(model))
    }
}

impl ImageModelProvider for OpenAiProvider {
    type Model = OpenAiImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl SpeechModelProvider for OpenAiProvider {
    type Model = OpenAiSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }
}

impl TranscriptionModelProvider for OpenAiProvider {
    type Model = OpenAiTranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }
}

impl fmt::Debug for OpenAiProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiProvider")
            .field("provider_id", self.provider_id())
            .field("recommended_mode", &OpenAiApiMode::Responses)
            .field("transport", &"shared")
            .finish()
    }
}

/// Builder for one immutable, shared OpenAI runtime.
pub struct OpenAiProviderBuilder {
    credential: OpenAiCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    custom_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    organization: Option<String>,
    project: Option<String>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    observer: Option<Arc<dyn TransportObserver>>,
    #[cfg(feature = "openai-realtime")]
    realtime_endpoint: Option<OpenAiRealtimeEndpoint>,
    #[cfg(feature = "openai-realtime")]
    translation_endpoint: Option<OpenAiRealtimeEndpoint>,
    #[cfg(feature = "openai-realtime")]
    realtime_session_timeout: Option<Duration>,
    #[cfg(feature = "openai-realtime")]
    realtime_io_timeout: Option<Duration>,
    #[cfg(feature = "openai-responses-websocket")]
    responses_websocket_endpoint: Option<WebSocketEndpoint>,
    #[cfg(feature = "openai-responses-websocket")]
    responses_websocket_session_timeout: Option<Duration>,
    #[cfg(feature = "openai-responses-websocket")]
    responses_websocket_io_timeout: Option<Duration>,
    #[cfg(feature = "openai-responses-websocket")]
    responses_websocket_turn_timeout: Option<Duration>,
    responses_defaults: OpenAiResponsesOptions,
    chat_completions_defaults: OpenAiChatCompletionsOptions,
    embedding_defaults: OpenAiEmbeddingOptions,
    image_defaults: OpenAiImageOptions,
    speech_defaults: OpenAiSpeechOptions,
    transcription_defaults: OpenAiTranscriptionOptions,
}

impl OpenAiProviderBuilder {
    fn new(credential: OpenAiCredential) -> Self {
        let endpoint = OfficialOrigin::new(OFFICIAL_ORIGIN)
            .and_then(|origin| EndpointConfig::official(OFFICIAL_BASE_URL, origin));
        Self {
            credential,
            endpoint,
            custom_endpoint: false,
            replay_domain: None,
            organization: None,
            project: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            observer: None,
            #[cfg(feature = "openai-realtime")]
            realtime_endpoint: None,
            #[cfg(feature = "openai-realtime")]
            translation_endpoint: None,
            #[cfg(feature = "openai-realtime")]
            realtime_session_timeout: None,
            #[cfg(feature = "openai-realtime")]
            realtime_io_timeout: None,
            #[cfg(feature = "openai-responses-websocket")]
            responses_websocket_endpoint: None,
            #[cfg(feature = "openai-responses-websocket")]
            responses_websocket_session_timeout: None,
            #[cfg(feature = "openai-responses-websocket")]
            responses_websocket_io_timeout: None,
            #[cfg(feature = "openai-responses-websocket")]
            responses_websocket_turn_timeout: None,
            responses_defaults: OpenAiResponsesOptions::default(),
            chat_completions_defaults: OpenAiChatCompletionsOptions::default(),
            embedding_defaults: OpenAiEmbeddingOptions::default(),
            image_defaults: OpenAiImageOptions::default(),
            speech_defaults: OpenAiSpeechOptions::default(),
            transcription_defaults: OpenAiTranscriptionOptions::default(),
        }
    }

    /// Replace the provider-owned endpoint with a caller-controlled endpoint.
    ///
    /// The endpoint's transport policy does not grant OpenAI support claims or an
    /// official replay audience. Callers must also select a custom replay domain.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self.custom_endpoint = true;
        self
    }

    /// Bind provider-native history to a non-secret endpoint and caller scope.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
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

    /// Observe sanitized HTTP transport lifecycle events without exposing request payloads.
    pub fn with_transport_observer(mut self, observer: Arc<dyn TransportObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Configure the credential audience for native Realtime conversations.
    #[cfg(feature = "openai-realtime")]
    pub fn with_realtime_endpoint(mut self, endpoint: OpenAiRealtimeEndpoint) -> Self {
        self.realtime_endpoint = Some(endpoint);
        self
    }

    /// Configure the credential audience for native Realtime Translation.
    #[cfg(feature = "openai-realtime")]
    pub fn with_translation_endpoint(mut self, endpoint: OpenAiRealtimeEndpoint) -> Self {
        self.translation_endpoint = Some(endpoint);
        self
    }

    #[cfg(feature = "openai-realtime")]
    pub fn with_realtime_session_timeout(mut self, timeout: Duration) -> Self {
        self.realtime_session_timeout = Some(timeout);
        self
    }

    #[cfg(feature = "openai-realtime")]
    pub fn with_realtime_io_timeout(mut self, timeout: Duration) -> Self {
        self.realtime_io_timeout = Some(timeout);
        self
    }

    /// Configure the caller-controlled Responses WebSocket endpoint.
    ///
    /// Custom HTTP providers must select this endpoint explicitly. An official
    /// HTTP provider cannot attach a caller-controlled WebSocket endpoint because
    /// that would give the relay OpenAI-owned replay provenance.
    #[cfg(feature = "openai-responses-websocket")]
    pub fn with_responses_websocket_endpoint(mut self, endpoint: WebSocketEndpoint) -> Self {
        self.responses_websocket_endpoint = Some(endpoint);
        self
    }

    /// Configure the maximum lifetime of one Responses WebSocket connection.
    #[cfg(feature = "openai-responses-websocket")]
    pub fn with_responses_websocket_session_timeout(mut self, timeout: Duration) -> Self {
        self.responses_websocket_session_timeout = Some(timeout);
        self
    }

    /// Configure the inactivity timeout for one Responses WebSocket send or receive.
    #[cfg(feature = "openai-responses-websocket")]
    pub fn with_responses_websocket_io_timeout(mut self, timeout: Duration) -> Self {
        self.responses_websocket_io_timeout = Some(timeout);
        self
    }

    /// Configure the maximum duration of one generated or warm-up turn.
    #[cfg(feature = "openai-responses-websocket")]
    pub fn with_responses_websocket_turn_timeout(mut self, timeout: Duration) -> Self {
        self.responses_websocket_turn_timeout = Some(timeout);
        self
    }

    pub fn with_responses_defaults(mut self, defaults: OpenAiResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn with_chat_completions_defaults(
        mut self,
        defaults: OpenAiChatCompletionsOptions,
    ) -> Self {
        self.chat_completions_defaults = defaults;
        self
    }

    pub fn with_embedding_defaults(mut self, defaults: OpenAiEmbeddingOptions) -> Self {
        self.embedding_defaults = defaults;
        self
    }

    pub fn with_image_defaults(mut self, defaults: OpenAiImageOptions) -> Self {
        self.image_defaults = defaults;
        self
    }

    pub fn with_speech_defaults(mut self, defaults: OpenAiSpeechOptions) -> Self {
        self.speech_defaults = defaults;
        self
    }

    pub fn with_transcription_defaults(mut self, defaults: OpenAiTranscriptionOptions) -> Self {
        self.transcription_defaults = defaults;
        self
    }

    /// Validate static configuration and build one shared provider runtime.
    pub fn build(self) -> Result<OpenAiProvider, OpenAiConfigError> {
        self.credential.validate()?;
        self.responses_defaults
            .validate_values()
            .map_err(OpenAiConfigError::InvalidResponsesDefaults)?;
        self.chat_completions_defaults
            .validate_values()
            .map_err(OpenAiConfigError::InvalidChatCompletionsDefaults)?;
        self.embedding_defaults
            .validate()
            .map_err(OpenAiConfigError::InvalidEmbeddingDefaults)?;
        self.image_defaults
            .validate()
            .map_err(OpenAiConfigError::InvalidImageDefaults)?;
        self.speech_defaults
            .validate()
            .map_err(OpenAiConfigError::InvalidSpeechDefaults)?;
        self.transcription_defaults
            .validate()
            .map_err(OpenAiConfigError::InvalidTranscriptionDefaults)?;
        let endpoint = self.endpoint?;
        let provider_verified_endpoint = !self.custom_endpoint;
        // The branded OpenAI provider owns the official Responses wire contract. Relays with
        // abbreviated fields belong in `siumai-openai-compatible`, where the caller selects an
        // explicit compatibility profile and dialect descriptor.
        let responses_wire_dialect = ResponsesWireDialect::openai();
        let replay_domain = match (self.replay_domain.clone(), provider_verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new("official")?),
            (None, false) => return Err(OpenAiConfigError::CustomEndpointRequiresReplayDomain),
        };
        if replay_domain.audience().is_official() != provider_verified_endpoint {
            return Err(OpenAiConfigError::ReplayAudienceMismatch);
        }
        if (self.organization.is_some() || self.project.is_some())
            && replay_domain.caller_scope().is_none()
        {
            return Err(OpenAiConfigError::AccountScopeRequiresReplayCallerScope);
        }
        let profile = if provider_verified_endpoint {
            OpenAiProfile::current()?.with_replay_domain(replay_domain)
        } else {
            OpenAiProfile::custom(replay_domain)?
        };
        if provider_verified_endpoint && self.credential.is_unauthenticated() {
            return Err(OpenAiConfigError::OfficialEndpointRequiresAuthentication);
        }
        #[cfg(feature = "openai-realtime")]
        let realtime_credential = self.credential.clone();
        #[cfg(feature = "openai-realtime")]
        let realtime_organization = self.organization.clone();
        #[cfg(feature = "openai-realtime")]
        let realtime_project = self.project.clone();
        #[cfg(feature = "openai-realtime")]
        let realtime_limits = self.limits.clone();
        #[cfg(feature = "openai-realtime")]
        let realtime_connect_timeout = self.connect_timeout;
        #[cfg(feature = "openai-realtime")]
        let realtime_session_timeout = self.realtime_session_timeout;
        #[cfg(feature = "openai-realtime")]
        let realtime_io_timeout = self.realtime_io_timeout;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_credential = self.credential.clone();
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_organization = self.organization.clone();
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_project = self.project.clone();
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_limits = self.limits.clone();
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_connect_timeout = self.connect_timeout;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_session_timeout = self.responses_websocket_session_timeout;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_io_timeout = self.responses_websocket_io_timeout;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_turn_timeout = self.responses_websocket_turn_timeout;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket_endpoint = match (
            self.responses_websocket_endpoint,
            provider_verified_endpoint,
        ) {
            (Some(_), true) => {
                return Err(
                    OpenAiConfigError::CallerControlledResponsesWebSocketOnOfficialProvider,
                );
            }
            (Some(endpoint), false) => Some(endpoint),
            (None, true) => Some(WebSocketEndpoint::official(
                OPENAI_RESPONSES_WEBSOCKET_URL,
                OfficialOrigin::new(OFFICIAL_ORIGIN)?,
            )?),
            (None, false) => None,
        };
        #[cfg(feature = "openai-realtime")]
        let realtime_endpoint = self
            .realtime_endpoint
            .or_else(|| (!self.custom_endpoint).then(OpenAiRealtimeEndpoint::official));
        #[cfg(feature = "openai-realtime")]
        let translation_endpoint = self
            .translation_endpoint
            .or_else(|| (!self.custom_endpoint).then(OpenAiRealtimeEndpoint::official));
        let mut native_claims = Vec::new();
        if provider_verified_endpoint {
            native_claims.extend([
                native_support_claim(
                    "responses-resource-lifecycle",
                    NativeSurfaceKind::Resource,
                    ApiStability::Stable,
                    RESPONSES_RESOURCE_SOURCE,
                    RESPONSES_SUPPORT_VERIFIED_ON,
                )?,
                native_support_claim(
                    "conversations-basic-items",
                    NativeSurfaceKind::Resource,
                    ApiStability::Stable,
                    CONVERSATIONS_SOURCE,
                    RESOURCE_SUPPORT_VERIFIED_ON,
                )?,
                native_support_claim(
                    "files-basic-lifecycle",
                    NativeSurfaceKind::Resource,
                    ApiStability::Stable,
                    FILES_SOURCE,
                    RESOURCE_SUPPORT_VERIFIED_ON,
                )?,
                native_support_claim(
                    "vector-stores-basic-files",
                    NativeSurfaceKind::Resource,
                    ApiStability::Stable,
                    VECTOR_STORES_SOURCE,
                    RESOURCE_SUPPORT_VERIFIED_ON,
                )?,
                native_support_claim(
                    "skills-directory-lifecycle",
                    NativeSurfaceKind::Resource,
                    ApiStability::Experimental,
                    SKILLS_SOURCE,
                    RESOURCE_SUPPORT_VERIFIED_ON,
                )?,
            ]);
            #[cfg(feature = "openai-responses-websocket")]
            native_claims.push(native_support_claim(
                "responses-websocket",
                NativeSurfaceKind::Session,
                ApiStability::Experimental,
                RESPONSES_WEBSOCKET_SOURCE,
                RESPONSES_WEBSOCKET_SUPPORT_VERIFIED_ON,
            )?);
        }
        #[cfg(feature = "openai-realtime")]
        if realtime_endpoint
            .as_ref()
            .is_some_and(OpenAiRealtimeEndpoint::is_official)
        {
            native_claims.push(native_support_claim(
                "realtime",
                NativeSurfaceKind::Session,
                ApiStability::Experimental,
                OPENAI_REALTIME_WEBSOCKET_SOURCE_URL,
                REALTIME_SUPPORT_VERIFIED_ON,
            )?);
        }
        #[cfg(feature = "openai-realtime")]
        if translation_endpoint
            .as_ref()
            .is_some_and(OpenAiRealtimeEndpoint::is_official)
        {
            native_claims.push(native_support_claim(
                "realtime-translation",
                NativeSurfaceKind::Session,
                ApiStability::Experimental,
                OPENAI_REALTIME_TRANSLATION_SOURCE_URL,
                REALTIME_SUPPORT_VERIFIED_ON,
            )?);
        }
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [profile.provider_profile().clone()],
            native_claims,
        )?);
        let auth = self.credential.into_auth(self.organization, self.project)?;
        let mut transport = ProviderTransport::builder(endpoint)
            .with_auth(auth)
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
        if let Some(observer) = self.observer {
            transport = transport.with_observer(observer);
        }
        let transport = transport.build()?;
        #[cfg(feature = "openai-responses-websocket")]
        let responses_websocket = if let Some(endpoint) = responses_websocket_endpoint {
            let auth = responses_websocket_credential.into_auth(
                responses_websocket_organization,
                responses_websocket_project,
            )?;
            let mut websocket = WebSocketTransport::builder(endpoint)
                .with_auth(auth)
                .with_limits(responses_websocket_limits);
            if let Some(timeout) = responses_websocket_connect_timeout {
                websocket = websocket.with_connect_timeout(timeout);
            }
            if let Some(timeout) = responses_websocket_session_timeout {
                websocket = websocket.with_session_timeout(timeout);
            }
            if let Some(timeout) = responses_websocket_io_timeout {
                websocket = websocket.with_io_timeout(timeout);
            }
            Some(OpenAiResponsesWebSocketRuntime::new(
                websocket.build()?,
                responses_websocket_turn_timeout,
            )?)
        } else {
            None
        };
        let policy = Arc::new(OpenAiModelPolicy::new(&profile));
        Ok(OpenAiProvider {
            runtime: Arc::new(OpenAiRuntime {
                profile,
                support_manifest,
                transport,
                policy,
                responses_wire_dialect,
                responses_options: OpenAiOptionMerger::responses(self.responses_defaults)?,
                chat_completions_options: OpenAiOptionMerger::chat_completions(
                    self.chat_completions_defaults,
                )?,
                embedding_options: self.embedding_defaults,
                image_options: self.image_defaults,
                speech_options: self.speech_defaults,
                transcription_options: self.transcription_defaults,
                replay_safety: ReplaySafety::Never,
                #[cfg(feature = "openai-realtime")]
                realtime_credential,
                #[cfg(feature = "openai-realtime")]
                realtime_organization,
                #[cfg(feature = "openai-realtime")]
                realtime_project,
                #[cfg(feature = "openai-realtime")]
                realtime_endpoint,
                #[cfg(feature = "openai-realtime")]
                translation_endpoint,
                #[cfg(feature = "openai-realtime")]
                realtime_limits,
                #[cfg(feature = "openai-realtime")]
                realtime_connect_timeout,
                #[cfg(feature = "openai-realtime")]
                realtime_session_timeout,
                #[cfg(feature = "openai-realtime")]
                realtime_io_timeout,
                #[cfg(feature = "openai-responses-websocket")]
                responses_websocket,
            }),
        })
    }
}

impl fmt::Debug for OpenAiProviderBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiProviderBuilder")
            .field("credential", &self.credential)
            .field("has_custom_endpoint", &self.custom_endpoint)
            .field("has_replay_domain", &self.replay_domain.is_some())
            .field(
                "organization",
                &self.organization.as_ref().map(|_| "[REDACTED]"),
            )
            .field("project", &self.project.as_ref().map(|_| "[REDACTED]"))
            .field("limits", &self.limits)
            .field("retry_policy", &self.retry_policy)
            .field("connect_timeout", &self.connect_timeout)
            .field("call_timeout", &self.call_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("has_transport_observer", &self.observer.is_some())
            .field("has_realtime_endpoint", &{
                #[cfg(feature = "openai-realtime")]
                {
                    self.realtime_endpoint.is_some()
                }
                #[cfg(not(feature = "openai-realtime"))]
                {
                    false
                }
            })
            .field("has_translation_endpoint", &{
                #[cfg(feature = "openai-realtime")]
                {
                    self.translation_endpoint.is_some()
                }
                #[cfg(not(feature = "openai-realtime"))]
                {
                    false
                }
            })
            .field("has_responses_websocket_endpoint", &{
                #[cfg(feature = "openai-responses-websocket")]
                {
                    self.responses_websocket_endpoint.is_some()
                }
                #[cfg(not(feature = "openai-responses-websocket"))]
                {
                    false
                }
            })
            .finish()
    }
}

pub(crate) struct OpenAiRuntime {
    pub(crate) profile: OpenAiProfile,
    pub(crate) support_manifest: Arc<ProviderSupportManifest>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<OpenAiModelPolicy>,
    pub(crate) responses_wire_dialect: ResponsesWireDialect,
    responses_options: OpenAiOptionMerger,
    chat_completions_options: OpenAiOptionMerger,
    embedding_options: OpenAiEmbeddingOptions,
    image_options: OpenAiImageOptions,
    speech_options: OpenAiSpeechOptions,
    transcription_options: OpenAiTranscriptionOptions,
    pub(crate) replay_safety: ReplaySafety,
    #[cfg(feature = "openai-realtime")]
    realtime_credential: OpenAiCredential,
    #[cfg(feature = "openai-realtime")]
    realtime_organization: Option<String>,
    #[cfg(feature = "openai-realtime")]
    realtime_project: Option<String>,
    #[cfg(feature = "openai-realtime")]
    realtime_endpoint: Option<OpenAiRealtimeEndpoint>,
    #[cfg(feature = "openai-realtime")]
    translation_endpoint: Option<OpenAiRealtimeEndpoint>,
    #[cfg(feature = "openai-realtime")]
    realtime_limits: TransportLimits,
    #[cfg(feature = "openai-realtime")]
    realtime_connect_timeout: Option<Duration>,
    #[cfg(feature = "openai-realtime")]
    realtime_session_timeout: Option<Duration>,
    #[cfg(feature = "openai-realtime")]
    realtime_io_timeout: Option<Duration>,
    #[cfg(feature = "openai-responses-websocket")]
    pub(crate) responses_websocket: Option<OpenAiResponsesWebSocketRuntime>,
}

fn native_support_claim(
    surface: &str,
    kind: NativeSurfaceKind,
    stability: ApiStability,
    source: &str,
    verified_on: &str,
) -> Result<VerifiedNativeSupportClaim, OpenAiConfigError> {
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            siumai_core::ProviderId::new("openai")?,
            siumai_core::PlatformId::new("openai-api")?,
            kind,
            NativeSurfaceId::new(surface)?,
        ),
        VerifiedFidelity::Native,
        stability,
        NativeVerificationEvidence::new(
            OfficialSource::new(source)?,
            VerificationDate::new(NaiveDate::parse_from_str(verified_on, "%Y-%m-%d")?),
        ),
    ))
}

impl OpenAiRuntime {
    pub(crate) fn scope(&self, mode: OpenAiApiMode) -> &ProviderScope {
        self.profile.provider_scope(mode)
    }

    pub(crate) fn scope_arc(&self, mode: OpenAiApiMode) -> Arc<ProviderScope> {
        self.profile.provider_scope(mode).clone()
    }

    pub(crate) fn family_scope_arc(&self, family: ModelFamily) -> Arc<ProviderScope> {
        self.profile
            .family_provider_scope(family)
            .expect("OpenAI runtime contains every exposed portable family")
            .clone()
    }

    pub(crate) fn merge_options(
        &self,
        mode: OpenAiApiMode,
        options: &CallOptions,
    ) -> Result<OpenAiMergedOptions, ProviderOptionError> {
        let scope = self.profile.provider_scope(mode);
        let layers =
            options.apply_provider_options(scope.provider_id(), ProviderOptionLayers::default())?;
        let merger = match mode {
            OpenAiApiMode::Responses => &self.responses_options,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_options,
        };
        layers.merge_for(
            ProviderOptionContext::new(
                scope.provider_id(),
                ModelFamily::Language,
                scope.api_mode(),
            ),
            merger,
        )
    }
}

impl fmt::Debug for OpenAiRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRuntime")
            .field("profile_id", self.profile.provider_profile().id())
            .field("transport", &"shared")
            .field("replay_safety", &self.replay_safety)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptionMode {
    Responses,
    ChatCompletions,
}

struct OpenAiOptionMerger {
    mode: OptionMode,
    defaults: Map<String, Value>,
}

pub(crate) struct OpenAiMergedOptions {
    pub(crate) wire: BTreeMap<String, Value>,
    pub(crate) native_tools: Vec<Value>,
    pub(crate) function_tools: BTreeMap<String, FunctionToolEncodingOptions>,
}

impl OpenAiOptionMerger {
    fn responses(defaults: OpenAiResponsesOptions) -> Result<Self, OpenAiConfigError> {
        Ok(Self {
            mode: OptionMode::Responses,
            defaults: serialize_object(defaults)
                .map_err(OpenAiConfigError::InvalidResponsesDefaults)?,
        })
    }

    fn chat_completions(defaults: OpenAiChatCompletionsOptions) -> Result<Self, OpenAiConfigError> {
        Ok(Self {
            mode: OptionMode::ChatCompletions,
            defaults: serialize_object(defaults)
                .map_err(OpenAiConfigError::InvalidChatCompletionsDefaults)?,
        })
    }

    fn allowed_fields(&self) -> &'static [&'static str] {
        match self.mode {
            OptionMode::Responses => RESPONSES_OPTION_FIELDS,
            OptionMode::ChatCompletions => CHAT_COMPLETIONS_OPTION_FIELDS,
        }
    }

    fn validate_typed(&self, value: &Map<String, Value>) -> Result<(), ProviderOptionError> {
        if let Some(field) = value
            .keys()
            .find(|field| !self.allowed_fields().contains(&field.as_str()))
        {
            return Err(ProviderOptionError::Rejected {
                path: field.clone(),
                reason: format!("field does not belong to the {:?} API mode", self.mode),
            });
        }
        match self.mode {
            OptionMode::Responses => {
                deserialize_options::<OpenAiResponsesOptions>(value)?.validate_values()
            }
            OptionMode::ChatCompletions => {
                deserialize_options::<OpenAiChatCompletionsOptions>(value)?.validate_values()
            }
        }
    }

    fn validate_raw(&self, value: &Map<String, Value>) -> Result<(), ProviderOptionError> {
        if let Some(field) = value
            .keys()
            .find(|field| is_protected_field(self.mode, field))
        {
            return Err(ProviderOptionError::Rejected {
                path: field.clone(),
                reason: "field is owned by the canonical language request".to_string(),
            });
        }
        validate_known_projection(self.mode, value)
    }
}

impl ProviderOptionMerger for OpenAiOptionMerger {
    type Output = OpenAiMergedOptions;

    fn validate_layer(
        &self,
        origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        match origin {
            ProviderOptionOrigin::RawOverride => self.validate_raw(options.value()),
            _ => self.validate_typed(options.value()),
        }
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut typed = self.defaults.clone();
        let mut raw = None;
        for (origin, options) in layers.in_precedence_order() {
            if origin == ProviderOptionOrigin::RawOverride {
                raw = Some(options.value());
            } else {
                merge_typed_layer(&mut typed, options.value());
            }
        }
        self.validate_typed(&typed)?;
        let (mut wire, native_tools, function_tools) = match self.mode {
            OptionMode::Responses => {
                let options = deserialize_options::<OpenAiResponsesOptions>(&typed)?
                    .into_request_options()?;
                (options.wire, options.native_tools, options.function_tools)
            }
            OptionMode::ChatCompletions => {
                let options = deserialize_options::<OpenAiChatCompletionsOptions>(&typed)?
                    .into_request_options()?;
                (options.wire, Vec::new(), BTreeMap::new())
            }
        };
        if let Some(raw) = raw {
            wire.extend(
                raw.iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
        }
        let validation_wire = wire.clone().into_iter().collect::<Map<_, _>>();
        validate_final_wire(self.mode, &validation_wire)?;
        Ok(OpenAiMergedOptions {
            wire,
            native_tools,
            function_tools,
        })
    }
}

const RESPONSES_OPTION_FIELDS: &[&str] = &[
    "conversation",
    "include",
    "instructions",
    "max_tool_calls",
    "top_logprobs",
    "metadata",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "text_verbosity",
    "truncation",
    "user",
    "context_management",
    "tools",
    "function_tool_options",
];

const CHAT_COMPLETIONS_OPTION_FIELDS: &[&str] = &[
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    "user",
    "reasoning_effort",
    "store",
    "metadata",
    "service_tier",
    "text_verbosity",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
];

fn merge_typed_layer(base: &mut Map<String, Value>, higher: &Map<String, Value>) {
    for (name, value) in higher {
        if matches!(
            name.as_str(),
            "reasoning" | "prompt_cache_options" | "metadata"
        ) && let (Some(Value::Object(base)), Value::Object(higher)) = (base.get_mut(name), value)
        {
            base.extend(higher.clone());
        } else {
            base.insert(name.clone(), value.clone());
        }
    }
}

fn validate_final_wire(
    mode: OptionMode,
    wire: &Map<String, Value>,
) -> Result<(), ProviderOptionError> {
    let mut projection = Map::new();
    let fields = match mode {
        OptionMode::Responses => RESPONSES_OPTION_FIELDS,
        OptionMode::ChatCompletions => CHAT_COMPLETIONS_OPTION_FIELDS,
    };
    for field in fields {
        if let Some(value) = wire.get(*field) {
            projection.insert((*field).to_string(), value.clone());
        }
    }
    match mode {
        OptionMode::Responses => {
            deserialize_options::<OpenAiResponsesOptions>(&projection)?.validate_values()
        }
        OptionMode::ChatCompletions => {
            if let Some(verbosity) = wire.get("verbosity") {
                projection.insert("text_verbosity".to_string(), verbosity.clone());
            }
            deserialize_options::<OpenAiChatCompletionsOptions>(&projection)?.validate_values()
        }
    }
}

fn validate_known_projection(
    mode: OptionMode,
    value: &Map<String, Value>,
) -> Result<(), ProviderOptionError> {
    validate_final_wire(mode, value)
}

fn is_protected_field(mode: OptionMode, field: &str) -> bool {
    let common = matches!(
        field,
        "model"
            | "stream"
            | "temperature"
            | "top_p"
            | "max_output_tokens"
            | "stop"
            | "seed"
            | "tools"
            | "tool_choice"
            | "prompt_cache_options"
            | "prompt_cache_retention"
            | "prompt_cache_breakpoints"
    );
    common
        || match mode {
            OptionMode::Responses => matches!(
                field,
                "input"
                    | "text"
                    | "background"
                    | "tools"
                    | "function_tool_options"
                    | "type"
                    | "generate"
            ),
            OptionMode::ChatCompletions => matches!(
                field,
                "messages"
                    | "response_format"
                    | "stream_options"
                    | "max_tokens"
                    | "max_completion_tokens"
            ),
        }
}

fn deserialize_options<T: DeserializeOwned>(
    value: &Map<String, Value>,
) -> Result<T, ProviderOptionError> {
    serde_json::from_value(Value::Object(value.clone())).map_err(|error| {
        ProviderOptionError::Rejected {
            path: "openai".to_string(),
            reason: error.to_string(),
        }
    })
}

fn serialize_object<T: Serialize>(value: T) -> Result<Map<String, Value>, ProviderOptionError> {
    match serde_json::to_value(value)
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
    {
        Value::Object(value) => Ok(value),
        _ => Err(ProviderOptionError::ExpectedObject {
            namespace: "openai".to_string(),
        }),
    }
}

fn parse_model_id(model: impl Into<String>) -> Result<ModelId, ModelLookupError> {
    ModelId::new(model.into()).map_err(ModelLookupError::from)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpenAiConfigError {
    #[error(transparent)]
    InvalidIdentity(#[from] InvalidId),
    #[error(transparent)]
    Credential(#[from] OpenAiCredentialError),
    #[error(transparent)]
    Endpoint(#[from] EndpointError),
    #[error(transparent)]
    Transport(#[from] TransportConfigError),
    #[error(transparent)]
    Profile(#[from] ProfileError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error(transparent)]
    SupportManifest(#[from] SupportManifestError),
    #[error("OpenAI support verification date is invalid: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("OpenAI verification date is invalid")]
    InvalidVerificationDate,
    #[error("the official OpenAI endpoint requires authenticated credentials")]
    OfficialEndpointRequiresAuthentication,
    #[error("a custom OpenAI endpoint requires an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured OpenAI endpoint identity")]
    ReplayAudienceMismatch,
    #[error("organization or project configuration requires a non-secret replay caller scope")]
    AccountScopeRequiresReplayCallerScope,
    #[cfg(feature = "openai-responses-websocket")]
    #[error(
        "a caller-controlled Responses WebSocket endpoint requires a caller-controlled HTTP provider"
    )]
    CallerControlledResponsesWebSocketOnOfficialProvider,
    #[cfg(feature = "openai-responses-websocket")]
    #[error(transparent)]
    ResponsesWebSocket(#[from] OpenAiResponsesWebSocketConfigError),
    #[error("invalid default Responses options: {0}")]
    InvalidResponsesDefaults(ProviderOptionError),
    #[error("invalid default Chat Completions options: {0}")]
    InvalidChatCompletionsDefaults(ProviderOptionError),
    #[error("invalid default embedding options: {0}")]
    InvalidEmbeddingDefaults(ProviderOptionError),
    #[error("invalid default image-generation options: {0}")]
    InvalidImageDefaults(ProviderOptionError),
    #[error("invalid default speech options: {0}")]
    InvalidSpeechDefaults(ProviderOptionError),
    #[error("invalid default transcription options: {0}")]
    InvalidTranscriptionDefaults(ProviderOptionError),
}

#[cfg(test)]
mod tests {
    use siumai_core::{ApiStability, Model, ModelAdvisory, ModelOperation, SupportState};

    use super::*;
    use crate::configured::{
        TEXT_EMBEDDING_3_SMALL,
        catalog::{GPT_5_6, GPT_5_6_SOL},
    };

    fn provider() -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[test]
    fn default_and_explicit_models_share_one_runtime_but_keep_modes() {
        let provider = provider();
        let model_id = ModelId::new(GPT_5_6_SOL).unwrap();
        let responses = provider.language_model(model_id.clone()).unwrap();
        let trait_responses =
            <OpenAiProvider as LanguageModelProvider>::language_model(&provider, model_id).unwrap();
        let chat = provider.chat_completions(GPT_5_6_SOL).unwrap();

        assert!(Arc::ptr_eq(&responses.runtime, &chat.runtime));
        assert_eq!(responses.descriptor(), trait_responses.descriptor());
        assert_eq!(responses.descriptor().provider().as_str(), "openai");
        assert_eq!(chat.descriptor().provider().as_str(), "openai");
        assert_eq!(responses.descriptor().api_mode(), Some("responses"));
        assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));
    }

    #[test]
    fn direct_and_mode_bound_registration_descriptors_match() {
        let provider = provider();
        let direct = provider.responses(GPT_5_6_SOL).unwrap();
        let erased = provider
            .responses_registration()
            .language_model(ModelId::new(GPT_5_6_SOL).unwrap())
            .unwrap();

        assert_eq!(direct.descriptor(), erased.descriptor());
    }

    #[test]
    fn default_registration_binds_all_portable_families_with_exact_scopes() {
        let provider = provider();
        let registration = provider.registration();

        assert_eq!(
            registration.families().collect::<Vec<_>>(),
            vec![
                ModelFamily::Language,
                ModelFamily::Embedding,
                ModelFamily::Image,
                ModelFamily::Speech,
                ModelFamily::Transcription,
            ]
        );

        let direct = provider.embedding(TEXT_EMBEDDING_3_SMALL).unwrap();
        let erased = registration
            .embedding_model(ModelId::new(TEXT_EMBEDDING_3_SMALL).unwrap())
            .unwrap();
        assert_eq!(direct.descriptor(), erased.descriptor());
        assert_eq!(direct.descriptor().protocol(), Some("openai.embeddings"));
        assert_eq!(direct.descriptor().api_mode(), Some("embeddings"));

        assert!(
            provider
                .responses_registration()
                .scope(ModelFamily::Embedding)
                .is_none()
        );
        assert!(
            provider
                .chat_completions_registration()
                .scope(ModelFamily::Embedding)
                .is_none()
        );
    }

    #[test]
    fn official_policy_keeps_alias_and_unknown_model_distinct() {
        let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .build()
            .unwrap();
        let registration = provider.responses_registration();
        let alias = registration.evaluate(ModelId::new(GPT_5_6).unwrap(), ModelOperation::Generate);
        assert_eq!(alias.state(), &SupportState::Supported);
        assert_eq!(alias.advisories(), &[ModelAdvisory::RollingAlias]);

        let future = registration.evaluate(
            ModelId::new("gpt-6-future").unwrap(),
            ModelOperation::Generate,
        );
        assert_eq!(future.state(), &SupportState::Unknown);
        assert_eq!(future.advisories(), &[ModelAdvisory::UnknownModel]);
    }

    #[test]
    fn custom_endpoint_uses_generic_profile_and_unknown_model_policy() {
        let provider = provider();
        let profile = provider.profile().provider_profile();

        assert!(profile.verified_claims().is_none());
        assert!(profile.catalog().is_none());
        let decision = provider
            .responses_registration()
            .evaluate(ModelId::new(GPT_5_6_SOL).unwrap(), ModelOperation::Generate);
        assert_eq!(decision.state(), &SupportState::Unknown);
        assert_eq!(decision.advisories(), &[ModelAdvisory::UnknownModel]);
        assert!(provider.support_manifest().native_claims().is_empty());
    }

    #[test]
    fn caller_supplied_official_policy_remains_a_custom_endpoint() {
        let endpoint = EndpointConfig::official(
            "https://relay.example/v1",
            OfficialOrigin::new("https://relay.example").unwrap(),
        )
        .unwrap();

        let missing_domain = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_err();
        assert!(matches!(
            missing_domain,
            OpenAiConfigError::CustomEndpointRequiresReplayDomain
        ));

        let official_domain = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .with_endpoint(endpoint.clone())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("forged-official").unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(
            official_domain,
            OpenAiConfigError::ReplayAudienceMismatch
        ));

        let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("caller-relay").unwrap(),
            ))
            .build()
            .unwrap();
        assert!(
            provider
                .profile()
                .provider_profile()
                .verified_claims()
                .is_none()
        );
        assert!(provider.support_manifest().native_claims().is_empty());
    }

    #[test]
    fn support_manifest_declares_official_resources_and_sessions() {
        let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-api-key"))
            .build()
            .unwrap();
        let manifest = provider.support_manifest();

        assert_eq!(manifest.profiles().len(), 1);
        for surface in [
            "responses-resource-lifecycle",
            "conversations-basic-items",
            "files-basic-lifecycle",
            "vector-stores-basic-files",
        ] {
            assert!(manifest.native_claims().iter().any(|claim| {
                claim
                    .scope()
                    .binding()
                    .surface_id()
                    .is_some_and(|candidate| candidate.as_str() == surface)
                    && claim.stability() == ApiStability::Stable
            }));
        }
        assert!(manifest.native_claims().iter().any(|claim| {
            claim
                .scope()
                .binding()
                .surface_id()
                .is_some_and(|surface| surface.as_str() == "skills-directory-lifecycle")
                && claim.stability() == ApiStability::Experimental
        }));
        #[cfg(feature = "openai-realtime")]
        assert!(manifest.native_claims().iter().any(|claim| {
            claim
                .scope()
                .binding()
                .surface_id()
                .is_some_and(|surface| surface.as_str() == "realtime")
                && claim.stability() == ApiStability::Experimental
        }));
        #[cfg(feature = "openai-responses-websocket")]
        assert!(manifest.native_claims().iter().any(|claim| {
            claim
                .scope()
                .binding()
                .surface_id()
                .is_some_and(|surface| surface.as_str() == "responses-websocket")
                && claim.scope().kind() == NativeSurfaceKind::Session
                && claim.stability() == ApiStability::Experimental
        }));
    }

    #[test]
    fn official_endpoint_rejects_unauthenticated_configuration() {
        let error = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .build()
            .unwrap_err();

        assert!(matches!(
            error,
            OpenAiConfigError::OfficialEndpointRequiresAuthentication
        ));
    }

    #[cfg(feature = "openai-realtime")]
    #[test]
    fn configured_provider_creates_distinct_realtime_session_configs() {
        use crate::configured::{OPENAI_REALTIME_MODEL, OPENAI_REALTIME_TRANSLATION_MODEL};

        let provider = OpenAiProvider::builder(OpenAiCredential::api_key("canary-secret"))
            .with_organization("org-example")
            .with_project("proj-example")
            .with_replay_domain(
                ReplayDomain::official(ReplayDomainId::new("official").unwrap())
                    .with_caller_scope(ReplayDomainId::new("test-account").unwrap()),
            )
            .build()
            .unwrap();
        let conversation = provider.realtime(OPENAI_REALTIME_MODEL).unwrap();
        let translation = provider
            .translation(OPENAI_REALTIME_TRANSLATION_MODEL)
            .unwrap();

        assert_eq!(conversation.model(), OPENAI_REALTIME_MODEL);
        assert_eq!(translation.model(), OPENAI_REALTIME_TRANSLATION_MODEL);
        assert!(conversation.endpoint().is_official());
        assert!(translation.endpoint().is_official());
        assert!(conversation.validate().is_ok());
        assert!(translation.validate().is_ok());
        assert!(!format!("{conversation:?}").contains("canary-secret"));
        assert!(!format!("{translation:?}").contains("canary-secret"));
    }

    #[cfg(feature = "openai-realtime")]
    #[test]
    fn custom_http_provider_requires_explicit_realtime_audiences() {
        let provider = provider();

        assert!(matches!(
            provider.realtime(crate::configured::OPENAI_REALTIME_MODEL),
            Err(OpenAiRealtimeConfigError::ExplicitEndpointRequiredForCustomProvider)
        ));
        assert!(matches!(
            provider.translation(crate::configured::OPENAI_REALTIME_TRANSLATION_MODEL),
            Err(OpenAiRealtimeConfigError::ExplicitEndpointRequiredForCustomProvider)
        ));
    }
}
