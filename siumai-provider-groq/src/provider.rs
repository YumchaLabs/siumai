//! Long-lived Groq provider and synchronous model construction.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, CatalogError, Error, GenericSupportClaim, InvalidId,
    LanguageCallError, LanguageModel, LanguageModelProvider, LanguageRequest, LanguageResponse,
    LanguageStream, Model, ModelCatalog, ModelDescriptor, ModelFamily, ModelId, ModelLifecycle,
    ModelLookupError, ModelOperation, ModelProfile, NativeSupportScope, NativeSurfaceId,
    NativeSurfaceKind, NativeVerificationEvidence, OfficialSource, PlatformId, ProfileError,
    ProfileId, ProtocolContractId, ProtocolId, Provider, ProviderId, ProviderInstanceId,
    ProviderOptionError, ProviderOptions, ProviderProfile, ProviderRegistration,
    ProviderRegistrationError, ProviderScope, ProviderSupportManifest, ReplayDomain,
    ReplayDomainId, SpeechModel, SpeechModelProvider, SupportManifestError, SupportScope,
    TranscriptionModel, TranscriptionModelProvider, TypedProviderOptions, VerificationDate,
    VerificationEvidence, VerifiedFidelity, VerifiedNativeSupportClaim, VerifiedSupportClaim,
};
use siumai_openai_compatible::{
    CredentialSourceError, DynamicCredentialSource, OpenAiCompatibleApiMode,
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProvider,
};
use siumai_transport::{
    EndpointConfig, EndpointError, EndpointPolicy, OfficialOrigin, ProviderTransport, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::audio::GroqAudio;
use crate::language::{DEFAULT_BASE_URL, GroqProfileError, PLATFORM_ID, PROVIDER_ID, profile};
use crate::options::{GroqLanguageOptions, GroqResponsesOptions, GroqTranscriptionOptions};
use crate::speech::{GroqSpeechModel, SPEECH_API_MODE_ID, SPEECH_PROTOCOL_ID, SPEECH_SOURCE};
use crate::transcription::{
    GroqTranscriptionModel, GroqTranscriptionRuntime, TRANSCRIPTION_API_MODE_ID,
    TRANSCRIPTION_PROTOCOL_ID, TRANSCRIPTION_SOURCE,
};

const OFFICIAL_ORIGIN: &str = "https://api.groq.com";
const OFFICIAL_REPLAY_DOMAIN_ID: &str = "groq-public-api";
const TRANSCRIPTION_VERIFIED_ON: &str = "2026-08-06";
const SPEECH_VERIFIED_ON: &str = "2026-08-09";
const AUDIO_NATIVE_VERIFIED_ON: &str = "2026-08-09";

/// Explicit Groq authentication configuration.
#[derive(Clone)]
pub struct GroqCredential {
    inner: OpenAiCompatibleCredential,
    authenticated: bool,
}

impl GroqCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleCredential::api_key(value),
            authenticated: true,
        }
    }

    pub fn dynamic(source: Arc<dyn DynamicCredentialSource>) -> Self {
        Self {
            inner: OpenAiCompatibleCredential::dynamic(source),
            authenticated: true,
        }
    }

    pub fn unauthenticated() -> Self {
        Self {
            inner: OpenAiCompatibleCredential::unauthenticated(),
            authenticated: false,
        }
    }
}

impl fmt::Debug for GroqCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqCredential")
            .field("authenticated", &self.authenticated)
            .field("material", &"[REDACTED]")
            .finish()
    }
}

/// Public Groq language API selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GroqLanguageApi {
    #[default]
    ChatCompletions,
    Responses,
}

impl From<GroqLanguageApi> for OpenAiCompatibleApiMode {
    fn from(value: GroqLanguageApi) -> Self {
        match value {
            GroqLanguageApi::ChatCompletions => Self::ChatCompletions,
            GroqLanguageApi::Responses => Self::Responses,
        }
    }
}

/// Long-lived, model-independent Groq provider.
#[derive(Clone)]
pub struct GroqProvider {
    language: OpenAiCompatibleProvider,
    chat_registration: ProviderRegistration,
    default_registration: ProviderRegistration,
    transcription: Arc<GroqTranscriptionRuntime>,
    speech: Arc<GroqSpeechRuntime>,
    audio: GroqAudio,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl GroqProvider {
    pub fn builder(credential: GroqCredential) -> GroqProviderBuilder {
        GroqProviderBuilder::new(credential)
    }

    pub fn from_api_key(api_key: impl Into<String>) -> Result<Self, GroqConfigError> {
        Self::builder(GroqCredential::api_key(api_key)).build()
    }

    /// Inspect the exact language, transcription, and speech claims for this configuration.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        &self.support_manifest
    }

    /// Create a lightweight Chat Completions model from an open textual model ID.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<GroqLanguageModel, ModelLookupError> {
        self.chat_completions(model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<GroqLanguageModel, ModelLookupError> {
        self.language_for(GroqLanguageApi::ChatCompletions, model)
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<GroqLanguageModel, ModelLookupError> {
        self.language_for(GroqLanguageApi::Responses, model)
    }

    pub fn language_for(
        &self,
        api: GroqLanguageApi,
        model: impl Into<String>,
    ) -> Result<GroqLanguageModel, ModelLookupError> {
        self.language
            .language_for(api.into(), model)
            .map(GroqLanguageModel)
    }

    /// Create a lightweight final-result transcription model from an open model ID.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Result<GroqTranscriptionModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_transcription_model(model))
    }

    /// Create a lightweight buffered Orpheus speech model from an open model ID.
    pub fn speech(&self, model: impl Into<String>) -> Result<GroqSpeechModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_speech_model(model))
    }

    pub fn default_speech_model(&self) -> Result<GroqSpeechModel, ModelLookupError> {
        self.speech(crate::models::DEFAULT_SPEECH)
    }

    /// Access provider-owned URL-audio transcription and translation operations.
    pub fn audio(&self) -> GroqAudio {
        self.audio.clone()
    }

    pub fn language_registration(&self) -> Option<ProviderRegistration> {
        Some(self.chat_registration.clone())
    }

    pub fn chat_completions_registration(&self) -> Option<ProviderRegistration> {
        Some(self.chat_registration.clone())
    }

    pub fn responses_registration(&self) -> Option<ProviderRegistration> {
        self.language.responses_registration()
    }

    pub fn registration_for(&self, api: GroqLanguageApi) -> Option<ProviderRegistration> {
        match api {
            GroqLanguageApi::ChatCompletions => self.chat_completions_registration(),
            GroqLanguageApi::Responses => self.responses_registration(),
        }
    }

    /// Register Groq's default language, transcription, and speech family bindings.
    pub fn registration(&self) -> ProviderRegistration {
        self.default_registration.clone()
    }

    pub fn transcription_registration(&self) -> ProviderRegistration {
        let runtime = self.transcription.clone();
        ProviderRegistration::from_transcription(
            self.transcription.scope.clone(),
            Arc::new(move |model| {
                Ok(
                    Arc::new(GroqTranscriptionModel::new(runtime.clone(), model))
                        as Arc<dyn TranscriptionModel>,
                )
            }),
        )
    }

    pub fn speech_registration(&self) -> ProviderRegistration {
        let runtime = self.speech.clone();
        ProviderRegistration::from_speech(
            self.speech.scope.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(GroqSpeechModel::new(runtime.clone(), model)) as Arc<dyn SpeechModel>)
            }),
        )
    }

    fn create_transcription_model(&self, model: ModelId) -> GroqTranscriptionModel {
        GroqTranscriptionModel::new(self.transcription.clone(), model)
    }

    fn create_speech_model(&self, model: ModelId) -> GroqSpeechModel {
        GroqSpeechModel::new(self.speech.clone(), model)
    }
}

impl Provider for GroqProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for GroqProvider {
    type Model = GroqLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.language
            .language_for(OpenAiCompatibleApiMode::ChatCompletions, model.to_string())
            .map(GroqLanguageModel)
    }
}

impl TranscriptionModelProvider for GroqProvider {
    type Model = GroqTranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }
}

impl SpeechModelProvider for GroqProvider {
    type Model = GroqSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }
}

impl fmt::Debug for GroqProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqProvider")
            .field("provider_id", &PROVIDER_ID)
            .field(
                "chat_scope",
                &self.chat_registration.scope(ModelFamily::Language),
            )
            .field("transcription_scope", &self.transcription.scope)
            .field("speech_scope", &self.speech.scope)
            .finish()
    }
}

/// Synchronous Groq provider configuration.
pub struct GroqProviderBuilder {
    credential: GroqCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: GroqLanguageOptions,
    responses_defaults: GroqResponsesOptions,
    transcription_defaults: GroqTranscriptionOptions,
}

impl GroqProviderBuilder {
    fn new(credential: GroqCredential) -> Self {
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
            chat_defaults: GroqLanguageOptions::default(),
            responses_defaults: GroqResponsesOptions::default(),
            transcription_defaults: GroqTranscriptionOptions::default(),
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

    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::public_custom(base_url);
        self.provider_selected_endpoint = false;
        self
    }

    /// Bind provider-native history to a non-secret replay domain.
    ///
    /// Caller-controlled endpoints require an explicit custom audience. The provider-selected
    /// endpoint uses Groq's official audience by default, but callers may add a material account
    /// boundary with [`ReplayDomain::with_caller_scope`].
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
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

    pub fn with_chat_completions_defaults(mut self, defaults: GroqLanguageOptions) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: GroqResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn with_transcription_defaults(mut self, defaults: GroqTranscriptionOptions) -> Self {
        self.transcription_defaults = defaults;
        self
    }

    /// Validate static configuration and construct shared runtimes without network I/O.
    pub fn build(self) -> Result<GroqProvider, GroqConfigError> {
        self.credential.inner.validate_static()?;
        self.chat_defaults.validate()?;
        self.responses_defaults.validate()?;
        self.transcription_defaults.validate()?;
        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let replay_domain =
            replay_domain_for_endpoint(self.replay_domain.clone(), verified_endpoint)?;
        if !self.credential.authenticated && verified_endpoint {
            return Err(GroqConfigError::OfficialEndpointRequiresCredential);
        }
        let auth = self.credential.inner.into_auth();
        let instance_id = ProviderInstanceId::new();
        let profile = profile(endpoint.clone(), replay_domain.clone(), verified_endpoint)?;
        let mut language = OpenAiCompatibleProvider::builder_with_auth(profile, auth.clone())
            .with_provider_instance(instance_id.clone())
            .with_limits(self.limits.clone())
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            language = language.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            language = language.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            language = language.with_read_timeout(timeout);
        }
        for (name, value) in option_map(&self.chat_defaults)? {
            language =
                language.with_default_option(OpenAiCompatibleApiMode::ChatCompletions, name, value);
        }
        for (name, value) in option_map(&self.responses_defaults)? {
            language =
                language.with_default_option(OpenAiCompatibleApiMode::Responses, name, value);
        }

        let mut media_transport = ProviderTransport::builder(endpoint.clone())
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            media_transport = media_transport.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            media_transport = media_transport.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            media_transport = media_transport.with_read_timeout(timeout);
        }
        let transcription_scope = Arc::new(transcription_scope(
            &endpoint,
            replay_domain.clone(),
            verified_endpoint,
        )?);
        let speech_scope = Arc::new(speech_scope(&endpoint, replay_domain, verified_endpoint)?);
        let transcription_profile = transcription_profile(&transcription_scope, verified_endpoint)?;
        let speech_profile = speech_profile(&speech_scope, verified_endpoint)?;
        let transcription_defaults = ProviderOptions::typed(&self.transcription_defaults)?;
        let language = language.build()?;
        let chat_registration = language
            .chat_completions_registration()
            .ok_or(GroqConfigError::MissingChatCompletionsMode)?;
        let media_transport = media_transport.build()?;
        let transcription = Arc::new(GroqTranscriptionRuntime::new(
            instance_id.clone(),
            transcription_scope,
            media_transport.clone(),
            transcription_defaults,
        ));
        let audio = GroqAudio::new(media_transport.clone());
        let speech = Arc::new(GroqSpeechRuntime::new(
            instance_id,
            speech_scope,
            media_transport,
        ));
        let transcription_registration =
            ProviderRegistration::from_transcription(transcription.scope.clone(), {
                let runtime = transcription.clone();
                Arc::new(move |model| {
                    Ok(
                        Arc::new(GroqTranscriptionModel::new(runtime.clone(), model))
                            as Arc<dyn TranscriptionModel>,
                    )
                })
            });
        let speech_registration = ProviderRegistration::from_speech(speech.scope.clone(), {
            let runtime = speech.clone();
            Arc::new(move |model| {
                Ok(Arc::new(GroqSpeechModel::new(runtime.clone(), model)) as Arc<dyn SpeechModel>)
            })
        });
        let default_registration = chat_registration
            .clone()
            .merge(transcription_registration)?
            .merge(speech_registration)?;
        let native_claims = if verified_endpoint {
            audio_native_claims()?
        } else {
            Vec::new()
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            ProviderId::new(PROVIDER_ID)?,
            [
                language.profile().provider_profile().clone(),
                transcription_profile,
                speech_profile,
            ],
            native_claims,
        )?);
        Ok(GroqProvider {
            language,
            chat_registration,
            default_registration,
            transcription,
            speech,
            audio,
            support_manifest,
        })
    }
}

impl fmt::Debug for GroqProviderBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqProviderBuilder")
            .field("credential", &self.credential)
            .field("endpoint", &self.endpoint)
            .field(
                "provider_selected_endpoint",
                &self.provider_selected_endpoint,
            )
            .field("replay_domain", &self.replay_domain)
            .field("limits", &self.limits)
            .field("retry_policy", &self.retry_policy)
            .field("connect_timeout", &self.connect_timeout)
            .field("call_timeout", &self.call_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("chat_defaults", &self.chat_defaults)
            .field("responses_defaults", &self.responses_defaults)
            .field("transcription_defaults", &self.transcription_defaults)
            .finish()
    }
}

/// Groq-branded wrapper around the shared compatible language model.
#[derive(Clone)]
pub struct GroqLanguageModel(OpenAiCompatibleLanguageModel);

impl Model for GroqLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.0.descriptor()
    }
}

#[async_trait]
impl LanguageModel for GroqLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.0.generate(request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.0.stream(request, options).await
    }
}

impl fmt::Debug for GroqLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqLanguageModel")
            .field("descriptor", self.descriptor())
            .finish()
    }
}

pub(crate) struct GroqSpeechRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
}

impl GroqSpeechRuntime {
    fn new(
        instance_id: ProviderInstanceId,
        scope: Arc<ProviderScope>,
        transport: ProviderTransport,
    ) -> Self {
        Self {
            instance_id,
            scope,
            transport,
        }
    }
}

impl fmt::Debug for GroqSpeechRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqSpeechRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .finish()
    }
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn transcription_scope(
    endpoint: &EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<ProviderScope, InvalidId> {
    let platform = match (verified_endpoint, endpoint.policy()) {
        (true, _) => PLATFORM_ID,
        (false, EndpointPolicy::Official(_)) => "official-custom-endpoint",
        (false, EndpointPolicy::PublicCustom) => "custom-endpoint",
        (false, EndpointPolicy::LocalExplicit(_)) => "local",
        (false, _) => "custom-endpoint",
    };
    Ok(ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
        .with_platform(PlatformId::new(platform)?)
        .with_protocol(ProtocolId::new(TRANSCRIPTION_PROTOCOL_ID)?)
        .with_api_mode(ApiModeId::new(TRANSCRIPTION_API_MODE_ID)?)
        .with_replay_domain(replay_domain))
}

fn speech_scope(
    endpoint: &EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<ProviderScope, InvalidId> {
    let platform = match (verified_endpoint, endpoint.policy()) {
        (true, _) => PLATFORM_ID,
        (false, EndpointPolicy::Official(_)) => "official-custom-endpoint",
        (false, EndpointPolicy::PublicCustom) => "custom-endpoint",
        (false, EndpointPolicy::LocalExplicit(_)) => "local",
        (false, _) => "custom-endpoint",
    };
    Ok(ProviderScope::new(ProviderId::new(PROVIDER_ID)?)
        .with_platform(PlatformId::new(platform)?)
        .with_protocol(ProtocolId::new(SPEECH_PROTOCOL_ID)?)
        .with_api_mode(ApiModeId::new(SPEECH_API_MODE_ID)?)
        .with_replay_domain(replay_domain))
}

fn replay_domain_for_endpoint(
    configured: Option<ReplayDomain>,
    verified_endpoint: bool,
) -> Result<ReplayDomain, GroqConfigError> {
    let replay_domain = match (configured, verified_endpoint) {
        (Some(replay_domain), _) => replay_domain,
        (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN_ID)?),
        (None, false) => return Err(GroqConfigError::CustomEndpointRequiresReplayDomain),
    };
    if replay_domain.audience().is_official() != verified_endpoint {
        return Err(GroqConfigError::ReplayAudienceMismatch);
    }
    Ok(replay_domain)
}

fn transcription_profile(
    scope: &ProviderScope,
    verified_endpoint: bool,
) -> Result<ProviderProfile, GroqConfigError> {
    let support_scope = SupportScope::new(
        scope.provider_id().clone(),
        scope
            .platform()
            .cloned()
            .ok_or(GroqConfigError::IncompleteTranscriptionScope)?,
        ModelFamily::Transcription,
        scope
            .protocol()
            .cloned()
            .ok_or(GroqConfigError::IncompleteTranscriptionScope)?,
        scope
            .api_mode()
            .cloned()
            .ok_or(GroqConfigError::IncompleteTranscriptionScope)?,
    );
    let profile_id = ProfileId::new("groq-transcription")?;
    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(support_scope, ApiStability::Experimental),
        ));
    }

    let verified_at = VerificationDate::new(NaiveDate::parse_from_str(
        TRANSCRIPTION_VERIFIED_ON,
        "%Y-%m-%d",
    )?);
    let evidence = VerificationEvidence::new(
        OfficialSource::new(TRANSCRIPTION_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-audio-transcriptions-2026-08")?,
    );
    let catalog = ModelCatalog::new(
        crate::models::transcription::KNOWN
            .iter()
            .map(|model| {
                Ok(ModelProfile::new(
                    ModelId::new(*model)?,
                    support_scope.clone(),
                    [ModelOperation::Transcribe],
                    ModelLifecycle::Active,
                    evidence.clone(),
                )?)
            })
            .collect::<Result<Vec<_>, GroqConfigError>>()?,
    )?;
    Ok(ProviderProfile::verified(
        profile_id,
        vec![VerifiedSupportClaim::new(
            support_scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            evidence,
        )],
        catalog,
    )?)
}

fn speech_profile(
    scope: &ProviderScope,
    verified_endpoint: bool,
) -> Result<ProviderProfile, GroqConfigError> {
    let support_scope = SupportScope::new(
        scope.provider_id().clone(),
        scope
            .platform()
            .cloned()
            .ok_or(GroqConfigError::IncompleteSpeechScope)?,
        ModelFamily::Speech,
        scope
            .protocol()
            .cloned()
            .ok_or(GroqConfigError::IncompleteSpeechScope)?,
        scope
            .api_mode()
            .cloned()
            .ok_or(GroqConfigError::IncompleteSpeechScope)?,
    );
    let profile_id = ProfileId::new("groq-speech")?;
    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(support_scope, ApiStability::Experimental),
        ));
    }

    let verified_at =
        VerificationDate::new(NaiveDate::parse_from_str(SPEECH_VERIFIED_ON, "%Y-%m-%d")?);
    let evidence = VerificationEvidence::new(
        OfficialSource::new(SPEECH_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-orpheus-speech-2026-08")?,
    );
    let catalog = ModelCatalog::new(
        crate::models::speech::KNOWN
            .iter()
            .map(|model| {
                Ok(ModelProfile::new(
                    ModelId::new(*model)?,
                    support_scope.clone(),
                    [ModelOperation::SynthesizeSpeech],
                    ModelLifecycle::Active,
                    evidence.clone(),
                )?)
            })
            .collect::<Result<Vec<_>, GroqConfigError>>()?,
    )?;
    Ok(ProviderProfile::verified(
        profile_id,
        vec![VerifiedSupportClaim::new(
            support_scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            evidence,
        )],
        catalog,
    )?)
}

fn audio_native_claims() -> Result<Vec<VerifiedNativeSupportClaim>, GroqConfigError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let verified_at = VerificationDate::new(NaiveDate::parse_from_str(
        AUDIO_NATIVE_VERIFIED_ON,
        "%Y-%m-%d",
    )?);
    let source = OfficialSource::new(TRANSCRIPTION_SOURCE)?;
    Ok(vec![
        VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider.clone(),
                platform.clone(),
                NativeSurfaceKind::Resource,
                NativeSurfaceId::new("url-audio-transcription")?,
            ),
            VerifiedFidelity::Native,
            ApiStability::Stable,
            NativeVerificationEvidence::new(source.clone(), verified_at),
        ),
        VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider,
                platform,
                NativeSurfaceKind::Resource,
                NativeSurfaceId::new("audio-translation")?,
            ),
            VerifiedFidelity::Native,
            ApiStability::Stable,
            NativeVerificationEvidence::new(source, verified_at),
        ),
    ])
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, GroqConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(GroqConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum GroqConfigError {
    #[error("invalid Groq identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Groq endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Groq language profile: {0}")]
    Profile(#[from] GroqProfileError),
    #[error("invalid Groq support profile: {0}")]
    SupportProfile(#[from] ProfileError),
    #[error("invalid Groq support catalog: {0}")]
    SupportCatalog(#[from] CatalogError),
    #[error("invalid Groq support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("invalid Groq support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Groq default registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("invalid Groq credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("invalid Groq language runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid Groq transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Groq provider options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("Groq default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("Groq default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("the configured Groq profile omitted Chat Completions")]
    MissingChatCompletionsMode,
    #[error("the configured Groq transcription scope is incomplete")]
    IncompleteTranscriptionScope,
    #[error("the configured Groq speech scope is incomplete")]
    IncompleteSpeechScope,
    #[error("the official Groq endpoint requires authenticated credentials")]
    OfficialEndpointRequiresCredential,
    #[error("a custom Groq endpoint requires an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Groq endpoint ownership")]
    ReplayAudienceMismatch,
}

#[cfg(test)]
mod tests {
    use siumai_core::ModelFamily;

    use super::*;

    #[test]
    fn configured_provider_native_models_share_one_instance_capability() {
        let first = GroqProvider::builder(GroqCredential::api_key("first-key"))
            .build()
            .unwrap();
        let second = GroqProvider::builder(GroqCredential::api_key("second-key"))
            .build()
            .unwrap();

        let speech = first.speech("future-speech").unwrap();
        let transcription = first.transcription("future-transcription").unwrap();
        let other = second.speech("future-speech").unwrap();

        assert_eq!(
            speech.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_ne!(
            speech.descriptor().instance_id(),
            other.descriptor().instance_id()
        );
    }

    #[test]
    fn construction_is_static_and_future_model_ids_are_open() {
        let provider = GroqProvider::builder(GroqCredential::api_key("test-key"))
            .build()
            .unwrap();
        let official_domain = ReplayDomain::official(
            ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN_ID).expect("official replay domain"),
        );
        for index in 0..1_000 {
            let language = provider
                .language(format!("future-language-{index}"))
                .unwrap();
            let responses = provider
                .responses(format!("future-responses-{index}"))
                .unwrap();
            let transcription = provider
                .transcription(format!("future-transcription-{index}"))
                .unwrap();
            assert_eq!(language.family(), ModelFamily::Language);
            assert_eq!(responses.family(), ModelFamily::Language);
            assert_eq!(transcription.family(), ModelFamily::Transcription);
            assert_eq!(language.provider_id().as_str(), "groq");
            assert_eq!(responses.provider_id().as_str(), "groq");
            assert_eq!(transcription.provider_id().as_str(), "groq");
        }
        assert_eq!(
            provider
                .language("future-language")
                .unwrap()
                .descriptor()
                .replay_domain(),
            Some(&official_domain)
        );
        assert_eq!(
            provider
                .responses("future-responses")
                .unwrap()
                .descriptor()
                .replay_domain(),
            Some(&official_domain)
        );
        assert_eq!(
            provider
                .transcription("future-transcription")
                .unwrap()
                .descriptor()
                .replay_domain(),
            Some(&official_domain)
        );
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Language)
                .unwrap()
                .as_str(),
            "chat-completions"
        );
        assert_eq!(
            provider
                .registration()
                .api_mode(ModelFamily::Transcription)
                .unwrap()
                .as_str(),
            TRANSCRIPTION_API_MODE_ID
        );
        let registration = provider.registration();
        assert_eq!(
            registration
                .language_model(ModelId::new("future-language").unwrap())
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Language
        );
        assert_eq!(
            registration
                .transcription_model(ModelId::new("future-transcription").unwrap())
                .unwrap()
                .descriptor()
                .family(),
            ModelFamily::Transcription
        );
        assert!(provider.responses_registration().is_some());
    }

    #[test]
    fn official_endpoint_rejects_unauthenticated_configuration() {
        let error = GroqProvider::builder(GroqCredential::unauthenticated())
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            GroqConfigError::OfficialEndpointRequiresCredential
        ));
    }

    #[test]
    fn credential_debug_never_exposes_material() {
        let debug = format!("{:?}", GroqCredential::api_key("groq-secret-canary"));
        assert!(!debug.contains("groq-secret-canary"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn support_manifest_exposes_exact_language_audio_claims() {
        let provider = GroqProvider::builder(GroqCredential::api_key("test-key"))
            .build()
            .unwrap();
        let manifest = provider.support_manifest();

        assert_eq!(manifest.provider_id().as_str(), PROVIDER_ID);
        assert_eq!(manifest.profiles().len(), 3);
        let transcription = manifest
            .profiles()
            .iter()
            .find(|profile| {
                profile
                    .verified_claims()
                    .is_some_and(|claims| claims[0].scope().family() == ModelFamily::Transcription)
            })
            .unwrap();
        let claim = &transcription.verified_claims().unwrap()[0];
        assert_eq!(claim.fidelity(), VerifiedFidelity::Native);
        assert_eq!(claim.stability(), ApiStability::Stable);
        assert_eq!(claim.scope().api_mode().as_str(), TRANSCRIPTION_API_MODE_ID);
        assert_eq!(transcription.catalog().unwrap().iter().count(), 2);

        let speech = manifest
            .profiles()
            .iter()
            .find(|profile| {
                profile
                    .verified_claims()
                    .is_some_and(|claims| claims[0].scope().family() == ModelFamily::Speech)
            })
            .unwrap();
        let claim = &speech.verified_claims().unwrap()[0];
        assert_eq!(claim.fidelity(), VerifiedFidelity::Native);
        assert_eq!(claim.stability(), ApiStability::Stable);
        assert_eq!(claim.scope().api_mode().as_str(), SPEECH_API_MODE_ID);
        assert_eq!(speech.catalog().unwrap().iter().count(), 2);
    }

    #[test]
    fn custom_endpoint_support_manifest_remains_generic() {
        let provider = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:9/v1").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap();
        let model = provider.language("future-model").unwrap();
        let replay_domain = model
            .descriptor()
            .replay_domain()
            .expect("custom replay domain");

        assert!(
            provider
                .support_manifest()
                .profiles()
                .iter()
                .all(|profile| profile.generic_claims().is_some())
        );
        assert!(!replay_domain.audience().is_official());
        assert_eq!(replay_domain.audience().id().as_str(), "test-relay");
    }

    #[test]
    fn custom_endpoint_requires_custom_replay_domain() {
        let endpoint = EndpointConfig::local_explicit("http://127.0.0.1:9/v1").unwrap();
        let missing = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_err();
        assert!(matches!(
            missing,
            GroqConfigError::CustomEndpointRequiresReplayDomain
        ));

        let wrong_audience = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("groq-public-api").unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(
            wrong_audience,
            GroqConfigError::ReplayAudienceMismatch
        ));
    }

    #[test]
    fn official_endpoint_rejects_custom_replay_audience() {
        let error = GroqProvider::builder(GroqCredential::api_key("test-key"))
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(error, GroqConfigError::ReplayAudienceMismatch));
    }

    #[test]
    fn caller_supplied_official_policy_does_not_gain_official_identity() {
        let endpoint = EndpointConfig::official(
            "https://relay.example/v1",
            OfficialOrigin::new("https://relay.example").unwrap(),
        )
        .unwrap();

        let missing = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_err();
        assert!(matches!(
            missing,
            GroqConfigError::CustomEndpointRequiresReplayDomain
        ));

        let mismatched = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN_ID).unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(
            mismatched,
            GroqConfigError::ReplayAudienceMismatch
        ));

        let provider = GroqProvider::builder(GroqCredential::unauthenticated())
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap();
        assert!(
            provider
                .support_manifest()
                .profiles()
                .iter()
                .all(|profile| profile.generic_claims().is_some())
        );
        assert!(
            !provider
                .language("future-model")
                .unwrap()
                .descriptor()
                .replay_domain()
                .expect("replay domain")
                .audience()
                .is_official()
        );
    }
}
