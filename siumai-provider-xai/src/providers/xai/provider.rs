//! Rust-first configured xAI language provider.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use serde::Serialize;
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CallOptions, CatalogError, Error, GenericSupportClaim, ImageModel,
    ImageModelProvider, InvalidId, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelCatalog, ModelDescriptor, ModelFamily, ModelId,
    ModelLifecycle, ModelLookupError, ModelOperation, ModelProfile, NativeSupportScope,
    NativeSurfaceId, NativeSurfaceKind, NativeVerificationEvidence, OfficialSource, PlatformId,
    ProfileError, ProfileId, ProtocolContractId, ProtocolId, Provider, ProviderId,
    ProviderInstanceId, ProviderOptionError, ProviderProfile, ProviderRegistration,
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

use crate::provider_options::{XaiChatOptions, XaiResponsesOptions};

use super::files::{FILES_SOURCE, FILES_VERIFIED_ON, XaiFiles};
use super::language::{
    DEFAULT_BASE_URL, OFFICIAL_ORIGIN, PLATFORM_ID, PROVIDER_ID, XaiProfileError, profile,
};
use super::media::{
    IMAGE_API_MODE_ID, IMAGE_PROTOCOL_ID, IMAGE_SOURCE, MEDIA_VERIFIED_ON, SPEECH_API_MODE_ID,
    SPEECH_PROTOCOL_ID, SPEECH_SOURCE, TRANSCRIPTION_API_MODE_ID, TRANSCRIPTION_PROTOCOL_ID,
    TRANSCRIPTION_SOURCE, XaiImageModel, XaiImageRuntime, XaiSpeechModel, XaiSpeechRuntime,
    XaiTranscriptionModel, XaiTranscriptionRuntime,
};
use super::models;
use super::video::{VIDEO_SOURCE, VIDEO_VERIFIED_ON, XaiVideoJobs};

const OFFICIAL_REPLAY_DOMAIN_ID: &str = "xai-public-api";

/// xAI language endpoint selected by a lightweight model handle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum XaiLanguageApi {
    #[default]
    Responses,
    ChatCompletions,
}

impl From<XaiLanguageApi> for OpenAiCompatibleApiMode {
    fn from(value: XaiLanguageApi) -> Self {
        match value {
            XaiLanguageApi::Responses => Self::Responses,
            XaiLanguageApi::ChatCompletions => Self::ChatCompletions,
        }
    }
}

/// xAI credential used by the configured language provider.
#[derive(Clone)]
pub struct XaiCredential(OpenAiCompatibleCredential);

impl XaiCredential {
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

impl fmt::Debug for XaiCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("XaiCredential")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// Long-lived, model-independent xAI language provider.
#[derive(Clone)]
pub struct XaiProvider {
    language: OpenAiCompatibleProvider,
    responses_registration: ProviderRegistration,
    chat_registration: ProviderRegistration,
    image: Arc<XaiImageRuntime>,
    speech: Arc<XaiSpeechRuntime>,
    transcription: Arc<XaiTranscriptionRuntime>,
    files: XaiFiles,
    video_jobs: XaiVideoJobs,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl XaiProvider {
    pub fn builder(credential: XaiCredential) -> XaiProviderBuilder {
        XaiProviderBuilder::new(credential)
    }

    /// Return the provider-owned support claims and exact advisory model catalog.
    ///
    /// The official xAI endpoint exposes verified Chat Completions and Responses claims. Custom
    /// endpoints expose generic compatibility claims instead of inheriting xAI's evidence.
    pub fn profile(&self) -> &ProviderProfile {
        self.language.profile().provider_profile()
    }

    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        &self.support_manifest
    }

    /// Create a lightweight model for the recommended Responses API.
    pub fn language(&self, model: impl Into<String>) -> Result<XaiLanguageModel, ModelLookupError> {
        self.responses(model)
    }

    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<XaiLanguageModel, ModelLookupError> {
        self.language_for(XaiLanguageApi::Responses, model)
    }

    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<XaiLanguageModel, ModelLookupError> {
        self.language_for(XaiLanguageApi::ChatCompletions, model)
    }

    pub fn language_for(
        &self,
        api: XaiLanguageApi,
        model: impl Into<String>,
    ) -> Result<XaiLanguageModel, ModelLookupError> {
        self.language
            .language_for(api.into(), model)
            .map(XaiLanguageModel)
    }

    pub fn image(&self, model: impl Into<String>) -> Result<XaiImageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_image_model(model))
    }

    pub fn default_image_model(&self) -> Result<XaiImageModel, ModelLookupError> {
        self.image(models::recommended::IMAGE)
    }

    /// Create the fixed xAI TTS endpoint handle.
    pub fn speech(&self) -> XaiSpeechModel {
        self.create_speech_model(ModelId::new(models::speech::TTS).expect("valid xAI TTS handle"))
    }

    /// Create the fixed xAI final-result STT endpoint handle.
    pub fn transcription(&self) -> XaiTranscriptionModel {
        self.create_transcription_model(
            ModelId::new(models::transcription::STT).expect("valid xAI STT handle"),
        )
    }

    pub fn files(&self) -> XaiFiles {
        self.files.clone()
    }

    /// Return the provider-owned xAI asynchronous video-generation resource.
    pub fn video_jobs(&self) -> XaiVideoJobs {
        self.video_jobs.clone()
    }

    /// Concrete Registry registration for the recommended Responses route.
    pub fn registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn responses_registration(&self) -> ProviderRegistration {
        self.responses_registration.clone()
    }

    pub fn chat_completions_registration(&self) -> ProviderRegistration {
        self.chat_registration.clone()
    }

    pub fn registration_for(&self, api: XaiLanguageApi) -> ProviderRegistration {
        match api {
            XaiLanguageApi::Responses => self.responses_registration(),
            XaiLanguageApi::ChatCompletions => self.chat_completions_registration(),
        }
    }

    fn create_image_model(&self, model: ModelId) -> XaiImageModel {
        XaiImageModel::new(self.image.clone(), model)
    }

    fn create_speech_model(&self, model: ModelId) -> XaiSpeechModel {
        XaiSpeechModel::new(self.speech.clone(), model)
    }

    fn create_transcription_model(&self, model: ModelId) -> XaiTranscriptionModel {
        XaiTranscriptionModel::new(self.transcription.clone(), model)
    }
}

impl Provider for XaiProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for XaiProvider {
    type Model = XaiLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.language.language_model(model).map(XaiLanguageModel)
    }
}

impl ImageModelProvider for XaiProvider {
    type Model = XaiImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl SpeechModelProvider for XaiProvider {
    type Model = XaiSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }
}

impl TranscriptionModelProvider for XaiProvider {
    type Model = XaiTranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }
}

impl fmt::Debug for XaiProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiProvider")
            .field("provider_id", self.provider_id())
            .field("image_scope", &self.image.scope)
            .field("speech_scope", &self.speech.scope)
            .field("transcription_scope", &self.transcription.scope)
            .finish()
    }
}

/// Synchronous xAI provider builder.
pub struct XaiProviderBuilder {
    credential: XaiCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    provider_selected_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: XaiChatOptions,
    responses_defaults: XaiResponsesOptions,
}

impl XaiProviderBuilder {
    fn new(credential: XaiCredential) -> Self {
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
            chat_defaults: XaiChatOptions::default(),
            responses_defaults: XaiResponsesOptions::default(),
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

    pub fn with_local_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::local_explicit(base_url);
        self.provider_selected_endpoint = false;
        self
    }

    /// Bind provider-native history to a non-secret replay domain.
    ///
    /// Caller-controlled endpoints require an explicit custom audience. The provider-selected
    /// endpoint uses xAI's official audience by default, but callers may add a material account
    /// boundary with [`ReplayDomain::with_caller_scope`].
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

    pub fn with_chat_defaults(mut self, defaults: XaiChatOptions) -> Self {
        self.chat_defaults = defaults;
        self
    }

    pub fn with_responses_defaults(mut self, defaults: XaiResponsesOptions) -> Self {
        self.responses_defaults = defaults;
        self
    }

    pub fn build(self) -> Result<XaiProvider, XaiConfigError> {
        self.chat_defaults
            .validate()
            .map_err(XaiConfigError::DefaultProviderOptions)?;
        self.responses_defaults
            .validate()
            .map_err(XaiConfigError::DefaultProviderOptions)?;
        self.credential.0.validate_static()?;

        let endpoint = self.endpoint?;
        let verified_endpoint = self.provider_selected_endpoint;
        let replay_domain = replay_domain_for_endpoint(self.replay_domain, verified_endpoint)?;
        let auth = self.credential.0.into_auth();
        let instance_id = ProviderInstanceId::new();
        let language_profile = profile(endpoint.clone(), replay_domain.clone(), verified_endpoint)?;
        let mut builder =
            OpenAiCompatibleProvider::builder_with_auth(language_profile, auth.clone())
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
            builder =
                builder.with_default_option(OpenAiCompatibleApiMode::ChatCompletions, name, value);
        }
        for (name, value) in option_map(&self.responses_defaults)? {
            builder = builder.with_default_option(OpenAiCompatibleApiMode::Responses, name, value);
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

        let image_scope = Arc::new(media_scope(
            &endpoint,
            replay_domain.clone(),
            verified_endpoint,
            ModelFamily::Image,
            IMAGE_PROTOCOL_ID,
            IMAGE_API_MODE_ID,
        )?);
        let speech_scope = Arc::new(media_scope(
            &endpoint,
            replay_domain.clone(),
            verified_endpoint,
            ModelFamily::Speech,
            SPEECH_PROTOCOL_ID,
            SPEECH_API_MODE_ID,
        )?);
        let transcription_scope = Arc::new(media_scope(
            &endpoint,
            replay_domain,
            verified_endpoint,
            ModelFamily::Transcription,
            TRANSCRIPTION_PROTOCOL_ID,
            TRANSCRIPTION_API_MODE_ID,
        )?);
        let image_profile = media_profile(
            &image_scope,
            verified_endpoint,
            "xai-image",
            IMAGE_SOURCE,
            "xai-image-generations-2026-08",
            ModelOperation::GenerateImage,
            models::image::HINTS,
        )?;
        let speech_profile = media_profile(
            &speech_scope,
            verified_endpoint,
            "xai-speech",
            SPEECH_SOURCE,
            "xai-tts-2026-08",
            ModelOperation::SynthesizeSpeech,
            &[models::speech::TTS],
        )?;
        let transcription_profile = media_profile(
            &transcription_scope,
            verified_endpoint,
            "xai-transcription",
            TRANSCRIPTION_SOURCE,
            "xai-stt-2026-08",
            ModelOperation::Transcribe,
            &[models::transcription::STT],
        )?;

        let language = builder.build()?;
        let language_responses_registration = language.responses_registration().ok_or(
            XaiConfigError::MissingLanguageRegistration(XaiLanguageApi::Responses),
        )?;
        let language_chat_registration = language.chat_completions_registration().ok_or(
            XaiConfigError::MissingLanguageRegistration(XaiLanguageApi::ChatCompletions),
        )?;
        let media_transport = media_transport.build()?;
        let image = Arc::new(XaiImageRuntime::new(
            instance_id.clone(),
            image_scope,
            media_transport.clone(),
            verified_endpoint,
        ));
        let speech = Arc::new(XaiSpeechRuntime::new(
            instance_id.clone(),
            speech_scope,
            media_transport.clone(),
            verified_endpoint,
        ));
        let transcription = Arc::new(XaiTranscriptionRuntime::new(
            instance_id,
            transcription_scope,
            media_transport.clone(),
            verified_endpoint,
        ));
        let video_jobs = XaiVideoJobs::new(media_transport.clone());
        let files = XaiFiles::new(media_transport);
        let image_registration =
            ProviderRegistration::from_image(image.scope.clone(), image.policy.clone(), {
                let runtime = image.clone();
                Arc::new(move |model| {
                    Ok(Arc::new(XaiImageModel::new(runtime.clone(), model)) as Arc<dyn ImageModel>)
                })
            });
        let speech_registration =
            ProviderRegistration::from_speech(speech.scope.clone(), speech.policy.clone(), {
                let runtime = speech.clone();
                Arc::new(move |model| {
                    Ok(Arc::new(XaiSpeechModel::new(runtime.clone(), model))
                        as Arc<dyn SpeechModel>)
                })
            });
        let transcription_registration = ProviderRegistration::from_transcription(
            transcription.scope.clone(),
            transcription.policy.clone(),
            {
                let runtime = transcription.clone();
                Arc::new(move |model| {
                    Ok(Arc::new(XaiTranscriptionModel::new(runtime.clone(), model))
                        as Arc<dyn TranscriptionModel>)
                })
            },
        );
        let media_registration = image_registration
            .merge(speech_registration)?
            .merge(transcription_registration)?;
        let responses_registration =
            language_responses_registration.merge(media_registration.clone())?;
        let chat_registration = language_chat_registration.merge(media_registration)?;
        let native_claims = if verified_endpoint {
            vec![files_support_claim()?, video_support_claim()?]
        } else {
            Vec::new()
        };
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            ProviderId::new(PROVIDER_ID)?,
            [
                language.profile().provider_profile().clone(),
                image_profile,
                speech_profile,
                transcription_profile,
            ],
            native_claims,
        )?);

        Ok(XaiProvider {
            language,
            responses_registration,
            chat_registration,
            image,
            speech,
            transcription,
            files,
            video_jobs,
            support_manifest,
        })
    }
}

/// Lightweight xAI language model backed by the configured compatible runtime.
#[derive(Clone)]
pub struct XaiLanguageModel(OpenAiCompatibleLanguageModel);

impl Model for XaiLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.0.descriptor()
    }
}

#[async_trait]
impl LanguageModel for XaiLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
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

impl fmt::Debug for XaiLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiLanguageModel")
            .field("descriptor", self.descriptor())
            .finish()
    }
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn replay_domain_for_endpoint(
    configured: Option<ReplayDomain>,
    verified_endpoint: bool,
) -> Result<ReplayDomain, XaiConfigError> {
    let replay_domain = match (configured, verified_endpoint) {
        (Some(replay_domain), _) => replay_domain,
        (None, true) => ReplayDomain::official(ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN_ID)?),
        (None, false) => return Err(XaiConfigError::CustomEndpointRequiresReplayDomain),
    };
    if replay_domain.audience().is_official() != verified_endpoint {
        return Err(XaiConfigError::ReplayAudienceMismatch);
    }
    Ok(replay_domain)
}

fn media_scope(
    endpoint: &EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
    _family: ModelFamily,
    protocol: &str,
    api_mode: &str,
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
        .with_protocol(ProtocolId::new(protocol)?)
        .with_api_mode(ApiModeId::new(api_mode)?)
        .with_replay_domain(replay_domain))
}

#[allow(clippy::too_many_arguments)]
fn media_profile(
    scope: &ProviderScope,
    verified_endpoint: bool,
    profile_id: &str,
    source: &str,
    contract: &str,
    operation: ModelOperation,
    models: &[&str],
) -> Result<ProviderProfile, XaiConfigError> {
    let family = operation.family();
    let support_scope = SupportScope::new(
        scope.provider_id().clone(),
        scope
            .platform()
            .cloned()
            .ok_or(XaiConfigError::IncompleteMediaScope)?,
        family,
        scope
            .protocol()
            .cloned()
            .ok_or(XaiConfigError::IncompleteMediaScope)?,
        scope
            .api_mode()
            .cloned()
            .ok_or(XaiConfigError::IncompleteMediaScope)?,
    );
    let profile_id = ProfileId::new(profile_id)?;
    if !verified_endpoint {
        return Ok(ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(support_scope, ApiStability::Experimental),
        ));
    }
    let verified_at =
        VerificationDate::new(NaiveDate::parse_from_str(MEDIA_VERIFIED_ON, "%Y-%m-%d")?);
    let evidence = VerificationEvidence::new(
        OfficialSource::new(source)?,
        verified_at,
        ProtocolContractId::new(contract)?,
    );
    let catalog = ModelCatalog::new(
        models
            .iter()
            .map(|model| {
                Ok(ModelProfile::new(
                    ModelId::new(*model)?,
                    support_scope.clone(),
                    [operation],
                    ModelLifecycle::Active,
                    evidence.clone(),
                )?)
            })
            .collect::<Result<Vec<_>, XaiConfigError>>()?,
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

fn files_support_claim() -> Result<VerifiedNativeSupportClaim, XaiConfigError> {
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            ProviderId::new(PROVIDER_ID)?,
            PlatformId::new(PLATFORM_ID)?,
            NativeSurfaceKind::Resource,
            NativeSurfaceId::new("files-lifecycle")?,
        ),
        VerifiedFidelity::Native,
        ApiStability::Stable,
        NativeVerificationEvidence::new(
            OfficialSource::new(FILES_SOURCE)?,
            VerificationDate::new(NaiveDate::parse_from_str(FILES_VERIFIED_ON, "%Y-%m-%d")?),
        ),
    ))
}

fn video_support_claim() -> Result<VerifiedNativeSupportClaim, XaiConfigError> {
    Ok(VerifiedNativeSupportClaim::new(
        NativeSupportScope::surface(
            ProviderId::new(PROVIDER_ID)?,
            PlatformId::new(PLATFORM_ID)?,
            NativeSurfaceKind::Job,
            NativeSurfaceId::new("video-generation-jobs")?,
        ),
        VerifiedFidelity::Native,
        ApiStability::Stable,
        NativeVerificationEvidence::new(
            OfficialSource::new(VIDEO_SOURCE)?,
            VerificationDate::new(NaiveDate::parse_from_str(VIDEO_VERIFIED_ON, "%Y-%m-%d")?),
        ),
    ))
}

fn option_map(options: &impl Serialize) -> Result<Map<String, Value>, XaiConfigError> {
    match serde_json::to_value(options)? {
        Value::Object(values) => Ok(values),
        _ => Err(XaiConfigError::InvalidDefaultsShape),
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum XaiConfigError {
    #[error("invalid xAI identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid xAI endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid xAI language profile: {0}")]
    Profile(#[from] XaiProfileError),
    #[error("invalid xAI credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("invalid xAI compatible runtime: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid xAI native transport: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid xAI media profile: {0}")]
    MediaProfile(#[from] ProfileError),
    #[error("invalid xAI media catalog: {0}")]
    MediaCatalog(#[from] CatalogError),
    #[error("invalid xAI support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid xAI provider registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("invalid xAI support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("xAI default options could not be serialized: {0}")]
    DefaultOptions(#[from] serde_json::Error),
    #[error("xAI default options must serialize to an object")]
    InvalidDefaultsShape,
    #[error("xAI typed provider defaults are invalid: {0}")]
    DefaultProviderOptions(ProviderOptionError),
    #[error("xAI profile is missing its required {0:?} language registration")]
    MissingLanguageRegistration(XaiLanguageApi),
    #[error("the configured xAI media scope is incomplete")]
    IncompleteMediaScope,
    #[error("a custom xAI endpoint requires an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured xAI endpoint ownership")]
    ReplayAudienceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::xai::models;
    use siumai_core::{ModelAdvisory, ModelFamily, ModelOperation, SupportState};

    #[test]
    fn model_construction_is_synchronous_and_future_model_safe() {
        let provider = XaiProvider::builder(XaiCredential::api_key("test-key"))
            .build()
            .expect("build xAI provider");
        let model = provider
            .language("future-grok-model")
            .expect("future model remains callable");
        assert_eq!(model.descriptor().model().as_str(), "future-grok-model");
        assert_eq!(
            model.descriptor().api_mode(),
            Some(siumai_protocol_openai::responses::API_MODE_ID)
        );
        let replay_domain = model
            .descriptor()
            .replay_domain()
            .expect("official replay domain");
        let chat = provider
            .chat_completions("future-grok-model")
            .expect("future Chat model remains callable");
        assert!(replay_domain.audience().is_official());
        assert_eq!(
            replay_domain.audience().id().as_str(),
            OFFICIAL_REPLAY_DOMAIN_ID
        );
        assert_eq!(chat.descriptor().replay_domain(), Some(replay_domain));
    }

    #[test]
    fn configured_provider_native_models_share_one_instance_capability() {
        let first = XaiProvider::builder(XaiCredential::api_key("first-key"))
            .build()
            .unwrap();
        let second = XaiProvider::builder(XaiCredential::api_key("second-key"))
            .build()
            .unwrap();

        let image = first.image("future-image").unwrap();
        let language = first.responses("future-language").unwrap();
        let speech = first.speech();
        let transcription = first.transcription();
        let other = second.image("future-image").unwrap();

        assert_eq!(
            image.descriptor().instance_id(),
            language.descriptor().instance_id()
        );
        assert_eq!(
            image.descriptor().instance_id(),
            speech.descriptor().instance_id()
        );
        assert_eq!(
            image.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_ne!(
            image.descriptor().instance_id(),
            other.descriptor().instance_id()
        );
    }

    #[test]
    fn credential_debug_is_redacted() {
        let debug = format!("{:?}", XaiCredential::api_key("secret-value"));
        assert!(!debug.contains("secret-value"));
    }

    #[test]
    fn official_profile_classifies_exact_ids_aliases_and_mode_specific_models() {
        let provider = XaiProvider::builder(XaiCredential::api_key("test-key"))
            .build()
            .expect("build xAI provider");

        for registration in [
            provider.chat_completions_registration(),
            provider.responses_registration(),
        ] {
            let exact = registration.evaluate(
                ModelId::new(models::language::GROK_4_5).expect("exact model id"),
                ModelOperation::Generate,
            );
            assert_eq!(exact.state(), &SupportState::Supported);
            assert!(exact.advisories().is_empty());

            let alias = registration.evaluate(
                ModelId::new(models::language::GROK_LATEST).expect("rolling alias"),
                ModelOperation::Stream,
            );
            assert_eq!(alias.state(), &SupportState::Supported);
            assert_eq!(alias.advisories(), &[ModelAdvisory::RollingAlias]);
        }

        let multi_agent =
            ModelId::new(models::language::GROK_4_20_MULTI_AGENT).expect("multi-agent model id");
        let chat = provider
            .chat_completions_registration()
            .evaluate(multi_agent.clone(), ModelOperation::Generate);
        assert_eq!(chat.state(), &SupportState::Unknown);
        assert_eq!(chat.advisories(), &[ModelAdvisory::UnknownModel]);

        let responses = provider
            .responses_registration()
            .evaluate(multi_agent, ModelOperation::Generate);
        assert_eq!(responses.state(), &SupportState::Supported);
        assert_eq!(responses.advisories(), &[ModelAdvisory::RollingAlias]);

        let future = provider.responses_registration().evaluate(
            ModelId::new("future-grok-model").expect("future model id"),
            ModelOperation::Generate,
        );
        assert_eq!(future.state(), &SupportState::Unknown);
        assert_eq!(future.advisories(), &[ModelAdvisory::UnknownModel]);
    }

    #[test]
    fn profile_accessor_exposes_both_verified_language_modes() {
        let provider = XaiProvider::builder(XaiCredential::api_key("test-key"))
            .build()
            .expect("build xAI provider");
        let profile = provider.profile();
        let claims = profile.verified_claims().expect("official xAI claims");

        assert_eq!(claims.len(), 2);
        assert!(
            claims
                .iter()
                .any(|claim| claim.scope().api_mode().as_str() == "chat-completions")
        );
        assert!(
            claims
                .iter()
                .any(|claim| claim.scope().api_mode().as_str() == "responses")
        );
        assert!(!profile.catalog().expect("official xAI catalog").is_empty());
        assert_eq!(provider.support_manifest().profiles().len(), 4);
        assert_eq!(provider.support_manifest().native_claims().len(), 2);
    }

    #[test]
    fn direct_and_registry_models_share_exact_language_scopes() {
        let provider = XaiProvider::builder(XaiCredential::api_key("test-key"))
            .build()
            .expect("build xAI provider");

        let model = ModelId::new("future-grok-language").expect("model id");
        let direct = provider
            .language_model(model.clone())
            .expect("direct language");
        let registered = provider
            .registration()
            .language_model(model)
            .expect("registered language");
        assert_eq!(direct.descriptor(), registered.descriptor());
        assert_eq!(
            provider.registration().scope(ModelFamily::Language),
            provider
                .responses_registration()
                .scope(ModelFamily::Language)
        );
        assert_ne!(
            provider.registration().scope(ModelFamily::Language),
            provider
                .chat_completions_registration()
                .scope(ModelFamily::Language)
        );

        let model = ModelId::new("future-grok-chat").expect("model id");
        let direct = provider
            .chat_completions(model.to_string())
            .expect("direct Chat Completions");
        let registered = provider
            .chat_completions_registration()
            .language_model(model)
            .expect("registered Chat Completions");
        assert_eq!(direct.descriptor(), registered.descriptor());
    }

    #[test]
    fn custom_endpoint_requires_custom_replay_domain() {
        let endpoint = EndpointConfig::local_explicit("http://127.0.0.1:9/v1").unwrap();
        let missing = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_err();
        assert!(matches!(
            missing,
            XaiConfigError::CustomEndpointRequiresReplayDomain
        ));

        let wrong_audience = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("xai-public-api").unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(
            wrong_audience,
            XaiConfigError::ReplayAudienceMismatch
        ));

        let provider = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_local_base_url("http://127.0.0.1:9/v1")
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap();
        let model = provider.language("future-grok-model").unwrap();
        let replay_domain = model
            .descriptor()
            .replay_domain()
            .expect("custom replay domain");
        assert!(!replay_domain.audience().is_official());
        assert_eq!(replay_domain.audience().id().as_str(), "test-relay");
    }

    #[test]
    fn official_endpoint_rejects_custom_replay_audience() {
        let error = XaiProvider::builder(XaiCredential::api_key("test-key"))
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(error, XaiConfigError::ReplayAudienceMismatch));
    }

    #[test]
    fn caller_supplied_official_policy_does_not_gain_official_identity() {
        let endpoint = EndpointConfig::official(
            "https://relay.example/v1",
            OfficialOrigin::new("https://relay.example").unwrap(),
        )
        .unwrap();

        let missing = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_err();
        assert!(matches!(
            missing,
            XaiConfigError::CustomEndpointRequiresReplayDomain
        ));

        let mismatched = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(endpoint.clone())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new(OFFICIAL_REPLAY_DOMAIN_ID).unwrap(),
            ))
            .build()
            .unwrap_err();
        assert!(matches!(mismatched, XaiConfigError::ReplayAudienceMismatch));

        let provider = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(endpoint)
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-relay").unwrap(),
            ))
            .build()
            .unwrap();
        assert!(provider.profile().generic_claims().is_some());
        assert!(provider.profile().verified_claims().is_none());
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
