use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use siumai_core::{
    InvalidId, Model, ModelId, ModelLookupError, Provider, ProviderInstanceId, ProviderOptionError,
    ProviderOptionSelection, ProviderOptions, ProviderRegistration, ProviderRegistrationError,
    ProviderScope, SpeechModelProvider, TranscriptionModelProvider, TypedProviderOptions,
};
use siumai_transport::{
    EndpointConfig, EndpointError, EndpointPolicy, OfficialOrigin, ProviderTransport,
    TransportConfigError, TransportLimits,
};
use thiserror::Error;

use crate::credential::{DeepgramCredential, DeepgramCredentialError};
use crate::model::DeepgramTranscriptionModel;
use crate::options::DeepgramTranscriptionOptions;
use crate::profile::{DeepgramProfile, DeepgramProfileError};
use crate::speech::DeepgramSpeechModel;

const DEEPGRAM_ORIGIN: &str = "https://api.deepgram.com";

/// Long-lived configured Deepgram provider sharing transport and auth state.
#[derive(Clone)]
pub struct DeepgramProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
    pub(crate) speech_runtime: Arc<DeepgramSpeechRuntime>,
    registration: ProviderRegistration,
    profile: DeepgramProfile,
}

impl DeepgramProvider {
    pub fn builder(credential: DeepgramCredential) -> DeepgramProviderBuilder {
        DeepgramProviderBuilder::new(credential)
    }

    pub fn from_api_key(api_key: impl Into<String>) -> Result<Self, DeepgramConfigError> {
        Self::builder(DeepgramCredential::api_key(api_key)).build()
    }

    /// Create a lightweight transcription model from a textual open model ID.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepgramTranscriptionModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_transcription_model(model))
    }

    /// Construct the canonical transcription family handle.
    pub fn transcription_model(
        &self,
        model: ModelId,
    ) -> Result<DeepgramTranscriptionModel, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }

    pub fn default_transcription_model(
        &self,
    ) -> Result<DeepgramTranscriptionModel, ModelLookupError> {
        self.transcription(crate::models::DEFAULT_TRANSCRIPTION)
    }

    /// Create a lightweight buffered Aura speech model from an open model ID.
    pub fn speech(
        &self,
        model: impl Into<String>,
    ) -> Result<DeepgramSpeechModel, ModelLookupError> {
        self.speech_model(ModelId::new(model.into())?)
    }

    pub fn speech_model(&self, model: ModelId) -> Result<DeepgramSpeechModel, ModelLookupError> {
        Ok(DeepgramSpeechModel::new(self.speech_runtime.clone(), model))
    }

    pub fn default_speech_model(&self) -> Result<DeepgramSpeechModel, ModelLookupError> {
        self.speech(crate::models::DEFAULT_SPEECH)
    }

    pub fn registration(&self) -> ProviderRegistration {
        self.registration.clone()
    }

    pub fn profile(&self) -> &DeepgramProfile {
        &self.profile
    }

    fn create_transcription_model(&self, model: ModelId) -> DeepgramTranscriptionModel {
        DeepgramTranscriptionModel::new(self.runtime.clone(), model)
    }
}

impl Provider for DeepgramProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.scope.provider_id()
    }
}

impl TranscriptionModelProvider for DeepgramProvider {
    type Model = DeepgramTranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }
}

impl SpeechModelProvider for DeepgramProvider {
    type Model = DeepgramSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        DeepgramProvider::speech_model(self, model)
    }
}

impl fmt::Debug for DeepgramProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepgramProvider")
            .field("scope", &self.runtime.scope)
            .field("speech_scope", &self.speech_runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

pub struct DeepgramProviderBuilder {
    credential: DeepgramCredential,
    endpoint: Option<EndpointConfig>,
    provider_selected_endpoint: bool,
    limits: TransportLimits,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    default_options: DeepgramTranscriptionOptions,
}

impl DeepgramProviderBuilder {
    fn new(credential: DeepgramCredential) -> Self {
        Self {
            credential,
            endpoint: None,
            provider_selected_endpoint: true,
            limits: TransportLimits::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            default_options: DeepgramTranscriptionOptions::new(),
        }
    }

    /// Select a statically validated endpoint, including an explicit local grant for tests.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Some(endpoint);
        self.provider_selected_endpoint = false;
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
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

    pub fn with_default_options(mut self, options: DeepgramTranscriptionOptions) -> Self {
        self.default_options = options;
        self
    }

    /// Validate static settings and create one shared provider runtime without I/O.
    pub fn build(self) -> Result<DeepgramProvider, DeepgramConfigError> {
        self.credential.validate()?;
        self.default_options.validate()?;
        let default_options = ProviderOptions::typed(&self.default_options)?;
        let endpoint = match self.endpoint {
            Some(endpoint) => endpoint,
            None => official_endpoint()?,
        };
        if self.credential.is_unauthenticated()
            && matches!(endpoint.policy(), EndpointPolicy::Official(_))
        {
            return Err(DeepgramConfigError::OfficialEndpointRequiresCredential);
        }
        let verified_endpoint = self.provider_selected_endpoint;
        let profile = if verified_endpoint {
            DeepgramProfile::current()?
        } else {
            DeepgramProfile::custom()?
        };

        let mut transport = ProviderTransport::builder(endpoint)
            .with_auth(self.credential.into_auth())
            .with_limits(self.limits);
        if let Some(timeout) = self.connect_timeout {
            transport = transport.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            transport = transport.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            transport = transport.with_read_timeout(timeout);
        }
        let transport = transport.build()?;
        let instance_id = ProviderInstanceId::new();
        let runtime = Arc::new(ProviderRuntime {
            scope: profile.scope(),
            instance_id: instance_id.clone(),
            transport: transport.clone(),
            default_options,
        });
        let speech_runtime = Arc::new(DeepgramSpeechRuntime {
            scope: profile.speech_scope(),
            instance_id,
            transport,
        });
        let transcription_registration =
            ProviderRegistration::from_transcription(runtime.scope.clone(), {
                let runtime = runtime.clone();
                Arc::new(move |model| {
                    Ok(
                        Arc::new(DeepgramTranscriptionModel::new(runtime.clone(), model))
                            as Arc<dyn siumai_core::TranscriptionModel>,
                    )
                })
            });
        let speech_registration =
            ProviderRegistration::from_speech(speech_runtime.scope.clone(), {
                let runtime = speech_runtime.clone();
                Arc::new(move |model| {
                    Ok(Arc::new(DeepgramSpeechModel::new(runtime.clone(), model))
                        as Arc<dyn siumai_core::SpeechModel>)
                })
            });
        Ok(DeepgramProvider {
            runtime,
            speech_runtime,
            registration: transcription_registration.merge(speech_registration)?,
            profile,
        })
    }
}

impl fmt::Debug for DeepgramProviderBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepgramProviderBuilder")
            .field("credential", &self.credential)
            .field("endpoint", &self.endpoint)
            .field(
                "provider_selected_endpoint",
                &self.provider_selected_endpoint,
            )
            .field("limits", &self.limits)
            .field("connect_timeout", &self.connect_timeout)
            .field("call_timeout", &self.call_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("default_options", &self.default_options)
            .finish()
    }
}

pub(crate) struct ProviderRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
    default_options: ProviderOptions,
}

pub(crate) struct DeepgramSpeechRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
}

impl ProviderRuntime {
    pub(crate) fn options<M: Model + ?Sized>(
        &self,
        model: &M,
        call: &siumai_core::CallOptions,
    ) -> Result<DeepgramTranscriptionOptions, ProviderOptionError> {
        let selection = call.provider_options_for(model)?;
        merge_options(&self.default_options, &selection)
    }
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("default_options", &self.default_options)
            .finish()
    }
}

fn merge_options(
    defaults: &ProviderOptions,
    selection: &ProviderOptionSelection<'_>,
) -> Result<DeepgramTranscriptionOptions, ProviderOptionError> {
    let mut merged = defaults.value().clone();
    for options in selection.typed() {
        decode_options(options.value())?.validate()?;
        merged.extend(options.value().clone());
    }
    if selection.raw_override().is_some() {
        return Err(ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: "Deepgram prerecorded transcription only accepts typed provider options"
                .to_string(),
        });
    }
    let options = decode_options(&merged)?;
    options.validate()?;
    Ok(options)
}

fn decode_options(
    options: &Map<String, Value>,
) -> Result<DeepgramTranscriptionOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.clone())).map_err(|error| {
        ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: error.to_string(),
        }
    })
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    let origin = OfficialOrigin::new(DEEPGRAM_ORIGIN)?;
    EndpointConfig::official(DEEPGRAM_ORIGIN, origin)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DeepgramConfigError {
    #[error("invalid Deepgram identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Deepgram support profile: {0}")]
    Profile(#[from] DeepgramProfileError),
    #[error("invalid Deepgram endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Deepgram transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Deepgram credential: {0}")]
    Credential(#[from] DeepgramCredentialError),
    #[error("invalid Deepgram default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("invalid Deepgram provider registration: {0}")]
    Registration(#[from] ProviderRegistrationError),
    #[error("the official Deepgram endpoint requires an API key")]
    OfficialEndpointRequiresCredential,
}

#[cfg(test)]
mod tests {
    use siumai_core::{ApiStability, CallOptions, Model, ModelFamily, VerifiedFidelity};

    use super::*;

    #[test]
    fn build_is_static_and_models_share_one_runtime() {
        let provider = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
        let runtime = Arc::as_ptr(&provider.runtime);
        for index in 0..1_000 {
            let model = provider
                .transcription(format!("future-model-{index}"))
                .unwrap();
            assert_eq!(Arc::as_ptr(&model.runtime), runtime);
            assert_eq!(model.family(), ModelFamily::Transcription);
        }
    }

    #[test]
    fn direct_and_registered_models_have_identical_identity() {
        let provider = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
        let direct = provider.transcription("nova-3").unwrap();
        let erased = provider
            .registration()
            .transcription_model(ModelId::new("nova-3").unwrap())
            .unwrap();

        assert_eq!(direct.descriptor(), erased.descriptor());
        assert_eq!(
            direct.descriptor().instance_id(),
            erased.descriptor().instance_id()
        );
        assert_eq!(direct.descriptor().protocol(), Some("deepgram-prerecorded"));
        assert_eq!(direct.descriptor().api_mode(), Some("prerecorded"));
    }

    #[test]
    fn configured_instance_identity_spans_families_and_is_fresh_per_build() {
        let first = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
        let second = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();

        let transcription = first.transcription("nova-3").unwrap();
        let speech = first.default_speech_model().unwrap();
        let separate = second.transcription("nova-3").unwrap();

        assert_eq!(
            transcription.descriptor().instance_id(),
            speech.descriptor().instance_id()
        );
        assert_ne!(
            transcription.descriptor().instance_id(),
            separate.descriptor().instance_id()
        );
    }

    #[test]
    fn official_profile_exposes_current_prerecorded_and_aura_models() {
        let provider = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
        let profile = provider.profile().provider_profile();
        let claims = profile.verified_claims().unwrap();
        assert_eq!(claims.len(), 2);
        assert!(claims.iter().all(|claim| {
            claim.fidelity() == VerifiedFidelity::Native
                && claim.stability() == ApiStability::Stable
        }));
        assert_eq!(
            profile.catalog().unwrap().iter().count(),
            crate::models::CURRENT_TRANSCRIPTION_MODELS.len()
                + crate::models::CURRENT_SPEECH_MODELS.len()
        );
    }

    #[test]
    fn invalid_default_options_fail_synchronously() {
        let error = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .with_default_options(DeepgramTranscriptionOptions::new().with_utt_split(f64::NAN))
            .build()
            .unwrap_err();
        assert!(matches!(error, DeepgramConfigError::Options(_)));
    }

    #[test]
    fn official_endpoint_rejects_unauthenticated_configuration() {
        assert!(matches!(
            DeepgramProvider::builder(DeepgramCredential::unauthenticated()).build(),
            Err(DeepgramConfigError::OfficialEndpointRequiresCredential)
        ));
    }

    #[test]
    fn call_options_override_defaults_and_unknown_raw_fields_are_rejected() {
        let provider = DeepgramProvider::builder(DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
        let model = provider.transcription("nova-3").unwrap();
        let call = CallOptions::default()
            .with_provider_options_for(
                &model,
                &DeepgramTranscriptionOptions::new()
                    .with_diarize_model(crate::DeepgramDiarizeModel::V1),
            )
            .unwrap();
        assert_eq!(
            provider
                .runtime
                .options(&model, &call)
                .unwrap()
                .diarize_model(),
            Some(crate::DeepgramDiarizeModel::V1)
        );

        let raw = CallOptions::default()
            .with_raw_provider_options_for(&model, serde_json::json!({"unknownFutureField": true}))
            .unwrap();
        let error = provider.runtime.options(&model, &raw).unwrap_err();
        assert!(matches!(
            error,
            ProviderOptionError::Rejected { path, reason }
                if path == "$" && reason.contains("typed provider options")
        ));
    }
}
