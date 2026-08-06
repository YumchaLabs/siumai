use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use siumai_core::{
    InvalidId, ModelFamily, ModelId, ModelLookupError, Provider, ProviderOptionContext,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, ProviderRegistration, ProviderScope, SpeechLimits, SpeechModel,
    SpeechModelProvider, TypedProviderOptions,
};
use siumai_transport::{
    EndpointError, ProviderTransport, RetryPolicy, TransportConfigError, TransportLimits,
};
use thiserror::Error;

use super::credentials::{ElevenLabsCredential, ElevenLabsCredentialError};
use super::model::ElevenLabsSpeechModel;
use super::models;
use super::options::ElevenLabsSpeechOptions;
use super::policy::ElevenLabsModelPolicy;
use super::profile::ElevenLabsProfile;

/// A synchronously configured ElevenLabs native provider.
#[derive(Clone)]
pub struct ElevenLabsProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
}

impl ElevenLabsProvider {
    pub fn builder(
        profile: ElevenLabsProfile,
        credential: ElevenLabsCredential,
    ) -> ElevenLabsProviderBuilder {
        ElevenLabsProviderBuilder::new(profile, credential)
    }

    /// Create a lightweight speech model from a textual open model ID.
    pub fn speech(
        &self,
        model: impl Into<String>,
    ) -> Result<ElevenLabsSpeechModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_speech_model(model))
    }

    /// Construct the canonical speech family handle.
    pub fn speech_model(&self, model: ModelId) -> Result<ElevenLabsSpeechModel, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_speech(
            self.runtime.scope.clone(),
            self.runtime.policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(provider.create_speech_model(model)) as Arc<dyn SpeechModel>)
            }),
        )
    }

    pub fn profile(&self) -> &ElevenLabsProfile {
        &self.runtime.profile
    }

    fn create_speech_model(&self, model: ModelId) -> ElevenLabsSpeechModel {
        ElevenLabsSpeechModel::new(self.runtime.clone(), model)
    }
}

impl Provider for ElevenLabsProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.scope.provider_id()
    }
}

impl SpeechModelProvider for ElevenLabsProvider {
    type Model = ElevenLabsSpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_speech_model(model))
    }
}

impl fmt::Debug for ElevenLabsProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElevenLabsProvider")
            .field("scope", &self.runtime.scope)
            .field("profile", &self.runtime.profile)
            .finish()
    }
}

pub struct ElevenLabsProviderBuilder {
    profile: ElevenLabsProfile,
    credential: ElevenLabsCredential,
    transport_limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    default_voice: String,
    default_options: ElevenLabsSpeechOptions,
    speech_limits: Option<SpeechLimits>,
}

impl ElevenLabsProviderBuilder {
    fn new(profile: ElevenLabsProfile, credential: ElevenLabsCredential) -> Self {
        Self {
            profile,
            credential,
            transport_limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            default_voice: models::DEFAULT_VOICE.to_string(),
            default_options: ElevenLabsSpeechOptions::default(),
            speech_limits: None,
        }
    }

    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Self {
        self.transport_limits = limits;
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

    pub fn with_default_voice(mut self, voice: impl Into<String>) -> Self {
        self.default_voice = voice.into();
        self
    }

    pub fn with_default_options(mut self, options: ElevenLabsSpeechOptions) -> Self {
        self.default_options = options;
        self
    }

    /// Override the model-profile speech budget for a deployment or route.
    pub fn with_speech_limits(mut self, limits: SpeechLimits) -> Self {
        self.speech_limits = Some(limits);
        self
    }

    /// Validate static settings and construct one shared provider runtime.
    pub fn build(self) -> Result<ElevenLabsProvider, ElevenLabsConfigError> {
        self.credential.validate()?;
        let default_voice = normalized_voice_id(&self.default_voice)
            .ok_or(ElevenLabsConfigError::InvalidDefaultVoice)?;
        self.default_options
            .validate()
            .map_err(ElevenLabsConfigError::InvalidDefaultOptions)?;
        if self.speech_limits.is_some_and(|limits| {
            limits.max_text_bytes == Some(0) || limits.max_text_chars == Some(0)
        }) {
            return Err(ElevenLabsConfigError::ZeroSpeechLimit);
        }

        let mut transport = ProviderTransport::builder(self.profile.endpoint().clone())
            .with_auth(self.credential.into_auth())
            .with_limits(self.transport_limits)
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
        let transport = transport.build()?;
        let policy = Arc::new(ElevenLabsModelPolicy::new(
            self.profile.profile_arc(),
            self.profile.support_scope().clone(),
        ));
        Ok(ElevenLabsProvider {
            runtime: Arc::new(ProviderRuntime {
                scope: self.profile.scope_arc(),
                profile: self.profile,
                transport,
                policy,
                default_voice,
                option_merger: ElevenLabsOptionMerger {
                    defaults: self.default_options,
                },
                speech_limits: self.speech_limits,
            }),
        })
    }
}

pub(crate) struct ProviderRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) profile: ElevenLabsProfile,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<ElevenLabsModelPolicy>,
    pub(crate) default_voice: String,
    option_merger: ElevenLabsOptionMerger,
    speech_limits: Option<SpeechLimits>,
}

impl ProviderRuntime {
    pub(crate) fn merge_options(
        &self,
        options: &siumai_core::CallOptions,
    ) -> Result<ElevenLabsSpeechOptions, ProviderOptionError> {
        let layers = options
            .apply_provider_options(self.scope.provider_id(), ProviderOptionLayers::default())?;
        layers.merge_for(
            ProviderOptionContext::new(
                self.scope.provider_id(),
                ModelFamily::Speech,
                self.scope.api_mode(),
            ),
            &self.option_merger,
        )
    }

    pub(crate) fn speech_limits(&self, model: &ModelId) -> SpeechLimits {
        self.speech_limits
            .unwrap_or_else(|| self.profile.limits_for(model))
    }
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("scope", &self.scope)
            .field("profile_id", self.profile.provider_profile().id())
            .field("transport", &"shared")
            .field("default_voice", &self.default_voice)
            .field("speech_limits", &self.speech_limits)
            .finish()
    }
}

struct ElevenLabsOptionMerger {
    defaults: ElevenLabsSpeechOptions,
}

impl ProviderOptionMerger for ElevenLabsOptionMerger {
    type Output = ElevenLabsSpeechOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options)?.validate()
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            merged.merge_from(decode_options(options)?);
        }
        merged.validate()?;
        Ok(merged)
    }
}

fn decode_options(
    options: &ProviderOptions,
) -> Result<ElevenLabsSpeechOptions, ProviderOptionError> {
    serde_json::from_value(serde_json::Value::Object(options.value().clone())).map_err(|error| {
        ProviderOptionError::Rejected {
            path: "$".to_string(),
            reason: format!("options do not match the ElevenLabs speech schema: {error}"),
        }
    })
}

pub(crate) fn normalized_voice_id(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')))
    .then(|| value.to_string())
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ElevenLabsConfigError {
    #[error("invalid ElevenLabs identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid ElevenLabs endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid ElevenLabs transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid ElevenLabs credential: {0}")]
    Credential(#[from] ElevenLabsCredentialError),
    #[error("ElevenLabs default voice ID is invalid")]
    InvalidDefaultVoice,
    #[error("ElevenLabs default speech options are invalid: {0}")]
    InvalidDefaultOptions(ProviderOptionError),
    #[error("ElevenLabs speech byte limit must be greater than zero")]
    ZeroSpeechLimit,
    #[error("support scope is not ElevenLabs native Text-to-Speech")]
    IncompatibleSupportScope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{Model, ModelFamily, ProviderId};

    #[test]
    fn builder_rejects_static_configuration_and_registration_matches_direct_model() {
        let profile = ElevenLabsProfile::local_explicit("http://127.0.0.1:9876").unwrap();
        let invalid =
            ElevenLabsProvider::builder(profile.clone(), ElevenLabsCredential::api_key("test-key"))
                .with_default_voice("../voice")
                .build();
        assert!(matches!(
            invalid,
            Err(ElevenLabsConfigError::InvalidDefaultVoice)
        ));

        let provider =
            ElevenLabsProvider::builder(profile, ElevenLabsCredential::api_key("test-key"))
                .build()
                .unwrap();
        let direct = provider.speech(models::DEFAULT).unwrap();
        let erased = provider
            .registration()
            .speech_model(ModelId::new(models::DEFAULT).unwrap())
            .unwrap();
        assert_eq!(direct.descriptor(), erased.descriptor());
        assert_eq!(direct.limits(), erased.limits());
        assert_eq!(direct.family(), ModelFamily::Speech);
        assert_eq!(
            direct.provider_id(),
            &ProviderId::new("elevenlabs").unwrap()
        );
    }
}
