use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, InvalidId, ModelFamily, ModelId, ModelLookupError, ModelOperation, ModelPolicy,
    ModelPolicyContext, ModelPolicyDecision, ProtocolId, Provider, ProviderId, ProviderOptionError,
    ProviderOptionLayers, ProviderOptionMerger, ProviderOptions, ProviderRegistration,
    ProviderScope, TranscriptionModelProvider, TypedProviderOptions, UnsupportedReason,
};
use siumai_transport::{
    EndpointConfig, EndpointError, EndpointPolicy, OfficialOrigin, ProviderTransport,
    TransportConfigError, TransportLimits,
};
use thiserror::Error;

use crate::credential::{DeepgramCredential, DeepgramCredentialError};
use crate::model::DeepgramTranscriptionModel;
use crate::options::DeepgramTranscriptionOptions;

const DEEPGRAM_ORIGIN: &str = "https://api.deepgram.com";

/// Long-lived configured Deepgram provider sharing transport and auth state.
#[derive(Clone)]
pub struct DeepgramProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
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
        let model = ModelId::new(model.into())
            .map_err(|error| ModelLookupError::InvalidReference(error.to_string()))?;
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

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_scope(self.runtime.scope.clone(), self.runtime.policy.clone())
            .with_transcription(Arc::new(move |model| {
                Ok(Arc::new(provider.create_transcription_model(model))
                    as Arc<dyn siumai_core::TranscriptionModel>)
            }))
    }

    fn create_transcription_model(&self, model: ModelId) -> DeepgramTranscriptionModel {
        DeepgramTranscriptionModel::new(self.runtime.clone(), model)
    }
}

impl Provider for DeepgramProvider {
    fn scope(&self) -> &ProviderScope {
        self.runtime.scope.as_ref()
    }
}

impl TranscriptionModelProvider for DeepgramProvider {
    type Model = DeepgramTranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_transcription_model(model))
    }
}

impl fmt::Debug for DeepgramProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepgramProvider")
            .field("scope", &self.runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

pub struct DeepgramProviderBuilder {
    credential: DeepgramCredential,
    endpoint: Option<EndpointConfig>,
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
        let scope = Arc::new(
            ProviderScope::new(ProviderId::new("deepgram")?)
                .with_protocol(ProtocolId::new("deepgram-prerecorded")?)
                .with_api_mode(ApiModeId::new("prerecorded")?),
        );
        let policy = Arc::new(DeepgramModelPolicy);
        Ok(DeepgramProvider {
            runtime: Arc::new(ProviderRuntime {
                scope,
                transport,
                policy,
                default_options,
                option_merger: DeepgramOptionMerger,
            }),
        })
    }
}

impl fmt::Debug for DeepgramProviderBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepgramProviderBuilder")
            .field("credential", &self.credential)
            .field("endpoint", &self.endpoint)
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
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<DeepgramModelPolicy>,
    default_options: ProviderOptions,
    option_merger: DeepgramOptionMerger,
}

impl ProviderRuntime {
    pub(crate) fn options(
        &self,
        call: &siumai_core::CallOptions,
    ) -> Result<DeepgramTranscriptionOptions, ProviderOptionError> {
        let layers =
            ProviderOptionLayers::default().with_provider_default(self.default_options.clone())?;
        call.apply_provider_options(self.scope.provider_id(), layers)?
            .merge_for(self.scope.provider_id(), &self.option_merger)
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

pub(crate) struct DeepgramModelPolicy;

impl ModelPolicy for DeepgramModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        if context.family != ModelFamily::Transcription {
            return ModelPolicyDecision::unsupported(UnsupportedReason::FamilyNotImplemented);
        }
        if context.operation != ModelOperation::Transcribe {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if crate::models::ALL_TRANSCRIPTION.contains(&context.model.as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

struct DeepgramOptionMerger;

impl ProviderOptionMerger for DeepgramOptionMerger {
    type Output = DeepgramTranscriptionOptions;

    fn validate_layer(
        &self,
        _origin: siumai_core::ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options.value()).and_then(|value| value.validate())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = Map::new();
        for (_, options) in layers.in_precedence_order() {
            for (name, value) in options.value() {
                merged.insert(name.clone(), value.clone());
            }
        }
        decode_options(&merged).and_then(|value| {
            value.validate()?;
            Ok(value)
        })
    }
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
    #[error("invalid Deepgram endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Deepgram transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid Deepgram credential: {0}")]
    Credential(#[from] DeepgramCredentialError),
    #[error("invalid Deepgram default options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("the official Deepgram endpoint requires an API key")]
    OfficialEndpointRequiresCredential,
}

#[cfg(test)]
mod tests {
    use siumai_core::{CallOptions, Model, ModelFamily};

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
        assert_eq!(direct.descriptor().protocol(), Some("deepgram-prerecorded"));
        assert_eq!(direct.descriptor().api_mode(), Some("prerecorded"));
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
        let call = CallOptions::default().with_provider_options(
            ProviderOptions::typed(
                &DeepgramTranscriptionOptions::new()
                    .with_diarize_model(crate::DeepgramDiarizeModel::V1),
            )
            .unwrap(),
        );
        assert_eq!(
            provider.runtime.options(&call).unwrap().diarize_model(),
            Some(crate::DeepgramDiarizeModel::V1)
        );

        let raw = ProviderOptions::checked_raw(
            ProviderId::new("deepgram").unwrap(),
            serde_json::json!({"unknownFutureField": true}),
        )
        .unwrap();
        assert!(
            provider
                .runtime
                .options(&CallOptions::default().with_provider_options(raw))
                .is_err()
        );
    }
}
