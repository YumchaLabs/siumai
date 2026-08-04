//! Provider identity, model policy, and dynamic registration contracts.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

use crate::error::Error;
use crate::model::{
    EmbeddingModel, ImageModel, LanguageModel, Model, ModelDescriptor, ModelFamily, RerankModel,
    SpeechModel, TranscriptionModel,
};

/// Identity shared by all configured provider instances.
///
/// This trait deliberately exposes no capability bag. Family support is
/// expressed by implementing one or more narrow provider traits below.
pub trait Provider: Send + Sync {
    fn scope(&self) -> &ProviderScope;

    fn provider_id(&self) -> &ProviderId {
        self.scope().provider_id()
    }

    /// Provider-owned deployment or public API identity, when distinct from
    /// the canonical provider ID.
    fn platform(&self) -> Option<&PlatformId> {
        self.scope().platform()
    }
}

/// A configured provider that constructs lightweight language model handles.
pub trait LanguageModelProvider: Provider {
    type Model: LanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// A configured provider that constructs lightweight embedding model handles.
pub trait EmbeddingModelProvider: Provider {
    type Model: EmbeddingModel;

    fn embedding_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// A configured provider that constructs lightweight rerank model handles.
pub trait RerankModelProvider: Provider {
    type Model: RerankModel;

    fn rerank_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// A configured provider that constructs lightweight image model handles.
pub trait ImageModelProvider: Provider {
    type Model: ImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// A configured provider that constructs lightweight speech model handles.
pub trait SpeechModelProvider: Provider {
    type Model: SpeechModel;

    fn speech_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// A configured provider that constructs lightweight transcription model handles.
pub trait TranscriptionModelProvider: Provider {
    type Model: TranscriptionModel;

    fn transcription_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError>;
}

/// Error returned when an identifier is empty or contains reserved characters.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid {kind} identifier `{value}`: {reason}")]
pub struct InvalidId {
    kind: &'static str,
    value: String,
    reason: &'static str,
}

impl InvalidId {
    fn new(kind: &'static str, value: impl Into<String>, reason: &'static str) -> Self {
        let value = value.into();
        let mut diagnostic = value
            .chars()
            .take(128)
            .map(|character| {
                if character.is_control() {
                    '?'
                } else {
                    character
                }
            })
            .collect::<String>();
        if value.chars().count() > 128 {
            diagnostic.push_str("...");
        }
        Self {
            kind,
            value: diagnostic,
            reason,
        }
    }
}

macro_rules! canonical_id {
    ($name:ident, $kind:literal) => {
        #[doc = concat!("A normalized ", $kind, " identifier.")]
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Parse and normalize an ASCII identifier.
            pub fn new(value: impl AsRef<str>) -> Result<Self, InvalidId> {
                let raw = value.as_ref();
                let normalized = raw.trim().to_ascii_lowercase();
                if normalized.is_empty() {
                    return Err(InvalidId::new($kind, raw, "must not be empty"));
                }
                if normalized.len() > 128 {
                    return Err(InvalidId::new($kind, raw, "must not exceed 128 bytes"));
                }
                if !normalized.is_ascii() {
                    return Err(InvalidId::new($kind, raw, "must be ASCII"));
                }
                if !normalized
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
                {
                    return Err(InvalidId::new(
                        $kind,
                        raw,
                        "may contain only ASCII letters, digits, '-', '_', and '.'",
                    ));
                }
                Ok(Self(normalized))
            }

            /// Return the normalized identifier.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = InvalidId;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

canonical_id!(ProviderId, "provider");
canonical_id!(RouteId, "route");
canonical_id!(PlatformId, "platform");
canonical_id!(ProtocolId, "protocol");
canonical_id!(ApiModeId, "API mode");
canonical_id!(ProfileId, "profile");
canonical_id!(ProtocolContractId, "protocol contract");

/// Immutable provider, platform, protocol, and API-mode identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderScope {
    provider: ProviderId,
    platform: Option<PlatformId>,
    protocol: Option<ProtocolId>,
    api_mode: Option<ApiModeId>,
}

impl ProviderScope {
    pub fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            platform: None,
            protocol: None,
            api_mode: None,
        }
    }

    pub fn with_platform(mut self, platform: PlatformId) -> Self {
        self.platform = Some(platform);
        self
    }

    pub fn with_protocol(mut self, protocol: ProtocolId) -> Self {
        self.protocol = Some(protocol);
        self
    }

    pub fn with_api_mode(mut self, api_mode: ApiModeId) -> Self {
        self.api_mode = Some(api_mode);
        self
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider
    }

    pub fn platform(&self) -> Option<&PlatformId> {
        self.platform.as_ref()
    }

    pub fn protocol(&self) -> Option<&ProtocolId> {
        self.protocol.as_ref()
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.api_mode.as_ref()
    }
}

/// An opaque provider model identifier.
///
/// Model IDs intentionally preserve case and embedded `:` characters. They are
/// open input, not a closed catalog or capability allowlist.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Create a model ID while preserving its spelling exactly.
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidId> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(InvalidId::new("model", value, "must not be empty"));
        }
        if value != value.trim() {
            return Err(InvalidId::new(
                "model",
                value,
                "must not contain leading or trailing whitespace",
            ));
        }
        if value.len() > 2048 {
            return Err(InvalidId::new("model", value, "must not exceed 2048 bytes"));
        }
        if value.chars().any(char::is_control) {
            return Err(InvalidId::new(
                "model",
                value,
                "must not contain control characters",
            ));
        }
        Ok(Self(value))
    }

    /// Return the provider-owned model ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("ModelId").field(&self.0).finish()
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<&str> for ModelId {
    type Error = InvalidId;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// The protocol operation evaluated by model policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelOperation {
    Generate,
    Stream,
    Embed,
    Rerank,
    GenerateImage,
    SynthesizeSpeech,
    Transcribe,
}

/// Why a provider or protocol cannot execute an operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UnsupportedReason {
    FamilyNotImplemented,
    OperationNotImplemented,
    ApiModeMismatch,
    ModelRetired,
    ProviderRestriction,
}

/// The callable state of one model operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SupportState {
    Supported,
    Unsupported { reason: UnsupportedReason },
    Unknown,
}

/// Typed, non-remapping guidance attached to a model-policy decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelAdvisory {
    UnknownModel,
    Deprecated { replacement: Option<ModelId> },
    Retired { replacement: Option<ModelId> },
    RollingAlias,
}

/// A model-policy result that keeps callability separate from lifecycle advice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPolicyDecision {
    state: SupportState,
    advisories: Box<[ModelAdvisory]>,
}

impl ModelPolicyDecision {
    pub fn supported() -> Self {
        Self {
            state: SupportState::Supported,
            advisories: Box::new([]),
        }
    }

    pub fn unsupported(reason: UnsupportedReason) -> Self {
        Self {
            state: SupportState::Unsupported { reason },
            advisories: Box::new([]),
        }
    }

    pub fn unknown_model() -> Self {
        Self {
            state: SupportState::Unknown,
            advisories: Box::new([ModelAdvisory::UnknownModel]),
        }
    }

    pub fn with_advisory(mut self, advisory: ModelAdvisory) -> Self {
        let mut advisories = self.advisories.into_vec();
        advisories.push(advisory);
        self.advisories = advisories.into_boxed_slice();
        self
    }

    pub fn state(&self) -> &SupportState {
        &self.state
    }

    pub fn advisories(&self) -> &[ModelAdvisory] {
        &self.advisories
    }
}

/// Complete identity and protocol context used by provider-owned model policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPolicyContext {
    pub scope: Arc<ProviderScope>,
    pub model: ModelId,
    pub family: ModelFamily,
    pub operation: ModelOperation,
}

/// Provider-owned model policy.
pub trait ModelPolicy: Send + Sync {
    /// Evaluate one model, family, operation, and protocol combination.
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision;
}

/// Failure to resolve a requested family or construct its lightweight model.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelLookupError {
    #[error("provider `{provider}` does not expose the {family:?} family")]
    UnsupportedFamily {
        provider: ProviderId,
        family: ModelFamily,
    },
    #[error("invalid model reference: {0}")]
    InvalidReference(String),
    #[error("model construction failed: {source}")]
    Construction {
        #[source]
        source: Error,
    },
    #[error(
        "provider registration returned an inconsistent model descriptor: expected {expected:?}, got {actual:?}"
    )]
    IdentityMismatch {
        expected: Box<ModelDescriptor>,
        actual: Box<ModelDescriptor>,
    },
}

pub type ModelFactory<T> =
    Arc<dyn Fn(ModelId) -> Result<Arc<T>, ModelLookupError> + Send + Sync + 'static>;

/// Narrow family constructors captured from one configured provider runtime.
///
/// Registry stores this value without importing concrete provider packages.
#[derive(Clone)]
pub struct ProviderRegistration {
    scope: Arc<ProviderScope>,
    model_policy: Arc<dyn ModelPolicy>,
    language: Option<ModelFactory<dyn LanguageModel>>,
    embedding: Option<ModelFactory<dyn EmbeddingModel>>,
    rerank: Option<ModelFactory<dyn RerankModel>>,
    image: Option<ModelFactory<dyn ImageModel>>,
    speech: Option<ModelFactory<dyn SpeechModel>>,
    transcription: Option<ModelFactory<dyn TranscriptionModel>>,
}

impl fmt::Debug for ProviderRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRegistration")
            .field("scope", &self.scope)
            .field("language", &self.language.is_some())
            .field("embedding", &self.embedding.is_some())
            .field("rerank", &self.rerank.is_some())
            .field("image", &self.image.is_some())
            .field("speech", &self.speech.is_some())
            .field("transcription", &self.transcription.is_some())
            .finish()
    }
}

impl ProviderRegistration {
    /// Begin a registration for one canonical provider and captured API mode.
    pub fn new(provider_id: ProviderId, model_policy: Arc<dyn ModelPolicy>) -> Self {
        Self::from_scope(Arc::new(ProviderScope::new(provider_id)), model_policy)
    }

    /// Begin a registration from the exact scope shared by direct models.
    pub fn from_scope(scope: Arc<ProviderScope>, model_policy: Arc<dyn ModelPolicy>) -> Self {
        Self {
            scope,
            model_policy,
            language: None,
            embedding: None,
            rerank: None,
            image: None,
            speech: None,
            transcription: None,
        }
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.scope.provider_id()
    }

    pub fn scope(&self) -> &Arc<ProviderScope> {
        &self.scope
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.scope.api_mode()
    }

    pub fn platform(&self) -> Option<&PlatformId> {
        self.scope.platform()
    }

    pub fn protocol(&self) -> Option<&ProtocolId> {
        self.scope.protocol()
    }

    /// Whether this configured registration exposes a constructor for a family.
    ///
    /// This reports factory availability only. Model and operation support
    /// remains the responsibility of [`ModelPolicy`].
    pub fn supports_family(&self, family: ModelFamily) -> bool {
        match family {
            ModelFamily::Language => self.language.is_some(),
            ModelFamily::Embedding => self.embedding.is_some(),
            ModelFamily::Rerank => self.rerank.is_some(),
            ModelFamily::Image => self.image.is_some(),
            ModelFamily::Speech => self.speech.is_some(),
            ModelFamily::Transcription => self.transcription.is_some(),
        }
    }

    /// Enumerate available family constructors in stable taxonomy order.
    pub fn families(&self) -> impl Iterator<Item = ModelFamily> + '_ {
        [
            ModelFamily::Language,
            ModelFamily::Embedding,
            ModelFamily::Rerank,
            ModelFamily::Image,
            ModelFamily::Speech,
            ModelFamily::Transcription,
        ]
        .into_iter()
        .filter(|family| self.supports_family(*family))
    }

    pub fn with_platform(mut self, platform: PlatformId) -> Self {
        Arc::make_mut(&mut self.scope).platform = Some(platform);
        self
    }

    pub fn with_protocol(mut self, protocol: ProtocolId) -> Self {
        Arc::make_mut(&mut self.scope).protocol = Some(protocol);
        self
    }

    pub fn with_api_mode(mut self, api_mode: ApiModeId) -> Self {
        Arc::make_mut(&mut self.scope).api_mode = Some(api_mode);
        self
    }

    pub fn evaluate(
        &self,
        model: ModelId,
        family: ModelFamily,
        operation: ModelOperation,
    ) -> ModelPolicyDecision {
        self.model_policy.evaluate(&ModelPolicyContext {
            scope: self.scope.clone(),
            model,
            family,
            operation,
        })
    }

    pub fn with_language(mut self, factory: ModelFactory<dyn LanguageModel>) -> Self {
        self.language = Some(factory);
        self
    }

    pub fn with_embedding(mut self, factory: ModelFactory<dyn EmbeddingModel>) -> Self {
        self.embedding = Some(factory);
        self
    }

    pub fn with_rerank(mut self, factory: ModelFactory<dyn RerankModel>) -> Self {
        self.rerank = Some(factory);
        self
    }

    pub fn with_image(mut self, factory: ModelFactory<dyn ImageModel>) -> Self {
        self.image = Some(factory);
        self
    }

    pub fn with_speech(mut self, factory: ModelFactory<dyn SpeechModel>) -> Self {
        self.speech = Some(factory);
        self
    }

    pub fn with_transcription(mut self, factory: ModelFactory<dyn TranscriptionModel>) -> Self {
        self.transcription = Some(factory);
        self
    }

    pub fn language_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn LanguageModel>, ModelLookupError> {
        let factory = self
            .language
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Language))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Language, factory(model)?)
    }

    pub fn embedding_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn EmbeddingModel>, ModelLookupError> {
        let factory = self
            .embedding
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Embedding))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Embedding, factory(model)?)
    }

    pub fn rerank_model(&self, model: ModelId) -> Result<Arc<dyn RerankModel>, ModelLookupError> {
        let factory = self
            .rerank
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Rerank))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Rerank, factory(model)?)
    }

    pub fn image_model(&self, model: ModelId) -> Result<Arc<dyn ImageModel>, ModelLookupError> {
        let factory = self
            .image
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Image))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Image, factory(model)?)
    }

    pub fn speech_model(&self, model: ModelId) -> Result<Arc<dyn SpeechModel>, ModelLookupError> {
        let factory = self
            .speech
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Speech))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Speech, factory(model)?)
    }

    pub fn transcription_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn TranscriptionModel>, ModelLookupError> {
        let factory = self
            .transcription
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Transcription))?;
        let requested = model.clone();
        self.validate_model(requested, ModelFamily::Transcription, factory(model)?)
    }

    fn unsupported(&self, family: ModelFamily) -> ModelLookupError {
        ModelLookupError::UnsupportedFamily {
            provider: self.scope.provider_id().clone(),
            family,
        }
    }

    fn validate_model<T: Model + ?Sized>(
        &self,
        expected_model: ModelId,
        expected_family: ModelFamily,
        model: Arc<T>,
    ) -> Result<Arc<T>, ModelLookupError> {
        let descriptor = model.descriptor();
        if descriptor.scope().as_ref() == self.scope.as_ref()
            && descriptor.model() == &expected_model
            && descriptor.family() == expected_family
        {
            return Ok(model);
        }

        let expected =
            ModelDescriptor::from_scope(self.scope.clone(), expected_model, expected_family);

        Err(ModelLookupError::IdentityMismatch {
            expected: Box::new(expected),
            actual: Box::new(descriptor.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AdvisoryPolicy;

    impl ModelPolicy for AdvisoryPolicy {
        fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
            if context.model.as_str() == "known" {
                ModelPolicyDecision::supported()
            } else {
                ModelPolicyDecision::unknown_model()
            }
        }
    }

    #[test]
    fn canonical_ids_normalize_without_accepting_route_separators() {
        assert_eq!(ProviderId::new(" OpenAI ").unwrap().as_str(), "openai");
        assert_eq!(RouteId::new("EU_PRIMARY").unwrap().as_str(), "eu_primary");
        assert!(RouteId::new("primary:model").is_err());
    }

    #[test]
    fn model_ids_preserve_colons_and_case() {
        let id = ModelId::new("Publisher:Model/V2").unwrap();
        assert_eq!(id.as_str(), "Publisher:Model/V2");
    }

    #[test]
    fn identifier_deserialization_reuses_validation_and_normalization() {
        let provider: ProviderId = serde_json::from_str("\" OpenAI \"").unwrap();
        assert_eq!(provider.as_str(), "openai");
        assert!(serde_json::from_str::<RouteId>("\"primary:model\"").is_err());
        assert!(serde_json::from_str::<ModelId>("\"\\n\"").is_err());
    }

    #[test]
    fn unknown_future_models_remain_unknown_instead_of_unsupported() {
        let context = ModelPolicyContext {
            scope: Arc::new(
                ProviderScope::new(ProviderId::new("custom").unwrap())
                    .with_protocol(ProtocolId::new("native").unwrap()),
            ),
            model: ModelId::new("future:model").unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
        };

        let decision = AdvisoryPolicy.evaluate(&context);
        assert_eq!(decision.state(), &SupportState::Unknown);
        assert_eq!(decision.advisories(), &[ModelAdvisory::UnknownModel]);
    }

    #[test]
    fn missing_family_is_a_typed_lookup_error() {
        let registration =
            ProviderRegistration::new(ProviderId::new("custom").unwrap(), Arc::new(AdvisoryPolicy));
        let error = registration
            .image_model(ModelId::new("image-future").unwrap())
            .err()
            .expect("missing image family should return an error");

        assert!(matches!(
            error,
            ModelLookupError::UnsupportedFamily {
                family: ModelFamily::Image,
                ..
            }
        ));
    }

    #[test]
    fn family_availability_is_factory_presence_not_model_policy() {
        let registration =
            ProviderRegistration::new(ProviderId::new("custom").unwrap(), Arc::new(AdvisoryPolicy))
                .with_language(Arc::new(|_| unreachable!("factory is not called")));

        assert!(registration.supports_family(ModelFamily::Language));
        assert!(!registration.supports_family(ModelFamily::Image));
        assert_eq!(
            registration.families().collect::<Vec<_>>(),
            [ModelFamily::Language]
        );
    }

    #[test]
    fn registration_carries_policy_and_full_protocol_context() {
        let registration =
            ProviderRegistration::new(ProviderId::new("custom").unwrap(), Arc::new(AdvisoryPolicy))
                .with_platform(PlatformId::new("public-api").unwrap())
                .with_protocol(ProtocolId::new("native").unwrap())
                .with_api_mode(ApiModeId::new("responses").unwrap());

        let status = registration.evaluate(
            ModelId::new("future:model").unwrap(),
            ModelFamily::Language,
            ModelOperation::Generate,
        );
        assert_eq!(status.state(), &SupportState::Unknown);
        assert_eq!(
            registration.platform().map(PlatformId::as_str),
            Some("public-api")
        );
        assert_eq!(
            registration.protocol().map(ProtocolId::as_str),
            Some("native")
        );
    }
}
