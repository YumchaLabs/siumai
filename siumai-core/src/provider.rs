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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// A model-policy answer that does not turn missing catalog data into support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CapabilityStatus {
    Supported,
    Unsupported { reason: String },
    Unknown { warning: String },
}

/// Complete identity and protocol context used by provider-owned model policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPolicyContext {
    pub provider: ProviderId,
    pub platform: Option<String>,
    pub model: ModelId,
    pub family: ModelFamily,
    pub operation: ModelOperation,
    pub protocol: Option<String>,
    pub api_mode: Option<String>,
}

/// Provider-owned model policy.
pub trait ModelPolicy: Send + Sync {
    /// Evaluate one model, family, operation, and protocol combination.
    fn evaluate(&self, context: &ModelPolicyContext) -> CapabilityStatus;
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
    provider_id: ProviderId,
    platform: Option<String>,
    protocol: Option<String>,
    api_mode: Option<String>,
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
            .field("provider_id", &self.provider_id)
            .field("platform", &self.platform)
            .field("protocol", &self.protocol)
            .field("api_mode", &self.api_mode)
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
        Self {
            provider_id,
            platform: None,
            protocol: None,
            api_mode: None,
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
        &self.provider_id
    }

    pub fn api_mode(&self) -> Option<&str> {
        self.api_mode.as_deref()
    }

    pub fn platform(&self) -> Option<&str> {
        self.platform.as_deref()
    }

    pub fn protocol(&self) -> Option<&str> {
        self.protocol.as_deref()
    }

    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = Some(platform.into());
        self
    }

    pub fn with_protocol(mut self, protocol: impl Into<String>) -> Self {
        self.protocol = Some(protocol.into());
        self
    }

    pub fn with_api_mode(mut self, api_mode: impl Into<String>) -> Self {
        self.api_mode = Some(api_mode.into());
        self
    }

    pub fn evaluate(
        &self,
        model: ModelId,
        family: ModelFamily,
        operation: ModelOperation,
    ) -> CapabilityStatus {
        self.model_policy.evaluate(&ModelPolicyContext {
            provider: self.provider_id.clone(),
            platform: self.platform.clone(),
            model,
            family,
            operation,
            protocol: self.protocol.clone(),
            api_mode: self.api_mode.clone(),
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
            provider: self.provider_id.clone(),
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
        if descriptor.provider() == &self.provider_id
            && descriptor.model() == &expected_model
            && descriptor.family() == expected_family
            && descriptor.platform() == self.platform.as_deref()
            && descriptor.protocol() == self.protocol.as_deref()
            && descriptor.api_mode() == self.api_mode.as_deref()
        {
            return Ok(model);
        }

        let mut expected =
            ModelDescriptor::new(self.provider_id.clone(), expected_model, expected_family);
        if let Some(platform) = &self.platform {
            expected = expected.with_platform(platform);
        }
        if let Some(protocol) = &self.protocol {
            expected = expected.with_protocol(protocol);
        }
        if let Some(api_mode) = &self.api_mode {
            expected = expected.with_api_mode(api_mode);
        }

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
        fn evaluate(&self, context: &ModelPolicyContext) -> CapabilityStatus {
            if context.model.as_str() == "known" {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unknown {
                    warning: "model is absent from the advisory catalog".to_string(),
                }
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
            provider: ProviderId::new("custom").unwrap(),
            platform: None,
            model: ModelId::new("future:model").unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
            protocol: Some("native".to_string()),
            api_mode: None,
        };

        assert!(matches!(
            AdvisoryPolicy.evaluate(&context),
            CapabilityStatus::Unknown { .. }
        ));
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
    fn registration_carries_policy_and_full_protocol_context() {
        let registration =
            ProviderRegistration::new(ProviderId::new("custom").unwrap(), Arc::new(AdvisoryPolicy))
                .with_platform("public-api")
                .with_protocol("native")
                .with_api_mode("responses");

        let status = registration.evaluate(
            ModelId::new("future:model").unwrap(),
            ModelFamily::Language,
            ModelOperation::Generate,
        );
        assert!(matches!(status, CapabilityStatus::Unknown { .. }));
        assert_eq!(registration.platform(), Some("public-api"));
        assert_eq!(registration.protocol(), Some("native"));
    }
}
