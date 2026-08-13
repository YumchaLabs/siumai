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
/// expressed by implementing one or more narrow provider traits below. Exact
/// platform, protocol, and API-mode identity belongs to model descriptors,
/// registrations, and separate provider-owned support evidence rather than
/// the provider as a whole.
pub trait Provider: Send + Sync {
    fn provider_id(&self) -> &ProviderId;
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

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl TryFrom<&str> for $name {
            type Error = InvalidId;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl std::str::FromStr for $name {
            type Err = InvalidId;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
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
canonical_id!(NativeSurfaceId, "provider-native surface");
canonical_id!(ReplayDomainId, "replay domain");

/// Opaque capability identifying one configured provider instance.
///
/// The token is intentionally neither serializable nor constructible from
/// provider labels. A configured provider runtime mints one token and shares it
/// with every model handle it creates. Profiles and technical scopes do not own
/// the token because they may be cloned across independently configured
/// credentials or transports.
#[derive(Clone)]
pub struct ProviderInstanceId(Arc<ProviderInstanceMarker>);

struct ProviderInstanceMarker;

impl ProviderInstanceId {
    /// Create a fresh configured-instance capability.
    pub fn new() -> Self {
        Self(Arc::new(ProviderInstanceMarker))
    }
}

impl Default for ProviderInstanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ProviderInstanceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<opaque-provider-instance>")
    }
}

impl PartialEq for ProviderInstanceId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ProviderInstanceId {}

/// Whether replay state belongs to an audited official audience or a caller-declared custom one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReplayAudience {
    Official(ReplayDomainId),
    Custom(ReplayDomainId),
}

impl ReplayAudience {
    pub fn id(&self) -> &ReplayDomainId {
        match self {
            Self::Official(id) | Self::Custom(id) => id,
        }
    }

    pub const fn is_official(&self) -> bool {
        matches!(self, Self::Official(_))
    }
}

/// Non-secret identity that bounds provider-native replay state.
///
/// IDs are caller-visible labels, never URLs, credentials, signed values, or
/// opaque provider payloads. `caller_scope` distinguishes material account,
/// workspace, project, or deployment boundaries within one audience.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReplayDomain {
    audience: ReplayAudience,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    caller_scope: Option<ReplayDomainId>,
}

impl ReplayDomain {
    pub fn official(audience: ReplayDomainId) -> Self {
        Self {
            audience: ReplayAudience::Official(audience),
            caller_scope: None,
        }
    }

    pub fn custom(audience: ReplayDomainId) -> Self {
        Self {
            audience: ReplayAudience::Custom(audience),
            caller_scope: None,
        }
    }

    pub fn with_caller_scope(mut self, caller_scope: ReplayDomainId) -> Self {
        self.caller_scope = Some(caller_scope);
        self
    }

    pub fn audience(&self) -> &ReplayAudience {
        &self.audience
    }

    pub fn caller_scope(&self) -> Option<&ReplayDomainId> {
        self.caller_scope.as_ref()
    }
}

/// Exact technical execution scope for a model, registration, or policy context.
///
/// This is not provider-wide identity: one configured provider may expose
/// multiple platforms, protocols, and API modes across model families.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProviderScope {
    provider: ProviderId,
    platform: Option<PlatformId>,
    protocol: Option<ProtocolId>,
    api_mode: Option<ApiModeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_domain: Option<ReplayDomain>,
}

impl ProviderScope {
    pub fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            platform: None,
            protocol: None,
            api_mode: None,
            replay_domain: None,
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

    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
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

    pub fn replay_domain(&self) -> Option<&ReplayDomain> {
        self.replay_domain.as_ref()
    }

    /// Return whether both configured execution scopes can replay native state.
    ///
    /// Missing replay identity always fails closed. Registry routes and model
    /// IDs deliberately do not participate in this comparison.
    pub fn shares_replay_domain(&self, other: &Self) -> bool {
        self.provider == other.provider
            && self.platform == other.platform
            && self.protocol == other.protocol
            && self.api_mode == other.api_mode
            && self.replay_domain.is_some()
            && self.replay_domain == other.replay_domain
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

/// Stable operation taxonomy used by support evidence and error context.
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

impl ModelOperation {
    /// Model family that owns this operation.
    pub const fn family(self) -> ModelFamily {
        match self {
            Self::Generate | Self::Stream => ModelFamily::Language,
            Self::Embed => ModelFamily::Embedding,
            Self::Rerank => ModelFamily::Rerank,
            Self::GenerateImage => ModelFamily::Image,
            Self::SynthesizeSpeech => ModelFamily::Speech,
            Self::Transcribe => ModelFamily::Transcription,
        }
    }
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
    #[error(transparent)]
    InvalidModelId(#[from] InvalidId),
    #[error("invalid model reference: {source}")]
    InvalidModelReference {
        #[source]
        source: Error,
    },
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

/// Invalid assembly of default family bindings for one Registry route.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderRegistrationError {
    #[error("provider registration expected `{expected}` but received scope for `{found}`")]
    ProviderMismatch {
        expected: ProviderId,
        found: ProviderId,
    },
    #[error(
        "provider registration already contains a {family:?} binding for {existing:?}; cannot add {incoming:?}"
    )]
    DuplicateFamily {
        family: ModelFamily,
        existing: Box<ProviderScope>,
        incoming: Box<ProviderScope>,
    },
}

pub type ModelFactory<T> =
    Arc<dyn Fn(ModelId) -> Result<Arc<T>, ModelLookupError> + Send + Sync + 'static>;

struct FamilyRegistration<T: ?Sized> {
    scope: Arc<ProviderScope>,
    factory: ModelFactory<T>,
}

impl<T: ?Sized> Clone for FamilyRegistration<T> {
    fn clone(&self) -> Self {
        Self {
            scope: self.scope.clone(),
            factory: self.factory.clone(),
        }
    }
}

impl<T: ?Sized> FamilyRegistration<T> {
    fn new(scope: Arc<ProviderScope>, factory: ModelFactory<T>) -> Self {
        Self { scope, factory }
    }
}

/// Host-selected family bindings for one canonical provider identity.
///
/// Each family owns its exact technical scope and constructor.
/// Registry stores this value without importing concrete provider packages.
/// Alternative API modes for the same family remain separate route
/// registrations instead of being selected implicitly. Explicit merge may
/// combine disjoint bindings from separate configured instances; matching
/// provider identity does not claim matching credentials or runtime origin.
#[derive(Clone)]
pub struct ProviderRegistration {
    provider_id: ProviderId,
    language: Option<FamilyRegistration<dyn LanguageModel>>,
    embedding: Option<FamilyRegistration<dyn EmbeddingModel>>,
    rerank: Option<FamilyRegistration<dyn RerankModel>>,
    image: Option<FamilyRegistration<dyn ImageModel>>,
    speech: Option<FamilyRegistration<dyn SpeechModel>>,
    transcription: Option<FamilyRegistration<dyn TranscriptionModel>>,
}

impl fmt::Debug for ProviderRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRegistration")
            .field("provider_id", &self.provider_id)
            .field(
                "language_scope",
                &self.language.as_ref().map(|binding| &binding.scope),
            )
            .field(
                "embedding_scope",
                &self.embedding.as_ref().map(|binding| &binding.scope),
            )
            .field(
                "rerank_scope",
                &self.rerank.as_ref().map(|binding| &binding.scope),
            )
            .field(
                "image_scope",
                &self.image.as_ref().map(|binding| &binding.scope),
            )
            .field(
                "speech_scope",
                &self.speech.as_ref().map(|binding| &binding.scope),
            )
            .field(
                "transcription_scope",
                &self.transcription.as_ref().map(|binding| &binding.scope),
            )
            .finish()
    }
}

impl ProviderRegistration {
    fn empty(provider_id: ProviderId) -> Self {
        Self {
            provider_id,
            language: None,
            embedding: None,
            rerank: None,
            image: None,
            speech: None,
            transcription: None,
        }
    }

    pub fn from_language(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn LanguageModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_language(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn from_embedding(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn EmbeddingModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_embedding(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn from_rerank(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn RerankModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_rerank(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn from_image(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn ImageModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_image(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn from_speech(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn SpeechModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_speech(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn from_transcription(
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn TranscriptionModel>,
    ) -> Self {
        let scope = scope.into();
        let provider_id = scope.provider_id().clone();
        Self::empty(provider_id)
            .bind_transcription(scope, factory)
            .expect("initial family scope defines the registration provider")
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn scope(&self, family: ModelFamily) -> Option<&ProviderScope> {
        self.scope_arc(family).map(Arc::as_ref)
    }

    fn scope_arc(&self, family: ModelFamily) -> Option<&Arc<ProviderScope>> {
        match family {
            ModelFamily::Language => self.language.as_ref().map(|binding| &binding.scope),
            ModelFamily::Embedding => self.embedding.as_ref().map(|binding| &binding.scope),
            ModelFamily::Rerank => self.rerank.as_ref().map(|binding| &binding.scope),
            ModelFamily::Image => self.image.as_ref().map(|binding| &binding.scope),
            ModelFamily::Speech => self.speech.as_ref().map(|binding| &binding.scope),
            ModelFamily::Transcription => self.transcription.as_ref().map(|binding| &binding.scope),
        }
    }

    pub fn api_mode(&self, family: ModelFamily) -> Option<&ApiModeId> {
        self.scope(family).and_then(|scope| scope.api_mode())
    }

    pub fn platform(&self, family: ModelFamily) -> Option<&PlatformId> {
        self.scope(family).and_then(|scope| scope.platform())
    }

    pub fn protocol(&self, family: ModelFamily) -> Option<&ProtocolId> {
        self.scope(family).and_then(|scope| scope.protocol())
    }

    /// Whether this configured registration exposes a constructor for a family.
    ///
    /// This reports factory availability only. Provider-specific request
    /// validation remains the responsibility of the concrete model.
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

    /// Clone one family binding into its own non-empty registration.
    ///
    /// This lets the host expose a narrower route than a provider's combined
    /// default registration without reconstructing provider internals.
    pub fn for_family(&self, family: ModelFamily) -> Option<Self> {
        let mut registration = Self::empty(self.provider_id.clone());
        match family {
            ModelFamily::Language => registration.language = self.language.clone(),
            ModelFamily::Embedding => registration.embedding = self.embedding.clone(),
            ModelFamily::Rerank => registration.rerank = self.rerank.clone(),
            ModelFamily::Image => registration.image = self.image.clone(),
            ModelFamily::Speech => registration.speech = self.speech.clone(),
            ModelFamily::Transcription => {
                registration.transcription = self.transcription.clone();
            }
        }
        registration.supports_family(family).then_some(registration)
    }

    pub fn bind_language(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn LanguageModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Language, &scope)?;
        self.language = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    pub fn bind_embedding(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn EmbeddingModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Embedding, &scope)?;
        self.embedding = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    pub fn bind_rerank(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn RerankModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Rerank, &scope)?;
        self.rerank = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    pub fn bind_image(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn ImageModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Image, &scope)?;
        self.image = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    pub fn bind_speech(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn SpeechModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Speech, &scope)?;
        self.speech = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    pub fn bind_transcription(
        mut self,
        scope: impl Into<Arc<ProviderScope>>,
        factory: ModelFactory<dyn TranscriptionModel>,
    ) -> Result<Self, ProviderRegistrationError> {
        let scope = scope.into();
        self.validate_binding(ModelFamily::Transcription, &scope)?;
        self.transcription = Some(FamilyRegistration::new(scope, factory));
        Ok(self)
    }

    /// Combine disjoint default family bindings for the same canonical provider.
    pub fn merge(mut self, mut other: Self) -> Result<Self, ProviderRegistrationError> {
        if self.provider_id != other.provider_id {
            return Err(ProviderRegistrationError::ProviderMismatch {
                expected: self.provider_id,
                found: other.provider_id,
            });
        }
        merge_family(
            ModelFamily::Language,
            &mut self.language,
            &mut other.language,
        )?;
        merge_family(
            ModelFamily::Embedding,
            &mut self.embedding,
            &mut other.embedding,
        )?;
        merge_family(ModelFamily::Rerank, &mut self.rerank, &mut other.rerank)?;
        merge_family(ModelFamily::Image, &mut self.image, &mut other.image)?;
        merge_family(ModelFamily::Speech, &mut self.speech, &mut other.speech)?;
        merge_family(
            ModelFamily::Transcription,
            &mut self.transcription,
            &mut other.transcription,
        )?;
        Ok(self)
    }

    pub fn language_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn LanguageModel>, ModelLookupError> {
        let binding = self
            .language
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Language))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Language,
            (binding.factory)(model)?,
        )
    }

    pub fn embedding_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn EmbeddingModel>, ModelLookupError> {
        let binding = self
            .embedding
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Embedding))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Embedding,
            (binding.factory)(model)?,
        )
    }

    pub fn rerank_model(&self, model: ModelId) -> Result<Arc<dyn RerankModel>, ModelLookupError> {
        let binding = self
            .rerank
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Rerank))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Rerank,
            (binding.factory)(model)?,
        )
    }

    pub fn image_model(&self, model: ModelId) -> Result<Arc<dyn ImageModel>, ModelLookupError> {
        let binding = self
            .image
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Image))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Image,
            (binding.factory)(model)?,
        )
    }

    pub fn speech_model(&self, model: ModelId) -> Result<Arc<dyn SpeechModel>, ModelLookupError> {
        let binding = self
            .speech
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Speech))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Speech,
            (binding.factory)(model)?,
        )
    }

    pub fn transcription_model(
        &self,
        model: ModelId,
    ) -> Result<Arc<dyn TranscriptionModel>, ModelLookupError> {
        let binding = self
            .transcription
            .as_ref()
            .ok_or_else(|| self.unsupported(ModelFamily::Transcription))?;
        let requested = model.clone();
        Self::validate_model(
            &binding.scope,
            requested,
            ModelFamily::Transcription,
            (binding.factory)(model)?,
        )
    }

    fn unsupported(&self, family: ModelFamily) -> ModelLookupError {
        ModelLookupError::UnsupportedFamily {
            provider: self.provider_id.clone(),
            family,
        }
    }

    fn validate_binding(
        &self,
        family: ModelFamily,
        scope: &ProviderScope,
    ) -> Result<(), ProviderRegistrationError> {
        if scope.provider_id() != &self.provider_id {
            return Err(ProviderRegistrationError::ProviderMismatch {
                expected: self.provider_id.clone(),
                found: scope.provider_id().clone(),
            });
        }
        if let Some(existing) = self.scope(family) {
            return Err(ProviderRegistrationError::DuplicateFamily {
                family,
                existing: Box::new(existing.clone()),
                incoming: Box::new(scope.clone()),
            });
        }
        Ok(())
    }

    fn validate_model<T: Model + ?Sized>(
        expected_scope: &Arc<ProviderScope>,
        expected_model: ModelId,
        expected_family: ModelFamily,
        model: Arc<T>,
    ) -> Result<Arc<T>, ModelLookupError> {
        let descriptor = model.descriptor();
        if descriptor.scope() == expected_scope.as_ref()
            && descriptor.model() == &expected_model
            && descriptor.family() == expected_family
        {
            return Ok(model);
        }

        let expected = ModelDescriptor::from_scope(
            expected_scope.clone(),
            expected_model,
            expected_family,
            descriptor.instance_id().clone(),
        );

        Err(ModelLookupError::IdentityMismatch {
            expected: Box::new(expected),
            actual: Box::new(descriptor.clone()),
        })
    }
}

fn merge_family<T: ?Sized>(
    family: ModelFamily,
    target: &mut Option<FamilyRegistration<T>>,
    source: &mut Option<FamilyRegistration<T>>,
) -> Result<(), ProviderRegistrationError> {
    if let (Some(existing), Some(incoming)) = (target.as_ref(), source.as_ref()) {
        return Err(ProviderRegistrationError::DuplicateFamily {
            family,
            existing: Box::new(existing.scope.as_ref().clone()),
            incoming: Box::new(incoming.scope.as_ref().clone()),
        });
    }
    if target.is_none() {
        *target = source.take();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn replay_domains_fail_closed_and_match_every_material_dimension() {
        let base = ProviderScope::new(ProviderId::new("openai").unwrap())
            .with_platform(PlatformId::new("public-api").unwrap())
            .with_protocol(ProtocolId::new("openai-responses").unwrap())
            .with_api_mode(ApiModeId::new("responses").unwrap());
        let official = ReplayDomain::official(ReplayDomainId::new("public-api").unwrap())
            .with_caller_scope(ReplayDomainId::new("account-a").unwrap());
        let matching = base.clone().with_replay_domain(official.clone());

        assert!(!base.shares_replay_domain(&base));
        assert!(matching.shares_replay_domain(&matching));
        assert!(
            !matching.shares_replay_domain(&base.clone().with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("public-api").unwrap(),
            )))
        );
        assert!(
            !matching.shares_replay_domain(
                &base.clone().with_replay_domain(
                    ReplayDomain::official(ReplayDomainId::new("public-api").unwrap())
                        .with_caller_scope(ReplayDomainId::new("account-b").unwrap()),
                )
            )
        );
        assert!(
            !matching.shares_replay_domain(
                &ProviderScope::new(ProviderId::new("openai").unwrap())
                    .with_platform(PlatformId::new("public-api").unwrap())
                    .with_protocol(ProtocolId::new("openai-chat-completions").unwrap())
                    .with_api_mode(ApiModeId::new("responses").unwrap())
                    .with_replay_domain(official),
            )
        );
    }

    #[test]
    fn identifier_deserialization_reuses_validation_and_normalization() {
        let provider: ProviderId = serde_json::from_str("\" OpenAI \"").unwrap();
        assert_eq!(provider.as_str(), "openai");
        assert!(serde_json::from_str::<RouteId>("\"primary:model\"").is_err());
        assert!(serde_json::from_str::<ModelId>("\"\\n\"").is_err());
    }

    #[test]
    fn missing_family_is_a_typed_lookup_error() {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("custom").unwrap()));
        let registration = ProviderRegistration::from_language(
            scope,
            Arc::new(|_| unreachable!("factory is not called")),
        );
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
    fn family_availability_is_factory_presence() {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("custom").unwrap()));
        let registration = ProviderRegistration::from_language(
            scope,
            Arc::new(|_| unreachable!("factory is not called")),
        );

        assert!(registration.supports_family(ModelFamily::Language));
        assert!(!registration.supports_family(ModelFamily::Image));
        assert_eq!(
            registration.families().collect::<Vec<_>>(),
            [ModelFamily::Language]
        );
    }

    #[test]
    fn registration_carries_full_protocol_context() {
        let scope = Arc::new(
            ProviderScope::new(ProviderId::new("custom").unwrap())
                .with_platform(PlatformId::new("public-api").unwrap())
                .with_protocol(ProtocolId::new("native").unwrap())
                .with_api_mode(ApiModeId::new("responses").unwrap()),
        );
        let registration = ProviderRegistration::from_language(
            scope,
            Arc::new(|_| unreachable!("factory is not called")),
        );

        assert_eq!(
            registration
                .platform(ModelFamily::Language)
                .map(PlatformId::as_str),
            Some("public-api")
        );
        assert_eq!(
            registration
                .protocol(ModelFamily::Language)
                .map(ProtocolId::as_str),
            Some("native")
        );
    }

    #[test]
    fn registration_keeps_distinct_scopes_per_family() {
        let language_scope = Arc::new(
            ProviderScope::new(ProviderId::new("composite").unwrap())
                .with_protocol(ProtocolId::new("openai").unwrap())
                .with_api_mode(ApiModeId::new("chat-completions").unwrap()),
        );
        let transcription_scope = Arc::new(
            ProviderScope::new(ProviderId::new("composite").unwrap())
                .with_protocol(ProtocolId::new("native-audio").unwrap())
                .with_api_mode(ApiModeId::new("transcriptions").unwrap()),
        );
        let registration = ProviderRegistration::from_language(
            language_scope.clone(),
            Arc::new(|_| unreachable!("factory is not called")),
        )
        .bind_transcription(
            transcription_scope.clone(),
            Arc::new(|_| unreachable!("factory is not called")),
        )
        .unwrap();

        assert_eq!(
            registration.scope(ModelFamily::Language),
            Some(language_scope.as_ref())
        );
        assert_eq!(
            registration.scope(ModelFamily::Transcription),
            Some(transcription_scope.as_ref())
        );
        assert_ne!(
            registration.api_mode(ModelFamily::Language),
            registration.api_mode(ModelFamily::Transcription)
        );
    }

    #[test]
    fn registration_rejects_cross_provider_and_duplicate_family_bindings() {
        let scope = Arc::new(ProviderScope::new(ProviderId::new("one").unwrap()));
        let registration = ProviderRegistration::from_language(
            scope.clone(),
            Arc::new(|_| unreachable!("factory is not called")),
        );
        let other_scope = Arc::new(ProviderScope::new(ProviderId::new("two").unwrap()));
        assert!(matches!(
            registration.clone().bind_embedding(
                other_scope,
                Arc::new(|_| unreachable!("factory is not called")),
            ),
            Err(ProviderRegistrationError::ProviderMismatch { .. })
        ));
        assert!(matches!(
            registration.bind_language(scope, Arc::new(|_| unreachable!("factory is not called")),),
            Err(ProviderRegistrationError::DuplicateFamily {
                family: ModelFamily::Language,
                ..
            })
        ));
    }

    #[test]
    fn operations_determine_their_model_family() {
        let cases = [
            (ModelOperation::Generate, ModelFamily::Language),
            (ModelOperation::Stream, ModelFamily::Language),
            (ModelOperation::Embed, ModelFamily::Embedding),
            (ModelOperation::Rerank, ModelFamily::Rerank),
            (ModelOperation::GenerateImage, ModelFamily::Image),
            (ModelOperation::SynthesizeSpeech, ModelFamily::Speech),
            (ModelOperation::Transcribe, ModelFamily::Transcription),
        ];

        for (operation, family) in cases {
            assert_eq!(operation.family(), family);
        }
    }

    #[test]
    fn combined_registration_can_be_narrowed_to_one_family() {
        let provider = ProviderId::new("composite").unwrap();
        let language_scope = ProviderScope::new(provider.clone())
            .with_api_mode(ApiModeId::new("responses").unwrap());
        let embedding_scope =
            ProviderScope::new(provider).with_api_mode(ApiModeId::new("embeddings").unwrap());
        let registration = ProviderRegistration::from_language(
            language_scope,
            Arc::new(|_| unreachable!("factory is not called")),
        )
        .bind_embedding(
            embedding_scope,
            Arc::new(|_| unreachable!("factory is not called")),
        )
        .unwrap();

        let embedding = registration.for_family(ModelFamily::Embedding).unwrap();
        assert_eq!(
            embedding.families().collect::<Vec<_>>(),
            [ModelFamily::Embedding]
        );
        assert!(registration.for_family(ModelFamily::Speech).is_none());
    }

    #[test]
    fn merge_rejects_cross_provider_and_same_family_scopes() {
        let chat_scope = ProviderScope::new(ProviderId::new("one").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap());
        let responses_scope = ProviderScope::new(ProviderId::new("one").unwrap())
            .with_api_mode(ApiModeId::new("responses").unwrap());
        let chat = ProviderRegistration::from_language(
            chat_scope.clone(),
            Arc::new(|_| unreachable!("factory is not called")),
        );
        let responses = ProviderRegistration::from_language(
            responses_scope.clone(),
            Arc::new(|_| unreachable!("factory is not called")),
        );

        assert!(matches!(
            chat.clone().merge(responses),
            Err(ProviderRegistrationError::DuplicateFamily {
                family: ModelFamily::Language,
                existing,
                incoming,
            }) if *existing == chat_scope && *incoming == responses_scope
        ));

        let other = ProviderRegistration::from_embedding(
            ProviderScope::new(ProviderId::new("two").unwrap()),
            Arc::new(|_| unreachable!("factory is not called")),
        );
        assert!(matches!(
            chat.merge(other),
            Err(ProviderRegistrationError::ProviderMismatch { .. })
        ));
    }
}
