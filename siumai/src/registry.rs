//! Optional immutable routing plus facade-owned provider integration.
//!
//! The underlying Registry remains provider-agnostic. This module adds only
//! feature-gated adapters from configured provider values to their
//! provider-owned registration descriptors.
//!
//! Route and model option defaults are intentionally not stored here. The U8
//! runtime composes those typed layers, while each provider owns its merge and
//! validation semantics.

use std::fmt;

pub use siumai_registry::{
    ModelReference, ModelReferenceError, ProviderRegistration, Registry, RegistryBuildError,
    RegistryBuilder, RegistryMiddleware, RegistryModelContext, RegistryResolveError,
    RegistrySnapshot, RouteId,
};

/// A configured provider that exposes one recommended Registry registration.
///
/// The registration may combine disjoint default model families, each with its
/// own exact scope and policy. Providers with multiple API modes for the same
/// family select only their documented default here; additional modes remain
/// explicit provider-owned registrations.
pub trait ProviderRegistrationSource {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError>;
}

/// Failure to obtain or add a provider's recommended Registry registration.
#[derive(Debug)]
#[non_exhaustive]
pub enum RegisterProviderError {
    /// The configured provider exposes only provider-native resources or jobs.
    NoPortableFamilyRegistration { provider: siumai_core::ProviderId },
    /// Registry rejected the caller-owned route or registration.
    Registry(RegistryBuildError),
}

impl fmt::Display for RegisterProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPortableFamilyRegistration { provider } => write!(
                formatter,
                "configured provider `{provider}` exposes no portable model-family registration"
            ),
            Self::Registry(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for RegisterProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoPortableFamilyRegistration { .. } => None,
            Self::Registry(source) => Some(source),
        }
    }
}

impl From<RegistryBuildError> for RegisterProviderError {
    fn from(source: RegistryBuildError) -> Self {
        Self::Registry(source)
    }
}

/// Ergonomic facade integration for configured providers.
pub trait RegistryBuilderExt {
    fn register_provider<S>(
        &mut self,
        route: impl AsRef<str>,
        provider: &S,
    ) -> Result<&mut Self, RegisterProviderError>
    where
        S: ProviderRegistrationSource;
}

impl RegistryBuilderExt for RegistryBuilder {
    fn register_provider<S>(
        &mut self,
        route: impl AsRef<str>,
        provider: &S,
    ) -> Result<&mut Self, RegisterProviderError>
    where
        S: ProviderRegistrationSource,
    {
        let registration = provider.provider_registration()?;
        self.register_named(route, registration)
            .map_err(RegisterProviderError::from)
    }
}

#[cfg(feature = "openai")]
impl ProviderRegistrationSource for siumai_provider_openai::configured::OpenAiProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "alibaba")]
impl ProviderRegistrationSource for siumai_provider_alibaba::AlibabaProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        self.registration()
            .ok_or_else(|| RegisterProviderError::NoPortableFamilyRegistration {
                provider: siumai_core::Provider::provider_id(self).clone(),
            })
    }
}

#[cfg(feature = "anthropic")]
impl ProviderRegistrationSource for siumai_provider_anthropic::AnthropicProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "openai-compatible")]
impl ProviderRegistrationSource for siumai_openai_compatible::OpenAiCompatibleProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "google")]
impl ProviderRegistrationSource for siumai_provider_gemini::GoogleImageProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "google-vertex-anthropic")]
impl ProviderRegistrationSource for siumai_provider_google_vertex::GoogleVertexAnthropicProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "cohere")]
impl ProviderRegistrationSource for siumai_provider_cohere::CohereProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "elevenlabs")]
impl ProviderRegistrationSource for siumai_provider_elevenlabs::configured::ElevenLabsProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "deepgram")]
impl ProviderRegistrationSource for siumai_provider_deepgram::DeepgramProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "deepseek")]
impl ProviderRegistrationSource for siumai_provider_deepseek::DeepSeekProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "minimax")]
impl ProviderRegistrationSource for siumai_provider_minimax::MinimaxProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "groq")]
impl ProviderRegistrationSource for siumai_provider_groq::GroqProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}

#[cfg(feature = "xai")]
impl ProviderRegistrationSource for siumai_provider_xai::XaiProvider {
    fn provider_registration(&self) -> Result<ProviderRegistration, RegisterProviderError> {
        Ok(self.registration())
    }
}
