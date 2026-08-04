//! Optional immutable routing plus facade-owned provider integration.
//!
//! The underlying Registry remains provider-agnostic. This module adds only
//! feature-gated adapters from configured provider values to their
//! provider-owned registration descriptors.
//!
//! Route and model option defaults are intentionally not stored here. The U8
//! runtime composes those typed layers, while each provider owns its merge and
//! validation semantics.

pub use siumai_registry::{
    ModelReference, ModelReferenceError, ProviderRegistration, Registry, RegistryBuildError,
    RegistryBuilder, RegistryMiddleware, RegistryModelContext, RegistryResolveError,
    RegistrySnapshot, RouteId,
};

/// A configured provider that exposes one recommended Registry registration.
///
/// Providers with multiple native API modes select only their documented
/// default here. Additional modes remain explicit provider-owned registrations.
pub trait ProviderRegistrationSource {
    fn provider_registration(&self) -> ProviderRegistration;
}

/// Ergonomic facade integration for configured providers.
pub trait RegistryBuilderExt {
    fn register_provider<S>(
        &mut self,
        route: impl AsRef<str>,
        provider: &S,
    ) -> Result<&mut Self, RegistryBuildError>
    where
        S: ProviderRegistrationSource;
}

impl RegistryBuilderExt for RegistryBuilder {
    fn register_provider<S>(
        &mut self,
        route: impl AsRef<str>,
        provider: &S,
    ) -> Result<&mut Self, RegistryBuildError>
    where
        S: ProviderRegistrationSource,
    {
        self.register_named(route, provider.provider_registration())
    }
}

#[cfg(feature = "openai")]
impl ProviderRegistrationSource for siumai_provider_openai::configured::OpenAiProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}

#[cfg(feature = "openai-compatible")]
impl ProviderRegistrationSource for siumai_provider_openai_compatible::OpenAiCompatibleProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}

#[cfg(feature = "google")]
impl ProviderRegistrationSource for siumai_provider_gemini::GoogleImagenProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}

#[cfg(feature = "cohere")]
impl ProviderRegistrationSource for siumai_provider_cohere::CohereProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}

#[cfg(feature = "elevenlabs")]
impl ProviderRegistrationSource for siumai_provider_elevenlabs::configured::ElevenLabsProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}

#[cfg(feature = "deepgram")]
impl ProviderRegistrationSource for siumai_provider_deepgram::DeepgramProvider {
    fn provider_registration(&self) -> ProviderRegistration {
        self.registration()
    }
}
