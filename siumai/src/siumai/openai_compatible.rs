use std::fmt;

use siumai_openai_compatible::{
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleProfile,
    OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The OpenAI-compatible construction stage that requires an explicit profile.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().openai_compatible().build();
/// ```
#[must_use = "supply an OpenAI-compatible profile to continue provider construction"]
pub struct OpenAiCompatibleProfileStage {
    _private: (),
}

impl OpenAiCompatibleProfileStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Select a verified or explicit custom provider-owned compatibility profile.
    pub fn profile(self, profile: OpenAiCompatibleProfile) -> OpenAiCompatibleCredentialStage {
        OpenAiCompatibleCredentialStage { profile }
    }
}

impl fmt::Debug for OpenAiCompatibleProfileStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpenAiCompatibleProfileStage")
    }
}

/// The OpenAI-compatible stage that requires a credential for the selected profile.
///
/// ```compile_fail
/// use siumai::Siumai;
/// use siumai::providers::openai_compatible::OpenAiCompatibleProfile;
///
/// let profile: OpenAiCompatibleProfile = todo!();
/// let _ = Siumai::builder()
///     .openai_compatible()
///     .profile(profile)
///     .build();
/// ```
#[must_use = "supply an OpenAI-compatible credential to continue provider construction"]
pub struct OpenAiCompatibleCredentialStage {
    profile: OpenAiCompatibleProfile,
}

impl OpenAiCompatibleCredentialStage {
    /// Use a static bearer API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> OpenAiCompatibleProviderStage {
        self.credential(OpenAiCompatibleCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(
        self,
        credential: OpenAiCompatibleCredential,
    ) -> OpenAiCompatibleProviderStage {
        OpenAiCompatibleProviderStage::new(OpenAiCompatibleProvider::builder(
            self.profile,
            credential,
        ))
    }
}

impl fmt::Debug for OpenAiCompatibleCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleCredentialStage")
            .field("profile_id", self.profile.provider_profile().id())
            .finish_non_exhaustive()
    }
}

/// A buildable compatible-provider stage wrapping the real provider builder.
#[must_use = "build the OpenAI-compatible provider or continue configuring it"]
pub struct OpenAiCompatibleProviderStage {
    builder: OpenAiCompatibleProviderBuilder,
}

impl OpenAiCompatibleProviderStage {
    fn new(builder: OpenAiCompatibleProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real compatible-provider builder without copying its setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(OpenAiCompatibleProviderBuilder) -> OpenAiCompatibleProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<OpenAiCompatibleProvider>, OpenAiCompatibleConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for OpenAiCompatibleProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select an explicit OpenAI-compatible profile.
    pub const fn openai_compatible(self) -> OpenAiCompatibleProfileStage {
        OpenAiCompatibleProfileStage::new()
    }
}
