use std::fmt;

use siumai_provider_elevenlabs::configured::{
    ElevenLabsConfigError, ElevenLabsCredential, ElevenLabsProfile, ElevenLabsProvider,
    ElevenLabsProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The ElevenLabs construction stage that requires an explicit provider profile.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().elevenlabs().api_key("test-api-key");
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().elevenlabs().build();
/// ```
#[must_use = "select an ElevenLabs profile before supplying credentials"]
pub struct ElevenLabsProfileStage {
    _private: (),
}

impl ElevenLabsProfileStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Select an ElevenLabs provider-owned profile.
    pub fn profile(self, profile: ElevenLabsProfile) -> ElevenLabsCredentialStage {
        ElevenLabsCredentialStage { profile }
    }
}

impl fmt::Debug for ElevenLabsProfileStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ElevenLabsProfileStage")
    }
}

/// The ElevenLabs stage that requires a credential for the selected profile.
///
/// ```compile_fail
/// use siumai::Siumai;
/// use siumai::providers::elevenlabs::ElevenLabsProfile;
///
/// let profile: ElevenLabsProfile = todo!();
/// let _ = Siumai::builder().elevenlabs().profile(profile).build();
/// ```
#[must_use = "supply an ElevenLabs credential to continue provider construction"]
pub struct ElevenLabsCredentialStage {
    profile: ElevenLabsProfile,
}

impl ElevenLabsCredentialStage {
    /// Use an ElevenLabs API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> ElevenLabsProviderStage {
        self.credential(ElevenLabsCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: ElevenLabsCredential) -> ElevenLabsProviderStage {
        ElevenLabsProviderStage::new(ElevenLabsProvider::builder(self.profile, credential))
    }
}

impl fmt::Debug for ElevenLabsCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElevenLabsCredentialStage")
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

/// A buildable ElevenLabs stage wrapping the real [`ElevenLabsProviderBuilder`].
#[must_use = "build the ElevenLabs provider or continue configuring it"]
pub struct ElevenLabsProviderStage {
    builder: ElevenLabsProviderBuilder,
}

impl ElevenLabsProviderStage {
    fn new(builder: ElevenLabsProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real ElevenLabs builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(ElevenLabsProviderBuilder) -> ElevenLabsProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<ElevenLabsProvider>, ElevenLabsConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for ElevenLabsProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElevenLabsProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select ElevenLabs and enter its profile-required stage.
    ///
    /// ```compile_fail
    /// use siumai::Siumai;
    /// use siumai::providers::elevenlabs::ElevenLabsProfile;
    ///
    /// let profile: ElevenLabsProfile = todo!();
    /// let hub = Siumai::builder()
    ///     .elevenlabs()
    ///     .profile(profile)
    ///     .api_key("test-api-key")
    ///     .build()
    ///     .unwrap();
    /// let _ = hub.language("unsupported-model");
    /// ```
    pub const fn elevenlabs(self) -> ElevenLabsProfileStage {
        ElevenLabsProfileStage::new()
    }
}
