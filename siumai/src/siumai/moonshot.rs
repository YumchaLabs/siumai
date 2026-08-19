use std::fmt;

use siumai_provider_moonshotai::{
    MoonshotConfigError, MoonshotCredential, MoonshotProvider, MoonshotProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The Moonshot AI construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().moonshot().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().moonshotai();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .moonshot()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.embedding("unsupported-model");
/// ```
#[must_use = "supply a Moonshot AI credential to continue provider construction"]
pub struct MoonshotCredentialStage {
    _private: (),
}

impl MoonshotCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Moonshot AI API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> MoonshotProviderStage {
        self.credential(MoonshotCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: MoonshotCredential) -> MoonshotProviderStage {
        MoonshotProviderStage::new(MoonshotProvider::builder(credential))
    }
}

impl fmt::Debug for MoonshotCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MoonshotCredentialStage")
    }
}

/// A buildable Moonshot AI stage wrapping the real [`MoonshotProviderBuilder`].
#[must_use = "build the Moonshot AI provider or continue configuring it"]
pub struct MoonshotProviderStage {
    builder: MoonshotProviderBuilder,
}

impl MoonshotProviderStage {
    fn new(builder: MoonshotProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Moonshot AI builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(MoonshotProviderBuilder) -> MoonshotProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<MoonshotProvider>, MoonshotConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for MoonshotProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoonshotProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Moonshot AI and enter its credential-required stage.
    pub const fn moonshot(self) -> MoonshotCredentialStage {
        MoonshotCredentialStage::new()
    }
}
