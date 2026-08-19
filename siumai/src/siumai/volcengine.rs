use std::fmt;

use siumai_provider_volcengine::{
    VolcengineConfigError, VolcengineCredential, VolcengineProvider, VolcengineProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The Volcengine construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().volcengine().build();
/// ```
#[must_use = "supply a Volcengine credential to continue provider construction"]
pub struct VolcengineCredentialStage {
    _private: (),
}

impl VolcengineCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Volcengine API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> VolcengineProviderStage {
        self.credential(VolcengineCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: VolcengineCredential) -> VolcengineProviderStage {
        VolcengineProviderStage::new(VolcengineProvider::builder(credential))
    }
}

impl fmt::Debug for VolcengineCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VolcengineCredentialStage")
    }
}

/// A buildable Volcengine stage wrapping the real [`VolcengineProviderBuilder`].
#[must_use = "build the Volcengine provider or continue configuring it"]
pub struct VolcengineProviderStage {
    builder: VolcengineProviderBuilder,
}

impl VolcengineProviderStage {
    fn new(builder: VolcengineProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Volcengine builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(VolcengineProviderBuilder) -> VolcengineProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<VolcengineProvider>, VolcengineConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for VolcengineProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VolcengineProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Volcengine ARK and enter its credential-required stage.
    pub const fn volcengine(self) -> VolcengineCredentialStage {
        VolcengineCredentialStage::new()
    }
}
