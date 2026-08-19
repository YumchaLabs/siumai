use std::fmt;

use siumai_provider_xai::{XaiConfigError, XaiCredential, XaiProvider, XaiProviderBuilder};

use super::{Siumai, SiumaiBuilder};

/// The xAI construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().xai().build();
/// ```
#[must_use = "supply an xAI credential to continue provider construction"]
pub struct XaiCredentialStage {
    _private: (),
}

impl XaiCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use an xAI API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> XaiProviderStage {
        self.credential(XaiCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: XaiCredential) -> XaiProviderStage {
        XaiProviderStage::new(XaiProvider::builder(credential))
    }
}

impl fmt::Debug for XaiCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("XaiCredentialStage")
    }
}

/// A buildable xAI stage wrapping the real [`XaiProviderBuilder`].
#[must_use = "build the xAI provider or continue configuring it"]
pub struct XaiProviderStage {
    builder: XaiProviderBuilder,
}

impl XaiProviderStage {
    fn new(builder: XaiProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real xAI builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(XaiProviderBuilder) -> XaiProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<XaiProvider>, XaiConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for XaiProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select xAI and enter its credential-required stage.
    pub const fn xai(self) -> XaiCredentialStage {
        XaiCredentialStage::new()
    }
}
