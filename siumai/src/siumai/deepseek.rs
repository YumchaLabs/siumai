use std::fmt;

use siumai_provider_deepseek::{
    DeepSeekConfigError, DeepSeekCredential, DeepSeekProvider, DeepSeekProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The DeepSeek construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().deepseek().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .deepseek()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.embedding("unsupported-model");
/// ```
#[must_use = "supply a DeepSeek credential to continue provider construction"]
pub struct DeepSeekCredentialStage {
    _private: (),
}

impl DeepSeekCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a DeepSeek API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> DeepSeekProviderStage {
        self.credential(DeepSeekCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: DeepSeekCredential) -> DeepSeekProviderStage {
        DeepSeekProviderStage::new(DeepSeekProvider::builder(credential))
    }
}

impl fmt::Debug for DeepSeekCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DeepSeekCredentialStage")
    }
}

/// A buildable DeepSeek stage wrapping the real [`DeepSeekProviderBuilder`].
#[must_use = "build the DeepSeek provider or continue configuring it"]
pub struct DeepSeekProviderStage {
    builder: DeepSeekProviderBuilder,
}

impl DeepSeekProviderStage {
    fn new(builder: DeepSeekProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real DeepSeek builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(DeepSeekProviderBuilder) -> DeepSeekProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<DeepSeekProvider>, DeepSeekConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for DeepSeekProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepSeekProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select DeepSeek and enter its credential-required stage.
    pub const fn deepseek(self) -> DeepSeekCredentialStage {
        DeepSeekCredentialStage::new()
    }
}
