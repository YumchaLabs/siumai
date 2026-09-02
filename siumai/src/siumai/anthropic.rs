use std::fmt;

use siumai_provider_anthropic::{
    AnthropicConfigError, AnthropicCredential, AnthropicProvider, AnthropicProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The Anthropic construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().anthropic().build();
/// ```
#[must_use = "supply an Anthropic credential to continue provider construction"]
pub struct AnthropicCredentialStage {
    _private: (),
}

impl AnthropicCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use an Anthropic API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> AnthropicProviderStage {
        self.credential(AnthropicCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: AnthropicCredential) -> AnthropicProviderStage {
        AnthropicProviderStage::new(AnthropicProvider::builder(credential))
    }
}

impl fmt::Debug for AnthropicCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AnthropicCredentialStage")
    }
}

/// A buildable Anthropic stage wrapping the real [`AnthropicProviderBuilder`].
#[must_use = "build the Anthropic provider or continue configuring it"]
pub struct AnthropicProviderStage {
    builder: AnthropicProviderBuilder,
}

impl AnthropicProviderStage {
    fn new(builder: AnthropicProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Anthropic builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(AnthropicProviderBuilder) -> AnthropicProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<AnthropicProvider>, AnthropicConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for AnthropicProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Anthropic and enter its credential-required stage.
    pub const fn anthropic(self) -> AnthropicCredentialStage {
        AnthropicCredentialStage::new()
    }
}
