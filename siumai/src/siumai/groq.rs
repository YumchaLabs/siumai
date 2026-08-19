use std::fmt;

use siumai_provider_groq::{GroqConfigError, GroqCredential, GroqProvider, GroqProviderBuilder};

use super::{Siumai, SiumaiBuilder};

/// The Groq construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().groq().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .groq()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.embedding("unsupported-model");
/// ```
#[must_use = "supply a Groq credential to continue provider construction"]
pub struct GroqCredentialStage {
    _private: (),
}

impl GroqCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Groq API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> GroqProviderStage {
        self.credential(GroqCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: GroqCredential) -> GroqProviderStage {
        GroqProviderStage::new(GroqProvider::builder(credential))
    }
}

impl fmt::Debug for GroqCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GroqCredentialStage")
    }
}

/// A buildable Groq stage wrapping the real [`GroqProviderBuilder`].
#[must_use = "build the Groq provider or continue configuring it"]
pub struct GroqProviderStage {
    builder: GroqProviderBuilder,
}

impl GroqProviderStage {
    fn new(builder: GroqProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Groq builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(GroqProviderBuilder) -> GroqProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<GroqProvider>, GroqConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for GroqProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroqProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Groq and enter its credential-required stage.
    pub const fn groq(self) -> GroqCredentialStage {
        GroqCredentialStage::new()
    }
}
