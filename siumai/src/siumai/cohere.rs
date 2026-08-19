use std::fmt;

use siumai_provider_cohere::{CohereConfigError, CohereProvider, CohereProviderBuilder};

use super::{Siumai, SiumaiBuilder};

/// The Cohere construction stage that requires the API key accepted by the real builder.
///
/// Cohere's provider builder intentionally accepts a key directly and does not expose a
/// provider-owned credential type, so the facade does not invent one.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().cohere().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().cohere().credential("test-api-key");
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .cohere()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.language("unsupported-model");
/// ```
#[must_use = "supply a Cohere API key to continue provider construction"]
pub struct CohereApiKeyStage {
    _private: (),
}

impl CohereApiKeyStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Cohere API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> CohereProviderStage {
        CohereProviderStage::new(CohereProvider::builder(api_key))
    }
}

impl fmt::Debug for CohereApiKeyStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CohereApiKeyStage")
    }
}

/// A buildable Cohere stage wrapping the real [`CohereProviderBuilder`].
#[must_use = "build the Cohere provider or continue configuring it"]
pub struct CohereProviderStage {
    builder: CohereProviderBuilder,
}

impl CohereProviderStage {
    fn new(builder: CohereProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Cohere builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(CohereProviderBuilder) -> CohereProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<CohereProvider>, CohereConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for CohereProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CohereProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Cohere and enter its API-key-required stage.
    pub const fn cohere(self) -> CohereApiKeyStage {
        CohereApiKeyStage::new()
    }
}
