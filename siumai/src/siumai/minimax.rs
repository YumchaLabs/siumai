use std::fmt;

use siumai_provider_minimax::{
    MinimaxConfigError, MinimaxCredential, MinimaxProvider, MinimaxProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The MiniMax construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().minimax().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .minimax()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.embedding("unsupported-model");
/// ```
#[must_use = "supply a MiniMax credential to continue provider construction"]
pub struct MinimaxCredentialStage {
    _private: (),
}

impl MinimaxCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a MiniMax API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> MinimaxProviderStage {
        self.credential(MinimaxCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: MinimaxCredential) -> MinimaxProviderStage {
        MinimaxProviderStage::new(MinimaxProvider::builder(credential))
    }
}

impl fmt::Debug for MinimaxCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MinimaxCredentialStage")
    }
}

/// A buildable MiniMax stage wrapping the real [`MinimaxProviderBuilder`].
#[must_use = "build the MiniMax provider or continue configuring it"]
pub struct MinimaxProviderStage {
    builder: MinimaxProviderBuilder,
}

impl MinimaxProviderStage {
    fn new(builder: MinimaxProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real MiniMax builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(MinimaxProviderBuilder) -> MinimaxProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<MinimaxProvider>, MinimaxConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for MinimaxProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select MiniMax and enter its credential-required stage.
    pub const fn minimax(self) -> MinimaxCredentialStage {
        MinimaxCredentialStage::new()
    }
}
