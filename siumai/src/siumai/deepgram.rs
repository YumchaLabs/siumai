use std::fmt;

use siumai_provider_deepgram::{
    DeepgramConfigError, DeepgramCredential, DeepgramProvider, DeepgramProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The Deepgram construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().deepgram().build();
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let hub = Siumai::builder()
///     .deepgram()
///     .api_key("test-api-key")
///     .build()
///     .unwrap();
/// let _ = hub.language("unsupported-model");
/// ```
#[must_use = "supply a Deepgram credential to continue provider construction"]
pub struct DeepgramCredentialStage {
    _private: (),
}

impl DeepgramCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Deepgram API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> DeepgramProviderStage {
        self.credential(DeepgramCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: DeepgramCredential) -> DeepgramProviderStage {
        DeepgramProviderStage::new(DeepgramProvider::builder(credential))
    }
}

impl fmt::Debug for DeepgramCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DeepgramCredentialStage")
    }
}

/// A buildable Deepgram stage wrapping the real [`DeepgramProviderBuilder`].
#[must_use = "build the Deepgram provider or continue configuring it"]
pub struct DeepgramProviderStage {
    builder: DeepgramProviderBuilder,
}

impl DeepgramProviderStage {
    fn new(builder: DeepgramProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Deepgram builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(DeepgramProviderBuilder) -> DeepgramProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<DeepgramProvider>, DeepgramConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for DeepgramProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepgramProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Deepgram and enter its credential-required stage.
    pub const fn deepgram(self) -> DeepgramCredentialStage {
        DeepgramCredentialStage::new()
    }
}
