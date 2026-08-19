use std::fmt;

use siumai_core::ModelLookupError;
use siumai_provider_gemini::{
    GeminiConfigError, GeminiCredential, GeminiGenerateContentModel, GeminiProvider,
    GeminiProviderBuilder,
};

use super::{LanguageClient, Siumai, SiumaiBuilder};

/// The Gemini construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().gemini().build();
/// ```
#[must_use = "supply a Gemini credential to continue provider construction"]
pub struct GeminiCredentialStage {
    _private: (),
}

impl GeminiCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use a Gemini API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> GeminiProviderStage {
        self.credential(GeminiCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: GeminiCredential) -> GeminiProviderStage {
        GeminiProviderStage::new(GeminiProvider::builder(credential))
    }
}

impl fmt::Debug for GeminiCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GeminiCredentialStage")
    }
}

/// A buildable Gemini stage wrapping the real [`GeminiProviderBuilder`].
#[must_use = "build the Gemini provider or continue configuring it"]
pub struct GeminiProviderStage {
    builder: GeminiProviderBuilder,
}

impl GeminiProviderStage {
    fn new(builder: GeminiProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Gemini builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(GeminiProviderBuilder) -> GeminiProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<GeminiProvider>, GeminiConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for GeminiProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Google Gemini and enter its credential-required stage.
    pub const fn gemini(self) -> GeminiCredentialStage {
        GeminiCredentialStage::new()
    }
}

impl Siumai<GeminiProvider> {
    /// Bind an explicit Gemini Generate Content model.
    ///
    /// [`Self::language`] remains the canonical Interactions path.
    pub fn generate_content(
        &self,
        model: impl Into<String>,
    ) -> Result<LanguageClient<GeminiProvider, GeminiGenerateContentModel>, ModelLookupError> {
        let model = self.provider().generate_content(model)?;
        Ok(LanguageClient::new(self.provider.clone(), model))
    }
}
