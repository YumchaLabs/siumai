use std::fmt;

use siumai_core::ModelLookupError;
use siumai_provider_openai::{
    OpenAiChatCompletionsModel, OpenAiConfigError, OpenAiCredential, OpenAiProvider,
    OpenAiProviderBuilder,
};

use super::{LanguageClient, Siumai, SiumaiBuilder};

/// The OpenAI construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().openai().build();
/// ```
#[must_use = "supply an OpenAI credential to continue provider construction"]
pub struct OpenAiCredentialStage {
    _private: (),
}

impl OpenAiCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use an OpenAI API key and enter the buildable provider stage.
    pub fn api_key(self, api_key: impl Into<String>) -> OpenAiProviderStage {
        self.credential(OpenAiCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the buildable stage.
    pub fn credential(self, credential: OpenAiCredential) -> OpenAiProviderStage {
        OpenAiProviderStage::new(OpenAiProvider::builder(credential))
    }
}

impl fmt::Debug for OpenAiCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpenAiCredentialStage")
    }
}

/// A buildable OpenAI stage wrapping the real [`OpenAiProviderBuilder`].
#[must_use = "build the OpenAI provider or continue configuring it"]
pub struct OpenAiProviderStage {
    builder: OpenAiProviderBuilder,
}

impl OpenAiProviderStage {
    fn new(builder: OpenAiProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real OpenAI builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(OpenAiProviderBuilder) -> OpenAiProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(self) -> Result<Siumai<OpenAiProvider>, OpenAiConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for OpenAiProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select OpenAI and enter its credential-required stage.
    pub const fn openai(self) -> OpenAiCredentialStage {
        OpenAiCredentialStage::new()
    }
}

impl Siumai<OpenAiProvider> {
    /// Bind an explicit OpenAI Chat Completions model.
    ///
    /// [`Self::language`] remains the canonical Responses path.
    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<LanguageClient<OpenAiProvider, OpenAiChatCompletionsModel>, ModelLookupError> {
        let model = self.provider().chat_completions(model)?;
        Ok(LanguageClient::new(self.provider.clone(), model))
    }
}
