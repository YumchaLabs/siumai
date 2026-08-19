use std::fmt;

use siumai_provider_alibaba::{
    AlibabaConfigError, AlibabaCredential, AlibabaProvider, AlibabaProviderBuilder,
};

use super::{Siumai, SiumaiBuilder};

/// The Alibaba construction stage that requires a provider-owned credential.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().alibaba().build();
/// ```
#[must_use = "supply an Alibaba credential to continue provider construction"]
pub struct AlibabaCredentialStage {
    _private: (),
}

impl AlibabaCredentialStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Use an Alibaba API key and enter the required configuration stage.
    pub fn api_key(self, api_key: impl Into<String>) -> AlibabaConfigurationStage {
        self.credential(AlibabaCredential::api_key(api_key))
    }

    /// Use a complete provider-owned credential and enter the required configuration stage.
    pub fn credential(self, credential: AlibabaCredential) -> AlibabaConfigurationStage {
        AlibabaConfigurationStage::new(AlibabaProvider::builder(credential))
    }
}

impl fmt::Debug for AlibabaCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AlibabaCredentialStage")
    }
}

/// An Alibaba stage that requires one provider-builder configuration transition.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder()
///     .alibaba()
///     .api_key("test-api-key")
///     .build();
/// ```
#[must_use = "select at least one Alibaba endpoint through configure_provider"]
pub struct AlibabaConfigurationStage {
    builder: AlibabaProviderBuilder,
}

impl AlibabaConfigurationStage {
    fn new(builder: AlibabaProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Alibaba builder and enter the buildable stage.
    ///
    /// The provider-owned builder remains responsible for verifying that the
    /// closure selected at least one valid endpoint.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(AlibabaProviderBuilder) -> AlibabaProviderBuilder,
    ) -> AlibabaProviderStage {
        AlibabaProviderStage::new(configure(self.builder))
    }
}

impl fmt::Debug for AlibabaConfigurationStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaConfigurationStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

/// A buildable Alibaba stage wrapping the real [`AlibabaProviderBuilder`].
#[must_use = "build the Alibaba provider"]
pub struct AlibabaProviderStage {
    builder: AlibabaProviderBuilder,
}

impl AlibabaProviderStage {
    fn new(builder: AlibabaProviderBuilder) -> Self {
        Self { builder }
    }

    /// Validate provider-owned endpoint configuration and build a typed hub.
    pub fn build(self) -> Result<Siumai<AlibabaProvider>, AlibabaConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for AlibabaProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Alibaba Cloud Model Studio and enter its credential-required stage.
    pub const fn alibaba(self) -> AlibabaCredentialStage {
        AlibabaCredentialStage::new()
    }
}
