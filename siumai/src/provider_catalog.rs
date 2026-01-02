#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
pub use siumai_registry::provider_catalog::*;

#[cfg(not(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
)))]
mod no_providers {
    use siumai_core::traits::ProviderCapabilities;
    use siumai_core::types::ProviderType;
    use std::borrow::Cow;

    #[derive(Debug, Clone)]
    pub struct ProviderInfo {
        pub provider_type: ProviderType,
        pub name: Cow<'static, str>,
        pub description: Cow<'static, str>,
        pub capabilities: ProviderCapabilities,
        pub default_base_url: Cow<'static, str>,
        pub supported_models: Vec<Cow<'static, str>>,
    }

    pub fn get_supported_providers() -> Vec<ProviderInfo> {
        Vec::new()
    }

    pub fn get_provider_info(_provider_type: &ProviderType) -> Option<ProviderInfo> {
        None
    }

    pub fn get_provider_info_by_id(_provider_id: &str) -> Option<ProviderInfo> {
        None
    }

    pub fn is_model_supported(_provider_type: &ProviderType, _model: &str) -> bool {
        false
    }

    pub fn is_model_supported_by_id(_provider_id: &str, _model: &str) -> bool {
        false
    }
}

#[cfg(not(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
)))]
pub use no_providers::*;
