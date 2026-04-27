#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod builder;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod client;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod config;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod ext;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod settings;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod spec;

#[cfg(any(feature = "azure-standard", feature = "azure"))]
#[allow(deprecated)]
pub use crate::provider_options::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use crate::provider_options::{
    OpenAILanguageModelChatOptions, OpenAILanguageModelResponsesOptions,
};
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use builder::AzureOpenAiBuilder;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use client::AzureOpenAiClient;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use config::AzureOpenAiConfig;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use ext::AzureOpenAiChatRequestExt;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use settings::AzureOpenAIProviderSettings;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use spec::{AzureChatMode, AzureOpenAiSpec, AzureUrlConfig};
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
