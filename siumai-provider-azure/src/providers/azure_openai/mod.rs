#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod client;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod config;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub mod spec;

#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use client::AzureOpenAiClient;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use config::AzureOpenAiConfig;
#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use spec::{AzureChatMode, AzureOpenAiSpec, AzureUrlConfig};
