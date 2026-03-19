//! Provider-owned typed response metadata.

pub mod azure;

pub use azure::{
    AzureChatResponseExt, AzureContentPartExt, AzureContentPartMetadata, AzureMetadata,
    AzureSource, AzureSourceExt, AzureSourceMetadata,
};
