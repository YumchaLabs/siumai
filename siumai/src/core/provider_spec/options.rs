//! Provider options helpers for core provider spec.
//!
//! This module contains shared hooks for injecting custom provider
//! options into outbound JSON bodies. It is used by `ProviderSpec`
//! default implementations for chat and embedding requests.

use crate::error::LlmError;
use crate::execution::executors::BeforeSendHook;
use crate::types::{ChatRequest, EmbeddingRequest, ProviderOptions};
use std::sync::Arc;

/// Default hook for `ProviderOptions::Custom` injection (chat).
///
/// This provides a standard implementation for injecting custom
/// provider options into the request JSON body. All providers
/// automatically support `CustomProviderOptions` through this
/// function when they use the default `chat_before_send`.
pub fn default_custom_options_hook(provider_id: &str, req: &ChatRequest) -> Option<BeforeSendHook> {
    if let ProviderOptions::Custom {
        provider_id: custom_provider_id,
        options,
    } = &req.provider_options
        && crate::registry::helpers::matches_provider_id(provider_id, custom_provider_id)
    {
        let custom_options = options.clone();
        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(obj) = out.as_object_mut() {
                // Merge custom options into the request body
                for (k, v) in &custom_options {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Ok(out)
        };
        return Some(Arc::new(hook));
    }
    None
}

/// Default hook for injecting custom provider options into embedding requests.
///
/// Mirrors [`default_custom_options_hook`] for chat, but reads from
/// `EmbeddingRequest::provider_options`. This enables user-defined provider
/// options to be merged into the outbound JSON body for embeddings.
pub fn default_custom_options_hook_embedding(
    provider_id: &str,
    req: &EmbeddingRequest,
) -> Option<BeforeSendHook> {
    if let ProviderOptions::Custom {
        provider_id: custom_provider_id,
        options,
    } = &req.provider_options
        && crate::registry::helpers::matches_provider_id(provider_id, custom_provider_id)
    {
        let custom_options = options.clone();
        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(obj) = out.as_object_mut() {
                for (k, v) in &custom_options {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Ok(out)
        };
        return Some(Arc::new(hook));
    }
    None
}
