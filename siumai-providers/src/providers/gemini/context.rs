//! Gemini ProviderContext helpers
//!
//! Centralizes how we build ProviderContext for Gemini, including async token_provider injection
//! (Vertex/enterprise Bearer auth) via http_extra_headers.

use crate::core::ProviderContext;
use secrecy::ExposeSecret;

use super::types::GeminiConfig;

pub fn build_base_context(config: &GeminiConfig) -> ProviderContext {
    ProviderContext::new(
        "gemini",
        config.base_url.clone(),
        Some(config.api_key.expose_secret().to_string()),
        config.http_config.headers.clone(),
    )
}

pub async fn build_context(config: &GeminiConfig) -> ProviderContext {
    // Gemini token_provider requires async access; we inject the Bearer token into ProviderContext
    // so the header builder can skip x-goog-api-key.
    let mut ctx = build_base_context(config);
    if let Some(tp) = &config.token_provider
        && let Ok(tok) = tp.token().await
    {
        ctx.http_extra_headers
            .insert("Authorization".to_string(), format!("Bearer {tok}"));
    }
    ctx
}
