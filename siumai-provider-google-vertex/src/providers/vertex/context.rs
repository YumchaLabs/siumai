//! Vertex ProviderContext helpers.
//!
//! Mirrors Gemini's pattern:
//! - `ProviderContext.api_key` is used only for express mode (query param `key`).
//! - Bearer auth is injected via `Authorization` header from `token_provider` when available.

use crate::core::ProviderContext;

use super::client::GoogleVertexConfig;

pub fn build_base_context(config: &GoogleVertexConfig) -> ProviderContext {
    ProviderContext::new(
        "vertex",
        config.base_url.clone(),
        config.api_key.clone(),
        config.http_config.headers.clone(),
    )
}

pub async fn build_context(config: &GoogleVertexConfig) -> ProviderContext {
    let mut ctx = build_base_context(config);
    if let Some(tp) = &config.token_provider
        && let Ok(tok) = tp.token().await
    {
        ctx.http_extra_headers
            .insert("Authorization".to_string(), format!("Bearer {tok}"));
    }
    ctx
}
