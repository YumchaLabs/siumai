//! Vertex ProviderContext helpers.
//!
//! Mirrors Gemini's pattern:
//! - `ProviderContext.api_key` is used only for express mode (query param `key`).
//! - Bearer auth is injected via `Authorization` header from `token_provider` when available.

use crate::core::ProviderContext;

use super::client::GoogleVertexConfig;

fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"))
}

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

    // Vercel AI SDK alignment (provider-node): auto-inject Bearer token when available,
    // but do not override user-supplied Authorization headers.
    if !has_auth_header(&ctx.http_extra_headers)
        && let Some(tp) = &config.token_provider
        && let Ok(tok) = tp.token().await
    {
        ctx.http_extra_headers
            .insert("Authorization".to_string(), format!("Bearer {tok}"));
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;
    use std::sync::Arc;

    #[tokio::test]
    async fn build_context_injects_bearer_token_when_missing() {
        let cfg = GoogleVertexConfig {
            base_url: "https://example".to_string(),
            model: "m".to_string(),
            api_key: None,
            http_config: crate::types::HttpConfig::default(),
            http_transport: None,
            token_provider: Some(Arc::new(StaticTokenProvider::new("tok"))),
        };

        let ctx = build_context(&cfg).await;
        assert_eq!(
            ctx.http_extra_headers
                .get("Authorization")
                .map(|s| s.as_str()),
            Some("Bearer tok")
        );
    }

    #[tokio::test]
    async fn build_context_does_not_override_authorization_header() {
        let mut http_config = crate::types::HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer user".to_string());

        let cfg = GoogleVertexConfig {
            base_url: "https://example".to_string(),
            model: "m".to_string(),
            api_key: None,
            http_config,
            http_transport: None,
            token_provider: Some(Arc::new(StaticTokenProvider::new("tok"))),
        };

        let ctx = build_context(&cfg).await;
        assert_eq!(
            ctx.http_extra_headers
                .get("Authorization")
                .map(|s| s.as_str()),
            Some("Bearer user")
        );
    }
}
