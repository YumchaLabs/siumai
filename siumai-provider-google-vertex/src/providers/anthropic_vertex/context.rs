//! Anthropic-on-Vertex ProviderContext helpers.
//!
//! Mirrors the Google Vertex pattern:
//! - auth is ultimately expressed as `Authorization: Bearer ...`
//! - user-supplied `Authorization` must win
//! - token providers can lazily inject a Bearer token into the runtime context

use crate::core::ProviderContext;

use super::client::VertexAnthropicConfig;

fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization"))
}

pub fn build_base_context(config: &VertexAnthropicConfig) -> ProviderContext {
    ProviderContext::new(
        "anthropic-vertex",
        config.base_url.clone(),
        None,
        config.http_config.headers.clone(),
    )
}

pub async fn build_context(config: &VertexAnthropicConfig) -> ProviderContext {
    let mut ctx = build_base_context(config);

    if !has_auth_header(&ctx.http_extra_headers)
        && let Some(tp) = &config.token_provider
        && let Ok(token) = tp.token().await
    {
        ctx.http_extra_headers
            .insert("Authorization".to_string(), format!("Bearer {token}"));
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
        let cfg = VertexAnthropicConfig::new("https://example", "claude")
            .with_token_provider(Arc::new(StaticTokenProvider::new("tok")));

        let ctx = build_context(&cfg).await;
        assert_eq!(
            ctx.http_extra_headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer tok")
        );
    }

    #[tokio::test]
    async fn build_context_does_not_override_authorization_header() {
        let mut http_config = crate::types::HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer user".to_string());

        let cfg = VertexAnthropicConfig::new("https://example", "claude")
            .with_http_config(http_config)
            .with_token_provider(Arc::new(StaticTokenProvider::new("tok")));

        let ctx = build_context(&cfg).await;
        assert_eq!(
            ctx.http_extra_headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer user")
        );
    }
}
