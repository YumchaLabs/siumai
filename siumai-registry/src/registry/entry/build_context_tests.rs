use super::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedBuildContext {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    has_http_client: bool,
    has_http_transport: bool,
    user_agent: Option<String>,
    has_retry_options: bool,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
}

#[derive(Clone, Default)]
struct NoopTransport;

#[async_trait::async_trait]
impl crate::execution::http::transport::HttpTransport for NoopTransport {
    async fn execute_json(
        &self,
        _request: crate::execution::http::transport::HttpTransportRequest,
    ) -> Result<crate::execution::http::transport::HttpTransportResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "noop transport should not execute requests".to_string(),
        ))
    }
}

struct ContextCapturingFactory {
    id: &'static str,
    seen: Arc<Mutex<Option<ObservedBuildContext>>>,
}

#[async_trait::async_trait]
impl ProviderFactory for ContextCapturingFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(TestProvClient))
    }

    async fn compat_language_client_with_ctx(
        &self,
        _model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        *self.seen.lock().unwrap() = Some(ObservedBuildContext {
            provider_id: ctx.provider_id.clone(),
            api_key: ctx.api_key.clone(),
            base_url: ctx.base_url.clone(),
            has_http_client: ctx.http_client.is_some(),
            has_http_transport: ctx.http_transport.is_some(),
            user_agent: ctx
                .http_config
                .as_ref()
                .and_then(|config| config.user_agent.clone()),
            has_retry_options: ctx.retry_options.is_some(),
            reasoning_enabled: ctx.reasoning_enabled,
            reasoning_budget: ctx.reasoning_budget,
        });
        Ok(Arc::new(TestProvClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(self.id)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }
}

#[cfg(any(feature = "google", feature = "google-vertex"))]
#[test]
fn build_context_resolves_google_token_provider_with_backward_compatibility() {
    let google: Arc<dyn crate::auth::TokenProvider> =
        Arc::new(crate::auth::StaticTokenProvider::new("google"));
    let gemini: Arc<dyn crate::auth::TokenProvider> =
        Arc::new(crate::auth::StaticTokenProvider::new("gemini"));

    let ctx = BuildContext {
        google_token_provider: Some(google.clone()),
        gemini_token_provider: Some(gemini.clone()),
        ..Default::default()
    };
    let resolved = ctx
        .resolved_google_token_provider()
        .expect("resolved provider");
    assert!(Arc::ptr_eq(&resolved, &google));

    let legacy_ctx = BuildContext {
        google_token_provider: None,
        gemini_token_provider: Some(gemini.clone()),
        ..Default::default()
    };
    let legacy_resolved = legacy_ctx
        .resolved_google_token_provider()
        .expect("legacy resolved provider");
    assert!(Arc::ptr_eq(&legacy_resolved, &gemini));
}

#[tokio::test]
async fn registry_builder_propagates_provider_build_overrides_to_language_model_handle() {
    let _g = reg_test_guard();

    let seen_specific = Arc::new(Mutex::new(None));
    let seen_global = Arc::new(Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_ctx".to_string(),
        Arc::new(ContextCapturingFactory {
            id: "testprov_ctx",
            seen: seen_specific.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "testprov_global".to_string(),
        Arc::new(ContextCapturingFactory {
            id: "testprov_global",
            seen: seen_global.clone(),
        }) as Arc<dyn ProviderFactory>,
    );

    let http_config = crate::types::HttpConfig {
        user_agent: Some("registry-test-agent".to_string()),
        ..Default::default()
    };

    let reg = crate::registry::builder::RegistryBuilder::new(providers)
        .with_http_client(reqwest::Client::new())
        .with_http_config(http_config)
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .with_reasoning(true)
        .with_reasoning_budget(1024)
        .with_provider_build_overrides(
            "testprov_ctx",
            crate::registry::ProviderBuildOverrides::default()
                .with_api_key("ctx-key")
                .with_base_url("https://example.com/custom")
                .with_reasoning(false)
                .with_reasoning_budget(2048)
                .fetch(Arc::new(NoopTransport)),
        )
        .with_retry_options(crate::retry_api::RetryOptions::default())
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let specific_handle = reg
        .language_model("testprov_ctx:model")
        .expect("build language handle");
    let global_handle = reg
        .language_model("testprov_global:model")
        .expect("build global language handle");

    let specific_response = specific_handle.chat(vec![]).await.expect("chat response");
    let global_response = global_handle.chat(vec![]).await.expect("chat response");
    assert_eq!(specific_response.content_text(), Some("ok"));
    assert_eq!(global_response.content_text(), Some("ok"));

    let specific_observed = seen_specific
        .lock()
        .unwrap()
        .clone()
        .expect("captured provider-specific build context");
    let global_observed = seen_global
        .lock()
        .unwrap()
        .clone()
        .expect("captured global build context");
    assert_eq!(
        specific_observed,
        ObservedBuildContext {
            provider_id: Some("testprov_ctx".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            has_http_client: true,
            has_http_transport: true,
            user_agent: Some("registry-test-agent".to_string()),
            has_retry_options: true,
            reasoning_enabled: Some(false),
            reasoning_budget: Some(2048),
        }
    );
    assert_eq!(
        global_observed,
        ObservedBuildContext {
            provider_id: Some("testprov_global".to_string()),
            api_key: Some("global-key".to_string()),
            base_url: Some("https://example.com/global".to_string()),
            has_http_client: true,
            has_http_transport: false,
            user_agent: Some("registry-test-agent".to_string()),
            has_retry_options: true,
            reasoning_enabled: Some(true),
            reasoning_budget: Some(1024),
        }
    );
}
