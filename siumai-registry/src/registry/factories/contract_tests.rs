//! Factory contract tests (no network).
//!
//! These tests validate shared construction precedence rules across provider factories:
//! - `ctx.http_client` overrides `ctx.http_config` (so invalid config is ignored when client is provided)
//! - `ctx.api_key` overrides env vars (when env fallback exists)
//! - `ctx.base_url` overrides provider defaults (where applicable)

#![allow(unsafe_code)]

use crate::error::LlmError;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use crate::registry::entry::{BuildContext, ProviderFactory};
use crate::types::{ChatMessage, ChatRequest, HttpConfig};
use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::{Arc, Mutex};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => unsafe {
                std::env::set_var(self.key, v);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().unwrap().take()
    }
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().unwrap() = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 401,
            headers,
            body:
                br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
        })
    }
}

fn make_chat_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi").build()])
}

#[cfg(feature = "openai")]
mod openai_contract {
    use super::*;
    use crate::registry::factories::{OpenAICompatibleProviderFactory, OpenAIProviderFactory};
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn openai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            http_transport: Some(Arc::new(transport)),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn openai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("OPENAI_API_KEY", "env-key");

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.ends_with("/responses"));
    }

    #[tokio::test]
    async fn openai_factory_prefers_ctx_api_key_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("OPENAI_API_KEY", "env-key");

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn openai_factory_prefers_ctx_base_url_over_default() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_openai::providers::openai::OpenAiClient>()
            .expect("OpenAiClient");
        assert_eq!(typed.base_url(), "https://example.com/custom/v1");
    }

    #[tokio::test]
    async fn openai_compatible_factory_prefers_ctx_api_key_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("DEEPSEEK_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("deepseek".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build openai-compatible client");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }
}

#[cfg(feature = "google")]
mod gemini_contract {
    use super::*;
    use crate::registry::factories::GeminiProviderFactory;

    #[tokio::test]
    async fn gemini_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = GeminiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn gemini_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client via env api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "env-key");
    }

    #[tokio::test]
    async fn gemini_factory_prefers_ctx_api_key_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client via ctx api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "ctx-key");
    }

    #[tokio::test]
    async fn gemini_factory_accepts_root_base_url() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(
            typed.base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[tokio::test]
    async fn gemini_factory_does_not_require_api_key_with_authorization_header() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::remove("GEMINI_API_KEY");
        let factory = GeminiProviderFactory;

        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer token".to_string());

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            http_config: Some(http_config),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client without api key when auth header is present");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "");
    }
}
