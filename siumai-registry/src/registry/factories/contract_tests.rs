//! Factory contract tests (no network).
//!
//! These tests validate shared construction precedence rules across provider factories:
//! - `ctx.http_client` overrides `ctx.http_config` (so invalid config is ignored when client is provided)
//! - `ctx.api_key` overrides env vars (when env fallback exists)
//! - `ctx.base_url` overrides provider defaults (where applicable)

use crate::error::LlmError;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use crate::registry::entry::{BuildContext, ProviderFactory};
use crate::test_support::{ENV_LOCK, EnvGuard};
use crate::types::{ChatMessage, ChatRequest, HttpConfig};
use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::{Arc, Mutex};

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

#[allow(dead_code)]
fn make_chat_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi").build()])
}

#[cfg(feature = "azure")]
mod azure_contract {
    use super::*;

    #[tokio::test]
    async fn azure_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/openai".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn azure_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");
        let _r = EnvGuard::set("AZURE_RESOURCE_NAME", "my-azure-resource");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("api-key").unwrap(), "env-key");
        assert!(
            req.url
                .starts_with("https://my-azure-resource.openai.azure.com/openai/"),
            "unexpected url: {}",
            req.url
        );
    }

    #[tokio::test]
    async fn azure_factory_prefers_ctx_api_key_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("api-key").unwrap(), "ctx-key");
        assert!(req.url.starts_with("https://example.com/custom/openai/"));
    }

    #[tokio::test]
    async fn azure_factory_prefers_ctx_base_url_over_resource_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");
        let _r = EnvGuard::set("AZURE_RESOURCE_NAME", "my-azure-resource");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            base_url: Some("https://example.com/override/openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via base_url override");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/override/openai/"));
    }
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

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_uses_env_api_key_when_ctx_missing() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("OPENROUTER_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("openai/gpt-4o", &ctx)
            .await
            .expect("build openai-compatible client via env api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_prefers_ctx_api_key_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("OPENROUTER_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("openai/gpt-4o", &ctx)
            .await
            .expect("build openai-compatible client via ctx api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert!(req.url.starts_with("https://example.com/v1"));
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

#[cfg(feature = "google-vertex")]
mod vertex_contract {
    use super::*;
    use crate::registry::factories::GoogleVertexProviderFactory;
    use crate::types::ImageGenerationRequest;

    fn make_image_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "hi".to_string(),
            count: 1,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = GoogleVertexProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn vertex_factory_uses_express_base_url_when_ctx_api_key_present() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(
            typed.base_url(),
            crate::utils::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
        );
    }

    #[tokio::test]
    async fn vertex_factory_uses_env_project_location_when_no_api_key_or_base_url() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _k = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
        let _p = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
        let _l = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            http_client: Some(reqwest::Client::new()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client via env project/location");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(
            typed.base_url(),
            crate::utils::vertex::google_vertex_base_url("test-project", "us-central1")
        );
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_base_url_over_express_default() {
        let _lock = ENV_LOCK.lock().unwrap();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_api_key_over_env_for_express_query() {
        let _lock = ENV_LOCK.lock().unwrap();

        let _g = EnvGuard::set("GOOGLE_VERTEX_API_KEY", "env-key");
        let factory = GoogleVertexProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let cap = client
            .as_image_generation_capability()
            .expect("image generation capability");
        let _ = cap.generate_images(make_image_request()).await;

        let req = transport.take().expect("captured request");
        assert!(
            req.url.contains("key=ctx-key"),
            "unexpected url: {}",
            req.url
        );
    }
}
