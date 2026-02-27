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
use crate::types::{ChatMessage, ChatRequest, HttpConfig, RerankRequest};
use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::{Arc, Mutex, MutexGuard};

#[allow(dead_code)]
#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    #[allow(dead_code)]
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

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[allow(dead_code)]
fn make_chat_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi").build()])
}

#[allow(dead_code)]
fn make_chat_request_with_model(model: &str) -> ChatRequest {
    let mut req = make_chat_request();
    req.common_params.model = model.to_string();
    req
}

#[allow(dead_code)]
fn make_rerank_request(model: &str) -> RerankRequest {
    RerankRequest::new(
        model.to_string(),
        "query".to_string(),
        vec!["doc-1".to_string(), "doc-2".to_string()],
    )
}

#[cfg(feature = "azure")]
mod azure_contract {
    use super::*;

    #[tokio::test]
    async fn azure_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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

#[cfg(feature = "cohere")]
mod cohere_contract {
    use super::*;
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::CohereProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn cohere_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via env api key");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(req.url, "https://api.cohere.com/v2/rerank");
    }

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via ctx api key");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            base_url: Some("https://example.com/cohere".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via base_url override");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/cohere/"));
        assert!(req.url.ends_with("/rerank"));
    }
}

#[cfg(feature = "togetherai")]
mod togetherai_contract {
    use super::*;
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::TogetherAiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn togetherai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via env api key");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(req.url, "https://api.together.xyz/v1/rerank");
    }

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via ctx api key");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            base_url: Some("https://example.com/together".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via base_url override");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/together/"));
        assert!(req.url.ends_with("/rerank"));
    }
}

#[cfg(feature = "openai")]
mod openai_contract {
    use super::*;
    use crate::registry::factories::{OpenAICompatibleProviderFactory, OpenAIProviderFactory};
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn openai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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

        let _ = client
            .chat_request(make_chat_request_with_model("openai/gpt-4o"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

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

        let _ = client
            .chat_request(make_chat_request_with_model("openai/gpt-4o"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }
}

#[cfg(feature = "anthropic")]
mod anthropic_contract {
    use super::*;

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn anthropic_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("ANTHROPIC_API_KEY", "env-key");
        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("x-api-key").unwrap(), "env-key");
        assert!(req.url.starts_with("https://example.com"));
    }

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("ANTHROPIC_API_KEY", "env-key");
        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("x-api-key").unwrap(), "ctx-key");
        assert!(req.url.starts_with("https://example.com"));
    }

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/custom"));
    }
}

#[cfg(feature = "groq")]
mod groq_contract {
    use super::*;
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn groq_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn groq_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GROQ_API_KEY", "env-key");
        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/openai/v1"));
    }

    #[tokio::test]
    async fn groq_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GROQ_API_KEY", "env-key");
        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert!(req.url.starts_with("https://example.com/openai/v1"));
    }

    #[tokio::test]
    async fn groq_factory_appends_openai_path_for_root_base_url() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(
            req.url.starts_with("https://example.com/openai/v1"),
            "unexpected url: {}",
            req.url
        );
    }
}

#[cfg(feature = "xai")]
mod xai_contract {
    use super::*;
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn xai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn xai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("XAI_API_KEY", "env-key");
        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn xai_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("XAI_API_KEY", "env-key");
        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn xai_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/custom"));
    }
}

#[cfg(feature = "ollama")]
mod ollama_contract {
    use super::*;

    #[tokio::test]
    async fn ollama_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn ollama_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_ollama::providers::ollama::OllamaClient>()
            .expect("OllamaClient");
        assert_eq!(typed.base_url(), "http://example.com:11434");
    }

    #[tokio::test]
    async fn ollama_factory_does_not_require_api_key() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("ollama should build without api key");
    }
}

#[cfg(feature = "minimaxi")]
mod minimaxi_contract {
    use super::*;

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn minimaxi_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("MINIMAXI_API_KEY", "env-key");
        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client via env api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().api_key, "env-key");
    }

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("MINIMAXI_API_KEY", "env-key");
        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client via ctx api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().api_key, "ctx-key");
    }

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().base_url, "https://example.com/custom");
    }
}

#[cfg(feature = "google-vertex")]
mod anthropic_vertex_contract {
    use super::*;

    #[tokio::test]
    async fn anthropic_vertex_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("anthropic-vertex".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn anthropic_vertex_factory_requires_base_url() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic-vertex".to_string()),
            ..Default::default()
        };

        let result = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await;
        match result {
            Err(LlmError::ConfigurationError(msg)) => {
                assert!(msg.to_lowercase().contains("base_url"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected base_url to be required"),
        }
    }
}

#[cfg(feature = "google")]
mod gemini_contract {
    use super::*;
    use crate::registry::factories::GeminiProviderFactory;

    #[tokio::test]
    async fn gemini_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
        let _lock = lock_env();

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
