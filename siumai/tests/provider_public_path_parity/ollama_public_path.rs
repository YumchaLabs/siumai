use super::*;
use futures_util::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::{EmbeddingExtensions, EmbeddingRequest, Tool, ToolChoice};
use siumai::provider_ext::ollama::{
    OllamaChatRequestExt, OllamaChatResponseExt, OllamaEmbeddingOptions, OllamaEmbeddingRequestExt,
    OllamaOptions,
};
use siumai::registry::builder::RegistryBuilder;

fn ollama_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("ollama", "ollama")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(ollama_registry_providers())
        .with_provider_base_url_fetch("ollama", base_url, transport)
        .build()
        .expect("build ollama registry")
}

fn make_ollama_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    ollama_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(ollama_registry_providers())
        .with_http_transport(global_transport)
        .with_base_url("http://example.com:11434/global")
        .with_provider_base_url_fetch(
            "ollama",
            "http://example.com:11434/ollama",
            ollama_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build ollama override registry")
}

fn assert_ollama_default_options_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{}api/chat", base_url));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
    assert_eq!(req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(req.body["raw"], serde_json::json!(true));
    assert_eq!(req.body["think"], serde_json::json!(true));
    assert_eq!(req.body["options"]["num_ctx"], serde_json::json!(4096));
    assert_eq!(req.body["format"], serde_json::json!("json"));
}

fn assert_ollama_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_ollama_default_options_request(req, base_url, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn ollama_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url("http://example.com:11434/")
            .model("llama3.2")
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });

    let request = make_chat_request_with_model("llama3.2")
        .with_provider_option(
            "ollama",
            serde_json::json!({
                "keep_alive": "1m",
                "extra_params": {
                    "think": true,
                    "num_ctx": 4096
                }
            }),
        )
        .with_response_format(siumai::prelude::unified::ResponseFormat::json_schema(
            schema.clone(),
        ));

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(siumai_req.body["think"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(siumai_req.body["format"], schema);
}

#[tokio::test]
async fn ollama_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_ollama_options(
        OllamaOptions::new()
            .with_keep_alive("1m")
            .with_format("json")
            .with_param("think", serde_json::json!(true))
            .with_param("num_ctx", serde_json::json!(4096)),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(siumai_req.body["think"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(siumai_req.body["format"], serde_json::json!("json"));
}

#[tokio::test]
async fn ollama_default_options_match_public_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_raw_mode(false),
        )
        .with_ollama_options(
            OllamaOptions::new()
                .with_format("json")
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        )
        .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_raw_mode(false),
        )
        .with_ollama_options(
            OllamaOptions::new()
                .with_format("json")
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        )
        .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .with_ollama_options(
                OllamaOptions::new()
                    .with_keep_alive("1m")
                    .with_raw_mode(false),
            )
            .with_ollama_options(
                OllamaOptions::new()
                    .with_format("json")
                    .with_param("think", serde_json::json!(true))
                    .with_param("num_ctx", serde_json::json!(4096)),
            )
            .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_ollama_default_options_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn ollama_registry_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_ollama_options(
        OllamaOptions::new()
            .with_keep_alive("1m")
            .with_format("json")
            .with_param("think", serde_json::json!(true))
            .with_param("num_ctx", serde_json::json!(4096)),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(registry_req.body["think"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(registry_req.body["format"], serde_json::json!("json"));
}

#[tokio::test]
async fn ollama_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "nomic-embed-text";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = EmbeddingRequest::single("hello ollama embedding")
        .with_model(model)
        .with_ollama_config(
            OllamaEmbeddingOptions::new()
                .with_keep_alive("5m")
                .with_truncate(false),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "http://example.com:11434/api/embed");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("nomic-embed-text")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello ollama embedding")
    );
    assert_eq!(siumai_req.body["truncate"], serde_json::json!(false));
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("5m"));
}

#[tokio::test]
async fn ollama_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "nomic-embed-text";
    let base_url = "http://example.com:11434/";

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .embedding_model("ollama:nomic-embed-text")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello ollama embedding")
        .with_model(model)
        .with_ollama_config(
            OllamaEmbeddingOptions::new()
                .with_keep_alive("5m")
                .with_truncate(false),
        );

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "http://example.com:11434/api/embed");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("nomic-embed-text")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!("hello ollama embedding")
    );
    assert_eq!(registry_req.body["truncate"], serde_json::json!(false));
    assert_eq!(registry_req.body["keep_alive"], serde_json::json!("5m"));
}

#[tokio::test]
async fn ollama_tool_choice_none_omits_tools_across_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "http://example.com:11434/api/chat");
    let siumai_body = siumai_req
        .body
        .as_object()
        .expect("ollama chat body object");
    assert!(!siumai_body.contains_key("tools"));
    assert!(!siumai_body.contains_key("tool_choice"));
}

#[tokio::test]
async fn ollama_required_tool_choice_fails_fast_on_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::Required)
        .build();

    let siumai_err = siumai_client
        .chat_request(request.clone())
        .await
        .expect_err("siumai request should fail");
    let provider_err = provider_client
        .chat_request(request.clone())
        .await
        .expect_err("provider request should fail");
    let config_err = config_client
        .chat_request(request.clone())
        .await
        .expect_err("config request should fail");
    let registry_err = registry_model
        .chat_request(request)
        .await
        .expect_err("registry request should fail");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);

    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
    assert!(registry_transport.take().is_none());
}

#[tokio::test]
async fn ollama_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let ollama_transport = CaptureTransport::default();

    let registry = make_ollama_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(ollama_transport.clone()),
    );

    let handle = registry
        .language_model("ollama:llama3.2")
        .expect("build ollama handle");

    let _ = handle
        .chat_request(make_chat_request_with_model("llama3.2"))
        .await;

    let req = ollama_transport.take().expect("captured ollama request");
    assert!(global_transport.take().is_none());
    assert_eq!(req.url, "http://example.com:11434/ollama/api/chat");
    assert_eq!(req.body["model"], serde_json::json!("llama3.2"));
}

#[tokio::test]
async fn ollama_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let ollama_transport = CaptureTransport::default();

    let registry = make_ollama_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(ollama_transport.clone()),
    );

    let handle = registry
        .language_model("ollama:llama3.2")
        .expect("build ollama handle");

    let _ = handle
        .chat_stream_request(make_chat_request_with_model("llama3.2"))
        .await;

    let req = ollama_transport
        .take_stream()
        .expect("captured ollama stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(req.url, "http://example.com:11434/ollama/api/chat");
    assert_eq!(req.body["model"], serde_json::json!("llama3.2"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn ollama_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let ollama_transport = CaptureTransport::default();

    let registry = make_ollama_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(ollama_transport.clone()),
    );

    let handle = registry
        .embedding_model("ollama:nomic-embed-text")
        .expect("build ollama embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello ollama embedding")
                .with_model("nomic-embed-text")
                .with_ollama_config(
                    OllamaEmbeddingOptions::new()
                        .with_keep_alive("5m")
                        .with_truncate(false),
                ),
        )
        .await;

    let req = ollama_transport
        .take()
        .expect("captured ollama embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(req.url, "http://example.com:11434/ollama/api/embed");
    assert_eq!(req.body["model"], serde_json::json!("nomic-embed-text"));
    assert_eq!(
        req.body["input"],
        serde_json::json!("hello ollama embedding")
    );
    assert_eq!(req.body["truncate"], serde_json::json!(false));
    assert_eq!(req.body["keep_alive"], serde_json::json!("5m"));
}

#[tokio::test]
async fn ollama_specific_tool_choice_fails_fast_on_public_paths() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::tool("get_weather"))
        .build();

    let siumai_err = siumai_client
        .chat_request(request.clone())
        .await
        .expect_err("siumai request should fail");
    let provider_err = provider_client
        .chat_request(request.clone())
        .await
        .expect_err("provider request should fail");
    let config_err = config_client
        .chat_request(request.clone())
        .await
        .expect_err("config request should fail");
    let registry_err = registry_model
        .chat_request(request)
        .await
        .expect_err("registry request should fail");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);

    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
    assert!(registry_transport.take().is_none());
}

#[tokio::test]
async fn ollama_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url("http://example.com:11434/")
            .model("llama3.2")
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });

    let request = make_chat_request_with_model("llama3.2")
        .with_provider_option(
            "ollama",
            serde_json::json!({
                "keep_alive": "1m",
                "extra_params": {
                    "think": true,
                    "num_ctx": 4096
                }
            }),
        )
        .with_response_format(siumai::prelude::unified::ResponseFormat::json_schema(
            schema.clone(),
        ));

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(siumai_req.body["think"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(siumai_req.body["format"], schema);
}

#[tokio::test]
async fn ollama_siumai_provider_config_stable_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_ollama_options(
        OllamaOptions::new()
            .with_keep_alive("1m")
            .with_format("json")
            .with_param("think", serde_json::json!(true))
            .with_param("num_ctx", serde_json::json!(4096)),
    );

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(siumai_req.body["think"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(siumai_req.body["format"], serde_json::json!("json"));
}

#[tokio::test]
async fn ollama_default_options_match_public_stream_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_raw_mode(false),
        )
        .with_ollama_options(
            OllamaOptions::new()
                .with_format("json")
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        )
        .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_raw_mode(false),
        )
        .with_ollama_options(
            OllamaOptions::new()
                .with_format("json")
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        )
        .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .with_ollama_options(
                OllamaOptions::new()
                    .with_keep_alive("1m")
                    .with_raw_mode(false),
            )
            .with_ollama_options(
                OllamaOptions::new()
                    .with_format("json")
                    .with_param("think", serde_json::json!(true))
                    .with_param("num_ctx", serde_json::json!(4096)),
            )
            .with_ollama_options(OllamaOptions::new().with_raw_mode(true))
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_ollama_default_options_stream_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn ollama_registry_stable_stream_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_ollama_options(
        OllamaOptions::new()
            .with_keep_alive("1m")
            .with_format("json")
            .with_param("think", serde_json::json!(true))
            .with_param("num_ctx", serde_json::json!(4096)),
    );

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(registry_req.body["keep_alive"], serde_json::json!("1m"));
    assert_eq!(registry_req.body["think"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["options"]["num_ctx"],
        serde_json::json!(4096)
    );
    assert_eq!(registry_req.body["format"], serde_json::json!("json"));
}

#[tokio::test]
async fn ollama_siumai_provider_config_reasoning_defaults_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-r1:1.5b";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .reasoning(true)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["think"], serde_json::json!(true));
}

#[tokio::test]
async fn ollama_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = serde_json::json!({
        "model": "llama3.2",
        "created_at": "2026-03-11T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Hello from Ollama"
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 1_250_000_000u64,
        "load_duration": 150_000_000u64,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200_000_000u64,
        "eval_count": 20,
        "eval_duration": 700_000_000u64
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request)
        .await
        .expect("config response ok");

    let siumai_meta = siumai_resp
        .ollama_metadata()
        .expect("siumai ollama metadata");
    let provider_meta = provider_resp
        .ollama_metadata()
        .expect("provider ollama metadata");
    let config_meta = config_resp
        .ollama_metadata()
        .expect("config ollama metadata");

    assert_eq!(
        siumai_resp.content_text().unwrap_or_default(),
        "Hello from Ollama"
    );
    assert_eq!(
        provider_resp.content_text().unwrap_or_default(),
        "Hello from Ollama"
    );
    assert_eq!(
        config_resp.content_text().unwrap_or_default(),
        "Hello from Ollama"
    );

    assert_eq!(siumai_meta.total_duration_ms, Some(1250));
    assert_eq!(provider_meta.total_duration_ms, Some(1250));
    assert_eq!(config_meta.total_duration_ms, Some(1250));
    assert_eq!(siumai_meta.load_duration_ms, Some(150));
    assert_eq!(provider_meta.load_duration_ms, Some(150));
    assert_eq!(config_meta.load_duration_ms, Some(150));
    assert_eq!(siumai_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(provider_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(config_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(siumai_meta.eval_duration_ms, Some(700));
    assert_eq!(provider_meta.eval_duration_ms, Some(700));
    assert_eq!(config_meta.eval_duration_ms, Some(700));
    assert!(
        (siumai_meta
            .tokens_per_second
            .expect("siumai tokens_per_second")
            - 28.5714285714)
            .abs()
            < 1e-6
    );
    assert!(
        (provider_meta
            .tokens_per_second
            .expect("provider tokens_per_second")
            - 28.5714285714)
            .abs()
            < 1e-6
    );
    assert!(
        (config_meta
            .tokens_per_second
            .expect("config tokens_per_second")
            - 28.5714285714)
            .abs()
            < 1e-6
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "http://example.com:11434/api/chat");
}

#[tokio::test]
async fn ollama_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = concat!(
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"done\":true,\"done_reason\":\"stop\",\"total_duration\":1250000000,\"load_duration\":150000000,\"prompt_eval_count\":10,\"prompt_eval_duration\":200000000,\"eval_count\":20,\"eval_duration\":700000000}\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut config_end = None;

    while let Some(event) = siumai_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            siumai_end = Some(response);
            break;
        }
    }
    while let Some(event) = provider_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            provider_end = Some(response);
            break;
        }
    }
    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");

    let siumai_meta = siumai_resp
        .ollama_metadata()
        .expect("siumai ollama metadata");
    let provider_meta = provider_resp
        .ollama_metadata()
        .expect("provider ollama metadata");
    let config_meta = config_resp
        .ollama_metadata()
        .expect("config ollama metadata");

    assert_eq!(
        siumai_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        provider_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    assert_eq!(siumai_meta.total_duration_ms, Some(1250));
    assert_eq!(provider_meta.total_duration_ms, Some(1250));
    assert_eq!(config_meta.total_duration_ms, Some(1250));
    assert_eq!(siumai_meta.load_duration_ms, Some(150));
    assert_eq!(provider_meta.load_duration_ms, Some(150));
    assert_eq!(config_meta.load_duration_ms, Some(150));
    assert_eq!(siumai_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(provider_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(config_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(siumai_meta.eval_duration_ms, Some(700));
    assert_eq!(provider_meta.eval_duration_ms, Some(700));
    assert_eq!(config_meta.eval_duration_ms, Some(700));

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "http://example.com:11434/api/chat");
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn ollama_registry_chat_response_metadata_match_config_path() {
    let response_json = serde_json::json!({
        "model": "llama3.2",
        "created_at": "2026-03-11T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Hello from Ollama"
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 1_250_000_000u64,
        "load_duration": 150_000_000u64,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200_000_000u64,
        "eval_count": 20,
        "eval_duration": 700_000_000u64
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let config_meta = config_resp
        .ollama_metadata()
        .expect("config ollama metadata");
    let registry_meta = registry_resp
        .ollama_metadata()
        .expect("registry ollama metadata");

    assert_eq!(
        config_resp.content_text().unwrap_or_default(),
        "Hello from Ollama"
    );
    assert_eq!(
        registry_resp.content_text().unwrap_or_default(),
        "Hello from Ollama"
    );
    assert_eq!(config_meta.total_duration_ms, Some(1250));
    assert_eq!(registry_meta.total_duration_ms, Some(1250));
    assert_eq!(config_meta.load_duration_ms, Some(150));
    assert_eq!(registry_meta.load_duration_ms, Some(150));
    assert_eq!(config_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(registry_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(config_meta.eval_duration_ms, Some(700));
    assert_eq!(registry_meta.eval_duration_ms, Some(700));
    assert!(
        (config_meta
            .tokens_per_second
            .expect("config tokens_per_second")
            - 28.5714285714)
            .abs()
            < 1e-6
    );
    assert!(
        (registry_meta
            .tokens_per_second
            .expect("registry tokens_per_second")
            - 28.5714285714)
            .abs()
            < 1e-6
    );

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "http://example.com:11434/api/chat");
}

#[tokio::test]
async fn ollama_registry_stream_end_metadata_match_config_path() {
    let stream_body = concat!(
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"done\":true,\"done_reason\":\"stop\",\"total_duration\":1250000000,\"load_duration\":150000000,\"prompt_eval_count\":10,\"prompt_eval_duration\":200000000,\"eval_count\":20,\"eval_duration\":700000000}\n"
        )
        .as_bytes()
        .to_vec();

    let config_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let registry_transport = JsonStreamSuccessTransport::new(stream_body);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let mut config_end = None;
    let mut registry_end = None;

    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    let config_meta = config_resp
        .ollama_metadata()
        .expect("config ollama metadata");
    let registry_meta = registry_resp
        .ollama_metadata()
        .expect("registry ollama metadata");

    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(config_meta.total_duration_ms, Some(1250));
    assert_eq!(registry_meta.total_duration_ms, Some(1250));
    assert_eq!(config_meta.load_duration_ms, Some(150));
    assert_eq!(registry_meta.load_duration_ms, Some(150));
    assert_eq!(config_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(registry_meta.prompt_eval_duration_ms, Some(200));
    assert_eq!(config_meta.eval_duration_ms, Some(700));
    assert_eq!(registry_meta.eval_duration_ms, Some(700));

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "http://example.com:11434/api/chat");
    assert_eq!(config_req.body["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn ollama_structured_output_stream_extraction_matches_across_public_paths() {
    let mut stream_body = [
        serde_json::json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "{\"summary\":\"hel"
            },
            "done": false
        })
        .to_string(),
        serde_json::json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "lo\"}"
            },
            "done": false
        })
        .to_string(),
        serde_json::json!({
            "model": "llama3.2",
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20
        })
        .to_string(),
    ]
    .join("\n")
    .into_bytes();
    stream_body.push(b'\n');

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let registry_transport = JsonStreamSuccessTransport::new(stream_body);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });

    let request = make_chat_request_with_model(model).with_response_format(
        siumai::prelude::unified::ResponseFormat::json_schema(schema.clone()),
    );

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect("config structured output");
    let registry_value = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect("registry structured output");

    assert_eq!(siumai_value["summary"], "hello");
    assert_eq!(provider_value["summary"], "hello");
    assert_eq!(config_value["summary"], "hello");
    assert_eq!(registry_value["summary"], "hello");

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["format"], schema);
}

#[tokio::test]
async fn ollama_structured_output_interrupted_stream_fails_consistently_across_public_paths() {
    let mut stream_body = [serde_json::json!({
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "{\"summary\":"
        },
        "done": false
    })
    .to_string()]
    .join("\n")
    .into_bytes();
    stream_body.push(b'\n');

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let registry_transport = JsonStreamSuccessTransport::new(stream_body);

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("ollama:llama3.2")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });

    let request = make_chat_request_with_model(model).with_response_format(
        siumai::prelude::unified::ResponseFormat::json_schema(schema.clone()),
    );

    let siumai_err = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect_err("siumai interrupted stream should fail");
    let provider_err = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect_err("provider interrupted stream should fail");
    let config_err = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect_err("config interrupted stream should fail");
    let registry_err = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect_err("registry interrupted stream should fail");

    for err in [siumai_err, provider_err, config_err, registry_err] {
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["format"], schema);
}

#[tokio::test]
async fn ollama_siumai_provider_config_non_embedding_requests_are_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let image_request = make_image_request_with_model(model);
    let rerank_request = make_rerank_request_with_model(model).with_top_n(1);

    let siumai_image_err = siumai_client
        .generate_images(image_request.clone())
        .await
        .expect_err("ollama image generation should be unsupported");
    let siumai_rerank_err = siumai_client
        .rerank(rerank_request.clone())
        .await
        .expect_err("ollama rerank should be unsupported");

    assert_unsupported_operation(&siumai_image_err);
    assert_unsupported_operation(&siumai_rerank_err);

    assert!(siumai_client.as_embedding_capability().is_some());
    assert!(provider_client.as_embedding_capability().is_some());
    assert!(config_client.as_embedding_capability().is_some());
    assert!(siumai_client.as_image_generation_capability().is_none());
    assert!(provider_client.as_image_generation_capability().is_none());
    assert!(config_client.as_image_generation_capability().is_none());
    assert!(siumai_client.as_rerank_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn ollama_registry_non_embedding_requests_are_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let base_url = "http://example.com:11434/";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let image_err = match registry.image_model("ollama:llama3.2") {
        Ok(_) => panic!("build registry image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("ollama:llama3.2") {
        Ok(_) => panic!("build registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn ollama_siumai_provider_config_audio_family_requests_are_intentionally_unsupported() {
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let config_transport = MixedCaptureTransport::default();

    let model = "llama3.2";
    let base_url = "http://example.com:11434/";

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(model)
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let tts_request = TtsRequest::new("hello ollama audio".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());
    let stt_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let siumai_tts_err = siumai_client
        .text_to_speech(tts_request.clone())
        .await
        .expect_err("ollama text-to-speech should be unsupported");

    let siumai_stt_err = siumai_client
        .speech_to_text(stt_request.clone())
        .await
        .expect_err("ollama speech-to-text should be unsupported");

    for err in [siumai_tts_err, siumai_stt_err] {
        assert_unsupported_operation(&err);
    }

    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_mixed_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
    ]);
}

#[tokio::test]
async fn ollama_registry_audio_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let base_url = "http://example.com:11434/";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let tts_err = match registry.speech_model("ollama:llama3.2") {
        Ok(_) => panic!("build ollama registry speech model should be unsupported"),
        Err(err) => err,
    };
    let stt_err = match registry.transcription_model("ollama:llama3.2") {
        Ok(_) => panic!("build ollama registry transcription model should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&tts_err);
    assert_unsupported_operation(&stt_err);
    assert_mixed_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn ollama_siumai_provider_config_registry_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .ollama()
        .base_url("http://example.com:11434/")
        .model("nomic-embed-text")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::ollama()
        .base_url("http://example.com:11434/")
        .model("nomic-embed-text")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::ollama::OllamaClient::from_config(
        siumai::provider_ext::ollama::OllamaConfig::builder()
            .base_url("http://example.com:11434/")
            .model("nomic-embed-text")
            .http_transport(Arc::new(config_transport.clone()))
            .build()
            .expect("build ollama config"),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "http://example.com:11434/",
    );
    let registry_client = registry
        .embedding_model("ollama:nomic-embed-text")
        .expect("build registry embedding handle");

    let request = EmbeddingRequest::new(vec!["text1".to_string(), "text2".to_string()])
        .with_model("nomic-embed-text")
        .with_ollama_config(
            OllamaEmbeddingOptions::new()
                .with_truncate(false)
                .with_keep_alive("5m")
                .with_option("temperature", serde_json::json!(0.1)),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "http://example.com:11434/api/embed");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("nomic-embed-text")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["text1", "text2"])
    );
    assert_eq!(siumai_req.body["truncate"], serde_json::json!(false));
    assert_eq!(siumai_req.body["keep_alive"], serde_json::json!("5m"));
    assert_eq!(
        siumai_req.body["options"]["temperature"],
        serde_json::json!(0.1)
    );
}
