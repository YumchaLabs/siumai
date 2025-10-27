//! Request headers flow tests against a mock server
//!
//! Verifies that high-level client/capability code sends expected headers,
//! including custom headers and provider-specific ones.

use mockito::Matcher;
use reqwest::Client;

#[tokio::test]
async fn openai_files_list_includes_custom_and_org_project_headers() {
    let _m = mockito::mock("GET", Matcher::Regex(r"/files\?.*".to_string()))
        .match_header("authorization", Matcher::Regex(r"^Bearer test-key$".to_string()))
        .match_header("content-type", "application/json")
        .match_header("OpenAI-Organization", "org-test")
        .match_header("OpenAI-Project", "proj-test")
        .match_header("X-Custom-Header", "custom-value")
        .with_status(200)
        .with_body(r#"{"data":[],"has_more":false}"#)
        .create();

    let base_url = &mockito::server_url();
    let config = siumai::providers::openai::OpenAiConfig::new("test-key")
        .with_base_url(base_url)
        .with_organization("org-test")
        .with_project("proj-test");

    let mut http_cfg = config.http_config.clone();
    http_cfg
        .headers
        .insert("X-Custom-Header".to_string(), "custom-value".to_string());
    let config = siumai::providers::openai::OpenAiConfig { http_config: http_cfg, ..config };

    let files = siumai::providers::openai::files::OpenAiFiles::new(config, Client::new());
    let resp = files.list_files(None).await.expect("list files should succeed");
    assert_eq!(resp.files.len(), 0);
}

#[tokio::test]
async fn openai_compatible_chat_includes_http_and_custom_and_adapter_headers() {
    // Mock OpenAI-compatible chat completions endpoint
    let _m = mockito::mock("POST", "/v1/chat/completions")
        .match_header("authorization", Matcher::Regex(r"^Bearer test-key$".to_string()))
        .match_header("content-type", "application/json")
        .match_header("X-Compat-H1", "v1") // from http_config.headers
        .match_header("X-Compat-H2", "v2") // from config.custom_headers
        .match_header("X-Adapter", "yes") // from adapter.custom_headers
        .with_status(200)
        .with_body(
            r#"{
            "id":"chatcmpl_test",
            "object":"chat.completion",
            "created": 123,
            "model":"gpt-3.5-turbo",
            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        }"#,
        )
        .create();

    // Define a minimal adapter that injects a custom header
    #[derive(Debug, Clone)]
    struct TestAdapter;
    impl siumai::providers::openai_compatible::adapter::ProviderAdapter for TestAdapter {
        fn provider_id(&self) -> &'static str { "test-adapter" }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _request_type: siumai::providers::openai_compatible::types::RequestType,
        ) -> Result<(), siumai::LlmError> { Ok(()) }
        fn get_field_mappings(&self, _model: &str) -> siumai::providers::openai_compatible::types::FieldMappings {
            siumai::providers::openai_compatible::types::FieldMappings {
                thinking_fields: vec!["thinking", "reasoning", "reasoning_content"],
                content_field: "content",
                tool_calls_field: "tool_calls",
                role_field: "role",
            }
        }
        fn get_field_accessor(&self) -> Box<dyn siumai::providers::openai_compatible::types::FieldAccessor> {
            Box::new(siumai::providers::openai_compatible::types::JsonFieldAccessor)
        }
        fn get_model_config(&self, _model: &str) -> siumai::providers::openai_compatible::types::ModelConfig { Default::default() }
        fn custom_headers(&self) -> reqwest::header::HeaderMap {
            let mut h = reqwest::header::HeaderMap::new();
            h.insert("X-Adapter", reqwest::header::HeaderValue::from_static("yes"));
            h
        }
        fn capabilities(&self) -> siumai::traits::ProviderCapabilities {
            siumai::traits::ProviderCapabilities::new().with_chat()
        }
        fn base_url(&self) -> &str { "" }
        fn clone_adapter(&self) -> Box<dyn siumai::providers::openai_compatible::adapter::ProviderAdapter> { Box::new(self.clone()) }
    }

    let base_url = format!("{}/v1", mockito::server_url());
    let adapter = std::sync::Arc::new(TestAdapter);
    let mut http_cfg = siumai::types::HttpConfig::default();
    http_cfg.headers.insert("X-Compat-H1".to_string(), "v1".to_string());

    let config = siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
        "compat",
        "test-key",
        &base_url,
        adapter,
    )
    .with_model("gpt-3.5-turbo")
    .with_http_config(http_cfg)
    .with_header("X-Compat-H2", "v2")
    .expect("custom header ok");

    let client = siumai::providers::openai_compatible::openai_client::OpenAiCompatibleClient::new(config)
        .await
        .expect("client ok");

    let messages = vec![siumai::types::ChatMessage::user("hi").build()];
    let resp = client.chat_with_tools(messages, None).await.expect("chat ok");
    // Simple sanity assertion
    assert!(matches!(resp.content, siumai::types::MessageContent::Text(_)));
}

#[tokio::test]
async fn openai_rerank_includes_org_project_and_custom_headers() {
    let _m = mockito::mock("POST", "/v1/rerank")
        .match_header("authorization", Matcher::Regex(r"^Bearer test-openai$".to_string()))
        .match_header("content-type", "application/json")
        .match_header("OpenAI-Organization", "org-1")
        .match_header("OpenAI-Project", "proj-1")
        .match_header("X-R-Header", "rv")
        .with_status(200)
        .with_body(
            r#"{"id":"rr_1","results":[{"document":null,"index":0,"relevance_score":0.9}],"tokens":{"input_tokens":1,"output_tokens":1}}"#,
        )
        .create();

    let base_url = &mockito::server_url();
    let mut cfg = siumai::providers::openai::OpenAiConfig::new("test-openai")
        .with_base_url(base_url)
        .with_organization("org-1")
        .with_project("proj-1");
    // Inject a custom header via http_config for rerank
    let mut http_cfg = cfg.http_config.clone();
    http_cfg.headers.insert("X-R-Header".to_string(), "rv".to_string());
    cfg.http_config = http_cfg;

    let client = siumai::providers::openai::client::OpenAiClient::new_with_config(cfg);
    let req = siumai::types::RerankRequest::new(
        "BAAI/bge-reranker-v2-m3".to_string(),
        "query".to_string(),
        vec!["doc1".to_string()],
    );
    let resp = client.rerank(req).await.expect("rerank ok");
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
async fn groq_chat_includes_auth_user_agent_and_custom_headers() {
    let _m = mockito::mock("POST", "/v1/chat/completions")
        .match_header("authorization", Matcher::Regex(r"^Bearer test-groq$".to_string()))
        .match_header("content-type", "application/json")
        .match_header("user-agent", Matcher::Any)
        .match_header("X-Custom-Header", "abc")
        .with_status(200)
        .with_body(r#"{"id":"cmpl_1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"model":"llama-3.1-8b-instant","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#)
        .create();

    let base_url = format!("{}/v1", mockito::server_url());
    let mut http_cfg = siumai::types::HttpConfig::default();
    http_cfg
        .headers
        .insert("X-Custom-Header".to_string(), "abc".to_string());

    let config = siumai::providers::groq::GroqConfig::new("test-groq")
        .with_base_url(base_url)
        .with_model(siumai::providers::groq::models::production::LLAMA_3_1_8B_INSTANT)
        .with_http_config(http_cfg);
    let client = siumai::providers::groq::GroqClient::new(config, Client::new());
    let messages = vec![siumai::types::ChatMessage::user("hi").build()];
    let resp = client.chat_with_tools(messages, None).await.expect("chat ok");
    assert!(matches!(resp.content, siumai::types::MessageContent::Text(_)));
}

#[tokio::test]
async fn xai_chat_includes_auth_and_custom_headers() {
    let _m = mockito::mock("POST", "/v1/chat/completions")
        .match_header("authorization", Matcher::Regex(r"^Bearer xai-key$".to_string()))
        .match_header("content-type", "application/json")
        .match_header("X-Debug", "1")
        .with_status(200)
        .with_body(r#"{"id":"cmpl_x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"model":"grok-beta","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#)
        .create();

    let base_url = format!("{}/v1", mockito::server_url());
    let mut http_cfg = siumai::types::HttpConfig::default();
    http_cfg.headers.insert("X-Debug".to_string(), "1".to_string());

    let mut cfg = siumai::providers::xai::config::XaiConfig::new("xai-key")
        .with_base_url(base_url)
        .with_model("grok-beta");
    cfg.http_config = http_cfg;
    let client = siumai::providers::xai::client::XaiClient::with_http_client(cfg, Client::new())
        .await
        .expect("build xai client");
    let messages = vec![siumai::types::ChatMessage::user("hi").build()];
    let resp = client.chat_with_tools(messages, None).await.expect("chat ok");
    assert!(matches!(resp.content, siumai::types::MessageContent::Text(_)));
}

#[tokio::test]
async fn groq_stt_uses_multipart_content_type() {
    // Expect multipart/form-data with boundary
    let _m = mockito::mock("POST", "/v1/audio/transcriptions")
        .match_header(
            "content-type",
            Matcher::Regex(r"^multipart/form-data; boundary=.*".to_string()),
        )
        .with_status(200)
        .with_body(r#"{"text":"hello"}"#)
        .create();

    let base_url = format!("{}/v1", mockito::server_url());
    let config = siumai::providers::groq::GroqConfig::new("test-groq")
        .with_base_url(base_url)
        .with_model(siumai::providers::groq::models::production::LLAMA_3_1_8B_INSTANT);
    let client = siumai::providers::groq::GroqClient::new(config, Client::new());
    let stt_req = siumai::types::SttRequest::from_audio(vec![1, 2, 3, 4]);
    let resp = client.speech_to_text(stt_req).await.expect("stt ok");
    assert_eq!(resp.text, "hello");
}

#[tokio::test]
async fn openai_file_upload_uses_multipart_content_type() {
    let _m = mockito::mock("POST", "/v1/files")
        .match_header(
            "content-type",
            Matcher::Regex(r"^multipart/form-data; boundary=.*".to_string()),
        )
        .with_status(200)
        .with_body(r#"{"id":"file_123","object":"file","bytes":5,"created_at":1710000000,"filename":"hello.txt","purpose":"assistants","status":"uploaded","status_details":null}"#)
        .create();

    let base_url = &mockito::server_url();
    let config = siumai::providers::openai::OpenAiConfig::new("test-openai").with_base_url(base_url);
    let files = siumai::providers::openai::files::OpenAiFiles::new(config, Client::new());
    let req = siumai::types::FileUploadRequest {
        content: b"hello".to_vec(),
        filename: "hello.txt".to_string(),
        mime_type: Some("text/plain".to_string()),
        purpose: "assistants".to_string(),
        metadata: std::collections::HashMap::new(),
    };
    let fo = files.upload_file(req).await.expect("upload ok");
    assert_eq!(fo.filename, "hello.txt");
}

#[tokio::test]
async fn anthropic_chat_includes_beta_header_when_provided() {
    // Anthropic messages endpoint
    let _m = mockito::mock("POST", "/v1/messages")
        .match_header("x-api-key", "test-key")
        .match_header("content-type", "application/json")
        .match_header("anthropic-version", Matcher::Any)
        .match_header("anthropic-beta", "messages-2023-12-15")
        .with_status(200)
        .with_body(r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-3-opus","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#)
        .create();

    let http_client = Client::new();
    let mut http_config = siumai::types::HttpConfig::default();
    http_config.headers.insert(
        "anthropic-beta".to_string(),
        "messages-2023-12-15".to_string(),
    );

    let client = siumai::providers::anthropic::AnthropicClient::new(
        "test-key".to_string(),
        mockito::server_url(),
        http_client,
        siumai::types::CommonParams { model: "claude-3-opus".to_string(), ..Default::default() },
        siumai::params::AnthropicParams::default(),
        http_config,
    );

    let messages = vec![siumai::types::ChatMessage::user("hi").build()];
    let resp = client.chat_with_tools(messages, None).await.expect("chat ok");
    assert_eq!(resp.model.unwrap(), "claude-3-opus");
}

#[tokio::test]
async fn gemini_files_list_includes_api_key_header() {
    // Gemini files list endpoint
    let _m = mockito::mock("GET", mockito::Matcher::Regex(r"/files\??.*".to_string()))
        .match_header("x-goog-api-key", "test-gemini-key")
        .match_header("content-type", "application/json")
        .with_status(200)
        .with_body(r#"{"files": [], "nextPageToken": null}"#)
        .create();

    let config = siumai::providers::gemini::types::GeminiConfig::new("test-gemini-key".to_string())
        .with_base_url(mockito::server_url())
        .with_model("gemini-1.5-flash".to_string());
    let files = siumai::providers::gemini::files::GeminiFiles::new(config, reqwest::Client::new());
    let resp = files.list_files(None).await.expect("list files should succeed");
    assert_eq!(resp.files.len(), 0);
}
