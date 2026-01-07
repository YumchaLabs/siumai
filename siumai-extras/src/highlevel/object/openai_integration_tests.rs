use super::*;

struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);

impl siumai::experimental::execution::http::interceptor::HttpInterceptor for Capture {
    fn on_before_send(
        &self,
        _ctx: &siumai::experimental::execution::http::interceptor::HttpRequestContext,
        _rb: reqwest::RequestBuilder,
        body: &serde_json::Value,
        _headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        *self.0.lock().unwrap() = Some(body.clone());
        Err(LlmError::InvalidParameter("stop".into()))
    }
}

#[test]
fn openai_generate_object_injects_response_format_object() {
    // Prepare OpenAI client
    let cfg =
        siumai::provider_ext::openai::OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
    let client = siumai::provider_ext::openai::OpenAiClient::new(cfg, reqwest::Client::new());
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let cap = Capture(captured.clone());
    let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

    // A simple object schema without name triggers json_object format
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    });
    let messages = vec![ChatMessage::user("hi").build()];
    let opts = GenerateObjectOptions {
        schema: Some(schema),
        ..Default::default()
    };
    // Invoke; interceptor will abort before network
    futures::executor::block_on(async {
        let _ = generate_object_openai::<serde_json::Value>(&client, messages, None, opts).await;
    });
    let body = captured.lock().unwrap().clone().expect("captured body");
    let rf = body
        .get("text")
        .and_then(|t| t.get("format"))
        .cloned()
        .expect("format");
    assert_eq!(rf.get("type").and_then(|v| v.as_str()), Some("json_object"));
}

#[test]
fn openai_stream_object_injects_response_format_named_schema() {
    let cfg =
        siumai::provider_ext::openai::OpenAiConfig::new("test-key").with_model("gpt-4.1-mini");
    let client = siumai::provider_ext::openai::OpenAiClient::new(cfg, reqwest::Client::new());
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let cap = Capture(captured.clone());
    let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);

    let schema = serde_json::json!({
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"]
    });
    let messages = vec![ChatMessage::user("hi").build()];
    let opts = StreamObjectOptions {
        schema: Some(schema),
        schema_name: Some("User".into()),
        ..Default::default()
    };
    // Invoke streaming helper; interceptor will abort before HTTP
    futures::executor::block_on(async {
        let _ = stream_object_openai::<serde_json::Value>(&client, messages, None, opts).await;
    });
    let body = captured.lock().unwrap().clone().expect("captured body");
    let rf = body
        .get("text")
        .and_then(|t| t.get("format"))
        .cloned()
        .expect("format");
    assert_eq!(rf.get("type").and_then(|v| v.as_str()), Some("json_schema"));
    assert_eq!(rf.get("name").and_then(|v| v.as_str()), Some("User"));
}
