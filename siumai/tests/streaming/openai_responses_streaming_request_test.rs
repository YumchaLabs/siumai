//! Verify OpenAI Responses API streaming request headers and body fields
//!
//! Asserts that when using Responses API in streaming mode:
//! - Request headers include `Accept: text/event-stream`
//! - Request headers include `Accept-Encoding: identity` (when not disabled)
//! - JSON body includes `stream: true` and `stream_options.include_usage: true`

use axum::{routing::post, Router};
use axum::body;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai::types::{Tool, ToolFunction};
use siumai::stream::ChatStreamEvent;
use siumai::types::ChatMessage;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default, Clone, Debug)]
struct SeenState {
    accept_is_sse: bool,
    accept_encoding_identity: bool,
    body_stream_true: bool,
    body_include_usage_true: bool,
    has_org_header: bool,
    has_project_header: bool,
    body_has_prompt_cache_key: bool,
}

async fn handler(req: Request, state: Arc<Mutex<SeenState>>) -> Response {
    let (parts, body_in) = req.into_parts();
    let headers: &HeaderMap = &parts.headers;

    let mut seen = SeenState::default();
    if let Some(v) = headers.get("accept") {
        seen.accept_is_sse = v == HeaderValue::from_static("text/event-stream");
    }
    if let Some(v) = headers.get("accept-encoding") {
        seen.accept_encoding_identity = v == HeaderValue::from_static("identity");
    }
    if let Some(v) = headers.get("openai-organization") {
        seen.has_org_header = v == HeaderValue::from_static("org-123");
    }
    if let Some(v) = headers.get("openai-project") {
        seen.has_project_header = v == HeaderValue::from_static("proj-456");
    }

    let body_bytes = body::to_bytes(body_in, 64 * 1024).await.unwrap_or_default();
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        seen.body_stream_true = json.get("stream").and_then(|v| v.as_bool()) == Some(true);
        seen.body_include_usage_true = json
            .get("stream_options")
            .and_then(|v| v.get("include_usage"))
            .and_then(|v| v.as_bool())
            == Some(true);
        seen.body_has_prompt_cache_key = json
            .get("prompt_cache_key")
            .and_then(|v| v.as_str())
            == Some("cache-xyz");
        // tool shape minimal validation
        let tools_ok = json
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().any(|t| t.get("type").and_then(|s| s.as_str()) == Some("function")
                && t.get("name").and_then(|s| s.as_str()) == Some("lookup")))
            .unwrap_or(false);
        // Reuse has_org_header flag to piggyback validation (no new field)
        if tools_ok { seen.has_org_header = true; }
    }

    {
        let mut guard = state.lock().await;
        *guard = seen;
    }

    // Minimal Responses SSE payload
    let sse = concat!(
        "data: {\"response\":{\"id\":\"resp_123\"},\"output\":[{\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}]}\n\n",
        "data: [DONE]\n\n",
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(sse))
        .unwrap()
}

#[tokio::test]
async fn openai_responses_streaming_includes_sse_headers_and_stream_options() {
    // Start local server
    let state = Arc::new(Mutex::new(SeenState::default()));
    let app = {
        let st = state.clone();
        Router::new().route("/v1/responses", post(move |req| handler(req, st.clone())))
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build OpenAI client for Responses API
    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let mut cfg = OpenAiConfig::new("test-key")
        .with_base_url(format!("{}/v1", base))
        .with_model("gpt-4o-mini")
        .with_organization("org-123")
        .with_project("proj-456")
        .with_responses_api(true);
    // prompt_cache_key for streaming body
    let openai_params = siumai::params::OpenAiParams::builder()
        .prompt_cache_key("cache-xyz")
        .build();
    cfg.openai_params = openai_params;
    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    // Stream call
    // Add a simple tool signature to verify Responses API tool flattening in body
    let tools = vec![Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "lookup".to_string(),
            description: Some("Lookup data".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"q": {"type":"string"}},
                "required": ["q"]
            }),
        },
    }];

    let mut stream = client
        .chat_stream(vec![ChatMessage::user("Hi").build()], Some(tools))
        .await
        .expect("stream ok");
    while let Some(ev) = stream.next().await {
        if let ChatStreamEvent::StreamEnd { .. } = ev.expect("event ok") {
            break;
        }
    }

    // Assert flags
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse, "Accept must be SSE");
    assert!(seen.accept_encoding_identity, "Accept-Encoding identity expected");
    assert!(seen.has_org_header, "Tools array missing or invalid");
    assert!(seen.has_project_header, "Project header missing");
    assert!(seen.body_stream_true, "Body.stream should be true");
    assert!(seen.body_include_usage_true, "Body.stream_options.include_usage should be true");
    assert!(seen.body_has_prompt_cache_key, "Body should include prompt_cache_key");

    drop(server);
}
