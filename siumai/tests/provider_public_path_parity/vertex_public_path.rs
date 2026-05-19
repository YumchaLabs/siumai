use super::*;
#[cfg(feature = "bedrock")]
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::extensions::VideoGenerationCapability;
use siumai::extensions::types::{VideoGenerationInput, VideoGenerationRequest};
use siumai::prelude::unified::{
    ContentPart, EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::anthropic_vertex::{
    VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
};
#[cfg(feature = "bedrock")]
use siumai::provider_ext::bedrock::{BedrockEmbeddingOptions, BedrockEmbeddingRequestExt};
use siumai::provider_ext::google_vertex::{
    GoogleVertexReferenceImage, GoogleVertexVideoModelOptions, VertexEmbeddingOptions,
    VertexEmbeddingRequestExt, VertexImagenEditOptions, VertexImagenOptions,
    VertexImagenRequestExt, VertexVideoRequestExt,
};
use siumai_core::types::EmbeddingTaskType;

fn vertex_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("vertex", "vertex")
}

fn anthropic_vertex_registry_providers()
-> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("anthropic-vertex", "anthropic-vertex")
}

fn anthropic_vertex_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("anthropic-vertex", "anthropic-vertex")
}

fn vertex_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("vertex", "vertex")
}

fn make_vertex_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    vertex_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    vertex_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "vertex",
            "ctx-key",
            "https://example.com/custom",
            vertex_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn vertex_reasoning_stream_body() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai-stream")
        .join("google-thought-signature-reasoning.1.chunks.txt");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read vertex reasoning fixture failed: {path:?}: {err}"));

    let mut sse = String::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        sse.push_str("data: ");
        sse.push_str(line);
        sse.push_str("\n\n");
    }
    sse.push_str("data: [DONE]\n\n");
    sse.into_bytes()
}

fn vertex_structured_output_success_stream_body() -> Vec<u8> {
    concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"{\\\"answer\\\":\\\"hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\\\"}\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"safetyRatings\":[{\"category\":\"HARM_CATEGORY_DEROGATORY\",\"probability\":\"NEGLIGIBLE\"}]}],\"usageMetadata\":{\"promptTokenCount\":7,\"candidatesTokenCount\":4,\"totalTokenCount\":11}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec()
}

fn vertex_structured_output_interrupted_stream_body() -> Vec<u8> {
    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"{\\\"answer\\\":\"}]}}]}\n\n"
        .as_bytes()
        .to_vec()
}

fn custom_events_by_type(
    events: &[siumai::prelude::unified::ChatStreamEvent],
    ty: &str,
) -> Vec<serde_json::Value> {
    let stable_parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Part { part }
            | siumai::prelude::unified::ChatStreamEvent::PartWithReplay { part, .. } => {
                serde_json::to_value(part).ok()
            }
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect();
    if !stable_parts.is_empty() {
        return stable_parts;
    }

    events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Custom { data, .. } => Some(data.clone()),
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect()
}

async fn collect_stream_events(
    stream: &mut siumai::prelude::unified::ChatStream,
) -> Vec<siumai::prelude::unified::ChatStreamEvent> {
    use futures_util::StreamExt;

    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(err) => panic!("collect vertex public-path stream event failed: {err:?}"),
        }
    }
    events
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    vertex_registry_builder()
        .with_provider_api_key_base_url_fetch("vertex", "test-key", base_url, transport)
        .build()
        .expect("build registry")
}

fn make_anthropic_vertex_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    let mut http_config = siumai::prelude::unified::HttpConfig::empty();
    http_config
        .headers
        .insert("authorization".to_string(), "Bearer test-token".to_string());

    anthropic_vertex_registry_builder()
        .with_provider_base_url_http_config_fetch(
            "anthropic-vertex",
            base_url,
            http_config,
            transport,
        )
        .build()
        .expect("build registry")
}

fn anthropic_vertex_structured_output_success_stream_body(model: &str) -> Vec<u8> {
    let mut body = String::new();
    body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_test","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":15,"output_tokens":0}}}}}}"#
        ));
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"json","input":{}}}"#,
        );
    body.push_str("\n\n");

    let part1 =
        serde_json::to_string("{\"value\":\"te").expect("serialize anthropic-vertex text part");
    body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{part1}}}}}"#
        ));
    body.push_str("\n\n");

    let part2 = serde_json::to_string("st\"}").expect("serialize anthropic-vertex text part");
    body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{part2}}}}}"#
        ));
    body.push_str("\n\n");

    body.push_str(r#"data: {"type":"content_block_stop","index":0}"#);
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":15,"output_tokens":4}}"#,
        );
    body.push_str("\n\n");
    body.push_str(r#"data: {"type":"message_stop"}"#);
    body.push_str("\n\n");
    body.into_bytes()
}

fn anthropic_vertex_structured_output_interrupted_stream_body(model: &str) -> Vec<u8> {
    let mut body = String::new();
    body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_test","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":15,"output_tokens":0}}}}}}"#
        ));
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"json","input":{}}}"#,
        );
    body.push_str("\n\n");

    let partial = serde_json::to_string("{\"value\":").expect("serialize anthropic-vertex partial");
    body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{partial}}}}}"#
        ));
    body.push_str("\n\n");
    body.into_bytes()
}

fn anthropic_vertex_structured_output_success_response(model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "json",
                "input": {
                    "value": "test"
                }
            }
        ],
        "model": model,
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "output_tokens": 4
        }
    })
}

fn anthropic_vertex_structured_output_invalid_response(model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "sorry, not valid json"
            }
        ],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "output_tokens": 4
        }
    })
}

fn anthropic_vertex_reasoning_response(model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_reasoning",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "thinking",
                "thinking": "Count the letters carefully. The word strawberry contains three r characters.",
                "signature": "sig-1"
            },
            {
                "type": "redacted_thinking",
                "data": "redacted-blob"
            },
            {
                "type": "text",
                "text": "There are three letter r's in strawberry."
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 18,
            "output_tokens": 24
        }
    })
}

fn anthropic_vertex_reasoning_stream_body(model: &str) -> Vec<u8> {
    let mut body = String::new();
    body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_reasoning","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":18,"output_tokens":1}}}}}}"#
        ));
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
        );
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Count the letters carefully. "}}"#,
        );
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The word strawberry contains three r characters."}}"#,
        );
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#,
        );
    body.push_str("\n\n");
    body.push_str(r#"data: {"type":"content_block_stop","index":0}"#);
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"redacted-blob"}}"#,
        );
    body.push_str("\n\n");
    body.push_str(r#"data: {"type":"content_block_stop","index":1}"#);
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}"#,
        );
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"There are three letter r's in strawberry."}}"#,
        );
    body.push_str("\n\n");
    body.push_str(r#"data: {"type":"content_block_stop","index":2}"#);
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":18,"output_tokens":24}}"#,
        );
    body.push_str("\n\n");
    body.push_str(r#"data: {"type":"message_stop"}"#);
    body.push_str("\n\n");
    body.into_bytes()
}

fn make_anthropic_vertex_structured_output_request(
    model: &str,
    schema: serde_json::Value,
) -> ChatRequest {
    ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema))
        .build()
}

fn make_anthropic_vertex_reasoning_request(model: &str) -> ChatRequest {
    ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_anthropic_vertex_options(
            siumai::provider_ext::anthropic_vertex::VertexAnthropicOptions::new()
                .with_thinking_mode(
                    siumai::provider_ext::anthropic_vertex::VertexAnthropicThinkingMode::enabled(
                        Some(2048),
                    ),
                ),
        )
}

fn anthropic_vertex_default_options_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    })
}

fn make_anthropic_vertex_default_options_request(model: &str) -> ChatRequest {
    let assistant = ChatMessage::assistant_with_content(vec![
        ContentPart::reasoning("secret reasoning that should not be replayed"),
        ContentPart::text("previous visible answer"),
    ])
    .build();

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build(), assistant])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .response_format(ResponseFormat::json_schema(
            anthropic_vertex_default_options_schema(),
        ))
        .temperature(0.5)
        .top_p(0.7)
        .max_tokens(2000)
        .build();
    request.common_params.top_k = Some(16.0);
    request
}

fn assert_anthropic_vertex_default_options_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
    schema: &serde_json::Value,
) {
    assert_eq!(req.url, format!("{base_url}/models/{model}:rawPredict"));
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert_eq!(
        req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert_eq!(
        req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048
        })
    );
    assert_eq!(req.body["max_tokens"], serde_json::json!(6096));
    assert!(req.body.get("temperature").is_none());
    assert!(req.body.get("top_p").is_none());
    assert!(req.body.get("top_k").is_none());
    assert!(req.body.get("output_format").is_none());
    assert!(req.body.get("model").is_none());
    assert!(req.body.get("stream").is_none());

    assert_eq!(
        req.body["tool_choice"],
        serde_json::json!({
            "type": "any",
            "disable_parallel_tool_use": true
        })
    );

    let tools = req.body["tools"].as_array().expect("anthropic tools array");
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("lookup_weather")
        }),
        "expected user tool in request body: {:?}",
        req.body["tools"]
    );
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("json")
                && tool.get("input_schema") == Some(schema)
        }),
        "expected reserved json tool in request body: {:?}",
        req.body["tools"]
    );

    let messages = req.body["messages"].as_array().expect("anthropic messages");
    let serialized_messages =
        serde_json::to_string(messages).expect("serialize anthropic messages");
    assert!(
        !serialized_messages.contains("<thinking>"),
        "expected no <thinking> replay when send_reasoning=false: {serialized_messages}"
    );
    assert!(
        !serialized_messages.contains("secret reasoning that should not be replayed"),
        "expected no raw reasoning replay when send_reasoning=false: {serialized_messages}"
    );

    let assistant_content = messages[1]["content"]
        .as_array()
        .expect("assistant content array");
    assert!(
        assistant_content.iter().all(|part| {
            !matches!(
                part.get("type").and_then(|value| value.as_str()),
                Some("thinking" | "redacted_thinking")
            )
        }),
        "expected assistant content without thinking blocks: {assistant_content:?}"
    );
    assert!(
        assistant_content.iter().any(|part| {
            part.get("type").and_then(|value| value.as_str()) == Some("text")
                && part.get("text").and_then(|value| value.as_str())
                    == Some("previous visible answer")
        }),
        "expected assistant visible text to remain after stripping reasoning: {assistant_content:?}"
    );
}

fn assert_anthropic_vertex_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
    schema: &serde_json::Value,
) {
    assert_eq!(
        req.url,
        format!("{base_url}/models/{model}:streamRawPredict")
    );
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert_eq!(
        req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048
        })
    );
    assert_eq!(req.body["max_tokens"], serde_json::json!(6096));
    assert!(req.body.get("temperature").is_none());
    assert!(req.body.get("top_p").is_none());
    assert!(req.body.get("top_k").is_none());
    assert!(req.body.get("output_format").is_none());
    assert!(req.body.get("model").is_none());
    assert_eq!(req.body["stream"], serde_json::json!(true));

    assert_eq!(
        req.body["tool_choice"],
        serde_json::json!({
            "type": "any",
            "disable_parallel_tool_use": true
        })
    );

    let tools = req.body["tools"].as_array().expect("anthropic tools array");
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("lookup_weather")
        }),
        "expected user tool in request body: {:?}",
        req.body["tools"]
    );
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("json")
                && tool.get("input_schema") == Some(schema)
        }),
        "expected reserved json tool in request body: {:?}",
        req.body["tools"]
    );

    let messages = req.body["messages"].as_array().expect("anthropic messages");
    let serialized_messages =
        serde_json::to_string(messages).expect("serialize anthropic messages");
    assert!(
        !serialized_messages.contains("<thinking>"),
        "expected no <thinking> replay when send_reasoning=false: {serialized_messages}"
    );
    assert!(
        !serialized_messages.contains("secret reasoning that should not be replayed"),
        "expected no raw reasoning replay when send_reasoning=false: {serialized_messages}"
    );

    let assistant_content = messages[1]["content"]
        .as_array()
        .expect("assistant content array");
    assert!(
        assistant_content.iter().all(|part| {
            !matches!(
                part.get("type").and_then(|value| value.as_str()),
                Some("thinking" | "redacted_thinking")
            )
        }),
        "expected assistant content without thinking blocks: {assistant_content:?}"
    );
    assert!(
        assistant_content.iter().any(|part| {
            part.get("type").and_then(|value| value.as_str()) == Some("text")
                && part.get("text").and_then(|value| value.as_str())
                    == Some("previous visible answer")
        }),
        "expected assistant visible text to remain after stripping reasoning: {assistant_content:?}"
    );
}

fn assert_anthropic_vertex_structured_output_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
    schema: &serde_json::Value,
) {
    assert_eq!(
        req.url,
        format!("{base_url}/models/{model}:streamRawPredict")
    );
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert_eq!(
        req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert!(req.body.get("output_format").is_none());
    assert_eq!(req.body["tool_choice"]["type"], serde_json::json!("any"));
    assert_eq!(
        req.body["tool_choice"]["disable_parallel_tool_use"],
        serde_json::json!(true)
    );
    let tools = req.body["tools"].as_array().expect("anthropic tools array");
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("json")
                && tool.get("input_schema") == Some(schema)
        }),
        "expected reserved json tool in request body: {:?}",
        req.body["tools"]
    );
    assert!(req.body.get("model").is_none());
}

fn assert_anthropic_vertex_structured_output_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
    schema: &serde_json::Value,
) {
    assert_eq!(req.url, format!("{base_url}/models/{model}:rawPredict"));
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert_eq!(
        req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert!(req.body.get("output_format").is_none());
    assert_eq!(req.body["tool_choice"]["type"], serde_json::json!("any"));
    assert_eq!(
        req.body["tool_choice"]["disable_parallel_tool_use"],
        serde_json::json!(true)
    );
    let tools = req.body["tools"].as_array().expect("anthropic tools array");
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|value| value.as_str()) == Some("json")
                && tool.get("input_schema") == Some(schema)
        }),
        "expected reserved json tool in request body: {:?}",
        req.body["tools"]
    );
    assert!(req.body.get("stream").is_none());
    assert!(req.body.get("model").is_none());
}

fn assert_anthropic_vertex_reasoning_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
    stream: bool,
) {
    let suffix = if stream {
        ":streamRawPredict"
    } else {
        ":rawPredict"
    };
    assert_eq!(req.url, format!("{base_url}/models/{model}{suffix}"));
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    if stream {
        assert_eq!(
            header_value(req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.body["stream"], serde_json::json!(true));
    } else {
        assert!(req.body.get("stream").is_none());
    }
    assert_eq!(
        req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert_eq!(
        req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048
        })
    );
    assert!(req.body.get("model").is_none());
}

#[tokio::test]
async fn vertex_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "gemini-2.5-flash",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("gemini-2.5-flash");

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req.url.contains("key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
}

#[tokio::test]
async fn vertex_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let vertex_transport = CaptureTransport::default();

    let registry = make_vertex_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(vertex_transport.clone()),
    );

    let handle = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build vertex handle");

    let _ = handle
        .chat_request(make_chat_request_with_model("gemini-2.5-flash"))
        .await;

    let req = vertex_transport.take().expect("captured vertex request");
    assert!(global_transport.take().is_none());
    assert!(
        req.url.starts_with("https://example.com/custom"),
        "unexpected url: {}",
        req.url
    );
    assert!(
        req.url
            .contains("/models/gemini-2.5-flash:generateContent?key=ctx-key"),
        "unexpected url: {}",
        req.url
    );
    assert_eq!(
        req.body["contents"][0]["parts"][0]["text"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn vertex_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let vertex_transport = CaptureTransport::default();

    let registry = make_vertex_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(vertex_transport.clone()),
    );

    let handle = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build vertex handle");

    let _ = handle
        .chat_stream_request(make_chat_request_with_model("gemini-2.5-flash"))
        .await;

    let req = vertex_transport
        .take_stream()
        .expect("captured vertex stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert!(
        req.url.starts_with("https://example.com/custom"),
        "unexpected url: {}",
        req.url
    );
    assert!(
        req.url
            .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=ctx-key"),
        "unexpected url: {}",
        req.url
    );
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        req.body["contents"][0]["parts"][0]["text"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_stream_reasoning_events_keep_vertex_namespace() {
    let stream_body = vertex_reasoning_stream_body();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

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
    let mut registry_stream = registry_model
        .chat_stream_request(make_chat_request_with_model(model))
        .await
        .expect("registry stream ok");

    let siumai_events = collect_stream_events(&mut siumai_stream).await;
    let provider_events = collect_stream_events(&mut provider_stream).await;
    let config_events = collect_stream_events(&mut config_stream).await;
    let registry_events = collect_stream_events(&mut registry_stream).await;

    let siumai_starts = custom_events_by_type(&siumai_events, "reasoning-start");
    let provider_starts = custom_events_by_type(&provider_events, "reasoning-start");
    let config_starts = custom_events_by_type(&config_events, "reasoning-start");
    let registry_starts = custom_events_by_type(&registry_events, "reasoning-start");
    assert_eq!(
        siumai_starts.len(),
        1,
        "expected one siumai reasoning-start"
    );
    assert_eq!(
        provider_starts.len(),
        1,
        "expected one provider reasoning-start"
    );
    assert_eq!(
        config_starts.len(),
        1,
        "expected one config reasoning-start"
    );
    assert_eq!(
        registry_starts.len(),
        1,
        "expected one registry reasoning-start"
    );

    let siumai_deltas = custom_events_by_type(&siumai_events, "reasoning-delta");
    let provider_deltas = custom_events_by_type(&provider_events, "reasoning-delta");
    let config_deltas = custom_events_by_type(&config_events, "reasoning-delta");
    let registry_deltas = custom_events_by_type(&registry_events, "reasoning-delta");
    assert_eq!(
        siumai_deltas.len(),
        1,
        "expected one siumai reasoning-delta"
    );
    assert_eq!(
        provider_deltas.len(),
        1,
        "expected one provider reasoning-delta"
    );
    assert_eq!(
        config_deltas.len(),
        1,
        "expected one config reasoning-delta"
    );
    assert_eq!(
        registry_deltas.len(),
        1,
        "expected one registry reasoning-delta"
    );

    for event in [
        &siumai_starts[0],
        &provider_starts[0],
        &config_starts[0],
        &registry_starts[0],
        &siumai_deltas[0],
        &provider_deltas[0],
        &config_deltas[0],
        &registry_deltas[0],
    ] {
        assert_eq!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("vertex"))
                .and_then(|meta| meta.get("thoughtSignature"))
                .and_then(|value| value.as_str()),
            Some("stream_sig")
        );
        assert!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("google"))
                .is_none(),
            "did not expect providerMetadata.google on vertex path"
        );
    }

    let collect_reasoning = |events: &[siumai::prelude::unified::ChatStreamEvent]| {
        events
            .iter()
            .filter_map(siumai::prelude::unified::ChatStreamEvent::reasoning_delta)
            .collect::<String>()
    };

    for reasoning in [
        collect_reasoning(&siumai_events),
        collect_reasoning(&provider_events),
        collect_reasoning(&config_events),
        collect_reasoning(&registry_events),
    ] {
        assert_eq!(reasoning, "thinking...");
    }

    let siumai_end = siumai_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected siumai StreamEnd");
    let provider_end = provider_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected provider StreamEnd");
    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");
    let registry_end = registry_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected registry StreamEnd");

    for response in [siumai_end, provider_end, config_end, registry_end] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected stream-end provider metadata");
        let vertex_meta = provider_metadata
            .get("vertex")
            .expect("expected provider_metadata.vertex");
        assert!(
            !provider_metadata.contains_key("google"),
            "did not expect provider_metadata.google on vertex path"
        );
        assert_eq!(
            vertex_meta
                .get("usageMetadata")
                .and_then(|usage| usage.get("totalTokenCount"))
                .and_then(|value| value.as_u64()),
            Some(3)
        );
        assert_eq!(
            vertex_meta
                .get("safetyRatings")
                .and_then(|ratings| ratings.as_array())
                .and_then(|ratings| ratings.first())
                .and_then(|rating| rating.get("category"))
                .and_then(|value| value.as_str()),
            Some("HARM_CATEGORY_DEROGATORY")
        );
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
    assert!(
        siumai_req.url.contains(":streamGenerateContent?alt=sse"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "gemini-2.5-flash",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("gemini-2.5-flash");

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
    assert!(
        siumai_req.url.contains(":streamGenerateContent?alt=sse"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[cfg(feature = "bedrock")]
#[tokio::test]
async fn bedrock_siumai_provider_config_embedding_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.titan-embed-text-v2:0";

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::single("bedrock embedding boundary")
        .with_model(model)
        .with_bedrock_embedding_options(
            BedrockEmbeddingOptions::new()
                .with_dimensions(512)
                .with_normalize(true),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert!(provider_client.as_embedding_capability().is_some());
    assert!(config_client.as_embedding_capability().is_some());
    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke"
    );
    assert_eq!(
        siumai_req.body,
        serde_json::json!({
            "inputText": "bedrock embedding boundary",
            "dimensions": 512,
            "normalize": true
        })
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "thinkingConfig": {
                    "thinkingBudget": 2048,
                    "includeThoughts": true
                },
                "structuredOutputs": true
            }),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("models/gemini-2.5-flash:generateContent?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn vertex_registry_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "thinkingConfig": {
                    "thinkingBudget": 2048,
                    "includeThoughts": true
                },
                "structuredOutputs": true
            }),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert!(
        registry_req
            .url
            .contains("models/gemini-2.5-flash:generateContent?key=test-key"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        registry_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        registry_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_tool_choice_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("models/gemini-2.5-flash:generateContent?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["toolConfig"],
        serde_json::json!({
            "functionCallingConfig": { "mode": "NONE" }
        })
    );
    assert_eq!(
        siumai_req.body["tools"][0]["functionDeclarations"][0]["name"],
        serde_json::json!("lookup_weather")
    );
}

#[tokio::test]
async fn vertex_registry_tool_choice_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build();

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert!(
        registry_req
            .url
            .contains("models/gemini-2.5-flash:generateContent?key=test-key"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        registry_req.body["toolConfig"],
        serde_json::json!({
            "functionCallingConfig": { "mode": "NONE" }
        })
    );
    assert_eq!(
        registry_req.body["tools"][0]["functionDeclarations"][0]["name"],
        serde_json::json!("lookup_weather")
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_chat_stream_with_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "thinkingConfig": {
                    "thinkingBudget": 2048,
                    "includeThoughts": true
                },
                "structuredOutputs": true
            }),
        );

    let _ = siumai_client.chat_stream_request(request.clone()).await;
    let _ = provider_client.chat_stream_request(request.clone()).await;
    let _ = config_client.chat_stream_request(request).await;

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
    assert!(
        siumai_req
            .url
            .contains("models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
    assert_eq!(
        siumai_req.body["toolConfig"],
        serde_json::json!({
            "functionCallingConfig": { "mode": "NONE" }
        })
    );
    assert_eq!(
        siumai_req.body["tools"][0]["functionDeclarations"][0]["name"],
        serde_json::json!("lookup_weather")
    );
}

#[tokio::test]
async fn vertex_registry_stream_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "thinkingConfig": {
                    "thinkingBudget": 2048,
                    "includeThoughts": true
                },
                "structuredOutputs": true
            }),
        );

    let _ = config_client.chat_stream_request(request.clone()).await;
    let _ = registry_model.chat_stream_request(request).await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert!(
        registry_req
            .url
            .contains("models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        registry_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        registry_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
    assert_eq!(
        registry_req.body["toolConfig"],
        serde_json::json!({
            "functionCallingConfig": { "mode": "NONE" }
        })
    );
    assert_eq!(
        registry_req.body["tools"][0]["functionDeclarations"][0]["name"],
        serde_json::json!("lookup_weather")
    );
}

#[tokio::test]
async fn vertex_structured_output_stream_end_preserves_metadata_and_extracts_json_consistently_across_public_paths()
 {
    let stream_body = vertex_structured_output_success_stream_body();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "structuredOutputs": true
            }),
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
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let siumai_events = collect_stream_events(&mut siumai_stream).await;
    let provider_events = collect_stream_events(&mut provider_stream).await;
    let config_events = collect_stream_events(&mut config_stream).await;
    let registry_events = collect_stream_events(&mut registry_stream).await;

    let collect_content = |events: &[siumai::prelude::unified::ChatStreamEvent]| {
        events
            .iter()
            .filter_map(siumai::prelude::unified::ChatStreamEvent::text_delta)
            .collect::<String>()
    };

    for content in [
        collect_content(&siumai_events),
        collect_content(&provider_events),
        collect_content(&config_events),
        collect_content(&registry_events),
    ] {
        assert_eq!(content, "{\"answer\":\"hello\"}");
    }

    let siumai_end = siumai_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected siumai StreamEnd");
    let provider_end = provider_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected provider StreamEnd");
    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");
    let registry_end = registry_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected registry StreamEnd");

    for response in [siumai_end, provider_end, config_end, registry_end] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected stream-end provider metadata");
        let vertex_meta = provider_metadata
            .get("vertex")
            .expect("expected provider_metadata.vertex");
        assert!(
            !provider_metadata.contains_key("google"),
            "did not expect provider_metadata.google on vertex path"
        );
        assert_eq!(
            vertex_meta
                .get("usageMetadata")
                .and_then(|usage| usage.get("totalTokenCount"))
                .and_then(|value| value.as_u64()),
            Some(11)
        );
        assert_eq!(
            vertex_meta
                .get("safetyRatings")
                .and_then(|ratings| ratings.as_array())
                .and_then(|ratings| ratings.first())
                .and_then(|rating| rating.get("category"))
                .and_then(|value| value.as_str()),
            Some("HARM_CATEGORY_DEROGATORY")
        );
    }

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(siumai_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(provider_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(config_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("config structured output");
    let registry_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(registry_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("registry structured output");

    assert_eq!(siumai_value["answer"], "hello");
    assert_eq!(provider_value["answer"], "hello");
    assert_eq!(config_value["answer"], "hello");
    assert_eq!(registry_value["answer"], "hello");

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
    assert!(
        siumai_req
            .url
            .contains("models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn vertex_structured_output_interrupted_stream_fails_consistently_across_public_paths() {
    let stream_body = vertex_structured_output_interrupted_stream_body();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("vertex:gemini-2.5-flash")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "vertex",
            serde_json::json!({
                "structuredOutputs": true
            }),
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
    assert!(
        siumai_req
            .url
            .contains("models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn anthropic_vertex_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://example.com/custom")
        .model("claude-3-5-sonnet-v2@20241022")
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .language_model("claude-3-5-sonnet-v2@20241022")
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(
            "https://example.com/custom",
            "claude-3-5-sonnet-v2@20241022",
        )
        .with_bearer_token("test-token")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("claude-3-5-sonnet-v2@20241022");

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/claude-3-5-sonnet-v2@20241022:rawPredict"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["messages"][0]["content"][0]["text"],
        serde_json::json!("hi")
    );
    assert_eq!(
        siumai_req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert!(siumai_req.body.get("model").is_none());
}

#[tokio::test]
async fn anthropic_vertex_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://example.com/custom")
        .model("claude-3-5-sonnet-v2@20241022")
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .model("claude-3-5-sonnet-v2@20241022")
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(
            "https://example.com/custom",
            "claude-3-5-sonnet-v2@20241022",
        )
        .with_bearer_token("test-token")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("claude-3-5-sonnet-v2@20241022");

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
    assert!(
        siumai_req
            .url
            .contains("/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
}

#[tokio::test]
async fn anthropic_vertex_siumai_provider_config_chat_request_respect_explicit_request_model() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let default_model = "claude-3-5-sonnet-v2@20241022";
    let request_model = "claude-3-7-sonnet@20250219";

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://example.com/custom")
        .model(default_model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .model(default_model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(
            "https://example.com/custom",
            default_model,
        )
        .with_bearer_token("test-token")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(request_model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/claude-3-7-sonnet@20250219:rawPredict"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert!(siumai_req.body.get("model").is_none());
}

#[tokio::test]
async fn anthropic_vertex_siumai_provider_config_chat_stream_request_respect_explicit_request_model()
 {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let default_model = "claude-3-5-sonnet-v2@20241022";
    let request_model = "claude-3-7-sonnet@20250219";

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url("https://example.com/custom")
        .model(default_model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url("https://example.com/custom")
        .model(default_model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(
            "https://example.com/custom",
            default_model,
        )
        .with_bearer_token("test-token")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(request_model);

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
    assert!(
        siumai_req
            .url
            .contains("/models/claude-3-7-sonnet@20250219:streamRawPredict"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert!(siumai_req.body.get("model").is_none());
}

#[tokio::test]
async fn anthropic_vertex_registry_chat_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let model = "claude-3-5-sonnet-v2@20241022";

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert!(
        registry_req
            .url
            .contains("/models/claude-3-5-sonnet-v2@20241022:rawPredict"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        registry_req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
    assert!(registry_req.body.get("model").is_none());
}

#[tokio::test]
async fn anthropic_vertex_registry_chat_stream_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let model = "claude-3-5-sonnet-v2@20241022";

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022")
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

    use futures_util::StreamExt;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert!(
        registry_req
            .url
            .contains("/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        registry_req.body["anthropic_version"],
        serde_json::json!("vertex-2023-10-16")
    );
}

#[tokio::test]
async fn anthropic_vertex_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let default_model = "claude-3-5-sonnet-v2@20241022";
    let request_model = "claude-3-7-sonnet@20250219";

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, default_model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert!(
        registry_req
            .url
            .contains("/models/claude-3-7-sonnet@20250219:rawPredict"),
        "unexpected url: {}",
        registry_req.url
    );
}

#[tokio::test]
async fn anthropic_vertex_registry_chat_stream_request_with_explicit_request_model_match_config_path()
 {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let default_model = "claude-3-5-sonnet-v2@20241022";
    let request_model = "claude-3-7-sonnet@20250219";

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, default_model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model);

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-token".to_string())
    );
    assert!(
        registry_req
            .url
            .contains("/models/claude-3-7-sonnet@20250219:streamRawPredict"),
        "unexpected url: {}",
        registry_req.url
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn anthropic_vertex_reasoning_response_is_equivalent_across_public_paths() {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let response_json = anthropic_vertex_reasoning_response(model);

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let request = make_anthropic_vertex_reasoning_request(model);

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let expected_reasoning =
        "Count the letters carefully. The word strawberry contains three r characters.".to_string();

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(response.reasoning(), vec![expected_reasoning.clone()]);
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );

        let anthropic_meta = response
            .anthropic_metadata()
            .expect("expected typed anthropic metadata");
        assert_eq!(anthropic_meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            anthropic_meta.redacted_thinking_data.as_deref(),
            Some("redacted-blob")
        );
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_anthropic_vertex_reasoning_request(&siumai_req, base_url, model, false);
}

#[tokio::test]
async fn anthropic_vertex_reasoning_stream_is_equivalent_across_public_paths() {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let stream_body = anthropic_vertex_reasoning_stream_body(model);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let request = make_anthropic_vertex_reasoning_request(model);

    use futures_util::StreamExt;
    let collect_stream_summary = async |stream: &mut siumai::prelude::unified::ChatStream| {
        let mut end = None;
        let mut reasoning = String::new();
        while let Some(event) = stream.next().await {
            match event.expect("stream event ok") {
                event if event.reasoning_delta().is_some() => {
                    reasoning.push_str(event.reasoning_delta().expect("reasoning delta"));
                }
                siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => {
                    end = Some(response);
                    break;
                }
                _ => {}
            }
        }
        (reasoning, end)
    };

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let (siumai_reasoning, siumai_resp) = collect_stream_summary(&mut siumai_stream).await;
    let (provider_reasoning, provider_resp) = collect_stream_summary(&mut provider_stream).await;
    let (config_reasoning, config_resp) = collect_stream_summary(&mut config_stream).await;
    let (registry_reasoning, registry_resp) = collect_stream_summary(&mut registry_stream).await;

    let siumai_resp = siumai_resp.expect("siumai stream end");
    let provider_resp = provider_resp.expect("provider stream end");
    let config_resp = config_resp.expect("config stream end");
    let registry_resp = registry_resp.expect("registry stream end");

    let expected_reasoning =
        "Count the letters carefully. The word strawberry contains three r characters.".to_string();

    for reasoning in [
        &siumai_reasoning,
        &provider_reasoning,
        &config_reasoning,
        &registry_reasoning,
    ] {
        assert_eq!(reasoning, &expected_reasoning);
    }

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );

        let anthropic_meta = response
            .anthropic_metadata()
            .expect("expected typed anthropic metadata");
        assert_eq!(anthropic_meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            anthropic_meta.redacted_thinking_data.as_deref(),
            Some("redacted-blob")
        );
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
    assert_anthropic_vertex_reasoning_request(&siumai_req, base_url, model, true);
}

#[tokio::test]
async fn anthropic_vertex_default_options_match_public_request_shape() {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let schema = anthropic_vertex_default_options_schema();

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .with_anthropic_vertex_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_anthropic_vertex_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_vertex_disable_parallel_tool_use(true)
        .with_anthropic_vertex_send_reasoning(false)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
            .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
            .with_disable_parallel_tool_use(true)
            .with_send_reasoning(false)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_anthropic_vertex_default_options_request(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&provider_req, &config_req);
    assert_anthropic_vertex_default_options_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_default_options_match_public_stream_request_shape() {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let schema = anthropic_vertex_default_options_schema();

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .with_anthropic_vertex_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_anthropic_vertex_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_vertex_disable_parallel_tool_use(true)
        .with_anthropic_vertex_send_reasoning(false)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
        .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
        .with_disable_parallel_tool_use(true)
        .with_send_reasoning(false)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048)))
            .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool)
            .with_disable_parallel_tool_use(true)
            .with_send_reasoning(false)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_anthropic_vertex_default_options_request(model);

    let siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    drop(siumai_stream);
    drop(provider_stream);
    drop(config_stream);

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
    assert_requests_equivalent(&provider_req, &config_req);
    assert_anthropic_vertex_default_options_stream_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_structured_output_stream_end_extracts_json_consistently_across_public_paths()
 {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let stream_body = anthropic_vertex_structured_output_success_stream_body(model);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });
    let request = make_anthropic_vertex_structured_output_request(model, schema.clone());

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let siumai_events = collect_stream_events(&mut siumai_stream).await;
    let provider_events = collect_stream_events(&mut provider_stream).await;
    let config_events = collect_stream_events(&mut config_stream).await;
    let registry_events = collect_stream_events(&mut registry_stream).await;

    let collect_content = |events: &[siumai::prelude::unified::ChatStreamEvent]| {
        events
            .iter()
            .filter_map(siumai::prelude::unified::ChatStreamEvent::text_delta)
            .collect::<String>()
    };

    for content in [
        collect_content(&siumai_events),
        collect_content(&provider_events),
        collect_content(&config_events),
        collect_content(&registry_events),
    ] {
        assert_eq!(content, "{\"value\":\"test\"}");
    }

    let siumai_end = siumai_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected siumai StreamEnd");
    let provider_end = provider_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected provider StreamEnd");
    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");
    let registry_end = registry_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected registry StreamEnd");

    for response in [siumai_end, provider_end, config_end, registry_end] {
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );
        assert_eq!(response.id.as_deref(), Some("msg_test"));
        assert_eq!(response.model.as_deref(), Some(model));
        assert_eq!(response.content_text(), Some("{\"value\":\"test\"}"));
    }

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(siumai_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(provider_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(config_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("config structured output");
    let registry_value = siumai::structured_output::extract_json_value_from_stream(Box::pin(
        futures::stream::iter(registry_events.into_iter().map(Ok::<_, LlmError>)),
    ))
    .await
    .expect("registry structured output");

    assert_eq!(siumai_value["value"], "test");
    assert_eq!(provider_value["value"], "test");
    assert_eq!(config_value["value"], "test");
    assert_eq!(registry_value["value"], "test");

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
    assert_anthropic_vertex_structured_output_stream_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_structured_output_response_extracts_json_consistently_across_public_paths()
 {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let response = anthropic_vertex_structured_output_success_response(model);

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });
    let request = make_anthropic_vertex_structured_output_request(model, schema.clone());

    let siumai_response = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_response = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_response = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_response = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    for response in [
        &siumai_response,
        &provider_response,
        &config_response,
        &registry_response,
    ] {
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );
        assert_eq!(response.id.as_deref(), Some("msg_test"));
        assert_eq!(response.model.as_deref(), Some(model));
        assert_eq!(response.content_text(), Some("{\"value\":\"test\"}"));
    }

    let siumai_value =
        siumai::structured_output::extract_json_value_from_response(&siumai_response)
            .expect("siumai structured output");
    let provider_value =
        siumai::structured_output::extract_json_value_from_response(&provider_response)
            .expect("provider structured output");
    let config_value =
        siumai::structured_output::extract_json_value_from_response(&config_response)
            .expect("config structured output");
    let registry_value =
        siumai::structured_output::extract_json_value_from_response(&registry_response)
            .expect("registry structured output");

    assert_eq!(siumai_value["value"], "test");
    assert_eq!(provider_value["value"], "test");
    assert_eq!(config_value["value"], "test");
    assert_eq!(registry_value["value"], "test");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_anthropic_vertex_structured_output_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_structured_output_response_invalid_json_fails_consistently_across_public_paths()
 {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let response = anthropic_vertex_structured_output_invalid_response(model);

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });
    let request = make_anthropic_vertex_structured_output_request(model, schema.clone());

    let siumai_err = siumai::structured_output::extract_json_value_from_response(
        &siumai_client
            .chat_request(request.clone())
            .await
            .expect("siumai response ok"),
    )
    .expect_err("siumai invalid response should fail");
    let provider_err = siumai::structured_output::extract_json_value_from_response(
        &provider_client
            .chat_request(request.clone())
            .await
            .expect("provider response ok"),
    )
    .expect_err("provider invalid response should fail");
    let config_err = siumai::structured_output::extract_json_value_from_response(
        &config_client
            .chat_request(request.clone())
            .await
            .expect("config response ok"),
    )
    .expect_err("config invalid response should fail");
    let registry_err = siumai::structured_output::extract_json_value_from_response(
        &registry_model
            .chat_request(request)
            .await
            .expect("registry response ok"),
    )
    .expect_err("registry invalid response should fail");

    for err in [siumai_err, provider_err, config_err, registry_err] {
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("no valid JSON candidate found"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_anthropic_vertex_structured_output_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_structured_output_interrupted_stream_fails_consistently_across_public_paths()
 {
    let model = "claude-sonnet-4-6";
    let base_url = "https://example.com/custom";
    let stream_body = anthropic_vertex_structured_output_interrupted_stream_body(model);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic-vertex:claude-sonnet-4-6")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });
    let request = make_anthropic_vertex_structured_output_request(model, schema.clone());

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
    assert_anthropic_vertex_structured_output_stream_request(&siumai_req, base_url, model, &schema);
}

#[tokio::test]
async fn anthropic_vertex_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let vertex_transport = CaptureTransport::default();

    let mut global_http_config = siumai::prelude::unified::HttpConfig::empty();
    global_http_config.headers.insert(
        "authorization".to_string(),
        "Bearer global-token".to_string(),
    );
    global_http_config
        .headers
        .insert("x-global-header".to_string(), "keep-me".to_string());

    let mut provider_http_config = siumai::prelude::unified::HttpConfig::empty();
    provider_http_config
        .headers
        .insert("authorization".to_string(), "Bearer ctx-token".to_string());

    let registry = anthropic_vertex_registry_builder()
        .with_base_url("https://example.com/global")
        .with_http_config(global_http_config)
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_base_url_http_config_fetch(
            "anthropic-vertex",
            "https://example.com/custom",
            provider_http_config,
            Arc::new(vertex_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022")
        .expect("build registry language model");

    let _ = handle
        .chat_request(make_chat_request_with_model(
            "claude-3-5-sonnet-v2@20241022",
        ))
        .await;

    let req = vertex_transport.take().expect("captured request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-token".to_string())
    );
    assert_eq!(
        header_value(&req, "x-global-header"),
        Some("keep-me".to_string())
    );
    assert!(
        req.url.starts_with(
            "https://example.com/custom/models/claude-3-5-sonnet-v2@20241022:rawPredict"
        ),
        "unexpected url: {}",
        req.url
    );
}

#[tokio::test]
async fn anthropic_vertex_siumai_provider_config_non_text_requests_are_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "claude-3-5-sonnet-v2@20241022";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .http_header("authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic_vertex()
        .base_url(base_url)
        .model(model)
        .bearer_token("test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::anthropic_vertex::VertexAnthropicClient::from_config(
        siumai::provider_ext::anthropic_vertex::VertexAnthropicConfig::new(base_url, model)
            .with_bearer_token("test-token")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let embedding_request =
        EmbeddingRequest::single("hello anthropic-vertex embedding").with_model(model);
    let rerank_request = make_rerank_request_with_model(model).with_top_n(1);
    let image_request = make_image_request_with_model(model);

    let siumai_embedding_err = siumai_client
        .embed_with_config(embedding_request.clone())
        .await
        .expect_err("anthropic-vertex embedding should be unsupported");
    let siumai_rerank_err = siumai_client
        .rerank(rerank_request.clone())
        .await
        .expect_err("anthropic-vertex rerank should be unsupported");
    let siumai_image_err = siumai_client
        .generate_images(image_request.clone())
        .await
        .expect_err("anthropic-vertex image generation should be unsupported");

    assert_unsupported_operation(&siumai_embedding_err);
    assert_unsupported_operation(&siumai_rerank_err);
    assert_unsupported_operation(&siumai_image_err);

    assert_no_deferred_capability_leaks(&provider_client);
    assert_no_deferred_capability_leaks(&config_client);
    assert!(siumai_client.as_embedding_capability().is_none());
    assert!(siumai_client.as_image_generation_capability().is_none());
    assert!(siumai_client.as_rerank_capability().is_none());
    assert!(siumai_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());

    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn anthropic_vertex_registry_non_text_requests_are_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let registry = make_anthropic_vertex_registry(Arc::new(registry_transport.clone()), base_url);
    let embedding_err =
        match registry.embedding_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022") {
            Ok(_) => {
                panic!("build anthropic-vertex registry embedding model should be unsupported")
            }
            Err(err) => err,
        };
    let image_err = match registry.image_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022") {
        Ok(_) => panic!("build anthropic-vertex registry image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err =
        match registry.reranking_model("anthropic-vertex:claude-3-5-sonnet-v2@20241022") {
            Ok(_) => panic!("build anthropic-vertex registry rerank handle should be unsupported"),
            Err(err) => err,
        };

    assert_unsupported_operation(&embedding_err);
    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn vertex_siumai_provider_config_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("text-embedding-004")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .embedding_model("text-embedding-004")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "text-embedding-004",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let input = vec!["hello vertex".to_string(), "embedding parity".to_string()];

    let _ = siumai_client.embed(input.clone()).await;
    let _ = EmbeddingModel::embed(&provider_client, EmbeddingRequest::new(input.clone())).await;
    let _ = EmbeddingModel::embed(&config_client, EmbeddingRequest::new(input)).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/text-embedding-004:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("text-embedding-004")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .embedding_model("text-embedding-004")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "text-embedding-004",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::new(vec![
        "hello vertex".to_string(),
        "embedding parity".to_string(),
    ])
    .with_model("text-embedding-004")
    .with_task_type(EmbeddingTaskType::RetrievalDocument)
    .with_title("vertex-doc")
    .with_vertex_embedding_options(VertexEmbeddingOptions {
        output_dimensionality: Some(256),
        auto_truncate: Some(true),
        ..Default::default()
    });

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/text-embedding-004:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["instances"][0]["content"],
        serde_json::json!("hello vertex")
    );
    assert_eq!(
        siumai_req.body["instances"][0]["task_type"],
        serde_json::json!("RETRIEVAL_DOCUMENT")
    );
    assert_eq!(
        siumai_req.body["instances"][0]["title"],
        serde_json::json!("vertex-doc")
    );
    assert_eq!(
        siumai_req.body["parameters"]["outputDimensionality"],
        serde_json::json!(256)
    );
    assert_eq!(
        siumai_req.body["parameters"]["autoTruncate"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn vertex_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let model = "text-embedding-004";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .embedding_model("vertex:text-embedding-004")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::new(vec![
        "hello vertex".to_string(),
        "embedding parity".to_string(),
    ])
    .with_model(model)
    .with_task_type(EmbeddingTaskType::RetrievalDocument)
    .with_title("vertex-doc")
    .with_vertex_embedding_options(VertexEmbeddingOptions {
        output_dimensionality: Some(256),
        auto_truncate: Some(true),
        ..Default::default()
    });

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["instances"][0]["task_type"],
        serde_json::json!("RETRIEVAL_DOCUMENT")
    );
    assert_eq!(
        registry_req.body["instances"][0]["title"],
        serde_json::json!("vertex-doc")
    );
    assert_eq!(
        registry_req.body["parameters"]["outputDimensionality"],
        serde_json::json!(256)
    );
    assert_eq!(
        registry_req.body["parameters"]["autoTruncate"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_image_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("imagen-4.0-generate-001")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("imagen-4.0-generate-001")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "imagen-4.0-generate-001",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ImageGenerationRequest {
        prompt: "a tiny orange robot".to_string(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some("imagen-4.0-generate-001".to_string()),
        quality: None,
        style: None,
        seed: Some(7),
        steps: None,
        guidance_scale: None,
        enhance_prompt: Some(true),
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_vertex_imagen_options(
        VertexImagenOptions::new()
            .with_negative_prompt("blurry")
            .with_person_generation("allow_adult")
            .with_safety_setting("block_medium_and_above")
            .with_add_watermark(false)
            .with_storage_uri("gs://bucket/images/")
            .with_sample_image_size("2K"),
    );

    let _ = siumai_client.generate_images(request.clone()).await;
    let _ = provider_client.generate_images(request.clone()).await;
    let _ = config_client.generate_images(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/imagen-4.0-generate-001:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
}

#[tokio::test]
async fn vertex_registry_imagen_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let model = "imagen-4.0-generate-001";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("vertex:imagen-4.0-generate-001")
        .expect("build registry image model");

    let request = ImageGenerationRequest {
        prompt: "a tiny orange robot".to_string(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some(model.to_string()),
        quality: None,
        style: None,
        seed: Some(7),
        steps: None,
        guidance_scale: None,
        enhance_prompt: Some(true),
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_vertex_imagen_options(VertexImagenOptions::new().with_negative_prompt("blurry"));

    let _ = config_client.generate_images(request.clone()).await;
    let _ = registry_model.generate_images(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert!(
        registry_req
            .url
            .contains("/models/imagen-4.0-generate-001:predict?key=test-key"),
        "unexpected url: {}",
        registry_req.url
    );
}

#[tokio::test]
async fn vertex_registry_image_handle_prefers_provider_specific_build_overrides() {
    let image_response = serde_json::json!({
        "predictions": [
            {
                "bytesBase64Encoded": "aGVsbG8=",
                "mimeType": "image/png",
                "prompt": "a tiny orange robot"
            }
        ]
    });

    let global_transport = JsonSuccessTransport::new(image_response.clone());
    let vertex_transport = JsonSuccessTransport::new(image_response);

    let registry = make_vertex_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(vertex_transport.clone()),
    );

    let handle = registry
        .image_model("vertex:imagen-4.0-generate-001")
        .expect("build vertex image model");

    let generated = handle
        .generate_images(
            ImageGenerationRequest {
                prompt: "a tiny orange robot".to_string(),
                negative_prompt: None,
                size: Some("1024x1024".to_string()),
                aspect_ratio: None,
                count: 1,
                model: Some("imagen-4.0-generate-001".to_string()),
                quality: None,
                style: None,
                seed: Some(7),
                steps: None,
                guidance_scale: None,
                enhance_prompt: Some(true),
                response_format: Some("b64_json".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            }
            .with_vertex_imagen_options(
                VertexImagenOptions::new()
                    .with_negative_prompt("blurry")
                    .with_person_generation("allow_adult")
                    .with_safety_setting("block_medium_and_above")
                    .with_add_watermark(false)
                    .with_storage_uri("gs://bucket/images/")
                    .with_sample_image_size("2K"),
            ),
        )
        .await
        .expect("generate images through registry handle");

    assert_eq!(generated.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert!(global_transport.take().is_none());

    let req = vertex_transport.take().expect("captured vertex request");
    assert!(
        req.url.starts_with("https://example.com/custom"),
        "unexpected url: {}",
        req.url
    );
    assert!(
        req.url
            .contains("/models/imagen-4.0-generate-001:predict?key=ctx-key"),
        "unexpected url: {}",
        req.url
    );
    assert_eq!(
        req.body["instances"][0]["prompt"],
        serde_json::json!("a tiny orange robot")
    );
    assert_eq!(
        req.body["parameters"]["negativePrompt"],
        serde_json::json!("blurry")
    );
    assert_eq!(
        req.body["parameters"]["personGeneration"],
        serde_json::json!("allow_adult")
    );
    assert_eq!(
        req.body["parameters"]["safetySetting"],
        serde_json::json!("block_medium_and_above")
    );
    assert_eq!(
        req.body["parameters"]["addWatermark"],
        serde_json::json!(false)
    );
    assert_eq!(
        req.body["parameters"]["storageUri"],
        serde_json::json!("gs://bucket/images/")
    );
    assert_eq!(
        req.body["parameters"]["sampleImageSize"],
        serde_json::json!("2K")
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_image_edit_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("imagen-3.0-edit-001")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("imagen-3.0-edit-001")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(
            "https://example.com/custom",
            "imagen-3.0-edit-001",
        )
        .with_api_key("test-key")
        .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = siumai::extensions::types::ImageEditRequest {
        images: vec![siumai::extensions::types::ImageEditInput::file(vec![
            1, 2, 3, 4,
        ])],
        mask: Some(siumai::extensions::types::ImageEditInput::file(vec![
            5, 6, 7, 8,
        ])),
        prompt: "replace the masked region with a paper airplane".to_string(),
        model: Some("imagen-3.0-edit-001".to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        seed: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_vertex_imagen_options(
        VertexImagenOptions::new().with_edit(
            VertexImagenEditOptions::new()
                .with_mode("EDIT_MODE_INPAINT_INSERTION")
                .with_mask_mode("MASK_MODE_USER_PROVIDED"),
        ),
    );

    let _ = siumai_client.edit_image(request.clone()).await;
    let _ = provider_client.edit_image(request.clone()).await;
    let _ = config_client.edit_image(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert!(
        siumai_req
            .url
            .contains("/models/imagen-3.0-edit-001:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_registry_image_edit_data_url_request_are_equivalent() {
    let response = serde_json::json!({
        "predictions": [
            {
                "bytesBase64Encoded": "aGVsbG8=",
                "mimeType": "image/png"
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let base_url = "https://example.com/custom";
    let model = "imagen-3.0-edit-001";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("vertex:imagen-3.0-edit-001")
        .expect("build registry image model");

    let request = make_data_url_image_edit_request_with_model(model).with_vertex_imagen_options(
        VertexImagenOptions::new().with_edit(
            VertexImagenEditOptions::new()
                .with_mode("EDIT_MODE_INPAINT_INSERTION")
                .with_mask_mode("MASK_MODE_USER_PROVIDED"),
        ),
    );

    let siumai_resp = siumai_client
        .edit_image(request.clone())
        .await
        .expect("siumai image edit ok");
    let provider_resp = provider_client
        .edit_image(request.clone())
        .await
        .expect("provider image edit ok");
    let config_resp = config_client
        .edit_image(request.clone())
        .await
        .expect("config image edit ok");
    let registry_resp = registry_model
        .edit_image(request)
        .await
        .expect("registry image edit ok");

    assert_eq!(siumai_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );
    assert_eq!(config_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        registry_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert!(
        siumai_req
            .url
            .contains("/models/imagen-3.0-edit-001:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["instances"][0]["referenceImages"][0]["referenceImage"]["bytesBase64Encoded"],
        serde_json::json!("aW1hZ2Utb25l")
    );
    assert_eq!(
        siumai_req.body["instances"][0]["referenceImages"][1]["referenceImage"]["bytesBase64Encoded"],
        serde_json::json!("bWFzay1vbmU=")
    );
    assert_eq!(
        siumai_req.body["instances"][0]["referenceImages"][1]["maskImageConfig"]["maskMode"],
        serde_json::json!("MASK_MODE_USER_PROVIDED")
    );
    assert_eq!(
        siumai_req.body["parameters"]["editMode"],
        serde_json::json!("EDIT_MODE_INPAINT_INSERTION")
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_registry_image_variation_data_url_request_are_equivalent() {
    let response = serde_json::json!({
        "predictions": [
            {
                "bytesBase64Encoded": "aGVsbG8=",
                "mimeType": "image/png"
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let base_url = "https://example.com/custom";
    let model = "imagen-3.0-generate-001";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("vertex:imagen-3.0-generate-001")
        .expect("build registry image model");

    let request = make_data_url_image_variation_request_with_model(model)
        .with_vertex_imagen_options(
            VertexImagenOptions::new()
                .with_negative_prompt("blurry")
                .with_person_generation("allow_adult")
                .with_safety_setting("block_medium_and_above")
                .with_add_watermark(false)
                .with_storage_uri("gs://bucket/images/")
                .with_sample_image_size("2K"),
        );

    let siumai_resp = siumai_client
        .create_variation(request.clone())
        .await
        .expect("siumai image variation ok");
    let provider_resp = provider_client
        .create_variation(request.clone())
        .await
        .expect("provider image variation ok");
    let config_resp = config_client
        .create_variation(request.clone())
        .await
        .expect("config image variation ok");
    let registry_resp = registry_model
        .create_variation(request)
        .await
        .expect("registry image variation ok");

    assert_eq!(siumai_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );
    assert_eq!(config_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        registry_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert!(
        siumai_req
            .url
            .contains("/models/imagen-3.0-generate-001:predict?key=test-key"),
        "unexpected url: {}",
        siumai_req.url
    );
    assert_eq!(
        siumai_req.body["instances"][0]["prompt"],
        serde_json::json!("keep the subject and explore new backgrounds")
    );
    assert_eq!(
        siumai_req.body["instances"][0]["referenceImages"][0]["referenceImage"]["bytesBase64Encoded"],
        serde_json::json!("aW1hZ2Utb25l")
    );
    assert_eq!(
        siumai_req.body["parameters"]["sampleCount"],
        serde_json::json!(2)
    );
    assert_eq!(
        siumai_req.body["parameters"]["aspectRatio"],
        serde_json::json!("16:9")
    );
    assert_eq!(siumai_req.body["parameters"]["seed"], serde_json::json!(7));
    assert_eq!(
        siumai_req.body["parameters"]["negativePrompt"],
        serde_json::json!("blurry")
    );
    assert!(siumai_req.body["parameters"].get("editMode").is_none());
}

#[tokio::test]
async fn vertex_registry_imagen_edit_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let model = "imagen-3.0-edit-001";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("vertex:imagen-3.0-edit-001")
        .expect("build registry image model");

    let request = siumai::extensions::types::ImageEditRequest {
        images: vec![siumai::extensions::types::ImageEditInput::file(vec![
            1, 2, 3, 4,
        ])],
        mask: Some(siumai::extensions::types::ImageEditInput::file(vec![
            5, 6, 7, 8,
        ])),
        prompt: "replace the masked region with a paper airplane".to_string(),
        model: Some(model.to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        seed: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_vertex_imagen_options(
        VertexImagenOptions::new().with_edit(
            VertexImagenEditOptions::new()
                .with_mode("EDIT_MODE_INPAINT_INSERTION")
                .with_mask_mode("MASK_MODE_USER_PROVIDED"),
        ),
    );

    let _ = config_client.edit_image(request.clone()).await;
    let _ = registry_model.edit_image(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert!(
        registry_req
            .url
            .contains("/models/imagen-3.0-edit-001:predict?key=test-key"),
        "unexpected url: {}",
        registry_req.url
    );
}

#[tokio::test]
async fn vertex_registry_image_edit_handle_prefers_provider_specific_build_overrides() {
    let image_response = serde_json::json!({
        "predictions": [
            {
                "bytesBase64Encoded": "aGVsbG8=",
                "mimeType": "image/png"
            }
        ]
    });

    let global_transport = JsonSuccessTransport::new(image_response.clone());
    let vertex_transport = JsonSuccessTransport::new(image_response);

    let registry = make_vertex_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(vertex_transport.clone()),
    );

    let handle = registry
        .image_model("vertex:imagen-3.0-edit-001")
        .expect("build vertex image model");

    let generated = handle
        .edit_image(
            siumai::extensions::types::ImageEditRequest {
                images: vec![siumai::extensions::types::ImageEditInput::file(vec![
                    1, 2, 3, 4,
                ])],
                mask: Some(siumai::extensions::types::ImageEditInput::file(vec![
                    5, 6, 7, 8,
                ])),
                prompt: "replace the masked region with a paper airplane".to_string(),
                model: Some("imagen-3.0-edit-001".to_string()),
                count: Some(1),
                size: Some("1024x1024".to_string()),
                aspect_ratio: None,
                seed: None,
                response_format: Some("b64_json".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            }
            .with_vertex_imagen_options(
                VertexImagenOptions::new().with_edit(
                    VertexImagenEditOptions::new()
                        .with_mode("EDIT_MODE_INPAINT_INSERTION")
                        .with_mask_mode("MASK_MODE_USER_PROVIDED"),
                ),
            ),
        )
        .await
        .expect("edit image through registry handle");

    assert_eq!(generated.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert!(global_transport.take().is_none());

    let req = vertex_transport.take().expect("captured vertex request");
    assert!(
        req.url.starts_with("https://example.com/custom"),
        "unexpected url: {}",
        req.url
    );
    assert!(
        req.url
            .contains("/models/imagen-3.0-edit-001:predict?key=ctx-key"),
        "unexpected url: {}",
        req.url
    );
    assert_eq!(
        req.body["instances"][0]["prompt"],
        serde_json::json!("replace the masked region with a paper airplane")
    );
    assert_eq!(
        req.body["parameters"]["editMode"],
        serde_json::json!("EDIT_MODE_INPAINT_INSERTION")
    );
    assert_eq!(
        req.body["instances"][0]["referenceImages"][1]["maskImageConfig"]["maskMode"],
        serde_json::json!("MASK_MODE_USER_PROVIDED")
    );
}

fn make_vertex_video_generation_request(model: &str) -> VideoGenerationRequest {
    VideoGenerationRequest::new(model, "animate a tiny robot walking through neon rain")
        .with_n(2)
        .with_duration(6)
        .with_resolution("1920x1080")
        .with_aspect_ratio("16:9")
        .with_seed(7)
        .with_image(VideoGenerationInput::file_with_media_type(
            vec![1, 2, 3, 4],
            "image/png",
        ))
        .with_vertex_video_options(
            GoogleVertexVideoModelOptions::new()
                .with_poll_interval_ms(1200)
                .with_poll_timeout_ms(30_000)
                .with_negative_prompt("blurry")
                .with_person_generation("allow_adult")
                .with_generate_audio(true)
                .with_gcs_output_directory("gs://bucket/output/")
                .with_reference_images(vec![
                    GoogleVertexReferenceImage::new().with_gcs_uri("gs://bucket/reference.png"),
                ]),
        )
        .with_header("x-video-test", "1")
}

fn assert_vertex_video_create_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(
        req.url,
        format!("{base_url}/models/{model}:predictLongRunning?key=test-key")
    );
    assert_eq!(header_value(req, "x-video-test"), Some("1".to_string()));
    assert_eq!(
        req.body["instances"][0]["prompt"],
        serde_json::json!("animate a tiny robot walking through neon rain")
    );
    assert_eq!(
        req.body["instances"][0]["image"]["bytesBase64Encoded"],
        serde_json::json!("AQIDBA==")
    );
    assert_eq!(
        req.body["instances"][0]["image"]["mimeType"],
        serde_json::json!("image/png")
    );
    assert_eq!(
        req.body["instances"][0]["referenceImages"][0]["gcsUri"],
        serde_json::json!("gs://bucket/reference.png")
    );
    assert_eq!(req.body["parameters"]["sampleCount"], serde_json::json!(2));
    assert_eq!(
        req.body["parameters"]["durationSeconds"],
        serde_json::json!(6)
    );
    assert_eq!(
        req.body["parameters"]["resolution"],
        serde_json::json!("1080p")
    );
    assert_eq!(
        req.body["parameters"]["aspectRatio"],
        serde_json::json!("16:9")
    );
    assert_eq!(req.body["parameters"]["seed"], serde_json::json!(7));
    assert_eq!(
        req.body["parameters"]["negativePrompt"],
        serde_json::json!("blurry")
    );
    assert_eq!(
        req.body["parameters"]["personGeneration"],
        serde_json::json!("allow_adult")
    );
    assert_eq!(
        req.body["parameters"]["generateAudio"],
        serde_json::json!(true)
    );
    assert_eq!(
        req.body["parameters"]["gcsOutputDirectory"],
        serde_json::json!("gs://bucket/output/")
    );
    assert!(req.body["parameters"].get("pollIntervalMs").is_none());
    assert!(req.body["parameters"].get("pollTimeoutMs").is_none());
}

fn assert_vertex_video_query_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(
        req.url,
        format!("{base_url}/models/{model}:fetchPredictOperation?key=test-key")
    );
    assert_eq!(
        req.body["operationName"],
        serde_json::json!("operations/test-video-123")
    );
}

#[tokio::test]
async fn vertex_siumai_provider_config_registry_video_create_are_equivalent() {
    let response_json = serde_json::json!({
        "name": "operations/test-video-123",
        "done": false
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "veo-3.1-generate-preview";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_client = registry
        .video_model("vertex:veo-3.1-generate-preview")
        .expect("build registry video model");

    let request = make_vertex_video_generation_request(model);

    let siumai_resp = siumai_client
        .create_video_task(request.clone())
        .await
        .expect("siumai create video ok");
    let provider_resp = provider_client
        .create_video_task(request.clone())
        .await
        .expect("provider create video ok");
    let config_resp = config_client
        .create_video_task(request.clone())
        .await
        .expect("config create video ok");
    let registry_resp = registry_client
        .create_video_task(request)
        .await
        .expect("registry create video ok");

    assert_eq!(siumai_resp.task_id, "operations/test-video-123");
    assert_eq!(provider_resp.task_id, "operations/test-video-123");
    assert_eq!(config_resp.task_id, "operations/test-video-123");
    assert_eq!(registry_resp.task_id, "operations/test-video-123");
    assert_eq!(siumai_resp.warnings, None);
    assert_eq!(provider_resp.warnings, siumai_resp.warnings);
    assert_eq!(config_resp.warnings, siumai_resp.warnings);
    assert_eq!(registry_resp.warnings, siumai_resp.warnings);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_vertex_video_create_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn vertex_siumai_provider_config_registry_query_video_task_are_equivalent() {
    let response_json = serde_json::json!({
        "name": "operations/test-video-123",
        "done": true,
        "response": {
            "videos": [
                {
                    "gcsUri": "https://cdn.example.com/video.mp4",
                    "mimeType": "video/mp4"
                }
            ],
            "raiMediaFilteredCount": 1
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "veo-3.1-generate-preview";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::google_vertex::GoogleVertexClient::from_config(
        siumai::provider_ext::google_vertex::GoogleVertexConfig::new(base_url, model)
            .with_api_key("test-key")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_client = registry
        .video_model("vertex:veo-3.1-generate-preview")
        .expect("build registry video model");

    let siumai_resp = siumai_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("siumai query video ok");
    let provider_resp = provider_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("provider query video ok");
    let config_resp = config_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("config query video ok");
    let registry_resp = registry_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("registry query video ok");

    assert_eq!(siumai_resp.status.to_string(), "Success");
    assert_eq!(provider_resp.status.to_string(), "Success");
    assert_eq!(config_resp.status.to_string(), "Success");
    assert_eq!(registry_resp.status.to_string(), "Success");
    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(response.file_id, None);
        assert!(response.provider_reference().is_none());
        assert_eq!(
            response.video_url.as_deref(),
            Some("https://cdn.example.com/video.mp4")
        );
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_vertex_video_query_request(&siumai_req, base_url, model);
}
