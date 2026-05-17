use super::*;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use crate::providers::openai_compatible::OpenAiCompatibleConfig;
use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};
use crate::streaming::{ChatStreamEvent, ChatStreamPart};
use crate::types::{ChatMessage, FinishReason, ResponseFormat, Tool, ToolChoice};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::{Arc, Mutex};

fn make_text_streaming_adapter() -> Arc<ConfigurableAdapter> {
    Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "compat-chat".to_string(),
        name: "Compat Chat".to_string(),
        base_url: "https://api.test.com/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec![
            "completion".to_string(),
            "chat".to_string(),
            "streaming".to_string(),
            "tools".to_string(),
        ],
        default_model: Some("compat-default-model".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }))
}

fn make_completion_adapter() -> Arc<ConfigurableAdapter> {
    Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "openrouter".to_string(),
        name: "OpenRouter".to_string(),
        base_url: "https://openrouter.ai/api/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["completion".to_string(), "streaming".to_string()],
        default_model: Some("openai/gpt-3.5-turbo-instruct".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }))
}

#[derive(Clone)]
struct JsonResponseTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl JsonResponseTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().unwrap().take()
    }
}

#[async_trait]
impl HttpTransport for JsonResponseTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().unwrap() = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last.lock().unwrap() = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportStreamResponse {
                status: 501,
                headers,
                body: HttpTransportStreamBody::from_bytes(
                    br#"{"error":{"message":"stream unsupported in test","type":"test_error","code":"unsupported"}}"#
                        .to_vec(),
                ),
            })
    }
}

#[derive(Clone)]
struct SseResponseTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl SseResponseTransport {
    fn new(body: impl Into<Vec<u8>>) -> Self {
        Self {
            response_body: Arc::new(body.into()),
            last_stream: Arc::new(Mutex::new(None)),
        }
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().unwrap().take()
    }
}

#[async_trait]
impl HttpTransport for SseResponseTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
                status: 501,
                headers,
                body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                    .to_vec(),
            })
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().unwrap() = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
        })
    }
}

#[tokio::test]
async fn completion_request_runtime_routes_to_completions_and_materializes_prompt() {
    let transport = JsonResponseTransport::new(serde_json::json!({
        "id": "cmpl_1",
        "model": "compat-model",
        "created": 1718345013u64,
        "choices": [{
            "text": "done",
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9
        }
    }));

    let config = OpenAiCompatibleConfig::new(
        "compat-chat",
        "test-key",
        "https://api.test.com/v1",
        make_text_streaming_adapter(),
    )
    .with_model("compat-model")
    .with_http_transport(Arc::new(transport.clone()));
    let client = OpenAiCompatibleClient::new(config).await.unwrap();

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model("compat-model");

    let response = crate::traits::CompletionCapability::complete(&client, request)
        .await
        .unwrap();

    assert_eq!(response.text(), "done");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.raw_finish_reason.as_deref(), Some("stop"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(9)
    );
    assert_eq!(
        response
            .response_metadata
            .as_ref()
            .and_then(|metadata| metadata.model.as_deref()),
        Some("compat-model")
    );

    let captured = transport.take().expect("captured completion request");
    assert_eq!(captured.url, "https://api.test.com/v1/completions");
    assert_eq!(captured.body["model"], serde_json::json!("compat-model"));
    assert_eq!(
        captured.body["prompt"],
        serde_json::json!(
            "Be terse.\n\nuser:\nHello\n\nassistant:\nHi\n\nuser:\nContinue\n\nassistant:\n"
        )
    );
    assert_eq!(captured.body["stop"], serde_json::json!(["\nuser:"]));
}

#[tokio::test]
async fn completion_request_runtime_emits_alignment_warnings_and_merges_provider_options() {
    let transport = JsonResponseTransport::new(serde_json::json!({
        "id": "cmpl_2",
        "model": "compat-model",
        "created": 1718345013u64,
        "choices": [{
            "text": "ok",
            "finish_reason": "stop"
        }]
    }));

    let config = OpenAiCompatibleConfig::new(
        "compat-chat",
        "test-key",
        "https://api.test.com/v1",
        make_text_streaming_adapter(),
    )
    .with_model("compat-model")
    .with_http_transport(Arc::new(transport.clone()));
    let client = OpenAiCompatibleClient::new(config).await.unwrap();

    let request = CompletionRequest::new("hi")
        .with_model("compat-model")
        .with_top_k(20.0)
        .with_tools(vec![Tool::function(
            "lookup",
            "lookup",
            serde_json::json!({ "type": "object" }),
        )])
        .with_tool_choice(ToolChoice::Required)
        .with_response_format(ResponseFormat::json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            }
        })))
        .with_provider_option(
            "openaiCompatible",
            serde_json::json!({
                "echo": true,
                "suffix": " after"
            }),
        )
        .with_provider_option(
            "compat-chat",
            serde_json::json!({
                "user": "provider-user",
                "logitBias": {
                    "42": 1.5
                }
            }),
        );

    let response = crate::traits::CompletionCapability::complete(&client, request)
        .await
        .unwrap();
    let warnings = response.warnings.expect("completion warnings");
    let unsupported_features = warnings
        .into_iter()
        .filter_map(|warning| match warning {
            Warning::Unsupported { feature, .. } => Some(feature),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        unsupported_features,
        vec![
            "topK".to_string(),
            "tools".to_string(),
            "toolChoice".to_string(),
            "responseFormat".to_string()
        ]
    );

    let captured = transport.take().expect("captured completion request");
    assert_eq!(captured.body["echo"], serde_json::json!(true));
    assert_eq!(captured.body["suffix"], serde_json::json!(" after"));
    assert_eq!(captured.body["user"], serde_json::json!("provider-user"));
    assert_eq!(captured.body["logit_bias"]["42"], serde_json::json!(1.5));
    assert!(captured.body.get("tools").is_none());
    assert!(captured.body.get("tool_choice").is_none());
    assert!(captured.body.get("response_format").is_none());
    assert!(captured.body.get("top_k").is_none());
}

#[tokio::test]
async fn completion_stream_request_runtime_routes_to_completions_and_emits_stream_end() {
    let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_3","model":"compat-model","choices":[{"text":"Hel","finish_reason":null}]}

data: {"id":"cmpl_3","model":"compat-model","choices":[{"text":"lo","finish_reason":"stop","logprobs":{"tokens":["lo"],"token_logprobs":[-0.3],"top_logprobs":[{"lo":-0.3}]}}],"sources":[{"url":"https://example.com/stream-source"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

data: [DONE]

"#,
        );

    let config = OpenAiCompatibleConfig::new(
        "compat-chat",
        "test-key",
        "https://api.test.com/v1",
        make_text_streaming_adapter(),
    )
    .with_model("compat-model")
    .with_include_usage(true)
    .with_http_transport(Arc::new(transport.clone()));
    let client = OpenAiCompatibleClient::new(config).await.unwrap();

    let request = CompletionRequest::new("hi").with_model("compat-model");
    let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
        .await
        .unwrap();

    let mut text = String::new();
    let mut end = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            event if event.text_delta().is_some() => {
                text.push_str(event.text_delta().expect("text delta"));
            }
            ChatStreamEvent::StreamEnd { response } => end = Some(response),
            _ => {}
        }
    }

    assert_eq!(text, "Hello");
    let end = end.expect("stream end response");
    assert_eq!(end.id.as_deref(), Some("cmpl_3"));
    assert_eq!(end.model.as_deref(), Some("compat-model"));
    assert_eq!(end.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        end.usage.as_ref().and_then(|usage| usage.total_tokens()),
        Some(3)
    );
    assert_eq!(end.content_text(), Some("Hello"));
    assert_eq!(
        end.provider_metadata
            .as_ref()
            .and_then(|root| root.get("compat-chat"))
            .and_then(|meta| meta.get("logprobs")),
        Some(&serde_json::json!({
            "tokens": ["lo"],
            "token_logprobs": [-0.3],
            "top_logprobs": [{ "lo": -0.3 }]
        }))
    );
    assert_eq!(
        end.provider_metadata
            .as_ref()
            .and_then(|root| root.get("compat-chat"))
            .and_then(|meta| meta.get("sources")),
        Some(&serde_json::json!([{ "url": "https://example.com/stream-source" }]))
    );

    let captured = transport.take_stream().expect("captured completion stream");
    assert_eq!(captured.url, "https://api.test.com/v1/completions");
    assert_eq!(captured.body["stream"], serde_json::json!(true));
    assert_eq!(
        captured.body["stream_options"],
        serde_json::json!({ "include_usage": true })
    );
}

#[tokio::test]
async fn completion_stream_request_runtime_preserves_empty_and_whitespace_text_deltas() {
    let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"A","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"\n","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"","finish_reason":null}]}

data: {"id":"cmpl_lossless","model":"compat-model","choices":[{"text":"B","finish_reason":"stop"}]}

data: [DONE]

"#,
        );

    let config = OpenAiCompatibleConfig::new(
        "compat-chat",
        "test-key",
        "https://api.test.com/v1",
        make_completion_adapter(),
    )
    .with_model("compat-model")
    .with_http_transport(Arc::new(transport.clone()));
    let client = OpenAiCompatibleClient::new(config).await.unwrap();

    let request = CompletionRequest::new("hi").with_model("compat-model");
    let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
        .await
        .unwrap();

    let mut deltas = Vec::new();
    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if let Some(delta) = event.text_delta() {
            deltas.push(delta.to_string());
        }
    }

    assert_eq!(deltas, vec!["A", "\n", "", "B"]);
}

#[tokio::test]
async fn completion_stream_request_runtime_emits_raw_chunks_on_part_lane() {
    let transport = SseResponseTransport::new(
            br#"data: {"id":"cmpl_4","model":"compat-model","created":1718345013,"choices":[{"text":"Hel","finish_reason":null}]}

data: {"id":"cmpl_4","model":"compat-model","created":1718345013,"choices":[{"text":"lo","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}

data: [DONE]

"#,
        );

    let config = OpenAiCompatibleConfig::new(
        "compat-chat",
        "test-key",
        "https://api.test.com/v1",
        make_text_streaming_adapter(),
    )
    .with_model("compat-model")
    .with_include_usage(true)
    .with_http_transport(Arc::new(transport.clone()));
    let client = OpenAiCompatibleClient::new(config).await.unwrap();

    let request = CompletionRequest::new("hi")
        .with_model("compat-model")
        .with_include_raw_chunks(true);
    let mut stream = crate::traits::CompletionCapability::complete_stream(&client, request)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    assert!(matches!(
        events.first(),
        Some(ChatStreamEvent::StreamStart { metadata })
            if metadata.id.as_deref() == Some("cmpl_4")
    ));

    let parts = events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part { part } => Some(part),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(matches!(
        parts.first(),
        Some(ChatStreamPart::StreamStart { .. })
    ));
    assert!(matches!(
        parts.get(1),
        Some(ChatStreamPart::Raw { raw_value })
            if raw_value["id"] == serde_json::json!("cmpl_4")
    ));
    assert!(matches!(
        parts.get(2),
        Some(ChatStreamPart::ResponseMetadata(metadata))
            if metadata.id.as_deref() == Some("cmpl_4")
    ));
    assert!(matches!(
        parts.get(3),
        Some(ChatStreamPart::TextStart { id, .. }) if id == "0"
    ));
    assert!(matches!(
        parts.get(4),
        Some(ChatStreamPart::TextDelta { id, delta, .. })
            if id == "0" && delta == "Hel"
    ));
    assert!(parts.iter().any(|part| {
        matches!(
            part,
            ChatStreamPart::TextEnd { id, .. } if id == "0"
        )
    }));
    assert!(parts.iter().any(|part| {
        matches!(
            part,
            ChatStreamPart::Finish { finish_reason, .. }
                if finish_reason.unified == FinishReason::Stop
        )
    }));
}

#[tokio::test]
async fn completion_response_preserves_raw_logprobs_metadata() {
    let client = OpenAiCompatibleClient::new(
        OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            make_completion_adapter(),
        )
        .with_model("openai/gpt-3.5-turbo-instruct"),
    )
    .await
    .expect("build completion client");

    let mut headers = HeaderMap::new();
    headers.insert("request-id", "req_compat_completion".parse().unwrap());

    let response = client.build_completion_response(
        serde_json::json!({
            "id": "cmpl_compat_1",
            "object": "text_completion",
            "created": 1_718_345_013,
            "model": "openai/gpt-3.5-turbo-instruct",
            "choices": [
                {
                    "text": "hello",
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["hello"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{"hello": -0.2}]
                    }
                }
            ],
            "sources": [{ "url": "https://example.com/source" }]
        }),
        &headers,
        Vec::new(),
    );

    assert_eq!(response.text(), "hello");
    let response_body = response
        .response_metadata
        .as_ref()
        .and_then(|metadata| metadata.body.as_ref())
        .expect("compatible completion response body");
    assert_eq!(
        response
            .response_metadata
            .as_ref()
            .and_then(|metadata| metadata.request_id.as_deref()),
        Some("req_compat_completion")
    );
    assert_eq!(response_body["id"], serde_json::json!("cmpl_compat_1"));
    assert_eq!(
        response_body["choices"][0]["text"],
        serde_json::json!("hello")
    );
    assert_eq!(
        response
            .provider_metadata
            .as_ref()
            .and_then(|root| root.get("openrouter"))
            .and_then(|meta| meta.get("logprobs")),
        Some(&serde_json::json!({
            "tokens": ["hello"],
            "token_logprobs": [-0.2],
            "top_logprobs": [{"hello": -0.2}]
        }))
    );
    assert_eq!(
        response
            .provider_metadata
            .as_ref()
            .and_then(|root| root.get("openrouter"))
            .and_then(|meta| meta.get("sources")),
        Some(&serde_json::json!([{ "url": "https://example.com/source" }]))
    );
}

#[test]
fn completion_logic_stays_out_of_monolithic_client_module() {
    let source = include_str!("../../openai_client.rs");
    for forbidden in [
        "struct CompletionSseConverter",
        "fn build_completion_body(",
        "fn build_completion_response(",
        "impl CompletionCapability for OpenAiCompatibleClient",
    ] {
        assert!(
            !source.contains(forbidden),
            "OpenAI-compatible completion logic should live in openai_client/completion/"
        );
    }
}

#[test]
fn completion_shell_keeps_streaming_converter_split() {
    let source = include_str!("mod.rs")
        .split("#[cfg(test)]")
        .next()
        .unwrap_or_default();

    for marker in ["mod streaming;", "use streaming::CompletionSseConverter;"] {
        assert!(
            source.contains(marker),
            "OpenAI-compatible completion shell should keep `{marker}`"
        );
    }

    for forbidden in [
        "struct CompletionStreamState",
        "struct CompletionSseConverter",
        "impl crate::streaming::SseEventConverter for CompletionSseConverter",
    ] {
        assert!(
            !source.contains(forbidden),
            "OpenAI-compatible completion streaming state should live in completion/streaming.rs"
        );
    }
}
