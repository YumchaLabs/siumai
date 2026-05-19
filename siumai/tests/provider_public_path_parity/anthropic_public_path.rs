use super::*;
use secrecy::ExposeSecret;
use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, FinishReason, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::anthropic::{
    AnthropicChatRequestExt, AnthropicChatResponseExt, AnthropicClient, AnthropicConfig,
    AnthropicContextManagementConfig, AnthropicContextManagementEdit,
    AnthropicContextManagementInputTokensValue, AnthropicEffort, AnthropicOptions,
    AnthropicProviderSettings, AnthropicStructuredOutputMode, ThinkingModeConfig,
};

fn anthropic_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("anthropic", "anthropic")
}

fn anthropic_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("anthropic", "anthropic")
}

fn make_anthropic_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    anthropic_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    anthropic_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "anthropic",
            "ctx-key",
            "https://example.com/anthropic/v1",
            anthropic_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    anthropic_registry_builder()
        .with_provider_api_key_base_url_fetch("anthropic", "test-key", base_url, transport)
        .build()
        .expect("build registry")
}

#[test]
fn anthropic_package_settings_preserve_supported_provider_inputs() {
    let transport = CaptureTransport::default();
    let config = AnthropicProviderSettings::new()
        .with_auth_token("test-token")
        .with_base_url("https://example.com/anthropic")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(transport.clone()))
        .into_config_for_model("claude-sonnet-4-5-20250929")
        .expect("settings into config");

    assert!(config.api_key.expose_secret().is_empty());
    assert_eq!(config.base_url, "https://example.com/anthropic");
    assert_eq!(config.common_params.model, "claude-sonnet-4-5-20250929");
    assert_eq!(
        config
            .http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer test-token")
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
    assert!(transport.take().is_none());
}

fn make_anthropic_request(model: &str, stream: bool) -> ChatRequest {
    let mut request = make_chat_request_with_model(model)
        .with_tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .with_tool_choice(ToolChoice::Required)
        .with_response_format(ResponseFormat::json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
            "additionalProperties": false
        })))
        .with_anthropic_options(
            AnthropicOptions::new()
                .with_thinking_mode(ThinkingModeConfig {
                    enabled: true,
                    thinking_budget: Some(1000),
                })
                .with_tool_streaming(false),
        );
    request.common_params.temperature = Some(0.5);
    request.common_params.top_p = Some(0.7);
    request.common_params.max_tokens = Some(2000);
    request.stream = stream;
    request
}

fn anthropic_json_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    })
}

fn make_anthropic_json_tool_request(model: &str) -> ChatRequest {
    let mut request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(anthropic_json_tool_schema()))
        .with_anthropic_options(
            AnthropicOptions::new()
                .with_structured_output_mode(AnthropicStructuredOutputMode::JsonTool),
        );
    request.stream = true;
    request
}

fn make_anthropic_default_options_request(model: &str) -> ChatRequest {
    let mut request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(anthropic_json_tool_schema()));
    request.common_params.temperature = Some(0.5);
    request.common_params.top_p = Some(0.7);
    request.common_params.max_tokens = Some(2000);
    request.stream = true;
    request
}

fn anthropic_reserved_json_tool_interrupted_stream_body(
    model: &str,
    partial_json: &str,
) -> Vec<u8> {
    let partial_json =
        serde_json::to_string(partial_json).expect("serialize anthropic partial json");
    let mut body = String::new();
    body.push_str(&format!(
            r#"data: {{"type":"message_start","message":{{"id":"msg_test","model":"{model}","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":15,"output_tokens":1}}}}}}"#
        ));
    body.push_str("\n\n");
    body.push_str(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"json","input":{}}}"#,
        );
    body.push_str("\n\n");
    body.push_str(&format!(
            r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{partial_json}}}}}"#
        ));
    body.push_str("\n\n");
    body.into_bytes()
}

fn assert_anthropic_json_tool_stream_request(req: &HttpTransportRequest, base_url: &str) {
    assert_eq!(req.url, format!("{base_url}/v1/messages"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert!(req.body.get("output_format").is_none());
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(req.body["tool_choice"]["type"], serde_json::json!("any"));
    assert_eq!(
        req.body["tool_choice"]["disable_parallel_tool_use"],
        serde_json::json!(true)
    );

    let tools = req.body["tools"].as_array().expect("anthropic tools array");
    assert!(
        tools.iter().any(|tool| {
            tool.get("name").and_then(|v| v.as_str()) == Some("json")
                && tool.get("input_schema") == Some(&anthropic_json_tool_schema())
        }),
        "expected reserved json tool in request body: {:?}",
        req.body["tools"]
    );

    let beta = header_value(req, "anthropic-beta").unwrap_or_default();
    assert!(
        !beta
            .split(',')
            .any(|token| token.trim() == "structured-outputs-2025-11-13"),
        "jsonTool fallback should not enable structured outputs beta: {beta}"
    );
}

fn assert_anthropic_default_options_stream_request(req: &HttpTransportRequest, base_url: &str) {
    assert_anthropic_json_tool_stream_request(req, base_url);
    assert_eq!(
        req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 1000
        })
    );
    assert_eq!(req.body["max_tokens"], serde_json::json!(3000));
    assert!(req.body.get("temperature").is_none());
    assert!(req.body.get("top_p").is_none());
    assert_eq!(
        req.body["context_management"],
        serde_json::json!({
            "edits": [{
                "type": "clear_tool_uses_20250919",
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 1
                },
                "exclude_tools": ["editor"]
            }]
        })
    );
    assert_eq!(
        req.body["output_config"],
        serde_json::json!({
            "effort": "high"
        })
    );

    let beta = header_value(req, "anthropic-beta").unwrap_or_default();
    assert!(
        beta.split(',')
            .any(|token| token.trim() == "context-management-2025-06-27"),
        "missing context-management beta token: {beta}"
    );
    assert!(
        beta.split(',')
            .any(|token| token.trim() == "effort-2025-11-24"),
        "missing effort beta token: {beta}"
    );
    assert!(
        !beta
            .split(',')
            .any(|token| token.trim() == "fine-grained-tool-streaming-2025-05-14"),
        "unexpected fine-grained tool streaming beta token: {beta}"
    );
}

#[tokio::test]
async fn anthropic_siumai_provider_config_chat_request_with_typed_options_are_equivalent() {
    let model = "claude-sonnet-4-5";
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_anthropic_request(model, false);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/messages");
    assert_eq!(
        header_value(&siumai_req, "x-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        header_value(&siumai_req, "anthropic-version"),
        Some("2023-06-01".to_string())
    );
    assert_eq!(
        siumai_req.body["thinking"]["type"],
        serde_json::json!("enabled")
    );
    assert_eq!(
        siumai_req.body["thinking"]["budget_tokens"],
        serde_json::json!(1000)
    );
    assert_eq!(siumai_req.body["max_tokens"], serde_json::json!(3000));
    assert!(siumai_req.body.get("temperature").is_none());
    assert!(siumai_req.body.get("top_p").is_none());
    assert_eq!(
        siumai_req.body["output_config"]["format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(
        siumai_req.body["tool_choice"]["type"],
        serde_json::json!("any")
    );
    assert_eq!(
        siumai_req.body["tools"][0]["name"],
        serde_json::json!("lookup_weather")
    );
    let beta = header_value(&siumai_req, "anthropic-beta").unwrap_or_default();
    assert!(
        beta.split(',')
            .any(|token| token.trim() == "structured-outputs-2025-11-13"),
        "missing structured outputs beta token: {beta}"
    );
}

#[tokio::test]
async fn anthropic_registry_typed_request_options_match_config_path() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_anthropic_request(model, false);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["thinking"]["type"],
        serde_json::json!("enabled")
    );
    assert_eq!(
        registry_req.body["thinking"]["budget_tokens"],
        serde_json::json!(1000)
    );
    assert_eq!(registry_req.body["max_tokens"], serde_json::json!(3000));
    assert!(registry_req.body.get("temperature").is_none());
    assert!(registry_req.body.get("top_p").is_none());
    assert_eq!(
        registry_req.body["output_config"]["format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(
        registry_req.body["tool_choice"]["type"],
        serde_json::json!("any")
    );
    let beta = header_value(&registry_req, "anthropic-beta").unwrap_or_default();
    assert!(
        beta.split(',')
            .any(|token| token.trim() == "structured-outputs-2025-11-13"),
        "missing structured outputs beta token: {beta}"
    );
}

#[tokio::test]
async fn anthropic_registry_handle_prefers_provider_specific_build_overrides() {
    let model = "claude-sonnet-4-5";
    let global_transport = CaptureTransport::default();
    let anthropic_transport = CaptureTransport::default();

    let registry = make_anthropic_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(anthropic_transport.clone()),
    );

    let handle = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build anthropic handle");

    let _ = handle
        .chat_request(make_anthropic_request(model, false))
        .await;

    let req = anthropic_transport
        .take()
        .expect("captured anthropic request");
    assert!(global_transport.take().is_none());
    assert_eq!(header_value(&req, "x-api-key"), Some("ctx-key".to_string()));
    assert_eq!(req.url, "https://example.com/anthropic/v1/messages");
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert!(req.body.get("stream").is_none());
}

#[tokio::test]
async fn anthropic_siumai_provider_config_chat_stream_with_typed_options_are_equivalent() {
    let model = "claude-sonnet-4-5";
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_anthropic_request(model, true);

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
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/messages");
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    let beta = header_value(&siumai_req, "anthropic-beta").unwrap_or_default();
    assert!(
        !beta
            .split(',')
            .any(|token| token.trim() == "fine-grained-tool-streaming-2025-05-14"),
        "unexpected fine-grained tool streaming beta token: {beta}"
    );
}

#[tokio::test]
async fn anthropic_default_options_match_public_stream_request_shape() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_anthropic_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(1000),
        })
        .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                trigger: None,
                keep: None,
                clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(1)),
                clear_tool_inputs: None,
                exclude_tools: Some(vec!["editor".to_string()]),
            },
        ))
        .with_anthropic_tool_streaming(false)
        .with_anthropic_effort(AnthropicEffort::High)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_anthropic_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(1000),
        })
        .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
        .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                trigger: None,
                keep: None,
                clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(1)),
                clear_tool_inputs: None,
                exclude_tools: Some(vec!["editor".to_string()]),
            },
        ))
        .with_anthropic_tool_streaming(false)
        .with_anthropic_effort(AnthropicEffort::High)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_anthropic_thinking_mode(ThinkingModeConfig {
                enabled: true,
                thinking_budget: Some(1000),
            })
            .with_anthropic_structured_output_mode(AnthropicStructuredOutputMode::JsonTool)
            .with_anthropic_context_management(AnthropicContextManagementConfig::new().with_edit(
                AnthropicContextManagementEdit::ClearToolUses20250919 {
                    trigger: None,
                    keep: None,
                    clear_at_least: Some(AnthropicContextManagementInputTokensValue::input_tokens(
                        1,
                    )),
                    clear_tool_inputs: None,
                    exclude_tools: Some(vec!["editor".to_string()]),
                },
            ))
            .with_anthropic_tool_streaming(false)
            .with_anthropic_effort(AnthropicEffort::High)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_anthropic_default_options_request(model);

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
    assert_anthropic_default_options_stream_request(&siumai_req, base_url);
}

#[tokio::test]
async fn anthropic_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let model = "claude-sonnet-4-5";
    let global_transport = CaptureTransport::default();
    let anthropic_transport = CaptureTransport::default();

    let registry = make_anthropic_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(anthropic_transport.clone()),
    );

    let handle = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build anthropic handle");

    use futures_util::StreamExt;
    let mut stream = handle
        .chat_stream_request(make_anthropic_request(model, true))
        .await
        .expect("anthropic stream ok");
    let _ = stream.next().await;

    let req = anthropic_transport
        .take_stream()
        .expect("captured anthropic stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(header_value(&req, "x-api-key"), Some("ctx-key".to_string()));
    assert_eq!(req.url, "https://example.com/anthropic/v1/messages");
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn anthropic_registry_typed_stream_request_options_match_config_path() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_anthropic_request(model, true);

    use futures_util::StreamExt;
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
    let beta = header_value(&registry_req, "anthropic-beta").unwrap_or_default();
    assert!(
        !beta
            .split(',')
            .any(|token| token.trim() == "fine-grained-tool-streaming-2025-05-14"),
        "unexpected fine-grained tool streaming beta token: {beta}"
    );
}

#[tokio::test]
async fn anthropic_structured_output_reserved_json_stream_extracts_across_public_paths() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let stream_body =
        anthropic_reserved_json_tool_interrupted_stream_body(model, r#"{"value":"test"}"#);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_anthropic_json_tool_request(model);

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
    assert_anthropic_json_tool_stream_request(&siumai_req, base_url);
    assert_anthropic_json_tool_stream_request(&provider_req, base_url);
    assert_anthropic_json_tool_stream_request(&config_req, base_url);
    assert_anthropic_json_tool_stream_request(&registry_req, base_url);
}

#[tokio::test]
async fn anthropic_structured_output_reserved_json_stream_fails_consistently_across_public_paths() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let stream_body = anthropic_reserved_json_tool_interrupted_stream_body(model, r#"{"value":"#);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_anthropic_json_tool_request(model);

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
    assert_anthropic_json_tool_stream_request(&siumai_req, base_url);
    assert_anthropic_json_tool_stream_request(&provider_req, base_url);
    assert_anthropic_json_tool_stream_request(&config_req, base_url);
    assert_anthropic_json_tool_stream_request(&registry_req, base_url);
}

#[tokio::test]
async fn anthropic_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": "hello anthropic"
            }
        ],
        "container": {
            "id": "container_test",
            "expires_at": "2025-10-20T12:27:25.107823Z",
            "skills": [
                {
                    "type": "anthropic",
                    "skill_id": "pptx",
                    "version": "20251013"
                }
            ]
        },
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 42,
            "service_tier": "standard",
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0
            }
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
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
        .anthropic_metadata()
        .expect("siumai anthropic metadata");
    let provider_meta = provider_resp
        .anthropic_metadata()
        .expect("provider anthropic metadata");
    let config_meta = config_resp
        .anthropic_metadata()
        .expect("config anthropic metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello anthropic"));
    assert_eq!(provider_resp.content_text(), Some("hello anthropic"));
    assert_eq!(config_resp.content_text(), Some("hello anthropic"));
    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(siumai_resp.service_tier.as_deref(), Some("standard"));
    assert_eq!(provider_resp.service_tier.as_deref(), Some("standard"));
    assert_eq!(config_resp.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        siumai_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        provider_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        siumai_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        provider_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/messages");
}

#[tokio::test]
async fn anthropic_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let stream_body = concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_test","model":"claude-sonnet-4-5","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1,"service_tier":"standard"}}}

"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello anthropic"}}

"#,
            r#"data: {"type":"content_block_stop","index":0}

"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"container":{"id":"container_test","expires_at":"2025-10-20T12:27:25.107823Z","skills":[{"type":"anthropic","skill_id":"pptx","version":"20251013"}]}},"usage":{"input_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":42,"service_tier":"standard","server_tool_use":{"web_search_requests":0,"web_fetch_requests":0}}}

"#,
            r#"data: {"type":"message_stop"}

"#
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
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

    use futures_util::StreamExt;

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
        .anthropic_metadata()
        .expect("siumai anthropic metadata");
    let provider_meta = provider_resp
        .anthropic_metadata()
        .expect("provider anthropic metadata");
    let config_meta = config_resp
        .anthropic_metadata()
        .expect("config anthropic metadata");

    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        siumai_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        provider_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        siumai_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        provider_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
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
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/messages");
}

#[tokio::test]
async fn anthropic_reasoning_response_is_equivalent_across_public_paths() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
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
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_anthropic_options(
        AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(2048),
        }),
    );

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
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));

        let meta = response
            .anthropic_metadata()
            .expect("expected anthropic metadata");
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            meta.redacted_thinking_data.as_deref(),
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
    assert_eq!(
        siumai_req.body["thinking"]["type"],
        serde_json::json!("enabled")
    );
    assert_eq!(
        siumai_req.body["thinking"]["budget_tokens"],
        serde_json::json!(2048)
    );
}

#[tokio::test]
async fn anthropic_reasoning_stream_is_equivalent_across_public_paths() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let stream_body = concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_reasoning","model":"claude-sonnet-4-5","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":18,"output_tokens":1}}}

"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Count the letters carefully. "}}

"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The word strawberry contains three r characters."}}

"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}

"#,
            r#"data: {"type":"content_block_stop","index":0}

"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"redacted-blob"}}

"#,
            r#"data: {"type":"content_block_stop","index":1}

"#,
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}

"#,
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"There are three letter r's in strawberry."}}

"#,
            r#"data: {"type":"content_block_stop","index":2}

"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":18,"output_tokens":24}}

"#,
            r#"data: {"type":"message_stop"}

"#
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_anthropic_options(
        AnthropicOptions::new().with_thinking_mode(ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(2048),
        }),
    );

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
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));

        let meta = response
            .anthropic_metadata()
            .expect("expected anthropic metadata");
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(
            meta.redacted_thinking_data.as_deref(),
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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["thinking"]["type"],
        serde_json::json!("enabled")
    );
    assert_eq!(
        siumai_req.body["thinking"]["budget_tokens"],
        serde_json::json!(2048)
    );
}

#[tokio::test]
async fn anthropic_registry_chat_response_metadata_match_config_path() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": "hello anthropic"
            }
        ],
        "container": {
            "id": "container_test",
            "expires_at": "2025-10-20T12:27:25.107823Z",
            "skills": [
                {
                    "type": "anthropic",
                    "skill_id": "pptx",
                    "version": "20251013"
                }
            ]
        },
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 42,
            "service_tier": "standard",
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0
            }
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
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
        .anthropic_metadata()
        .expect("config anthropic metadata");
    let registry_meta = registry_resp
        .anthropic_metadata()
        .expect("registry anthropic metadata");

    assert_eq!(config_resp.content_text(), Some("hello anthropic"));
    assert_eq!(registry_resp.content_text(), Some("hello anthropic"));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.service_tier.as_deref(), Some("standard"));
    assert_eq!(registry_resp.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        registry_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        registry_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "https://example.com/custom/v1/messages");
}

#[tokio::test]
async fn anthropic_registry_stream_end_metadata_match_config_path() {
    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";
    let stream_body = concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_test","model":"claude-sonnet-4-5","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1,"service_tier":"standard"}}}

"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello anthropic"}}

"#,
            r#"data: {"type":"content_block_stop","index":0}

"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"container":{"id":"container_test","expires_at":"2025-10-20T12:27:25.107823Z","skills":[{"type":"anthropic","skill_id":"pptx","version":"20251013"}]}},"usage":{"input_tokens":15,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":42,"service_tier":"standard","server_tool_use":{"web_search_requests":0,"web_fetch_requests":0}}}

"#,
            r#"data: {"type":"message_stop"}

"#
        )
        .as_bytes()
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("anthropic:claude-sonnet-4-5")
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
        .anthropic_metadata()
        .expect("config anthropic metadata");
    let registry_meta = registry_resp
        .anthropic_metadata()
        .expect("registry anthropic metadata");

    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        registry_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );
    assert_eq!(
        registry_meta
            .container
            .as_ref()
            .and_then(|container| container.skills.as_ref())
            .and_then(|skills| skills.first())
            .and_then(|skill| skill.skill_id.as_deref()),
        Some("pptx")
    );

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "https://example.com/custom/v1/messages");
}

#[tokio::test]
async fn anthropic_siumai_provider_config_non_text_requests_are_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let embedding_request = EmbeddingRequest::single("hello anthropic embedding").with_model(model);
    let rerank_request = make_rerank_request_with_model(model).with_top_n(1);
    let image_request = make_image_request_with_model(model);

    let siumai_embedding_err = siumai_client
        .embed_with_config(embedding_request.clone())
        .await
        .expect_err("anthropic embedding should be unsupported");
    let siumai_rerank_err = siumai_client
        .rerank(rerank_request.clone())
        .await
        .expect_err("anthropic rerank should be unsupported");
    let siumai_image_err = siumai_client
        .generate_images(image_request.clone())
        .await
        .expect_err("anthropic image generation should be unsupported");

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
async fn anthropic_registry_non_text_requests_are_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let embedding_err = match registry.embedding_model("anthropic:claude-sonnet-4-5") {
        Ok(_) => panic!("build registry embedding model should be unsupported"),
        Err(err) => err,
    };
    let image_err = match registry.image_model("anthropic:claude-sonnet-4-5") {
        Ok(_) => panic!("build registry image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("anthropic:claude-sonnet-4-5") {
        Ok(_) => panic!("build anthropic registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&embedding_err);
    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn anthropic_siumai_provider_config_audio_family_requests_are_intentionally_unsupported() {
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let config_transport = MixedCaptureTransport::default();

    let model = "claude-sonnet-4-5";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::anthropic()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let tts_request = TtsRequest::new("hello anthropic audio".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());
    let stt_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let siumai_tts_err = siumai_client
        .text_to_speech(tts_request.clone())
        .await
        .expect_err("anthropic text-to-speech should be unsupported");

    let siumai_stt_err = siumai_client
        .speech_to_text(stt_request.clone())
        .await
        .expect_err("anthropic speech-to-text should be unsupported");

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
async fn anthropic_registry_audio_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let base_url = "https://example.com/custom";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let tts_err = match registry.speech_model("anthropic:claude-sonnet-4-5") {
        Ok(_) => panic!("build anthropic registry speech model should be unsupported"),
        Err(err) => err,
    };
    let stt_err = match registry.transcription_model("anthropic:claude-sonnet-4-5") {
        Ok(_) => panic!("build anthropic registry transcription model should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&tts_err);
    assert_unsupported_operation(&stt_err);
    assert_mixed_capture_transports_unused(&[&registry_transport]);
}
