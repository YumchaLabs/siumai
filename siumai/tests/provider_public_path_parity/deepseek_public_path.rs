use super::*;
use futures_util::StreamExt;
use siumai::experimental::execution::middleware::language_model::LanguageModelMiddleware;
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::deepseek::{
    DeepSeekChatRequestExt, DeepSeekChatResponseExt, DeepSeekOptions, DeepSeekProviderSettings,
};
use siumai_registry::registry::builder::RegistryBuilder;

fn deepseek_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("deepseek", "deepseek")
}

fn deepseek_registry_builder() -> RegistryBuilder {
    RegistryBuilder::new(deepseek_registry_providers())
}

#[test]
fn deepseek_package_settings_preserve_supported_provider_inputs() {
    let config = DeepSeekProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepseek")
        .with_header("x-test", "1")
        .into_config_for_model("deepseek-chat")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.com/deepseek");
    assert_eq!(config.common_params.model, "deepseek-chat");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

fn make_registry_with_global_reasoning_defaults(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    deepseek_registry_builder()
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .with_provider_api_key_base_url_fetch("deepseek", "test-key", base_url, transport)
        .build()
        .expect("build deepseek registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    deepseek_registry_builder()
        .with_provider_api_key_base_url_fetch("deepseek", "test-key", base_url, transport)
        .build()
        .expect("build deepseek registry")
}

fn make_registry_builder_with_global_reasoning_defaults(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    deepseek_registry_builder()
        .with_api_key("test-key")
        .with_base_url(base_url)
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .fetch(transport)
        .build()
        .expect("build registry")
}

fn assert_deepseek_default_options_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_eq!(req.url, format!("{base_url}/chat/completions"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
    assert_deepseek_thinking_enabled(&req.body);
    assert_eq!(req.body["foo"], serde_json::json!("bar"));
}

fn assert_deepseek_thinking_enabled(body: &serde_json::Value) {
    assert_eq!(
        body["thinking"],
        serde_json::json!({
            "type": "enabled"
        })
    );
    assert!(body.get("enableReasoning").is_none());
    assert!(body.get("enable_reasoning").is_none());
    assert!(body.get("reasoningBudget").is_none());
    assert!(body.get("reasoning_budget").is_none());
}

fn assert_deepseek_legacy_reasoning_budget_removed(body: &serde_json::Value) {
    assert!(body.get("thinking").is_none());
    assert!(body.get("enableReasoning").is_none());
    assert!(body.get("enable_reasoning").is_none());
    assert!(body.get("reasoningBudget").is_none());
    assert!(body.get("reasoning_budget").is_none());
}

fn assert_deepseek_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_deepseek_default_options_request(req, base_url, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        req.body["stream_options"],
        serde_json::json!({ "include_usage": true })
    );
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
}

struct DeepSeekFooOverrideMiddleware;

impl LanguageModelMiddleware for DeepSeekFooOverrideMiddleware {
    fn transform_params(&self, req: ChatRequest) -> ChatRequest {
        req.with_deepseek_options(
            DeepSeekOptions::new().with_param("foo", serde_json::json!("middleware")),
        )
    }
}

fn make_deepseek_tool_call_request(model: &str) -> ChatRequest {
    ChatRequest::builder()
        .model(model)
        .messages(vec![
            ChatMessage::user("What's the weather in Tokyo?").build(),
        ])
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
        .build()
}

fn assert_deepseek_weather_tool_call_response(response: &siumai::prelude::unified::ChatResponse) {
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::ToolCalls)
    );
    assert_eq!(response.tool_calls().len(), 1);

    let call = response.tool_calls()[0].as_tool_call().expect("tool call");
    assert_eq!(call.tool_call_id, "call_1");
    assert_eq!(call.tool_name, "get_weather");
    assert_eq!(call.arguments, &serde_json::json!({ "city": "Tokyo" }));
}

async fn collect_deepseek_streamed_tool_call(
    stream: &mut siumai::prelude::unified::ChatStream,
) -> (
    siumai::prelude::unified::ChatResponse,
    String,
    String,
    serde_json::Value,
) {
    let mut tool_call_id = None;
    let mut tool_name = None;
    let mut arguments = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                record_streamed_tool_part(
                    &event,
                    &mut tool_call_id,
                    &mut tool_name,
                    &mut arguments,
                );
                if let siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } = event {
                    return (
                        response,
                        tool_call_id.expect("stream tool call id"),
                        tool_name.expect("stream tool name"),
                        serde_json::from_str(&arguments).expect("stream tool arguments json"),
                    );
                }
            }
            Err(err) => panic!("stream error: {err}"),
        }
    }

    panic!("stream end event");
}

#[tokio::test]
async fn deepseek_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-chat")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-chat")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
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
    assert_deepseek_thinking_enabled(&siumai_req.body);
}

#[tokio::test]
async fn deepseek_registry_chat_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
        }),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        registry_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
    assert_deepseek_thinking_enabled(&registry_req.body);
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "deepseek-chat";
    let request_model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model).with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
        }),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_deepseek_thinking_enabled(&registry_req.body);
}

#[tokio::test]
async fn deepseek_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(4096)
            .with_param("foo", serde_json::json!("bar")),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_deepseek_thinking_enabled(&siumai_req.body);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_default_options_match_public_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
        .with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
        .with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
            .with_deepseek_options(
                DeepSeekOptions::new()
                    .with_reasoning_budget(4096)
                    .with_param("foo", serde_json::json!("bar")),
            )
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
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
    assert_deepseek_default_options_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn deepseek_default_options_match_public_stream_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
        .with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
        .with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_deepseek_options(DeepSeekOptions::new().with_reasoning(false))
            .with_deepseek_options(
                DeepSeekOptions::new()
                    .with_reasoning_budget(4096)
                    .with_param("foo", serde_json::json!("bar")),
            )
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
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
    assert_deepseek_default_options_stream_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn deepseek_shared_builder_later_default_options_override_earlier_defaults() {
    let transport = CaptureTransport::default();
    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_reasoning(false)
        .with_deepseek_reasoning_budget(4096)
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let _ = client
        .chat_request(make_chat_request_with_model(model))
        .await;

    let captured = transport.take().expect("captured request");
    assert_deepseek_thinking_enabled(&captured.body);
}

#[tokio::test]
async fn deepseek_shared_builder_custom_middleware_overrides_default_options() {
    let transport = CaptureTransport::default();
    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_deepseek_options(
            DeepSeekOptions::new().with_param("foo", serde_json::json!("builder")),
        )
        .add_model_middleware(Arc::new(DeepSeekFooOverrideMiddleware))
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let _ = client
        .chat_request(make_chat_request_with_model(model))
        .await;

    let captured = transport.take().expect("captured request");
    assert_eq!(captured.body["foo"], serde_json::json!("middleware"));
}

#[tokio::test]
async fn deepseek_registry_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(4096)
            .with_param("foo", serde_json::json!("bar")),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_deepseek_thinking_enabled(&registry_req.body);
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_siumai_provider_config_tool_choice_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .model(model)
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build()
        .with_provider_option(
            "deepseek",
            serde_json::json!({
                "tool_choice": "auto",
                "reasoningBudget": 4096
            }),
        );

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
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_deepseek_legacy_reasoning_budget_removed(&siumai_req.body);
    assert_eq!(
        siumai_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn deepseek_registry_tool_choice_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .model(model)
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build()
        .with_provider_option(
            "deepseek",
            serde_json::json!({
                "tool_choice": "auto",
                "reasoningBudget": 4096
            }),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["tool_choice"], serde_json::json!("none"));
    assert_deepseek_legacy_reasoning_budget_removed(&registry_req.body);
    assert_eq!(
        registry_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn deepseek_structured_output_stream_is_equivalent_across_public_paths() {
    let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"{\"answer\":\"he","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"llo\"}","role":null},"finish_reason":"stop"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
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
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_provider_option(
            "deepseek",
            serde_json::json!({
                "response_format": { "type": "json_object" },
                "reasoningBudget": 2048
            }),
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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_deepseek_legacy_reasoning_budget_removed(&siumai_req.body);
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({ "type": "json_object" })
    );
}

#[tokio::test]
async fn deepseek_structured_output_synthetic_unknown_stream_end_fails_consistently_across_public_paths()
 {
    let stream_body = "data: {\"id\":\"1\",\"model\":\"deepseek-chat\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\" ,\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
            .as_bytes()
            .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
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
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_provider_option(
            "deepseek",
            serde_json::json!({
                "response_format": { "type": "json_object" },
                "reasoningBudget": 2048
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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_deepseek_legacy_reasoning_budget_removed(&siumai_req.body);
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({ "type": "json_object" })
    );
}

#[tokio::test]
async fn deepseek_tool_calls_stream_and_non_stream_are_equivalent_across_public_paths() {
    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let non_stream_response = serde_json::json!({
        "id": "chatcmpl-deepseek-tool-call",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\""}}]},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo\"}"}}]},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let provider_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let config_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let registry_non_stream_transport = JsonSuccessTransport::new(non_stream_response);

    let siumai_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let config_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_stream_transport = SseSuccessTransport::new(stream_body);

    let siumai_non_stream_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_non_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai non-stream client");

    let provider_non_stream_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_non_stream_transport.clone()))
        .build()
        .await
        .expect("build provider non-stream client");

    let config_non_stream_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_non_stream_transport.clone())),
    )
    .await
    .expect("build config non-stream client");

    let registry_non_stream =
        make_registry(Arc::new(registry_non_stream_transport.clone()), base_url);
    let registry_non_stream_model = registry_non_stream
        .language_model("deepseek:deepseek-chat")
        .expect("build registry non-stream language model");

    let siumai_stream_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai stream client");

    let provider_stream_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_stream_transport.clone()))
        .build()
        .await
        .expect("build provider stream client");

    let config_stream_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_stream_transport.clone())),
    )
    .await
    .expect("build config stream client");

    let registry_stream = make_registry(Arc::new(registry_stream_transport.clone()), base_url);
    let registry_stream_model = registry_stream
        .language_model("deepseek:deepseek-chat")
        .expect("build registry stream language model");

    let request = make_deepseek_tool_call_request(model);

    let siumai_non_stream_resp = siumai_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("siumai non-stream response");
    let provider_non_stream_resp = provider_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("provider non-stream response");
    let config_non_stream_resp = config_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("config non-stream response");
    let registry_non_stream_resp = registry_non_stream_model
        .chat_request(request.clone())
        .await
        .expect("registry non-stream response");

    let mut siumai_stream = siumai_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream");
    let mut provider_stream = provider_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream");
    let mut config_stream = config_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream");
    let mut registry_stream_handle = registry_stream_model
        .chat_stream_request(request)
        .await
        .expect("registry stream");

    let (
        siumai_stream_resp,
        siumai_stream_tool_call_id,
        siumai_stream_tool_name,
        siumai_stream_arguments,
    ) = collect_deepseek_streamed_tool_call(&mut siumai_stream).await;
    let (
        provider_stream_resp,
        provider_stream_tool_call_id,
        provider_stream_tool_name,
        provider_stream_arguments,
    ) = collect_deepseek_streamed_tool_call(&mut provider_stream).await;
    let (
        config_stream_resp,
        config_stream_tool_call_id,
        config_stream_tool_name,
        config_stream_arguments,
    ) = collect_deepseek_streamed_tool_call(&mut config_stream).await;
    let (
        registry_stream_resp,
        registry_stream_tool_call_id,
        registry_stream_tool_name,
        registry_stream_arguments,
    ) = collect_deepseek_streamed_tool_call(&mut registry_stream_handle).await;

    for response in [
        &siumai_non_stream_resp,
        &provider_non_stream_resp,
        &config_non_stream_resp,
        &registry_non_stream_resp,
    ] {
        assert_deepseek_weather_tool_call_response(response);
    }

    for response in [
        &siumai_stream_resp,
        &provider_stream_resp,
        &config_stream_resp,
        &registry_stream_resp,
    ] {
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::ToolCalls)
        );
        assert!(response.tool_calls().is_empty());
    }

    let expected_call = siumai_non_stream_resp.tool_calls()[0]
        .as_tool_call()
        .expect("baseline tool call");
    for response in [
        &provider_non_stream_resp,
        &config_non_stream_resp,
        &registry_non_stream_resp,
    ] {
        let call = response.tool_calls()[0]
            .as_tool_call()
            .expect("matching tool call");
        assert_eq!(call.tool_call_id, expected_call.tool_call_id);
        assert_eq!(call.tool_name, expected_call.tool_name);
        assert_eq!(call.arguments, expected_call.arguments);
    }

    for (tool_call_id, tool_name, arguments) in [
        (
            &siumai_stream_tool_call_id,
            &siumai_stream_tool_name,
            &siumai_stream_arguments,
        ),
        (
            &provider_stream_tool_call_id,
            &provider_stream_tool_name,
            &provider_stream_arguments,
        ),
        (
            &config_stream_tool_call_id,
            &config_stream_tool_name,
            &config_stream_arguments,
        ),
        (
            &registry_stream_tool_call_id,
            &registry_stream_tool_name,
            &registry_stream_arguments,
        ),
    ] {
        assert_eq!(tool_call_id, expected_call.tool_call_id);
        assert_eq!(tool_name, expected_call.tool_name);
        assert_eq!(arguments, expected_call.arguments);
    }

    let siumai_non_stream_req = siumai_non_stream_transport
        .take()
        .expect("siumai non-stream request");
    let provider_non_stream_req = provider_non_stream_transport
        .take()
        .expect("provider non-stream request");
    let config_non_stream_req = config_non_stream_transport
        .take()
        .expect("config non-stream request");
    let registry_non_stream_req = registry_non_stream_transport
        .take()
        .expect("registry non-stream request");

    assert_requests_equivalent(&siumai_non_stream_req, &provider_non_stream_req);
    assert_requests_equivalent(&siumai_non_stream_req, &config_non_stream_req);
    assert_requests_equivalent(&siumai_non_stream_req, &registry_non_stream_req);
    assert_eq!(
        siumai_non_stream_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );

    let siumai_stream_req = siumai_stream_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_stream_req = provider_stream_transport
        .take_stream()
        .expect("provider stream request");
    let config_stream_req = config_stream_transport
        .take_stream()
        .expect("config stream request");
    let registry_stream_req = registry_stream_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_stream_req, &provider_stream_req);
    assert_requests_equivalent(&siumai_stream_req, &config_stream_req);
    assert_requests_equivalent(&siumai_stream_req, &registry_stream_req);
    assert_eq!(
        header_value(&siumai_stream_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_stream_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_stream_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn deepseek_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-chat")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-chat")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
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
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_registry_chat_stream_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
        }),
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
    assert_eq!(
        header_value(&registry_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        registry_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_deepseek_thinking_enabled(&registry_req.body);
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "deepseek-chat";
    let request_model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model).with_provider_option(
        "deepseek",
        serde_json::json!({
            "enableReasoning": true,
            "reasoningBudget": 4096,
            "foo": "bar"
        }),
    );

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
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_registry_chat_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let deepseek_transport = CaptureTransport::default();

    let registry = deepseek_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "deepseek",
            "ctx-key",
            "https://example.com/deepseek/v1",
            Arc::new(deepseek_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let _ = handle
        .chat_request(
            make_chat_request_with_model("deepseek-chat").with_provider_option(
                "deepseek",
                serde_json::json!({
                    "enableReasoning": true,
                    "reasoningBudget": 4096,
                    "foo": "bar"
                }),
            ),
        )
        .await;

    let req = deepseek_transport
        .take()
        .expect("captured deepseek request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("deepseek-chat"));
    assert_deepseek_thinking_enabled(&req.body);
    assert_eq!(req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_registry_chat_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let deepseek_transport = CaptureTransport::default();

    let registry = deepseek_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "deepseek",
            "ctx-key",
            "https://example.com/deepseek/v1",
            Arc::new(deepseek_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let _ = handle
        .chat_stream_request(
            make_chat_request_with_model("deepseek-chat").with_deepseek_options(
                DeepSeekOptions::new()
                    .with_reasoning_budget(4096)
                    .with_param("foo", serde_json::json!("bar")),
            ),
        )
        .await;

    let req = deepseek_transport
        .take_stream()
        .expect("captured deepseek stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("deepseek-chat"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_deepseek_thinking_enabled(&req.body);
    assert_eq!(req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_siumai_provider_config_stable_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(4096)
            .with_param("foo", serde_json::json!("bar")),
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
    assert_deepseek_thinking_enabled(&siumai_req.body);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_registry_stable_stream_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(4096)
            .with_param("foo", serde_json::json!("bar")),
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
    assert_deepseek_thinking_enabled(&registry_req.body);
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_siumai_provider_config_reasoning_defaults_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning(true)
            .with_reasoning_budget(2048)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
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
    assert_deepseek_thinking_enabled(&siumai_req.body);
}

#[tokio::test]
async fn deepseek_registry_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning(true)
            .with_reasoning_budget(1024)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        Arc::new(registry_transport.clone()),
        base_url,
        true,
        1024,
    );
    let registry_model = registry
        .language_model("deepseek:deepseek-reasoner")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_deepseek_thinking_enabled(&registry_req.body);
}

#[tokio::test]
async fn deepseek_registry_builder_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning(true)
            .with_reasoning_budget(1024)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry_builder_with_global_reasoning_defaults(
        Arc::new(registry_transport.clone()),
        base_url,
        true,
        1024,
    );
    let registry_model = registry
        .language_model("deepseek:deepseek-reasoner")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_deepseek_thinking_enabled(&registry_req.body);
}

#[tokio::test]
async fn deepseek_reasoning_response_is_equivalent_across_public_paths() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-deepseek-reasoning",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "There are three letter r's in strawberry.",
                    "reasoning_content": "Count the letters in strawberry carefully. There are three r characters."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 24,
            "total_tokens": 42,
            "completion_tokens_details": {
                "reasoning_tokens": 12
            }
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-reasoner")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(1536)
            .with_param("foo", serde_json::json!("bar")),
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
        "Count the letters in strawberry carefully. There are three r characters.".to_string();

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(response.reasoning(), vec![expected_reasoning.clone()]);
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens_details.as_ref())
                .and_then(|details| details.reasoning_tokens),
            Some(12)
        );
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_deepseek_thinking_enabled(&siumai_req.body);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_reasoning_stream_is_equivalent_across_public_paths() {
    let stream_body = br#"data: {"id":"1","model":"deepseek-reasoner","created":1718345013,"choices":[{"index":0,"delta":{"reasoning_content":"Count the letters in strawberry carefully. ","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-reasoner","created":1718345013,"choices":[{"index":0,"delta":{"reasoning_content":"There are three r characters.","content":"There are three letter r's in strawberry.","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":24,"total_tokens":42,"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "deepseek-reasoner";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-reasoner")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_deepseek_options(
        DeepSeekOptions::new()
            .with_reasoning_budget(1536)
            .with_param("foo", serde_json::json!("bar")),
    );

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
        "Count the letters in strawberry carefully. There are three r characters.".to_string();

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
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens_details.as_ref())
                .and_then(|details| details.reasoning_tokens),
            Some(12)
        );
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
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
    assert_deepseek_thinking_enabled(&siumai_req.body);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn deepseek_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-deepseek-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from deepseek"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "reasoning_tokens": 2
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
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
        .deepseek_metadata()
        .expect("siumai deepseek metadata");
    let provider_meta = provider_resp
        .deepseek_metadata()
        .expect("provider deepseek metadata");
    let config_meta = config_resp
        .deepseek_metadata()
        .expect("config deepseek metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(provider_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(config_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));
    assert_eq!(siumai_resp.id.as_deref(), Some("chatcmpl-deepseek-test"));
    assert_eq!(provider_resp.id.as_deref(), Some("chatcmpl-deepseek-test"));
    assert_eq!(config_resp.id.as_deref(), Some("chatcmpl-deepseek-test"));
    assert_eq!(siumai_resp.model.as_deref(), Some(model));
    assert_eq!(provider_resp.model.as_deref(), Some(model));
    assert_eq!(config_resp.model.as_deref(), Some(model));
    assert!(siumai_meta.sources.is_none());
    assert!(provider_meta.sources.is_none());
    assert!(config_meta.sources.is_none());
    assert!(siumai_meta.extra.is_empty());
    assert!(provider_meta.extra.is_empty());
    assert!(config_meta.extra.is_empty());

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
}

#[tokio::test]
async fn deepseek_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":" from deepseek","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"reasoning_tokens":2}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let events = async |stream: &mut siumai::prelude::unified::ChatStream| {
        let mut end = None;
        while let Some(event) = stream.next().await {
            if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
                end = Some(response);
                break;
            }
        }
        end
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
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let siumai_resp = events(&mut siumai_stream).await.expect("siumai stream end");
    let provider_resp = events(&mut provider_stream)
        .await
        .expect("provider stream end");
    let config_resp = events(&mut config_stream).await.expect("config stream end");

    let siumai_meta = siumai_resp
        .deepseek_metadata()
        .expect("siumai deepseek metadata");
    let provider_meta = provider_resp
        .deepseek_metadata()
        .expect("provider deepseek metadata");
    let config_meta = config_resp
        .deepseek_metadata()
        .expect("config deepseek metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(provider_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(config_resp.content_text(), Some("hello from deepseek"));
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
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));
    assert_eq!(siumai_resp.id.as_deref(), Some("1"));
    assert_eq!(provider_resp.id.as_deref(), Some("1"));
    assert_eq!(config_resp.id.as_deref(), Some("1"));
    assert_eq!(siumai_resp.model.as_deref(), Some(model));
    assert_eq!(provider_resp.model.as_deref(), Some(model));
    assert_eq!(config_resp.model.as_deref(), Some(model));
    assert!(siumai_meta.sources.is_none());
    assert!(provider_meta.sources.is_none());
    assert!(config_meta.sources.is_none());
    assert!(siumai_meta.extra.is_empty());
    assert!(provider_meta.extra.is_empty());
    assert!(config_meta.extra.is_empty());

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
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_registry_chat_response_metadata_match_config_path() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-deepseek-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from deepseek"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "reasoning_tokens": 2
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
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
        .deepseek_metadata()
        .expect("config deepseek metadata");
    let registry_meta = registry_resp
        .deepseek_metadata()
        .expect("registry deepseek metadata");

    assert_eq!(config_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(registry_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs));
    assert_eq!(config_resp.id.as_deref(), Some("chatcmpl-deepseek-test"));
    assert_eq!(registry_resp.id.as_deref(), Some("chatcmpl-deepseek-test"));
    assert_eq!(config_resp.model.as_deref(), Some(model));
    assert_eq!(registry_resp.model.as_deref(), Some(model));
    assert!(config_meta.sources.is_none());
    assert!(registry_meta.sources.is_none());
    assert!(config_meta.extra.is_empty());
    assert!(registry_meta.extra.is_empty());

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        config_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
}

#[tokio::test]
async fn deepseek_registry_stream_end_metadata_match_config_path() {
    let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":" from deepseek","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"reasoning_tokens":2}}

data: [DONE]

"#
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "deepseek-chat";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let events = async |stream: &mut siumai::prelude::unified::ChatStream| {
        let mut end = None;
        while let Some(event) = stream.next().await {
            if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
                end = Some(response);
                break;
            }
        }
        end
    };

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let config_resp = events(&mut config_stream).await.expect("config stream end");
    let registry_resp = events(&mut registry_stream)
        .await
        .expect("registry stream end");

    let config_meta = config_resp
        .deepseek_metadata()
        .expect("config deepseek metadata");
    let registry_meta = registry_resp
        .deepseek_metadata()
        .expect("registry deepseek metadata");

    assert_eq!(config_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(registry_resp.content_text(), Some("hello from deepseek"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs));
    assert_eq!(config_resp.id.as_deref(), Some("1"));
    assert_eq!(registry_resp.id.as_deref(), Some("1"));
    assert_eq!(config_resp.model.as_deref(), Some(model));
    assert_eq!(registry_resp.model.as_deref(), Some(model));
    assert!(config_meta.sources.is_none());
    assert!(registry_meta.sources.is_none());
    assert!(config_meta.extra.is_empty());
    assert!(registry_meta.extra.is_empty());

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        header_value(&config_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn deepseek_registry_override_chat_response_metadata_preserves_vendor_namespace() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-deepseek-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from deepseek"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "reasoning_tokens": 2
        }
    });

    let global_transport = CaptureTransport::default();
    let deepseek_transport = JsonSuccessTransport::new(response_json);

    let registry = deepseek_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1/")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "deepseek",
            "ctx-key",
            "https://example.com/deepseek/v1/",
            Arc::new(deepseek_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let response = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build deepseek handle")
        .chat_request(make_chat_request_with_model("deepseek-chat"))
        .await
        .expect("registry response ok");

    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("deepseek").is_some());

    let metadata = response
        .deepseek_metadata()
        .expect("registry deepseek metadata");
    assert_eq!(response.content_text(), Some("hello from deepseek"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(metadata.logprobs, Some(expected_logprobs));

    let req = deepseek_transport
        .take()
        .expect("captured deepseek request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
}

#[tokio::test]
async fn deepseek_registry_override_stream_end_metadata_preserves_vendor_namespace() {
    let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":" from deepseek","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"reasoning_tokens":2}}

data: [DONE]

"#
        .to_vec();

    let global_transport = CaptureTransport::default();
    let deepseek_transport = SseSuccessTransport::new(stream_body);

    let registry = deepseek_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1/")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "deepseek",
            "ctx-key",
            "https://example.com/deepseek/v1/",
            Arc::new(deepseek_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let mut stream = registry
        .language_model("deepseek:deepseek-chat")
        .expect("build deepseek handle")
        .chat_stream_request(make_chat_request_with_model("deepseek-chat"))
        .await
        .expect("registry stream ok");

    let mut stream_end = None;
    while let Some(event) = stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            stream_end = Some(response);
            break;
        }
    }

    let response = stream_end.expect("registry stream end");
    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("deepseek").is_some());

    let metadata = response
        .deepseek_metadata()
        .expect("registry deepseek metadata");
    assert_eq!(response.content_text(), Some("hello from deepseek"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(metadata.logprobs, Some(expected_logprobs));

    let req = deepseek_transport
        .take_stream()
        .expect("captured deepseek stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
}

#[tokio::test]
async fn deepseek_siumai_provider_config_embedding_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-embedding-test")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("deepseek-embedding-test")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("deepseek-embedding-test")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request =
        EmbeddingRequest::single("hello deepseek embedding").with_model("deepseek-embedding-test");

    let siumai_err = siumai_client
        .embed_with_config(request.clone())
        .await
        .expect_err("deepseek embedding should be unsupported");
    let provider_err = provider_client
        .embed_with_config(request.clone())
        .await
        .expect_err("deepseek embedding should be unsupported");
    let config_err = config_client
        .embed_with_config(request)
        .await
        .expect_err("deepseek embedding should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(provider_client.as_embedding_capability().is_none());
    assert!(config_client.as_embedding_capability().is_none());
    assert!(provider_client.as_image_generation_capability().is_none());
    assert!(config_client.as_image_generation_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn deepseek_siumai_provider_config_rerank_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-rerank-test";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let siumai_err = siumai_client
        .rerank(request.clone())
        .await
        .expect_err("deepseek rerank should be unsupported");
    let provider_err = provider_client
        .rerank(request.clone())
        .await
        .expect_err("deepseek rerank should be unsupported");
    let config_err = config_client
        .rerank(request)
        .await
        .expect_err("deepseek rerank should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(provider_client.as_embedding_capability().is_none());
    assert!(config_client.as_embedding_capability().is_none());
    assert!(provider_client.as_image_generation_capability().is_none());
    assert!(config_client.as_image_generation_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn deepseek_siumai_provider_config_image_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deepseek-image-test";

    let siumai_client = Siumai::builder()
        .deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepseek()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::deepseek::DeepSeekClient::from_config(
        siumai::provider_ext::deepseek::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_image_request_with_model(model);

    let siumai_err = siumai_client
        .generate_images(request.clone())
        .await
        .expect_err("deepseek image generation should be unsupported");
    let provider_err = provider_client
        .generate_images(request.clone())
        .await
        .expect_err("deepseek image generation should be unsupported");
    let config_err = config_client
        .generate_images(request)
        .await
        .expect_err("deepseek image generation should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(provider_client.as_embedding_capability().is_none());
    assert!(config_client.as_embedding_capability().is_none());
    assert!(provider_client.as_image_generation_capability().is_none());
    assert!(config_client.as_image_generation_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn deepseek_registry_non_text_requests_are_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/custom/v1",
    );

    let embedding_err = match registry.embedding_model("deepseek:deepseek-embedding-test") {
        Ok(_) => panic!("build registry embedding model should be unsupported"),
        Err(err) => err,
    };
    let image_err = match registry.image_model("deepseek:deepseek-image-test") {
        Ok(_) => panic!("build registry image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("deepseek:deepseek-rerank-test") {
        Ok(_) => panic!("build registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&embedding_err);
    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_capture_transports_unused(&[&registry_transport]);
}
