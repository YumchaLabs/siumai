use super::*;
use futures_util::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::extensions::types::{
    ImageEditInput, ImageEditRequest, VideoGenerationInput, VideoGenerationRequest,
};
use siumai::extensions::{SpeechExtras, VideoGenerationCapability};
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice, Warning,
};
use siumai::provider_ext::xai::{
    SearchMode, WebSearchSource, XaiChatRequestExt, XaiChatResponseExt, XaiImageOptions,
    XaiImageRequestExt, XaiOptions, XaiSearchParameters, XaiTtsOptions, XaiTtsRequestExt,
    XaiVideoOptions, XaiVideoRequestExt,
};
use siumai_registry::registry::builder::RegistryBuilder;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn xai_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("xai", "xai")
}

fn xai_registry_builder() -> RegistryBuilder {
    RegistryBuilder::new(xai_registry_providers())
}

fn make_xai_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    xai_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    xai_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1/")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "xai",
            "ctx-key",
            "https://example.com/xai/v1/",
            xai_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn make_registry(transport: Arc<dyn HttpTransport>) -> siumai::registry::ProviderRegistryHandle {
    xai_registry_builder()
        .with_provider_api_key_base_url_fetch(
            "xai",
            "test-key",
            "https://example.com/custom/v1",
            transport,
        )
        .build()
        .expect("build registry")
}

fn make_registry_with_global_reasoning_defaults(
    transport: Arc<dyn HttpTransport>,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    xai_registry_builder()
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .with_provider_api_key_base_url_fetch(
            "xai",
            "test-key",
            "https://example.com/custom/v1",
            transport,
        )
        .build()
        .expect("build registry")
}

fn make_registry_builder_with_global_reasoning_defaults(
    transport: Arc<dyn HttpTransport>,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    xai_registry_builder()
        .with_api_key("test-key")
        .with_base_url("https://example.com/custom/v1")
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .fetch(transport)
        .build()
        .expect("build registry")
}

fn make_xai_tool_call_request(model: &str) -> ChatRequest {
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

#[test]
fn xai_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::xai::XaiProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/xai")
        .with_header("x-test", "1")
        .into_config_for_model("grok-4")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.com/xai");
    assert_eq!(config.common_params.model, "grok-4");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

fn assert_xai_weather_tool_call_response(response: &siumai::prelude::unified::ChatResponse) {
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

fn make_xai_default_search_parameters() -> XaiSearchParameters {
    XaiSearchParameters {
        mode: SearchMode::On,
        return_citations: Some(true),
        max_search_results: Some(4),
        from_date: Some("2026-03-01".to_string()),
        to_date: Some("2026-03-12".to_string()),
        sources: Some(vec![
            WebSearchSource {
                country: Some("US".to_string()),
                allowed_websites: Some(vec![
                    "blog.rust-lang.org".to_string(),
                    "this-week-in-rust.org".to_string(),
                ]),
                excluded_websites: Some(vec!["example.com".to_string()]),
                safe_search: Some(true),
            }
            .into(),
        ]),
    }
}

fn assert_xai_default_options_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/chat/completions"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
    assert_xai_reasoning_effort_high(&req.body);
    assert_eq!(
        req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        req.body["search_parameters"]["max_search_results"],
        serde_json::json!(4)
    );
    assert_eq!(
        req.body["search_parameters"]["from_date"],
        serde_json::json!("2026-03-01")
    );
    assert_eq!(
        req.body["search_parameters"]["to_date"],
        serde_json::json!("2026-03-12")
    );
    assert_eq!(
        req.body["search_parameters"]["sources"][0]["type"],
        serde_json::json!("web")
    );
    assert_eq!(
        req.body["search_parameters"]["sources"][0]["country"],
        serde_json::json!("US")
    );
    assert_eq!(
        req.body["search_parameters"]["sources"][0]["allowed_websites"],
        serde_json::json!(["blog.rust-lang.org", "this-week-in-rust.org"])
    );
    assert_eq!(
        req.body["search_parameters"]["sources"][0]["excluded_websites"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        req.body["search_parameters"]["sources"][0]["safe_search"],
        serde_json::json!(true)
    );
}

fn assert_xai_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_xai_default_options_request(req, base_url, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
}

fn assert_xai_chat_request_filters_responses_only_options(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_eq!(req.url, format!("{base_url}/chat/completions"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["logprobs"], serde_json::json!(true));
    assert_eq!(req.body["top_logprobs"], serde_json::json!(2));
    assert_xai_reasoning_effort_high(&req.body);
    assert!(req.body.get("reasoning_summary").is_none());
    assert!(req.body.get("previous_response_id").is_none());
    assert!(req.body.get("include").is_none());
    assert!(req.body.get("store").is_none());
}

fn assert_xai_reasoning_effort_high(body: &serde_json::Value) {
    assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
    assert!(body.get("enable_reasoning").is_none());
    assert!(body.get("reasoning_budget").is_none());
}

fn make_xai_image_generation_request(model: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a tiny purple robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some(model.to_string()),
        quality: None,
        style: Some("comic".to_string()),
        aspect_ratio: None,
        seed: Some(7),
        steps: Some(20),
        guidance_scale: Some(7.5),
        enhance_prompt: Some(true),
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_xai_image_options(
        XaiImageOptions::new()
            .with_aspect_ratio("16:9")
            .with_output_format("png")
            .with_sync_mode(true)
            .with_resolution("2k")
            .with_quality("high")
            .with_user("user-123"),
    )
}

fn assert_xai_image_generation_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/images/generations"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["prompt"], serde_json::json!("a tiny purple robot"));
    assert_eq!(req.body["n"], serde_json::json!(1));
    assert_eq!(req.body["response_format"], serde_json::json!("b64_json"));
    assert_eq!(req.body["aspect_ratio"], serde_json::json!("16:9"));
    assert_eq!(req.body["output_format"], serde_json::json!("png"));
    assert_eq!(req.body["sync_mode"], serde_json::json!(true));
    assert_eq!(req.body["resolution"], serde_json::json!("2k"));
    assert_eq!(req.body["quality"], serde_json::json!("high"));
    assert_eq!(req.body["user"], serde_json::json!("user-123"));
    assert!(req.body.get("size").is_none());
    assert!(req.body.get("negative_prompt").is_none());
    assert!(req.body.get("seed").is_none());
    assert!(req.body.get("style").is_none());
    assert!(req.body.get("steps").is_none());
    assert!(req.body.get("guidance_scale").is_none());
    assert!(req.body.get("enhance_prompt").is_none());
}

fn make_xai_image_edit_request(model: &str) -> ImageEditRequest {
    ImageEditRequest {
        images: vec![ImageEditInput::file(vec![1, 2, 3, 4])],
        mask: Some(ImageEditInput::file(vec![5, 6, 7, 8])),
        prompt: "replace the background with a neon city".to_string(),
        model: Some(model.to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        seed: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_xai_image_options(
        XaiImageOptions::new()
            .with_aspect_ratio("9:16")
            .with_output_format("webp")
            .with_quality("medium"),
    )
}

fn make_xai_multi_image_edit_request(model: &str) -> ImageEditRequest {
    ImageEditRequest {
        images: vec![
            ImageEditInput::file(vec![1, 2, 3, 4]),
            ImageEditInput::url("https://example.com/input-2.png"),
        ],
        mask: Some(ImageEditInput::file(vec![5, 6, 7, 8])),
        prompt: "blend the two source images into a synthwave poster".to_string(),
        model: Some(model.to_string()),
        count: Some(2),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        seed: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_xai_image_options(
        XaiImageOptions::new()
            .with_aspect_ratio("1:1")
            .with_output_format("png")
            .with_quality("high"),
    )
}

fn assert_xai_image_edit_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/images/edits"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["prompt"],
        serde_json::json!("replace the background with a neon city")
    );
    assert_eq!(req.body["n"], serde_json::json!(1));
    assert_eq!(req.body["response_format"], serde_json::json!("b64_json"));
    assert_eq!(req.body["aspect_ratio"], serde_json::json!("9:16"));
    assert_eq!(req.body["output_format"], serde_json::json!("webp"));
    assert_eq!(req.body["quality"], serde_json::json!("medium"));
    assert_eq!(req.body["image"]["type"], serde_json::json!("image_url"));
    assert!(
        req.body["image"]["url"]
            .as_str()
            .is_some_and(|value| value.starts_with("data:image/png;base64,"))
    );
    assert!(req.body.get("mask").is_none());
    assert!(req.body.get("size").is_none());
}

fn assert_xai_multi_image_edit_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/images/edits"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["prompt"],
        serde_json::json!("blend the two source images into a synthwave poster")
    );
    assert_eq!(req.body["n"], serde_json::json!(2));
    assert_eq!(req.body["response_format"], serde_json::json!("b64_json"));
    assert_eq!(req.body["aspect_ratio"], serde_json::json!("1:1"));
    assert_eq!(req.body["output_format"], serde_json::json!("png"));
    assert_eq!(req.body["quality"], serde_json::json!("high"));
    assert!(req.body.get("image").is_none());
    assert_eq!(
        req.body["images"][0]["type"],
        serde_json::json!("image_url")
    );
    assert!(
        req.body["images"][0]["url"]
            .as_str()
            .is_some_and(|value| value.starts_with("data:image/png;base64,"))
    );
    assert_eq!(
        req.body["images"][1]["url"],
        serde_json::json!("https://example.com/input-2.png")
    );
    assert!(req.body.get("mask").is_none());
    assert!(req.body.get("size").is_none());
}

fn make_xai_video_generation_request(model: &str) -> VideoGenerationRequest {
    VideoGenerationRequest::new(model, "a tiny robot walking in rain")
        .with_count(2)
        .with_duration(5)
        .with_aspect_ratio("16:9")
        .with_fps(24)
        .with_seed(7)
        .with_image(VideoGenerationInput::url(
            "https://example.com/start-frame.png",
        ))
        .with_xai_video_options(
            XaiVideoOptions::new()
                .with_resolution("720p")
                .with_poll_interval_ms(1200)
                .with_poll_timeout_ms(30_000)
                .with_extra_field("style", serde_json::json!("cinematic")),
        )
        .with_header("x-video-test", "1")
}

fn assert_xai_video_create_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/videos/generations"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["prompt"],
        serde_json::json!("a tiny robot walking in rain")
    );
    assert_eq!(
        req.body["image"]["url"],
        serde_json::json!("https://example.com/start-frame.png")
    );
    assert_eq!(req.body["duration"], serde_json::json!(5));
    assert_eq!(req.body["aspect_ratio"], serde_json::json!("16:9"));
    assert_eq!(req.body["resolution"], serde_json::json!("720p"));
    assert_eq!(req.body["style"], serde_json::json!("cinematic"));
    assert_eq!(header_value(req, "x-video-test"), Some("1".to_string()));
    assert!(req.body.get("poll_interval_ms").is_none());
    assert!(req.body.get("poll_timeout_ms").is_none());
    assert!(req.body.get("fps").is_none());
    assert!(req.body.get("seed").is_none());
    assert!(req.body.get("n").is_none());
}

fn make_xai_video_extension_request(model: &str) -> VideoGenerationRequest {
    VideoGenerationRequest::new(model, "extend the clip")
        .with_duration(6)
        .with_aspect_ratio("16:9")
        .with_xai_video_options(
            XaiVideoOptions::new()
                .with_mode("extend-video")
                .with_video_url("https://example.com/input.mp4")
                .with_resolution("720p")
                .with_poll_interval_ms(1500)
                .with_poll_timeout_ms(45_000),
        )
        .with_header("x-video-mode", "extend")
}

fn assert_xai_video_extension_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/videos/extensions"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["prompt"], serde_json::json!("extend the clip"));
    assert_eq!(req.body["duration"], serde_json::json!(6));
    assert_eq!(
        req.body["video"]["url"],
        serde_json::json!("https://example.com/input.mp4")
    );
    assert_eq!(
        header_value(req, "x-video-mode"),
        Some("extend".to_string())
    );
    assert!(req.body.get("aspect_ratio").is_none());
    assert!(req.body.get("resolution").is_none());
    assert!(req.body.get("poll_interval_ms").is_none());
    assert!(req.body.get("poll_timeout_ms").is_none());
}

fn make_xai_reference_to_video_request(model: &str) -> VideoGenerationRequest {
    VideoGenerationRequest::new(model, "animate this style")
        .with_duration(4)
        .with_aspect_ratio("16:9")
        .with_xai_video_options(
            XaiVideoOptions::new()
                .with_mode("reference-to-video")
                .with_resolution("720p")
                .with_reference_image_urls([
                    "https://example.com/ref-1.png",
                    "https://example.com/ref-2.png",
                ])
                .with_poll_interval_ms(900)
                .with_poll_timeout_ms(20_000)
                .with_extra_field("style", serde_json::json!("cinematic")),
        )
        .with_header("x-video-mode", "reference")
}

fn assert_xai_reference_to_video_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/videos/generations"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["prompt"], serde_json::json!("animate this style"));
    assert_eq!(req.body["duration"], serde_json::json!(4));
    assert_eq!(req.body["aspect_ratio"], serde_json::json!("16:9"));
    assert_eq!(req.body["resolution"], serde_json::json!("720p"));
    assert_eq!(req.body["style"], serde_json::json!("cinematic"));
    assert_eq!(
        req.body["reference_images"],
        serde_json::json!([
            { "url": "https://example.com/ref-1.png" },
            { "url": "https://example.com/ref-2.png" }
        ])
    );
    assert_eq!(
        header_value(req, "x-video-mode"),
        Some("reference".to_string())
    );
    assert!(req.body.get("poll_interval_ms").is_none());
    assert!(req.body.get("poll_timeout_ms").is_none());
}

fn wiremock_header_value(req: &WiremockRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string)
}

async fn collect_streamed_tool_call(
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
async fn xai_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-beta")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-beta")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("grok-beta")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("grok-beta")
        .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn xai_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-beta")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-beta")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("grok-beta")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("grok-beta")
        .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

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
async fn xai_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .stop_sequences(vec!["END".to_string()])
        .response_format(siumai::prelude::unified::ResponseFormat::json_schema(
            schema.clone(),
        ))
        .build()
        .with_provider_option(
            "xai",
            serde_json::json!({
                "response_format": { "type": "json_object" },
                "reasoningEffort": "high"
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
    assert!(siumai_req.body.get("stop").is_none());
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        })
    );
}

#[tokio::test]
async fn xai_siumai_provider_config_filters_responses_only_chat_options_consistently() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_xai_reasoning_effort("high")
        .with_xai_reasoning_summary("detailed")
        .with_xai_top_logprobs(2)
        .with_xai_store(false)
        .with_xai_previous_response("resp_prev_123")
        .with_xai_include(["file_search_call.results"])
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_reasoning_effort("high")
        .with_reasoning_summary("detailed")
        .with_top_logprobs(2)
        .with_store(false)
        .with_previous_response("resp_prev_123")
        .with_include(["file_search_call.results"])
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning_effort("high")
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"])
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
    assert_xai_chat_request_filters_responses_only_options(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_default_options_match_public_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
        .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
        .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
            .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
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
    assert_xai_default_options_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_default_options_match_public_stream_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
        .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
        .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_xai_options(XaiOptions::new().with_reasoning_effort("high"))
            .with_xai_options(XaiOptions::new().with_search(make_xai_default_search_parameters()))
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
    assert_xai_default_options_stream_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_reasoning_defaults_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
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
    assert_xai_reasoning_effort_high(&siumai_req.body);
}

#[tokio::test]
async fn xai_registry_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
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
        true,
        1024,
    );
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_xai_reasoning_effort_high(&registry_req.body);
}

#[tokio::test]
async fn xai_registry_builder_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
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
        true,
        1024,
    );
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_xai_reasoning_effort_high(&registry_req.body);
}

#[tokio::test]
async fn xai_reasoning_response_is_equivalent_across_public_paths() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-xai-reasoning",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "There are three letter r's in strawberry.",
                    "thinking": "Count the letters carefully. The word strawberry contains three r characters."
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

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning(true)
            .with_reasoning_budget(1536)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        Arc::new(registry_transport.clone()),
        true,
        1536,
    );
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

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
    assert_xai_reasoning_effort_high(&siumai_req.body);
}

#[tokio::test]
async fn xai_reasoning_stream_is_equivalent_across_public_paths() {
    let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"Count the letters carefully. ","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"The word strawberry contains three r characters.","content":"There are three letter r's in strawberry.","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":24,"total_tokens":42,"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning(true)
            .with_reasoning_budget(1536)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        Arc::new(registry_transport.clone()),
        true,
        1536,
    );
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

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
    assert_xai_reasoning_effort_high(&siumai_req.body);
}

#[tokio::test]
async fn xai_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-xai-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from xai"
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
        "sources": [
            {
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com",
                "title": "Example"
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
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

    let siumai_meta = siumai_resp.xai_metadata().expect("siumai xai metadata");
    let provider_meta = provider_resp.xai_metadata().expect("provider xai metadata");
    let config_meta = config_resp.xai_metadata().expect("config xai metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from xai"));
    assert_eq!(provider_resp.content_text(), Some("hello from xai"));
    assert_eq!(config_resp.content_text(), Some("hello from xai"));
    assert_eq!(siumai_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        provider_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        config_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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
async fn xai_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://example.com","title":"Example"}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"content":" from xai","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
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

    let siumai_meta = siumai_resp.xai_metadata().expect("siumai xai metadata");
    let provider_meta = provider_resp.xai_metadata().expect("provider xai metadata");
    let config_meta = config_resp.xai_metadata().expect("config xai metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from xai"));
    assert_eq!(provider_resp.content_text(), Some("hello from xai"));
    assert_eq!(config_resp.content_text(), Some("hello from xai"));
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
    assert_eq!(siumai_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        provider_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        config_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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
async fn xai_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "grok-beta";
    let request_model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-beta")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model)
        .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
    assert_eq!(
        registry_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
}

#[tokio::test]
async fn xai_registry_filters_responses_only_chat_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let mut request_options = serde_json::to_value(
        siumai::provider_ext::xai::XaiOptions::new().with_reasoning_effort("high"),
    )
    .expect("serialize xai chat options");
    let responses_options = serde_json::to_value(
        siumai::provider_ext::xai::XaiResponsesOptions::new()
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"]),
    )
    .expect("serialize xai responses options");
    request_options
        .as_object_mut()
        .expect("xai request options object")
        .extend(
            responses_options
                .as_object()
                .expect("xai responses options object")
                .clone(),
        );

    let request =
        ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(request_options);

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_xai_chat_request_filters_responses_only_options(&registry_req, base_url, model);
}

#[tokio::test]
async fn xai_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "grok-beta";
    let request_model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-beta")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model)
        .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

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
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn xai_registry_chat_response_metadata_match_config_path() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-xai-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from xai"
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
        "sources": [
            {
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com",
                "title": "Example"
            }
        ]
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
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

    let config_meta = config_resp.xai_metadata().expect("config xai metadata");
    let registry_meta = registry_resp.xai_metadata().expect("registry xai metadata");

    assert_eq!(config_resp.content_text(), Some("hello from xai"));
    assert_eq!(registry_resp.content_text(), Some("hello from xai"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        config_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        config_req.url,
        "https://example.com/custom/v1/chat/completions"
    );
}

#[tokio::test]
async fn xai_registry_stream_end_metadata_match_config_path() {
    let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://example.com","title":"Example"}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"content":" from xai","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
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

    let config_meta = config_resp.xai_metadata().expect("config xai metadata");
    let registry_meta = registry_resp.xai_metadata().expect("registry xai metadata");

    assert_eq!(config_resp.content_text(), Some("hello from xai"));
    assert_eq!(registry_resp.content_text(), Some("hello from xai"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        config_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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
async fn xai_registry_override_chat_response_metadata_preserves_vendor_namespace() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-xai-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from xai"
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
        "sources": [
            {
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com",
                "title": "Example"
            }
        ]
    });

    let global_transport = CaptureTransport::default();
    let xai_transport = JsonSuccessTransport::new(response_json);

    let registry = make_xai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(xai_transport.clone()),
    );

    let response = registry
        .language_model("xai:grok-4")
        .expect("build xai handle")
        .chat_request(make_chat_request_with_model("grok-4"))
        .await
        .expect("registry response ok");

    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("xai").is_some());

    let metadata = response.xai_metadata().expect("xai metadata");
    assert_eq!(response.content_text(), Some("hello from xai"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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

    let req = xai_transport.take().expect("captured xai request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
}

#[tokio::test]
async fn xai_registry_override_stream_end_metadata_preserves_vendor_namespace() {
    let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://example.com","title":"Example"}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"content":" from xai","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let global_transport = CaptureTransport::default();
    let xai_transport = SseSuccessTransport::new(stream_body);

    let registry = make_xai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(xai_transport.clone()),
    );

    let mut stream = registry
        .language_model("xai:grok-4")
        .expect("build xai handle")
        .chat_stream_request(make_chat_request_with_model("grok-4"))
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
    assert!(root.get("xai").is_some());

    let metadata = response.xai_metadata().expect("xai metadata");
    assert_eq!(response.content_text(), Some("hello from xai"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://example.com")
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

    let req = xai_transport
        .take_stream()
        .expect("captured xai stream request");
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
    assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
}

#[tokio::test]
async fn xai_structured_output_synthetic_unknown_stream_end_extracts_across_public_paths() {
    let stream_body = concat!(
            "data: {\"id\":\"1\",\"model\":\"grok-4\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\\\"hel\",\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"model\":\"grok-4\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\\\"}\"},\"finish_reason\":null}]}\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

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
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        })
    );
}

#[tokio::test]
async fn xai_structured_output_synthetic_unknown_stream_end_fails_consistently_across_public_paths()
{
    let stream_body = "data: {\"id\":\"1\",\"model\":\"grok-4\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\" ,\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
            .as_bytes()
            .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

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
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        })
    );
}

#[tokio::test]
async fn xai_siumai_provider_config_web_search_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
        siumai::provider_ext::xai::XaiOptions::new()
            .with_reasoning_effort("high")
            .with_parallel_function_calling(false)
            .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                mode: siumai::provider_ext::xai::SearchMode::On,
                return_citations: Some(true),
                max_search_results: Some(5),
                from_date: Some("2026-03-01".to_string()),
                to_date: Some("2026-03-11".to_string()),
                sources: Some(vec![
                    siumai::provider_ext::xai::WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec![
                            "blog.rust-lang.org".to_string(),
                            "this-week-in-rust.org".to_string(),
                        ]),
                        excluded_websites: Some(vec!["example.com".to_string()]),
                        safe_search: Some(true),
                    }
                    .into(),
                ]),
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
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["from_date"],
        serde_json::json!("2026-03-01")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["to_date"],
        serde_json::json!("2026-03-11")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["type"],
        serde_json::json!("web")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["country"],
        serde_json::json!("US")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["allowed_websites"],
        serde_json::json!(["blog.rust-lang.org", "this-week-in-rust.org"])
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["excluded_websites"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["safe_search"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn xai_registry_web_search_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
        siumai::provider_ext::xai::XaiOptions::new()
            .with_reasoning_effort("high")
            .with_parallel_function_calling(false)
            .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                mode: siumai::provider_ext::xai::SearchMode::On,
                return_citations: Some(true),
                max_search_results: Some(5),
                from_date: Some("2026-03-01".to_string()),
                to_date: Some("2026-03-11".to_string()),
                sources: Some(vec![
                    siumai::provider_ext::xai::WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec![
                            "blog.rust-lang.org".to_string(),
                            "this-week-in-rust.org".to_string(),
                        ]),
                        excluded_websites: Some(vec!["example.com".to_string()]),
                        safe_search: Some(true),
                    }
                    .into(),
                ]),
            }),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        registry_req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["from_date"],
        serde_json::json!("2026-03-01")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["to_date"],
        serde_json::json!("2026-03-11")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["type"],
        serde_json::json!("web")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["country"],
        serde_json::json!("US")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["allowed_websites"],
        serde_json::json!(["blog.rust-lang.org", "this-week-in-rust.org"])
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["excluded_websites"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["safe_search"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn xai_siumai_provider_config_tool_choice_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
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
            "xai",
            serde_json::json!({
                "tool_choice": "auto",
                "reasoningEffort": "high"
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
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn xai_tool_calls_stream_and_non_stream_are_equivalent_across_public_paths() {
    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let non_stream_response = serde_json::json!({
        "id": "chatcmpl-xai-tool-call",
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

    let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\""}}]},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo\"}"}}]},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

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
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_non_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai non-stream client");

    let provider_non_stream_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_non_stream_transport.clone()))
        .build()
        .await
        .expect("build provider non-stream client");

    let config_non_stream_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_non_stream_transport.clone())),
    )
    .await
    .expect("build config non-stream client");

    let registry_non_stream = make_registry(Arc::new(registry_non_stream_transport.clone()));
    let registry_non_stream_model = registry_non_stream
        .language_model("xai:grok-4")
        .expect("build registry non-stream language model");

    let siumai_stream_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai stream client");

    let provider_stream_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_stream_transport.clone()))
        .build()
        .await
        .expect("build provider stream client");

    let config_stream_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_stream_transport.clone())),
    )
    .await
    .expect("build config stream client");

    let registry_stream = make_registry(Arc::new(registry_stream_transport.clone()));
    let registry_stream_model = registry_stream
        .language_model("xai:grok-4")
        .expect("build registry stream language model");

    let request = make_xai_tool_call_request(model);

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
    ) = collect_streamed_tool_call(&mut siumai_stream).await;
    let (
        provider_stream_resp,
        provider_stream_tool_call_id,
        provider_stream_tool_name,
        provider_stream_arguments,
    ) = collect_streamed_tool_call(&mut provider_stream).await;
    let (
        config_stream_resp,
        config_stream_tool_call_id,
        config_stream_tool_name,
        config_stream_arguments,
    ) = collect_streamed_tool_call(&mut config_stream).await;
    let (
        registry_stream_resp,
        registry_stream_tool_call_id,
        registry_stream_tool_name,
        registry_stream_arguments,
    ) = collect_streamed_tool_call(&mut registry_stream_handle).await;

    for response in [
        &siumai_non_stream_resp,
        &provider_non_stream_resp,
        &config_non_stream_resp,
        &registry_non_stream_resp,
    ] {
        assert_xai_weather_tool_call_response(response);
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

    for response in [
        &siumai_stream_resp,
        &provider_stream_resp,
        &config_stream_resp,
        &registry_stream_resp,
    ] {
        assert!(
            response.tool_calls().is_empty(),
            "stream end response should keep finish reason while tool-call equivalence is asserted via typed tool input accumulation"
        );
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
async fn xai_siumai_provider_config_web_search_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
        siumai::provider_ext::xai::XaiOptions::new()
            .with_reasoning_effort("high")
            .with_parallel_function_calling(false)
            .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                mode: siumai::provider_ext::xai::SearchMode::On,
                return_citations: Some(true),
                max_search_results: Some(5),
                from_date: Some("2026-03-01".to_string()),
                to_date: Some("2026-03-11".to_string()),
                sources: Some(vec![
                    siumai::provider_ext::xai::WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec![
                            "blog.rust-lang.org".to_string(),
                            "this-week-in-rust.org".to_string(),
                        ]),
                        excluded_websites: Some(vec!["example.com".to_string()]),
                        safe_search: Some(true),
                    }
                    .into(),
                ]),
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
        siumai_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["from_date"],
        serde_json::json!("2026-03-01")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["to_date"],
        serde_json::json!("2026-03-11")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["type"],
        serde_json::json!("web")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["country"],
        serde_json::json!("US")
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["allowed_websites"],
        serde_json::json!(["blog.rust-lang.org", "this-week-in-rust.org"])
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["excluded_websites"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        siumai_req.body["search_parameters"]["sources"][0]["safe_search"],
        serde_json::json!(true)
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn xai_registry_web_search_stream_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
        siumai::provider_ext::xai::XaiOptions::new()
            .with_reasoning_effort("high")
            .with_parallel_function_calling(false)
            .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                mode: siumai::provider_ext::xai::SearchMode::On,
                return_citations: Some(true),
                max_search_results: Some(5),
                from_date: Some("2026-03-01".to_string()),
                to_date: Some("2026-03-11".to_string()),
                sources: Some(vec![
                    siumai::provider_ext::xai::WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec![
                            "blog.rust-lang.org".to_string(),
                            "this-week-in-rust.org".to_string(),
                        ]),
                        excluded_websites: Some(vec!["example.com".to_string()]),
                        safe_search: Some(true),
                    }
                    .into(),
                ]),
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
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        registry_req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
    assert_eq!(
        registry_req.body["search_parameters"]["from_date"],
        serde_json::json!("2026-03-01")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["to_date"],
        serde_json::json!("2026-03-11")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["type"],
        serde_json::json!("web")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["country"],
        serde_json::json!("US")
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["allowed_websites"],
        serde_json::json!(["blog.rust-lang.org", "this-week-in-rust.org"])
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["excluded_websites"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(
        registry_req.body["search_parameters"]["sources"][0]["safe_search"],
        serde_json::json!(true)
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn xai_registry_chat_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let xai_transport = CaptureTransport::default();

    let registry = make_xai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(xai_transport.clone()),
    );

    let handle = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let _ = handle
        .chat_request(
            ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
                siumai::provider_ext::xai::XaiOptions::new()
                    .with_reasoning_effort("high")
                    .with_parallel_function_calling(false)
                    .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                        mode: siumai::provider_ext::xai::SearchMode::On,
                        return_citations: Some(true),
                        max_search_results: Some(5),
                        from_date: Some("2026-03-01".to_string()),
                        to_date: Some("2026-03-11".to_string()),
                        sources: Some(vec![
                            siumai::provider_ext::xai::WebSearchSource {
                                country: Some("US".to_string()),
                                allowed_websites: Some(vec![
                                    "blog.rust-lang.org".to_string(),
                                    "this-week-in-rust.org".to_string(),
                                ]),
                                excluded_websites: Some(vec!["example.com".to_string()]),
                                safe_search: Some(true),
                            }
                            .into(),
                        ]),
                    }),
            ),
        )
        .await;

    let req = xai_transport.take().expect("captured xai request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("grok-4"));
    assert_eq!(req.body["reasoning_effort"], serde_json::json!("high"));
    assert_eq!(
        req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
}

#[tokio::test]
async fn xai_registry_chat_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let xai_transport = CaptureTransport::default();

    let registry = make_xai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(xai_transport.clone()),
    );

    let handle = registry
        .language_model("xai:grok-4")
        .expect("build registry language model");

    let _ = handle
        .chat_stream_request(
            ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
                siumai::provider_ext::xai::XaiOptions::new()
                    .with_reasoning_effort("high")
                    .with_parallel_function_calling(false)
                    .with_search(siumai::provider_ext::xai::XaiSearchParameters {
                        mode: siumai::provider_ext::xai::SearchMode::On,
                        return_citations: Some(true),
                        max_search_results: Some(5),
                        from_date: Some("2026-03-01".to_string()),
                        to_date: Some("2026-03-11".to_string()),
                        sources: Some(vec![
                            siumai::provider_ext::xai::WebSearchSource {
                                country: Some("US".to_string()),
                                allowed_websites: Some(vec![
                                    "blog.rust-lang.org".to_string(),
                                    "this-week-in-rust.org".to_string(),
                                ]),
                                excluded_websites: Some(vec!["example.com".to_string()]),
                                safe_search: Some(true),
                            }
                            .into(),
                        ]),
                    }),
            ),
        )
        .await;

    let req = xai_transport
        .take_stream()
        .expect("captured xai stream request");
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
    assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("grok-4"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(req.body["reasoning_effort"], serde_json::json!("high"));
    assert_eq!(
        req.body["parallel_function_calling"],
        serde_json::json!(false)
    );
    assert_eq!(
        req.body["search_parameters"]["mode"],
        serde_json::json!("on")
    );
    assert_eq!(
        req.body["search_parameters"]["return_citations"],
        serde_json::json!(true)
    );
    assert_eq!(
        req.body["search_parameters"]["max_search_results"],
        serde_json::json!(5)
    );
}

#[tokio::test]
async fn xai_siumai_provider_config_embedding_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-embedding-test")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model("grok-embedding-test")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model("grok-embedding-test")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = EmbeddingRequest::single("hello xai embedding").with_model("grok-embedding-test");

    let siumai_err = siumai_client
        .embed_with_config(request.clone())
        .await
        .expect_err("xai embedding should be unsupported");
    let provider_err = provider_client
        .embed_with_config(request.clone())
        .await
        .expect_err("xai embedding should be unsupported");
    let config_err = config_client
        .embed_with_config(request)
        .await
        .expect_err("xai embedding should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(provider_client.as_embedding_capability().is_none());
    assert!(config_client.as_embedding_capability().is_none());
    assert!(provider_client.as_image_generation_capability().is_some());
    assert!(config_client.as_image_generation_capability().is_some());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn xai_siumai_provider_config_rerank_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "grok-rerank-test";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
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
        .expect_err("xai rerank should be unsupported");
    let provider_err = provider_client
        .rerank(request.clone())
        .await
        .expect_err("xai rerank should be unsupported");
    let config_err = config_client
        .rerank(request)
        .await
        .expect_err("xai rerank should be unsupported");

    assert!(matches!(siumai_err, LlmError::UnsupportedOperation(_)));
    assert!(matches!(provider_err, LlmError::UnsupportedOperation(_)));
    assert!(matches!(config_err, LlmError::UnsupportedOperation(_)));
    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
}

#[tokio::test]
async fn xai_siumai_provider_config_image_request_are_equivalent() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8=",
                "revised_prompt": "a tiny purple robot"
            }
        ],
        "usage": {
            "cost_in_usd_ticks": 321
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-image";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_xai_image_generation_request(model);

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image generation ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image generation ok");
    let config_resp = config_client
        .generate_images(request)
        .await
        .expect("config image generation ok");

    assert_eq!(siumai_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );
    assert_eq!(config_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_xai_image_generation_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_registry_image_request_options_match_config_path() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8="
            }
        ]
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-image-pro";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .image_model("xai:grok-imagine-image-pro")
        .expect("build registry image model");

    let request = make_xai_image_generation_request(model);

    let _ = config_client.generate_images(request.clone()).await;
    let _ = registry_model.generate_images(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_xai_image_generation_request(&registry_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_image_edit_request_are_equivalent() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8="
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-image";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_xai_image_edit_request(model);

    let _ = siumai_client.edit_image(request.clone()).await;
    let _ = provider_client.edit_image(request.clone()).await;
    let _ = config_client.edit_image(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_xai_image_edit_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_registry_image_edit_request_options_match_config_path() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8="
            }
        ]
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-image";
    let base_url = "https://example.com/custom/v1";

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .image_model("xai:grok-imagine-image")
        .expect("build registry image model");

    let request = make_xai_image_edit_request(model);

    let _ = config_client.edit_image(request.clone()).await;
    let _ = registry_model.edit_image(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_xai_image_edit_request(&registry_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_multi_image_edit_request_are_equivalent() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8="
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-image";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_xai_multi_image_edit_request(model);

    let _ = siumai_client.edit_image(request.clone()).await;
    let _ = provider_client.edit_image(request.clone()).await;
    let _ = config_client.edit_image(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_xai_multi_image_edit_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_registry_video_create_are_equivalent() {
    let response_json = serde_json::json!({
        "request_id": "task-123",
        "queued": true
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-video";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_client = registry
        .video_model("xai:grok-imagine-video")
        .expect("build registry video model");

    let request = make_xai_video_generation_request(model);

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

    assert_eq!(siumai_resp.task_id, "task-123");
    assert_eq!(provider_resp.task_id, "task-123");
    assert_eq!(config_resp.task_id, "task-123");
    assert_eq!(registry_resp.task_id, "task-123");
    assert_eq!(
        siumai_resp.warnings,
        Some(vec![
            Warning::unsupported(
                "n",
                Some("xAI video models do not support generating multiple videos per call."),
            ),
            Warning::unsupported("fps", Some("xAI video models do not support custom FPS."),),
            Warning::unsupported(
                "seed",
                Some("xAI video models do not support deterministic seeds."),
            ),
        ])
    );
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
    assert_xai_video_create_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_registry_video_extension_create_are_equivalent() {
    let response_json = serde_json::json!({
        "request_id": "task-extend-123",
        "queued": true
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-video";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_client = registry
        .video_model("xai:grok-imagine-video")
        .expect("build registry video model");

    let request = make_xai_video_extension_request(model);

    let siumai_resp = siumai_client
        .create_video_task(request.clone())
        .await
        .expect("siumai create extension ok");
    let provider_resp = provider_client
        .create_video_task(request.clone())
        .await
        .expect("provider create extension ok");
    let config_resp = config_client
        .create_video_task(request.clone())
        .await
        .expect("config create extension ok");
    let registry_resp = registry_client
        .create_video_task(request)
        .await
        .expect("registry create extension ok");

    assert_eq!(siumai_resp.task_id, "task-extend-123");
    assert_eq!(provider_resp.task_id, "task-extend-123");
    assert_eq!(config_resp.task_id, "task-extend-123");
    assert_eq!(registry_resp.task_id, "task-extend-123");

    let siumai_req = siumai_transport.take().expect("siumai extension request");
    let provider_req = provider_transport
        .take()
        .expect("provider extension request");
    let config_req = config_transport.take().expect("config extension request");
    let registry_req = registry_transport
        .take()
        .expect("registry extension request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_xai_video_extension_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_registry_reference_to_video_create_are_equivalent() {
    let response_json = serde_json::json!({
        "request_id": "task-reference-123",
        "queued": true
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "grok-imagine-video";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_client = registry
        .video_model("xai:grok-imagine-video")
        .expect("build registry video model");

    let request = make_xai_reference_to_video_request(model);

    let siumai_resp = siumai_client
        .create_video_task(request.clone())
        .await
        .expect("siumai create reference-to-video ok");
    let provider_resp = provider_client
        .create_video_task(request.clone())
        .await
        .expect("provider create reference-to-video ok");
    let config_resp = config_client
        .create_video_task(request.clone())
        .await
        .expect("config create reference-to-video ok");
    let registry_resp = registry_client
        .create_video_task(request)
        .await
        .expect("registry create reference-to-video ok");

    assert_eq!(siumai_resp.task_id, "task-reference-123");
    assert_eq!(provider_resp.task_id, "task-reference-123");
    assert_eq!(config_resp.task_id, "task-reference-123");
    assert_eq!(registry_resp.task_id, "task-reference-123");

    let siumai_req = siumai_transport.take().expect("siumai reference request");
    let provider_req = provider_transport
        .take()
        .expect("provider reference request");
    let config_req = config_transport.take().expect("config reference request");
    let registry_req = registry_transport
        .take()
        .expect("registry reference request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_xai_reference_to_video_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn xai_siumai_provider_config_registry_query_video_task_are_equivalent() {
    async fn mount_video_query_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/videos/task-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "done",
                "model": "grok-imagine-video",
                "video": {
                    "url": "https://cdn.example.com/video.mp4",
                    "duration": 6.5,
                    "respect_moderation": true,
                    "poster": "https://cdn.example.com/poster.jpg"
                },
                "usage": {
                    "cost_in_usd_ticks": 456
                },
                "request_type": "generation"
            })))
            .mount(&server)
            .await;
        server
    }

    let siumai_server = mount_video_query_server().await;
    let provider_server = mount_video_query_server().await;
    let config_server = mount_video_query_server().await;
    let registry_server = mount_video_query_server().await;

    let model = "grok-imagine-video";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(format!("{}/v1", siumai_server.uri()))
        .model(model)
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(format!("{}/v1", provider_server.uri()))
        .model(model)
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(format!("{}/v1", config_server.uri()))
            .with_model(model),
    )
    .await
    .expect("build config client");

    let registry = xai_registry_builder()
        .with_provider_api_key_base_url("xai", "test-key", format!("{}/v1", registry_server.uri()))
        .build()
        .expect("build registry");
    let registry_client = registry
        .video_model("xai:grok-imagine-video")
        .expect("build registry video model");

    let siumai_resp = siumai_client
        .query_video_task("task-123")
        .await
        .expect("siumai query video ok");
    let provider_resp = provider_client
        .query_video_task("task-123")
        .await
        .expect("provider query video ok");
    let config_resp = config_client
        .query_video_task("task-123")
        .await
        .expect("config query video ok");
    let registry_resp = registry_client
        .query_video_task("task-123")
        .await
        .expect("registry query video ok");

    assert_eq!(siumai_resp.status.to_string(), "Success");
    assert_eq!(provider_resp.status.to_string(), "Success");
    assert_eq!(config_resp.status.to_string(), "Success");
    assert_eq!(registry_resp.status.to_string(), "Success");
    assert_eq!(
        siumai_resp.video_url.as_deref(),
        Some("https://cdn.example.com/video.mp4")
    );
    assert_eq!(
        registry_resp.video_url.as_deref(),
        Some("https://cdn.example.com/video.mp4")
    );
    assert_eq!(siumai_resp.duration, Some(6.5));
    assert_eq!(registry_resp.duration, Some(6.5));

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai query request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider query request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config query request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry query request");

    assert_eq!(siumai_req.url.path(), "/v1/videos/task-123");
    assert_eq!(provider_req.url.path(), "/v1/videos/task-123");
    assert_eq!(config_req.url.path(), "/v1/videos/task-123");
    assert_eq!(registry_req.url.path(), "/v1/videos/task-123");
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&provider_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&config_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
}

#[tokio::test]
async fn xai_siumai_provider_config_stt_request_is_intentionally_unsupported() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));

    let model = "grok-voice-mini";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url("https://example.com/custom/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url("https://example.com/custom/v1")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some(model.to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let siumai_err = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect_err("xai speech-to-text should be unsupported");
    let provider_err = provider_client
        .speech_to_text(request.clone())
        .await
        .expect_err("xai speech-to-text should be unsupported");
    let config_err = config_client
        .speech_to_text(request)
        .await
        .expect_err("xai speech-to-text should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(siumai_client.as_audio_capability().is_some());
    assert!(provider_client.as_audio_capability().is_some());
    assert!(config_client.as_audio_capability().is_some());
    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());
    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
}

#[tokio::test]
async fn xai_registry_stt_request_is_intentionally_unsupported() {
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_err = match registry.transcription_model("xai:grok-voice-mini") {
        Ok(_) => panic!("build registry transcription model should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert!(registry_transport.take().is_none());
}

#[tokio::test]
async fn xai_siumai_provider_config_speech_extras_are_intentionally_unavailable() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "grok-4";
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .speech_model("xai:grok-4")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from xai".to_string())
        .with_voice("aria".to_string())
        .with_format("mp3".to_string());

    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_speech_extras().is_none());
    assert!(provider_client.as_speech_extras().is_none());
    assert!(config_client.as_speech_extras().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());
    assert!(siumai_client.as_transcription_extras().is_none());
    assert!(provider_client.as_transcription_extras().is_none());
    assert!(config_client.as_transcription_extras().is_none());

    let registry_err = match registry_model.tts_stream(request).await {
        Ok(_) => panic!("xai registry tts stream should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn xai_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let base_url = "https://example.com/custom/v1";

    let siumai_client = Siumai::builder()
        .xai()
        .api_key("test-key")
        .base_url(base_url)
        .model("grok-4")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::xai()
        .api_key("test-key")
        .base_url(base_url)
        .model("grok-4")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::xai::XaiClient::from_config(
        siumai::provider_ext::xai::XaiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model("grok-4")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .speech_model("xai:grok-4")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from xai".to_string())
        .with_model("grok-4".to_string())
        .with_voice("aria".to_string())
        .with_format("mp3".to_string())
        .with_xai_tts_options(
            XaiTtsOptions::new()
                .with_sample_rate(44_100)
                .with_bit_rate(192_000),
        );

    let siumai_resp = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect("siumai tts ok");
    let provider_resp = provider_client
        .text_to_speech(request.clone())
        .await
        .expect("provider tts ok");
    let config_resp = config_client
        .text_to_speech(request.clone())
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(&registry_model, request)
        .await
        .expect("registry tts ok");

    assert_eq!(siumai_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(provider_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(config_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(registry_resp.audio_data, vec![1, 2, 3, 4]);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/tts");
    assert_eq!(siumai_req.body["model"], serde_json::json!("grok-4"));
    assert_eq!(siumai_req.body["text"], serde_json::json!("hello from xai"));
    assert_eq!(siumai_req.body["voice_id"], serde_json::json!("aria"));
    assert_eq!(
        siumai_req.body["output_format"],
        serde_json::json!({
            "codec": "mp3",
            "sample_rate": 44_100,
            "bit_rate": 192_000
        })
    );
}
