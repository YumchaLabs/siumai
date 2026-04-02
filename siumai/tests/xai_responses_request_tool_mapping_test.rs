#![cfg(feature = "xai")]

use siumai::experimental::execution::transformers::request::RequestTransformer;
use siumai::prelude::unified::*;
use siumai_provider_xai::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;

fn transform_request(request: ChatRequest) -> serde_json::Value {
    let tx = OpenAiResponsesRequestTransformer;
    tx.transform_chat(&request).expect("transform request")
}

#[test]
fn xai_code_execution_tool_maps_to_code_interpreter_in_responses_request() {
    let req = ChatRequest {
        messages: vec![user!("hello")],
        tools: Some(vec![tools::xai::code_execution()]),
        common_params: CommonParams {
            model: "grok-4-fast".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let body = transform_request(req);

    let tools = body.get("tools").and_then(|v| v.as_array()).expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(
        tools[0].get("type").and_then(|v| v.as_str()),
        Some("code_interpreter")
    );
}

#[test]
fn xai_provider_tools_serialize_sdk_aligned_args_in_responses_request() {
    let req = ChatRequest {
        messages: vec![user!("hello")],
        tools: Some(vec![
            tools::xai::web_search_with(
                tools::xai::WebSearchArgs::new()
                    .with_allowed_domains(["wikipedia.org"])
                    .with_excluded_domains(["spam.com"])
                    .with_enable_image_understanding(true),
            ),
            tools::xai::x_search_with(
                tools::xai::XSearchArgs::new()
                    .with_allowed_x_handles(["xai"])
                    .with_excluded_x_handles(["spam_handle"])
                    .with_from_date("2025-01-01")
                    .with_to_date("2025-01-31")
                    .with_enable_video_understanding(true),
            ),
            tools::xai::view_image(),
            tools::xai::view_x_video(),
            tools::xai::file_search_with(
                tools::xai::FileSearchArgs::new(["collection_1"]).with_max_num_results(5),
            ),
            tools::xai::mcp_server_with(
                tools::xai::McpArgs::new("https://example.com/mcp")
                    .with_server_label("docs")
                    .with_server_description("Docs MCP")
                    .with_allowed_tools(["search_docs"])
                    .with_headers([("X-Test", "1")])
                    .with_authorization("Bearer token"),
            ),
        ]),
        common_params: CommonParams {
            model: "grok-4-fast".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let body = transform_request(req);
    let tools = body["tools"].as_array().expect("tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["allowed_domains"],
        serde_json::json!(["wikipedia.org"])
    );
    assert_eq!(
        tools[0]["excluded_domains"],
        serde_json::json!(["spam.com"])
    );
    assert_eq!(
        tools[0]["enable_image_understanding"],
        serde_json::json!(true)
    );

    assert_eq!(tools[1]["type"], serde_json::json!("x_search"));
    assert_eq!(tools[1]["allowed_x_handles"], serde_json::json!(["xai"]));
    assert_eq!(
        tools[1]["excluded_x_handles"],
        serde_json::json!(["spam_handle"])
    );
    assert_eq!(tools[1]["from_date"], serde_json::json!("2025-01-01"));
    assert_eq!(tools[1]["to_date"], serde_json::json!("2025-01-31"));
    assert_eq!(
        tools[1]["enable_video_understanding"],
        serde_json::json!(true)
    );

    assert_eq!(tools[2], serde_json::json!({ "type": "view_image" }));
    assert_eq!(tools[3], serde_json::json!({ "type": "view_x_video" }));

    assert_eq!(tools[4]["type"], serde_json::json!("file_search"));
    assert_eq!(
        tools[4]["vector_store_ids"],
        serde_json::json!(["collection_1"])
    );
    assert_eq!(tools[4]["max_num_results"], serde_json::json!(5));

    assert_eq!(tools[5]["type"], serde_json::json!("mcp"));
    assert_eq!(
        tools[5]["server_url"],
        serde_json::json!("https://example.com/mcp")
    );
    assert_eq!(tools[5]["server_label"], serde_json::json!("docs"));
    assert_eq!(
        tools[5]["server_description"],
        serde_json::json!("Docs MCP")
    );
    assert_eq!(
        tools[5]["allowed_tools"],
        serde_json::json!(["search_docs"])
    );
    assert_eq!(tools[5]["headers"], serde_json::json!({ "X-Test": "1" }));
    assert_eq!(tools[5]["authorization"], serde_json::json!("Bearer token"));
}

#[test]
fn xai_tool_choice_omits_server_side_provider_tools() {
    let req = ChatRequest {
        messages: vec![user!("hello")],
        tools: Some(vec![tools::xai::web_search()]),
        tool_choice: Some(ToolChoice::tool("web_search")),
        common_params: CommonParams {
            model: "grok-4-fast".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let body = transform_request(req);
    assert!(body.get("tool_choice").is_none());
}

#[test]
fn xai_tool_choice_keeps_function_tools() {
    let req = ChatRequest {
        messages: vec![user!("hello")],
        tools: Some(vec![Tool::function(
            "weather",
            "weather lookup",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        )]),
        tool_choice: Some(ToolChoice::tool("weather")),
        common_params: CommonParams {
            model: "grok-4-fast".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let body = transform_request(req);
    assert_eq!(
        body["tool_choice"],
        serde_json::json!({
            "type": "function",
            "name": "weather"
        })
    );
}

#[test]
fn xai_assistant_unsupported_parts_are_skipped_on_responses_request_path() {
    let req = ChatRequest::builder()
        .message(ChatMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![
                ContentPart::text("kept"),
                ContentPart::reasoning("internal chain"),
                ContentPart::reasoning_file_base64("aGVsbG8=", "image/png"),
                ContentPart::custom("openai.compaction"),
                ContentPart::file_base64("aGVsbG8=", "text/plain", Some("note.txt".to_string())),
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        })
        .provider_option("xai", serde_json::json!({}))
        .model("grok-4-fast")
        .build();

    let body = transform_request(req);
    let input = body["input"].as_array().expect("input array");

    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], serde_json::json!("assistant"));
    assert_eq!(input[0]["content"], serde_json::json!("kept"));
    assert!(input[0].get("type").is_none());
}
