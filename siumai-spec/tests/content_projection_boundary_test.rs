use siumai_spec::types::{
    AssistantContent, AssistantContentPart, AssistantModelMessage, ChatMessage, ChatResponse,
    ContentPart, FilePartSource, GenerateTextContentPartProjectionError, MessageContent,
    MessageRole, ModelMessage, ModelMessageConversionError, ProviderMetadataMap,
    ProviderOptionsMap, TextPart, ToolCallPart, ToolContentPart, ToolModelMessage,
    ToolResultOutput, ToolResultPart, UserContent, UserContentPart, UserModelMessage,
    project_chat_message_to_prompt_message, project_chat_response_to_generate_text_content_parts,
    project_prompt_messages_to_chat_messages,
    project_response_content_part_to_generate_text_content_part,
};

#[test]
fn response_content_projection_to_generate_text_parts_preserves_response_metadata_only() {
    let mut provider_options = ProviderOptionsMap::default();
    provider_options.insert(
        "openai",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );
    let provider_metadata = ProviderMetadataMap::from([(
        "openai".to_string(),
        serde_json::json!({ "responseId": "resp_1" }),
    )]);

    let text_part = ContentPart::Text {
        text: "hello".to_string(),
        provider_options: provider_options.clone(),
        provider_metadata: Some(provider_metadata.clone()),
    };
    let projected_text = project_response_content_part_to_generate_text_content_part(&text_part)
        .expect("text response part should project");
    let projected_text_value =
        serde_json::to_value(&projected_text).expect("serialize projected text");

    assert_eq!(projected_text_value["type"], serde_json::json!("text"));
    assert_eq!(projected_text_value["text"], serde_json::json!("hello"));
    assert_eq!(
        projected_text_value["providerMetadata"]["openai"]["responseId"],
        serde_json::json!("resp_1")
    );
    assert_json_has_no_key(&projected_text_value, "providerOptions");
    assert_json_has_no_key(&projected_text_value, "provider_options");

    let response = ChatResponse::new(MessageContent::MultiModal(vec![
        text_part,
        ContentPart::File {
            source: FilePartSource::base64("aGVsbG8="),
            media_type: "text/plain".to_string(),
            filename: Some("hello.txt".to_string()),
            provider_options: provider_options.clone(),
            provider_metadata: Some(provider_metadata.clone()),
        },
        ContentPart::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "search".to_string(),
            output: ToolResultOutput::text("ok"),
            input: Some(serde_json::json!({ "query": "rust" })),
            provider_executed: Some(true),
            dynamic: Some(false),
            preliminary: Some(false),
            title: Some("Search result".to_string()),
            provider_options,
            provider_metadata: Some(provider_metadata),
        },
    ]));

    let projected_parts = project_chat_response_to_generate_text_content_parts(&response)
        .expect("response content should project");
    let projected_value =
        serde_json::to_value(&projected_parts).expect("serialize projected response");

    assert_eq!(projected_parts.len(), 3);
    assert_eq!(projected_value[1]["type"], serde_json::json!("file"));
    assert_eq!(
        projected_value[1]["file"]["base64"],
        serde_json::json!("aGVsbG8=")
    );
    assert_eq!(
        projected_value[2]["input"],
        serde_json::json!({ "query": "rust" })
    );
    assert_json_has_no_key(&projected_value, "providerOptions");
    assert_json_has_no_key(&projected_value, "provider_options");
}

#[test]
fn response_content_projection_rejects_ambiguous_legacy_carriers() {
    let image = ContentPart::image_base64("aGVsbG8=").with_image_media_type("image/png");
    let err = project_response_content_part_to_generate_text_content_part(&image)
        .expect_err("image response projection should be ambiguous");
    assert_eq!(
        err,
        GenerateTextContentPartProjectionError::UnsupportedContentPart {
            part_type: "image",
            reason: "image content is ambiguous in generated text output projection",
        }
    );

    let file_url = ContentPart::File {
        source: FilePartSource::url("https://example.com/report.pdf"),
        media_type: "application/pdf".to_string(),
        filename: Some("report.pdf".to_string()),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    };
    let err = project_response_content_part_to_generate_text_content_part(&file_url)
        .expect_err("URL-backed file projection should be ambiguous");
    assert_eq!(
        err,
        GenerateTextContentPartProjectionError::UnsupportedContentPart {
            part_type: "file",
            reason: "generated file output requires base64 or binary data",
        }
    );

    let tool_result_without_input = ContentPart::tool_result_text("call_1", "search", "ok");
    let err =
        project_response_content_part_to_generate_text_content_part(&tool_result_without_input)
            .expect_err("tool-result output projection should require original input");
    assert_eq!(
        err,
        GenerateTextContentPartProjectionError::UnsupportedContentPart {
            part_type: "tool-result",
            reason: "tool-result generated output requires original input",
        }
    );
}

#[test]
fn adr_0008_root_content_part_move_has_serde_parity_fixture_gate() {
    let mut provider_options = ProviderOptionsMap::default();
    provider_options.insert(
        "openai",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );
    let provider_metadata = ProviderMetadataMap::from([(
        "openai".to_string(),
        serde_json::json!({ "itemId": "msg_1", "responseId": "resp_1" }),
    )]);

    let root_part = ContentPart::Text {
        text: "hello".to_string(),
        provider_options: provider_options.clone(),
        provider_metadata: Some(provider_metadata.clone()),
    };
    let compat_part = siumai_spec::types::content::compat::ContentPart::Text {
        text: "hello".to_string(),
        provider_options: provider_options.clone(),
        provider_metadata: Some(provider_metadata.clone()),
    };
    assert_eq!(
        serde_json::to_value(&root_part).expect("serialize root ContentPart"),
        serde_json::to_value(&compat_part).expect("serialize compat ContentPart"),
        "root and compat ContentPart paths must serialize identically until the root namespace move lands"
    );

    let message = ChatMessage {
        role: MessageRole::User,
        content: MessageContent::MultiModal(vec![root_part.clone()]),
        provider_options: provider_options.clone(),
        metadata: Default::default(),
    };
    let message_value = serde_json::to_value(&message).expect("serialize ChatMessage fixture");
    assert_eq!(message_value["role"], serde_json::json!("user"));
    assert_eq!(
        message_value["providerOptions"]["openai"]["cacheControl"]["type"],
        serde_json::json!("ephemeral")
    );
    assert_eq!(
        message_value["content"]["MultiModal"][0],
        serde_json::to_value(&compat_part).expect("serialize compat fixture part")
    );

    let message_roundtrip: ChatMessage =
        serde_json::from_value(message_value.clone()).expect("deserialize ChatMessage fixture");
    assert_eq!(
        serde_json::to_value(&message_roundtrip).expect("reserialize ChatMessage fixture"),
        message_value,
        "ChatMessage serde payload must remain stable before root ContentPart movement"
    );

    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![ContentPart::ToolCall {
        tool_call_id: "call_1".to_string(),
        tool_name: "search".to_string(),
        arguments: serde_json::json!({ "query": "rust" }),
        provider_executed: Some(true),
        dynamic: Some(false),
        invalid: Some(false),
        error: None,
        title: Some("Search".to_string()),
        provider_options,
        provider_metadata: Some(provider_metadata.clone()),
    }]));
    response.provider_metadata = Some(provider_metadata);

    let response_value = serde_json::to_value(&response).expect("serialize ChatResponse fixture");
    assert_eq!(
        response_value["content"]["MultiModal"][0]["providerMetadata"]["openai"]["itemId"],
        serde_json::json!("msg_1")
    );
    assert_eq!(
        response_value["provider_metadata"]["openai"]["responseId"],
        serde_json::json!("resp_1")
    );

    let response_roundtrip: ChatResponse =
        serde_json::from_value(response_value.clone()).expect("deserialize ChatResponse fixture");
    assert_eq!(
        serde_json::to_value(&response_roundtrip).expect("reserialize ChatResponse fixture"),
        response_value,
        "ChatResponse serde payload must remain stable before root ContentPart movement"
    );
}

fn assert_json_has_no_key(value: &serde_json::Value, key: &str) {
    match value {
        serde_json::Value::Object(map) => {
            assert!(
                !map.contains_key(key),
                "projected generated output must not contain `{key}`"
            );
            for nested in map.values() {
                assert_json_has_no_key(nested, key);
            }
        }
        serde_json::Value::Array(values) => {
            for nested in values {
                assert_json_has_no_key(nested, key);
            }
        }
        _ => {}
    }
}

#[test]
fn prompt_projection_rejects_response_side_provider_metadata_on_legacy_content_parts() {
    let provider_metadata = ProviderMetadataMap::from([(
        "openai".to_string(),
        serde_json::json!({ "responseId": "resp_123" }),
    )]);

    let cases = [
        (
            "text",
            MessageRole::User,
            ContentPart::Text {
                text: "hello".to_string(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "image",
            MessageRole::User,
            ContentPart::Image {
                source: FilePartSource::url("https://example.com/image.png"),
                media_type: Some("image/png".to_string()),
                detail: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "file",
            MessageRole::User,
            ContentPart::File {
                source: FilePartSource::url("https://example.com/file.pdf"),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "reasoning",
            MessageRole::Assistant,
            ContentPart::Reasoning {
                text: "thinking".to_string(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "custom",
            MessageRole::Assistant,
            ContentPart::Custom {
                kind: "openai.compaction".to_string(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "tool-call",
            MessageRole::Assistant,
            ContentPart::ToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                arguments: serde_json::json!({ "query": "rust" }),
                provider_executed: None,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata.clone()),
            },
        ),
        (
            "tool-result",
            MessageRole::Tool,
            ContentPart::tool_result_text("call_1", "search", "ok")
                .with_provider_metadata_for_test(provider_metadata.clone()),
        ),
    ];

    for (part_type, role, part) in cases {
        let message = ChatMessage {
            role,
            content: MessageContent::MultiModal(vec![part]),
            provider_options: ProviderOptionsMap::default(),
            metadata: Default::default(),
        };

        let err = project_chat_message_to_prompt_message(&message)
            .expect_err("response-side provider metadata must not project into prompt messages");
        assert_eq!(
            err,
            ModelMessageConversionError::UnsupportedContentPart {
                context: "prompt",
                part_type,
                reason: "provider metadata is response-side only",
            }
        );
    }
}

#[test]
fn prompt_projection_to_legacy_content_parts_never_emits_response_provider_metadata() {
    let mut provider_options = ProviderOptionsMap::default();
    provider_options.insert(
        "openai",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );

    let messages = vec![
        ModelMessage::User(UserModelMessage::new(UserContent::parts(vec![
            UserContentPart::Text(
                TextPart::new("hello").with_provider_options_map(provider_options.clone()),
            ),
        ]))),
        ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::parts(vec![
            AssistantContentPart::Text(
                TextPart::new("thinking").with_provider_options_map(provider_options.clone()),
            ),
            AssistantContentPart::ToolCall(
                ToolCallPart::new("call_1", "search", serde_json::json!({ "query": "rust" }))
                    .with_provider_options_map(provider_options.clone()),
            ),
        ]))),
        ModelMessage::Tool(ToolModelMessage::new(vec![ToolContentPart::ToolResult(
            ToolResultPart::new("call_1", "search", ToolResultOutput::text("ok"))
                .with_provider_options_map(provider_options.clone()),
        )])),
    ];

    let chat_messages = project_prompt_messages_to_chat_messages(&messages);

    for message in &chat_messages {
        let MessageContent::MultiModal(parts) = &message.content else {
            continue;
        };

        for part in parts {
            assert_content_part_has_no_provider_metadata(part);
            assert_eq!(
                part.provider_options(),
                Some(&provider_options),
                "prompt projection should preserve request-side providerOptions"
            );
        }
    }
}

fn assert_content_part_has_no_provider_metadata(part: &ContentPart) {
    match part {
        ContentPart::Text {
            provider_metadata, ..
        }
        | ContentPart::Image {
            provider_metadata, ..
        }
        | ContentPart::Audio {
            provider_metadata, ..
        }
        | ContentPart::File {
            provider_metadata, ..
        }
        | ContentPart::ReasoningFile {
            provider_metadata, ..
        }
        | ContentPart::Custom {
            provider_metadata, ..
        }
        | ContentPart::Source {
            provider_metadata, ..
        }
        | ContentPart::ToolCall {
            provider_metadata, ..
        }
        | ContentPart::ToolApprovalRequest {
            provider_metadata, ..
        }
        | ContentPart::ToolResult {
            provider_metadata, ..
        }
        | ContentPart::Reasoning {
            provider_metadata, ..
        } => assert!(
            provider_metadata.is_none(),
            "prompt projection must not emit response-side providerMetadata"
        ),
        ContentPart::ToolApprovalResponse { .. } => {}
    }
}

trait ContentPartProviderMetadataTestExt {
    fn with_provider_metadata_for_test(self, provider_metadata: ProviderMetadataMap) -> Self;
}

impl ContentPartProviderMetadataTestExt for ContentPart {
    fn with_provider_metadata_for_test(self, provider_metadata: ProviderMetadataMap) -> Self {
        match self {
            Self::ToolResult {
                tool_call_id,
                tool_name,
                output,
                input,
                provider_executed,
                dynamic,
                preliminary,
                title,
                provider_options,
                ..
            } => Self::ToolResult {
                tool_call_id,
                tool_name,
                output,
                input,
                provider_executed,
                dynamic,
                preliminary,
                title,
                provider_options,
                provider_metadata: Some(provider_metadata),
            },
            _ => self,
        }
    }
}
