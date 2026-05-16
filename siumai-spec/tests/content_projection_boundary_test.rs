use std::fs;
use std::path::{Path, PathBuf};

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

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_source(relative_path: &str) -> String {
    fs::read_to_string(crate_root().join(relative_path))
        .unwrap_or_else(|error| panic!("failed to read {relative_path}: {error}"))
}

fn content_part_variant_block(source: &str, variant: &str) -> String {
    let variants = [
        "Text",
        "Image",
        "Audio",
        "File",
        "ReasoningFile",
        "Custom",
        "Source",
        "ToolCall",
        "ToolApprovalResponse",
        "ToolApprovalRequest",
        "ToolResult",
        "Reasoning",
    ];
    let enum_start = source
        .find("pub enum ContentPart")
        .expect("ContentPart enum should exist");
    let marker = format!("    {variant} {{");
    let start = enum_start
        + source[enum_start..]
            .find(&marker)
            .unwrap_or_else(|| panic!("ContentPart::{variant} should exist"));

    let mut end = source.len();
    for next_variant in variants {
        if next_variant == variant {
            continue;
        }

        let next_marker = format!("    {next_variant} {{");
        if let Some(relative_next) = source[start + marker.len()..].find(&next_marker) {
            let candidate = start + marker.len() + relative_next;
            end = end.min(candidate);
        }
    }

    source[start..end].to_string()
}

fn rust_item_block(source: &str, item_name: &str) -> String {
    let start = find_rust_item_start(source, item_name)
        .unwrap_or_else(|| panic!("Rust item `{item_name}` should exist"));
    let open_brace = start
        + source[start..]
            .find('{')
            .unwrap_or_else(|| panic!("Rust item `{item_name}` should have a body"));

    let mut depth = 0_u32;
    for (offset, ch) in source[open_brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = open_brace + offset + ch.len_utf8();
                    return source[start..end].to_string();
                }
            }
            _ => {}
        }
    }

    panic!("Rust item `{item_name}` body should be balanced");
}

fn find_rust_item_start(source: &str, item_name: &str) -> Option<usize> {
    for keyword in ["pub struct", "pub enum"] {
        let needle = format!("{keyword} {item_name}");
        let mut offset = 0;

        while let Some(relative_start) = source[offset..].find(&needle) {
            let start = offset + relative_start;
            let after_name = start + needle.len();
            let has_exact_name_boundary = source[after_name..]
                .chars()
                .next()
                .is_some_and(|ch| ch == ' ' || ch == '<' || ch == '{');

            if has_exact_name_boundary {
                return Some(start);
            }

            offset = after_name;
        }
    }

    None
}

#[test]
fn legacy_content_part_dual_provider_maps_stay_explicitly_audited() {
    let source = read_source("src/types/chat/content/part.rs");
    let all_variants = [
        "Text",
        "Image",
        "Audio",
        "File",
        "ReasoningFile",
        "Custom",
        "Source",
        "ToolCall",
        "ToolApprovalResponse",
        "ToolApprovalRequest",
        "ToolResult",
        "Reasoning",
    ];
    let audited_dual_use_variants = [
        "Text",
        "Image",
        "Audio",
        "File",
        "ReasoningFile",
        "Custom",
        "ToolCall",
        "ToolApprovalRequest",
        "ToolResult",
        "Reasoning",
    ];

    for variant in all_variants {
        let block = content_part_variant_block(&source, variant);
        let has_provider_options = block.contains("provider_options: ProviderOptionsMap");
        let has_provider_metadata =
            block.contains("provider_metadata: Option<ProviderMetadataMap>");
        let is_audited_dual_use = audited_dual_use_variants.contains(&variant);

        assert!(
            !has_provider_options || !has_provider_metadata || is_audited_dual_use,
            "ContentPart::{variant} must not become a new dual providerOptions/providerMetadata carrier without updating the Track C audit"
        );
    }

    for variant in audited_dual_use_variants {
        let block = content_part_variant_block(&source, variant);
        assert!(
            block.contains("provider_options: ProviderOptionsMap"),
            "ContentPart::{variant} should keep the audited request-side providerOptions field until a replacement projection is shipped"
        );
        assert!(
            block.contains("provider_metadata: Option<ProviderMetadataMap>"),
            "ContentPart::{variant} should keep the audited response-side providerMetadata field until a replacement projection is shipped"
        );
    }

    let source_block = content_part_variant_block(&source, "Source");
    assert!(
        !source_block.contains("provider_options: ProviderOptionsMap"),
        "ContentPart::Source is response-side citation data and must not grow request-side providerOptions"
    );
    assert!(
        source_block.contains("provider_metadata: Option<ProviderMetadataMap>"),
        "ContentPart::Source should keep response-side providerMetadata"
    );

    let approval_response_block = content_part_variant_block(&source, "ToolApprovalResponse");
    assert!(
        approval_response_block.contains("provider_options: ProviderOptionsMap"),
        "ContentPart::ToolApprovalResponse should keep request-side providerOptions"
    );
    assert!(
        !approval_response_block.contains("provider_metadata: Option<ProviderMetadataMap>"),
        "ContentPart::ToolApprovalResponse must not grow response-side providerMetadata without an explicit audit"
    );
}

#[test]
fn ai_sdk_v4_prompt_and_generated_content_keep_provider_maps_directional() {
    let prompt_source = read_source("src/types/ai_sdk/language_model_v4/prompt.rs");
    assert!(
        prompt_source.contains("provider_options: ProviderOptionsMap"),
        "AI SDK V4 prompt projections should carry request-side providerOptions"
    );
    for forbidden in ["provider_metadata", "ProviderMetadata", "providerMetadata"] {
        assert!(
            !prompt_source.contains(forbidden),
            "AI SDK V4 prompt projections must not carry response-side provider metadata fragment `{forbidden}`"
        );
    }

    let content_source = read_source("src/types/ai_sdk/language_model_v4/content.rs");
    assert!(
        content_source.contains("provider_metadata: Option<ProviderMetadata>"),
        "AI SDK V4 generated content projections should carry response-side providerMetadata"
    );
    for forbidden in ["provider_options", "ProviderOptions", "providerOptions"] {
        assert!(
            !content_source.contains(forbidden),
            "AI SDK V4 generated content projections must not carry request-side provider options fragment `{forbidden}`"
        );
    }
}

#[test]
fn non_v4_prompt_projection_types_keep_provider_maps_directional() {
    let source = read_source("src/types/prompt.rs");
    let prompt_projection_items = [
        "TextPart",
        "ImagePart",
        "FilePart",
        "ReasoningPart",
        "CustomPart",
        "ReasoningFilePart",
        "ToolCallPart",
        "ToolResultPart",
        "ToolApprovalRequest",
        "ToolApprovalResponse",
        "UserContentPart",
        "AssistantContentPart",
        "ToolContentPart",
        "UserContent",
        "AssistantContent",
        "SystemModelMessage",
        "UserModelMessage",
        "AssistantModelMessage",
        "ToolModelMessage",
        "ModelMessage",
        "Prompt",
        "StandardizedPrompt",
    ];

    for item in prompt_projection_items {
        let block = rust_item_block(&source, item);
        for forbidden in ["provider_metadata", "ProviderMetadata", "providerMetadata"] {
            assert!(
                !block.contains(forbidden),
                "non-V4 prompt projection item `{item}` must not carry response-side provider metadata fragment `{forbidden}`"
            );
        }
    }

    for item in [
        "TextPart",
        "ImagePart",
        "FilePart",
        "ReasoningPart",
        "CustomPart",
        "ReasoningFilePart",
        "ToolCallPart",
        "ToolResultPart",
        "ToolApprovalResponse",
        "SystemModelMessage",
        "UserModelMessage",
        "AssistantModelMessage",
        "ToolModelMessage",
    ] {
        let block = rust_item_block(&source, item);
        assert!(
            block.contains("provider_options: ProviderOptionsMap"),
            "non-V4 prompt projection item `{item}` should carry request-side providerOptions"
        );
    }

    assert!(
        source.contains("pub fn project_chat_message_to_prompt_message"),
        "non-V4 prompt projection should expose a named stable ChatMessage -> ModelMessage helper"
    );
    assert!(
        source.contains("pub fn project_prompt_message_to_chat_message"),
        "non-V4 prompt projection should expose a named ModelMessage -> stable ChatMessage helper"
    );
}

#[test]
fn non_v4_generate_text_output_projection_keeps_provider_maps_directional() {
    let generate_text_source = read_source("src/types/ai_sdk/generate_text.rs");
    let output_parts_source = read_source("src/types/ai_sdk/output_parts.rs");

    for (source, item) in [
        (&generate_text_source, "GenerateTextContentPart"),
        (&generate_text_source, "GenerateTextReasoningPart"),
        (&output_parts_source, "TextOutput"),
        (&output_parts_source, "CustomOutput"),
        (&output_parts_source, "FileOutput"),
        (&output_parts_source, "ReasoningOutput"),
        (&output_parts_source, "ReasoningFileOutput"),
        (&output_parts_source, "ToolCall"),
        (&output_parts_source, "ToolResult"),
        (&output_parts_source, "ToolError"),
        (&output_parts_source, "ToolOutput"),
        (&output_parts_source, "ToolOutputDenied"),
        (&output_parts_source, "ToolApprovalRequestOutput"),
        (&output_parts_source, "ToolApprovalResponseOutput"),
    ] {
        let block = rust_item_block(source, item);
        for forbidden in ["provider_options", "ProviderOptions", "providerOptions"] {
            assert!(
                !block.contains(forbidden),
                "non-V4 generated output item `{item}` must not carry request-side provider options fragment `{forbidden}`"
            );
        }
    }

    for item in [
        "TextOutput",
        "CustomOutput",
        "FileOutput",
        "ReasoningOutput",
        "ReasoningFileOutput",
        "ToolCall",
        "ToolResult",
        "ToolError",
    ] {
        let block = rust_item_block(&output_parts_source, item);
        assert!(
            block.contains("provider_metadata: Option<ProviderMetadata>"),
            "non-V4 generated output item `{item}` should carry response-side providerMetadata"
        );
    }

    assert!(
        generate_text_source
            .contains("pub fn project_response_content_part_to_generate_text_content_part"),
        "non-V4 response projection should expose a named ContentPart -> GenerateTextContentPart helper"
    );
    assert!(
        generate_text_source
            .contains("pub fn project_chat_response_to_generate_text_content_parts"),
        "non-V4 response projection should expose a named ChatResponse -> GenerateTextContentPart helper"
    );
}

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
