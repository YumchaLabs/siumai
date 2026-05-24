use std::collections::HashMap;

use super::{
    ConvertUiMessagesOptions, SafeValidateUiMessagesResult, ValidateUiMessagesSchemaOptions,
    convert_to_model_messages, convert_to_model_messages_with,
    convert_to_model_messages_with_tooling, safe_validate_ui_messages,
    safe_validate_ui_messages_with_schemas, validate_ui_messages,
    validate_ui_messages_with_schemas,
};
use crate::tooling::{ExecutableTool, ExecutableTools};
use crate::types::{
    ChatMessage, ContentPart, MessageContent, ProviderOptionsMap, ProviderReference, Tool,
    ToolResultOutput, UiFilePart, UiMessage, UiMessagePart, UiToolApproval, UiToolPart,
    UiToolPartState,
};

fn provider_map(provider_id: &str, value: serde_json::Value) -> ProviderOptionsMap {
    let mut map = ProviderOptionsMap::default();
    map.insert(provider_id, value);
    map
}

fn multimodal_parts(message: &ChatMessage) -> &[ContentPart] {
    let MessageContent::MultiModal(parts) = &message.content else {
        panic!("expected multimodal content");
    };
    parts
}

fn assert_request_provider_options_only(part: &ContentPart, expected: &ProviderOptionsMap) {
    assert_eq!(part.provider_options(), Some(expected));

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
        } => assert!(provider_metadata.is_none()),
        ContentPart::ToolApprovalResponse { .. } => {}
        ContentPart::Source { .. } => panic!("source parts are response-side only"),
    }
}

#[test]
fn ui_conversion_centralizes_legacy_request_content_constructors() {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ui/conversion.rs"));
    let helper_start = source
        .find("fn ui_request_text_part")
        .expect("UI request content adapter helpers should exist");
    let helper_end = source
        .find("fn convert_text_part")
        .expect("UI request content adapter helpers should precede UI part conversion");
    assert!(helper_start < helper_end);

    let mut outside_helper_provider_metadata_lines = Vec::new();
    let mut offset = 0usize;
    for (index, line) in source.lines().enumerate() {
        let line_start = offset;
        offset += line.len() + 1;
        if line.trim() == "provider_metadata: None,"
            && !(helper_start..helper_end).contains(&line_start)
        {
            outside_helper_provider_metadata_lines.push((index + 1, line.trim().to_string()));
        }
    }

    assert_eq!(
        outside_helper_provider_metadata_lines.len(),
        1,
        "UI request ContentPart provider_metadata construction should stay centralized; the only outside-helper occurrence is the plain-text collapse match: {outside_helper_provider_metadata_lines:?}"
    );
    assert_eq!(
        outside_helper_provider_metadata_lines[0].1,
        "provider_metadata: None,"
    );
}

#[test]
fn validate_rejects_missing_output_error_text() {
    let message = UiMessage::assistant(
        "msg_1",
        vec![UiMessagePart::Tool(UiToolPart::named(
            "search",
            "call_1",
            UiToolPartState::OutputError,
        ))],
    );

    let err = validate_ui_messages(&[message]).expect_err("validation should fail");
    assert!(format!("{err}").contains("errorText is required"));
}

#[test]
fn safe_validate_returns_ai_sdk_style_result_union() {
    let ok =
        safe_validate_ui_messages(&[UiMessage::user("user", vec![UiMessagePart::text("hello")])]);
    assert!(ok.success());
    assert_eq!(ok.data().expect("validated messages")[0].id, "user");
    assert!(ok.clone().into_result().is_ok());

    let failed = safe_validate_ui_messages(&[UiMessage::user("empty", Vec::new())]);
    assert!(!failed.success());
    assert!(matches!(
        failed.error(),
        Some(crate::ui::UiMessageError::EmptyMessageParts { message_id })
            if message_id == "empty"
    ));
    assert!(matches!(
        failed.into_result(),
        Err(crate::ui::UiMessageError::EmptyMessageParts { message_id })
            if message_id == "empty"
    ));
}

#[test]
fn safe_schema_validate_preserves_schema_errors() {
    let mut data_schemas = HashMap::new();
    data_schemas.insert("other".to_string(), serde_json::json!({ "kind": "other" }));

    let result = safe_validate_ui_messages_with_schemas(
        &[UiMessage::user(
            "user",
            vec![UiMessagePart::data(
                "weather",
                serde_json::json!({ "city": "Tokyo" }),
            )],
        )],
        ValidateUiMessagesSchemaOptions {
            metadata_schema: None,
            data_schemas: Some(&data_schemas),
        },
        None,
        &|_schema: &serde_json::Value, _instance: &serde_json::Value| Ok(()),
    );

    let SafeValidateUiMessagesResult::Failure { error } = result else {
        panic!("schema validation should fail");
    };
    assert!(format!("{error}").contains("no schema found for data part `weather`"));
}

#[test]
fn convert_merges_system_text_parts_and_provider_metadata() {
    let mut provider_metadata = ProviderOptionsMap::default();
    provider_metadata.insert(
        "provider-a",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );

    let mut text_part = crate::types::UiTextPart::new("sys-");
    text_part.provider_metadata = provider_metadata.clone();

    let messages = vec![UiMessage::system(
        "sys",
        vec![
            UiMessagePart::Text(text_part),
            UiMessagePart::text("prompt"),
        ],
    )];

    let converted = convert_to_model_messages(&messages).expect("convert ok");
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].content_text(), Some("sys-prompt"));
    assert_eq!(converted[0].provider_options(), &provider_metadata);
}

#[test]
fn convert_normalizes_ui_part_provider_metadata_to_request_provider_options() {
    let text_provider_metadata = provider_map(
        "provider-a",
        serde_json::json!({ "cacheControl": { "type": "ephemeral" } }),
    );
    let reasoning_provider_metadata = provider_map(
        "provider-b",
        serde_json::json!({ "thoughtSignature": "thought_sig_1" }),
    );
    let custom_provider_metadata =
        provider_map("provider-a", serde_json::json!({ "itemId": "custom_1" }));
    let file_provider_metadata = provider_map(
        "provider-c",
        serde_json::json!({ "cacheControl": "ephemeral" }),
    );
    let reasoning_file_provider_metadata = provider_map(
        "provider-b",
        serde_json::json!({ "fileId": "reasoning_file_1" }),
    );

    let mut text = crate::types::UiTextPart::new("draft");
    text.provider_metadata = text_provider_metadata.clone();

    let mut reasoning = crate::types::UiReasoningPart::new("thinking");
    reasoning.provider_metadata = reasoning_provider_metadata.clone();

    let mut custom = crate::types::UiCustomPart::new("provider-a.compaction");
    custom.provider_metadata = custom_provider_metadata.clone();

    let mut file = UiFilePart::new("https://example.com/file.pdf", "application/pdf");
    file.provider_metadata = file_provider_metadata.clone();

    let mut reasoning_file =
        crate::types::UiReasoningFilePart::new("https://example.com/reasoning.txt", "text/plain");
    reasoning_file.provider_metadata = reasoning_file_provider_metadata.clone();

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![
            UiMessagePart::Text(text),
            UiMessagePart::Reasoning(reasoning),
            UiMessagePart::Custom(custom),
            UiMessagePart::File(file),
            UiMessagePart::ReasoningFile(reasoning_file),
        ],
    )])
    .expect("convert ok");

    let parts = multimodal_parts(&converted[0]);
    assert_eq!(parts.len(), 5);
    assert_request_provider_options_only(&parts[0], &text_provider_metadata);
    assert_request_provider_options_only(&parts[1], &reasoning_provider_metadata);
    assert_request_provider_options_only(&parts[2], &custom_provider_metadata);
    assert_request_provider_options_only(&parts[3], &file_provider_metadata);
    assert_request_provider_options_only(&parts[4], &reasoning_file_provider_metadata);
}

#[test]
fn convert_normalizes_ui_tool_provider_metadata_to_request_provider_options() {
    let call_provider_metadata =
        provider_map("provider-a", serde_json::json!({ "itemId": "call_1" }));
    let result_provider_metadata =
        provider_map("provider-a", serde_json::json!({ "itemId": "result_1" }));

    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.output = Some(serde_json::json!({ "forecast": "sunny" }));
    tool.call_provider_metadata = call_provider_metadata.clone();
    tool.result_provider_metadata = result_provider_metadata.clone();

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect("convert ok");

    let assistant_parts = multimodal_parts(&converted[0]);
    assert_request_provider_options_only(&assistant_parts[0], &call_provider_metadata);

    let tool_parts = multimodal_parts(&converted[1]);
    assert_request_provider_options_only(&tool_parts[0], &result_provider_metadata);
}

#[test]
fn convert_maps_user_provider_reference_file() {
    let mut file_part = UiFilePart::new("https://example.com/ignored.pdf", "application/pdf");
    file_part.provider_reference = Some(ProviderReference::single("provider-a", "file_123"));

    let messages = vec![UiMessage::user(
        "user",
        vec![UiMessagePart::File(file_part)],
    )];

    let converted = convert_to_model_messages(&messages).expect("convert ok");
    let ChatMessage { content, .. } = &converted[0];
    let crate::types::MessageContent::MultiModal(parts) = content else {
        panic!("expected multimodal user content");
    };
    let ContentPart::File { source, .. } = &parts[0] else {
        panic!("expected file part");
    };
    assert!(source.is_provider_reference());
    assert_eq!(
        source
            .as_provider_reference()
            .and_then(|reference| reference.get("provider-a")),
        Some("file_123")
    );
}

#[test]
fn convert_splits_assistant_tool_blocks() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.output = Some(serde_json::json!({ "temp": 18 }));

    let messages = vec![UiMessage::assistant(
        "assistant",
        vec![
            UiMessagePart::text("Before "),
            UiMessagePart::Tool(tool),
            UiMessagePart::step_start(),
            UiMessagePart::text("After"),
        ],
    )];

    let converted = convert_to_model_messages(&messages).expect("convert ok");
    assert_eq!(converted.len(), 3);
    assert_eq!(converted[0].role, crate::types::MessageRole::Assistant);
    assert_eq!(converted[1].role, crate::types::MessageRole::Tool);
    assert_eq!(converted[2].role, crate::types::MessageRole::Assistant);
}

#[test]
fn convert_ignores_incomplete_tool_calls_when_requested() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::InputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));

    let messages = vec![UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool), UiMessagePart::text("done")],
    )];

    let converted = convert_to_model_messages_with(
        &messages,
        ConvertUiMessagesOptions {
            ignore_incomplete_tool_calls: true,
        },
        |_part| Ok(None),
    )
    .expect("convert ok");

    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].content_text(), Some("done"));
}

#[test]
fn convert_data_parts_with_callback() {
    let messages = vec![UiMessage::user(
        "user",
        vec![UiMessagePart::data(
            "weather",
            serde_json::json!({ "city": "Tokyo" }),
        )],
    )];

    let converted =
        convert_to_model_messages_with(&messages, ConvertUiMessagesOptions::default(), |part| {
            Ok(Some(ContentPart::text(format!(
                "city={}",
                part.data["city"]
            ))))
        })
        .expect("convert ok");

    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].content_text(), Some("city=\"Tokyo\""));
}

#[test]
fn convert_tool_approval_response_preserves_provider_executed() {
    let mut tool = UiToolPart::dynamic("shell", "call_1", UiToolPartState::ApprovalResponded);
    tool.input = Some(serde_json::json!({ "command": "ls" }));
    tool.provider_executed = Some(true);
    tool.approval = Some(UiToolApproval {
        id: "approval_1".to_string(),
        approved: Some(true),
        reason: Some("ok".to_string()),
    });

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect("convert ok");

    let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
        panic!("expected tool message");
    };
    let ContentPart::ToolApprovalResponse {
        provider_executed, ..
    } = &parts[0]
    else {
        panic!("expected tool approval response");
    };
    assert_eq!(*provider_executed, Some(true));
}

#[test]
fn explicit_tool_result_output_shape_roundtrips_from_ui_output() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.output = Some(serde_json::json!({
        "type": "content",
        "value": [
            { "type": "text", "text": "sunny" }
        ]
    }));

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect("convert ok");

    let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
        panic!("expected tool message");
    };
    let ContentPart::ToolResult { output, .. } = &parts[0] else {
        panic!("expected tool result");
    };
    assert_eq!(
        output,
        &ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text("sunny")])
    );
}

#[test]
fn provider_executed_output_error_uses_error_json() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputError);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.error_text = Some("boom".to_string());
    tool.provider_executed = Some(true);

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect("convert ok");

    let crate::types::MessageContent::MultiModal(parts) = &converted[0].content else {
        panic!("expected assistant multimodal content");
    };
    let ContentPart::ToolResult { output, .. } = &parts[1] else {
        panic!("expected assistant tool result");
    };
    assert_eq!(
        output,
        &ToolResultOutput::error_json(serde_json::json!("boom"))
    );
}

#[test]
fn local_output_error_uses_error_text() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputError);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.error_text = Some("boom".to_string());

    let converted = convert_to_model_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect("convert ok");

    let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
        panic!("expected tool multimodal content");
    };
    let ContentPart::ToolResult { output, .. } = &parts[0] else {
        panic!("expected tool result");
    };
    assert_eq!(output, &ToolResultOutput::error_text("boom"));
}

#[test]
fn runtime_tool_mapper_overrides_default_ui_tool_output_conversion() {
    let tools = ExecutableTools::from_tools([ExecutableTool::new(Tool::function(
        "weather",
        "Weather tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_to_model_output_fn(|ctx| {
        Ok(ToolResultOutput::content(vec![
            crate::types::ToolResultContentPart::text(format!(
                "{}:{}",
                ctx.tool_call_id, ctx.output["temp"]
            )),
        ]))
    })]);

    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.output = Some(serde_json::json!({ "temp": 18 }));

    let converted = convert_to_model_messages_with_tooling(
        &[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )],
        ConvertUiMessagesOptions::default(),
        &tools,
        |_part| Ok(None),
    )
    .expect("convert ok");

    let crate::types::MessageContent::MultiModal(parts) = &converted[1].content else {
        panic!("expected tool multimodal content");
    };
    let ContentPart::ToolResult { output, .. } = &parts[0] else {
        panic!("expected tool result");
    };
    assert_eq!(
        output,
        &ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text("call_1:18")])
    );
}

#[test]
fn schema_validation_rejects_invalid_tool_output_against_tool_schema() {
    let tools =
        ExecutableTools::from_tools([ExecutableTool::new(Tool::function_with_output_schema(
            "weather",
            "Weather tool",
            serde_json::json!({ "kind": "input" }),
            serde_json::json!({ "kind": "output" }),
        ))]);

    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.output = Some(serde_json::json!({ "temp": 18 }));

    let err = validate_ui_messages_with_schemas(
        &[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool)],
        )],
        ValidateUiMessagesSchemaOptions::default(),
        Some(&tools),
        &|schema: &serde_json::Value, instance: &serde_json::Value| match schema["kind"].as_str() {
            Some("input")
                if instance
                    .get("city")
                    .and_then(|value| value.as_str())
                    .is_some() =>
            {
                Ok(())
            }
            Some("output")
                if instance
                    .get("forecast")
                    .and_then(|value| value.as_str())
                    .is_some() =>
            {
                Ok(())
            }
            Some(kind) => Err(format!("expected valid {kind} payload")),
            None => Ok(()),
        },
    )
    .expect_err("schema validation should fail");

    assert!(format!("{err}").contains("output failed schema validation"));
}

#[test]
fn schema_validation_rejects_data_parts_without_matching_schema() {
    let mut data_schemas = HashMap::new();
    data_schemas.insert("other".to_string(), serde_json::json!({ "kind": "other" }));

    let err = validate_ui_messages_with_schemas(
        &[UiMessage::user(
            "user",
            vec![UiMessagePart::data(
                "weather",
                serde_json::json!({ "city": "Tokyo" }),
            )],
        )],
        ValidateUiMessagesSchemaOptions {
            metadata_schema: None,
            data_schemas: Some(&data_schemas),
        },
        None,
        &|_schema: &serde_json::Value, _instance: &serde_json::Value| Ok(()),
    )
    .expect_err("missing data schema should fail");

    assert!(format!("{err}").contains("no schema found for data part `weather`"));
}
