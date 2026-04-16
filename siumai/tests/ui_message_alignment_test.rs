use siumai::tooling::{ExecutableTool, ExecutableTools};
use siumai::types::{
    ContentPart, Tool, ToolResultOutput, UiMessage, UiMessagePart, UiToolApproval,
    UiToolApprovedApproval, UiToolInvocation, UiToolInvocationState, UiToolKind, UiToolPart,
    UiToolPartState,
};
use siumai::ui::{
    ConvertUiMessagesOptions, convert_to_chat_request, convert_to_model_messages_with,
    convert_to_model_messages_with_tooling, validate_ui_messages,
};

#[test]
fn facade_converts_ui_messages_into_chat_request() {
    let request = convert_to_chat_request(&[
        UiMessage::system("sys", vec![UiMessagePart::text("You are helpful.")]),
        UiMessage::user("user", vec![UiMessagePart::text("Hello")]),
    ])
    .expect("convert chat request");

    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages[0].content_text(), Some("You are helpful."));
    assert_eq!(request.messages[1].content_text(), Some("Hello"));
}

#[test]
fn facade_ignores_incomplete_tool_calls_when_requested() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::InputAvailable);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));

    let converted = convert_to_model_messages_with(
        &[UiMessage::assistant(
            "assistant",
            vec![UiMessagePart::Tool(tool), UiMessagePart::text("done")],
        )],
        ConvertUiMessagesOptions {
            ignore_incomplete_tool_calls: true,
        },
        |_part| Ok(None),
    )
    .expect("convert messages");

    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].content_text(), Some("done"));
}

#[test]
fn facade_converts_data_parts_with_callback() {
    let converted = convert_to_model_messages_with(
        &[UiMessage::user(
            "user",
            vec![UiMessagePart::data(
                "weather",
                serde_json::json!({ "city": "Tokyo" }),
            )],
        )],
        ConvertUiMessagesOptions::default(),
        |part| {
            Ok(Some(ContentPart::text(format!(
                "city={}",
                part.data["city"]
            ))))
        },
    )
    .expect("convert messages");

    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].content_text(), Some("city=\"Tokyo\""));
}

#[test]
fn facade_rejects_approval_requested_with_decision() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::ApprovalRequested);
    tool.input = Some(serde_json::json!({ "city": "Tokyo" }));
    tool.approval = Some(UiToolApproval {
        id: "approval_1".to_string(),
        approved: Some(true),
        reason: None,
    });

    let err = validate_ui_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect_err("validation should fail");

    assert!(format!("{err}").contains("must not include approved"));
}

#[test]
fn facade_rejects_output_error_with_output_payload() {
    let mut tool = UiToolPart::named("weather", "call_1", UiToolPartState::OutputError);
    tool.error_text = Some("boom".to_string());
    tool.output = Some(serde_json::json!({ "temp": 18 }));

    let err = validate_ui_messages(&[UiMessage::assistant(
        "assistant",
        vec![UiMessagePart::Tool(tool)],
    )])
    .expect_err("validation should fail");

    assert!(format!("{err}").contains("output is not allowed for output-error"));
}

#[test]
fn facade_supports_runtime_tool_output_mapping() {
    let tools = ExecutableTools::from_tools([ExecutableTool::new(Tool::function(
        "weather",
        "Weather tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_to_model_output_fn(|ctx| {
        Ok(ToolResultOutput::content(vec![
            siumai::types::ToolResultContentPart::text(format!(
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
    .expect("convert messages");

    let siumai::types::MessageContent::MultiModal(parts) = &converted[1].content else {
        panic!("expected tool multimodal content");
    };
    let ContentPart::ToolResult { output, .. } = &parts[0] else {
        panic!("expected tool result");
    };
    assert_eq!(
        output,
        &ToolResultOutput::content(vec![siumai::types::ToolResultContentPart::text(
            "call_1:18"
        )])
    );
}

#[test]
fn facade_exposes_state_discriminated_ui_tool_invocation_overlay() {
    let invocation = UiToolInvocation {
        tool_call_id: "call_1".to_string(),
        title: Some("Weather".to_string()),
        provider_executed: Some(true),
        call_provider_metadata: Default::default(),
        state: UiToolInvocationState::OutputAvailable {
            input: serde_json::json!({ "city": "Tokyo" }),
            output: serde_json::json!({ "temp": 18 }),
            result_provider_metadata: Default::default(),
            preliminary: Some(true),
            approval: Some(UiToolApprovedApproval {
                id: "approval_1".to_string(),
                reason: Some("ok".to_string()),
            }),
        },
    };

    let tool = UiToolPart::from_invocation(
        UiToolKind::Static {
            tool_name: "weather".to_string(),
        },
        invocation.clone(),
    );

    assert_eq!(tool.invocation().expect("typed invocation"), invocation);
}
