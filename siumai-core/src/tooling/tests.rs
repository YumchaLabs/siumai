use super::*;
use std::sync::Arc;

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::types::{Context, Tool, ToolResultOutput};

#[tokio::test]
async fn typed_tool_parses_args_and_serializes_output() {
    #[derive(Deserialize)]
    struct Args {
        x: i64,
        y: i64,
    }

    #[derive(Serialize)]
    struct Out {
        sum: i64,
    }

    let tool = ExecutableTool::typed_function::<Args, Out, _, _>(
        "add",
        "Add two integers",
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer" },
                "y": { "type": "integer" }
            },
            "required": ["x", "y"]
        }),
        |args| async move {
            Ok(Out {
                sum: args.x + args.y,
            })
        },
    );

    let out = tool
        .execute_json(serde_json::json!({ "x": 1, "y": 2 }))
        .await
        .unwrap();

    assert_eq!(out, serde_json::json!({ "sum": 3 }));
}

#[tokio::test]
async fn tool_set_executes_by_name() {
    let mut tools = ExecutableTools::new();
    tools.insert(ExecutableTool::function(
        "echo",
        "Echo input",
        serde_json::json!({"type":"object"}),
        |v| async move { Ok(v) },
    ));

    let out = tools
        .execute("echo", serde_json::json!({"a":1}))
        .await
        .unwrap();
    assert_eq!(out, serde_json::json!({"a":1}));
}

#[tokio::test]
async fn execute_tool_normalizes_streaming_outputs() {
    let tool = tool(Tool::function(
        "search",
        "Search tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_execute_stream_fn(|_args, options| {
        assert_eq!(options.tool_call_id, "call_1");
        Box::pin(futures::stream::iter(vec![
            Ok(serde_json::json!({ "progress": 50 })),
            Ok(serde_json::json!({ "progress": 100 })),
        ]))
    });

    let results = execute_tool(
        &tool,
        serde_json::json!({ "q": "rust" }),
        ToolExecutionOptions::new("call_1"),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].as_ref().unwrap(),
        &ToolExecutionResult::preliminary(serde_json::json!({ "progress": 50 }))
    );
    assert_eq!(
        results[1].as_ref().unwrap(),
        &ToolExecutionResult::preliminary(serde_json::json!({ "progress": 100 }))
    );
    assert_eq!(
        results[2].as_ref().unwrap(),
        &ToolExecutionResult::final_result(serde_json::json!({ "progress": 100 }))
    );
}

#[tokio::test]
async fn execute_tool_emits_single_final_output_for_one_shot_execution() {
    let tool = tool(Tool::function(
        "weather",
        "Weather tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_execute_with_options_fn(|args, options| async move {
        Ok(serde_json::json!({
            "city": args["city"].clone(),
            "toolCallId": options.tool_call_id,
        }))
    });

    let results = execute_tool(
        &tool,
        serde_json::json!({ "city": "Berlin" }),
        ToolExecutionOptions::new("call_oneshot"),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].as_ref().unwrap(),
        &ToolExecutionResult::final_result(serde_json::json!({
            "city": "Berlin",
            "toolCallId": "call_oneshot",
        }))
    );
}

#[tokio::test]
async fn tool_set_executes_streams_with_options() {
    let tools = ExecutableTools::from_tools([tool(Tool::function(
        "search",
        "Search tool",
        serde_json::json!({ "type": "object" }),
    ))
    .with_execute_with_options_fn(|args, options| async move {
        assert_eq!(args["q"], serde_json::json!("rust"));
        assert_eq!(options.tool_call_id, "call_2");
        Ok(serde_json::json!({ "ok": true }))
    })]);

    let out = tools
        .execute_with_options(
            "search",
            serde_json::json!({ "q": "rust" }),
            ToolExecutionOptions::new("call_2"),
        )
        .await
        .unwrap();

    assert_eq!(out, serde_json::json!({ "ok": true }));
}

#[test]
fn tool_execution_options_can_project_chat_messages() {
    let options = ToolExecutionOptions::new("call_3")
        .try_with_chat_messages(&[crate::types::ChatMessage::user("hello").build()])
        .expect("chat messages should convert");

    assert_eq!(options.messages.len(), 1);
    assert!(matches!(
        options.messages.first(),
        Some(crate::types::ModelMessage::User(_))
    ));
}

#[test]
fn callback_contexts_project_from_shared_execution_options() {
    let abort_signal = crate::utils::cancel::new_cancel_handle();
    let options = ToolExecutionOptions::new("call_ctx")
        .with_messages(vec![
            crate::types::ModelMessage::try_from(crate::types::ChatMessage::user("hello").build())
                .expect("user model message"),
        ])
        .with_abort_signal(abort_signal.clone())
        .with_context(Context::from([(
            "requestId".to_string(),
            serde_json::json!("req_1"),
        )]));

    let delta = ToolInputDeltaContext::from_execution_options("{", &options);
    assert_eq!(delta.tool_call_id, "call_ctx");
    assert_eq!(delta.input_text_delta, "{");
    assert!(delta.abort_signal.is_some());
    assert!(matches!(
        delta.messages.first(),
        Some(crate::types::ModelMessage::User(_))
    ));

    let available = ToolInputAvailableContext::from_execution_options(
        serde_json::json!({ "city": "Berlin" }),
        &options,
    );
    assert_eq!(available.input["city"], serde_json::json!("Berlin"));
    assert!(available.abort_signal.is_some());

    let approval = ToolNeedsApprovalContext::from_execution_options(
        serde_json::json!({ "city": "Berlin" }),
        &options,
    );
    assert_eq!(approval.tool_call_id, "call_ctx");
    assert_eq!(
        approval.context.get("requestId"),
        Some(&serde_json::json!("req_1"))
    );
}

#[test]
fn tool_builder_keeps_output_schema_on_portable_function_tool() {
    #[derive(Deserialize)]
    struct Args {
        city: String,
    }

    #[derive(Serialize)]
    struct Out {
        forecast: String,
    }

    let tool = ExecutableTool::typed_function_with_output_schema::<Args, Out, _, _>(
        "weather",
        "Weather tool",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
        serde_json::json!({
            "type": "object",
            "properties": {
                "forecast": { "type": "string" }
            },
            "required": ["forecast"]
        }),
        |args| async move {
            Ok(Out {
                forecast: format!("sunny:{}", args.city),
            })
        },
    );

    let schema = tool
        .tool()
        .output_schema()
        .expect("output schema should be attached");

    assert_eq!(
        schema,
        &serde_json::json!({
            "type": "object",
            "properties": {
                "forecast": { "type": "string" }
            },
            "required": ["forecast"]
        })
    );
}

#[test]
fn tool_set_uses_runtime_model_output_mapper() {
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

    let output = tools
        .to_model_output(
            "weather",
            ToolModelOutputContext {
                tool_call_id: "call_1".to_string(),
                input: serde_json::json!({ "city": "Tokyo" }),
                output: serde_json::json!({ "temp": 18 }),
            },
        )
        .expect("map ok")
        .expect("mapper should exist");

    assert_eq!(
        output,
        ToolResultOutput::content(vec![crate::types::ToolResultContentPart::text("call_1:18")])
    );
}

#[test]
fn provider_tool_factories_preserve_execution_ownership() {
    let input_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": { "type": "string" }
        },
        "required": ["query"]
    });
    let output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"]
    });

    let provider_defined =
        create_provider_defined_tool_factory("acme.search", "search", input_schema.clone())
            .create_tool_with_output_schema(
                serde_json::json!({ "region": "us" }),
                output_schema.clone(),
            );

    let Tool::ProviderDefined(provider_defined) = provider_defined else {
        panic!("expected provider-defined tool");
    };
    assert!(!provider_defined.is_provider_executed());
    assert_eq!(provider_defined.input_schema(), Some(&input_schema));
    assert_eq!(provider_defined.output_schema(), Some(&output_schema));

    let provider_defined_with_output = create_provider_defined_tool_factory_with_output_schema(
        "acme.summarize",
        "summarize",
        input_schema.clone(),
        output_schema.clone(),
    )
    .create_tool(serde_json::json!({ "format": "short" }));

    let Tool::ProviderDefined(provider_defined_with_output) = provider_defined_with_output else {
        panic!("expected provider-defined tool with fixed output schema");
    };
    assert!(!provider_defined_with_output.is_provider_executed());
    assert_eq!(
        provider_defined_with_output.input_schema(),
        Some(&input_schema)
    );
    assert_eq!(
        provider_defined_with_output.output_schema(),
        Some(&output_schema)
    );

    let provider_executed = create_provider_executed_tool_factory(
        "acme.hosted_search",
        "hostedSearch",
        input_schema.clone(),
        output_schema.clone(),
    )
    .with_supports_deferred_results(true)
    .create_tool(serde_json::json!({ "region": "eu" }));

    let Tool::ProviderDefined(provider_executed) = provider_executed else {
        panic!("expected provider-executed tool");
    };
    assert!(provider_executed.is_provider_executed());
    assert_eq!(provider_executed.input_schema(), Some(&input_schema));
    assert_eq!(provider_executed.output_schema(), Some(&output_schema));
    assert_eq!(provider_executed.supports_deferred_results, Some(true));
}

#[tokio::test]
async fn tool_runtime_metadata_supports_callbacks_and_dynamic_flags() {
    let tool = ExecutableTool::function(
        "dangerous",
        "Dangerous tool",
        serde_json::json!({ "type": "object" }),
        |args| async move { Ok(args) },
    )
    .with_dynamic(true)
    .with_context_schema(serde_json::json!({
        "type": "object",
        "properties": {
            "role": { "type": "string" }
        }
    }))
    .with_needs_approval_fn(|context| async move {
        Ok(context
            .context
            .get("role")
            .and_then(|value| value.as_str())
            .is_some_and(|role| role != "admin"))
    })
    .with_on_input_available_fn(|_context| async move { Ok(()) });

    let metadata = tool.runtime_metadata();
    assert!(metadata.dynamic());
    assert!(metadata.context_schema().is_some());
    assert!(metadata.has_needs_approval());
    assert!(metadata.has_on_input_available());
    assert!(
        metadata
            .needs_approval(ToolNeedsApprovalContext {
                tool_call_id: "call_1".to_string(),
                input: serde_json::json!({}),
                messages: Vec::new(),
                context: Context::from([("role".to_string(), serde_json::json!("viewer"),)]),
            })
            .await
            .unwrap()
    );
}

#[test]
fn executable_tool_helpers_match_ai_sdk_style_facade() {
    let executable = tool(Tool::function(
        "weather",
        "Weather tool",
        serde_json::json!({ "type": "object" }),
    ));
    assert!(!is_executable_tool(Some(&executable)));

    let executable = executable.with_execute(Arc::new(|args| Box::pin(async move { Ok(args) })));
    assert!(is_executable_tool(Some(&executable)));
    assert!(
        dynamic_tool(executable.clone())
            .runtime_metadata()
            .dynamic()
    );
}
