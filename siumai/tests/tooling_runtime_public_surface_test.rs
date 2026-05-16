use std::mem::size_of;

use futures::StreamExt;
use serde_json::{Value, json};

#[tokio::test]
async fn public_surface_tooling_runtime_contract_compiles() {
    use siumai::prelude::unified::{CancelHandle, ChatMessage, Context, Tool, ToolResultOutput};
    use siumai::tooling::{
        ExecutableTool, ExecutableTools, ToolExecutionOptions, ToolExecutionResult,
        ToolExecutionStream, ToolInputAvailableContext, ToolInputDeltaContext,
        ToolModelOutputContext, ToolNeedsApprovalContext, ToolRuntimeContext, ToolRuntimeMetadata,
        ToolSet, execute_tool, model_messages_from_chat_messages, tool,
    };

    let _ = size_of::<ExecutableTool>();
    let _ = size_of::<ExecutableTools>();
    let _ = size_of::<ToolExecutionOptions>();
    let _ = size_of::<ToolExecutionResult>();
    let _ = size_of::<ToolExecutionStream>();
    let _ = size_of::<ToolNeedsApprovalContext>();
    let _ = size_of::<ToolInputDeltaContext>();
    let _ = size_of::<ToolInputAvailableContext>();
    let _ = size_of::<ToolModelOutputContext>();
    let _ = size_of::<ToolRuntimeContext>();
    let _ = size_of::<ToolRuntimeMetadata>();
    let _ = size_of::<ToolSet>();

    let _ = ToolNeedsApprovalContext::from_execution_options
        as fn(Value, &ToolExecutionOptions) -> ToolNeedsApprovalContext;
    let _ = ToolInputDeltaContext::from_execution_options
        as fn(String, &ToolExecutionOptions) -> ToolInputDeltaContext;
    let _ = ToolInputAvailableContext::from_execution_options
        as fn(Value, &ToolExecutionOptions) -> ToolInputAvailableContext;
    let _ = ToolRuntimeMetadata::dynamic as fn(&ToolRuntimeMetadata) -> bool;
    let _ = ToolRuntimeMetadata::context_schema as fn(&ToolRuntimeMetadata) -> Option<&Value>;

    let shared_messages = model_messages_from_chat_messages(&[ChatMessage::user("hello").build()])
        .expect("project model messages");
    let context: Context = [("requestId".to_string(), json!("req-1"))]
        .into_iter()
        .collect();
    let runtime_options = ToolExecutionOptions::new("call_runtime")
        .with_messages(shared_messages.clone())
        .with_context(context.clone())
        .with_abort_signal(CancelHandle::new());

    let runtime_tool = tool(Tool::function(
        "search",
        "Search tool",
        json!({ "type": "object" }),
    ))
    .with_context_schema(json!({
        "type": "object",
        "properties": {
            "requestId": { "type": "string" }
        }
    }))
    .with_needs_approval_fn(|ctx| async move {
        assert_eq!(ctx.tool_call_id, "call_runtime");
        assert_eq!(ctx.input["q"], json!("needs-approval"));
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.context["requestId"], json!("req-1"));
        Ok(true)
    })
    .with_on_input_start_fn(|ctx| async move {
        assert_eq!(ctx.tool_call_id, "call_runtime");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.context["requestId"], json!("req-1"));
        assert!(ctx.abort_signal.is_some());
        Ok(())
    })
    .with_on_input_delta_fn(|ctx| async move {
        assert_eq!(ctx.tool_call_id, "call_runtime");
        assert_eq!(ctx.input_text_delta, "{\"q\":");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.context["requestId"], json!("req-1"));
        assert!(ctx.abort_signal.is_some());
        Ok(())
    })
    .with_on_input_available_fn(|ctx| async move {
        assert_eq!(ctx.tool_call_id, "call_runtime");
        assert_eq!(ctx.input["q"], json!("rust"));
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.context["requestId"], json!("req-1"));
        assert!(ctx.abort_signal.is_some());
        Ok(())
    })
    .with_to_model_output_fn(|ctx| {
        assert_eq!(ctx.tool_call_id, "call_runtime");
        assert_eq!(ctx.input["q"], json!("rust"));
        assert_eq!(ctx.output["echo"], json!("rust"));
        Ok(ToolResultOutput::text("search=rust"))
    })
    .with_execute_with_options_fn(|args, options| async move {
        assert_eq!(options.tool_call_id, "call_runtime");
        assert_eq!(options.messages.len(), 1);
        assert_eq!(options.context["requestId"], json!("req-1"));
        assert!(options.abort_signal.is_some());
        Ok(json!({ "echo": args["q"].clone() }))
    });

    let metadata = runtime_tool.runtime_metadata();
    assert!(metadata.context_schema().is_some());
    assert!(
        metadata
            .needs_approval(ToolNeedsApprovalContext::from_execution_options(
                json!({ "q": "needs-approval" }),
                &runtime_options,
            ))
            .await
            .expect("needs approval"),
    );
    metadata
        .invoke_on_input_start(runtime_options.clone())
        .await
        .expect("invoke on input start");
    metadata
        .invoke_on_input_delta(ToolInputDeltaContext::from_execution_options(
            "{\"q\":".to_string(),
            &runtime_options,
        ))
        .await
        .expect("invoke on input delta");
    metadata
        .invoke_on_input_available(ToolInputAvailableContext::from_execution_options(
            json!({ "q": "rust" }),
            &runtime_options,
        ))
        .await
        .expect("invoke on input available");

    let mut results = execute_tool(
        &runtime_tool,
        json!({ "q": "rust" }),
        runtime_options.clone(),
    )
    .await
    .expect("execute tool");
    let first = results
        .next()
        .await
        .expect("tool result")
        .expect("tool result ok");
    assert!(matches!(first, ToolExecutionResult::Final { .. }));
    assert!(results.next().await.is_none());

    let model_output = runtime_tool
        .to_model_output(ToolModelOutputContext {
            tool_call_id: "call_runtime".to_string(),
            input: json!({ "q": "rust" }),
            output: json!({ "echo": "rust" }),
        })
        .expect("to model output")
        .expect("mapped output");
    assert_eq!(model_output.to_string_lossy(), "search=rust");

    let tools: ToolSet = ExecutableTools::from_tools([runtime_tool]);
    let output = tools
        .execute_with_options("search", json!({ "q": "rust" }), runtime_options.clone())
        .await
        .expect("execute tool set");
    assert_eq!(output["echo"], json!("rust"));

    let mapped = tools
        .to_model_output(
            "search",
            ToolModelOutputContext {
                tool_call_id: "call_runtime".to_string(),
                input: json!({ "q": "rust" }),
                output: json!({ "echo": "rust" }),
            },
        )
        .expect("tool set model output")
        .expect("mapped output");
    assert_eq!(mapped.to_string_lossy(), "search=rust");
}

#[test]
fn public_surface_provider_defined_tool_metadata_compiles() {
    use siumai::prelude::unified::{ProviderDefinedTool, ProviderOptionsMap, Tool};

    let mut provider_options = ProviderOptionsMap::new();
    provider_options.insert("openai", json!({ "defer_loading": true }));

    let tool = Tool::provider_defined("openai.web_search", "web_search")
        .with_title("Web Search")
        .with_provider_options_map(provider_options.clone())
        .with_supports_deferred_results(true)
        .with_args(json!({ "searchContextSize": "high" }));

    assert_eq!(tool.title(), Some("Web Search"));
    assert_eq!(tool.provider_options_map(), Some(&provider_options));

    let provider_tool = match tool {
        Tool::ProviderDefined(provider_tool) => provider_tool,
        Tool::Function { .. } => panic!("expected provider-defined tool"),
    };

    assert_eq!(provider_tool.supports_deferred_results, Some(true));
    assert_eq!(provider_tool.args["searchContextSize"], json!("high"));

    let from_id = Tool::provider_defined_id("openai.web_search").expect("known provider tool id");
    assert!(matches!(from_id, Tool::ProviderDefined(_)));

    let typed =
        ProviderDefinedTool::from_id("openai.web_search").expect("known provider-defined tool");
    assert_eq!(typed.id, "openai.web_search");
}
