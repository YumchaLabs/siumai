use std::collections::{HashMap, HashSet};

use crate::orchestrator::{OrchestratorContext, ToolResolver};
use futures::StreamExt;
use serde_json::{Value, json};
use siumai::prelude::unified::*;

pub(crate) fn merge_step_tool_results(
    response: &ChatResponse,
    local_tool_results: &[ContentPart],
) -> Vec<ContentPart> {
    let mut tool_results = response
        .tool_results()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    tool_results.extend(local_tool_results.iter().cloned());
    tool_results
}

pub(crate) fn client_tool_call_count(tool_calls: &[ContentPart]) -> usize {
    tool_calls
        .iter()
        .filter_map(ContentPart::as_tool_call)
        .filter(|tool_call| tool_call.provider_executed != Some(&true))
        .count()
}

pub(crate) fn update_pending_deferred_tool_calls(
    pending_tool_calls: &mut HashSet<String>,
    tools: Option<&[Tool]>,
    tool_calls: &[ContentPart],
    tool_results: &[ContentPart],
) {
    let resolved_ids = tool_results
        .iter()
        .filter_map(ContentPart::as_tool_result)
        .map(|tool_result| tool_result.tool_call_id)
        .collect::<HashSet<_>>();

    for tool_call in tool_calls.iter().filter_map(ContentPart::as_tool_call) {
        if tool_call.provider_executed != Some(&true) {
            continue;
        }

        if !provider_tool_supports_deferred_results(tools, tool_call.tool_name) {
            continue;
        }

        if !resolved_ids.contains(tool_call.tool_call_id) {
            pending_tool_calls.insert(tool_call.tool_call_id.to_string());
        }
    }

    for tool_result in tool_results.iter().filter_map(ContentPart::as_tool_result) {
        pending_tool_calls.remove(tool_result.tool_call_id);
    }
}

pub(crate) fn should_continue_after_tool_step(
    resolver_available: bool,
    client_tool_calls: usize,
    local_tool_results: usize,
    pending_tool_calls: &HashSet<String>,
) -> bool {
    (resolver_available && client_tool_calls > 0 && local_tool_results >= client_tool_calls)
        || !pending_tool_calls.is_empty()
}

fn provider_tool_supports_deferred_results(tools: Option<&[Tool]>, tool_name: &str) -> bool {
    tools
        .and_then(|tools| {
            tools.iter().find_map(|tool| match tool {
                Tool::ProviderDefined(provider_tool) if provider_tool.name == tool_name => {
                    provider_tool.supports_deferred_results
                }
                _ => None,
            })
        })
        .unwrap_or(false)
}

pub(crate) type PreliminaryToolResultCallback = dyn Fn(&str, &str, &Value) + Send + Sync;

#[derive(Debug, Clone)]
pub(crate) struct CollectedToolApproval {
    pub approval_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: Value,
    pub approved: bool,
    pub reason: Option<String>,
    pub provider_executed: bool,
    pub dynamic: bool,
}

#[derive(Debug, Default)]
pub(crate) struct CollectedToolApprovals {
    pub approved: Vec<CollectedToolApproval>,
    pub denied: Vec<CollectedToolApproval>,
}

#[derive(Debug, Default)]
pub(crate) struct ToolApprovalPreprocessResult {
    pub local_tool_message: Option<ChatMessage>,
    pub provider_forward_message: Option<ChatMessage>,
    pub processed_tool_call_ids: HashSet<String>,
}

#[derive(Debug, Clone)]
struct CollectedToolCall {
    tool_name: String,
    input: Value,
    provider_executed: bool,
    dynamic: bool,
}

pub(crate) fn collect_tool_approvals(
    messages: &[ChatMessage],
) -> Result<CollectedToolApprovals, LlmError> {
    let Some(last_message) = messages.last() else {
        return Ok(CollectedToolApprovals::default());
    };

    if last_message.role != MessageRole::Tool {
        return Ok(CollectedToolApprovals::default());
    }

    let MessageContent::MultiModal(last_parts) = &last_message.content else {
        return Ok(CollectedToolApprovals::default());
    };

    let mut tool_calls_by_id = HashMap::<String, CollectedToolCall>::new();
    let mut approval_requests_by_id = HashMap::<String, String>::new();

    for message in messages {
        if message.role != MessageRole::Assistant {
            continue;
        }

        let MessageContent::MultiModal(parts) = &message.content else {
            continue;
        };

        for part in parts {
            match part {
                ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    provider_executed,
                    dynamic,
                    ..
                } => {
                    tool_calls_by_id.insert(
                        tool_call_id.clone(),
                        CollectedToolCall {
                            tool_name: tool_name.clone(),
                            input: arguments.clone(),
                            provider_executed: provider_executed.unwrap_or(false),
                            dynamic: dynamic.unwrap_or(false),
                        },
                    );
                }
                ContentPart::ToolApprovalRequest {
                    approval_id,
                    tool_call_id,
                    ..
                } => {
                    approval_requests_by_id.insert(approval_id.clone(), tool_call_id.clone());
                }
                _ => {}
            }
        }
    }

    let tool_results_in_last = last_parts
        .iter()
        .filter_map(ContentPart::as_tool_result)
        .map(|tool_result| tool_result.tool_call_id.to_string())
        .collect::<HashSet<_>>();

    let mut approvals = CollectedToolApprovals::default();

    for part in last_parts {
        let ContentPart::ToolApprovalResponse {
            approval_id,
            approved,
            reason,
            provider_executed,
            ..
        } = part
        else {
            continue;
        };

        let Some(tool_call_id) = approval_requests_by_id.get(approval_id) else {
            return Err(LlmError::InvalidParameter(format!(
                "Tool approval response references unknown approvalId: \"{}\"",
                approval_id
            )));
        };

        if tool_results_in_last.contains(tool_call_id) {
            continue;
        }

        let Some(tool_call) = tool_calls_by_id.get(tool_call_id) else {
            return Err(LlmError::InvalidParameter(format!(
                "Tool approval response references missing toolCallId: \"{}\" for approvalId: \"{}\"",
                tool_call_id, approval_id
            )));
        };

        let approval = CollectedToolApproval {
            approval_id: approval_id.clone(),
            tool_call_id: tool_call_id.clone(),
            tool_name: tool_call.tool_name.clone(),
            input: tool_call.input.clone(),
            approved: *approved,
            reason: reason.clone(),
            provider_executed: provider_executed.unwrap_or(tool_call.provider_executed),
            dynamic: tool_call.dynamic,
        };

        if *approved {
            approvals.approved.push(approval);
        } else {
            approvals.denied.push(approval);
        }
    }

    Ok(approvals)
}

pub(crate) async fn execute_local_tool_call(
    resolver: &dyn ToolResolver,
    tool_name: &str,
    tool_call_id: &str,
    execution_args: Value,
    tool_dynamic: bool,
    context: &OrchestratorContext,
    on_preliminary_tool_result: Option<&PreliminaryToolResultCallback>,
) -> ContentPart {
    let input = execution_args.clone();
    let out_val = match resolver
        .call_tool_stream_with_context(tool_name, execution_args, context)
        .await
    {
        Ok(mut stream) => {
            let mut final_output = None;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(tool_result) => {
                        if tool_result.is_preliminary() {
                            if let Some(callback) = on_preliminary_tool_result {
                                callback(tool_name, tool_call_id, tool_result.output());
                            }
                        } else {
                            final_output = Some(tool_result.into_output());
                        }
                    }
                    Err(error) => {
                        final_output = Some(Value::String(format!("<tool error: {}>", error)));
                        break;
                    }
                }
            }

            final_output
                .unwrap_or_else(|| Value::String("<tool error: no final result>".to_string()))
        }
        Err(error) => Value::String(format!("<tool error: {}>", error)),
    };

    let mut part =
        ContentPart::tool_result_json(tool_call_id.to_string(), tool_name.to_string(), out_val)
            .with_tool_result_input(input);

    if tool_dynamic {
        part = part.with_tool_dynamic(true);
    }

    part
}

pub(crate) fn execution_denied_tool_result(
    tool_call_id: &str,
    tool_name: &str,
    input: Value,
    reason: Option<String>,
    tool_dynamic: bool,
    provider_approval_id: Option<&str>,
) -> ContentPart {
    let mut part =
        ContentPart::tool_execution_denied(tool_call_id.to_string(), tool_name.to_string(), reason)
            .with_tool_result_input(input);

    if tool_dynamic {
        part = part.with_tool_dynamic(true);
    }

    if let Some(approval_id) = provider_approval_id
        && let ContentPart::ToolResult { output, .. } = &mut part
    {
        output
            .provider_options_mut()
            .insert("openai", json!({ "approvalId": approval_id }));
    }

    part
}

pub(crate) async fn preprocess_tool_approval_responses(
    messages: &[ChatMessage],
    resolver: Option<&dyn ToolResolver>,
    context: &OrchestratorContext,
    on_preliminary_tool_result: Option<&PreliminaryToolResultCallback>,
) -> Result<ToolApprovalPreprocessResult, LlmError> {
    let approvals = collect_tool_approvals(messages)?;
    let requires_local_execution = approvals
        .approved
        .iter()
        .any(|approval| !approval.provider_executed);

    if requires_local_execution && resolver.is_none() {
        return Err(LlmError::UnsupportedOperation(
            "tool approval response requires a resolver to continue local tool execution"
                .to_string(),
        ));
    }

    let mut local_tool_parts = Vec::new();
    let mut provider_forward_parts = Vec::new();
    let mut processed_tool_call_ids = HashSet::new();

    for approval in approvals.approved {
        if approval.provider_executed {
            provider_forward_parts.push(provider_tool_approval_response(&approval));
            continue;
        }

        let Some(resolver) = resolver else {
            unreachable!("validated above");
        };
        let tool_call_id = approval.tool_call_id.clone();
        local_tool_parts.push(
            execute_local_tool_call(
                resolver,
                approval.tool_name.as_str(),
                approval.tool_call_id.as_str(),
                approval.input,
                approval.dynamic,
                context,
                on_preliminary_tool_result,
            )
            .await,
        );
        processed_tool_call_ids.insert(tool_call_id);
    }

    for approval in approvals.denied {
        let provider_forward_part = approval
            .provider_executed
            .then(|| provider_tool_approval_response(&approval));
        let tool_call_id = approval.tool_call_id.clone();
        local_tool_parts.push(execution_denied_tool_result(
            approval.tool_call_id.as_str(),
            approval.tool_name.as_str(),
            approval.input,
            approval.reason.clone(),
            approval.dynamic,
            approval
                .provider_executed
                .then_some(approval.approval_id.as_str()),
        ));
        processed_tool_call_ids.insert(tool_call_id);

        if let Some(part) = provider_forward_part {
            provider_forward_parts.push(part);
        }
    }

    Ok(ToolApprovalPreprocessResult {
        local_tool_message: tool_message_from_parts(local_tool_parts),
        provider_forward_message: tool_message_from_parts(provider_forward_parts),
        processed_tool_call_ids,
    })
}

fn provider_tool_approval_response(approval: &CollectedToolApproval) -> ContentPart {
    ContentPart::ToolApprovalResponse {
        approval_id: approval.approval_id.clone(),
        approved: approval.approved,
        reason: approval.reason.clone(),
        provider_executed: Some(true),
        provider_options: ProviderOptionsMap::default(),
    }
}

fn tool_message_from_parts(parts: Vec<ContentPart>) -> Option<ChatMessage> {
    (!parts.is_empty()).then_some(ChatMessage {
        role: MessageRole::Tool,
        content: MessageContent::MultiModal(parts),
        provider_options: ProviderOptionsMap::default(),
        metadata: MessageMetadata::default(),
    })
}
