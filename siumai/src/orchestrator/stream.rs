//! Streaming orchestrator implementation.

use std::collections::HashSet;

use futures::StreamExt;
use serde_json::Value;
use tokio::sync::oneshot;

use super::types::{OrchestratorStreamOptions, StepResult, ToolApproval, ToolResolver};
use super::validation::validate_args_with_schema;
use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, ContentPart, MessageContent, Tool};

/// Stream handle that carries the stream and a oneshot receiver for steps summary.
pub struct StreamOrchestration {
    pub stream: ChatStream,
    pub steps: oneshot::Receiver<Vec<StepResult>>,
    /// A cancel handle to abort the orchestration.
    pub cancel: crate::utils::cancel::CancelHandle,
}

/// Orchestrate multi-step streaming. Concatenates provider streams across steps.
///
/// This is a simplified version that doesn't execute tools.
/// Use `generate_stream_owned` for full tool execution support.
pub async fn generate_stream(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    _resolver: Option<&dyn ToolResolver>,
    _opts: OrchestratorStreamOptions,
) -> Result<StreamOrchestration, LlmError> {
    let (tx, rx) = oneshot::channel();
    let stream = model.chat_stream(messages, tools).await?;
    let (stream, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
    let _ = tx.send(Vec::new());
    Ok(StreamOrchestration {
        stream,
        steps: rx,
        cancel,
    })
}

/// A Sync stream wrapper around mpsc::Receiver using a Mutex for interior mutability.
struct MpscStream(
    std::sync::Arc<
        std::sync::Mutex<tokio::sync::mpsc::Receiver<Result<ChatStreamEvent, LlmError>>>,
    >,
);

impl futures::Stream for MpscStream {
    type Item = Result<ChatStreamEvent, LlmError>;
    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut guard = self.0.lock().unwrap();
        tokio::sync::mpsc::Receiver::poll_recv(&mut *guard, cx)
    }
}

/// Owned-variant multi-step streaming with tool execution and callbacks.
///
/// This function spawns a background task that:
/// 1. Streams the first step's response
/// 2. Executes any tool calls
/// 3. Continues with subsequent steps (non-streaming for efficiency)
/// 4. Returns a stream that emits all events
///
/// # Arguments
///
/// * `model` - The chat model to use (must be Send + Sync + 'static)
/// * `messages` - Initial message history
/// * `tools` - Available tools for the model to call
/// * `resolver` - Tool resolver for executing tool calls
/// * `opts` - Orchestrator options including callbacks and telemetry
///
/// # Returns
///
/// Returns a `StreamOrchestration` containing:
/// - `stream`: The event stream
/// - `steps`: A receiver for the final step results
/// - `cancel`: A handle to cancel the orchestration
pub async fn generate_stream_owned<M, R>(
    model: M,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    resolver: Option<R>,
    opts: OrchestratorStreamOptions,
) -> Result<StreamOrchestration, LlmError>
where
    M: ChatCapability + Send + Sync + 'static,
    R: ToolResolver + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<ChatStreamEvent, LlmError>>(64);
    let (steps_tx, steps_rx) = oneshot::channel();
    let mut history = messages;
    let max_steps = if opts.max_steps == 0 {
        1
    } else {
        opts.max_steps
    };
    let tools_opt = tools.clone();
    let on_chunk = opts.on_chunk.clone();
    let on_step_finish = opts.on_step_finish.clone();
    let on_finish = opts.on_finish.clone();
    let on_tool_approval = opts.on_tool_approval.clone();
    let on_preliminary_tool_result = opts.on_preliminary_tool_result.clone();
    let on_abort = opts.on_abort.clone();
    let orchestrator_cancel = crate::utils::cancel::new_cancel_handle();
    let orchestrator_cancel_clone = orchestrator_cancel.clone();

    tokio::spawn(async move {
        let mut step_results: Vec<StepResult> = Vec::new();
        let mut encountered_error = false;
        let resolver = resolver
            .map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn ToolResolver + Send + Sync>);
        let sender = tx;
        // Track processed tool_call IDs to avoid duplicate executions across steps.
        let mut processed_call_ids: HashSet<String> = HashSet::new();
        'outer: for step_idx in 0..max_steps {
            if orchestrator_cancel_clone.is_cancelled() {
                break 'outer;
            }
            // First step streams; subsequent steps may be non-streaming providers.
            let resp = if step_idx == 0 {
                let handle = match model
                    .chat_stream_with_cancel(history.clone(), tools.clone())
                    .await
                {
                    Ok(h) => h,
                    Err(e) => {
                        encountered_error = true;
                        let _ = sender.send(Err(e)).await;
                        break;
                    }
                };
                let mut s = handle.stream;
                let mut acc_text = String::new();
                let mut final_resp: Option<ChatResponse> = None;
                while let Some(item) = s.next().await {
                    if orchestrator_cancel_clone.is_cancelled() {
                        handle.cancel.cancel();
                        break 'outer;
                    }
                    match item {
                        Ok(ev) => {
                            match &ev {
                                ChatStreamEvent::ContentDelta { delta, .. } => {
                                    if let Some(cb) = &on_chunk {
                                        cb(&ev);
                                    }
                                    acc_text.push_str(delta)
                                }
                                ChatStreamEvent::StreamEnd { response } => {
                                    final_resp = Some(response.clone())
                                }
                                _ => {}
                            }
                            let _ = sender.send(Ok(ev)).await;
                        }
                        Err(e) => {
                            encountered_error = true;
                            let _ = sender.send(Err(e)).await;
                            break;
                        }
                    }
                }
                final_resp
                    .unwrap_or_else(|| ChatResponse::new(MessageContent::Text(acc_text.clone())))
            } else {
                // Non-streaming follow-up to advance conversation efficiently
                // Use chat_request if we have common_params
                let result = if opts.common_params.is_some() {
                    let mut request = crate::types::ChatRequest::new(history.clone());

                    if let Some(tools) = tools.clone() {
                        request = request.with_tools(tools);
                    }

                    // Apply agent-level common_params
                    if let Some(ref common_params) = opts.common_params {
                        request = request.with_common_params(common_params.clone());
                    }

                    model.chat_request(request).await
                } else {
                    model.chat_with_tools(history.clone(), tools.clone()).await
                };

                match result {
                    Ok(r) => r,
                    Err(e) => {
                        encountered_error = true;
                        let _ = sender.send(Err(e)).await;
                        break;
                    }
                }
            };
            let mut step_msgs: Vec<ChatMessage> = Vec::new();
            let _assistant_text = resp
                .content_text()
                .map(|s| s.to_string())
                .unwrap_or_else(String::new);
            // The response content already contains tool calls, so we just use it directly
            let assistant_built = ChatMessage {
                role: crate::types::MessageRole::Assistant,
                content: resp.content.clone(),
                metadata: crate::types::MessageMetadata::default(),
            };
            history.push(assistant_built.clone());
            step_msgs.push(assistant_built);

            // Execute tools if any
            let tool_calls: Vec<_> = resp.tool_calls().into_iter().cloned().collect();
            if !tool_calls.is_empty()
                && let (Some(resolver), Some(ts)) = (resolver.as_ref(), tools_opt.as_ref())
            {
                for call in tool_calls.iter() {
                    if let crate::types::ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } = call
                    {
                        // Skip duplicate tool-call IDs already executed in previous steps.
                        if processed_call_ids.contains(tool_call_id) {
                            continue;
                        }
                        if let Some(def) = ts.iter().find(|t| match t {
                            Tool::Function { function } => &function.name == tool_name,
                            Tool::ProviderDefined(provider_tool) => {
                                &provider_tool.name == tool_name
                            }
                        }) {
                            let parameters = match def {
                                Tool::Function { function } => &function.parameters,
                                Tool::ProviderDefined(_) => {
                                    // Provider-defined tools don't have parameters schema
                                    // Skip validation for them
                                    continue;
                                }
                            };
                            if let Err(reason) = validate_args_with_schema(parameters, arguments) {
                                let out =
                                    serde_json::json!({"error":"invalid_args","reason":reason});
                                let tool_msg = ChatMessage::tool_error_json(
                                    tool_call_id.clone(),
                                    tool_name.clone(),
                                    out,
                                )
                                .build();
                                history.push(tool_msg.clone());
                                step_msgs.push(tool_msg);
                                continue;
                            }
                        }
                        let decision = if let Some(cb) = &on_tool_approval {
                            cb(tool_name, arguments)
                        } else {
                            ToolApproval::Approve(arguments.clone())
                        };
                        let out_val = match decision {
                            ToolApproval::Approve(args) | ToolApproval::Modify(args) => {
                                // Use streaming tool execution
                                match resolver.call_tool_stream(tool_name, args).await {
                                    Ok(mut stream) => {
                                        let mut final_output = None;

                                        // Process stream
                                        while let Some(result) = stream.next().await {
                                            match result {
                                                Ok(tool_result) => {
                                                    if tool_result.is_preliminary() {
                                                        // Call preliminary callback if provided
                                                        if let Some(cb) =
                                                            &on_preliminary_tool_result
                                                        {
                                                            cb(
                                                                tool_name,
                                                                tool_call_id,
                                                                tool_result.output(),
                                                            );
                                                        }
                                                    } else {
                                                        // Final result
                                                        final_output =
                                                            Some(tool_result.into_output());
                                                    }
                                                }
                                                Err(e) => {
                                                    // Error during streaming
                                                    final_output = Some(Value::String(format!(
                                                        "<tool error: {}>",
                                                        e
                                                    )));
                                                    break;
                                                }
                                            }
                                        }

                                        final_output.unwrap_or_else(|| {
                                            Value::String(
                                                "<tool error: no final result>".to_string(),
                                            )
                                        })
                                    }
                                    Err(e) => {
                                        serde_json::Value::String(format!("<tool error: {}>", e))
                                    }
                                }
                            }
                            ToolApproval::Deny { reason } => {
                                serde_json::json!({"error":"denied","reason":reason})
                            }
                        };
                        let tool_msg = ChatMessage::tool_result_json(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            out_val,
                        )
                        .build();
                        history.push(tool_msg.clone());
                        step_msgs.push(tool_msg);
                        processed_call_ids.insert(tool_call_id.clone());
                    }
                }
            }

            // Extract tool results from tool messages
            let tool_results: Vec<ContentPart> = step_msgs
                .iter()
                .filter(|msg| matches!(msg.role, crate::types::MessageRole::Tool))
                .flat_map(|msg| msg.tool_results())
                .cloned()
                .collect();

            let step = StepResult {
                messages: step_msgs,
                finish_reason: resp.finish_reason.clone(),
                usage: resp.usage.clone(),
                tool_calls: tool_calls.clone(),
                tool_results,
                warnings: resp.warnings.clone(),
            };
            if let Some(cb) = &on_step_finish {
                cb(&step);
            }
            step_results.push(step);

            if tool_calls.is_empty() {
                break;
            }
        }
        if orchestrator_cancel_clone.is_cancelled() {
            if let Some(cb) = &on_abort {
                cb(&step_results);
            }
        } else if !encountered_error && let Some(cb) = &on_finish {
            cb(&step_results);
        }
        let _ = steps_tx.send(step_results);
    });

    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let stream: ChatStream = std::pin::Pin::from(Box::new(MpscStream(rx))
        as Box<dyn futures::Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync>);
    Ok(StreamOrchestration {
        stream,
        steps: steps_rx,
        cancel: orchestrator_cancel,
    })
}
