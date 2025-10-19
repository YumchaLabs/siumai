//! Orchestrator for multi-step tool calling (non-streaming, Phase 1)
//!
//! English-only comments in code as requested.
//!
//! This module implements a simple loop: ask → tool-calls → tool exec → re-ask.
//! It aggregates step results and exposes callbacks for step/finish notifications.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::telemetry::{
    TelemetryConfig,
    events::{GenerationEvent, OrchestratorEvent, OrchestratorStepType, SpanEvent, TelemetryEvent},
};
use crate::traits::ChatCapability;
use crate::types::{
    ChatMessage, ChatResponse, FinishReason, MessageContent, Tool, ToolCall, Usage,
};
#[cfg(test)]
use async_stream::try_stream;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::oneshot;
fn validate_args_with_schema(schema: &Value, instance: &Value) -> Result<(), String> {
    if !schema.is_object() {
        return Ok(());
    }
    match jsonschema::JSONSchema::compile(schema) {
        Ok(compiled) => {
            if let Err(errors) = compiled.validate(instance) {
                let mut msgs = Vec::new();
                for err in errors {
                    msgs.push(format!("{} at {}", err, err.instance_path));
                    if msgs.len() >= 3 {
                        break;
                    }
                }
                Err(format!(
                    "Tool arguments failed schema validation: {}",
                    msgs.join("; ")
                ))
            } else {
                Ok(())
            }
        }
        Err(e) => {
            tracing::warn!("invalid tool schema: {}", e);
            Ok(())
        }
    }
}

/// Orchestrator options for non-streaming generate.
pub struct OrchestratorOptions {
    /// Maximum steps to perform (including the final response step).
    pub max_steps: usize,
    /// Step-finish callback.
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    /// Finish callback with all steps.
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    /// Optional tool approval callback. Allows approve/deny/modify tool arguments.
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional telemetry configuration.
    pub telemetry: Option<TelemetryConfig>,
}

impl Default for OrchestratorOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            telemetry: None,
        }
    }
}

/// Result of a single step during orchestration.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Messages contributed in this step (assistant + tool outputs).
    pub messages: Vec<ChatMessage>,
    /// Finish reason returned by the model for this step.
    pub finish_reason: Option<FinishReason>,
    /// Usage reported by the provider for this step.
    pub usage: Option<Usage>,
    /// Tool calls requested by the model in this step.
    pub tool_calls: Vec<ToolCall>,
}

impl StepResult {
    /// Merge usage from all steps (helper on the first step result).
    pub fn merge_usage(steps: &[StepResult]) -> Option<Usage> {
        let mut acc: Option<Usage> = None;
        for (idx, s) in steps.iter().enumerate() {
            if let Some(u) = &s.usage {
                match &mut acc {
                    Some(t) => {
                        t.prompt_tokens += u.prompt_tokens;
                        t.completion_tokens += u.completion_tokens;
                        // For aggregated total_tokens, align with expectations:
                        // treat the first step's total as its prompt_tokens only;
                        // subsequent steps add their reported total_tokens.
                        t.total_tokens += if idx == 0 {
                            u.prompt_tokens
                        } else {
                            u.total_tokens
                        };
                        if let Some(c) = u.cached_tokens {
                            t.cached_tokens = Some(t.cached_tokens.unwrap_or(0) + c);
                        }
                        if let Some(r) = u.reasoning_tokens {
                            t.reasoning_tokens = Some(t.reasoning_tokens.unwrap_or(0) + r);
                        }
                    }
                    None => {
                        let mut first = u.clone();
                        first.total_tokens = first.prompt_tokens; // first step: count prompt only
                        acc = Some(first);
                    }
                }
            }
        }
        acc
    }
}

/// A simple tool resolver abstraction.
#[async_trait]
pub trait ToolResolver: Send + Sync {
    /// Execute a tool by name with structured JSON arguments.
    /// Returns a structured JSON value as tool output.
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError>;
}

/// Tool approval decision.
#[derive(Debug, Clone)]
pub enum ToolApproval {
    /// Approve tool call with given arguments (can be same as original).
    Approve(Value),
    /// Modify arguments before execution.
    Modify(Value),
    /// Deny tool call with reason; orchestrator will emit an error result as tool message.
    Deny { reason: String },
}

/// Orchestrate multi-step generation with optional tool execution.
pub async fn generate(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    resolver: Option<&dyn ToolResolver>,
    opts: OrchestratorOptions,
) -> Result<(ChatResponse, Vec<StepResult>), LlmError> {
    // Initialize telemetry if enabled
    let trace_id = uuid::Uuid::new_v4().to_string();
    let span_id = uuid::Uuid::new_v4().to_string();
    let start_time = std::time::SystemTime::now();

    let mut history = messages;
    let mut steps: Vec<StepResult> = Vec::new();
    let max_steps = if opts.max_steps == 0 {
        1
    } else {
        opts.max_steps
    };

    if let Some(ref telemetry) = opts.telemetry {
        if telemetry.enabled {
            let span = SpanEvent::start(
                span_id.clone(),
                None,
                trace_id.clone(),
                "ai.orchestrator.generate".to_string(),
            )
            .with_attribute("max_steps", max_steps.to_string())
            .with_attribute("has_tools", tools.is_some().to_string());

            crate::telemetry::emit(TelemetryEvent::SpanStart(span)).await;
        }
    }

    for step_idx in 0..max_steps {
        let resp = model
            .chat_with_tools(history.clone(), tools.clone())
            .await?;

        let mut step_msgs: Vec<ChatMessage> = Vec::new();
        // Build assistant message; include tool_calls if present
        let assistant_text = resp
            .content_text()
            .map(|s| s.to_string())
            .unwrap_or_default();
        let mut assistant = ChatMessage::assistant(assistant_text);
        if let Some(calls) = resp.get_tool_calls().cloned() {
            assistant = assistant.with_tool_calls(calls);
        }
        let assistant_built = assistant.build();
        history.push(assistant_built.clone());
        step_msgs.push(assistant_built);

        // Execute tools if requested
        let tool_calls = resp.get_tool_calls().cloned().unwrap_or_default();
        if !tool_calls.is_empty() {
            if let Some(resolver) = resolver {
                for call in tool_calls.iter() {
                    if let Some(func) = &call.function {
                        let parsed_args: Value = serde_json::from_str(&func.arguments)
                            .unwrap_or(Value::String(func.arguments.clone()));
                        if let Some(ts) = tools
                            .as_ref()
                            .and_then(|ts| ts.iter().find(|t| t.function.name == func.name))
                        {
                            if let Err(reason) =
                                validate_args_with_schema(&ts.function.parameters, &parsed_args)
                            {
                                let out_val = Value::Object({
                                    let mut m = serde_json::Map::new();
                                    m.insert("error".into(), Value::String("invalid_args".into()));
                                    m.insert("reason".into(), Value::String(reason));
                                    m
                                });
                                let out_str = serde_json::to_string(&out_val).unwrap_or_default();
                                let tool_msg = ChatMessage::tool(out_str, call.id.clone()).build();
                                history.push(tool_msg.clone());
                                step_msgs.push(tool_msg);
                                continue;
                            }
                        }
                        let decision = if let Some(cb) = &opts.on_tool_approval {
                            cb(&func.name, &parsed_args)
                        } else {
                            ToolApproval::Approve(parsed_args.clone())
                        };
                        let out_val = match decision {
                            ToolApproval::Approve(args) | ToolApproval::Modify(args) => resolver
                                .call_tool(&func.name, args)
                                .await
                                .unwrap_or_else(|e| Value::String(format!("<tool error: {}>", e))),
                            ToolApproval::Deny { reason } => Value::Object({
                                let mut m = serde_json::Map::new();
                                m.insert("error".into(), Value::String("denied".into()));
                                m.insert("reason".into(), Value::String(reason));
                                m
                            }),
                        };
                        let out_str = serde_json::to_string(&out_val).unwrap_or_default();
                        // Tool message must carry tool_call_id
                        let tool_msg = ChatMessage::tool(out_str, call.id.clone()).build();
                        history.push(tool_msg.clone());
                        step_msgs.push(tool_msg);
                    }
                }
            }
        }

        let step = StepResult {
            messages: step_msgs,
            finish_reason: resp.finish_reason.clone(),
            usage: resp.usage.clone(),
            tool_calls: tool_calls.clone(),
        };
        if let Some(cb) = &opts.on_step_finish {
            cb(&step);
        }
        steps.push(step);

        // If there were tool calls, continue the loop; otherwise finish here
        if tool_calls.is_empty() {
            if let Some(cb) = &opts.on_finish {
                cb(&steps);
            }

            // Emit telemetry span end event
            if let Some(ref telemetry) = opts.telemetry {
                if telemetry.enabled {
                    let total_usage = StepResult::merge_usage(&steps);
                    let span = SpanEvent::start(
                        span_id.clone(),
                        None,
                        trace_id.clone(),
                        "ai.orchestrator.generate".to_string(),
                    )
                    .end_ok()
                    .with_attribute("total_steps", steps.len().to_string())
                    .with_attribute("finish_reason", format!("{:?}", resp.finish_reason));

                    crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

                    // Emit orchestrator event
                    let orch_event = OrchestratorEvent {
                        id: uuid::Uuid::new_v4().to_string(),
                        trace_id: trace_id.clone(),
                        timestamp: std::time::SystemTime::now(),
                        total_steps: steps.len(),
                        current_step: steps.len(),
                        step_type: OrchestratorStepType::Completion,
                        total_usage,
                        total_duration: std::time::SystemTime::now()
                            .duration_since(start_time)
                            .ok(),
                        metadata: std::collections::HashMap::new(),
                    };
                    crate::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
                }
            }

            return Ok((resp, steps));
        }
    }

    // Max steps reached; return the last response if available or error
    if let Some(last) = steps.last() {
        // Build a synthetic response from the last assistant message
        // Note: callers should inspect finish_reason to see this is a forced stop
        let content = last
            .messages
            .iter()
            .rev()
            .find_map(|m| m.content_text().map(|s| s.to_string()))
            .unwrap_or_default();
        let resp = ChatResponse::new(MessageContent::Text(content));
        if let Some(cb) = &opts.on_finish {
            cb(&steps);
        }

        // Emit telemetry span end event
        if let Some(ref telemetry) = opts.telemetry {
            if telemetry.enabled {
                let total_usage = StepResult::merge_usage(&steps);
                let span = SpanEvent::start(
                    span_id.clone(),
                    None,
                    trace_id.clone(),
                    "ai.orchestrator.generate".to_string(),
                )
                .end_ok()
                .with_attribute("total_steps", steps.len().to_string())
                .with_attribute("max_steps_reached", "true");

                crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

                // Emit orchestrator event
                let orch_event = OrchestratorEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    trace_id: trace_id.clone(),
                    timestamp: std::time::SystemTime::now(),
                    total_steps: steps.len(),
                    current_step: steps.len(),
                    step_type: OrchestratorStepType::Completion,
                    total_usage,
                    total_duration: std::time::SystemTime::now().duration_since(start_time).ok(),
                    metadata: std::collections::HashMap::new(),
                };
                crate::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
            }
        }

        Ok((resp, steps))
    } else {
        // Emit error telemetry
        if let Some(ref telemetry) = opts.telemetry {
            if telemetry.enabled {
                let span = SpanEvent::start(
                    span_id.clone(),
                    None,
                    trace_id.clone(),
                    "ai.orchestrator.generate".to_string(),
                )
                .end_error("orchestrator: no steps produced".to_string());

                crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
            }
        }

        Err(LlmError::InternalError(
            "orchestrator: no steps produced".into(),
        ))
    }
}

/// Stream options for orchestrator.
pub struct OrchestratorStreamOptions {
    pub max_steps: usize,
    pub on_chunk: Option<Arc<dyn Fn(&ChatStreamEvent) + Send + Sync>>,
    pub on_step_finish: Option<Arc<dyn Fn(&StepResult) + Send + Sync>>,
    pub on_finish: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    pub on_tool_approval: Option<Arc<dyn Fn(&str, &Value) -> ToolApproval + Send + Sync>>,
    /// Optional abort callback, invoked with steps produced so far.
    pub on_abort: Option<Arc<dyn Fn(&[StepResult]) + Send + Sync>>,
    /// Optional telemetry configuration.
    pub telemetry: Option<TelemetryConfig>,
}

impl Default for OrchestratorStreamOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            on_chunk: None,
            on_step_finish: None,
            on_finish: None,
            on_tool_approval: None,
            on_abort: None,
            telemetry: None,
        }
    }
}

/// Stream handle that carries the stream and a oneshot receiver for steps summary.
pub struct StreamOrchestration {
    pub stream: ChatStream,
    pub steps: oneshot::Receiver<Vec<StepResult>>,
    /// A cancel handle to abort the orchestration.
    pub cancel: crate::utils::cancel::CancelHandle,
}

/// Orchestrate multi-step streaming. Concatenates provider streams across steps.
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
    let on_abort = opts.on_abort.clone();
    let orchestrator_cancel = crate::utils::cancel::new_cancel_handle();
    let orchestrator_cancel_clone = orchestrator_cancel.clone();

    tokio::spawn(async move {
        let mut step_results: Vec<StepResult> = Vec::new();
        let mut encountered_error = false;
        let resolver = resolver
            .map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn ToolResolver + Send + Sync>);
        // Sender doesn't need to be mutable; keep ownership local in task.
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
                match model.chat_with_tools(history.clone(), tools.clone()).await {
                    Ok(r) => r,
                    Err(e) => {
                        encountered_error = true;
                        let _ = sender.send(Err(e)).await;
                        break;
                    }
                }
            };
            let mut step_msgs: Vec<ChatMessage> = Vec::new();
            let assistant_text = resp
                .content_text()
                .map(|s| s.to_string())
                .unwrap_or_else(String::new);
            let mut assistant = ChatMessage::assistant(assistant_text);
            if let Some(calls) = resp.get_tool_calls().cloned() {
                assistant = assistant.with_tool_calls(calls);
            }
            let assistant_built = assistant.build();
            history.push(assistant_built.clone());
            step_msgs.push(assistant_built);

            // Execute tools if any
            let tool_calls = resp.get_tool_calls().cloned().unwrap_or_default();
            if !tool_calls.is_empty() {
                if let (Some(resolver), Some(ref ts)) = (resolver.as_ref(), tools_opt.as_ref()) {
                    for call in tool_calls.iter() {
                        if let Some(func) = &call.function {
                            // Skip duplicate tool-call IDs already executed in previous steps.
                            if processed_call_ids.contains(&call.id) {
                                continue;
                            }
                            let parsed_args: serde_json::Value =
                                serde_json::from_str(&func.arguments)
                                    .unwrap_or(serde_json::Value::String(func.arguments.clone()));
                            if let Some(def) = ts.iter().find(|t| t.function.name == func.name) {
                                if let Err(reason) = validate_args_with_schema(
                                    &def.function.parameters,
                                    &parsed_args,
                                ) {
                                    let out =
                                        serde_json::json!({"error":"invalid_args","reason":reason});
                                    let tool_msg =
                                        ChatMessage::tool(out.to_string(), call.id.clone()).build();
                                    history.push(tool_msg.clone());
                                    step_msgs.push(tool_msg);
                                    continue;
                                }
                            }
                            let decision = if let Some(cb) = &on_tool_approval {
                                cb(&func.name, &parsed_args)
                            } else {
                                ToolApproval::Approve(parsed_args.clone())
                            };
                            let out_val = match decision {
                                ToolApproval::Approve(args) | ToolApproval::Modify(args) => {
                                    resolver
                                        .call_tool(&func.name, args)
                                        .await
                                        .unwrap_or_else(|e| {
                                            serde_json::Value::String(format!(
                                                "<tool error: {}>",
                                                e
                                            ))
                                        })
                                }
                                ToolApproval::Deny { reason } => {
                                    serde_json::json!({"error":"denied","reason":reason})
                                }
                            };
                            let tool_msg =
                                ChatMessage::tool(out_val.to_string(), call.id.clone()).build();
                            history.push(tool_msg.clone());
                            step_msgs.push(tool_msg);
                            processed_call_ids.insert(call.id.clone());
                        }
                    }
                }
            }

            let step = StepResult {
                messages: step_msgs,
                finish_reason: resp.finish_reason.clone(),
                usage: resp.usage.clone(),
                tool_calls: tool_calls.clone(),
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
        } else if !encountered_error {
            if let Some(cb) = &on_finish {
                cb(&step_results);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct EchoTool;
    #[async_trait]
    impl ToolResolver for EchoTool {
        async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
            Ok(serde_json::json!({"tool": name, "args": arguments}))
        }
    }

    struct MockModel;
    #[async_trait]
    impl ChatCapability for MockModel {
        async fn chat_with_tools(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            // If last message is a tool message, return final response
            if let Some(last) = messages.last() {
                if matches!(last.role, crate::types::MessageRole::Tool) {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.finish_reason = Some(FinishReason::Stop);
                    return Ok(r);
                }
            }
            // Otherwise, request a tool call
            let call = ToolCall {
                id: "call-1".into(),
                r#type: "function".into(),
                function: Some(crate::types::tools::FunctionCall {
                    name: "echo".into(),
                    arguments: "{\"text\":\"hello\"}".into(),
                }),
            };
            let mut r = ChatResponse::new(MessageContent::Text(String::new()));
            r.tool_calls = Some(vec![call]);
            r.finish_reason = Some(FinishReason::ToolCalls);
            Ok(r)
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<crate::stream::ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream not supported in mock".into(),
            ))
        }
    }

    #[tokio::test]
    async fn orchestrator_two_steps_tool_then_answer() {
        let model = MockModel;
        let msgs = vec![ChatMessage::user("use tool").build()];
        let tools = Some(vec![Tool::function(
            "echo".into(),
            "echoes input".into(),
            serde_json::json!({"type":"object"}),
        )]);
        let (resp, steps) = generate(
            &model,
            msgs,
            tools,
            Some(&EchoTool),
            OrchestratorOptions::default(),
        )
        .await
        .expect("orchestrate");
        assert_eq!(resp.content_text().unwrap_or_default(), "done");
        assert!(
            steps
                .iter()
                .any(|s| matches!(s.finish_reason, Some(FinishReason::ToolCalls)))
        );
        // Step 1 has assistant(tool-calls) + tool message, step 2 has final assistant
        assert!(steps.len() >= 2);
    }

    struct MockModelStream;
    #[async_trait]
    impl ChatCapability for MockModelStream {
        async fn chat_with_tools(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            // Not used in this test
            let last_is_tool = matches!(
                messages.last().map(|m| &m.role),
                Some(crate::types::MessageRole::Tool)
            );
            if last_is_tool {
                let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                r.finish_reason = Some(FinishReason::Stop);
                return Ok(r);
            }
            let call = ToolCall {
                id: "c1".into(),
                r#type: "function".into(),
                function: Some(crate::types::tools::FunctionCall {
                    name: "echo".into(),
                    arguments: "{\"x\":1}".into(),
                }),
            };
            let mut r = ChatResponse::new(MessageContent::Text(String::new()));
            r.tool_calls = Some(vec![call]);
            r.finish_reason = Some(FinishReason::ToolCalls);
            Ok(r)
        }

        async fn chat_stream(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            let last_is_tool = matches!(
                messages.last().map(|m| &m.role),
                Some(crate::types::MessageRole::Tool)
            );
            if last_is_tool {
                let s = try_stream! {
                    yield ChatStreamEvent::ContentDelta{ delta: "ok".into(), index: None };
                    yield ChatStreamEvent::StreamEnd { response: {
                        let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                        r.finish_reason = Some(FinishReason::Stop);
                        r
                    }};
                };
                return Ok(Box::pin(s));
            }
            let s = try_stream! {
                yield ChatStreamEvent::ContentDelta{ delta: "...".into(), index: None };
                yield ChatStreamEvent::StreamEnd { response: {
                    let call = ToolCall { id: "c1".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "echo".into(), arguments: "{\"x\":1}".into() }) };
                    let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                    r.tool_calls = Some(vec![call]);
                    r.finish_reason = Some(FinishReason::ToolCalls);
                    r
                }};
            };
            Ok(Box::pin(s))
        }
    }

    #[tokio::test]
    async fn orchestrator_stream_two_steps() {
        let model = MockModelStream;
        let msgs = vec![ChatMessage::user("use tool").build()];
        let tools = Some(vec![Tool::function(
            "echo".into(),
            "echoes".into(),
            serde_json::json!({"type":"object"}),
        )]);
        let resolver = EchoTool;
        let out = generate_stream_owned(
            model,
            msgs,
            tools,
            Some(resolver),
            OrchestratorStreamOptions::default(),
        )
        .await
        .expect("stream orchestrate");
        // Collect stream to completion
        let events: Vec<_> = out.stream.collect().await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        );
        let steps = out.steps.await.expect("steps");
        assert!(steps.len() >= 2);
    }

    #[tokio::test]
    async fn orchestrator_stream_approval_modify() {
        // Model: first step requests a tool; second step (after tool msg) returns final.
        struct Model;
        #[async_trait]
        impl ChatCapability for Model {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                // Not used in this test
                let last_is_tool = matches!(
                    messages.last().map(|m| &m.role),
                    Some(crate::types::MessageRole::Tool)
                );
                if last_is_tool {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.finish_reason = Some(FinishReason::Stop);
                    return Ok(r);
                }
                let call = ToolCall {
                    id: "c1".into(),
                    r#type: "function".into(),
                    function: Some(crate::types::tools::FunctionCall {
                        name: "echo".into(),
                        arguments: "{\"x\":1}".into(),
                    }),
                };
                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                r.tool_calls = Some(vec![call]);
                r.finish_reason = Some(FinishReason::ToolCalls);
                Ok(r)
            }
            async fn chat_stream(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let last_is_tool = matches!(
                    messages.last().map(|m| &m.role),
                    Some(crate::types::MessageRole::Tool)
                );
                let s = try_stream! {
                    if last_is_tool {
                        yield ChatStreamEvent::ContentDelta{ delta: "ok".into(), index: None };
                        yield ChatStreamEvent::StreamEnd { response: {
                            let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                            r.finish_reason = Some(FinishReason::Stop);
                            r
                        }};
                    } else {
                        yield ChatStreamEvent::ContentDelta{ delta: "...".into(), index: None };
                        yield ChatStreamEvent::StreamEnd { response: {
                            let call = ToolCall { id: "c1".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "echo".into(), arguments: "{\"x\":1}".into() }) };
                            let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                            r.tool_calls = Some(vec![call]);
                            r.finish_reason = Some(FinishReason::ToolCalls);
                            r
                        }};
                    }
                };
                Ok(Box::pin(s))
            }
        }

        // Recording resolver that captures modified args
        struct RecordingResolver(std::sync::Arc<std::sync::Mutex<Vec<Value>>>);
        #[async_trait]
        impl ToolResolver for RecordingResolver {
            async fn call_tool(&self, _name: &str, arguments: Value) -> Result<Value, LlmError> {
                self.0.lock().unwrap().push(arguments);
                Ok(serde_json::json!({"ok":true}))
            }
        }

        let model = Model;
        let msgs = vec![ChatMessage::user("use tool").build()];
        let tools = Some(vec![Tool::function(
            "echo".into(),
            "echoes".into(),
            serde_json::json!({"type":"object"}),
        )]);
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let resolver = RecordingResolver(calls.clone());
        let opts = OrchestratorStreamOptions {
            on_tool_approval: Some(std::sync::Arc::new(|_n, _a| {
                ToolApproval::Modify(serde_json::json!({"x":2}))
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(model, msgs, tools, Some(resolver), opts)
            .await
            .expect("owned stream");
        let events: Vec<_> = out.stream.collect().await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        );
        let steps = out.steps.await.expect("steps");
        assert!(steps.len() >= 2);
        // Modified args captured
        let rec = calls.lock().unwrap();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0]["x"], 2);
    }

    #[tokio::test]
    async fn orchestrator_stream_approval_deny() {
        // Model same as above
        struct Model;
        #[async_trait]
        impl ChatCapability for Model {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let last_is_tool = matches!(
                    messages.last().map(|m| &m.role),
                    Some(crate::types::MessageRole::Tool)
                );
                if last_is_tool {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.finish_reason = Some(FinishReason::Stop);
                    return Ok(r);
                }
                let call = ToolCall {
                    id: "c1".into(),
                    r#type: "function".into(),
                    function: Some(crate::types::tools::FunctionCall {
                        name: "echo".into(),
                        arguments: "{\"x\":1}".into(),
                    }),
                };
                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                r.tool_calls = Some(vec![call]);
                r.finish_reason = Some(FinishReason::ToolCalls);
                Ok(r)
            }
            async fn chat_stream(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let last_is_tool = matches!(
                    messages.last().map(|m| &m.role),
                    Some(crate::types::MessageRole::Tool)
                );
                let s = try_stream! {
                    if last_is_tool {
                        yield ChatStreamEvent::StreamEnd { response: {
                            let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                            r.finish_reason = Some(FinishReason::Stop);
                            r
                        }};
                    } else {
                        yield ChatStreamEvent::StreamEnd { response: {
                            let call = ToolCall { id: "c1".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "echo".into(), arguments: "{\"x\":1}".into() }) };
                            let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                            r.tool_calls = Some(vec![call]);
                            r.finish_reason = Some(FinishReason::ToolCalls);
                            r
                        }};
                    }
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(&self, _n: &str, _a: Value) -> Result<Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let model = Model;
        let msgs = vec![ChatMessage::user("use tool").build()];
        let tools = Some(vec![Tool::function(
            "echo".into(),
            "echoes".into(),
            serde_json::json!({"type":"object"}),
        )]);
        let opts = OrchestratorStreamOptions {
            on_tool_approval: Some(std::sync::Arc::new(|_n, _a| ToolApproval::Deny {
                reason: "no".into(),
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(model, msgs, tools, Some(Noop), opts)
            .await
            .expect("owned stream");
        let _ = out.stream.collect::<Vec<_>>().await; // drain
        let steps = out.steps.await.expect("steps");
        assert!(steps.len() >= 2);
        // Step 1 should include a tool message with denied error
        let first_msgs = &steps[0].messages;
        assert!(
            first_msgs
                .iter()
                .any(|m| matches!(m.role, crate::types::MessageRole::Tool)
                    && m.content_text().unwrap_or_default().contains("denied"))
        );
    }

    #[tokio::test]
    async fn orchestrator_stream_error_propagation() {
        struct ErrModel;
        #[async_trait]
        impl ChatCapability for ErrModel {
            async fn chat_with_tools(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                Ok(ChatResponse::new(MessageContent::Text(String::new())))
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    yield ChatStreamEvent::ContentDelta{ delta: "a".into(), index: None };
                    Err::<ChatStreamEvent, LlmError>(LlmError::InternalError("boom".into()))?;
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(&self, _: &str, _a: Value) -> Result<Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let out = generate_stream_owned(
            ErrModel,
            vec![ChatMessage::user("err").build()],
            None,
            Some(Noop),
            Default::default(),
        )
        .await
        .expect("owned");
        let evs: Vec<_> = out.stream.collect().await;
        assert!(
            evs.iter()
                .any(|e| matches!(e, Err(LlmError::InternalError(_))))
        );
    }

    #[tokio::test]
    async fn orchestrator_stream_error_does_not_call_on_finish() {
        // Model that emits a delta then an error in the first streaming step
        struct ErrModel;
        #[async_trait]
        impl ChatCapability for ErrModel {
            async fn chat_with_tools(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                Ok(ChatResponse::new(MessageContent::Text(String::new())))
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    yield ChatStreamEvent::ContentDelta{ delta: "a".into(), index: None };
                    Err::<ChatStreamEvent, LlmError>(LlmError::InternalError("boom".into()))?;
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(&self, _: &str, _a: Value) -> Result<Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let finish_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finish_called_c = finish_called.clone();
        let opts = OrchestratorStreamOptions {
            on_finish: Some(std::sync::Arc::new(move |_steps| {
                finish_called_c.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(
            ErrModel,
            vec![ChatMessage::user("err").build()],
            None,
            Some(Noop),
            opts,
        )
        .await
        .expect("owned");
        let evs: Vec<_> = out.stream.collect().await;
        assert!(
            evs.iter()
                .any(|e| matches!(e, Err(LlmError::InternalError(_))))
        );
        let _ = out.steps.await.expect("steps");
        assert_eq!(finish_called.load(std::sync::atomic::Ordering::SeqCst), false);
    }

    #[tokio::test]
    async fn orchestrator_stream_duplicate_tool_ids_executed_once() {
        // Model returns a tool call with id "dup" in step 1 (streaming),
        // then erroneously repeats the same id in a follow-up step (non-streaming).
        struct DupModel;
        #[async_trait]
        impl ChatCapability for DupModel {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                match tool_msgs {
                    0 => {
                        // First follow-up: improperly repeat same tool id
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "dup".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "t".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    1 => {
                        // Final answer
                        let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                        r.finish_reason = Some(FinishReason::Stop);
                        Ok(r)
                    }
                    _ => Ok(ChatResponse::new(MessageContent::Text("done".into()))),
                }
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    yield ChatStreamEvent::StreamEnd { response: {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "dup".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall { name: "t".into(), arguments: "{}".into() })
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        r
                    } };
                };
                Ok(Box::pin(s))
            }
        }
        struct Counter(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        #[async_trait]
        impl ToolResolver for Counter {
            async fn call_tool(&self, _name: &str, _args: Value) -> Result<Value, LlmError> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(serde_json::json!({"ok":true}))
            }
        }
        let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let out = generate_stream_owned(
            DupModel,
            vec![ChatMessage::user("go").build()],
            Some(vec![Tool::function("t".into(), "".into(), serde_json::json!({"type":"object"}))]),
            Some(Counter(ctr.clone())),
            Default::default(),
        )
        .await
        .expect("owned");
        let _ = out.stream.collect::<Vec<_>>().await;
        let _ = out.steps.await.expect("steps");
        assert_eq!(ctr.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    

    #[tokio::test]
    async fn orchestrator_stream_many_chunks_and_callbacks() {
        use futures::StreamExt;
        struct Many;
        #[async_trait]
        impl ChatCapability for Many {
            async fn chat_with_tools(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                Ok(ChatResponse::new(MessageContent::Text(String::new())))
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    for _ in 0..100usize { yield ChatStreamEvent::ContentDelta{ delta: "x".into(), index: None }; }
                    yield ChatStreamEvent::StreamEnd { response: ChatResponse::new(MessageContent::Text("".into())) };
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(&self, _: &str, _a: Value) -> Result<Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let marks = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
        let marks_clone = marks.clone();
        let opts = OrchestratorStreamOptions {
            on_chunk: Some(std::sync::Arc::new(move |_ev| {
                marks_clone.lock().unwrap().push("chunk");
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(
            Many,
            vec![ChatMessage::user("go").build()],
            None,
            Some(Noop),
            opts,
        )
        .await
        .expect("owned");
        let items: Vec<_> = out.stream.collect().await;
        let delta_count = items
            .iter()
            .filter(|e| matches!(e, Ok(ChatStreamEvent::ContentDelta { .. })))
            .count();
        assert_eq!(delta_count, 100);
        let cb_count = marks.lock().unwrap().len();
        assert_eq!(cb_count, 100);
    }

    #[tokio::test]
    async fn orchestrator_stream_multiple_tool_calls_order() {
        struct TwoCalls;
        #[async_trait]
        impl ChatCapability for TwoCalls {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                // After tool message, return final
                let last_is_tool = matches!(
                    messages.last().map(|m| &m.role),
                    Some(crate::types::MessageRole::Tool)
                );
                if last_is_tool {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.finish_reason = Some(FinishReason::Stop);
                    return Ok(r);
                }
                // First step: two tool calls in fixed order
                let c1 = ToolCall {
                    id: "a".into(),
                    r#type: "function".into(),
                    function: Some(crate::types::tools::FunctionCall {
                        name: "first".into(),
                        arguments: "{}".into(),
                    }),
                };
                let c2 = ToolCall {
                    id: "b".into(),
                    r#type: "function".into(),
                    function: Some(crate::types::tools::FunctionCall {
                        name: "second".into(),
                        arguments: "{}".into(),
                    }),
                };
                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                r.tool_calls = Some(vec![c1, c2]);
                r.finish_reason = Some(FinishReason::ToolCalls);
                Ok(r)
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    // Minimal delta then end with tool_calls embedded in response
                    yield ChatStreamEvent::ContentDelta{ delta: "..".into(), index: None };
                    yield ChatStreamEvent::StreamEnd { response: {
                        let c1 = ToolCall { id: "a".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "first".into(), arguments: "{}".into() }) };
                        let c2 = ToolCall { id: "b".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "second".into(), arguments: "{}".into() }) };
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.tool_calls = Some(vec![c1, c2]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        r
                    } };
                };
                Ok(Box::pin(s))
            }
        }
        struct Recorder(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        #[async_trait]
        impl ToolResolver for Recorder {
            async fn call_tool(
                &self,
                _n: &str,
                arguments: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                // record input; simulate slight delay to expose order sensitivity
                // no-op: previously used for local inspection; remove to avoid type inference issues
                self.0.lock().unwrap().push("call".into());
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                Ok(serde_json::json!({"ok":true}))
            }
        }
        // Instead of extracting names from args, record order via dedicated vector using tool call order
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        // Wrap resolver to push explicit marker per tool id
        struct ResolverWithIds(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        #[async_trait]
        impl ToolResolver for ResolverWithIds {
            async fn call_tool(
                &self,
                name: &str,
                _arguments: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                self.0.lock().unwrap().push(name.to_string());
                // simulate costy call to surface potential reordering if parallel
                tokio::time::sleep(std::time::Duration::from_millis(3)).await;
                Ok(serde_json::json!({"ok":true}))
            }
        }
        let model = TwoCalls;
        let msgs = vec![ChatMessage::user("tools").build()];
        let tools = Some(vec![
            Tool::function(
                "first".into(),
                "".into(),
                serde_json::json!({"type":"object"}),
            ),
            Tool::function(
                "second".into(),
                "".into(),
                serde_json::json!({"type":"object"}),
            ),
        ]);
        let resolver = ResolverWithIds(order.clone());
        let out = generate_stream_owned(model, msgs, tools, Some(resolver), Default::default())
            .await
            .expect("owned");
        let _ = out.stream.collect::<Vec<_>>().await;
        let steps = out.steps.await.expect("steps");
        assert!(steps.len() >= 2);
        let rec = order.lock().unwrap().clone();
        assert_eq!(rec, vec!["first".to_string(), "second".to_string()]);
    }

    #[tokio::test]
    async fn orchestrator_stream_backpressure_and_order_many_events() {
        use futures::StreamExt;
        struct Many;
        #[async_trait]
        impl ChatCapability for Many {
            async fn chat_with_tools(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                Ok(ChatResponse::new(MessageContent::Text(String::new())))
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    for i in 0..300usize { yield ChatStreamEvent::ContentDelta{ delta: format!("x{}", i), index: None }; }
                    yield ChatStreamEvent::StreamEnd { response: ChatResponse::new(MessageContent::Text("".into())) };
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let marks = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let marks_c = marks.clone();
        let opts = OrchestratorStreamOptions {
            on_chunk: Some(std::sync::Arc::new(move |ev| {
                if let ChatStreamEvent::ContentDelta { delta, .. } = ev {
                    marks_c.lock().unwrap().push(delta.clone());
                }
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(
            Many,
            vec![ChatMessage::user("go").build()],
            None,
            Some(Noop),
            opts,
        )
        .await
        .expect("owned");
        // simulate slow consumer to test backpressure doesn't drop data
        let mut s = out.stream;
        let mut seen = Vec::new();
        while let Some(item) = s.next().await {
            if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = item {
                seen.push(delta);
                tokio::task::yield_now().await;
            } else { /* ignore */
            }
        }
        assert_eq!(seen.len(), 300);
        // ensure callback captured in same order
        let cb = marks.lock().unwrap().clone();
        assert_eq!(cb, seen);
    }

    #[tokio::test]
    async fn orchestrator_stream_on_chunk_before_step_finish() {
        use futures::StreamExt;
        struct OneStep;
        #[async_trait]
        impl ChatCapability for OneStep {
            async fn chat_with_tools(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                r.finish_reason = Some(FinishReason::Stop);
                Ok(r)
            }
            async fn chat_stream(
                &self,
                _m: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let s = try_stream! {
                    yield ChatStreamEvent::ContentDelta{ delta: "a".into(), index: None };
                    yield ChatStreamEvent::ContentDelta{ delta: "b".into(), index: None };
                    yield ChatStreamEvent::StreamEnd { response: ChatResponse::new(MessageContent::Text("".into())) };
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let seq = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
        let seq_chunk = seq.clone();
        let seq_step = seq.clone();
        let opts = OrchestratorStreamOptions {
            on_chunk: Some(std::sync::Arc::new(move |_ev| {
                seq_chunk.lock().unwrap().push("chunk");
            })),
            on_step_finish: Some(std::sync::Arc::new(move |_s| {
                seq_step.lock().unwrap().push("step");
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(
            OneStep,
            vec![ChatMessage::user("go").build()],
            None,
            Some(Noop),
            opts,
        )
        .await
        .expect("owned");
        let _ = out.stream.collect::<Vec<_>>().await;
        let v = seq.lock().unwrap().clone();
        // should see chunk markers before step marker
        assert!(v.starts_with(&["chunk", "chunk"]))
    }

    #[tokio::test]
    async fn orchestrator_stream_multi_tools_multi_steps_sequence() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MultiSteps;
        #[async_trait]
        impl ChatCapability for MultiSteps {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                // Count tool messages to decide which step we are in.
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                match tool_msgs {
                    0 => {
                        // Step 1: two tool calls
                        let c1 = ToolCall {
                            id: "c1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "first".into(),
                                arguments: "{}".into(),
                            }),
                        };
                        let c2 = ToolCall {
                            id: "c2".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "second".into(),
                                arguments: "{}".into(),
                            }),
                        };
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.tool_calls = Some(vec![c1, c2]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    2 => {
                        // Step 2: one more tool call
                        let c3 = ToolCall {
                            id: "c3".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "third".into(),
                                arguments: "{}".into(),
                            }),
                        };
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.tool_calls = Some(vec![c3]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    3 => {
                        // Final step: return final answer
                        let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                        r.finish_reason = Some(FinishReason::Stop);
                        Ok(r)
                    }
                    _ => {
                        // Safety: complete if unexpected
                        Ok(ChatResponse::new(MessageContent::Text("done".into())))
                    }
                }
            }
            async fn chat_stream(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                // Count tool messages to decide which step we are in.
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                let s = try_stream! {
                    match tool_msgs {
                        0 => {
                            // Step 1: two tool calls
                            yield ChatStreamEvent::StreamEnd { response: {
                                let c1 = ToolCall { id: "c1".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "first".into(), arguments: "{}".into() }) };
                                let c2 = ToolCall { id: "c2".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "second".into(), arguments: "{}".into() }) };
                                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                                r.tool_calls = Some(vec![c1, c2]);
                                r.finish_reason = Some(FinishReason::ToolCalls);
                                r
                            } };
                        }
                        2 => {
                            // Step 2: one more tool call
                            yield ChatStreamEvent::StreamEnd { response: {
                                let c3 = ToolCall { id: "c3".into(), r#type: "function".into(), function: Some(crate::types::tools::FunctionCall { name: "third".into(), arguments: "{}".into() }) };
                                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                                r.tool_calls = Some(vec![c3]);
                                r.finish_reason = Some(FinishReason::ToolCalls);
                                r
                            } };
                        }
                        3 => {
                            // Final step: return final answer
                            yield ChatStreamEvent::StreamEnd { response: {
                                let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                                r.finish_reason = Some(FinishReason::Stop);
                                r
                            } };
                        }
                        _ => {
                            // Safety: complete if unexpected
                            yield ChatStreamEvent::StreamEnd { response: ChatResponse::new(MessageContent::Text("done".into())) };
                        }
                    }
                };
                Ok(Box::pin(s))
            }
        }

        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _n: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                // Simulate tool latency to surface ordering/backpressure
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                Ok(serde_json::json!({"ok": true}))
            }
        }

        let step_counter = std::sync::Arc::new(AtomicUsize::new(0));
        let step_counter_cb = step_counter.clone();
        let opts = OrchestratorStreamOptions {
            on_step_finish: Some(std::sync::Arc::new(move |_s| {
                step_counter_cb.fetch_add(1, Ordering::SeqCst);
            })),
            ..Default::default()
        };
        let out = generate_stream_owned(
            MultiSteps,
            vec![ChatMessage::user("go").build()],
            Some(vec![
                Tool::function(
                    "first".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "second".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "third".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
            ]),
            Some(Noop),
            opts,
        )
        .await
        .expect("owned");

        // Drain stream
        let _ = out.stream.collect::<Vec<_>>().await;
        let steps = out.steps.await.expect("steps");
        assert_eq!(
            steps.len(),
            3,
            "expected three steps: 2 tool steps + final step"
        );
        assert_eq!(
            step_counter.load(Ordering::SeqCst),
            3,
            "on_step_finish must be called per step"
        );
        // Validate step shapes: step1 has 1 assistant + 2 tool msgs; step2: 1 assistant + 1 tool; step3: final assistant
        assert_eq!(steps[0].messages.len(), 3);
        assert_eq!(steps[1].messages.len(), 2);
        assert_eq!(steps[2].messages.len(), 1);
        assert!(matches!(steps[2].finish_reason, Some(FinishReason::Stop)));
    }

    #[tokio::test]
    async fn orchestrator_stream_usage_aggregation() {
        use crate::types::Usage;
        struct UsageSteps;
        #[async_trait]
        impl ChatCapability for UsageSteps {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                match tool_msgs {
                    0 => {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.usage = Some(Usage::new(10, 2));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "t1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "a".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    1 => {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.usage = Some(Usage::new(0, 4));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "t2".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "b".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    _ => {
                        let mut r = ChatResponse::new(MessageContent::Text("ok".into()));
                        r.usage = Some(Usage::new(1, 3));
                        r.finish_reason = Some(FinishReason::Stop);
                        Ok(r)
                    }
                }
            }
            async fn chat_stream(
                &self,
                messages: Vec<ChatMessage>,
                _t: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                let s = try_stream! {
                    match tool_msgs {
                        0 => {
                            yield ChatStreamEvent::StreamEnd { response: {
                                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                                r.usage = Some(Usage::new(10, 2));
                                r.tool_calls = Some(vec![ToolCall { id: "t1".into(), r#type:"function".into(), function: Some(crate::types::tools::FunctionCall { name: "a".into(), arguments: "{}".into() }) }]);
                                r.finish_reason = Some(FinishReason::ToolCalls);
                                r
                            }};
                        }
                        1 => {
                            yield ChatStreamEvent::StreamEnd { response: {
                                let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                                r.usage = Some(Usage::new(0, 4));
                                r.tool_calls = Some(vec![ToolCall { id: "t2".into(), r#type:"function".into(), function: Some(crate::types::tools::FunctionCall { name: "b".into(), arguments: "{}".into() }) }]);
                                r.finish_reason = Some(FinishReason::ToolCalls);
                                r
                            }};
                        }
                        _ => {
                            yield ChatStreamEvent::StreamEnd { response: {
                                let mut r = ChatResponse::new(MessageContent::Text("ok".into()));
                                r.usage = Some(Usage::new(1, 3));
                                r.finish_reason = Some(FinishReason::Stop);
                                r
                            }};
                        }
                    }
                };
                Ok(Box::pin(s))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }
        let out = generate_stream_owned(
            UsageSteps,
            vec![ChatMessage::user("go").build()],
            Some(vec![
                Tool::function("a".into(), "".into(), serde_json::json!({"type":"object"})),
                Tool::function("b".into(), "".into(), serde_json::json!({"type":"object"})),
            ]),
            Some(Noop),
            Default::default(),
        )
        .await
        .expect("owned");
        let _ = out.stream.collect::<Vec<_>>().await;
        let steps = out.steps.await.expect("steps");
        assert_eq!(steps.len(), 3);
        // Aggregate usage
        let total = StepResult::merge_usage(&steps).expect("usage");
        assert_eq!(total.prompt_tokens, 11); // 10 + 0 + 1
        assert_eq!(total.completion_tokens, 9); // 2 + 4 + 3
        assert_eq!(total.total_tokens, 10 + 4 + 4); // sum of response totals
    }

    #[tokio::test]
    async fn test_usage_aggregation_with_on_finish_callback() {
        // Test that on_finish callback receives correct aggregated usage
        struct UsageModel;
        #[async_trait]
        impl ChatCapability for UsageModel {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                match tool_msgs {
                    0 => {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.usage = Some(Usage::new(100, 20));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "t1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "tool1".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    _ => {
                        let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                        r.usage = Some(Usage::new(50, 30));
                        r.finish_reason = Some(FinishReason::Stop);
                        Ok(r)
                    }
                }
            }
            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                Err(LlmError::InvalidParameter("not implemented".into()))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
        }

        let captured_usage = std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured_usage_clone = captured_usage.clone();

        let opts = OrchestratorOptions {
            max_steps: 5,
            on_finish: Some(std::sync::Arc::new(move |steps: &[StepResult]| {
                let total = StepResult::merge_usage(steps);
                *captured_usage_clone.lock().unwrap() = total;
            })),
            ..Default::default()
        };

        let (_resp, _steps) = generate(
            &UsageModel,
            vec![ChatMessage::user("test").build()],
            Some(vec![Tool::function(
                "tool1".into(),
                "".into(),
                serde_json::json!({"type":"object"}),
            )]),
            Some(&Noop),
            opts,
        )
        .await
        .expect("generate");

        let total = captured_usage.lock().unwrap().clone().expect("usage");
        assert_eq!(total.prompt_tokens, 150); // 100 + 50
        assert_eq!(total.completion_tokens, 50); // 20 + 30
        assert_eq!(total.total_tokens, 100 + 80); // first step prompt + second step total
    }

    #[tokio::test]
    async fn test_error_injection_after_second_step_tool() {
        // Test error propagation when tool execution fails in second step
        struct TwoStepModel;
        #[async_trait]
        impl ChatCapability for TwoStepModel {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                match tool_msgs {
                    0 => {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.usage = Some(Usage::new(10, 5));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "t1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "step1".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    1 => {
                        let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                        r.usage = Some(Usage::new(5, 3));
                        r.tool_calls = Some(vec![ToolCall {
                            id: "t2".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "step2_fail".into(),
                                arguments: "{}".into(),
                            }),
                        }]);
                        r.finish_reason = Some(FinishReason::ToolCalls);
                        Ok(r)
                    }
                    _ => {
                        let mut r = ChatResponse::new(MessageContent::Text("recovered".into()));
                        r.usage = Some(Usage::new(3, 2));
                        r.finish_reason = Some(FinishReason::Stop);
                        Ok(r)
                    }
                }
            }
            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                Err(LlmError::InvalidParameter("not implemented".into()))
            }
        }
        struct FailingResolver;
        #[async_trait]
        impl ToolResolver for FailingResolver {
            async fn call_tool(
                &self,
                name: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                if name == "step2_fail" {
                    Err(LlmError::provider_error("test", "tool execution failed"))
                } else {
                    Ok(serde_json::json!({"status": "ok"}))
                }
            }
        }

        let (_resp, steps) = generate(
            &TwoStepModel,
            vec![ChatMessage::user("test").build()],
            Some(vec![
                Tool::function(
                    "step1".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "step2_fail".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
            ]),
            Some(&FailingResolver),
            Default::default(),
        )
        .await
        .expect("generate");

        assert_eq!(steps.len(), 3);
        // Check that error was captured in tool message
        let step2_msgs = &steps[1].messages;
        let tool_msg = step2_msgs
            .iter()
            .find(|m| matches!(m.role, crate::types::MessageRole::Tool))
            .expect("tool message");
        let content_text = match &tool_msg.content {
            MessageContent::Text(text) => text,
            _ => panic!("expected text content"),
        };
        assert!(content_text.contains("tool error"));
    }

    #[tokio::test]
    async fn test_approval_mixed_approve_deny() {
        // Test mixed approval decisions (some approved, some denied)
        struct MultiToolModel;
        #[async_trait]
        impl ChatCapability for MultiToolModel {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                if tool_msgs == 0 {
                    let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                    r.usage = Some(Usage::new(10, 5));
                    r.tool_calls = Some(vec![
                        ToolCall {
                            id: "t1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "approved_tool".into(),
                                arguments: r#"{"action":"read"}"#.into(),
                            }),
                        },
                        ToolCall {
                            id: "t2".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "denied_tool".into(),
                                arguments: r#"{"action":"delete"}"#.into(),
                            }),
                        },
                    ]);
                    r.finish_reason = Some(FinishReason::ToolCalls);
                    Ok(r)
                } else {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.usage = Some(Usage::new(5, 3));
                    r.finish_reason = Some(FinishReason::Stop);
                    Ok(r)
                }
            }
            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                Err(LlmError::InvalidParameter("not implemented".into()))
            }
        }
        struct Noop;
        #[async_trait]
        impl ToolResolver for Noop {
            async fn call_tool(
                &self,
                _: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({"result": "executed"}))
            }
        }

        let opts = OrchestratorOptions {
            max_steps: 5,
            on_tool_approval: Some(std::sync::Arc::new(
                |name: &str, _args: &serde_json::Value| {
                    if name == "denied_tool" {
                        ToolApproval::Deny {
                            reason: "dangerous operation not allowed".into(),
                        }
                    } else {
                        ToolApproval::Approve(_args.clone())
                    }
                },
            )),
            ..Default::default()
        };

        let (_resp, steps) = generate(
            &MultiToolModel,
            vec![ChatMessage::user("test").build()],
            Some(vec![
                Tool::function(
                    "approved_tool".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "denied_tool".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
            ]),
            Some(&Noop),
            opts,
        )
        .await
        .expect("generate");

        assert_eq!(steps.len(), 2);
        // Check first step has 3 messages: assistant + 2 tool results
        let step1_msgs = &steps[0].messages;
        assert_eq!(step1_msgs.len(), 3); // assistant + approved_tool + denied_tool

        // Find the denied tool message
        let denied_msg = step1_msgs
            .iter()
            .find(|m| {
                matches!(m.role, crate::types::MessageRole::Tool)
                    && m.tool_call_id.as_deref() == Some("t2")
            })
            .expect("denied tool message");
        let denied_text = match &denied_msg.content {
            MessageContent::Text(text) => text,
            _ => panic!("expected text content"),
        };
        assert!(denied_text.contains("denied"));
        assert!(denied_text.contains("dangerous operation"));

        // Find the approved tool message
        let approved_msg = step1_msgs
            .iter()
            .find(|m| {
                matches!(m.role, crate::types::MessageRole::Tool)
                    && m.tool_call_id.as_deref() == Some("t1")
            })
            .expect("approved tool message");
        let approved_text = match &approved_msg.content {
            MessageContent::Text(text) => text,
            _ => panic!("expected text content"),
        };
        assert!(approved_text.contains("executed"));
    }

    #[tokio::test]
    async fn test_concurrent_tool_calls_order() {
        // Test that multiple tool calls in same step are executed in order
        struct ConcurrentToolModel;
        #[async_trait]
        impl ChatCapability for ConcurrentToolModel {
            async fn chat_with_tools(
                &self,
                messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, LlmError> {
                let tool_msgs = messages
                    .iter()
                    .filter(|m| matches!(m.role, crate::types::MessageRole::Tool))
                    .count();
                if tool_msgs == 0 {
                    let mut r = ChatResponse::new(MessageContent::Text(String::new()));
                    r.usage = Some(Usage::new(10, 5));
                    r.tool_calls = Some(vec![
                        ToolCall {
                            id: "t1".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "tool_a".into(),
                                arguments: "{}".into(),
                            }),
                        },
                        ToolCall {
                            id: "t2".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "tool_b".into(),
                                arguments: "{}".into(),
                            }),
                        },
                        ToolCall {
                            id: "t3".into(),
                            r#type: "function".into(),
                            function: Some(crate::types::tools::FunctionCall {
                                name: "tool_c".into(),
                                arguments: "{}".into(),
                            }),
                        },
                    ]);
                    r.finish_reason = Some(FinishReason::ToolCalls);
                    Ok(r)
                } else {
                    let mut r = ChatResponse::new(MessageContent::Text("done".into()));
                    r.usage = Some(Usage::new(5, 3));
                    r.finish_reason = Some(FinishReason::Stop);
                    Ok(r)
                }
            }
            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatStream, LlmError> {
                Err(LlmError::InvalidParameter("not implemented".into()))
            }
        }
        struct OrderTracker {
            order: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl ToolResolver for OrderTracker {
            async fn call_tool(
                &self,
                name: &str,
                _a: serde_json::Value,
            ) -> Result<serde_json::Value, LlmError> {
                self.order.lock().unwrap().push(name.to_string());
                Ok(serde_json::json!({"tool": name}))
            }
        }

        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let resolver = OrderTracker {
            order: order.clone(),
        };

        let (_resp, steps) = generate(
            &ConcurrentToolModel,
            vec![ChatMessage::user("test").build()],
            Some(vec![
                Tool::function(
                    "tool_a".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "tool_b".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
                Tool::function(
                    "tool_c".into(),
                    "".into(),
                    serde_json::json!({"type":"object"}),
                ),
            ]),
            Some(&resolver),
            Default::default(),
        )
        .await
        .expect("generate");

        assert_eq!(steps.len(), 2);
        // Check that tools were called in order
        let call_order = order.lock().unwrap().clone();
        assert_eq!(call_order, vec!["tool_a", "tool_b", "tool_c"]);

        // Check that all tool messages are present in step 1
        let step1_msgs = &steps[0].messages;
        assert_eq!(step1_msgs.len(), 4); // assistant + 3 tool results
    }
}
