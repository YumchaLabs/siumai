//! Tool-loop gateway helpers.
//!
//! This module is intended for gateway/proxy use-cases:
//! - Stream from an upstream provider
//! - Detect tool calls
//! - Execute tools in-process
//! - Feed tool results back into the next model step
//! - Keep a single downstream stream open (only emit one StreamStart + one StreamEnd)
//!
//! English-only comments in code as requested.

use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;
use siumai::prelude::unified::*;
use tokio::sync::mpsc;

use crate::orchestrator::types::ToolResolver;

fn validate_args_with_schema(schema: &Value, instance: &Value) -> Result<(), String> {
    #[cfg(feature = "schema")]
    {
        if let Err(e) = crate::schema::validate_json(schema, instance) {
            return Err(e.to_string());
        }
    }

    let _ = (schema, instance);
    Ok(())
}

/// Options for tool-loop gateway streaming.
#[derive(Debug, Clone)]
pub struct ToolLoopGatewayOptions {
    /// Maximum tool-loop steps (tool-call rounds + final answer).
    pub max_steps: usize,
}

impl Default for ToolLoopGatewayOptions {
    fn default() -> Self {
        Self { max_steps: 8 }
    }
}

#[derive(Debug, Default, Clone)]
struct ToolCallAcc {
    tool_name: Option<String>,
    args_json: String,
    provider_executed: Option<bool>,
    source: Option<ToolCallAccSource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallAccSource {
    LegacyDelta,
    StablePart,
}

fn parse_json_best_effort(s: &str) -> Value {
    let s = s.trim();
    if s.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

fn ensure_tool_call_acc<'a>(
    acc: &'a mut HashMap<String, ToolCallAcc>,
    order: &mut Vec<String>,
    id: &str,
    source: ToolCallAccSource,
) -> Option<&'a mut ToolCallAcc> {
    match acc.entry(id.to_string()) {
        Entry::Vacant(entry) => {
            order.push(id.to_string());
            let slot = entry.insert(ToolCallAcc {
                source: Some(source),
                ..ToolCallAcc::default()
            });
            Some(slot)
        }
        Entry::Occupied(entry) => {
            let slot = entry.into_mut();
            match slot.source {
                Some(existing) if existing != source => None,
                Some(_) => Some(slot),
                None => {
                    slot.source = Some(source);
                    Some(slot)
                }
            }
        }
    }
}

fn accumulate_runtime_tool_part(
    part: &ChatStreamPart,
    acc: &mut HashMap<String, ToolCallAcc>,
    order: &mut Vec<String>,
) -> bool {
    match part {
        ChatStreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            let Some(entry) = ensure_tool_call_acc(acc, order, id, ToolCallAccSource::StablePart)
            else {
                return true;
            };
            entry.tool_name = Some(tool_name.clone());
            entry.provider_executed = *provider_executed;
            true
        }
        ChatStreamPart::ToolInputDelta { id, delta, .. } => {
            let Some(entry) = ensure_tool_call_acc(acc, order, id, ToolCallAccSource::StablePart)
            else {
                return true;
            };
            entry.args_json.push_str(delta);
            true
        }
        ChatStreamPart::ToolCall(call) => {
            let Some(entry) = ensure_tool_call_acc(
                acc,
                order,
                &call.tool_call_id,
                ToolCallAccSource::StablePart,
            ) else {
                return true;
            };
            entry.tool_name = Some(call.tool_name.clone());
            entry.args_json = call.input.clone();
            entry.provider_executed = call.provider_executed;
            true
        }
        _ => false,
    }
}

fn accumulate_loose_tool_part(
    data: &Value,
    acc: &mut HashMap<String, ToolCallAcc>,
    order: &mut Vec<String>,
) -> bool {
    let Some(part) =
        siumai::experimental::streaming::LanguageModelV3StreamPart::parse_loose_json(data)
    else {
        return false;
    };

    match part {
        siumai::experimental::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            let Some(entry) = ensure_tool_call_acc(acc, order, &id, ToolCallAccSource::StablePart)
            else {
                return true;
            };
            entry.tool_name = Some(tool_name);
            entry.provider_executed = provider_executed;
            true
        }
        siumai::experimental::streaming::LanguageModelV3StreamPart::ToolInputDelta {
            id,
            delta,
            ..
        } => {
            let Some(entry) = ensure_tool_call_acc(acc, order, &id, ToolCallAccSource::StablePart)
            else {
                return true;
            };
            entry.args_json.push_str(&delta);
            true
        }
        siumai::experimental::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            let Some(entry) = ensure_tool_call_acc(
                acc,
                order,
                &call.tool_call_id,
                ToolCallAccSource::StablePart,
            ) else {
                return true;
            };
            entry.tool_name = Some(call.tool_name);
            entry.args_json = call.input;
            entry.provider_executed = call.provider_executed;
            true
        }
        _ => false,
    }
}

fn tool_calls_from_acc_ordered(
    mut acc: HashMap<String, ToolCallAcc>,
    order: Vec<String>,
) -> Vec<ContentPart> {
    let mut out = Vec::new();

    for id in order {
        let Some(item) = acc.remove(&id) else {
            continue;
        };
        let Some(name) = item.tool_name else {
            continue;
        };
        let args = parse_json_best_effort(&item.args_json);
        out.push(ContentPart::ToolCall {
            tool_call_id: id,
            tool_name: name,
            arguments: args,
            provider_executed: item.provider_executed,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        });
    }

    // Deterministic fallback for any remaining IDs (should be rare).
    let mut rest: Vec<_> = acc.into_iter().collect();
    rest.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (id, item) in rest {
        let Some(name) = item.tool_name else {
            continue;
        };
        let args = parse_json_best_effort(&item.args_json);
        out.push(ContentPart::ToolCall {
            tool_call_id: id,
            tool_name: name,
            arguments: args,
            provider_executed: item.provider_executed,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        });
    }

    out
}

fn v3_tool_result_event(
    tool_call_id: String,
    tool_name: String,
    result: Value,
    is_error: bool,
) -> ChatStreamEvent {
    ChatStreamEvent::Custom {
        event_type: "gateway:tool-result".to_string(),
        data: serde_json::json!({
            "type": "tool-result",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "result": result,
            "isError": is_error,
        }),
    }
}

/// Create a single `ChatStream` that keeps the downstream stream open across tool-loop steps.
///
/// This function:
/// - emits the upstream `StreamStart` only once (from the first step)
/// - suppresses intermediate `StreamEnd` events until the final step completes
/// - emits v3 `tool-result` events between steps so downstream protocols can surface results
pub async fn tool_loop_chat_stream(
    model: Arc<dyn ChatCapability + Send + Sync>,
    initial_messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    resolver: Arc<dyn ToolResolver + Send + Sync>,
    opts: ToolLoopGatewayOptions,
) -> Result<ChatStream, LlmError> {
    let max_steps = opts.max_steps.max(1);

    let (tx, rx) = mpsc::channel::<Result<ChatStreamEvent, LlmError>>(64);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

    // A Sync stream wrapper around mpsc::Receiver using a Mutex for interior mutability.
    struct MpscStream(
        std::sync::Arc<std::sync::Mutex<mpsc::Receiver<Result<ChatStreamEvent, LlmError>>>>,
    );

    impl futures::Stream for MpscStream {
        type Item = Result<ChatStreamEvent, LlmError>;
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            let mut guard = self.0.lock().unwrap();
            mpsc::Receiver::poll_recv(&mut *guard, cx)
        }
    }

    tokio::spawn(async move {
        let sender = tx;
        let mut history = initial_messages;
        let mut emitted_stream_start = false;
        let mut final_stream_end: Option<ChatResponse> = None;

        'outer: for _step_idx in 0..max_steps {
            let handle = match model
                .chat_stream_with_cancel(history.clone(), Some(tools.clone()))
                .await
            {
                Ok(h) => h,
                Err(e) => {
                    let _ = sender.send(Err(e)).await;
                    break 'outer;
                }
            };

            let mut upstream = handle.stream;
            let cancel = handle.cancel;

            let mut acc_text = String::new();
            let mut tool_calls_acc: HashMap<String, ToolCallAcc> = HashMap::new();
            let mut tool_call_order: Vec<String> = Vec::new();

            while let Some(item) = upstream.next().await {
                let ev = match item {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = sender.send(Err(e)).await;
                        cancel.cancel();
                        break 'outer;
                    }
                };

                match &ev {
                    ChatStreamEvent::StreamStart { .. } => {
                        if !emitted_stream_start {
                            emitted_stream_start = true;
                            if sender.send(Ok(ev)).await.is_err() {
                                cancel.cancel();
                                break 'outer;
                            }
                        }
                    }
                    ChatStreamEvent::ContentDelta { delta, .. } => {
                        acc_text.push_str(delta);
                        if sender.send(Ok(ev)).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                    ChatStreamEvent::ThinkingDelta { .. } => {
                        if sender.send(Ok(ev)).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                    ChatStreamEvent::ToolCallDelta {
                        id,
                        function_name,
                        arguments_delta,
                        ..
                    } => {
                        let Some(entry) = ensure_tool_call_acc(
                            &mut tool_calls_acc,
                            &mut tool_call_order,
                            id,
                            ToolCallAccSource::LegacyDelta,
                        ) else {
                            if sender.send(Ok(ev)).await.is_err() {
                                cancel.cancel();
                                break 'outer;
                            }
                            continue;
                        };
                        if let Some(name) = function_name.clone()
                            && !name.trim().is_empty()
                        {
                            entry.tool_name = Some(name);
                        }
                        if let Some(delta) = arguments_delta.clone() {
                            entry.args_json.push_str(&delta);
                        }
                        if sender.send(Ok(ev)).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                    ChatStreamEvent::Part { part }
                    | ChatStreamEvent::PartWithReplay { part, .. } => {
                        let _ = accumulate_runtime_tool_part(
                            part,
                            &mut tool_calls_acc,
                            &mut tool_call_order,
                        );
                        if sender.send(Ok(ev)).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                    ChatStreamEvent::Custom { data, .. } => {
                        let _ = accumulate_loose_tool_part(
                            data,
                            &mut tool_calls_acc,
                            &mut tool_call_order,
                        );
                        if sender.send(Ok(ev)).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                    ChatStreamEvent::StreamEnd { response } => {
                        final_stream_end = Some(response.clone());
                    }
                    other => {
                        if sender.send(Ok(other.clone())).await.is_err() {
                            cancel.cancel();
                            break 'outer;
                        }
                    }
                }
            }

            let tool_calls = tool_calls_from_acc_ordered(tool_calls_acc, tool_call_order);

            let mut assistant_parts: Vec<ContentPart> = Vec::new();
            if !acc_text.trim().is_empty() {
                assistant_parts.push(ContentPart::Text {
                    text: acc_text.clone(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                });
            }
            assistant_parts.extend(tool_calls.clone());

            // Always add the assistant message to history (tool-call steps require it).
            history.push(ChatMessage {
                role: MessageRole::Assistant,
                content: if assistant_parts.is_empty() {
                    MessageContent::Text(String::new())
                } else if assistant_parts.len() == 1
                    && matches!(assistant_parts[0], ContentPart::Text { .. })
                {
                    MessageContent::Text(acc_text)
                } else {
                    MessageContent::MultiModal(assistant_parts)
                },
                provider_options: ProviderOptionsMap::default(),
                metadata: MessageMetadata::default(),
            });

            // Execute tools (skip provider-executed tool calls).
            let mut executed_any = false;
            for call in tool_calls {
                let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    provider_executed,
                    ..
                } = call
                else {
                    continue;
                };

                if provider_executed.unwrap_or(false) {
                    continue;
                }

                // Validate args for function tools (best-effort).
                if let Some(def) = tools.iter().find(|t| match t {
                    Tool::Function { function } => function.name == tool_name,
                    Tool::ProviderDefined(_) => false,
                }) {
                    let schema = match def {
                        Tool::Function { function } => &function.parameters,
                        Tool::ProviderDefined(_) => continue,
                    };
                    if let Err(reason) = validate_args_with_schema(schema, &arguments) {
                        let err = serde_json::json!({"error":"invalid_args","reason":reason});
                        if sender
                            .send(Ok(v3_tool_result_event(
                                tool_call_id.clone(),
                                tool_name.clone(),
                                err.clone(),
                                true,
                            )))
                            .await
                            .is_err()
                        {
                            break 'outer;
                        }
                        history.push(
                            ChatMessage::tool_error_json(tool_call_id, tool_name, err).build(),
                        );
                        executed_any = true;
                        continue;
                    }
                }

                let (value, is_error) =
                    match resolver.call_tool(&tool_name, arguments.clone()).await {
                        Ok(v) => (v, false),
                        Err(e) => (
                            serde_json::json!({"error":"tool_error","message":e.user_message()}),
                            true,
                        ),
                    };

                if is_error {
                    history.push(
                        ChatMessage::tool_error_json(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            value.clone(),
                        )
                        .build(),
                    );
                } else {
                    history.push(
                        ChatMessage::tool_result_json(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            value.clone(),
                        )
                        .build(),
                    );
                }
                executed_any = true;

                if sender
                    .send(Ok(v3_tool_result_event(
                        tool_call_id,
                        tool_name,
                        value,
                        is_error,
                    )))
                    .await
                    .is_err()
                {
                    break 'outer;
                }
            }

            if !executed_any {
                break 'outer;
            }
        }

        let end = if let Some(resp) = final_stream_end {
            ChatStreamEvent::StreamEnd { response: resp }
        } else {
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: None,
                    content: MessageContent::Text(String::new()),
                    usage: None,
                    finish_reason: Some(FinishReason::Unknown),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            }
        };

        let _ = sender.send(Ok(end)).await;
    });

    let stream: ChatStream = std::pin::Pin::from(Box::new(MpscStream(rx))
        as Box<dyn futures::Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync>);
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use futures::stream;
    use serde_json::json;
    use siumai::prelude::unified::ChatStreamToolCall;
    use siumai::prelude::unified::ResponseMetadata;
    use std::sync::Mutex;

    struct MockResolver;

    #[async_trait]
    impl ToolResolver for MockResolver {
        async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, LlmError> {
            match name {
                "get_weather" => Ok(json!({
                    "city": arguments.get("city").and_then(|v| v.as_str()).unwrap_or("Unknown"),
                    "temperature_c": 26
                })),
                _ => Err(LlmError::InternalError(format!("unknown tool: {name}"))),
            }
        }
    }

    #[derive(Default)]
    struct MockModel {
        calls: Mutex<usize>,
        requests: Mutex<Vec<Vec<ChatMessage>>>,
    }

    #[async_trait]
    impl ChatCapability for MockModel {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(MessageContent::Text("ok".to_string())))
        }

        async fn chat_stream(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            self.requests.lock().unwrap().push(messages);
            let idx = {
                let mut g = self.calls.lock().unwrap();
                let idx = *g;
                *g += 1;
                idx
            };

            let meta = ResponseMetadata {
                id: Some(format!("resp_{idx}")),
                model: Some("mock".to_string()),
                created: None,
                provider: "mock".to_string(),
                request_id: None,
            };

            match idx {
                0 => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::ToolCallDelta {
                            id: "call_1".to_string(),
                            function_name: Some("get_weather".to_string()),
                            arguments_delta: Some("{\"city\":\"Guangzhou\"}".to_string()),
                            index: None,
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse {
                                id: Some("resp_0".to_string()),
                                model: Some("mock".to_string()),
                                content: MessageContent::MultiModal(vec![ContentPart::tool_call(
                                    "call_1",
                                    "get_weather",
                                    json!({"city":"Guangzhou"}),
                                    None,
                                )]),
                                usage: None,
                                finish_reason: Some(FinishReason::ToolCalls),
                                audio: None,
                                system_fingerprint: None,
                                service_tier: None,
                                warnings: None,
                                provider_metadata: None,
                            },
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
                _ => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::ContentDelta {
                            delta: "It's sunny.".to_string(),
                            index: None,
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse::new(MessageContent::Text(
                                "It's sunny.".to_string(),
                            )),
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_loop_inserts_tool_result_and_keeps_single_stream_end() {
        let model = Arc::new(MockModel::default());
        let model_dyn: Arc<dyn ChatCapability + Send + Sync> = model.clone();
        let resolver: Arc<dyn ToolResolver + Send + Sync> = Arc::new(MockResolver);

        let tools = vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )];

        let mut stream = tool_loop_chat_stream(
            model_dyn,
            vec![ChatMessage::user("weather?").build()],
            tools,
            resolver,
            ToolLoopGatewayOptions { max_steps: 4 },
        )
        .await
        .expect("create tool-loop stream");

        let mut seen_start = 0;
        let mut seen_end = 0;
        let mut seen_tool_call_delta = false;
        let mut seen_gateway_tool_result = false;
        let mut seen_final_text = false;

        while let Some(item) = stream.next().await {
            let ev = item.expect("event");
            match ev {
                ChatStreamEvent::StreamStart { .. } => seen_start += 1,
                ChatStreamEvent::StreamEnd { .. } => seen_end += 1,
                ChatStreamEvent::ToolCallDelta { .. } => seen_tool_call_delta = true,
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    if delta.contains("sunny") {
                        seen_final_text = true;
                    }
                }
                ChatStreamEvent::Custom { event_type, data } => {
                    if event_type == "gateway:tool-result"
                        && data.get("type").and_then(|v| v.as_str()) == Some("tool-result")
                    {
                        seen_gateway_tool_result = true;
                    }
                }
                _ => {}
            }
        }

        assert_eq!(seen_start, 1, "should emit StreamStart only once");
        assert_eq!(seen_end, 1, "should emit StreamEnd only once");
        assert!(seen_tool_call_delta, "should forward tool call deltas");
        assert!(
            seen_gateway_tool_result,
            "should insert gateway tool-result v3 part"
        );
        assert!(seen_final_text, "should forward final answer content");

        let recorded = model.requests.lock().unwrap();
        assert_eq!(
            recorded.len(),
            2,
            "should call upstream twice (tool + final answer)"
        );
        assert!(
            recorded[1].iter().any(|m| m.role == MessageRole::Tool),
            "second upstream request should include tool result messages"
        );
    }

    #[derive(Default)]
    struct StablePartToolModel {
        calls: Mutex<usize>,
    }

    #[async_trait]
    impl ChatCapability for StablePartToolModel {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(MessageContent::Text("ok".to_string())))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            let idx = {
                let mut g = self.calls.lock().unwrap();
                let idx = *g;
                *g += 1;
                idx
            };

            let meta = ResponseMetadata {
                id: Some(format!("stable_resp_{idx}")),
                model: Some("mock".to_string()),
                created: None,
                provider: "mock".to_string(),
                request_id: None,
            };

            match idx {
                0 => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputStart {
                                id: "call_1".to_string(),
                                tool_name: "get_weather".to_string(),
                                provider_metadata: None,
                                provider_executed: None,
                                dynamic: None,
                                title: None,
                            },
                        }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputDelta {
                                id: "call_1".to_string(),
                                delta: "{\"city\":\"Guangzhou\"}".to_string(),
                                provider_metadata: None,
                            },
                        }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputEnd {
                                id: "call_1".to_string(),
                                provider_metadata: None,
                            },
                        }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                tool_call_id: "call_1".to_string(),
                                tool_name: "get_weather".to_string(),
                                input: "{\"city\":\"Guangzhou\"}".to_string(),
                                provider_executed: None,
                                dynamic: None,
                                provider_metadata: None,
                            }),
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse {
                                id: Some("stable_resp_0".to_string()),
                                model: Some("mock".to_string()),
                                content: MessageContent::MultiModal(vec![ContentPart::tool_call(
                                    "call_1",
                                    "get_weather",
                                    json!({"city":"Guangzhou"}),
                                    None,
                                )]),
                                usage: None,
                                finish_reason: Some(FinishReason::ToolCalls),
                                audio: None,
                                system_fingerprint: None,
                                service_tier: None,
                                warnings: None,
                                provider_metadata: None,
                            },
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
                _ => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::ContentDelta {
                            delta: "Stable parts worked.".to_string(),
                            index: None,
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse::new(MessageContent::Text(
                                "Stable parts worked.".to_string(),
                            )),
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_loop_accepts_stable_tool_parts() {
        let model: Arc<dyn ChatCapability + Send + Sync> = Arc::new(StablePartToolModel::default());
        let resolver: Arc<dyn ToolResolver + Send + Sync> = Arc::new(MockResolver);

        let tools = vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )];

        let mut stream = tool_loop_chat_stream(
            model,
            vec![ChatMessage::user("weather?").build()],
            tools,
            resolver,
            ToolLoopGatewayOptions { max_steps: 4 },
        )
        .await
        .expect("create stable-part tool-loop stream");

        let mut saw_tool_call_part = false;
        let mut saw_final_text = false;

        while let Some(item) = stream.next().await {
            match item.expect("event") {
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(call),
                } if call.tool_call_id == "call_1" && call.tool_name == "get_weather" => {
                    saw_tool_call_part = true;
                }
                ChatStreamEvent::ContentDelta { delta, .. } if delta.contains("Stable parts") => {
                    saw_final_text = true;
                }
                _ => {}
            }
        }

        assert!(saw_tool_call_part, "should forward stable tool-call part");
        assert!(
            saw_final_text,
            "should continue to the post-tool answer step"
        );
    }
}
