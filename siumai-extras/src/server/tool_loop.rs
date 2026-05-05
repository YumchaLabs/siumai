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

use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;
use siumai::prelude::unified::*;
use tokio::sync::mpsc;

use crate::orchestrator::types::ToolResolver;
use crate::tool_runtime::update_pending_deferred_tool_calls;

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
) -> &'a mut ToolCallAcc {
    match acc.entry(id.to_string()) {
        Entry::Vacant(entry) => {
            order.push(id.to_string());
            entry.insert(ToolCallAcc::default())
        }
        Entry::Occupied(entry) => entry.into_mut(),
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
            let entry = ensure_tool_call_acc(acc, order, id);
            entry.tool_name = Some(tool_name.clone());
            entry.provider_executed = *provider_executed;
            true
        }
        ChatStreamPart::ToolInputDelta { id, delta, .. } => {
            let entry = ensure_tool_call_acc(acc, order, id);
            entry.args_json.push_str(delta);
            true
        }
        ChatStreamPart::ToolCall(call) => {
            let entry = ensure_tool_call_acc(acc, order, &call.tool_call_id);
            entry.tool_name = Some(call.tool_name.clone());
            entry.args_json = call.input.clone();
            entry.provider_executed = call.provider_executed;
            true
        }
        _ => false,
    }
}

fn accumulate_runtime_text_part(part: &ChatStreamPart, acc_text: &mut String) -> bool {
    match part {
        ChatStreamPart::TextDelta { delta, .. } => {
            acc_text.push_str(delta);
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
            let entry = ensure_tool_call_acc(acc, order, &id);
            entry.tool_name = Some(tool_name);
            entry.provider_executed = provider_executed;
            true
        }
        siumai::experimental::streaming::LanguageModelV3StreamPart::ToolInputDelta {
            id,
            delta,
            ..
        } => {
            let entry = ensure_tool_call_acc(acc, order, &id);
            entry.args_json.push_str(&delta);
            true
        }
        siumai::experimental::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            let entry = ensure_tool_call_acc(acc, order, &call.tool_call_id);
            entry.tool_name = Some(call.tool_name);
            entry.args_json = call.input;
            entry.provider_executed = call.provider_executed;
            true
        }
        _ => false,
    }
}

fn accumulate_loose_text_part(data: &Value, acc_text: &mut String) -> bool {
    let Some(part) =
        siumai::experimental::streaming::LanguageModelV3StreamPart::parse_loose_json(data)
    else {
        return false;
    };

    match part {
        siumai::experimental::streaming::LanguageModelV3StreamPart::TextDelta { delta, .. } => {
            acc_text.push_str(&delta);
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
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
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
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        });
    }

    out
}

fn gateway_tool_result_events(
    tool_call_id: String,
    tool_name: String,
    result: Value,
    is_error: bool,
) -> [ChatStreamEvent; 2] {
    let stable = ChatStreamEvent::Part {
        part: ChatStreamPart::ToolResult(ChatStreamToolResult {
            tool_call_id: tool_call_id.clone(),
            tool_name: tool_name.clone(),
            result: result.clone(),
            is_error: Some(is_error),
            preliminary: None,
            dynamic: None,
            provider_metadata: None,
        }),
    };

    let legacy = ChatStreamEvent::Custom {
        event_type: "gateway:tool-result".to_string(),
        data: serde_json::json!({
            "type": "tool-result",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "result": result,
            "isError": is_error,
        }),
    };

    [stable, legacy]
}

/// Create a single `ChatStream` that keeps the downstream stream open across tool-loop steps.
///
/// This function:
/// - emits the upstream `StreamStart` only once (from the first step)
/// - suppresses intermediate `StreamEnd` events until the final step completes
/// - emits stable `tool-result` parts plus legacy v3 `tool-result` custom events between steps so
///   downstream protocols and older consumers can both surface results
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
        let mut pending_deferred_tool_calls: HashSet<String> = HashSet::new();

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
            let mut step_response: Option<ChatResponse> = None;

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
                    ChatStreamEvent::Part { part }
                    | ChatStreamEvent::PartWithReplay { part, .. } => {
                        let _ = accumulate_runtime_text_part(part, &mut acc_text);
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
                        let _ = accumulate_loose_text_part(data, &mut acc_text);
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
                        step_response = Some(response.clone());
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
            let response_tool_results = step_response
                .as_ref()
                .map(|response| {
                    response
                        .tool_results()
                        .into_iter()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            update_pending_deferred_tool_calls(
                &mut pending_deferred_tool_calls,
                Some(tools.as_slice()),
                &tool_calls,
                &response_tool_results,
            );

            let mut assistant_parts: Vec<ContentPart> = Vec::new();
            if !acc_text.trim().is_empty() {
                assistant_parts.push(ContentPart::Text {
                    text: acc_text.clone(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                });
            }
            assistant_parts.extend(tool_calls.clone());
            assistant_parts.extend(response_tool_results.clone());

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
                        for event in gateway_tool_result_events(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            err.clone(),
                            true,
                        ) {
                            if sender.send(Ok(event)).await.is_err() {
                                break 'outer;
                            }
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

                for event in gateway_tool_result_events(tool_call_id, tool_name, value, is_error) {
                    if sender.send(Ok(event)).await.is_err() {
                        break 'outer;
                    }
                }
            }

            if !executed_any && pending_deferred_tool_calls.is_empty() {
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
                    raw_finish_reason: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    request: None,
                    response: None,
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
    use siumai::prelude::unified::{
        ChatStreamToolCall, ChatStreamToolResult, ProviderOptionsMap, ResponseMetadata,
    };
    use siumai::types::ToolResultOutput;
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
                headers: None,
                body: None,
            };

            match idx {
                0 => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::tool_input_start_part(
                            "call_1",
                            "get_weather",
                        )),
                        Ok(ChatStreamEvent::tool_input_delta_part(
                            "call_1",
                            "{\"city\":\"Guangzhou\"}",
                        )),
                        Ok(ChatStreamEvent::tool_input_end_part("call_1")),
                        Ok(ChatStreamEvent::tool_call_part(
                            "call_1",
                            "get_weather",
                            "{\"city\":\"Guangzhou\"}",
                        )),
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
                                raw_finish_reason: None,
                                audio: None,
                                system_fingerprint: None,
                                service_tier: None,
                                warnings: None,
                                request: None,
                                response: None,
                                provider_metadata: None,
                            },
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
                _ => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::text_delta_part("0", "It's sunny.")),
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
        let mut seen_tool_call_part = false;
        let mut seen_gateway_tool_result = false;
        let mut seen_tool_result_part = false;
        let mut seen_final_text = false;

        while let Some(item) = stream.next().await {
            let ev = item.expect("event");
            match ev {
                ChatStreamEvent::StreamStart { .. } => seen_start += 1,
                ChatStreamEvent::StreamEnd { .. } => seen_end += 1,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(_),
                }
                | ChatStreamEvent::PartWithReplay {
                    part: ChatStreamPart::ToolCall(_),
                    ..
                } => seen_tool_call_part = true,
                event
                    if event
                        .text_delta()
                        .is_some_and(|delta| delta.contains("sunny")) =>
                {
                    seen_final_text = true;
                }
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolResult(result),
                } if result.tool_call_id == "call_1"
                    && result.tool_name == "get_weather"
                    && result.result == json!({"city":"Guangzhou","temperature_c":26}) =>
                {
                    seen_tool_result_part = true;
                }
                ChatStreamEvent::Custom { event_type, data }
                    if event_type == "gateway:tool-result"
                        && data.get("type").and_then(|v| v.as_str()) == Some("tool-result") =>
                {
                    seen_gateway_tool_result = true;
                }
                _ => {}
            }
        }

        assert_eq!(seen_start, 1, "should emit StreamStart only once");
        assert_eq!(seen_end, 1, "should emit StreamEnd only once");
        assert!(seen_tool_call_part, "should forward typed tool call parts");
        assert!(
            seen_tool_result_part,
            "should insert stable tool-result part between steps"
        );
        assert!(
            seen_gateway_tool_result,
            "should keep gateway tool-result legacy compatibility event"
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
        requests: Mutex<Vec<Vec<ChatMessage>>>,
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
                id: Some(format!("stable_resp_{idx}")),
                model: Some("mock".to_string()),
                created: None,
                provider: "mock".to_string(),
                request_id: None,
                headers: None,
                body: None,
            };

            match idx {
                0 => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::TextDelta {
                                id: "txt_1".to_string(),
                                delta: "Checking weather. ".to_string(),
                                provider_metadata: None,
                            },
                        }),
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
                                raw_finish_reason: None,
                                audio: None,
                                system_fingerprint: None,
                                service_tier: None,
                                warnings: None,
                                request: None,
                                response: None,
                                provider_metadata: None,
                            },
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
                _ => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::TextDelta {
                                id: "txt_2".to_string(),
                                delta: "Stable parts worked.".to_string(),
                                provider_metadata: None,
                            },
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse::new(MessageContent::Text(String::new())),
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_loop_accepts_stable_tool_parts() {
        let model = Arc::new(StablePartToolModel::default());
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
                ChatStreamEvent::Part {
                    part: ChatStreamPart::TextDelta { delta, .. },
                } if delta.contains("Stable parts") => {
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

        let recorded = model.requests.lock().unwrap();
        assert_eq!(
            recorded.len(),
            2,
            "stable-part tool loop should issue a follow-up request"
        );
        let follow_up_assistant = recorded[1]
            .iter()
            .find(|message| message.role == MessageRole::Assistant)
            .expect("assistant follow-up message");
        assert_eq!(
            follow_up_assistant.content_text(),
            Some("Checking weather. ")
        );
        assert_eq!(follow_up_assistant.tool_calls().len(), 1);
    }

    #[derive(Default)]
    struct DeferredProviderToolModel {
        calls: Mutex<usize>,
        requests: Mutex<Vec<Vec<ChatMessage>>>,
    }

    #[async_trait]
    impl ChatCapability for DeferredProviderToolModel {
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
                id: Some(format!("deferred_resp_{idx}")),
                model: Some("mock".to_string()),
                created: None,
                provider: "mock".to_string(),
                request_id: None,
                headers: None,
                body: None,
            };

            match idx {
                0 => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputStart {
                                id: "call_code_execution".to_string(),
                                tool_name: "code_execution".to_string(),
                                provider_metadata: None,
                                provider_executed: Some(true),
                                dynamic: None,
                                title: None,
                            },
                        }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolInputDelta {
                                id: "call_code_execution".to_string(),
                                delta: "{\"code\":\"print('hi')\"}".to_string(),
                                provider_metadata: None,
                            },
                        }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                                tool_call_id: "call_code_execution".to_string(),
                                tool_name: "code_execution".to_string(),
                                input: "{\"code\":\"print('hi')\"}".to_string(),
                                provider_executed: Some(true),
                                dynamic: None,
                                provider_metadata: None,
                            }),
                        }),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse {
                                id: Some("deferred_resp_0".to_string()),
                                model: Some("mock".to_string()),
                                content: MessageContent::MultiModal(vec![ContentPart::tool_call(
                                    "call_code_execution",
                                    "code_execution",
                                    json!({"code":"print('hi')"}),
                                    Some(true),
                                )]),
                                usage: None,
                                finish_reason: Some(FinishReason::ToolCalls),
                                raw_finish_reason: None,
                                audio: None,
                                system_fingerprint: None,
                                service_tier: None,
                                warnings: None,
                                request: None,
                                response: None,
                                provider_metadata: None,
                            },
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
                _ => {
                    let events = vec![
                        Ok(ChatStreamEvent::StreamStart { metadata: meta }),
                        Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::ToolResult(ChatStreamToolResult {
                                tool_call_id: "call_code_execution".to_string(),
                                tool_name: "code_execution".to_string(),
                                result: json!({"stdout":"hi"}),
                                is_error: Some(false),
                                preliminary: None,
                                dynamic: None,
                                provider_metadata: None,
                            }),
                        }),
                        Ok(ChatStreamEvent::text_delta_part(
                            "0",
                            "Deferred provider result.",
                        )),
                        Ok(ChatStreamEvent::StreamEnd {
                            response: ChatResponse::new(MessageContent::MultiModal(vec![
                                ContentPart::ToolResult {
                                    tool_call_id: "call_code_execution".to_string(),
                                    tool_name: "code_execution".to_string(),
                                    output: ToolResultOutput::json(json!({"stdout":"hi"})),
                                    input: None,
                                    provider_executed: Some(true),
                                    dynamic: None,
                                    preliminary: None,
                                    title: None,
                                    provider_options: ProviderOptionsMap::default(),
                                    provider_metadata: None,
                                },
                                ContentPart::text("Deferred provider result."),
                            ])),
                        }),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        }
    }

    #[tokio::test]
    async fn tool_loop_continues_for_deferred_provider_results() {
        let model = Arc::new(DeferredProviderToolModel::default());
        let model_dyn: Arc<dyn ChatCapability + Send + Sync> = model.clone();
        let resolver: Arc<dyn ToolResolver + Send + Sync> = Arc::new(MockResolver);

        let tools = vec![
            Tool::provider_defined("mock.code_execution", "code_execution")
                .with_supports_deferred_results(true),
        ];

        let mut stream = tool_loop_chat_stream(
            model_dyn,
            vec![ChatMessage::user("run code").build()],
            tools,
            resolver,
            ToolLoopGatewayOptions { max_steps: 4 },
        )
        .await
        .expect("create deferred provider tool-loop stream");

        let mut saw_final_text = false;
        while let Some(item) = stream.next().await {
            match item.expect("event") {
                event
                    if event
                        .text_delta()
                        .is_some_and(|delta| delta.contains("Deferred provider result")) =>
                {
                    saw_final_text = true;
                }
                _ => {}
            }
        }

        assert!(
            saw_final_text,
            "should continue after deferred provider tool call"
        );
        assert_eq!(
            model.requests.lock().unwrap().len(),
            2,
            "should re-enter the upstream model after a deferred provider tool call"
        );
    }
}
