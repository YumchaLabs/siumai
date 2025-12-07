#![allow(clippy::collapsible_if)]
//! Non-streaming orchestrator implementation.

use futures::StreamExt;
use serde_json::Value;

use super::prepare_step::{PrepareStepContext, filter_active_tools};
use super::stop_condition::StopCondition;
use super::types::{OrchestratorOptions, StepResult, ToolApproval, ToolResolver};
use super::validation::validate_args_with_schema;
use siumai::error::LlmError;
use siumai::observability::telemetry::{
    TelemetryConfig,
    events::{OrchestratorEvent, OrchestratorStepType, SpanEvent, TelemetryEvent},
};
use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, ChatResponse, ContentPart, MessageContent, Tool};

/// Convert orchestrator ToolChoice to types::ToolChoice
fn convert_tool_choice(choice: super::prepare_step::ToolChoice) -> siumai::types::ToolChoice {
    match choice {
        super::prepare_step::ToolChoice::Auto => siumai::types::ToolChoice::Auto,
        super::prepare_step::ToolChoice::Required => siumai::types::ToolChoice::Required,
        super::prepare_step::ToolChoice::None => siumai::types::ToolChoice::None,
        super::prepare_step::ToolChoice::Specific { tool_name } => {
            siumai::types::ToolChoice::Tool { name: tool_name }
        }
    }
}

/// Orchestrate multi-step generation with optional tool execution.
///
/// This function implements a loop: ask → tool-calls → tool exec → re-ask.
/// It continues until a stop condition is met or the model generates a response without tool calls.
///
/// # Arguments
///
/// * `model` - The chat model to use
/// * `messages` - Initial message history
/// * `tools` - Available tools for the model to call
/// * `resolver` - Tool resolver for executing tool calls
/// * `stop_conditions` - Conditions that determine when to stop the loop
/// * `opts` - Orchestrator options including callbacks and telemetry
///
/// # Returns
///
/// Returns a tuple of (final ChatResponse, all StepResults)
pub async fn generate(
    model: &impl ChatCapability,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    resolver: Option<&dyn ToolResolver>,
    stop_conditions: &[&dyn StopCondition],
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

            siumai::observability::telemetry::emit(TelemetryEvent::SpanStart(span)).await;
        }
    }

    for step_idx in 0..max_steps {
        // Call prepare_step callback if provided
        let mut current_tools = tools.clone();
        let mut current_messages = history.clone();
        let mut current_tool_choice: Option<siumai::types::ToolChoice> = None;
        let mut current_system: Option<String> = None;

        if let Some(ref prepare_fn) = opts.prepare_step {
            let ctx = PrepareStepContext {
                step_number: step_idx,
                steps: &steps,
                messages: &history,
            };
            let prepare_result = prepare_fn(ctx);

            // Apply prepare_step overrides
            if let Some(active_tools) = prepare_result.active_tools {
                if let Some(ref tools) = current_tools {
                    current_tools = Some(filter_active_tools(tools, &Some(active_tools)));
                }
            }

            if let Some(messages) = prepare_result.messages {
                current_messages = messages;
            }

            // Apply tool_choice override
            if let Some(tool_choice) = prepare_result.tool_choice {
                current_tool_choice = Some(convert_tool_choice(tool_choice));
            }

            // Apply system override
            if let Some(system) = prepare_result.system {
                current_system = Some(system);
            }
        }

        // Apply system message override if provided
        if let Some(system) = current_system {
            // Prepend system message to the beginning
            current_messages.insert(0, ChatMessage::system(system).build());
        }

        // Use chat_request if we have tool_choice override or common_params
        let resp = if current_tool_choice.is_some() || opts.common_params.is_some() {
            let mut request = siumai::types::ChatRequest::new(current_messages.clone());

            if let Some(tools) = current_tools.clone() {
                request = request.with_tools(tools);
            }

            if let Some(tool_choice) = current_tool_choice {
                request = request.with_tool_choice(tool_choice);
            }

            // Apply agent-level common_params
            if let Some(ref common_params) = opts.common_params {
                request = request.with_common_params(common_params.clone());
            }

            model.chat_request(request).await?
        } else {
            model
                .chat_with_tools(current_messages.clone(), current_tools.clone())
                .await?
        };

        let mut step_msgs: Vec<ChatMessage> = Vec::new();
        // Build assistant message; include tool_calls if present
        let _assistant_text = resp
            .content_text()
            .map(|s| s.to_string())
            .unwrap_or_default();
        // The response content already contains tool calls, so we just use it directly
        let assistant_built = ChatMessage {
            role: siumai::types::MessageRole::Assistant,
            content: resp.content.clone(),
            metadata: siumai::types::MessageMetadata::default(),
        };
        history.push(assistant_built.clone());
        step_msgs.push(assistant_built);

        // Execute tools if requested
        let tool_calls: Vec<_> = resp.tool_calls().into_iter().cloned().collect();
        if !tool_calls.is_empty() {
            if let Some(resolver) = resolver {
                for call in tool_calls.iter() {
                    if let siumai::types::ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } = call
                    {
                        if let Some(ts) = current_tools.as_ref().and_then(|ts| {
                            ts.iter().find(|t| match t {
                                Tool::Function { function } => &function.name == tool_name,
                                Tool::ProviderDefined(provider_tool) => {
                                    &provider_tool.name == tool_name
                                }
                            })
                        }) {
                            let parameters = match ts {
                                Tool::Function { function } => &function.parameters,
                                Tool::ProviderDefined(_) => {
                                    // Provider-defined tools don't have parameters schema
                                    // Skip validation for them
                                    continue;
                                }
                            };
                            if let Err(reason) = validate_args_with_schema(parameters, arguments) {
                                let out_val = Value::Object({
                                    let mut m = serde_json::Map::new();
                                    m.insert("error".into(), Value::String("invalid_args".into()));
                                    m.insert("reason".into(), Value::String(reason));
                                    m
                                });
                                let tool_msg = ChatMessage::tool_error(
                                    tool_call_id.clone(),
                                    tool_name.clone(),
                                    serde_json::to_string(&out_val).unwrap_or_default(),
                                )
                                .build();
                                history.push(tool_msg.clone());
                                step_msgs.push(tool_msg);
                                continue;
                            }
                        }
                        let decision = if let Some(cb) = &opts.on_tool_approval {
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
                                                            &opts.on_preliminary_tool_result
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
                                    Err(e) => Value::String(format!("<tool error: {}>", e)),
                                }
                            }
                            ToolApproval::Deny { reason } => Value::Object({
                                let mut m = serde_json::Map::new();
                                m.insert("error".into(), Value::String("denied".into()));
                                m.insert("reason".into(), Value::String(reason));
                                m
                            }),
                        };
                        // Tool message must carry tool_call_id
                        let tool_msg = ChatMessage::tool_result_json(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            out_val,
                        )
                        .build();
                        history.push(tool_msg.clone());
                        step_msgs.push(tool_msg);
                    }
                }
            }
        }

        // Extract tool results from tool messages
        let tool_results: Vec<ContentPart> = step_msgs
            .iter()
            .filter(|msg| matches!(msg.role, siumai::types::MessageRole::Tool))
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
        if let Some(cb) = &opts.on_step_finish {
            cb(&step);
        }
        steps.push(step);

        // Check stop conditions
        let should_stop = stop_conditions.iter().any(|c| c.should_stop(&steps));

        if should_stop || tool_calls.is_empty() {
            if let Some(cb) = &opts.on_finish {
                cb(&steps);
            }

            emit_telemetry_success(
                &opts.telemetry,
                &span_id,
                &trace_id,
                &steps,
                &resp,
                start_time,
            )
            .await;

            return Ok((resp, steps));
        }
    }

    // Max steps reached; return the last response if available or error
    if let Some(last) = steps.last() {
        // Build a synthetic response from the last assistant message
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

        emit_telemetry_max_steps(&opts.telemetry, &span_id, &trace_id, &steps, start_time).await;

        Ok((resp, steps))
    } else {
        emit_telemetry_error(&opts.telemetry, &span_id, &trace_id).await;

        Err(LlmError::InternalError(
            "orchestrator: no steps produced".into(),
        ))
    }
}

async fn emit_telemetry_success(
    telemetry: &Option<TelemetryConfig>,
    span_id: &str,
    trace_id: &str,
    steps: &[StepResult],
    resp: &ChatResponse,
    start_time: std::time::SystemTime,
) {
    if let Some(telemetry) = telemetry {
        if telemetry.enabled {
            let total_usage = StepResult::merge_usage(steps);
            let span = SpanEvent::start(
                span_id.to_string(),
                None,
                trace_id.to_string(),
                "ai.orchestrator.generate".to_string(),
            )
            .end_ok()
            .with_attribute("total_steps", steps.len().to_string())
            .with_attribute("finish_reason", format!("{:?}", resp.finish_reason));

            siumai::observability::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

            let orch_event = OrchestratorEvent {
                id: uuid::Uuid::new_v4().to_string(),
                trace_id: trace_id.to_string(),
                timestamp: std::time::SystemTime::now(),
                total_steps: steps.len(),
                current_step: steps.len(),
                step_type: OrchestratorStepType::Completion,
                total_usage,
                total_duration: std::time::SystemTime::now().duration_since(start_time).ok(),
                metadata: std::collections::HashMap::new(),
            };
            siumai::observability::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
        }
    }
}

async fn emit_telemetry_max_steps(
    telemetry: &Option<TelemetryConfig>,
    span_id: &str,
    trace_id: &str,
    steps: &[StepResult],
    start_time: std::time::SystemTime,
) {
    if let Some(telemetry) = telemetry {
        if telemetry.enabled {
            let total_usage = StepResult::merge_usage(steps);
            let span = SpanEvent::start(
                span_id.to_string(),
                None,
                trace_id.to_string(),
                "ai.orchestrator.generate".to_string(),
            )
            .end_ok()
            .with_attribute("total_steps", steps.len().to_string())
            .with_attribute("max_steps_reached", "true");

            siumai::observability::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

            let orch_event = OrchestratorEvent {
                id: uuid::Uuid::new_v4().to_string(),
                trace_id: trace_id.to_string(),
                timestamp: std::time::SystemTime::now(),
                total_steps: steps.len(),
                current_step: steps.len(),
                step_type: OrchestratorStepType::Completion,
                total_usage,
                total_duration: std::time::SystemTime::now().duration_since(start_time).ok(),
                metadata: std::collections::HashMap::new(),
            };
            siumai::observability::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
        }
    }
}

async fn emit_telemetry_error(telemetry: &Option<TelemetryConfig>, span_id: &str, trace_id: &str) {
    if let Some(telemetry) = telemetry {
        if telemetry.enabled {
            let span = SpanEvent::start(
                span_id.to_string(),
                None,
                trace_id.to_string(),
                "ai.orchestrator.generate".to_string(),
            )
            .end_error("orchestrator: no steps produced".to_string());

            siumai::observability::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
        }
    }
}
