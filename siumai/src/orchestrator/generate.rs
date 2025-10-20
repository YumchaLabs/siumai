//! Non-streaming orchestrator implementation.

use serde_json::Value;

use super::prepare_step::{PrepareStepContext, filter_active_tools};
use super::stop_condition::StopCondition;
use super::types::{OrchestratorOptions, StepResult, ToolApproval, ToolResolver};
use crate::error::LlmError;
use crate::telemetry::{
    TelemetryConfig,
    events::{OrchestratorEvent, OrchestratorStepType, SpanEvent, TelemetryEvent},
};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, MessageContent, Tool};

fn validate_args_with_schema(_schema: &Value, _instance: &Value) -> Result<(), String> {
    // Schema validation has been moved to siumai-extras
    // If you need schema validation, use siumai-extras::schema::validate_json
    // For now, we skip validation
    tracing::debug!(
        "Schema validation is no longer built-in. Use siumai-extras::schema::validate_json for validation."
    );
    Ok(())
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

            crate::telemetry::emit(TelemetryEvent::SpanStart(span)).await;
        }
    }

    for step_idx in 0..max_steps {
        // Call prepare_step callback if provided
        let mut current_tools = tools.clone();
        let mut current_messages = history.clone();

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

            // Note: tool_choice and system overrides would require changes to ChatCapability trait
            // For now, we only support active_tools and messages overrides
        }

        let resp = model
            .chat_with_tools(current_messages.clone(), current_tools.clone())
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
                        if let Some(ts) = current_tools
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

            crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

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
            crate::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
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

            crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

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
            crate::telemetry::emit(TelemetryEvent::Orchestrator(orch_event)).await;
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

            crate::telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
        }
    }
}
