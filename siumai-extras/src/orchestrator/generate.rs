#![allow(clippy::collapsible_if)]
//! Non-streaming orchestrator implementation.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::tool_runtime::{
    client_tool_call_count, execute_local_tool_call, execution_denied_tool_result,
    merge_step_tool_results, preprocess_tool_approval_responses, should_continue_after_tool_step,
    update_pending_deferred_tool_calls,
};

use super::prepare_step::{PrepareStepContext, filter_active_tools};
use super::stop_condition::StopCondition;
use super::types::{
    OrchestratorFinishEvent, OrchestratorOptions, StepLanguageModel, StepModelInfo, StepResult,
    ToolApproval, ToolResolver,
};
use super::validation::validate_args_with_schema;
use siumai::experimental::observability::telemetry::{
    TelemetryConfig,
    events::{OrchestratorEvent, OrchestratorStepType, SpanEvent, TelemetryEvent},
};
use siumai::prelude::unified::*;
use siumai::tooling::{ToolInputAvailableContext, ToolRuntimeMetadata};

/// Convert orchestrator ToolChoice to types::ToolChoice
fn convert_tool_choice(choice: super::prepare_step::ToolChoice) -> ToolChoice {
    match choice {
        super::prepare_step::ToolChoice::Auto => ToolChoice::Auto,
        super::prepare_step::ToolChoice::Required => ToolChoice::Required,
        super::prepare_step::ToolChoice::None => ToolChoice::None,
        super::prepare_step::ToolChoice::Specific { tool_name } => {
            ToolChoice::Tool { name: tool_name }
        }
    }
}

fn telemetry_metadata_values(
    telemetry: Option<&TelemetryConfig>,
) -> Option<HashMap<String, Value>> {
    let telemetry = telemetry?;
    (!telemetry.metadata.is_empty()).then(|| {
        telemetry
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect()
    })
}

fn find_declared_tool<'a>(tools: Option<&'a [Tool]>, tool_name: &str) -> Option<&'a Tool> {
    tools.and_then(|tools| {
        tools.iter().find(|tool| match tool {
            Tool::Function { function } => function.name == tool_name,
            Tool::ProviderDefined(provider_tool) => provider_tool.name == tool_name,
        })
    })
}

fn annotate_response_tool_calls(response: &mut ChatResponse, resolver: Option<&dyn ToolResolver>) {
    let Some(resolver) = resolver else {
        return;
    };

    let MessageContent::MultiModal(parts) = &mut response.content else {
        return;
    };

    for part in parts.iter_mut() {
        let ContentPart::ToolCall {
            tool_name, dynamic, ..
        } = part
        else {
            continue;
        };

        if resolver
            .runtime_tool_metadata(tool_name)
            .is_some_and(|metadata| metadata.dynamic())
        {
            *dynamic = Some(true);
        }
    }
}

fn append_response_parts(response: &mut ChatResponse, extra_parts: Vec<ContentPart>) {
    if extra_parts.is_empty() {
        return;
    }

    #[allow(unreachable_patterns)]
    match &mut response.content {
        MessageContent::Text(text) => {
            let mut parts = (!text.is_empty())
                .then(|| ContentPart::text(text.clone()))
                .into_iter()
                .collect::<Vec<_>>();
            parts.extend(extra_parts);
            response.content = MessageContent::MultiModal(parts);
        }
        MessageContent::MultiModal(parts) => parts.extend(extra_parts),
        _ => {}
    }
}

fn tool_message_from_part(part: ContentPart) -> ChatMessage {
    ChatMessage {
        role: MessageRole::Tool,
        content: MessageContent::MultiModal(vec![part]),
        provider_options: ProviderOptionsMap::default(),
        metadata: MessageMetadata::default(),
    }
}

fn input_available_context(
    tool_call_id: &str,
    input: &Value,
    step_input_messages: &[ChatMessage],
    context: &super::types::OrchestratorContext,
) -> ToolInputAvailableContext {
    ToolInputAvailableContext {
        tool_call_id: tool_call_id.to_string(),
        input: input.clone(),
        messages: step_input_messages.to_vec(),
        context: context.as_map().clone(),
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
    model: &impl LanguageModel,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    resolver: Option<&dyn ToolResolver>,
    stop_conditions: &[&dyn StopCondition],
    opts: OrchestratorOptions,
) -> Result<(ChatResponse, Vec<StepResult>), LlmError> {
    // Initialize telemetry if enabled
    let call_id = uuid::Uuid::new_v4().to_string();
    let trace_id = uuid::Uuid::new_v4().to_string();
    let span_id = uuid::Uuid::new_v4().to_string();
    let start_time = std::time::SystemTime::now();

    let mut history = messages;
    let mut steps: Vec<StepResult> = Vec::new();
    let mut current_context = opts.context.clone();
    let mut pending_deferred_tool_calls: HashSet<String> = HashSet::new();
    let max_steps = if opts.max_steps == 0 {
        1
    } else {
        opts.max_steps
    };

    let preprocessed_approvals = preprocess_tool_approval_responses(
        &history,
        resolver,
        &current_context,
        opts.on_preliminary_tool_result.as_deref(),
    )
    .await?;
    if let Some(message) = preprocessed_approvals.local_tool_message {
        history.push(message);
    }
    if let Some(message) = preprocessed_approvals.provider_forward_message {
        history.push(message);
    }

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

            siumai::experimental::observability::telemetry::emit(TelemetryEvent::SpanStart(span))
                .await;
        }
    }

    for step_idx in 0..max_steps {
        // Call prepare_step callback if provided
        let mut current_tools = tools.clone();
        let mut current_messages = history.clone();
        let mut current_tool_choice: Option<ToolChoice> = None;
        let mut current_system: Option<String> = None;
        let mut current_provider_options_map: Option<ProviderOptionsMap> = None;
        let mut step_model_override: Option<StepLanguageModel> = None;

        if let Some(ref prepare_fn) = opts.prepare_step {
            let ctx = PrepareStepContext {
                step_number: step_idx,
                steps: &steps,
                model,
                messages: &history,
                context: &current_context,
            };
            let prepare_result = prepare_fn(ctx);
            if let Some(model_override) = prepare_result.model {
                step_model_override = Some(model_override);
            }

            if let Some(context) = prepare_result.context {
                current_context = context;
            }

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

            // Apply provider options override
            if let Some(map) = prepare_result.provider_options_map {
                current_provider_options_map = Some(map);
            }
        }

        let step_model = step_model_override
            .as_deref()
            .map(|model| model as &dyn LanguageModel)
            .unwrap_or(model);
        let step_input_messages = current_messages.clone();

        // Apply system message override if provided
        if let Some(system) = current_system {
            // Prepend system message to the beginning
            current_messages.insert(0, ChatMessage::system(system).build());
        }

        let mut request = ChatRequest::new(current_messages.clone());

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

        // Apply provider options override
        if let Some(map) = current_provider_options_map {
            request.provider_options_map.merge_overrides(map);
        }

        request.telemetry = opts.telemetry.clone();
        let step_request = request.clone();

        let mut resp = siumai::text::generate(
            step_model,
            request,
            siumai::text::GenerateOptions::default(),
        )
        .await?;
        annotate_response_tool_calls(&mut resp, resolver);

        let mut step_msgs: Vec<ChatMessage> = Vec::new();
        let mut pending_tool_messages: Vec<ChatMessage> = Vec::new();
        let mut assistant_extra_parts: Vec<ContentPart> = Vec::new();

        // Execute tools if requested
        let tool_calls: Vec<_> = resp.tool_calls().into_iter().cloned().collect();
        let client_tool_calls = client_tool_call_count(&tool_calls);
        if !tool_calls.is_empty() {
            if let Some(resolver) = resolver {
                for call in tool_calls.iter() {
                    if let ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        provider_executed,
                        dynamic,
                        ..
                    } = call
                    {
                        if provider_executed == &Some(true) {
                            continue;
                        }

                        let declared_tool =
                            find_declared_tool(current_tools.as_deref(), tool_name.as_str());
                        let runtime_metadata = resolver.runtime_tool_metadata(tool_name.as_str());

                        if declared_tool.is_none() && runtime_metadata.is_none() {
                            continue;
                        }

                        if let Some(Tool::Function { function }) = declared_tool {
                            if let Err(reason) =
                                validate_args_with_schema(&function.parameters, arguments)
                            {
                                let out_val = Value::Object({
                                    let mut m = serde_json::Map::new();
                                    m.insert("error".into(), Value::String("invalid_args".into()));
                                    m.insert("reason".into(), Value::String(reason));
                                    m
                                });
                                let tool_msg = ChatMessage::tool_error_json(
                                    tool_call_id.clone(),
                                    tool_name.clone(),
                                    out_val,
                                )
                                .build();
                                pending_tool_messages.push(tool_msg);
                                continue;
                            }
                        }

                        if let Some(metadata) = runtime_metadata.as_ref() {
                            metadata
                                .invoke_on_input_available(input_available_context(
                                    tool_call_id,
                                    arguments,
                                    &step_input_messages,
                                    &current_context,
                                ))
                                .await?;
                        }

                        let approval_required = if let Some(metadata) = runtime_metadata.as_ref() {
                            metadata
                                .needs_approval(input_available_context(
                                    tool_call_id,
                                    arguments,
                                    &step_input_messages,
                                    &current_context,
                                ))
                                .await?
                        } else {
                            false
                        };

                        if approval_required && opts.on_tool_approval.is_none() {
                            assistant_extra_parts.push(ContentPart::tool_approval_request(
                                format!("approval_{}", uuid::Uuid::new_v4()),
                                tool_call_id.clone(),
                            ));
                            continue;
                        }

                        let decision = if let Some(cb) = &opts.on_tool_approval {
                            cb(tool_name, arguments)
                        } else {
                            ToolApproval::Approve(arguments.clone())
                        };
                        let tool_dynamic = (*dynamic)
                            .or_else(|| runtime_metadata.as_ref().map(ToolRuntimeMetadata::dynamic))
                            .unwrap_or(false);
                        let out_part = match decision {
                            ToolApproval::Approve(args) | ToolApproval::Modify(args) => {
                                execute_local_tool_call(
                                    resolver,
                                    tool_name,
                                    tool_call_id,
                                    args,
                                    tool_dynamic,
                                    Some(&step_input_messages),
                                    &current_context,
                                    opts.on_preliminary_tool_result.as_deref(),
                                )
                                .await
                            }
                            ToolApproval::Deny { reason } => execution_denied_tool_result(
                                tool_call_id,
                                tool_name,
                                arguments.clone(),
                                Some(reason),
                                tool_dynamic,
                                None,
                            ),
                        };
                        pending_tool_messages.push(tool_message_from_part(out_part));
                    }
                }
            }
        }

        append_response_parts(&mut resp, assistant_extra_parts);

        let assistant_built = ChatMessage {
            role: MessageRole::Assistant,
            content: resp.content.clone(),
            provider_options: ProviderOptionsMap::default(),
            metadata: MessageMetadata::default(),
        };
        history.push(assistant_built.clone());
        step_msgs.push(assistant_built);
        for tool_msg in pending_tool_messages {
            history.push(tool_msg.clone());
            step_msgs.push(tool_msg);
        }

        // Extract locally executed tool results from tool messages.
        let local_tool_results: Vec<ContentPart> = step_msgs
            .iter()
            .filter(|msg| matches!(msg.role, MessageRole::Tool))
            .flat_map(|msg| msg.tool_results())
            .cloned()
            .collect();
        let tool_results = merge_step_tool_results(&resp, &local_tool_results);
        update_pending_deferred_tool_calls(
            &mut pending_deferred_tool_calls,
            current_tools.as_deref(),
            &tool_calls,
            &tool_results,
        );
        let step_content = StepResult::compose_content(&resp, &tool_results);

        let step = StepResult {
            call_id: call_id.clone(),
            step_number: step_idx,
            model: StepModelInfo::from_language_model(step_model),
            request: step_request,
            response: resp.clone(),
            raw_finish_reason: resp.raw_finish_reason.clone(),
            function_id: opts
                .telemetry
                .as_ref()
                .and_then(|telemetry| telemetry.function_id.clone()),
            metadata: telemetry_metadata_values(opts.telemetry.as_ref()),
            context: current_context.clone(),
            content: step_content,
            messages: step_msgs,
            finish_reason: resp.finish_reason.clone(),
            usage: resp.usage.clone(),
            tool_calls: tool_calls.clone(),
            tool_results,
            warnings: resp.warnings.clone(),
            provider_metadata: resp.provider_metadata.clone(),
        };
        if let Some(cb) = &opts.on_step_finish {
            cb(&step);
        }
        steps.push(step);

        // Check stop conditions
        let should_stop = stop_conditions.iter().any(|c| c.should_stop(&steps));

        let can_continue_tool_loop = should_continue_after_tool_step(
            resolver.is_some(),
            client_tool_calls,
            local_tool_results.len(),
            &pending_deferred_tool_calls,
        );

        if should_stop || !can_continue_tool_loop {
            if let Some(cb) = &opts.on_finish {
                if let Some(event) =
                    OrchestratorFinishEvent::from_response_and_steps(resp.clone(), steps.clone())
                {
                    cb(&event);
                }
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
            if let Some(event) =
                OrchestratorFinishEvent::from_response_and_steps(resp.clone(), steps.clone())
            {
                cb(&event);
            }
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

            siumai::experimental::observability::telemetry::emit(TelemetryEvent::SpanEnd(span))
                .await;

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
            siumai::experimental::observability::telemetry::emit(TelemetryEvent::Orchestrator(
                orch_event,
            ))
            .await;
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

            siumai::experimental::observability::telemetry::emit(TelemetryEvent::SpanEnd(span))
                .await;

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
            siumai::experimental::observability::telemetry::emit(TelemetryEvent::Orchestrator(
                orch_event,
            ))
            .await;
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

            siumai::experimental::observability::telemetry::emit(TelemetryEvent::SpanEnd(span))
                .await;
        }
    }
}
