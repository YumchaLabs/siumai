//! Streaming orchestrator implementation.

use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use serde_json::Value;
use tokio::sync::oneshot;

use crate::tool_runtime::{
    client_tool_call_count, execute_local_tool_call, execution_denied_tool_result,
    merge_step_tool_results, preprocess_tool_approval_responses, should_continue_after_tool_step,
    update_pending_deferred_tool_calls,
};

use super::prepare_step::{PrepareStepContext, filter_active_tools};
use super::types::{
    OrchestratorFinishEvent, OrchestratorStreamOptions, StepLanguageModel, StepModelInfo,
    StepResult, ToolApproval, ToolResolver,
};
use super::validation::validate_args_with_schema;
use siumai::experimental::observability::telemetry::TelemetryConfig;
use siumai::prelude::unified::*;
use siumai::tooling::{
    ToolInputAvailableContext, ToolInputDeltaContext, ToolRuntimeContext, ToolRuntimeMetadata,
};

/// Stream handle that carries the stream, step summary, and a cancel handle.
pub struct StreamOrchestration {
    /// Combined chat stream across all orchestration steps.
    pub stream: ChatStream,
    /// Receiver for the list of step results produced by the orchestrator.
    pub steps: oneshot::Receiver<Vec<StepResult>>,
    /// Receiver for the aggregated normalized usage across all recorded steps.
    pub total_usage: oneshot::Receiver<Option<Usage>>,
    /// A cancel handle to abort the orchestration.
    pub cancel: siumai::experimental::utils::cancel::CancelHandle,
}

struct DisabledToolResolver;

#[async_trait::async_trait]
impl ToolResolver for DisabledToolResolver {
    async fn call_tool(&self, name: &str, _arguments: Value) -> Result<Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "tool execution is disabled for streaming path: {}",
            name
        )))
    }
}

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

#[derive(Debug, Clone, Default)]
struct StreamToolInputState {
    tool_name: Option<String>,
    input_text: String,
    input_available_invoked: bool,
    input_start_invoked: bool,
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

fn find_declared_tool<'a>(tools: Option<&'a [Tool]>, tool_name: &str) -> Option<&'a Tool> {
    tools.and_then(|tools| {
        tools.iter().find(|tool| match tool {
            Tool::Function { function } => function.name == tool_name,
            Tool::ProviderDefined(provider_tool) => provider_tool.name == tool_name,
        })
    })
}

fn annotate_response_tool_calls(
    response: &mut ChatResponse,
    resolver: Option<&(dyn ToolResolver + Send + Sync)>,
) {
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

fn input_start_context(
    tool_call_id: &str,
    step_input_messages: &[ChatMessage],
    context: &super::types::OrchestratorContext,
) -> ToolRuntimeContext {
    ToolRuntimeContext {
        tool_call_id: tool_call_id.to_string(),
        messages: step_input_messages.to_vec(),
        context: context.as_map().clone(),
    }
}

fn input_delta_context(
    tool_call_id: &str,
    delta: &str,
    step_input_messages: &[ChatMessage],
    context: &super::types::OrchestratorContext,
) -> ToolInputDeltaContext {
    ToolInputDeltaContext {
        tool_call_id: tool_call_id.to_string(),
        input_text_delta: delta.to_string(),
        messages: step_input_messages.to_vec(),
        context: context.as_map().clone(),
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

/// Orchestrate multi-step streaming. Concatenates provider streams across steps.
///
/// When no resolver is provided, tool calls are surfaced in the step result but
/// the loop stops after that step instead of attempting tool execution.
pub async fn generate_stream<M>(
    model: &M,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    _resolver: Option<&dyn ToolResolver>,
    opts: OrchestratorStreamOptions,
) -> Result<StreamOrchestration, LlmError>
where
    M: LanguageModel + Clone + Send + Sync + 'static,
{
    generate_stream_owned(
        (*model).clone(),
        messages,
        tools,
        Option::<DisabledToolResolver>::None,
        opts,
    )
    .await
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
/// - `total_usage`: A receiver for the aggregated normalized usage
/// - `cancel`: A handle to cancel the orchestration
pub async fn generate_stream_owned<M, R>(
    model: M,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    resolver: Option<R>,
    opts: OrchestratorStreamOptions,
) -> Result<StreamOrchestration, LlmError>
where
    M: LanguageModel + Send + Sync + 'static,
    R: ToolResolver + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<ChatStreamEvent, LlmError>>(64);
    let (steps_tx, steps_rx) = oneshot::channel();
    let (total_usage_tx, total_usage_rx) = oneshot::channel();
    let mut history = messages;
    let max_steps = if opts.max_steps == 0 {
        1
    } else {
        opts.max_steps
    };
    let on_chunk = opts.on_chunk.clone();
    let on_step_finish = opts.on_step_finish.clone();
    let on_finish = opts.on_finish.clone();
    let on_tool_approval = opts.on_tool_approval.clone();
    let on_preliminary_tool_result = opts.on_preliminary_tool_result.clone();
    let on_abort = opts.on_abort.clone();
    let call_id = uuid::Uuid::new_v4().to_string();
    let orchestrator_cancel = siumai::experimental::utils::cancel::new_cancel_handle();
    let orchestrator_cancel_clone = orchestrator_cancel.clone();

    tokio::spawn(async move {
        let mut step_results: Vec<StepResult> = Vec::new();
        let mut encountered_error = false;
        let mut final_response: Option<ChatResponse> = None;
        let mut current_context = opts.context.clone();
        let resolver = resolver
            .map(|r| std::sync::Arc::new(r) as std::sync::Arc<dyn ToolResolver + Send + Sync>);
        let sender = tx;
        // Track processed tool_call IDs to avoid duplicate executions across steps.
        let mut processed_call_ids: HashSet<String> = HashSet::new();
        let mut pending_deferred_tool_calls: HashSet<String> = HashSet::new();
        let preprocessed_approvals = match preprocess_tool_approval_responses(
            &history,
            resolver
                .as_deref()
                .map(|resolver| resolver as &dyn ToolResolver),
            &current_context,
            on_preliminary_tool_result.as_deref(),
        )
        .await
        {
            Ok(result) => result,
            Err(err) => {
                let _ = sender.send(Err(err)).await;
                let _ = total_usage_tx.send(StepResult::merge_usage(&step_results));
                let _ = steps_tx.send(step_results);
                return;
            }
        };
        if let Some(message) = preprocessed_approvals.local_tool_message {
            history.push(message);
        }
        if let Some(message) = preprocessed_approvals.provider_forward_message {
            history.push(message);
        }
        processed_call_ids.extend(preprocessed_approvals.processed_tool_call_ids);
        'outer: for step_idx in 0..max_steps {
            if orchestrator_cancel_clone.is_cancelled() {
                break 'outer;
            }
            macro_rules! stream_or_break {
                ($expr:expr) => {
                    match $expr {
                        Ok(value) => value,
                        Err(err) => {
                            encountered_error = true;
                            let _ = sender.send(Err(err)).await;
                            break 'outer;
                        }
                    }
                };
            }
            let mut current_tools = tools.clone();
            let mut current_messages = history.clone();
            let mut current_tool_choice: Option<ToolChoice> = None;
            let mut current_system: Option<String> = None;
            let mut current_provider_options_map: Option<ProviderOptionsMap> = None;
            let mut step_model_override: Option<StepLanguageModel> = None;

            if let Some(ref prepare_fn) = opts.prepare_step {
                let ctx = PrepareStepContext {
                    step_number: step_idx,
                    steps: &step_results,
                    model: &model,
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

                if let Some(active_tools) = prepare_result.active_tools {
                    if let Some(ref tools) = current_tools {
                        current_tools = Some(filter_active_tools(tools, &Some(active_tools)));
                    }
                }

                if let Some(messages) = prepare_result.messages {
                    current_messages = messages;
                }

                if let Some(tool_choice) = prepare_result.tool_choice {
                    current_tool_choice = Some(convert_tool_choice(tool_choice));
                }

                if let Some(system) = prepare_result.system {
                    current_system = Some(system);
                }

                if let Some(map) = prepare_result.provider_options_map {
                    current_provider_options_map = Some(map);
                }
            }

            let step_model = step_model_override
                .as_deref()
                .map(|model| model as &dyn LanguageModel)
                .unwrap_or(&model);
            let step_input_messages = current_messages.clone();

            if let Some(system) = current_system {
                current_messages.insert(0, ChatMessage::system(system).build());
            }

            // First step streams; subsequent steps may be non-streaming providers.
            let (
                step_request,
                mut resp,
                streamed_approval_parts,
                streamed_input_available_ids,
                streamed_approval_required,
            ) = if step_idx == 0 {
                let mut request = ChatRequest::new(current_messages.clone());

                if let Some(tools) = current_tools.clone() {
                    request = request.with_tools(tools);
                }

                if let Some(tool_choice) = current_tool_choice.clone() {
                    request = request.with_tool_choice(tool_choice);
                }

                if let Some(ref common_params) = opts.common_params {
                    request = request.with_common_params(common_params.clone());
                }

                if let Some(map) = current_provider_options_map.clone() {
                    request.provider_options_map.merge_overrides(map);
                }

                request = request.with_streaming(true);
                request.telemetry = opts.telemetry.clone();
                let step_request = request.clone();

                let handle = match siumai::text::stream_with_cancel(
                    step_model,
                    request,
                    siumai::text::StreamOptions::default(),
                )
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
                let mut local_approval_parts = HashMap::<String, ContentPart>::new();
                let mut tool_input_states = HashMap::<String, StreamToolInputState>::new();
                let mut local_approval_required = HashMap::<String, bool>::new();
                while let Some(item) = s.next().await {
                    if orchestrator_cancel_clone.is_cancelled() {
                        handle.cancel.cancel();
                        break 'outer;
                    }
                    match item {
                        Ok(ev) => {
                            let mut extra_event: Option<ChatStreamEvent> = None;
                            match &ev {
                                ChatStreamEvent::ContentDelta { delta, .. } => {
                                    acc_text.push_str(delta)
                                }
                                ChatStreamEvent::ToolCallDelta {
                                    id,
                                    function_name,
                                    arguments_delta,
                                    ..
                                } => {
                                    let state = tool_input_states.entry(id.clone()).or_default();

                                    if let Some(tool_name) = function_name {
                                        state.tool_name = Some(tool_name.clone());
                                        if !state.input_start_invoked
                                            && let Some(metadata) =
                                                resolver.as_ref().and_then(|resolver| {
                                                    resolver.runtime_tool_metadata(tool_name)
                                                })
                                        {
                                            stream_or_break!(
                                                metadata
                                                    .invoke_on_input_start(input_start_context(
                                                        id,
                                                        &step_input_messages,
                                                        &current_context,
                                                    ))
                                                    .await
                                            );
                                            state.input_start_invoked = true;
                                        }
                                    }

                                    if let Some(delta) = arguments_delta {
                                        state.input_text.push_str(delta);
                                        if let Some(tool_name) = state.tool_name.as_deref()
                                            && let Some(metadata) =
                                                resolver.as_ref().and_then(|resolver| {
                                                    resolver.runtime_tool_metadata(tool_name)
                                                })
                                        {
                                            stream_or_break!(
                                                metadata
                                                    .invoke_on_input_delta(input_delta_context(
                                                        id,
                                                        delta,
                                                        &step_input_messages,
                                                        &current_context,
                                                    ))
                                                    .await
                                            );
                                        }
                                    }
                                }
                                ChatStreamEvent::Part { part }
                                | ChatStreamEvent::PartWithReplay { part, .. } => match part {
                                    ChatStreamPart::TextDelta { .. } => {
                                        let _ = accumulate_runtime_text_part(part, &mut acc_text);
                                    }
                                    ChatStreamPart::ToolInputStart { id, tool_name, .. } => {
                                        let state =
                                            tool_input_states.entry(id.clone()).or_default();
                                        state.tool_name = Some(tool_name.clone());
                                        if !state.input_start_invoked
                                            && let Some(metadata) =
                                                resolver.as_ref().and_then(|resolver| {
                                                    resolver.runtime_tool_metadata(tool_name)
                                                })
                                        {
                                            stream_or_break!(
                                                metadata
                                                    .invoke_on_input_start(input_start_context(
                                                        id,
                                                        &step_input_messages,
                                                        &current_context,
                                                    ))
                                                    .await
                                            );
                                            state.input_start_invoked = true;
                                        }
                                    }
                                    ChatStreamPart::ToolInputDelta { id, delta, .. } => {
                                        let state =
                                            tool_input_states.entry(id.clone()).or_default();
                                        state.input_text.push_str(delta);
                                        if let Some(tool_name) = state.tool_name.as_deref()
                                            && let Some(metadata) =
                                                resolver.as_ref().and_then(|resolver| {
                                                    resolver.runtime_tool_metadata(tool_name)
                                                })
                                        {
                                            stream_or_break!(
                                                metadata
                                                    .invoke_on_input_delta(input_delta_context(
                                                        id,
                                                        delta,
                                                        &step_input_messages,
                                                        &current_context,
                                                    ))
                                                    .await
                                            );
                                        }
                                    }
                                    ChatStreamPart::ToolCall(call) => {
                                        let state = tool_input_states
                                            .entry(call.tool_call_id.clone())
                                            .or_default();
                                        state.tool_name = Some(call.tool_name.clone());

                                        let declared_tool = find_declared_tool(
                                            current_tools.as_deref(),
                                            call.tool_name.as_str(),
                                        );
                                        let runtime_metadata =
                                            resolver.as_ref().and_then(|resolver| {
                                                resolver
                                                    .runtime_tool_metadata(call.tool_name.as_str())
                                            });

                                        if declared_tool.is_none() && runtime_metadata.is_none() {
                                            // Still forward the original event, but ignore local runtime hooks.
                                        } else {
                                            let input = serde_json::from_str::<Value>(&call.input)
                                                .unwrap_or_else(|_| {
                                                    Value::String(call.input.clone())
                                                });

                                            if !state.input_available_invoked
                                                && let Some(metadata) = runtime_metadata.as_ref()
                                            {
                                                stream_or_break!(
                                                    metadata
                                                        .invoke_on_input_available(
                                                            input_available_context(
                                                                &call.tool_call_id,
                                                                &input,
                                                                &step_input_messages,
                                                                &current_context,
                                                            ),
                                                        )
                                                        .await
                                                );
                                                state.input_available_invoked = true;
                                            }

                                            let approval_required = if let Some(metadata) =
                                                runtime_metadata.as_ref()
                                            {
                                                stream_or_break!(
                                                    metadata
                                                        .needs_approval(input_available_context(
                                                            &call.tool_call_id,
                                                            &input,
                                                            &step_input_messages,
                                                            &current_context,
                                                        ))
                                                        .await
                                                )
                                            } else {
                                                false
                                            };
                                            local_approval_required.insert(
                                                call.tool_call_id.clone(),
                                                approval_required,
                                            );

                                            if approval_required && on_tool_approval.is_none() {
                                                let approval_id =
                                                    format!("approval_{}", uuid::Uuid::new_v4());
                                                local_approval_parts.insert(
                                                    call.tool_call_id.clone(),
                                                    ContentPart::tool_approval_request(
                                                        approval_id.clone(),
                                                        call.tool_call_id.clone(),
                                                    ),
                                                );
                                                extra_event = Some(ChatStreamEvent::Part {
                                                    part: ChatStreamPart::ToolApprovalRequest(
                                                        ChatStreamToolApprovalRequest {
                                                            approval_id,
                                                            tool_call_id: call.tool_call_id.clone(),
                                                            provider_metadata: None,
                                                        },
                                                    ),
                                                });
                                            }
                                        }
                                    }
                                    ChatStreamPart::ToolApprovalRequest(request) => {
                                        local_approval_parts.insert(
                                            request.tool_call_id.clone(),
                                            ContentPart::tool_approval_request(
                                                request.approval_id.clone(),
                                                request.tool_call_id.clone(),
                                            ),
                                        );
                                    }
                                    _ => {}
                                },
                                ChatStreamEvent::StreamEnd { response } => {
                                    final_resp = Some(response.clone())
                                }
                                _ => {}
                            }
                            if let Some(cb) = &on_chunk {
                                cb(&ev);
                            }
                            let _ = sender.send(Ok(ev)).await;
                            if let Some(extra_event) = extra_event {
                                if let Some(cb) = &on_chunk {
                                    cb(&extra_event);
                                }
                                let _ = sender.send(Ok(extra_event)).await;
                            }
                        }
                        Err(e) => {
                            encountered_error = true;
                            let _ = sender.send(Err(e)).await;
                            break;
                        }
                    }
                }
                if let Some(response) = final_resp {
                    (
                        step_request,
                        response,
                        local_approval_parts,
                        tool_input_states
                            .into_iter()
                            .filter_map(|(tool_call_id, state)| {
                                state.input_available_invoked.then_some(tool_call_id)
                            })
                            .collect::<HashSet<_>>(),
                        local_approval_required,
                    )
                } else {
                    (
                        step_request,
                        ChatResponse::new(MessageContent::Text(acc_text.clone())),
                        local_approval_parts,
                        tool_input_states
                            .into_iter()
                            .filter_map(|(tool_call_id, state)| {
                                state.input_available_invoked.then_some(tool_call_id)
                            })
                            .collect::<HashSet<_>>(),
                        local_approval_required,
                    )
                }
            } else {
                // Non-streaming follow-up to advance conversation efficiently
                let mut request = ChatRequest::new(current_messages.clone());

                if let Some(tools) = current_tools.clone() {
                    request = request.with_tools(tools);
                }

                if let Some(tool_choice) = current_tool_choice {
                    request = request.with_tool_choice(tool_choice);
                }

                if let Some(ref common_params) = opts.common_params {
                    request = request.with_common_params(common_params.clone());
                }

                if let Some(map) = current_provider_options_map {
                    request.provider_options_map.merge_overrides(map);
                }

                request.telemetry = opts.telemetry.clone();
                let step_request = request.clone();

                let result = siumai::text::generate(
                    step_model,
                    request,
                    siumai::text::GenerateOptions::default(),
                )
                .await;

                match result {
                    Ok(r) => (
                        step_request,
                        r,
                        HashMap::new(),
                        HashSet::new(),
                        HashMap::new(),
                    ),
                    Err(e) => {
                        encountered_error = true;
                        let _ = sender.send(Err(e)).await;
                        break;
                    }
                }
            };
            annotate_response_tool_calls(&mut resp, resolver.as_deref());
            let mut step_msgs: Vec<ChatMessage> = Vec::new();
            let mut pending_tool_messages: Vec<ChatMessage> = Vec::new();
            let mut assistant_extra_parts =
                streamed_approval_parts.into_values().collect::<Vec<_>>();

            // Execute tools if any
            let tool_calls: Vec<_> = resp.tool_calls().into_iter().cloned().collect();
            let client_tool_calls = client_tool_call_count(&tool_calls);
            if !tool_calls.is_empty()
                && let Some(resolver) = resolver.as_ref()
            {
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

                        // Skip duplicate tool-call IDs already executed in previous steps.
                        if processed_call_ids.contains(tool_call_id) {
                            continue;
                        }

                        if let Some(Tool::Function { function }) = declared_tool {
                            if let Err(reason) =
                                validate_args_with_schema(&function.parameters, arguments)
                            {
                                let out =
                                    serde_json::json!({"error":"invalid_args","reason":reason});
                                let tool_msg = ChatMessage::tool_error_json(
                                    tool_call_id.clone(),
                                    tool_name.clone(),
                                    out,
                                )
                                .build();
                                pending_tool_messages.push(tool_msg);
                                continue;
                            }
                        }

                        if !streamed_input_available_ids.contains(tool_call_id)
                            && let Some(metadata) = runtime_metadata.as_ref()
                        {
                            stream_or_break!(
                                metadata
                                    .invoke_on_input_available(input_available_context(
                                        tool_call_id,
                                        arguments,
                                        &step_input_messages,
                                        &current_context,
                                    ))
                                    .await
                            );
                        }

                        let approval_required = streamed_approval_required
                            .get(tool_call_id)
                            .copied()
                            .unwrap_or_else(|| false);
                        let approval_required =
                            if streamed_approval_required.contains_key(tool_call_id) {
                                approval_required
                            } else {
                                if let Some(metadata) = runtime_metadata.as_ref() {
                                    stream_or_break!(
                                        metadata
                                            .needs_approval(input_available_context(
                                                tool_call_id,
                                                arguments,
                                                &step_input_messages,
                                                &current_context,
                                            ))
                                            .await
                                    )
                                } else {
                                    false
                                }
                            };

                        if approval_required && on_tool_approval.is_none() {
                            let already_present = assistant_extra_parts.iter().any(|part| {
                                matches!(
                                    part,
                                    ContentPart::ToolApprovalRequest {
                                        tool_call_id: existing_tool_call_id,
                                        ..
                                    } if existing_tool_call_id == tool_call_id
                                )
                            });
                            if !already_present {
                                assistant_extra_parts.push(ContentPart::tool_approval_request(
                                    format!("approval_{}", uuid::Uuid::new_v4()),
                                    tool_call_id.clone(),
                                ));
                            }
                            continue;
                        }

                        let decision = if let Some(cb) = &on_tool_approval {
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
                                    resolver.as_ref(),
                                    tool_name,
                                    tool_call_id,
                                    args,
                                    tool_dynamic,
                                    &current_context,
                                    on_preliminary_tool_result.as_deref(),
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
                        processed_call_ids.insert(tool_call_id.clone());
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
            if let Some(cb) = &on_step_finish {
                cb(&step);
            }
            step_results.push(step);
            final_response = Some(resp.clone());

            let should_stop = opts
                .stop_conditions
                .iter()
                .any(|condition| condition.should_stop(&step_results));
            let can_continue_tool_loop = should_continue_after_tool_step(
                resolver.is_some(),
                client_tool_calls,
                local_tool_results.len(),
                &pending_deferred_tool_calls,
            );

            if should_stop || !can_continue_tool_loop {
                break;
            }
        }
        if orchestrator_cancel_clone.is_cancelled() {
            if let Some(cb) = &on_abort {
                cb(&step_results);
            }
        } else if !encountered_error
            && let (Some(cb), Some(response)) = (&on_finish, final_response)
        {
            if let Some(event) =
                OrchestratorFinishEvent::from_response_and_steps(response, step_results.clone())
            {
                cb(&event);
            }
        }
        let _ = total_usage_tx.send(StepResult::merge_usage(&step_results));
        let _ = steps_tx.send(step_results);
    });

    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let stream: ChatStream = std::pin::Pin::from(Box::new(MpscStream(rx))
        as Box<dyn futures::Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync>);
    Ok(StreamOrchestration {
        stream,
        steps: steps_rx,
        total_usage: total_usage_rx,
        cancel: orchestrator_cancel,
    })
}
