//! Non-streaming Responses decoding and canonical projection.

use std::collections::BTreeMap;

use serde_json::{Value, json};
use siumai_core::{
    Citation, ContentPart, Error, ErrorContext, ErrorKind, ExecutionOwner, FinishReason,
    LanguageIncompleteReason, LanguageResponse, LanguageResponseStatus, ModelId,
    OpaqueProviderItem, ProviderItemRelation, ProviderProvenance, ProviderScope, ToolCall,
    ToolOutcome, ToolResult, Usage, Warning, WarningKind,
};

use super::wire::{
    AnnotationWire, MessageItemWire, OutputContentPart, OutputItem, ResponseStatus,
    ResponseUsageWire, ResponseWire,
};
use super::{OPENAI_RESPONSES_OPAQUE_KIND, OPENAI_RESPONSES_PROTOCOL};

/// A lossless native response paired with its portable canonical projection.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedResponse {
    native: ResponseWire,
    canonical: LanguageResponse,
}

impl DecodedResponse {
    pub fn native(&self) -> &ResponseWire {
        &self.native
    }

    pub fn canonical(&self) -> &LanguageResponse {
        &self.canonical
    }

    pub fn status(&self) -> &ResponseStatus {
        &self.native.status
    }

    pub fn into_parts(self) -> (ResponseWire, LanguageResponse) {
        (self.native, self.canonical)
    }

    /// Convert a decoded terminal resource into the stable call result contract.
    ///
    /// Provider-returned failed and cancelled resources are still successful
    /// protocol decodes. Their terminal state remains on `LanguageResponse`;
    /// outer errors are reserved for failures that never formed a response.
    pub fn into_result(self) -> Result<LanguageResponse, Error> {
        match &self.native.status {
            ResponseStatus::Completed
            | ResponseStatus::Incomplete
            | ResponseStatus::Cancelled
            | ResponseStatus::Failed => Ok(self.canonical),
            ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => {
                Err(protocol_error(
                    "OpenAI returned a non-terminal Responses resource to a non-streaming call",
                ))
            }
        }
    }
}

/// Decode any Responses resource, including background queued/in-progress state.
///
/// This native entry point deliberately does not project into `LanguageResponse`,
/// whose stable contract is terminal-only. Provider-specific background APIs can
/// poll this resource and pass a terminal body to [`decode_response`] later.
pub fn decode_response_resource(body: &[u8]) -> Result<ResponseWire, Error> {
    let native = serde_json::from_slice::<ResponseWire>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "provider returned malformed OpenAI Responses JSON",
        )
        .with_source(source)
    })?;
    validate_resource_identity(&native)?;
    Ok(native)
}

/// Decode a complete non-streaming Responses JSON body without flattening native
/// output items into chat-only structures.
pub fn decode_response(
    body: &[u8],
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<DecodedResponse, Error> {
    let native = decode_response_resource(body)?;
    decode_response_wire(native, scope, requested_model)
}

pub(crate) fn decode_response_wire(
    native: ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<DecodedResponse, Error> {
    validate_resource_identity(&native)?;
    if matches!(
        &native.status,
        ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_)
    ) {
        return Err(protocol_error(
            "OpenAI returned a non-terminal or unknown Responses status to a terminal decoder",
        ));
    }

    let canonical = project_response(&native, scope, requested_model)?;
    Ok(DecodedResponse { native, canonical })
}

fn validate_resource_identity(native: &ResponseWire) -> Result<(), Error> {
    if native.id.trim().is_empty() {
        return Err(protocol_error("OpenAI Responses resource omitted its ID"));
    }
    if native.model.trim().is_empty() {
        return Err(protocol_error(
            "OpenAI Responses resource omitted its model ID",
        ));
    }
    ModelId::new(native.model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses resource contained an invalid model ID",
        )
        .with_source(source)
    })?;
    Ok(())
}

pub(crate) fn project_response(
    native: &ResponseWire,
    scope: &ProviderScope,
    _requested_model: &ModelId,
) -> Result<LanguageResponse, Error> {
    let response_model = ModelId::new(native.model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses resource contained an invalid model ID",
        )
        .with_source(source)
    })?;
    let mut content = Vec::new();
    for item in &native.output {
        project_item(item, &native.status, scope, &response_model, &mut content)?;
    }

    let status = response_status(native);
    let finish_reason = finish_reason(native, &content);
    let mut warnings = Vec::new();
    if matches!(&native.status, ResponseStatus::Incomplete) {
        warnings.push(Warning::new(
            WarningKind::PartialResult,
            "OpenAI returned an incomplete Responses result",
        ));
    }

    let mut provider = BTreeMap::new();
    provider.insert(
        OPENAI_RESPONSES_PROTOCOL.to_string(),
        response_metadata(native),
    );

    LanguageResponse::new(
        status,
        content,
        finish_reason,
        native.usage.as_ref().map(decode_usage).unwrap_or_default(),
    )
    .map(|response| {
        response
            .with_id(native.id.clone())
            .with_model(response_model)
            .with_warnings(warnings)
            .with_provider_metadata(provider)
    })
    .map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses resource produced an invalid canonical response",
        )
        .with_source(source)
    })
}

fn project_item(
    item: &OutputItem,
    response_status: &ResponseStatus,
    scope: &ProviderScope,
    model: &ModelId,
    content: &mut Vec<ContentPart>,
) -> Result<(), Error> {
    match item {
        OutputItem::Message(message) => project_message(message, content),
        OutputItem::Reasoning(reasoning) => {
            content.extend(
                reasoning
                    .summary
                    .iter()
                    .chain(reasoning.content.iter())
                    .map(|part| ContentPart::Reasoning {
                        text: part.text.clone(),
                    }),
            );
        }
        OutputItem::FunctionCall(call) => match serde_json::from_str::<Value>(&call.arguments) {
            Ok(arguments) => content.push(ContentPart::ToolCall(ToolCall {
                id: call.call_id.clone(),
                name: call.name.clone(),
                arguments,
                owner: ExecutionOwner::Local,
            })),
            Err(source) if matches!(response_status, ResponseStatus::Completed) => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "OpenAI function call contained incomplete or invalid JSON arguments",
                )
                .with_source(source));
            }
            Err(_) => {}
        },
        OutputItem::CustomToolCall(call) => {
            content.push(ContentPart::ToolCall(ToolCall {
                id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: Value::String(call.input.clone()),
                owner: ExecutionOwner::Local,
            }));
        }
        OutputItem::Program(program) => {
            content.push(ContentPart::ToolCall(ToolCall {
                id: program.call_id.clone(),
                name: "openai.programmatic_tool_calling".to_string(),
                arguments: json!({
                    "code": program.code,
                    "fingerprint": program.fingerprint,
                }),
                owner: ExecutionOwner::Provider {
                    provider: scope.provider_id().clone(),
                },
            }));
        }
        OutputItem::ProgramOutput(output) => {
            content.push(ContentPart::ToolResult(project_program_output(output)));
        }
        OutputItem::ProviderTool(_) | OutputItem::Unknown(_) => {}
    }

    content.push(ContentPart::ProviderOpaque(opaque_item(
        item, scope, model,
    )?));
    Ok(())
}

fn project_message(message: &MessageItemWire, content: &mut Vec<ContentPart>) {
    for part in &message.content {
        match part {
            OutputContentPart::Text(text) => {
                content.push(ContentPart::Text {
                    text: text.text.clone(),
                });
                content.extend(
                    text.annotations
                        .iter()
                        .enumerate()
                        .map(|(index, annotation)| {
                            ContentPart::Citation(project_citation(&message.id, index, annotation))
                        }),
                );
            }
            OutputContentPart::Refusal(refusal) => {
                content.push(ContentPart::Refusal {
                    reason: Some(refusal.refusal.clone()),
                });
            }
            OutputContentPart::Unknown(_) => {}
        }
    }
}

pub(crate) fn project_citation(
    message_id: &str,
    index: usize,
    annotation: &AnnotationWire,
) -> Citation {
    let source_id = annotation
        .fields
        .get("file_id")
        .or_else(|| annotation.fields.get("container_id"))
        .or_else(|| annotation.fields.get("url"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{message_id}:annotation:{index}"));
    let mut provider = BTreeMap::new();
    provider.insert(
        OPENAI_RESPONSES_PROTOCOL.to_string(),
        serde_json::to_value(annotation).unwrap_or(Value::Null),
    );
    Citation {
        source_id,
        title: annotation
            .fields
            .get("title")
            .or_else(|| annotation.fields.get("filename"))
            .and_then(Value::as_str)
            .map(str::to_string),
        url: annotation
            .fields
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string),
        start: annotation.fields.get("start_index").and_then(Value::as_u64),
        end: annotation.fields.get("end_index").and_then(Value::as_u64),
        provider,
    }
}

pub(crate) fn project_program_output(output: &super::wire::ProgramOutputItemWire) -> ToolResult {
    let value = serde_json::from_str(&output.result)
        .unwrap_or_else(|_| Value::String(output.result.clone()));
    let outcome = match &output.status {
        super::wire::ItemStatus::Completed => ToolOutcome::Success {
            value: json!({
                "result": value,
                "status": output.status.as_str(),
            }),
        },
        super::wire::ItemStatus::InProgress
        | super::wire::ItemStatus::Incomplete
        | super::wire::ItemStatus::Failed
        | super::wire::ItemStatus::Other(_) => ToolOutcome::ExecutionFailed {
            message: "OpenAI programmatic tool execution did not complete".to_string(),
            retryable: false,
            details: None,
        },
    };
    ToolResult {
        call_id: output.call_id.clone(),
        name: "openai.programmatic_tool_calling".to_string(),
        outcome,
    }
}

pub(crate) fn opaque_item(
    item: &OutputItem,
    scope: &ProviderScope,
    model: &ModelId,
) -> Result<OpaqueProviderItem, Error> {
    let data = item.to_value().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "failed to preserve an OpenAI Responses output item",
        )
        .with_source(source)
    })?;
    let mut builder = OpaqueProviderItem::builder(
        ProviderProvenance {
            provider: scope.provider_id().clone(),
            platform: scope.platform().map(ToString::to_string),
            protocol: OPENAI_RESPONSES_PROTOCOL.to_string(),
            model: model.clone(),
        },
        OPENAI_RESPONSES_OPAQUE_KIND,
        data,
    );
    if let Some(item_id) = item.id() {
        builder = builder.item_id(item_id);
    }
    let relations = native_relations(item).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses item contained an invalid identity relation",
        )
        .with_source(source)
    })?;
    builder.relations(relations).build().map_err(|source| {
        Error::new(
            ErrorKind::ResponseLimit,
            "OpenAI Responses output item exceeded the opaque replay limit",
        )
        .with_source(source)
    })
}

fn native_relations(
    item: &OutputItem,
) -> Result<Vec<ProviderItemRelation>, siumai_core::OpaqueProviderItemError> {
    let mut relations = Vec::new();
    if let Some(call_id) = item.call_id() {
        relations.push(ProviderItemRelation::call(call_id)?);
    }
    match item {
        OutputItem::FunctionCall(call) => {
            if let Some(caller_id) = call
                .caller
                .as_ref()
                .and_then(|caller| caller.caller_id.as_deref())
            {
                relations.push(ProviderItemRelation::caller(caller_id)?);
            }
        }
        OutputItem::CustomToolCall(call) => {
            if let Some(caller_id) = call
                .caller
                .as_ref()
                .and_then(|caller| caller.caller_id.as_deref())
            {
                relations.push(ProviderItemRelation::caller(caller_id)?);
            }
        }
        OutputItem::ProviderTool(tool) => {
            if let Some(caller_id) = tool
                .caller()
                .and_then(|caller| caller.get("caller_id"))
                .and_then(Value::as_str)
            {
                relations.push(ProviderItemRelation::caller(caller_id)?);
            }
        }
        OutputItem::Unknown(item) => {
            if let Some(caller_id) = item
                .caller()
                .and_then(|caller| caller.get("caller_id"))
                .and_then(Value::as_str)
            {
                relations.push(ProviderItemRelation::caller(caller_id)?);
            }
        }
        _ => {}
    }
    Ok(relations)
}

pub(crate) fn decode_usage(wire: &ResponseUsageWire) -> Usage {
    let mut usage = Usage::default()
        .with_input_tokens(wire.input_tokens)
        .with_output_tokens(wire.output_tokens)
        .with_total_tokens(wire.total_tokens);

    if let Some(details) = &wire.input_tokens_details {
        usage = usage
            .with_cache_read_tokens(details.cached_tokens)
            .with_cache_write_tokens(details.cache_write_tokens);
        if let Some(value) = details.orchestration_input_tokens {
            usage = usage.with_provider_value("orchestration_input_tokens", value);
        }
        if let Some(value) = details.orchestration_input_cached_tokens {
            usage = usage.with_provider_value("orchestration_input_cached_tokens", value);
        }
        for (key, value) in &details.extra {
            usage = usage.with_provider_value(format!("input_tokens_details.{key}"), value.clone());
        }
    }
    if let Some(details) = &wire.output_tokens_details {
        usage = usage.with_reasoning_tokens(details.reasoning_tokens);
        if let Some(value) = details.orchestration_output_tokens {
            usage = usage.with_provider_value("orchestration_output_tokens", value);
        }
        for (key, value) in &details.extra {
            usage =
                usage.with_provider_value(format!("output_tokens_details.{key}"), value.clone());
        }
    }
    let orchestration = wire
        .input_tokens_details
        .as_ref()
        .and_then(|details| details.orchestration_input_tokens)
        .unwrap_or(0)
        .saturating_add(
            wire.output_tokens_details
                .as_ref()
                .and_then(|details| details.orchestration_output_tokens)
                .unwrap_or(0),
        );
    if orchestration > 0 {
        usage = usage.with_orchestration_tokens(orchestration);
    }
    for (key, value) in &wire.extra {
        usage = usage.with_provider_value(key.clone(), value.clone());
    }
    usage
}

fn response_metadata(native: &ResponseWire) -> Value {
    json!({
        "status": native.status.as_str(),
        "created_at": native.created_at,
        "incomplete_details": native.incomplete_details,
        "error": native.error,
        "reasoning": native.reasoning,
        "extra": native.extra,
    })
}

fn finish_reason(native: &ResponseWire, content: &[ContentPart]) -> FinishReason {
    match &native.status {
        ResponseStatus::Completed => {
            if content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(_)))
            {
                FinishReason::ToolCalls
            } else if content
                .iter()
                .any(|part| matches!(part, ContentPart::Refusal { .. }))
            {
                FinishReason::Refusal
            } else {
                FinishReason::Stop
            }
        }
        ResponseStatus::Incomplete => match native
            .incomplete_details
            .as_ref()
            .map(|details| details.reason.as_str())
        {
            Some("max_output_tokens" | "max_tokens") => FinishReason::Length,
            Some("content_filter" | "safety") => FinishReason::ContentFilter,
            Some(reason) => FinishReason::Other(format!("incomplete:{reason}")),
            None => FinishReason::Other("incomplete".to_string()),
        },
        ResponseStatus::Cancelled => FinishReason::Cancelled,
        ResponseStatus::Failed => FinishReason::Error,
        ResponseStatus::Queued => FinishReason::Other("queued".to_string()),
        ResponseStatus::InProgress => FinishReason::Other("in_progress".to_string()),
        ResponseStatus::Other(status) => FinishReason::Other(status.clone()),
    }
}

fn response_status(native: &ResponseWire) -> LanguageResponseStatus {
    match &native.status {
        ResponseStatus::Completed => LanguageResponseStatus::Completed,
        ResponseStatus::Incomplete => LanguageResponseStatus::Incomplete {
            reason: native.incomplete_details.as_ref().map(|details| {
                match details.reason.as_str() {
                    "max_output_tokens" | "max_tokens" => LanguageIncompleteReason::MaxOutputTokens,
                    "content_filter" | "safety" => LanguageIncompleteReason::ContentFilter,
                    reason => LanguageIncompleteReason::Other(reason.to_string()),
                }
            }),
        },
        ResponseStatus::Failed => LanguageResponseStatus::Failed,
        ResponseStatus::Cancelled => LanguageResponseStatus::Cancelled,
        ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => {
            LanguageResponseStatus::Completed
        }
    }
}

pub(crate) fn failed_response_error(
    response: &ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Error {
    let mut error = Error::new(ErrorKind::Provider, "OpenAI Responses generation failed")
        .with_context(error_context(scope, requested_model));
    if let Some(source) = &response.error {
        error = error.with_source(NativeResponseFailure {
            code: source.code.clone(),
            kind: source.kind.clone(),
            message: source.message.clone(),
            param: source.param.clone(),
        });
    }
    error
}

pub(crate) fn error_context(scope: &ProviderScope, requested_model: &ModelId) -> ErrorContext {
    ErrorContext {
        operation: None,
        provider: Some(scope.provider_id().clone()),
        route: None,
        model: Some(requested_model.clone()),
    }
}

pub(crate) fn protocol_error(message: &'static str) -> Error {
    Error::new(ErrorKind::Protocol, message)
}

struct NativeResponseFailure {
    code: Option<String>,
    kind: Option<String>,
    message: String,
    param: Option<String>,
}

impl std::fmt::Debug for NativeResponseFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeResponseFailure")
            .field("code", &self.code)
            .field("kind", &self.kind)
            .field("message_present", &!self.message.is_empty())
            .field("param_present", &self.param.is_some())
            .finish()
    }
}

impl std::fmt::Display for NativeResponseFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("native OpenAI Responses failure")
    }
}

impl std::error::Error for NativeResponseFailure {}
