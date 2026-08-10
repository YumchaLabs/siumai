//! Non-streaming Responses decoding and canonical projection.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Value, json};
use siumai_core::{
    Citation, ContentPart, DEFAULT_TOOL_INPUT_BYTE_LIMIT, Error, ErrorContext, ErrorKind,
    LanguageCallError, LanguageCompletionReason, LanguageIncompleteReason, LanguageResponse,
    LanguageTermination, ModelId, OpaqueProviderItem, PartialLanguageOutput,
    PartialLanguageOutputPart, ProviderItemRelation, ProviderProvenance, ProviderScope,
    ResponseDiagnostics, ToolCall, Usage, Warning, WarningKind,
};

use crate::openai_error::classify_stream_error;

use super::wire::{
    AnnotationWire, MessageItemWire, OutputContentPart, OutputItem, ResponseStatus,
    ResponseUsageWire, ResponseWire,
};
use super::{OPENAI_RESPONSES_OPAQUE_KIND, OPENAI_RESPONSES_PROTOCOL};

/// A lossless native response paired with its portable call outcome.
pub struct DecodedResponse {
    native: ResponseWire,
    portable: Result<LanguageResponse, LanguageCallError>,
}

impl fmt::Debug for DecodedResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedResponse")
            .field("native_status", &self.native.status)
            .field("native_output_items", &self.native.output.len())
            .field(
                "portable_outcome",
                &match &self.portable {
                    Ok(response) => match response.termination() {
                        LanguageTermination::Completed(_) => "completed",
                        LanguageTermination::Incomplete(_) => "incomplete",
                        _ => "other",
                    },
                    Err(error) if error.error().kind() == ErrorKind::Cancelled => "cancelled",
                    Err(_) => "failed",
                },
            )
            .field(
                "portable_content_parts",
                &self
                    .portable
                    .as_ref()
                    .ok()
                    .map(|response| response.content().len()),
            )
            .field(
                "has_partial_output",
                &self
                    .portable
                    .as_ref()
                    .err()
                    .is_some_and(|error| error.partial().is_some()),
            )
            .finish_non_exhaustive()
    }
}

impl DecodedResponse {
    pub fn native(&self) -> &ResponseWire {
        &self.native
    }

    pub fn portable(&self) -> Result<&LanguageResponse, &LanguageCallError> {
        self.portable.as_ref()
    }

    pub fn status(&self) -> &ResponseStatus {
        &self.native.status
    }

    pub fn into_parts(self) -> (ResponseWire, Result<LanguageResponse, LanguageCallError>) {
        (self.native, self.portable)
    }

    /// Convert a decoded terminal resource into the stable call result contract.
    ///
    /// Convert a decoded terminal resource into the portable language-call outcome.
    pub fn into_result(self) -> Result<LanguageResponse, LanguageCallError> {
        self.portable
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
    decode_response_wire_with_replay(native, scope, requested_model, true)
}

pub(crate) fn decode_response_wire_with_replay(
    native: ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
    include_native_replay: bool,
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

    let portable = match &native.status {
        ResponseStatus::Completed | ResponseStatus::Incomplete => Ok(project_response(
            &native,
            scope,
            requested_model,
            include_native_replay,
        )?),
        ResponseStatus::Failed => Err(failed_call_error(
            &native,
            scope,
            requested_model,
            ResponseDiagnostics::default(),
        )),
        ResponseStatus::Cancelled => Err(cancelled_call_error(&native, scope, requested_model)),
        ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => {
            return Err(protocol_error(
                "OpenAI returned a non-terminal or unknown Responses status to a terminal decoder",
            ));
        }
    };
    Ok(DecodedResponse { native, portable })
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
    include_native_replay: bool,
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
        project_item(
            item,
            &native.status,
            scope,
            &response_model,
            include_native_replay,
            &mut content,
        )?;
    }

    let termination = response_termination(native, &content)?;
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
        termination,
        content,
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
    include_native_replay: bool,
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
        OutputItem::FunctionCall(call) if call.arguments.len() > DEFAULT_TOOL_INPUT_BYTE_LIMIT => {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI function call arguments exceeded the byte limit",
            ));
        }
        OutputItem::FunctionCall(call) => match serde_json::from_str::<Value>(&call.arguments) {
            Ok(arguments) => {
                let tool_call = ToolCall::local(call.call_id.clone(), call.name.clone(), arguments)
                    .map_err(|source| {
                        Error::new(
                            ErrorKind::Protocol,
                            "OpenAI function call violated the canonical tool contract",
                        )
                        .with_source(source)
                    })?;
                content.push(ContentPart::ToolCall(tool_call));
            }
            Err(source) if matches!(response_status, ResponseStatus::Completed) => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "OpenAI function call contained incomplete or invalid JSON arguments",
                )
                .with_source(source));
            }
            Err(_) => {}
        },
        OutputItem::CustomToolCall(_) => {}
        OutputItem::Program(_) | OutputItem::ProgramOutput(_) => {}
        OutputItem::ProviderTool(_) | OutputItem::Unknown(_) => {}
    }

    if include_native_replay {
        content.push(ContentPart::ProviderOpaque(opaque_item(
            item, scope, model,
        )?));
    }
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
        provider: BTreeMap::new(),
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
    let provenance = ProviderProvenance::from_scope(scope, model.clone()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "OpenAI Responses replay requires an explicit provider replay domain",
        )
        .with_source(source)
    })?;
    let mut builder = OpaqueProviderItem::builder(provenance, OPENAI_RESPONSES_OPAQUE_KIND, data);
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
    }
    if let Some(details) = &wire.output_tokens_details {
        usage = usage.with_reasoning_tokens(details.reasoning_tokens);
        if let Some(value) = details.orchestration_output_tokens {
            usage = usage.with_provider_value("orchestration_output_tokens", value);
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
    usage
}

fn response_metadata(native: &ResponseWire) -> Value {
    json!({
        "status": public_response_status(&native.status),
        "created_at": native.created_at,
        "has_incomplete_details": native.incomplete_details.is_some(),
        "has_error": native.error.is_some(),
        "has_reasoning": native.reasoning.is_some(),
        "extra_field_count": native.extra.len(),
    })
}

fn public_response_status(status: &ResponseStatus) -> &'static str {
    match status {
        ResponseStatus::Queued => "queued",
        ResponseStatus::InProgress => "in_progress",
        ResponseStatus::Completed => "completed",
        ResponseStatus::Incomplete => "incomplete",
        ResponseStatus::Cancelled => "cancelled",
        ResponseStatus::Failed => "failed",
        ResponseStatus::Other(_) => "other",
    }
}

fn response_termination(
    native: &ResponseWire,
    content: &[ContentPart],
) -> Result<LanguageTermination, Error> {
    match &native.status {
        ResponseStatus::Completed => {
            if content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall(_)))
            {
                Ok(LanguageTermination::Completed(
                    LanguageCompletionReason::ToolCalls,
                ))
            } else if content
                .iter()
                .any(|part| matches!(part, ContentPart::Refusal { .. }))
            {
                Ok(LanguageTermination::Completed(
                    LanguageCompletionReason::Refusal,
                ))
            } else {
                Ok(LanguageTermination::Completed(
                    LanguageCompletionReason::Stop,
                ))
            }
        }
        ResponseStatus::Incomplete => match native
            .incomplete_details
            .as_ref()
            .map(|details| details.reason.as_str())
        {
            Some("max_output_tokens" | "max_tokens") => Ok(LanguageTermination::Incomplete(
                LanguageIncompleteReason::MaxOutputTokens,
            )),
            Some("content_filter" | "safety") => Ok(LanguageTermination::Incomplete(
                LanguageIncompleteReason::ContentFilter,
            )),
            Some(reason) => Ok(LanguageTermination::Incomplete(
                LanguageIncompleteReason::Other(reason.to_string()),
            )),
            None => Ok(LanguageTermination::Incomplete(
                LanguageIncompleteReason::Other("incomplete".to_string()),
            )),
        },
        ResponseStatus::Cancelled | ResponseStatus::Failed => Err(protocol_error(
            "failed or cancelled Responses resources are portable call outcomes, not responses",
        )),
        ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => Err(
            protocol_error("non-terminal Responses resources cannot become language responses"),
        ),
    }
}

fn project_partial_output(
    response: &ResponseWire,
) -> Result<Option<PartialLanguageOutput>, siumai_core::PartialLanguageOutputError> {
    let mut content = Vec::new();
    for item in &response.output {
        match item {
            OutputItem::Message(message) => {
                for part in &message.content {
                    match part {
                        OutputContentPart::Text(text) => {
                            content.push(PartialLanguageOutputPart::Text {
                                text: text.text.clone(),
                            });
                        }
                        OutputContentPart::Refusal(refusal) => {
                            content.push(PartialLanguageOutputPart::Refusal {
                                reason: Some(refusal.refusal.clone()),
                            });
                        }
                        OutputContentPart::Unknown(_) => {}
                    }
                }
            }
            OutputItem::Reasoning(reasoning) => {
                content.extend(
                    reasoning
                        .summary
                        .iter()
                        .chain(reasoning.content.iter())
                        .map(|part| PartialLanguageOutputPart::Reasoning {
                            text: part.text.clone(),
                        }),
                );
            }
            OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::Program(_)
            | OutputItem::ProgramOutput(_)
            | OutputItem::ProviderTool(_)
            | OutputItem::Unknown(_) => {}
        }
    }
    let usage = response
        .usage
        .as_ref()
        .map(decode_usage)
        .unwrap_or_default();
    if content.is_empty() && usage == Usage::default() {
        Ok(None)
    } else {
        PartialLanguageOutput::new(content, usage).map(Some)
    }
}

fn partial_or_source(
    error: Error,
    response: &ResponseWire,
) -> (Error, Option<PartialLanguageOutput>) {
    match project_partial_output(response) {
        Ok(partial) => (error, partial),
        Err(source) => (error.with_source(source), None),
    }
}

fn failed_call_error(
    response: &ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
    diagnostics: ResponseDiagnostics,
) -> LanguageCallError {
    let error = failed_response_error(response, scope, requested_model, diagnostics);
    let (error, partial) = partial_or_source(error, response);
    LanguageCallError::new(error, partial)
}

fn cancelled_call_error(
    response: &ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> LanguageCallError {
    let error = Error::cancelled("OpenAI cancelled the Responses generation")
        .with_context(error_context(scope, requested_model));
    let (error, partial) = partial_or_source(error, response);
    LanguageCallError::new(error, partial)
}

pub(crate) fn failed_response_error(
    response: &ResponseWire,
    scope: &ProviderScope,
    requested_model: &ModelId,
    diagnostics: ResponseDiagnostics,
) -> Error {
    if let Some(source) = &response.error {
        return classify_stream_error(
            &json!({ "error": source }),
            diagnostics,
            "OpenAI Responses generation failed",
        )
        .with_context(error_context(scope, requested_model));
    }
    Error::new(ErrorKind::Provider, "OpenAI Responses generation failed")
        .with_diagnostics(diagnostics)
        .with_context(error_context(scope, requested_model))
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
