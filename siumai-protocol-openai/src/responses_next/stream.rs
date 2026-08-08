//! Stateful decoder for OpenAI Responses SSE data values.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use siumai_core::{
    DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error, ErrorKind, ExecutionOwner,
    LanguageStreamDecoder, LanguageStreamEvent, ModelId, ProviderScope, StreamTerminal, ToolCall,
};

use super::OPENAI_RESPONSES_OPAQUE_KIND;
use super::response::{
    decode_response_wire, decode_usage, failed_response_error, opaque_item, project_citation,
    protocol_error,
};
use super::wire::{
    AnnotationWire, OutputContentPart, OutputItem, ResponseErrorWire, ResponseStatus, ResponseWire,
    StreamEventWire,
};

/// Stateful, native Responses stream decoder.
///
/// SSE framing is transport-owned. This type consumes one `data` value at a
/// time, preserves native items, and emits exactly one canonical terminal.
pub struct ResponsesStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    lifecycle: DecoderLifecycle,
    started: bool,
    response_id: Option<String>,
    response_model: Option<ModelId>,
    last_sequence: Option<u64>,
    items: BTreeMap<u64, OutputItem>,
    item_indices: BTreeMap<String, u64>,
    completed_items: BTreeSet<u64>,
    text_started: BTreeSet<String>,
    text_ended: BTreeSet<String>,
    reasoning_started: BTreeSet<String>,
    reasoning_ended: BTreeSet<String>,
    tool_inputs: BTreeMap<String, ToolInputAssembly>,
    refusals: BTreeMap<String, String>,
    emitted_refusals: BTreeSet<String>,
    emitted_citations: BTreeSet<String>,
    terminal_native: Option<ResponseWire>,
}

impl std::fmt::Debug for ResponsesStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResponsesStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("started", &self.started)
            .field("terminal", &self.lifecycle.terminal_seen())
            .field("finished", &self.lifecycle.finish_seen())
            .field("item_count", &self.items.len())
            .finish()
    }
}

#[derive(Debug, Default)]
struct ToolInputAssembly {
    call_id: String,
    name: String,
    input: String,
    custom: bool,
}

impl ResponsesStreamDecoder {
    pub fn new(scope: ProviderScope, requested_model: ModelId) -> Self {
        Self {
            scope,
            requested_model,
            lifecycle: DecoderLifecycle::default(),
            started: false,
            response_id: None,
            response_model: None,
            last_sequence: None,
            items: BTreeMap::new(),
            item_indices: BTreeMap::new(),
            completed_items: BTreeSet::new(),
            text_started: BTreeSet::new(),
            text_ended: BTreeSet::new(),
            reasoning_started: BTreeSet::new(),
            reasoning_ended: BTreeSet::new(),
            tool_inputs: BTreeMap::new(),
            refusals: BTreeMap::new(),
            emitted_refusals: BTreeSet::new(),
            emitted_citations: BTreeSet::new(),
            terminal_native: None,
        }
    }

    pub fn decode(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::decode(self, data)
    }

    pub fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::finish(self)
    }

    pub fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }

    pub fn terminal_native(&self) -> Option<&ResponseWire> {
        self.terminal_native.as_ref()
    }

    fn decode_frame(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if data.trim() == "[DONE]" {
            return Err(Error::unexpected_eof());
        }
        let event = serde_json::from_str::<StreamEventWire>(data).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed OpenAI Responses stream JSON",
            )
            .with_source(source)
        })?;
        self.observe_sequence(event.sequence_number)?;
        match event.kind.as_str() {
            "response.created" | "response.queued" | "response.in_progress" => {
                self.observe_progress_response(&event)
            }
            "response.output_item.added" => self.output_item_added(&event),
            "response.output_item.done" => self.output_item_done(&event),
            "response.output_text.delta" => self.text_delta(&event),
            "response.output_text.done" => self.text_done(&event),
            "response.refusal.delta" => self.refusal_delta(&event),
            "response.refusal.done" => self.refusal_done(&event),
            "response.output_text.annotation.added" => self.annotation_added(&event),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                self.reasoning_delta(&event)
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                self.reasoning_done(&event)
            }
            "response.function_call_arguments.delta" => self.tool_input_delta(&event, false),
            "response.function_call_arguments.done" => self.tool_input_done(&event, false),
            "response.custom_tool_call_input.delta" => self.tool_input_delta(&event, true),
            "response.custom_tool_call_input.done" => self.tool_input_done(&event, true),
            "response.completed"
            | "response.incomplete"
            | "response.cancelled"
            | "response.failed" => self.terminal_event(&event),
            "error" => self.error_event(&event),
            _ => self.opaque_stream_event(&event),
        }
    }

    fn observe_sequence(&mut self, sequence: Option<u64>) -> Result<(), Error> {
        let Some(sequence) = sequence else {
            return Ok(());
        };
        if self
            .last_sequence
            .is_some_and(|previous| sequence <= previous)
        {
            return Err(protocol_error(
                "OpenAI Responses stream sequence numbers were not strictly increasing",
            ));
        }
        self.last_sequence = Some(sequence);
        Ok(())
    }

    fn observe_progress_response(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let response = decode_event_response(event)?.ok_or_else(|| {
            protocol_error("OpenAI Responses lifecycle event omitted its response")
        })?;
        if matches!(
            &response.status,
            ResponseStatus::Completed
                | ResponseStatus::Incomplete
                | ResponseStatus::Cancelled
                | ResponseStatus::Failed
        ) {
            return Err(protocol_error(
                "OpenAI Responses progress event carried a terminal response status",
            ));
        }
        self.observe_identity(&response)
    }

    fn observe_identity(
        &mut self,
        response: &ResponseWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        if response.id.trim().is_empty() || response.model.trim().is_empty() {
            return Err(protocol_error(
                "OpenAI Responses lifecycle event omitted response identity",
            ));
        }
        if self
            .response_id
            .as_deref()
            .is_some_and(|existing| existing != response.id)
        {
            return Err(protocol_error(
                "OpenAI Responses stream changed its response ID",
            ));
        }
        let model = ModelId::new(response.model.clone()).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses stream contained an invalid model ID",
            )
            .with_source(source)
        })?;
        if self
            .response_model
            .as_ref()
            .is_some_and(|existing| existing != &model)
        {
            return Err(protocol_error(
                "OpenAI Responses stream changed its model ID",
            ));
        }
        self.response_id.get_or_insert_with(|| response.id.clone());
        self.response_model.get_or_insert(model);
        if self.started {
            return Ok(Vec::new());
        }
        self.started = true;
        Ok(vec![LanguageStreamEvent::Started {
            id: self.response_id.clone(),
            model: self.response_model.clone(),
        }])
    }

    fn output_item_added(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let output_index = required_u64(event, "output_index")?;
        if self.items.contains_key(&output_index) {
            return Err(protocol_error(
                "OpenAI Responses stream reused an output index",
            ));
        }
        let item = decode_event_item(event)?
            .ok_or_else(|| protocol_error("OpenAI Responses output-item event omitted its item"))?;
        let item_id = item
            .id()
            .ok_or_else(|| protocol_error("OpenAI Responses output item omitted its item ID"))?;
        if item_id.is_empty() || self.item_indices.contains_key(item_id) {
            return Err(protocol_error(
                "OpenAI Responses stream reused or omitted an output item ID",
            ));
        }
        self.item_indices.insert(item_id.to_string(), output_index);

        let mut events = Vec::new();
        match &item {
            OutputItem::FunctionCall(call) => {
                self.start_tool_input(
                    item_id,
                    &call.call_id,
                    &call.name,
                    &call.arguments,
                    false,
                    &mut events,
                )?;
            }
            OutputItem::CustomToolCall(call) => {
                self.start_tool_input(
                    item_id,
                    &call.call_id,
                    &call.name,
                    &call.input,
                    true,
                    &mut events,
                )?;
            }
            _ => {}
        }
        self.items.insert(output_index, item);
        Ok(events)
    }

    fn output_item_done(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let output_index = required_u64(event, "output_index")?;
        if self.completed_items.contains(&output_index) {
            return Err(protocol_error(
                "OpenAI Responses stream completed an output item more than once",
            ));
        }
        let item = decode_event_item(event)?
            .ok_or_else(|| protocol_error("OpenAI Responses output-item event omitted its item"))?;
        let item_id = item
            .id()
            .ok_or_else(|| protocol_error("OpenAI Responses output item omitted its item ID"))?;
        let expected_index = self.item_indices.get(item_id).copied().ok_or_else(|| {
            protocol_error("OpenAI Responses completed an item that was never added")
        })?;
        if expected_index != output_index {
            return Err(protocol_error(
                "OpenAI Responses output item changed its output index",
            ));
        }
        if self
            .items
            .get(&output_index)
            .is_some_and(|added| added.kind() != item.kind())
        {
            return Err(protocol_error(
                "OpenAI Responses output item changed its type",
            ));
        }

        let mut events = Vec::new();
        match &item {
            OutputItem::Message(message) => self.finish_message(message, &mut events),
            OutputItem::Reasoning(reasoning) => {
                self.finish_reasoning_item(item_id, reasoning, &mut events);
            }
            OutputItem::FunctionCall(call) => {
                self.emit_function_call(item_id, call, &mut events)?;
            }
            OutputItem::CustomToolCall(call) => {
                self.validate_custom_tool_call(item_id, call)?;
            }
            OutputItem::Program(_) | OutputItem::ProgramOutput(_) => {}
            OutputItem::ProviderTool(_) | OutputItem::Unknown(_) => {}
        }
        events.push(LanguageStreamEvent::ProviderOpaque(opaque_item(
            &item,
            &self.scope,
            self.response_model
                .as_ref()
                .unwrap_or(&self.requested_model),
        )?));
        self.items.insert(output_index, item);
        self.completed_items.insert(output_index);
        Ok(events)
    }

    fn text_delta(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "text", content_index);
        let delta = required_str(event, "delta")?.to_string();
        let mut events = Vec::new();
        if self.text_started.insert(id.clone()) {
            events.push(LanguageStreamEvent::TextStart { id: id.clone() });
        }
        events.push(LanguageStreamEvent::TextDelta { id, delta });
        Ok(events)
    }

    fn text_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "text", content_index);
        let mut events = Vec::new();
        if self.text_started.insert(id.clone()) {
            let text = event
                .field("text")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol_error("OpenAI text completion omitted its final text"))?;
            events.push(LanguageStreamEvent::TextStart { id: id.clone() });
            events.push(LanguageStreamEvent::TextDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        if self.text_ended.insert(id.clone()) {
            events.push(LanguageStreamEvent::TextEnd { id });
        }
        Ok(events)
    }

    fn reasoning_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let lane = if event.kind.contains("summary") {
            "summary"
        } else {
            "content"
        };
        let id = content_id(item_id, lane, index);
        let delta = required_str(event, "delta")?.to_string();
        let mut events = Vec::new();
        self.start_reasoning(&id, &mut events);
        events.push(LanguageStreamEvent::ReasoningDelta { id, delta });
        Ok(events)
    }

    fn reasoning_done(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let lane = if event.kind.contains("summary") {
            "summary"
        } else {
            "content"
        };
        let id = content_id(item_id, lane, index);
        let mut events = Vec::new();
        if !self.reasoning_started.contains(&id)
            && let Some(text) = event.field("text").and_then(Value::as_str)
        {
            self.start_reasoning(&id, &mut events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        self.end_reasoning(&id, &mut events);
        Ok(events)
    }

    fn start_reasoning(&mut self, id: &str, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.insert(id.to_string()) {
            events.push(LanguageStreamEvent::ReasoningStart { id: id.to_string() });
        }
    }

    fn end_reasoning(&mut self, id: &str, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.contains(id) && self.reasoning_ended.insert(id.to_string()) {
            events.push(LanguageStreamEvent::ReasoningEnd { id: id.to_string() });
        }
    }

    fn finish_reasoning_item(
        &mut self,
        item_id: &str,
        reasoning: &super::wire::ReasoningItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        for (index, part) in reasoning.summary.iter().enumerate() {
            self.finish_reasoning_part(item_id, "summary", index as u64, &part.text, events);
        }
        for (index, part) in reasoning.content.iter().enumerate() {
            self.finish_reasoning_part(item_id, "content", index as u64, &part.text, events);
        }
    }

    fn finish_reasoning_part(
        &mut self,
        item_id: &str,
        lane: &str,
        index: u64,
        text: &str,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        let id = content_id(item_id, lane, index);
        if !self.reasoning_started.contains(&id) {
            self.start_reasoning(&id, events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        self.end_reasoning(&id, events);
    }

    fn refusal_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "refusal", content_index);
        self.refusals
            .entry(id)
            .or_default()
            .push_str(required_str(event, "delta")?);
        Ok(Vec::new())
    }

    fn refusal_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "refusal", content_index);
        let reason = event
            .field("refusal")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.refusals.remove(&id));
        if self.emitted_refusals.insert(id) {
            Ok(vec![LanguageStreamEvent::Refusal { reason }])
        } else {
            Ok(Vec::new())
        }
    }

    fn annotation_added(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let annotation_index = required_u64(event, "annotation_index")?;
        let annotation_position = usize::try_from(annotation_index).map_err(|_| {
            protocol_error("OpenAI Responses annotation index exceeded the platform limit")
        })?;
        let annotation = event
            .field("annotation")
            .cloned()
            .ok_or_else(|| protocol_error("OpenAI citation event omitted its annotation"))?;
        let annotation =
            serde_json::from_value::<AnnotationWire>(annotation).map_err(|source| {
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI citation event contained a malformed annotation",
                )
                .with_source(source)
            })?;
        let key = format!("{item_id}:{annotation_index}");
        if !self.emitted_citations.insert(key) {
            return Ok(Vec::new());
        }
        Ok(vec![LanguageStreamEvent::Citation(project_citation(
            item_id,
            annotation_position,
            &annotation,
        ))])
    }

    fn tool_input_delta(
        &mut self,
        event: &StreamEventWire,
        custom: bool,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        let delta = required_str(event, "delta")?.to_string();
        let assembly = self
            .tool_inputs
            .get_mut(item_id)
            .ok_or_else(|| protocol_error("OpenAI tool-input delta preceded its output item"))?;
        if assembly.custom != custom {
            return Err(protocol_error(
                "OpenAI tool-input event changed its tool kind",
            ));
        }
        append_tool_input(&mut assembly.input, &delta)?;
        if custom {
            return Ok(Vec::new());
        }
        Ok(vec![LanguageStreamEvent::ToolInputDelta {
            id: assembly.call_id.clone(),
            delta,
        }])
    }

    fn tool_input_done(
        &mut self,
        event: &StreamEventWire,
        custom: bool,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        let field = if custom { "input" } else { "arguments" };
        let completed = required_str(event, field)?.to_string();
        ensure_tool_input_limit(&completed)?;
        let assembly = self.tool_inputs.get_mut(item_id).ok_or_else(|| {
            protocol_error("OpenAI tool-input completion preceded its output item")
        })?;
        if assembly.custom != custom || (!assembly.input.is_empty() && assembly.input != completed)
        {
            return Err(protocol_error(
                "OpenAI tool-input completion disagreed with its streamed deltas",
            ));
        }
        assembly.input = completed;
        Ok(Vec::new())
    }

    fn start_tool_input(
        &mut self,
        item_id: &str,
        call_id: &str,
        name: &str,
        initial_input: &str,
        custom: bool,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        if call_id.is_empty() || name.is_empty() || self.tool_inputs.contains_key(item_id) {
            return Err(protocol_error(
                "OpenAI tool output item omitted or reused its identity",
            ));
        }
        ensure_tool_input_limit(initial_input)?;
        self.tool_inputs.insert(
            item_id.to_string(),
            ToolInputAssembly {
                call_id: call_id.to_string(),
                name: name.to_string(),
                input: initial_input.to_string(),
                custom,
            },
        );
        if !custom {
            events.push(LanguageStreamEvent::ToolInputStart {
                id: call_id.to_string(),
                name: name.to_string(),
                owner: ExecutionOwner::Local,
            });
        }
        Ok(())
    }

    fn emit_function_call(
        &mut self,
        item_id: &str,
        call: &super::wire::FunctionCallItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        let assembly = self
            .tool_inputs
            .get(item_id)
            .ok_or_else(|| protocol_error("OpenAI function call completed before it was added"))?;
        if assembly.call_id != call.call_id || assembly.name != call.name {
            return Err(protocol_error("OpenAI function call changed its identity"));
        }
        ensure_tool_input_limit(&call.arguments)?;
        if !assembly.input.is_empty() && assembly.input != call.arguments {
            return Err(protocol_error(
                "OpenAI function call arguments disagreed with streamed deltas",
            ));
        }
        let arguments = serde_json::from_str(&call.arguments).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI function call ended with incomplete JSON arguments",
            )
            .with_source(source)
        })?;
        let call = ToolCall::local(call.call_id.clone(), call.name.clone(), arguments).map_err(
            |source| {
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI function call violated the canonical tool contract",
                )
                .with_source(source)
            },
        )?;
        events.push(LanguageStreamEvent::ToolCall(call));
        Ok(())
    }

    fn validate_custom_tool_call(
        &self,
        item_id: &str,
        call: &super::wire::CustomToolCallItemWire,
    ) -> Result<(), Error> {
        let assembly = self.tool_inputs.get(item_id).ok_or_else(|| {
            protocol_error("OpenAI custom tool call completed before it was added")
        })?;
        if assembly.call_id != call.call_id
            || assembly.name != call.name
            || (!assembly.input.is_empty() && assembly.input != call.input)
        {
            return Err(protocol_error(
                "OpenAI custom tool call disagreed with its streamed state",
            ));
        }
        Ok(())
    }

    fn finish_message(
        &mut self,
        message: &super::wire::MessageItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        for (content_index, part) in message.content.iter().enumerate() {
            match part {
                OutputContentPart::Text(text) => {
                    let id = content_id(&message.id, "text", content_index as u64);
                    if self.text_started.insert(id.clone()) {
                        events.push(LanguageStreamEvent::TextStart { id: id.clone() });
                        events.push(LanguageStreamEvent::TextDelta {
                            id: id.clone(),
                            delta: text.text.clone(),
                        });
                    }
                    if self.text_ended.insert(id.clone()) {
                        events.push(LanguageStreamEvent::TextEnd { id });
                    }
                    for (annotation_index, annotation) in text.annotations.iter().enumerate() {
                        let key = format!("{}:{annotation_index}", message.id);
                        if self.emitted_citations.insert(key) {
                            events.push(LanguageStreamEvent::Citation(project_citation(
                                &message.id,
                                annotation_index,
                                annotation,
                            )));
                        }
                    }
                }
                OutputContentPart::Refusal(refusal) => {
                    let id = content_id(&message.id, "refusal", content_index as u64);
                    if !self.emitted_refusals.insert(id) {
                        continue;
                    }
                    events.push(LanguageStreamEvent::Refusal {
                        reason: Some(refusal.refusal.clone()),
                    });
                }
                OutputContentPart::Unknown(_) => {}
            }
        }
    }

    fn terminal_event(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let native = decode_event_response(event)?.ok_or_else(|| {
            protocol_error("OpenAI Responses terminal event omitted its response")
        })?;
        let expected = match event.kind.as_str() {
            "response.completed" => "completed",
            "response.incomplete" => "incomplete",
            "response.cancelled" => "cancelled",
            "response.failed" => "failed",
            _ => unreachable!("terminal dispatch only passes terminal events"),
        };
        if native.status.as_str() != expected {
            return Err(protocol_error(
                "OpenAI Responses terminal event disagreed with response status",
            ));
        }
        self.validate_terminal_items(&native)?;
        let mut events = self.observe_identity(&native)?;
        let decoded = decode_response_wire(native.clone(), &self.scope, &self.requested_model)?;
        if let Some(usage) = &native.usage {
            events.push(LanguageStreamEvent::Usage(decode_usage(usage)));
        }
        let (_, canonical) = decoded.into_parts();
        let terminal = match &native.status {
            ResponseStatus::Completed | ResponseStatus::Incomplete => StreamTerminal::Completed {
                response: Box::new(canonical),
            },
            ResponseStatus::Cancelled => StreamTerminal::Cancelled {
                reason: "OpenAI cancelled the Responses generation".to_string(),
                response: Some(Box::new(canonical)),
            },
            ResponseStatus::Failed => StreamTerminal::Failed {
                error: failed_response_error(&native, &self.scope, &self.requested_model),
                response: Some(Box::new(canonical)),
            },
            ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => {
                return Err(protocol_error(
                    "OpenAI Responses terminal event carried a non-terminal status",
                ));
            }
        };
        self.terminal_native = Some(native);
        events.push(LanguageStreamEvent::Terminal(terminal));
        Ok(events)
    }

    fn error_event(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let wire = if let Some(error) = event.field("error") {
            serde_json::from_value::<ResponseErrorWire>(error.clone())
        } else {
            serde_json::from_value::<ResponseErrorWire>(encode_event_value(event)?)
        }
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses error event was malformed",
            )
            .with_source(source)
        })?;
        let error = Error::new(
            ErrorKind::Provider,
            "OpenAI emitted an error after establishing the Responses stream",
        )
        .with_source(NativeStreamFailure(wire));
        Ok(vec![LanguageStreamEvent::Terminal(
            StreamTerminal::Failed {
                error,
                response: None,
            },
        )])
    }

    fn opaque_stream_event(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let data = event.to_value().map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "failed to preserve an OpenAI Responses stream event",
            )
            .with_source(source)
        })?;
        let model = self
            .response_model
            .clone()
            .unwrap_or_else(|| self.requested_model.clone());
        let provenance =
            siumai_core::ProviderProvenance::from_scope(&self.scope, model).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "OpenAI Responses replay requires an explicit provider replay domain",
                )
                .with_source(source)
            })?;
        let item = siumai_core::OpaqueProviderItem::new(
            provenance,
            format!("{OPENAI_RESPONSES_OPAQUE_KIND}.stream_event"),
            data,
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI Responses stream event exceeded the opaque replay limit",
            )
            .with_source(source)
        })?;
        Ok(vec![LanguageStreamEvent::ProviderOpaque(item)])
    }

    fn ensure_known_item(&self, item_id: &str) -> Result<(), Error> {
        if self
            .item_indices
            .get(item_id)
            .is_some_and(|index| !self.completed_items.contains(index))
        {
            Ok(())
        } else {
            Err(protocol_error(
                "OpenAI Responses delta referenced an unknown or completed output item",
            ))
        }
    }

    fn validate_terminal_items(&self, response: &ResponseWire) -> Result<(), Error> {
        for (index, streamed) in &self.items {
            let index = usize::try_from(*index).map_err(|_| {
                protocol_error("OpenAI Responses output index exceeded the platform limit")
            })?;
            let terminal = response.output.get(index).ok_or_else(|| {
                protocol_error("OpenAI terminal response omitted a streamed output item")
            })?;
            if streamed.id() != terminal.id() || streamed.kind() != terminal.kind() {
                return Err(protocol_error(
                    "OpenAI terminal response changed a streamed output item identity",
                ));
            }
        }
        if matches!(
            &response.status,
            ResponseStatus::Completed | ResponseStatus::Incomplete
        ) && (self.completed_items.len() != self.items.len()
            || response.output.len() != self.items.len())
        {
            return Err(protocol_error(
                "OpenAI terminal response arrived before all output items completed",
            ));
        }
        Ok(())
    }
}

fn ensure_tool_input_limit(input: &str) -> Result<(), Error> {
    if input.len() > DEFAULT_TOOL_INPUT_BYTE_LIMIT {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "OpenAI Responses streamed tool input exceeded the byte limit",
        ));
    }
    Ok(())
}

fn append_tool_input(buffer: &mut String, delta: &str) -> Result<(), Error> {
    if buffer
        .len()
        .checked_add(delta.len())
        .is_none_or(|total| total > DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "OpenAI Responses streamed tool input exceeded the byte limit",
        ));
    }
    buffer.push_str(delta);
    Ok(())
}

impl LanguageStreamDecoder for ResponsesStreamDecoder {
    type ProtocolFrame = str;

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        let events = self.decode_frame(frame)?;
        self.lifecycle.record(&events).map_err(Error::from)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish().map_err(Error::from)? {
            Ok(Vec::new())
        } else {
            Err(Error::unexpected_eof())
        }
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

fn required_str<'a>(event: &'a StreamEventWire, field: &str) -> Result<&'a str, Error> {
    event
        .field(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| protocol_error("OpenAI Responses stream event omitted a string field"))
}

fn required_u64(event: &StreamEventWire, field: &str) -> Result<u64, Error> {
    event
        .field(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| protocol_error("OpenAI Responses stream event omitted an integer field"))
}

fn optional_u64(event: &StreamEventWire, field: &str) -> Option<u64> {
    event.field(field).and_then(Value::as_u64)
}

fn decode_event_response(event: &StreamEventWire) -> Result<Option<ResponseWire>, Error> {
    event.response().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses stream event contained a malformed response resource",
        )
        .with_source(source)
    })
}

fn decode_event_item(event: &StreamEventWire) -> Result<Option<OutputItem>, Error> {
    event.item().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI Responses stream event contained a malformed output item",
        )
        .with_source(source)
    })
}

fn encode_event_value(event: &StreamEventWire) -> Result<Value, Error> {
    event.to_value().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "failed to preserve an OpenAI Responses stream event",
        )
        .with_source(source)
    })
}

fn content_id(item_id: &str, lane: &str, index: u64) -> String {
    format!("{item_id}:{lane}:{index}")
}

struct NativeStreamFailure(ResponseErrorWire);

impl std::fmt::Debug for NativeStreamFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeStreamFailure")
            .field("code", &self.0.code)
            .field("kind", &self.0.kind)
            .field("message_present", &!self.0.message.is_empty())
            .field("param_present", &self.0.param.is_some())
            .finish()
    }
}

impl std::fmt::Display for NativeStreamFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("native OpenAI Responses stream failure")
    }
}

impl std::error::Error for NativeStreamFailure {}
