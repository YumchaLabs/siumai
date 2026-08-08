use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error, ErrorKind, ExecutionOwner,
    LanguageResponse, LanguageResponseStatus, LanguageStreamDecoder, LanguageStreamEvent, ModelId,
    OpaqueProviderItem, ProviderScope, PublicDiagnosticText, ResponseDiagnostics, StreamTerminal,
    ToolCall, Usage,
};

use super::language::{
    InteractionStatus, InteractionUsageWire, InteractionWire, decode_usage, project_interaction,
    project_step, step_type, valid_bounded_text,
};

const MAX_STREAM_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_STREAM_SIGNATURE_BYTES: usize = 1024 * 1024;
const MAX_STREAM_STEPS: usize = 256;
const MAX_STREAM_ID_BYTES: usize = 4 * 1024;
const MAX_STREAM_CONTENT_PARTS: usize = 256;
const MAX_STREAM_STEP_BYTES: usize = 64 * 1024 * 1024;

/// Stable v1 Gemini Interactions SSE decoder.
pub struct InteractionsStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    lifecycle: DecoderLifecycle,
    diagnostics: ResponseDiagnostics,
    interaction_id: Option<String>,
    response_model: Option<ModelId>,
    started: bool,
    open_steps: BTreeMap<usize, OpenStep>,
    finished_steps: BTreeMap<usize, Value>,
    usage: Option<InteractionUsageWire>,
    step_usage: Option<Usage>,
}

impl std::fmt::Debug for InteractionsStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InteractionsStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("lifecycle", &self.lifecycle)
            .field("interaction_id", &self.interaction_id)
            .field("response_model", &self.response_model)
            .field("started", &self.started)
            .field("open_step_count", &self.open_steps.len())
            .field("finished_step_count", &self.finished_steps.len())
            .finish()
    }
}

impl InteractionsStreamDecoder {
    pub fn new(scope: ProviderScope, requested_model: ModelId) -> Self {
        Self {
            scope,
            requested_model,
            lifecycle: DecoderLifecycle::default(),
            diagnostics: ResponseDiagnostics::default(),
            interaction_id: None,
            response_model: None,
            started: false,
            open_steps: BTreeMap::new(),
            finished_steps: BTreeMap::new(),
            usage: None,
            step_usage: None,
        }
    }

    fn decode_event(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        let envelope = serde_json::from_str::<EventEnvelope>(frame).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed Gemini Interactions SSE JSON",
            )
            .with_source(source)
        })?;
        let EventEnvelope { event_type, value } = envelope;
        let value = Value::Object(value);
        match event_type.as_str() {
            "interaction.created" => self.created(value),
            "interaction.status_update" => self.status_update(value),
            "step.start" => self.step_start(value),
            "step.delta" => self.step_delta(value),
            "step.stop" => self.step_stop(value),
            "interaction.completed" => self.completed(value),
            "error" => Err(self.in_band_error(value)),
            _ => Err(Error::protocol_violation(
                "Gemini Interactions stream emitted an unknown event type",
            )),
        }
    }

    fn created(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let event =
            serde_json::from_value::<InteractionLifecycleEvent>(value).map_err(|source| {
                Error::protocol_violation("Gemini interaction.created event is malformed")
                    .with_source(source)
            })?;
        if self.started {
            return Err(Error::protocol_violation(
                "Gemini Interactions stream emitted interaction.created more than once",
            ));
        }
        if event.interaction.status != InteractionStatus::InProgress {
            return Err(Error::protocol_violation(
                "Gemini interaction.created event carried a non-starting status",
            ));
        }
        let id = checked_id(event.interaction.id)?;
        let model = checked_model(event.interaction.model, &self.requested_model)?;
        self.interaction_id = Some(id.clone());
        self.response_model = Some(model.clone());
        self.started = true;
        Ok(vec![LanguageStreamEvent::Started {
            id: Some(id),
            model: Some(model),
        }])
    }

    fn status_update(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let event = serde_json::from_value::<StatusUpdateEvent>(value).map_err(|source| {
            Error::protocol_violation("Gemini interaction status event is malformed")
                .with_source(source)
        })?;
        let id = checked_id(Some(event.interaction_id))?;
        self.ensure_interaction_id(&id)?;
        let _status = event.status;
        Ok(Vec::new())
    }

    fn step_start(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let event = serde_json::from_value::<StepStartEvent>(value).map_err(|source| {
            Error::protocol_violation("Gemini step.start event is malformed").with_source(source)
        })?;
        self.ensure_step_index(event.index)?;
        if self.open_steps.contains_key(&event.index)
            || self.finished_steps.contains_key(&event.index)
        {
            return Err(Error::protocol_violation(
                "Gemini Interactions stream reused a step index",
            ));
        }
        let kind = step_type(&event.step)?;
        let block_id = format!(
            "{}:{}",
            self.interaction_id.as_deref().unwrap_or("interaction"),
            event.index
        );
        let (state, events) = match kind {
            "model_output" => OpenStep::model_output(event.step, block_id)?,
            "thought" => OpenStep::thought(event.step, block_id)?,
            "function_call" => OpenStep::function_call(event.step)?,
            _ => {
                let accumulated_bytes = encoded_value_len(&event.step)?;
                if accumulated_bytes > MAX_STREAM_STEP_BYTES {
                    return Err(Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini provider-native step exceeded the stream limit",
                    ));
                }
                (
                    OpenStep::Opaque {
                        raw: event.step,
                        accumulated_bytes,
                    },
                    Vec::new(),
                )
            }
        };
        self.open_steps.insert(event.index, state);
        Ok(events)
    }

    fn step_delta(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let event = serde_json::from_value::<StepDeltaEvent>(value).map_err(|source| {
            Error::protocol_violation("Gemini step.delta event is malformed").with_source(source)
        })?;
        let state = self.open_steps.get_mut(&event.index).ok_or_else(|| {
            Error::protocol_violation("Gemini step.delta referenced an unopened step")
        })?;
        let mut events = state.apply_delta(event.delta)?;
        if let Some(usage) = event.metadata.and_then(|metadata| metadata.total_usage) {
            self.usage = Some(usage.clone());
            events.push(LanguageStreamEvent::Usage(decode_usage(Some(usage))?));
        }
        Ok(events)
    }

    fn step_stop(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let event = serde_json::from_value::<StepStopEvent>(value).map_err(|source| {
            Error::protocol_violation("Gemini step.stop event is malformed").with_source(source)
        })?;
        let state = self.open_steps.remove(&event.index).ok_or_else(|| {
            Error::protocol_violation("Gemini step.stop referenced an unopened step")
        })?;
        let (raw, mut events) = state.finish()?;
        let model = self
            .response_model
            .as_ref()
            .unwrap_or(&self.requested_model)
            .clone();
        let mut projected = Vec::new();
        let mut warnings = Vec::new();
        project_step(&raw, &self.scope, &model, &mut projected, &mut warnings)?;
        for part in &projected {
            match part {
                ContentPart::ToolCall(call) => {
                    if !events
                        .iter()
                        .any(|event| matches!(event, LanguageStreamEvent::ToolCall(existing) if existing == call))
                    {
                        events.push(LanguageStreamEvent::ToolCall(call.clone()));
                    }
                }
                ContentPart::ProviderOpaque(item) => {
                    events.push(LanguageStreamEvent::ProviderOpaque(item.clone()));
                }
                _ => {}
            }
        }
        self.finished_steps.insert(event.index, raw);
        if let Some(usage) = event.usage {
            self.usage = Some(usage.clone());
            events.push(LanguageStreamEvent::Usage(decode_usage(Some(usage))?));
        } else if let Some(step_usage) = event.step_usage {
            let step_usage = decode_usage(Some(step_usage))?;
            let aggregate = self
                .step_usage
                .as_ref()
                .map(|existing| existing.checked_add(&step_usage))
                .unwrap_or(step_usage);
            self.step_usage = Some(aggregate.clone());
            events.push(LanguageStreamEvent::Usage(aggregate));
        }
        Ok(events)
    }

    fn completed(&mut self, value: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        if !self.open_steps.is_empty() {
            return Err(Error::protocol_violation(
                "Gemini interaction completed before all open steps stopped",
            ));
        }
        let event =
            serde_json::from_value::<InteractionLifecycleEvent>(value).map_err(|source| {
                Error::protocol_violation("Gemini interaction.completed event is malformed")
                    .with_source(source)
            })?;
        let id = checked_id(event.interaction.id)?;
        self.ensure_interaction_id(&id)?;
        let model = checked_model(event.interaction.model, &self.requested_model)?;
        if let Some(existing) = &self.response_model
            && existing != &model
        {
            return Err(Error::protocol_violation(
                "Gemini Interactions stream changed its response model",
            ));
        }
        self.response_model = Some(model.clone());

        let mut steps = event.interaction.steps;
        let reconstructed = self.finished_steps.values().cloned().collect::<Vec<_>>();
        if steps.is_empty() {
            steps = reconstructed;
        } else if !reconstructed.is_empty() {
            let streamed = project_interaction(
                InteractionWire {
                    id: Some(id.clone()),
                    status: event.interaction.status,
                    model: Some(model.as_str().to_string()),
                    steps: reconstructed,
                    usage: event
                        .interaction
                        .usage
                        .clone()
                        .or_else(|| self.usage.clone()),
                    service_tier: event.interaction.service_tier.clone(),
                    created: event.interaction.created.clone(),
                    updated: event.interaction.updated.clone(),
                },
                &self.scope,
                &self.requested_model,
            )?;
            let terminal = project_interaction(
                InteractionWire {
                    id: Some(id.clone()),
                    status: event.interaction.status,
                    model: Some(model.as_str().to_string()),
                    steps: steps.clone(),
                    usage: event
                        .interaction
                        .usage
                        .clone()
                        .or_else(|| self.usage.clone()),
                    service_tier: event.interaction.service_tier.clone(),
                    created: event.interaction.created.clone(),
                    updated: event.interaction.updated.clone(),
                },
                &self.scope,
                &self.requested_model,
            )?;
            if portable_content(&streamed) != portable_content(&terminal)
                || replay_content(&streamed) != replay_content(&terminal)
            {
                return Err(Error::protocol_violation(
                    "Gemini streamed steps disagree with the terminal interaction",
                ));
            }
        }

        let terminal_usage = event.interaction.usage.or_else(|| self.usage.clone());
        let mut response = project_interaction(
            InteractionWire {
                id: Some(id.clone()),
                status: event.interaction.status,
                model: Some(model.as_str().to_string()),
                steps,
                usage: terminal_usage.clone(),
                service_tier: event.interaction.service_tier,
                created: event.interaction.created,
                updated: event.interaction.updated,
            },
            &self.scope,
            &self.requested_model,
        )?;
        if terminal_usage.is_none()
            && let Some(step_usage) = self.step_usage.clone()
        {
            response = replace_usage(response, step_usage)?;
        }
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(LanguageStreamEvent::Started {
                id: Some(id),
                model: Some(model),
            });
        }
        events.push(LanguageStreamEvent::Terminal(terminal_event(response)));
        Ok(events)
    }

    fn in_band_error(&self, value: Value) -> Error {
        let event = serde_json::from_value::<ErrorEvent>(value).ok();
        let code = event
            .and_then(|event| event.error)
            .and_then(|error| error.code)
            .filter(|code| valid_bounded_text(code, 256));
        let kind = match code.as_deref() {
            Some("rate_limit_exceeded" | "resource_exhausted") => ErrorKind::RateLimited,
            Some("quota_exceeded") => ErrorKind::QuotaExceeded,
            Some("invalid_argument" | "not_found") => ErrorKind::InvalidInput,
            Some("unauthenticated") => ErrorKind::Authentication,
            Some("permission_denied") => ErrorKind::Authorization,
            Some("context_length_exceeded") => ErrorKind::ContextWindowExceeded,
            Some("unavailable") => ErrorKind::Unavailable,
            _ => ErrorKind::Provider,
        };
        let mut diagnostics = self.diagnostics.clone();
        if let Some(code) = code.and_then(|code| PublicDiagnosticText::new(code).ok()) {
            diagnostics = diagnostics.with_provider_code(code);
        }
        let message = match kind {
            ErrorKind::RateLimited => "Gemini rate limited the streaming request",
            ErrorKind::QuotaExceeded => "Gemini quota was exceeded",
            ErrorKind::InvalidInput => "Gemini rejected the streaming request",
            ErrorKind::Authentication => "Gemini authentication failed during streaming",
            ErrorKind::Authorization => "Gemini authorization failed during streaming",
            ErrorKind::ContextWindowExceeded => "Gemini context window was exceeded",
            ErrorKind::Unavailable => "Gemini was unavailable during streaming",
            _ => "Gemini streaming failed",
        };
        Error::new(kind, message).with_diagnostics(diagnostics)
    }

    fn ensure_interaction_id(&mut self, id: &str) -> Result<(), Error> {
        match &self.interaction_id {
            Some(existing) if existing != id => Err(Error::protocol_violation(
                "Gemini Interactions stream changed its interaction ID",
            )),
            Some(_) => Ok(()),
            None => {
                self.interaction_id = Some(id.to_string());
                Ok(())
            }
        }
    }

    fn ensure_step_index(&self, index: usize) -> Result<(), Error> {
        if index >= MAX_STREAM_STEPS {
            Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Interactions stream exceeded the step-index limit",
            ))
        } else {
            Ok(())
        }
    }
}

impl LanguageStreamDecoder for InteractionsStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.diagnostics = diagnostics;
    }

    fn decode(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle.ensure_decode_allowed()?;
        let events = self.decode_event(frame)?;
        self.lifecycle.record(&events)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish()? {
            Ok(Vec::new())
        } else {
            Err(Error::unexpected_eof())
        }
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

enum OpenStep {
    ModelOutput {
        raw: Value,
        id: String,
        content: Vec<Value>,
        text_bytes: usize,
        accumulated_bytes: usize,
        text_started: bool,
    },
    Thought {
        raw: Value,
        id: String,
        text: String,
        signature: String,
        reasoning_started: bool,
    },
    FunctionCall {
        raw: Value,
        id: String,
        name: String,
        initial_arguments: Value,
        argument_text: String,
        delta_seen: bool,
    },
    Opaque {
        raw: Value,
        accumulated_bytes: usize,
    },
}

impl OpenStep {
    fn model_output(raw: Value, id: String) -> Result<(Self, Vec<LanguageStreamEvent>), Error> {
        let content = match raw.get("content") {
            Some(Value::Array(blocks)) if blocks.len() <= MAX_STREAM_CONTENT_PARTS => {
                blocks.clone()
            }
            Some(Value::Array(_)) => {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini model-output start exceeded the content-part limit",
                ));
            }
            Some(_) => {
                return Err(Error::protocol_violation(
                    "Gemini model-output start content was not an array",
                ));
            }
            None => Vec::new(),
        };
        let accumulated_bytes = encoded_value_len(&raw)?;
        ensure_step_size(&raw)?;
        let mut text_bytes = 0usize;
        let mut events = Vec::new();
        let mut started = false;
        for block in &content {
            if block.get("type").and_then(Value::as_str) == Some("text")
                && let Some(initial) = block.get("text").and_then(Value::as_str)
            {
                text_bytes = text_bytes.checked_add(initial.len()).ok_or_else(|| {
                    Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini model output exceeded the stream text limit",
                    )
                })?;
                if text_bytes > MAX_STREAM_TEXT_BYTES {
                    return Err(Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini model output exceeded the stream text limit",
                    ));
                }
                if !started {
                    started = true;
                    events.push(LanguageStreamEvent::TextStart { id: id.clone() });
                }
                if !initial.is_empty() {
                    events.push(LanguageStreamEvent::TextDelta {
                        id: id.clone(),
                        delta: initial.to_string(),
                    });
                }
            }
        }
        Ok((
            Self::ModelOutput {
                raw,
                id,
                content,
                text_bytes,
                accumulated_bytes,
                text_started: started,
            },
            events,
        ))
    }

    fn thought(raw: Value, id: String) -> Result<(Self, Vec<LanguageStreamEvent>), Error> {
        ensure_step_size(&raw)?;
        let text = super::language::thought_text(&raw)?;
        let signature = raw
            .get("signature")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if signature.len() > MAX_STREAM_SIGNATURE_BYTES {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini thought signature exceeded the stream limit",
            ));
        }
        let mut events = vec![LanguageStreamEvent::ReasoningStart { id: id.clone() }];
        if !text.is_empty() {
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: id.clone(),
                delta: text.clone(),
            });
        }
        Ok((
            Self::Thought {
                raw,
                id,
                text,
                signature,
                reasoning_started: true,
            },
            events,
        ))
    }

    fn function_call(raw: Value) -> Result<(Self, Vec<LanguageStreamEvent>), Error> {
        ensure_step_size(&raw)?;
        let fields = super::language::function_call_fields(&raw)?;
        if !fields.arguments.is_object() {
            return Err(Error::protocol_violation(
                "Gemini function-call start arguments must be a JSON object",
            ));
        }
        let id = fields.id.to_string();
        let name = fields.name.to_string();
        let initial_arguments = fields.arguments.clone();
        Ok((
            Self::FunctionCall {
                raw,
                id: id.clone(),
                name: name.clone(),
                initial_arguments,
                argument_text: String::new(),
                delta_seen: false,
            },
            vec![LanguageStreamEvent::ToolInputStart {
                id,
                name,
                owner: ExecutionOwner::Local,
            }],
        ))
    }

    fn apply_delta(&mut self, delta: Value) -> Result<Vec<LanguageStreamEvent>, Error> {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::protocol_violation("Gemini step delta omitted its type"))?
            .to_string();
        match (self, kind.as_str()) {
            (
                Self::ModelOutput {
                    id,
                    content,
                    text_bytes,
                    accumulated_bytes,
                    text_started,
                    ..
                },
                "text",
            ) => {
                let text_delta = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::protocol_violation("Gemini text delta omitted its text"))?
                    .to_string();
                *text_bytes = text_bytes.checked_add(text_delta.len()).ok_or_else(|| {
                    Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini model output exceeded the stream text limit",
                    )
                })?;
                if *text_bytes > MAX_STREAM_TEXT_BYTES {
                    return Err(Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini model output exceeded the stream text limit",
                    ));
                }
                record_delta_bytes(accumulated_bytes, &delta)?;
                merge_model_output_delta(content, delta)?;
                let mut events = Vec::new();
                if !*text_started {
                    *text_started = true;
                    events.push(LanguageStreamEvent::TextStart { id: id.clone() });
                }
                events.push(LanguageStreamEvent::TextDelta {
                    id: id.clone(),
                    delta: text_delta,
                });
                Ok(events)
            }
            (
                Self::ModelOutput {
                    content,
                    accumulated_bytes,
                    ..
                },
                _,
            ) => {
                record_delta_bytes(accumulated_bytes, &delta)?;
                merge_model_output_delta(content, delta)?;
                Ok(Vec::new())
            }
            (
                Self::Thought {
                    id,
                    text,
                    reasoning_started,
                    ..
                },
                "thought_summary",
            ) => {
                let content = delta.get("content").ok_or_else(|| {
                    Error::protocol_violation("Gemini thought summary delta omitted content")
                })?;
                if content.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(Error::protocol_violation(
                        "Gemini thought summary delta was not text",
                    ));
                }
                let delta = content.get("text").and_then(Value::as_str).ok_or_else(|| {
                    Error::protocol_violation("Gemini thought summary delta omitted text")
                })?;
                push_bounded(text, delta, MAX_STREAM_TEXT_BYTES, "thought summary")?;
                let mut events = Vec::new();
                if !*reasoning_started {
                    *reasoning_started = true;
                    events.push(LanguageStreamEvent::ReasoningStart { id: id.clone() });
                }
                events.push(LanguageStreamEvent::ReasoningDelta {
                    id: id.clone(),
                    delta: delta.to_string(),
                });
                Ok(events)
            }
            (Self::Thought { signature, .. }, "thought_signature") => {
                let delta = delta
                    .get("signature")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::protocol_violation(
                            "Gemini thought signature delta omitted its value",
                        )
                    })?;
                push_bounded(
                    signature,
                    delta,
                    MAX_STREAM_SIGNATURE_BYTES,
                    "thought signature",
                )?;
                Ok(Vec::new())
            }
            (
                Self::FunctionCall {
                    id,
                    argument_text,
                    delta_seen,
                    ..
                },
                "arguments_delta",
            ) => {
                let delta = delta
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::protocol_violation("Gemini arguments delta omitted its text")
                    })?;
                push_bounded(
                    argument_text,
                    delta,
                    DEFAULT_TOOL_INPUT_BYTE_LIMIT,
                    "function arguments",
                )?;
                *delta_seen = true;
                Ok(vec![LanguageStreamEvent::ToolInputDelta {
                    id: id.clone(),
                    delta: delta.to_string(),
                }])
            }
            (
                Self::Opaque {
                    raw,
                    accumulated_bytes,
                },
                _,
            ) => {
                *accumulated_bytes = accumulated_bytes
                    .checked_add(encoded_value_len(&delta)?)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::ResponseLimit,
                            "Gemini provider-native step exceeded the stream limit",
                        )
                    })?;
                if *accumulated_bytes > MAX_STREAM_STEP_BYTES {
                    return Err(Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini provider-native step exceeded the stream limit",
                    ));
                }
                merge_delta_value(raw, delta, 0)?;
                Ok(Vec::new())
            }
            _ => Err(Error::protocol_violation(
                "Gemini step delta does not match its open step",
            )),
        }
    }

    fn finish(self) -> Result<(Value, Vec<LanguageStreamEvent>), Error> {
        match self {
            Self::ModelOutput {
                mut raw,
                id,
                content,
                text_started,
                ..
            } => {
                raw["content"] = Value::Array(content);
                ensure_step_size(&raw)?;
                let mut events = Vec::new();
                if text_started {
                    events.push(LanguageStreamEvent::TextEnd { id });
                }
                Ok((raw, events))
            }
            Self::Thought {
                mut raw,
                id,
                text,
                signature,
                reasoning_started,
            } => {
                if !text.is_empty() {
                    raw["summary"] = json!([{"type": "text", "text": text}]);
                }
                if !signature.is_empty() {
                    raw["signature"] = Value::String(signature);
                }
                let events = reasoning_started
                    .then(|| LanguageStreamEvent::ReasoningEnd { id })
                    .into_iter()
                    .collect();
                Ok((raw, events))
            }
            Self::FunctionCall {
                mut raw,
                id,
                name,
                initial_arguments,
                argument_text,
                delta_seen,
            } => {
                let arguments = if delta_seen {
                    let parsed =
                        serde_json::from_str::<Value>(&argument_text).map_err(|source| {
                            Error::protocol_violation(
                                "Gemini streamed function arguments are not valid JSON",
                            )
                            .with_source(source)
                        })?;
                    if !parsed.is_object() {
                        return Err(Error::protocol_violation(
                            "Gemini streamed function arguments must be a JSON object",
                        ));
                    }
                    if initial_arguments
                        .as_object()
                        .is_some_and(|object| !object.is_empty())
                        && initial_arguments != parsed
                    {
                        return Err(Error::protocol_violation(
                            "Gemini function-call start and delta arguments disagree",
                        ));
                    }
                    parsed
                } else {
                    initial_arguments
                };
                raw["arguments"] = arguments.clone();
                let call = ToolCall::local(id, name, arguments).map_err(|source| {
                    Error::protocol_violation(
                        "Gemini streamed an invalid caller-executed function call",
                    )
                    .with_source(source)
                })?;
                Ok((raw, vec![LanguageStreamEvent::ToolCall(call)]))
            }
            Self::Opaque { raw, .. } => Ok((raw, Vec::new())),
        }
    }
}

fn terminal_event(response: LanguageResponse) -> StreamTerminal {
    match response.status() {
        LanguageResponseStatus::Completed | LanguageResponseStatus::Incomplete { .. } => {
            StreamTerminal::Completed {
                response: Box::new(response),
            }
        }
        LanguageResponseStatus::Failed => StreamTerminal::Failed {
            error: Error::new(ErrorKind::Provider, "Gemini interaction failed"),
            response: Some(Box::new(response)),
        },
        LanguageResponseStatus::Cancelled => StreamTerminal::Cancelled {
            reason: "Gemini interaction was cancelled".to_string(),
            response: Some(Box::new(response)),
        },
        _ => StreamTerminal::Failed {
            error: Error::protocol_violation("Gemini terminal response has an unsupported status"),
            response: None,
        },
    }
}

fn portable_content(response: &LanguageResponse) -> Vec<ContentPart> {
    response
        .content()
        .iter()
        .filter(|part| !matches!(part, ContentPart::ProviderOpaque(_)))
        .cloned()
        .collect()
}

fn replay_content(response: &LanguageResponse) -> Vec<&OpaqueProviderItem> {
    response
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .collect()
}

fn replace_usage(response: LanguageResponse, usage: Usage) -> Result<LanguageResponse, Error> {
    let mut rebuilt = LanguageResponse::new(
        response.status().clone(),
        response.content().to_vec(),
        response.finish_reason().clone(),
        usage,
    )
    .map_err(|source| {
        Error::protocol_violation("Gemini stream produced invalid aggregated usage")
            .with_source(source)
    })?
    .with_warnings(response.warnings().to_vec())
    .with_provider_metadata(response.provider_metadata().clone());
    if let Some(id) = response.id() {
        rebuilt = rebuilt.with_id(id.to_string());
    }
    if let Some(model) = response.model() {
        rebuilt = rebuilt.with_model(model.clone());
    }
    Ok(rebuilt)
}

fn checked_id(value: Option<String>) -> Result<String, Error> {
    let value = value.ok_or_else(|| {
        Error::protocol_violation("Gemini interaction lifecycle event omitted its ID")
    })?;
    if valid_bounded_text(&value, MAX_STREAM_ID_BYTES) {
        Ok(value)
    } else {
        Err(Error::protocol_violation(
            "Gemini interaction lifecycle event contained an invalid ID",
        ))
    }
}

fn checked_model(value: Option<String>, fallback: &ModelId) -> Result<ModelId, Error> {
    value
        .map(ModelId::new)
        .transpose()
        .map_err(|source| {
            Error::protocol_violation("Gemini stream returned an invalid model ID")
                .with_source(source)
        })
        .map(|model| model.unwrap_or_else(|| fallback.clone()))
}

fn push_bounded(
    target: &mut String,
    delta: &str,
    maximum: usize,
    _field: &'static str,
) -> Result<(), Error> {
    if target.len().saturating_add(delta.len()) > maximum {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini stream accumulation exceeded its protocol limit",
        ));
    }
    target.push_str(delta);
    Ok(())
}

fn encoded_value_len(value: &Value) -> Result<usize, Error> {
    serde_json::to_vec(value)
        .map(|encoded| encoded.len())
        .map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "Gemini stream value could not be measured",
            )
            .with_source(source)
        })
}

fn ensure_step_size(step: &Value) -> Result<(), Error> {
    if encoded_value_len(step)? > MAX_STREAM_STEP_BYTES {
        Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini streamed step exceeded the protocol limit",
        ))
    } else {
        Ok(())
    }
}

fn record_delta_bytes(total: &mut usize, delta: &Value) -> Result<(), Error> {
    *total = total
        .checked_add(encoded_value_len(delta)?)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::ResponseLimit,
                "Gemini streamed step exceeded the protocol limit",
            )
        })?;
    if *total > MAX_STREAM_STEP_BYTES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini streamed step exceeded the protocol limit",
        ));
    }
    Ok(())
}

fn merge_model_output_delta(content: &mut Vec<Value>, delta: Value) -> Result<(), Error> {
    let kind = delta
        .get("type")
        .and_then(Value::as_str)
        .filter(|value| valid_bounded_text(value, 128))
        .ok_or_else(|| Error::protocol_violation("Gemini model-output delta has an invalid type"))?
        .to_string();
    if let Some(last) = content.last_mut()
        && last.get("type").and_then(Value::as_str) == Some(kind.as_str())
    {
        return merge_delta_value(last, delta, 0);
    }
    if content.len() >= MAX_STREAM_CONTENT_PARTS {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini model output exceeded the content-part limit",
        ));
    }
    content.push(delta);
    Ok(())
}

fn merge_delta_value(target: &mut Value, delta: Value, depth: usize) -> Result<(), Error> {
    if depth > 16 {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini provider-native delta exceeded the nesting limit",
        ));
    }
    match (target, delta) {
        (Value::Object(target), Value::Object(delta)) => {
            for (key, value) in delta {
                if key == "type" {
                    if depth > 0 && target.get(&key).is_some_and(|existing| existing != &value) {
                        return Err(Error::protocol_violation(
                            "Gemini provider-native delta changed a nested content type",
                        ));
                    }
                    target.entry(key).or_insert(value);
                    continue;
                }
                match target.get_mut(&key) {
                    Some(existing)
                        if matches!(
                            key.as_str(),
                            "id" | "call_id" | "name" | "mime_type" | "uri"
                        ) && existing != &value =>
                    {
                        return Err(Error::protocol_violation(
                            "Gemini provider-native delta changed a stable identity field",
                        ));
                    }
                    Some(existing) => merge_delta_value(existing, value, depth + 1)?,
                    None => {
                        target.insert(key, value);
                    }
                }
            }
            Ok(())
        }
        (Value::Array(target), Value::Array(mut delta)) => {
            if target.len().saturating_add(delta.len()) > MAX_STREAM_CONTENT_PARTS {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini provider-native delta exceeded the array limit",
                ));
            }
            target.append(&mut delta);
            Ok(())
        }
        (Value::String(target), Value::String(delta)) => {
            if target != &delta {
                push_bounded(
                    target,
                    &delta,
                    MAX_STREAM_STEP_BYTES,
                    "provider-native text",
                )?;
            }
            Ok(())
        }
        (target, delta) if *target == delta => Ok(()),
        (target, delta) => {
            *target = delta;
            Ok(())
        }
    }
}

#[derive(Deserialize)]
struct EventEnvelope {
    event_type: String,
    #[serde(flatten)]
    value: Map<String, Value>,
}

#[derive(Deserialize)]
struct InteractionLifecycleEvent {
    interaction: PartialInteractionWire,
}

#[derive(Deserialize)]
struct PartialInteractionWire {
    #[serde(default)]
    id: Option<String>,
    status: InteractionStatus,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    steps: Vec<Value>,
    #[serde(default)]
    usage: Option<InteractionUsageWire>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    created: Option<String>,
    #[serde(default)]
    updated: Option<String>,
}

#[derive(Deserialize)]
struct StatusUpdateEvent {
    interaction_id: String,
    status: InteractionStatus,
}

#[derive(Deserialize)]
struct StepStartEvent {
    index: usize,
    step: Value,
}

#[derive(Deserialize)]
struct StepDeltaEvent {
    index: usize,
    delta: Value,
    #[serde(default)]
    metadata: Option<StepDeltaMetadata>,
}

#[derive(Deserialize)]
struct StepDeltaMetadata {
    #[serde(default)]
    total_usage: Option<InteractionUsageWire>,
}

#[derive(Deserialize)]
struct StepStopEvent {
    index: usize,
    #[serde(default)]
    step_usage: Option<InteractionUsageWire>,
    #[serde(default)]
    usage: Option<InteractionUsageWire>,
}

#[derive(Deserialize)]
struct ErrorEvent {
    #[serde(default)]
    error: Option<ProviderErrorWire>,
}

#[derive(Deserialize)]
struct ProviderErrorWire {
    #[serde(default)]
    code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        ApiModeId, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId,
    };

    fn decoder() -> InteractionsStreamDecoder {
        let scope = ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new("gemini-interactions").unwrap())
            .with_api_mode(ApiModeId::new("interactions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ));
        InteractionsStreamDecoder::new(scope, ModelId::new("gemini-3.6-flash").unwrap())
    }

    #[test]
    fn streamed_tool_arguments_settle_once_as_canonical_json() {
        let mut decoder = decoder();
        let frames = [
            json!({
                "event_type": "interaction.created",
                "interaction": {
                    "id": "interaction-1",
                    "status": "in_progress",
                    "model": "gemini-3.6-flash"
                }
            }),
            json!({
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "function_call",
                    "id": "call/1",
                    "name": "lookup",
                    "arguments": {}
                }
            }),
            json!({
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "arguments_delta",
                    "arguments": "{\"key\":\"x\"}"
                }
            }),
            json!({"event_type": "step.stop", "index": 0}),
            json!({
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "interaction-1",
                    "status": "requires_action",
                    "model": "gemini-3.6-flash",
                    "usage": {"total_input_tokens": 4, "total_output_tokens": 1}
                }
            }),
        ];
        let mut events = Vec::new();
        for frame in frames {
            events.extend(decoder.decode(&frame.to_string()).unwrap());
        }

        assert_eq!(
            events
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        assert!(events.iter().any(|event| {
            matches!(
                event,
                LanguageStreamEvent::ToolCall(call)
                    if call.arguments() == &json!({"key": "x"})
            )
        }));
        assert!(matches!(decoder.finish(), Ok(events) if events.is_empty()));
    }

    #[test]
    fn streamed_model_output_preserves_media_and_cumulative_usage() {
        let mut decoder = decoder();
        let frames = [
            json!({
                "event_type": "interaction.created",
                "interaction": {
                    "id": "interaction-media",
                    "status": "in_progress",
                    "model": "gemini-3.6-flash"
                }
            }),
            json!({
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {
                            "type": "image",
                            "uri": "https://example.invalid/image.png",
                            "mime_type": "image/png"
                        }
                    ]
                }
            }),
            json!({
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "world"},
                "metadata": {
                    "total_usage": {
                        "total_input_tokens": 3,
                        "total_output_tokens": 2,
                        "total_tokens": 5
                    }
                }
            }),
            json!({"event_type": "step.stop", "index": 0}),
            json!({
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "interaction-media",
                    "status": "completed",
                    "model": "gemini-3.6-flash"
                }
            }),
        ];
        let mut events = Vec::new();
        for frame in frames {
            events.extend(decoder.decode(&frame.to_string()).unwrap());
        }

        let terminal = events
            .iter()
            .find_map(LanguageStreamEvent::terminal)
            .unwrap();
        let StreamTerminal::Completed { response } = terminal else {
            panic!("expected completed Gemini stream");
        };
        assert!(matches!(
            &response.content()[0],
            ContentPart::Text { text } if text == "hello "
        ));
        assert!(matches!(
            &response.content()[1],
            ContentPart::Media(media)
                if matches!(&media.data, siumai_core::MediaData::Url(uri) if uri == "https://example.invalid/image.png")
        ));
        assert!(matches!(
            &response.content()[2],
            ContentPart::Text { text } if text == "world"
        ));
        assert_eq!(
            response.usage().total_tokens,
            siumai_core::UsageValue::Known(5)
        );
    }

    #[test]
    fn terminal_replay_state_must_match_streamed_state() {
        let mut decoder = decoder();
        for frame in [
            json!({
                "event_type": "interaction.created",
                "interaction": {
                    "id": "interaction-replay",
                    "status": "in_progress",
                    "model": "gemini-3.6-flash"
                }
            }),
            json!({
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "thought", "signature": "sig-", "summary": []}
            }),
            json!({
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "stream"}
            }),
            json!({"event_type": "step.stop", "index": 0}),
        ] {
            decoder.decode(&frame.to_string()).unwrap();
        }
        let error = decoder
            .decode(
                &json!({
                    "event_type": "interaction.completed",
                    "interaction": {
                        "id": "interaction-replay",
                        "status": "completed",
                        "model": "gemini-3.6-flash",
                        "steps": [{
                            "type": "thought",
                            "signature": "different",
                            "summary": []
                        }]
                    }
                })
                .to_string(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }

    #[test]
    fn eof_without_interaction_completed_is_typed_unexpected_eof() {
        let mut decoder = decoder();
        decoder
            .decode(
                &json!({
                    "event_type": "interaction.created",
                    "interaction": {
                        "id": "interaction-1",
                        "status": "in_progress",
                        "model": "gemini-3.6-flash"
                    }
                })
                .to_string(),
            )
            .unwrap();

        let error = decoder.finish().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    }

    #[test]
    fn in_band_error_keeps_typed_classification_and_safe_diagnostics() {
        let mut decoder = decoder();
        let error = decoder
            .decode(
                &json!({
                    "event_type": "error",
                    "error": {
                        "code": "resource_exhausted",
                        "message": "tenant secret should stay private"
                    }
                })
                .to_string(),
            )
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        assert_eq!(
            error
                .diagnostics()
                .and_then(ResponseDiagnostics::provider_code),
            Some("resource_exhausted")
        );
        assert!(!format!("{error:?}").contains("tenant secret"));
    }
}
