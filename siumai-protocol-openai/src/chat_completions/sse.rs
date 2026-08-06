//! Outbound OpenAI Chat Completions SSE encoding.
//!
//! This module projects canonical [`LanguageStreamEvent`] values onto the Chat Completions wire
//! format. It owns no transport, provider, endpoint, credential, or inbound-decoding behavior.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde_json::{Map, Value};
use siumai_core::{
    Error, ErrorKind, ExecutionOwner, FinishReason, LanguageResponse, LanguageStreamEvent,
    StreamTerminal, ToolCall, Usage,
};

const MAX_RESPONSE_ID_BYTES: usize = 512;
const MAX_TEXT_PART_ID_BYTES: usize = 512;
const MAX_TEXT_DELTA_BYTES: usize = 1024 * 1024;
const MAX_REFUSAL_BYTES: usize = 64 * 1024;
const MAX_TOOL_CALLS: usize = 128;
const MAX_TOOL_CALL_ID_BYTES: usize = 512;
const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_TOOL_ARGUMENT_BYTES: usize = 1024 * 1024;
const MAX_TOTAL_TOOL_ARGUMENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_FINISH_REASON_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum EncoderLifecycle {
    #[default]
    Fresh,
    Started,
    Terminal,
}

#[derive(Debug)]
struct ToolCallState {
    index: u32,
    name: String,
    arguments: String,
    completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UsageProjection {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    completion_audio_tokens: Option<u64>,
}

impl UsageProjection {
    fn from_usage(usage: &Usage) -> Self {
        Self {
            prompt_tokens: usage.input_tokens.value(),
            completion_tokens: usage.output_tokens.value(),
            total_tokens: usage.total_tokens.value(),
            cached_tokens: usage.cache_read_tokens.value(),
            reasoning_tokens: usage.reasoning_tokens.value(),
            completion_audio_tokens: usage.audio_output_tokens.value(),
        }
    }

    fn is_empty(self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
            && self.cached_tokens.is_none()
            && self.reasoning_tokens.is_none()
            && self.completion_audio_tokens.is_none()
    }

    fn into_value(self) -> Value {
        let mut usage = Map::new();
        insert_optional_u64(&mut usage, "prompt_tokens", self.prompt_tokens);
        insert_optional_u64(&mut usage, "completion_tokens", self.completion_tokens);
        insert_optional_u64(&mut usage, "total_tokens", self.total_tokens);

        if let Some(cached_tokens) = self.cached_tokens {
            usage.insert(
                "prompt_tokens_details".to_string(),
                Value::Object(Map::from_iter([(
                    "cached_tokens".to_string(),
                    Value::from(cached_tokens),
                )])),
            );
        }

        let mut completion_details = Map::new();
        insert_optional_u64(
            &mut completion_details,
            "reasoning_tokens",
            self.reasoning_tokens,
        );
        insert_optional_u64(
            &mut completion_details,
            "audio_tokens",
            self.completion_audio_tokens,
        );
        if !completion_details.is_empty() {
            usage.insert(
                "completion_tokens_details".to_string(),
                Value::Object(completion_details),
            );
        }

        Value::Object(usage)
    }
}

/// Stateful encoder for one outbound OpenAI Chat Completions SSE stream.
///
/// The encoder is intentionally mutable and non-cloneable: one value owns one stream lifecycle.
/// A fresh encoder must be constructed for each independent response. The encoder never infers a
/// successful terminal from EOF; callers must provide exactly one canonical terminal event.
#[derive(Debug, Default)]
pub struct ChatCompletionsSseEncoder {
    lifecycle: EncoderLifecycle,
    response_id: Option<String>,
    model: Option<String>,
    active_text_id: Option<String>,
    tool_calls: BTreeMap<String, ToolCallState>,
    total_tool_argument_bytes: usize,
    last_emitted_usage: Option<UsageProjection>,
}

impl ChatCompletionsSseEncoder {
    /// Construct an encoder for one logical response stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode one canonical stream event into zero or more complete SSE frames.
    ///
    /// Each returned [`Bytes`] value contains exactly one `data:` frame, including its trailing
    /// blank line. A terminal event emits one terminal frame followed by exactly one `[DONE]`
    /// frame. Any event after a terminal event is rejected.
    pub fn encode(&mut self, event: &LanguageStreamEvent) -> Result<Vec<Bytes>, Error> {
        if self.lifecycle == EncoderLifecycle::Terminal {
            return Err(protocol_violation(
                "Chat Completions encoder received an event after its terminal event",
            ));
        }

        match event {
            LanguageStreamEvent::Started { id, model } => {
                self.encode_started(id.as_deref(), model.as_ref().map(|model| model.as_str()))
            }
            _ if self.lifecycle == EncoderLifecycle::Fresh => Err(protocol_violation(
                "Chat Completions encoder requires Started as its first event",
            )),
            LanguageStreamEvent::TextStart { id } => self.encode_text_start(id),
            LanguageStreamEvent::TextDelta { id, delta } => self.encode_text_delta(id, delta),
            LanguageStreamEvent::TextEnd { id } => self.encode_text_end(id),
            LanguageStreamEvent::ToolInputStart { id, name, owner } => {
                self.encode_tool_input_start(id, name, owner)
            }
            LanguageStreamEvent::ToolInputDelta { id, delta } => {
                self.encode_tool_input_delta(id, delta)
            }
            LanguageStreamEvent::ToolCall(call) => self.encode_tool_call(call),
            LanguageStreamEvent::Refusal { reason } => self.encode_refusal(reason.as_deref()),
            LanguageStreamEvent::Usage(usage) => self.encode_usage(usage),
            LanguageStreamEvent::Terminal(terminal) => self.encode_terminal(terminal),
            LanguageStreamEvent::ReasoningStart { .. }
            | LanguageStreamEvent::ReasoningDelta { .. }
            | LanguageStreamEvent::ReasoningEnd { .. } => Err(unsupported_projection(
                "canonical reasoning events require a configured Chat Completions dialect",
            )),
            LanguageStreamEvent::ToolResult(_) => Err(unsupported_projection(
                "tool result events are not assistant Chat Completions output",
            )),
            LanguageStreamEvent::Citation(_) => Err(unsupported_projection(
                "canonical citations have no provider-neutral Chat Completions projection",
            )),
            LanguageStreamEvent::ProviderDeferred { .. }
            | LanguageStreamEvent::ProviderOpaque(_) => Err(unsupported_projection(
                "provider-owned stream state cannot be projected into Chat Completions SSE",
            )),
            _ => Err(unsupported_projection(
                "the canonical stream event has no audited Chat Completions projection",
            )),
        }
    }

    /// Whether this encoder has emitted its unique terminal marker.
    pub fn terminal_seen(&self) -> bool {
        self.lifecycle == EncoderLifecycle::Terminal
    }

    /// Validate that the upstream canonical stream ended with its required terminal event.
    ///
    /// This method never emits or synthesizes a successful terminal. Integrations should call it
    /// when the upstream stream reaches clean EOF.
    pub fn finish(&self) -> Result<(), Error> {
        if self.terminal_seen() {
            Ok(())
        } else {
            Err(Error::unexpected_eof())
        }
    }

    fn encode_started(
        &mut self,
        response_id: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<Bytes>, Error> {
        if self.lifecycle != EncoderLifecycle::Fresh {
            return Err(protocol_violation(
                "Chat Completions encoder received more than one Started event",
            ));
        }
        if let Some(response_id) = response_id {
            validate_identifier(
                response_id,
                MAX_RESPONSE_ID_BYTES,
                "Chat Completions response ID",
            )?;
        }

        let frame = data_frame(&chunk_payload(
            response_id,
            model,
            serde_json::json!([{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": Value::Null,
            }]),
            None,
        ))?;

        self.response_id = response_id.map(ToOwned::to_owned);
        self.model = model.map(ToOwned::to_owned);
        self.lifecycle = EncoderLifecycle::Started;
        Ok(vec![frame])
    }

    fn encode_text_start(&mut self, id: &str) -> Result<Vec<Bytes>, Error> {
        validate_identifier(id, MAX_TEXT_PART_ID_BYTES, "canonical text part ID")?;
        if self.active_text_id.is_some() {
            return Err(protocol_violation(
                "Chat Completions encoder received overlapping text parts",
            ));
        }
        self.active_text_id = Some(id.to_string());
        Ok(Vec::new())
    }

    fn encode_text_delta(&self, id: &str, delta: &str) -> Result<Vec<Bytes>, Error> {
        if self.active_text_id.as_deref() != Some(id) {
            return Err(protocol_violation(
                "Chat Completions encoder received a text delta outside its matching text part",
            ));
        }
        if delta.len() > MAX_TEXT_DELTA_BYTES {
            return Err(response_limit(
                "Chat Completions text delta exceeded the encoder byte limit",
            ));
        }
        if delta.is_empty() {
            return Ok(Vec::new());
        }

        Ok(vec![data_frame(&chunk_payload(
            self.response_id.as_deref(),
            self.model.as_deref(),
            serde_json::json!([{
                "index": 0,
                "delta": { "content": delta },
                "finish_reason": Value::Null,
            }]),
            None,
        ))?])
    }

    fn encode_text_end(&mut self, id: &str) -> Result<Vec<Bytes>, Error> {
        if self.active_text_id.as_deref() != Some(id) {
            return Err(protocol_violation(
                "Chat Completions encoder received a text end outside its matching text part",
            ));
        }
        self.active_text_id = None;
        Ok(Vec::new())
    }

    fn encode_refusal(&self, reason: Option<&str>) -> Result<Vec<Bytes>, Error> {
        let reason = reason.ok_or_else(|| {
            unsupported_projection(
                "a refusal without a public reason cannot be represented faithfully",
            )
        })?;
        if reason.len() > MAX_REFUSAL_BYTES {
            return Err(response_limit(
                "Chat Completions refusal text exceeded the encoder byte limit",
            ));
        }

        Ok(vec![data_frame(&chunk_payload(
            self.response_id.as_deref(),
            self.model.as_deref(),
            serde_json::json!([{
                "index": 0,
                "delta": { "refusal": reason },
                "finish_reason": Value::Null,
            }]),
            None,
        ))?])
    }

    fn encode_usage(&mut self, usage: &Usage) -> Result<Vec<Bytes>, Error> {
        let projection = UsageProjection::from_usage(usage);
        if projection.is_empty() || self.last_emitted_usage == Some(projection) {
            return Ok(Vec::new());
        }

        let frame = self.usage_frame(projection)?;
        self.last_emitted_usage = Some(projection);
        Ok(vec![frame])
    }

    fn encode_tool_input_start(
        &mut self,
        id: &str,
        name: &str,
        owner: &ExecutionOwner,
    ) -> Result<Vec<Bytes>, Error> {
        ensure_local_owner(owner)?;
        validate_tool_identity(id, name)?;
        if self.tool_calls.contains_key(id) {
            return Err(protocol_violation(
                "Chat Completions encoder received a duplicate tool call ID",
            ));
        }
        let index = self.next_tool_call_index()?;
        let frame = self.tool_delta_frame(index, id, Some(name), None)?;
        self.tool_calls.insert(
            id.to_string(),
            ToolCallState {
                index,
                name: name.to_string(),
                arguments: String::new(),
                completed: false,
            },
        );
        Ok(vec![frame])
    }

    fn encode_tool_input_delta(&mut self, id: &str, delta: &str) -> Result<Vec<Bytes>, Error> {
        let state = self.tool_calls.get(id).ok_or_else(|| {
            protocol_violation(
                "Chat Completions encoder received tool arguments before their tool start",
            )
        })?;
        if state.completed {
            return Err(protocol_violation(
                "Chat Completions encoder received tool arguments after tool completion",
            ));
        }
        let call_argument_bytes =
            state
                .arguments
                .len()
                .checked_add(delta.len())
                .ok_or_else(|| {
                    response_limit(
                        "Chat Completions tool arguments exceeded the encoder byte limit",
                    )
                })?;
        if call_argument_bytes > MAX_TOOL_ARGUMENT_BYTES {
            return Err(response_limit(
                "Chat Completions tool arguments exceeded the per-call byte limit",
            ));
        }
        let total_argument_bytes = self
            .total_tool_argument_bytes
            .checked_add(delta.len())
            .ok_or_else(|| {
                response_limit("Chat Completions tool state exceeded the encoder byte limit")
            })?;
        if total_argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
            return Err(response_limit(
                "Chat Completions tool state exceeded the aggregate byte limit",
            ));
        }
        if delta.is_empty() {
            return Ok(Vec::new());
        }

        let index = state.index;
        let frame = self.tool_delta_frame(index, id, None, Some(delta))?;
        let state = self.tool_calls.get_mut(id).ok_or_else(missing_tool_state)?;
        state.arguments.push_str(delta);
        self.total_tool_argument_bytes = total_argument_bytes;
        Ok(vec![frame])
    }

    fn encode_tool_call(&mut self, call: &ToolCall) -> Result<Vec<Bytes>, Error> {
        ensure_local_owner(&call.owner)?;
        validate_tool_identity(&call.id, &call.name)?;
        let final_arguments = serde_json::to_string(&call.arguments).map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "failed to serialize canonical tool arguments for Chat Completions",
            )
            .with_source(source)
        })?;
        if final_arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(response_limit(
                "Chat Completions tool arguments exceeded the per-call byte limit",
            ));
        }

        if let Some(state) = self.tool_calls.get(&call.id) {
            if state.completed {
                return Err(protocol_violation(
                    "Chat Completions encoder received a duplicate completed tool call",
                ));
            }
            if state.name != call.name {
                return Err(protocol_violation(
                    "Chat Completions tool name changed during streaming",
                ));
            }

            if state.arguments.is_empty() {
                let total_argument_bytes = self
                    .total_tool_argument_bytes
                    .checked_add(final_arguments.len())
                    .ok_or_else(|| {
                        response_limit(
                            "Chat Completions tool state exceeded the encoder byte limit",
                        )
                    })?;
                if total_argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
                    return Err(response_limit(
                        "Chat Completions tool state exceeded the aggregate byte limit",
                    ));
                }

                let frame =
                    self.tool_delta_frame(state.index, &call.id, None, Some(&final_arguments))?;
                let state = self
                    .tool_calls
                    .get_mut(&call.id)
                    .ok_or_else(missing_tool_state)?;
                state.arguments = final_arguments;
                state.completed = true;
                self.total_tool_argument_bytes = total_argument_bytes;
                return Ok(vec![frame]);
            }

            let streamed_arguments =
                serde_json::from_str::<Value>(&state.arguments).map_err(|source| {
                    Error::new(
                        ErrorKind::ProtocolViolation,
                        "streamed Chat Completions tool arguments did not form complete JSON",
                    )
                    .with_source(source)
                })?;
            if streamed_arguments != call.arguments {
                return Err(protocol_violation(
                    "streamed Chat Completions tool arguments changed at completion",
                ));
            }
            self.tool_calls
                .get_mut(&call.id)
                .ok_or_else(missing_tool_state)?
                .completed = true;
            return Ok(Vec::new());
        }

        let index = self.next_tool_call_index()?;
        let total_argument_bytes = self
            .total_tool_argument_bytes
            .checked_add(final_arguments.len())
            .ok_or_else(|| {
                response_limit("Chat Completions tool state exceeded the encoder byte limit")
            })?;
        if total_argument_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
            return Err(response_limit(
                "Chat Completions tool state exceeded the aggregate byte limit",
            ));
        }

        let frame =
            self.tool_delta_frame(index, &call.id, Some(&call.name), Some(&final_arguments))?;
        self.tool_calls.insert(
            call.id.clone(),
            ToolCallState {
                index,
                name: call.name.clone(),
                arguments: final_arguments,
                completed: true,
            },
        );
        self.total_tool_argument_bytes = total_argument_bytes;
        Ok(vec![frame])
    }

    fn encode_terminal(&mut self, terminal: &StreamTerminal) -> Result<Vec<Bytes>, Error> {
        match terminal {
            StreamTerminal::Completed { response } => self.encode_completed(response),
            StreamTerminal::Failed { error, response } => {
                if let Some(response) = response.as_deref() {
                    validate_terminal_response(response)?;
                }
                let frames = vec![
                    error_frame(error.message(), error_kind_code(error.kind()))?,
                    done_frame(),
                ];
                self.lifecycle = EncoderLifecycle::Terminal;
                Ok(frames)
            }
            StreamTerminal::Cancelled { response, .. } => {
                if let Some(response) = response.as_deref() {
                    validate_terminal_response(response)?;
                }
                let frames = vec![
                    error_frame("language stream was cancelled", "cancelled")?,
                    done_frame(),
                ];
                self.lifecycle = EncoderLifecycle::Terminal;
                Ok(frames)
            }
            _ => Err(unsupported_projection(
                "the terminal outcome has no audited Chat Completions projection",
            )),
        }
    }

    fn encode_completed(&mut self, response: &LanguageResponse) -> Result<Vec<Bytes>, Error> {
        if self.active_text_id.is_some() {
            return Err(protocol_violation(
                "completed Chat Completions stream ended with an open text part",
            ));
        }
        if self.tool_calls.values().any(|state| !state.completed) {
            return Err(protocol_violation(
                "completed Chat Completions stream ended with unfinished tool input",
            ));
        }
        validate_terminal_response(response)?;
        let (response_id, model) = self.resolve_response_identity(response)?;
        let finish_reason = finish_reason_value(response.finish_reason())?;
        let projection = UsageProjection::from_usage(response.usage());

        let mut frames = vec![data_frame(&chunk_payload(
            response_id.as_deref(),
            model.as_deref(),
            serde_json::json!([{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }]),
            None,
        ))?];
        if !projection.is_empty() && self.last_emitted_usage != Some(projection) {
            frames.push(data_frame(&chunk_payload(
                response_id.as_deref(),
                model.as_deref(),
                Value::Array(Vec::new()),
                Some(projection.into_value()),
            ))?);
            self.last_emitted_usage = Some(projection);
        }
        frames.push(done_frame());

        self.response_id = response_id;
        self.model = model;
        self.lifecycle = EncoderLifecycle::Terminal;
        Ok(frames)
    }

    fn resolve_response_identity(
        &self,
        response: &LanguageResponse,
    ) -> Result<(Option<String>, Option<String>), Error> {
        if let Some(response_id) = response.id() {
            validate_identifier(
                response_id,
                MAX_RESPONSE_ID_BYTES,
                "Chat Completions response ID",
            )?;
            if self
                .response_id
                .as_deref()
                .is_some_and(|existing| existing != response_id)
            {
                return Err(protocol_violation(
                    "Chat Completions response ID changed before completion",
                ));
            }
        }
        if let Some(model) = response.model()
            && self
                .model
                .as_deref()
                .is_some_and(|existing| existing != model.as_str())
        {
            return Err(protocol_violation(
                "Chat Completions model ID changed before completion",
            ));
        }

        Ok((
            self.response_id
                .clone()
                .or_else(|| response.id().map(ToOwned::to_owned)),
            self.model
                .clone()
                .or_else(|| response.model().map(|model| model.as_str().to_string())),
        ))
    }

    fn next_tool_call_index(&self) -> Result<u32, Error> {
        if self.tool_calls.len() >= MAX_TOOL_CALLS {
            return Err(response_limit(
                "Chat Completions tool call count exceeded the encoder limit",
            ));
        }
        u32::try_from(self.tool_calls.len())
            .map_err(|_| response_limit("Chat Completions tool call index exceeded the wire range"))
    }

    fn usage_frame(&self, projection: UsageProjection) -> Result<Bytes, Error> {
        data_frame(&chunk_payload(
            self.response_id.as_deref(),
            self.model.as_deref(),
            Value::Array(Vec::new()),
            Some(projection.into_value()),
        ))
    }

    fn tool_delta_frame(
        &self,
        index: u32,
        id: &str,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> Result<Bytes, Error> {
        let mut function = Map::new();
        if let Some(name) = name {
            function.insert("name".to_string(), Value::String(name.to_string()));
        }
        if let Some(arguments) = arguments {
            function.insert(
                "arguments".to_string(),
                Value::String(arguments.to_string()),
            );
        }

        data_frame(&chunk_payload(
            self.response_id.as_deref(),
            self.model.as_deref(),
            serde_json::json!([{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "id": id,
                        "type": "function",
                        "function": Value::Object(function),
                    }],
                },
                "finish_reason": Value::Null,
            }]),
            None,
        ))
    }
}

fn validate_terminal_response(response: &LanguageResponse) -> Result<(), Error> {
    response.validate().map_err(|source| {
        Error::new(
            ErrorKind::ProtocolViolation,
            "canonical terminal response was internally inconsistent",
        )
        .with_source(source)
    })
}

fn validate_tool_identity(id: &str, name: &str) -> Result<(), Error> {
    validate_identifier(id, MAX_TOOL_CALL_ID_BYTES, "Chat Completions tool call ID")?;
    if name.len() > MAX_TOOL_NAME_BYTES {
        return Err(response_limit(
            "Chat Completions tool name exceeded the encoder byte limit",
        ));
    }
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(protocol_violation(
            "Chat Completions tool name was not a valid function name",
        ));
    }
    Ok(())
}

fn validate_identifier(value: &str, maximum: usize, label: &'static str) -> Result<(), Error> {
    if value.len() > maximum {
        return Err(response_limit(match label {
            "Chat Completions response ID" => {
                "Chat Completions response ID exceeded the encoder byte limit"
            }
            "canonical text part ID" => "canonical text part ID exceeded the encoder byte limit",
            "Chat Completions tool call ID" => {
                "Chat Completions tool call ID exceeded the encoder byte limit"
            }
            _ => "Chat Completions identifier exceeded the encoder byte limit",
        }));
    }
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(protocol_violation(match label {
            "Chat Completions response ID" => "Chat Completions response ID was invalid",
            "canonical text part ID" => "canonical text part ID was invalid",
            "Chat Completions tool call ID" => "Chat Completions tool call ID was invalid",
            _ => "Chat Completions identifier was invalid",
        }));
    }
    Ok(())
}

fn ensure_local_owner(owner: &ExecutionOwner) -> Result<(), Error> {
    match owner {
        ExecutionOwner::Local => Ok(()),
        ExecutionOwner::Provider { .. } => Err(unsupported_projection(
            "provider-owned tools cannot be represented as client-executed function calls",
        )),
        _ => Err(unsupported_projection(
            "the tool execution owner has no audited Chat Completions projection",
        )),
    }
}

fn finish_reason_value(reason: &FinishReason) -> Result<Value, Error> {
    let value = match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Refusal => "refusal",
        FinishReason::Other(reason) => {
            if reason.is_empty() || reason.chars().any(char::is_control) {
                return Err(protocol_violation(
                    "custom Chat Completions finish reason was invalid",
                ));
            }
            if reason.len() > MAX_FINISH_REASON_BYTES {
                return Err(response_limit(
                    "custom Chat Completions finish reason exceeded the encoder byte limit",
                ));
            }
            reason
        }
        FinishReason::Error | FinishReason::Cancelled => {
            return Err(protocol_violation(
                "completed Chat Completions response used an error terminal reason",
            ));
        }
        _ => {
            return Err(unsupported_projection(
                "the finish reason has no audited Chat Completions projection",
            ));
        }
    };
    Ok(Value::String(value.to_string()))
}

fn insert_optional_u64(map: &mut Map<String, Value>, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        map.insert(key.to_string(), Value::from(value));
    }
}

fn chunk_payload(
    response_id: Option<&str>,
    model: Option<&str>,
    choices: Value,
    usage: Option<Value>,
) -> Value {
    let mut payload = Map::new();
    if let Some(response_id) = response_id {
        payload.insert("id".to_string(), Value::String(response_id.to_string()));
    }
    payload.insert(
        "object".to_string(),
        Value::String("chat.completion.chunk".to_string()),
    );
    if let Some(model) = model {
        payload.insert("model".to_string(), Value::String(model.to_string()));
    }
    payload.insert("choices".to_string(), choices);
    if let Some(usage) = usage {
        payload.insert("usage".to_string(), usage);
    }
    Value::Object(payload)
}

fn data_frame(value: &Value) -> Result<Bytes, Error> {
    let data = serde_json::to_vec(value).map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "failed to serialize an OpenAI Chat Completions SSE frame",
        )
        .with_source(source)
    })?;
    let mut frame = Vec::with_capacity(data.len() + 8);
    frame.extend_from_slice(b"data: ");
    frame.extend_from_slice(&data);
    frame.extend_from_slice(b"\n\n");
    Ok(Bytes::from(frame))
}

fn error_frame(message: &str, code: &str) -> Result<Bytes, Error> {
    data_frame(&serde_json::json!({
        "error": {
            "message": message,
            "type": "stream_error",
            "code": code,
        },
    }))
}

fn done_frame() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

fn error_kind_code(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidInput => "invalid_input",
        ErrorKind::Configuration => "configuration_error",
        ErrorKind::Authentication => "authentication_error",
        ErrorKind::Authorization => "authorization_error",
        ErrorKind::Unsupported => "unsupported_operation",
        ErrorKind::RateLimited => "rate_limited",
        ErrorKind::QuotaExceeded => "quota_exceeded",
        ErrorKind::Timeout => "timeout",
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::Transport => "transport_error",
        ErrorKind::Protocol => "protocol_error",
        ErrorKind::Provider => "provider_error",
        ErrorKind::UnexpectedEof => "unexpected_eof",
        ErrorKind::ResponseLimit => "response_limit",
        ErrorKind::LimitExceeded => "limit_exceeded",
        ErrorKind::PartialResult => "partial_result",
        ErrorKind::ProtocolViolation => "protocol_violation",
        ErrorKind::StructuredOutput => "structured_output_error",
        ErrorKind::Tool => "tool_error",
        ErrorKind::Internal => "internal_error",
        _ => "stream_error",
    }
}

fn protocol_violation(message: &'static str) -> Error {
    Error::new(ErrorKind::ProtocolViolation, message)
}

fn unsupported_projection(message: &'static str) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn response_limit(message: &'static str) -> Error {
    Error::new(ErrorKind::ResponseLimit, message)
}

fn missing_tool_state() -> Error {
    Error::new(
        ErrorKind::Internal,
        "Chat Completions encoder lost validated tool state",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::str;

    use serde_json::json;
    use siumai_core::{
        Error, ErrorKind, ExecutionOwner, FinishReason, LanguageResponse, LanguageStreamEvent,
        ModelId, OpaqueProviderItem, ProviderId, ProviderProvenance, SensitiveResponse,
        StreamTerminal, ToolCall, Usage, UsageValue,
    };

    use super::*;

    fn started() -> LanguageStreamEvent {
        LanguageStreamEvent::Started {
            id: Some("chatcmpl_test".to_string()),
            model: Some(ModelId::new("model-test").expect("valid model ID")),
        }
    }

    fn completed_response(finish_reason: FinishReason, usage: Usage) -> LanguageResponse {
        LanguageResponse::completed(Vec::new(), finish_reason, usage)
            .expect("valid completed response")
            .with_id("chatcmpl_test")
            .with_model(ModelId::new("model-test").expect("valid model ID"))
    }

    fn completed_event(finish_reason: FinishReason, usage: Usage) -> LanguageStreamEvent {
        LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(completed_response(finish_reason, usage)),
        })
    }

    fn frame_json(frame: &Bytes) -> Value {
        let text = str::from_utf8(frame).expect("SSE frame is UTF-8");
        let data = text
            .strip_prefix("data: ")
            .and_then(|value| value.strip_suffix("\n\n"))
            .expect("complete data frame");
        assert_ne!(data, "[DONE]");
        serde_json::from_str(data).expect("valid SSE JSON")
    }

    fn is_done(frame: &Bytes) -> bool {
        frame.as_ref() == b"data: [DONE]\n\n"
    }

    #[test]
    fn encodes_canonical_text_usage_and_completed_terminal() {
        let usage = Usage::default()
            .with_input_tokens(3)
            .with_output_tokens(5)
            .with_total_tokens(8)
            .with_cache_read_tokens(2)
            .with_reasoning_tokens(1)
            .with_audio_output_tokens(0);
        let mut encoder = ChatCompletionsSseEncoder::new();

        let role = encoder.encode(&started()).expect("encode start");
        assert_eq!(role.len(), 1);
        let role = frame_json(&role[0]);
        assert_eq!(role["id"], "chatcmpl_test");
        assert_eq!(role["model"], "model-test");
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");

        assert!(
            encoder
                .encode(&LanguageStreamEvent::TextStart {
                    id: "text-0".to_string(),
                })
                .expect("encode text start")
                .is_empty()
        );
        let text = encoder
            .encode(&LanguageStreamEvent::TextDelta {
                id: "text-0".to_string(),
                delta: "Hello".to_string(),
            })
            .expect("encode text delta");
        assert_eq!(
            frame_json(&text[0])["choices"][0]["delta"]["content"],
            "Hello"
        );
        assert!(
            encoder
                .encode(&LanguageStreamEvent::TextEnd {
                    id: "text-0".to_string(),
                })
                .expect("encode text end")
                .is_empty()
        );

        let usage_frames = encoder
            .encode(&LanguageStreamEvent::Usage(usage.clone()))
            .expect("encode usage");
        let usage_json = frame_json(&usage_frames[0]);
        assert_eq!(usage_json["choices"], json!([]));
        assert_eq!(usage_json["usage"]["prompt_tokens"], 3);
        assert_eq!(usage_json["usage"]["completion_tokens"], 5);
        assert_eq!(usage_json["usage"]["total_tokens"], 8);
        assert_eq!(
            usage_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            2
        );
        assert_eq!(
            usage_json["usage"]["completion_tokens_details"]["reasoning_tokens"],
            1
        );
        assert_eq!(
            usage_json["usage"]["completion_tokens_details"]["audio_tokens"],
            0
        );

        let terminal = encoder
            .encode(&completed_event(FinishReason::Stop, usage))
            .expect("encode terminal");
        assert_eq!(terminal.len(), 2);
        assert_eq!(
            frame_json(&terminal[0])["choices"][0]["finish_reason"],
            "stop"
        );
        assert!(is_done(&terminal[1]));
        assert!(encoder.terminal_seen());
    }

    #[test]
    fn terminal_is_unique_and_all_later_events_are_rejected() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        let terminal = encoder
            .encode(&completed_event(FinishReason::Stop, Usage::default()))
            .expect("encode terminal");
        assert_eq!(terminal.iter().filter(|frame| is_done(frame)).count(), 1);

        let duplicate = encoder
            .encode(&completed_event(FinishReason::Stop, Usage::default()))
            .expect_err("duplicate terminal must fail");
        assert_eq!(duplicate.kind(), ErrorKind::ProtocolViolation);

        let after_terminal = encoder
            .encode(&LanguageStreamEvent::Usage(Usage::default()))
            .expect_err("event after terminal must fail");
        assert_eq!(after_terminal.kind(), ErrorKind::ProtocolViolation);

        let restart = encoder
            .encode(&started())
            .expect_err("Started must not reset a terminal encoder");
        assert_eq!(restart.kind(), ErrorKind::ProtocolViolation);
    }

    #[test]
    fn second_started_event_cannot_reset_an_active_encoder() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        let error = encoder
            .encode(&started())
            .expect_err("second start must fail");
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }

    #[test]
    fn failed_and_cancelled_terminals_are_explicit_and_sanitized() {
        let mut failed_encoder = ChatCompletionsSseEncoder::new();
        failed_encoder.encode(&started()).expect("encode start");
        let failure =
            Error::new(ErrorKind::Provider, "public provider failure").with_sensitive_response(
                SensitiveResponse::new(BTreeMap::new(), b"PRIVATE_RAW_RESPONSE".to_vec()),
            );
        let failed = failed_encoder
            .encode(&LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                error: failure,
                response: None,
            }))
            .expect("encode failure");
        assert_eq!(failed.len(), 2);
        let failure_text = str::from_utf8(&failed[0]).expect("UTF-8 failure frame");
        assert!(failure_text.contains("public provider failure"));
        assert!(failure_text.contains("provider_error"));
        assert!(!failure_text.contains("PRIVATE_RAW_RESPONSE"));
        assert!(is_done(&failed[1]));

        let mut cancelled_encoder = ChatCompletionsSseEncoder::new();
        cancelled_encoder.encode(&started()).expect("encode start");
        let cancelled = cancelled_encoder
            .encode(&LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                reason: "PRIVATE_CANCELLATION_REASON".to_string(),
                response: None,
            }))
            .expect("encode cancellation");
        let cancellation_text = str::from_utf8(&cancelled[0]).expect("UTF-8 cancellation frame");
        assert!(cancellation_text.contains("language stream was cancelled"));
        assert!(!cancellation_text.contains("PRIVATE_CANCELLATION_REASON"));
        assert!(is_done(&cancelled[1]));
    }

    #[test]
    fn usage_preserves_unknown_and_known_zero_without_provider_payloads() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        assert!(
            encoder
                .encode(&LanguageStreamEvent::Usage(Usage::default()))
                .expect("encode unknown usage")
                .is_empty()
        );

        let usage = Usage::default()
            .with_input_tokens(UsageValue::Known(0))
            .with_output_tokens(UsageValue::Unknown)
            .with_total_tokens(UsageValue::Known(u64::MAX))
            .with_cache_write_tokens(99)
            .with_provider_value("private_raw_usage", "PRIVATE_USAGE_PAYLOAD");
        let frames = encoder
            .encode(&LanguageStreamEvent::Usage(usage))
            .expect("encode known usage");
        let text = str::from_utf8(&frames[0]).expect("UTF-8 usage frame");
        let usage = &frame_json(&frames[0])["usage"];
        assert_eq!(usage["prompt_tokens"], 0);
        assert!(usage.get("completion_tokens").is_none());
        assert_eq!(usage["total_tokens"], u64::MAX);
        assert!(usage.get("cache_write_tokens").is_none());
        assert!(!text.contains("PRIVATE_USAGE_PAYLOAD"));
        assert!(!text.contains("private_raw_usage"));
    }

    #[test]
    fn only_local_tools_are_projected_as_function_calls() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        let provider = ProviderId::new("hosted-runtime").expect("valid provider ID");
        let error = encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-provider".to_string(),
                name: "search".to_string(),
                owner: ExecutionOwner::Provider { provider },
            })
            .expect_err("provider-owned tool must not become a function call");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(!format!("{error:?}").contains("hosted-runtime"));

        let direct_error = encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-provider-direct".to_string(),
                name: "search".to_string(),
                arguments: json!({}),
                owner: ExecutionOwner::Provider {
                    provider: ProviderId::new("hosted-runtime").expect("valid provider ID"),
                },
            }))
            .expect_err("provider-owned completed tool must not become a function call");
        assert_eq!(direct_error.kind(), ErrorKind::Unsupported);
        assert!(!format!("{direct_error:?}").contains("hosted-runtime"));

        let start = encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-local".to_string(),
                name: "lookup".to_string(),
                owner: ExecutionOwner::Local,
            })
            .expect("encode local tool start");
        assert_eq!(
            frame_json(&start[0])["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
    }

    #[test]
    fn local_tool_lifecycle_validates_name_arguments_and_completion() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-0".to_string(),
                name: "lookup".to_string(),
                owner: ExecutionOwner::Local,
            })
            .expect("encode tool start");
        encoder
            .encode(&LanguageStreamEvent::ToolInputDelta {
                id: "call-0".to_string(),
                delta: "{\"city\":\"Paris\"}".to_string(),
            })
            .expect("encode tool arguments");
        assert!(
            encoder
                .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                    id: "call-0".to_string(),
                    name: "lookup".to_string(),
                    arguments: json!({"city": "Paris"}),
                    owner: ExecutionOwner::Local,
                }))
                .expect("complete tool call")
                .is_empty()
        );

        let duplicate = encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-0".to_string(),
                name: "lookup".to_string(),
                arguments: json!({"city": "Paris"}),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("duplicate tool completion must fail");
        assert_eq!(duplicate.kind(), ErrorKind::ProtocolViolation);

        let terminal = encoder
            .encode(&completed_event(FinishReason::ToolCalls, Usage::default()))
            .expect("encode tool terminal");
        assert!(is_done(terminal.last().expect("DONE frame")));
    }

    #[test]
    fn tool_name_changes_and_incomplete_arguments_are_rejected() {
        let mut name_encoder = ChatCompletionsSseEncoder::new();
        name_encoder.encode(&started()).expect("encode start");
        name_encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-0".to_string(),
                name: "lookup".to_string(),
                owner: ExecutionOwner::Local,
            })
            .expect("encode tool start");
        let name_error = name_encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-0".to_string(),
                name: "changed".to_string(),
                arguments: json!({}),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("changed name must fail");
        assert_eq!(name_error.kind(), ErrorKind::ProtocolViolation);

        let mut arguments_encoder = ChatCompletionsSseEncoder::new();
        arguments_encoder.encode(&started()).expect("encode start");
        arguments_encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-0".to_string(),
                name: "lookup".to_string(),
                owner: ExecutionOwner::Local,
            })
            .expect("encode tool start");
        arguments_encoder
            .encode(&LanguageStreamEvent::ToolInputDelta {
                id: "call-0".to_string(),
                delta: "{".to_string(),
            })
            .expect("encode partial arguments");
        let arguments_error = arguments_encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-0".to_string(),
                name: "lookup".to_string(),
                arguments: json!({}),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("incomplete arguments must fail");
        assert_eq!(arguments_error.kind(), ErrorKind::ProtocolViolation);
    }

    #[test]
    fn private_opaque_provider_payload_is_never_projected() {
        let item = OpaqueProviderItem::new(
            ProviderProvenance {
                provider: ProviderId::new("custom-provider").expect("valid provider ID"),
                platform: None,
                protocol: "custom-protocol".to_string(),
                model: ModelId::new("model-test").expect("valid model ID"),
            },
            "private-state",
            json!({"secret": "PRIVATE_OPAQUE_PAYLOAD"}),
        )
        .expect("bounded opaque item");
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        let error = encoder
            .encode(&LanguageStreamEvent::ProviderOpaque(item))
            .expect_err("opaque payload must not be projected");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let diagnostic = format!("{error:?}");
        assert!(!diagnostic.contains("PRIVATE_OPAQUE_PAYLOAD"));
        assert!(!diagnostic.contains("custom-provider"));
    }

    #[test]
    fn ordering_errors_are_rejected_without_synthesizing_a_terminal() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        let before_start = encoder
            .encode(&LanguageStreamEvent::TextDelta {
                id: "text-0".to_string(),
                delta: "late".to_string(),
            })
            .expect_err("Started must be first");
        assert_eq!(before_start.kind(), ErrorKind::ProtocolViolation);
        assert!(!encoder.terminal_seen());

        encoder.encode(&started()).expect("encode start");
        let missing_text_start = encoder
            .encode(&LanguageStreamEvent::TextDelta {
                id: "text-0".to_string(),
                delta: "late".to_string(),
            })
            .expect_err("text delta requires start");
        assert_eq!(missing_text_start.kind(), ErrorKind::ProtocolViolation);

        encoder
            .encode(&LanguageStreamEvent::TextStart {
                id: "text-0".to_string(),
            })
            .expect("encode text start");
        let mismatched_end = encoder
            .encode(&LanguageStreamEvent::TextEnd {
                id: "text-1".to_string(),
            })
            .expect_err("text end ID must match");
        assert_eq!(mismatched_end.kind(), ErrorKind::ProtocolViolation);

        let open_terminal = encoder
            .encode(&completed_event(FinishReason::Stop, Usage::default()))
            .expect_err("completed terminal cannot close an open text part");
        assert_eq!(open_terminal.kind(), ErrorKind::ProtocolViolation);
        assert!(!encoder.terminal_seen());
    }

    #[test]
    fn tool_count_identity_and_argument_bounds_are_enforced() {
        let mut id_encoder = ChatCompletionsSseEncoder::new();
        id_encoder.encode(&started()).expect("encode start");
        let oversized_id = "x".repeat(MAX_TOOL_CALL_ID_BYTES + 1);
        let id_error = id_encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: oversized_id,
                name: "lookup".to_string(),
                owner: ExecutionOwner::Local,
            })
            .expect_err("oversized tool ID must fail");
        assert_eq!(id_error.kind(), ErrorKind::ResponseLimit);

        let oversized_name = "x".repeat(MAX_TOOL_NAME_BYTES + 1);
        let name_error = id_encoder
            .encode(&LanguageStreamEvent::ToolInputStart {
                id: "call-name".to_string(),
                name: oversized_name,
                owner: ExecutionOwner::Local,
            })
            .expect_err("oversized tool name must fail");
        assert_eq!(name_error.kind(), ErrorKind::ResponseLimit);

        let mut argument_encoder = ChatCompletionsSseEncoder::new();
        argument_encoder.encode(&started()).expect("encode start");
        let argument_error = argument_encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-large".to_string(),
                name: "lookup".to_string(),
                arguments: Value::String("x".repeat(MAX_TOOL_ARGUMENT_BYTES + 1)),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("oversized tool arguments must fail");
        assert_eq!(argument_error.kind(), ErrorKind::ResponseLimit);

        argument_encoder.total_tool_argument_bytes = MAX_TOTAL_TOOL_ARGUMENT_BYTES;
        let aggregate_error = argument_encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-aggregate".to_string(),
                name: "lookup".to_string(),
                arguments: json!({}),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("aggregate tool state over limit must fail");
        assert_eq!(aggregate_error.kind(), ErrorKind::ResponseLimit);

        let mut count_encoder = ChatCompletionsSseEncoder::new();
        count_encoder.encode(&started()).expect("encode start");
        for index in 0..MAX_TOOL_CALLS {
            count_encoder
                .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                    id: format!("call-{index}"),
                    name: "lookup".to_string(),
                    arguments: json!({}),
                    owner: ExecutionOwner::Local,
                }))
                .expect("tool within count bound");
        }
        let count_error = count_encoder
            .encode(&LanguageStreamEvent::ToolCall(ToolCall {
                id: "call-over-limit".to_string(),
                name: "lookup".to_string(),
                arguments: json!({}),
                owner: ExecutionOwner::Local,
            }))
            .expect_err("tool count over limit must fail");
        assert_eq!(count_error.kind(), ErrorKind::ResponseLimit);
    }

    #[test]
    fn clean_eof_is_unexpected_without_success_synthesis() {
        let mut encoder = ChatCompletionsSseEncoder::new();
        encoder.encode(&started()).expect("encode start");
        assert!(!encoder.terminal_seen());
        let error = encoder
            .finish()
            .expect_err("clean EOF without terminal must fail");
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);

        encoder
            .encode(&completed_event(FinishReason::Stop, Usage::default()))
            .expect("encode terminal");
        encoder
            .finish()
            .expect("terminal stream may finish cleanly");
    }
}
