use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{
    ContentPart, DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error, ErrorKind, ExecutionOwner,
    FinishReason, LanguageStreamDecoder, LanguageStreamEvent, ModelId, ProviderScope,
    ResponseDiagnostics, StreamTerminal, ToolCall, Usage,
};

use crate::openai_error::classify_stream_error;

use super::ChatCompletionsDialect;
use super::reasoning::ReasoningDetailsSnapshot;
use super::response::{
    build_response, decode_finish_reason, decode_usage, merge_response_metadata, merge_usage,
    parse_model, protocol_error, selected_response_metadata,
};
use super::wire::{ChatStreamChunkWire, ToolCallDeltaWire};

/// Stateful Chat Completions stream decoder.
///
/// SSE framing remains transport-owned. This decoder owns protocol identity,
/// delta assembly, tool JSON finalization, usage, and the unique terminal event.
pub struct ChatCompletionsStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    dialect: ChatCompletionsDialect,
    lifecycle: DecoderLifecycle,
    started: bool,
    response_id: Option<String>,
    response_model: Option<ModelId>,
    text: String,
    text_started: bool,
    reasoning: String,
    reasoning_started: bool,
    reasoning_details: Option<ReasoningDetailsSnapshot>,
    refusals: Vec<Option<String>>,
    tools: BTreeMap<u32, ToolAssembly>,
    order: Vec<ContentOrder>,
    usage: Usage,
    response_metadata: BTreeMap<String, Value>,
    finish_reason: Option<FinishReason>,
    response_diagnostics: ResponseDiagnostics,
}

impl std::fmt::Debug for ChatCompletionsStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatCompletionsStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("started", &self.started)
            .field("terminal", &self.lifecycle.terminal_seen())
            .field("finished", &self.lifecycle.finish_seen())
            .field("text_bytes", &self.text.len())
            .field("reasoning_bytes", &self.reasoning.len())
            .field("has_reasoning_details", &self.reasoning_details.is_some())
            .field("refusal_count", &self.refusals.len())
            .field("tool_count", &self.tools.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentOrder {
    Text,
    Reasoning,
    ReasoningDetails,
    Refusal(usize),
    Tool(u32),
}

#[derive(Debug, Default)]
struct ToolAssembly {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    start_emitted: bool,
}

impl ChatCompletionsStreamDecoder {
    pub fn new(
        scope: ProviderScope,
        requested_model: ModelId,
        dialect: ChatCompletionsDialect,
    ) -> Self {
        Self {
            scope,
            requested_model,
            dialect,
            lifecycle: DecoderLifecycle::default(),
            started: false,
            response_id: None,
            response_model: None,
            text: String::new(),
            text_started: false,
            reasoning: String::new(),
            reasoning_started: false,
            reasoning_details: None,
            refusals: Vec::new(),
            tools: BTreeMap::new(),
            order: Vec::new(),
            usage: Usage::default(),
            response_metadata: BTreeMap::new(),
            finish_reason: None,
            response_diagnostics: ResponseDiagnostics::default(),
        }
    }

    pub fn with_response_diagnostics(mut self, diagnostics: ResponseDiagnostics) -> Self {
        self.response_diagnostics = diagnostics;
        self
    }

    /// Decode one framed SSE `data` value.
    pub fn decode(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::decode(self, data)
    }

    /// Signal clean SSE EOF and finalize the protocol lifecycle.
    pub fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::finish(self)
    }

    pub fn terminal_seen(&self) -> bool {
        <Self as LanguageStreamDecoder>::terminal_seen(self)
    }

    fn decode_frame(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.dialect
            .validate_reasoning_configuration()
            .map_err(|source| {
                Error::new(
                    ErrorKind::Configuration,
                    "Chat Completions reasoning dialect configuration is invalid",
                )
                .with_source(source)
            })?;
        if data.trim() == "[DONE]" {
            return self.complete();
        }

        let value = serde_json::from_str::<Value>(data).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed Chat Completions stream JSON",
            )
            .with_source(source)
        })?;
        if value.get("error").is_some_and(|error| !error.is_null())
            || value.get("type").and_then(Value::as_str) == Some("error")
        {
            return Ok(vec![LanguageStreamEvent::Terminal(
                StreamTerminal::Failed {
                    error: classify_stream_error(
                        &value,
                        self.response_diagnostics.clone(),
                        "provider reported an error after establishing the Chat Completions stream",
                    ),
                    response: None,
                },
            )]);
        }
        let chunk = serde_json::from_value::<ChatStreamChunkWire>(value).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed Chat Completions stream JSON",
            )
            .with_source(source)
        })?;
        let mut events = Vec::new();
        self.observe_identity(&chunk, &mut events)?;
        merge_response_metadata(&mut self.response_metadata, chunk.extra)?;

        if chunk.choices.len() > 1
            || chunk
                .choices
                .first()
                .is_some_and(|choice| choice.index != 0)
        {
            return Err(protocol_error(
                "Chat Completions stream must contain at most choice index zero",
            ));
        }

        let mut choice = chunk.choices.into_iter().next();
        let usage = chunk.usage.or_else(|| {
            self.dialect
                .supports_stream_choice_usage()
                .then(|| choice.as_mut().and_then(|choice| choice.usage.take()))
                .flatten()
        });
        if let Some(usage) = usage {
            let update = decode_usage(usage, &self.dialect)?;
            merge_usage(&mut self.usage, update);
            events.push(LanguageStreamEvent::Usage(self.usage.clone()));
        }
        let Some(choice) = choice else {
            return Ok(events);
        };

        if let Some(delta) = choice.delta.content {
            if !self.text_started {
                self.text_started = true;
                self.order.push(ContentOrder::Text);
                events.push(LanguageStreamEvent::TextStart {
                    id: "text-0".to_string(),
                });
            }
            self.text.push_str(&delta);
            events.push(LanguageStreamEvent::TextDelta {
                id: "text-0".to_string(),
                delta,
            });
        }
        if let Some(field) = self.dialect.reasoning_output_field()
            && let Some(value) = choice.delta.extra.get(field)
            && !value.is_null()
        {
            let delta = value.as_str().ok_or_else(|| {
                protocol_error("Chat Completions reasoning delta was not a string")
            })?;
            if !self.reasoning_started {
                self.reasoning_started = true;
                self.order.push(ContentOrder::Reasoning);
                events.push(LanguageStreamEvent::ReasoningStart {
                    id: "reasoning-0".to_string(),
                });
            }
            self.reasoning.push_str(delta);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: "reasoning-0".to_string(),
                delta: delta.to_string(),
            });
        }
        if let Some(field) = self.dialect.reasoning_details_field()
            && let Some(value) = choice.delta.extra.get(field)
            && !value.is_null()
        {
            let snapshot = ReasoningDetailsSnapshot::response(value)?;
            if self.reasoning_details.is_none() {
                self.order.push(ContentOrder::ReasoningDetails);
            }
            self.reasoning_details = Some(snapshot);
        }
        if let Some(reason) = choice.delta.refusal {
            let index = self.refusals.len();
            self.refusals.push(Some(reason.clone()));
            self.order.push(ContentOrder::Refusal(index));
            events.push(LanguageStreamEvent::Refusal {
                reason: Some(reason),
            });
        }
        for call in choice.delta.tool_calls {
            self.apply_tool_delta(call, &mut events)?;
        }
        if let Some(reason) = choice.finish_reason {
            let reason = decode_finish_reason(&reason);
            if self
                .finish_reason
                .as_ref()
                .is_some_and(|existing| existing != &reason)
            {
                return Err(protocol_error(
                    "Chat Completions stream changed its finish reason",
                ));
            }
            self.finish_reason = Some(reason);
        }
        Ok(events)
    }

    fn observe_identity(
        &mut self,
        chunk: &ChatStreamChunkWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        if let Some(id) = chunk.id.as_deref().filter(|id| !id.is_empty()) {
            if self
                .response_id
                .as_deref()
                .is_some_and(|existing| existing != id)
            {
                return Err(protocol_error(
                    "Chat Completions stream changed its response ID",
                ));
            }
            self.response_id.get_or_insert_with(|| id.to_string());
        }
        if let Some(model) = chunk.model.as_deref().filter(|model| !model.is_empty()) {
            let model = parse_model(Some(model), &self.requested_model)?;
            if self
                .response_model
                .as_ref()
                .is_some_and(|existing| existing != &model)
            {
                return Err(protocol_error(
                    "Chat Completions stream changed its model ID",
                ));
            }
            self.response_model.get_or_insert(model);
        }
        if !self.started {
            self.started = true;
            events.push(LanguageStreamEvent::Started {
                id: self.response_id.clone(),
                model: self.response_model.clone(),
            });
        }
        Ok(())
    }

    fn apply_tool_delta(
        &mut self,
        delta: ToolCallDeltaWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        if delta.kind.as_deref().is_some_and(|kind| kind != "function") {
            return Err(protocol_error(
                "Chat Completions stream emitted a non-function tool call",
            ));
        }
        let index = delta.index;
        let is_new = !self.tools.contains_key(&index);
        let tool = self.tools.entry(index).or_default();
        if is_new {
            self.order.push(ContentOrder::Tool(index));
        }
        merge_identity(&mut tool.id, delta.id, "tool call ID")?;
        if let Some(function) = delta.function {
            merge_identity(&mut tool.name, function.name, "tool function name")?;
            let argument_delta = function.arguments.unwrap_or_default();
            if !argument_delta.is_empty() {
                append_tool_input(&mut tool.arguments, &argument_delta)?;
            }
            if !tool.start_emitted
                && let (Some(id), Some(name)) = (&tool.id, &tool.name)
            {
                tool.start_emitted = true;
                events.push(LanguageStreamEvent::ToolInputStart {
                    id: id.clone(),
                    name: name.clone(),
                    owner: ExecutionOwner::Local,
                });
                if !tool.arguments.is_empty() {
                    events.push(LanguageStreamEvent::ToolInputDelta {
                        id: id.clone(),
                        delta: tool.arguments.clone(),
                    });
                }
            } else if tool.start_emitted && !argument_delta.is_empty() {
                events.push(LanguageStreamEvent::ToolInputDelta {
                    id: tool.id.clone().expect("started tools have an ID"),
                    delta: argument_delta,
                });
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if !self.started {
            return Err(protocol_error(
                "Chat Completions stream ended before a response chunk",
            ));
        }
        let finish_reason = self
            .finish_reason
            .clone()
            .ok_or_else(|| protocol_error("Chat Completions stream omitted its finish reason"))?;
        let mut events = Vec::new();
        if self.text_started {
            events.push(LanguageStreamEvent::TextEnd {
                id: "text-0".to_string(),
            });
        }
        if self.reasoning_started {
            events.push(LanguageStreamEvent::ReasoningEnd {
                id: "reasoning-0".to_string(),
            });
        }

        let mut completed_tools = BTreeMap::new();
        for (index, tool) in &self.tools {
            let id = tool.id.clone().ok_or_else(|| {
                protocol_error("Chat Completions tool stream omitted its call ID")
            })?;
            let name = tool.name.clone().ok_or_else(|| {
                protocol_error("Chat Completions tool stream omitted its function name")
            })?;
            if !tool.start_emitted {
                events.push(LanguageStreamEvent::ToolInputStart {
                    id: id.clone(),
                    name: name.clone(),
                    owner: ExecutionOwner::Local,
                });
                if !tool.arguments.is_empty() {
                    events.push(LanguageStreamEvent::ToolInputDelta {
                        id: id.clone(),
                        delta: tool.arguments.clone(),
                    });
                }
            }
            let arguments = serde_json::from_str::<Value>(&tool.arguments).map_err(|source| {
                Error::new(
                    ErrorKind::Protocol,
                    "Chat Completions streamed tool arguments were not complete JSON",
                )
                .with_source(source)
            })?;
            let call = ToolCall::local(id, name, arguments).map_err(|source| {
                Error::new(
                    ErrorKind::Protocol,
                    "Chat Completions stream violated the canonical tool contract",
                )
                .with_source(source)
            })?;
            events.push(LanguageStreamEvent::ToolCall(call.clone()));
            completed_tools.insert(*index, call);
        }

        let response_model = self
            .response_model
            .clone()
            .unwrap_or_else(|| self.requested_model.clone());
        let reasoning_details = self
            .reasoning_details
            .clone()
            .map(|snapshot| snapshot.into_opaque(&self.scope, &response_model))
            .transpose()?;
        let mut content = Vec::new();
        for part in &self.order {
            match *part {
                ContentOrder::Text => content.push(ContentPart::Text {
                    text: self.text.clone(),
                }),
                ContentOrder::Reasoning => content.push(ContentPart::Reasoning {
                    text: self.reasoning.clone(),
                }),
                ContentOrder::ReasoningDetails => {
                    let item = reasoning_details.clone().ok_or_else(|| {
                        protocol_error("Chat Completions reasoning_details assembly was incomplete")
                    })?;
                    events.push(LanguageStreamEvent::ProviderOpaque(item.clone()));
                    content.push(ContentPart::ProviderOpaque(item));
                }
                ContentOrder::Refusal(index) => content.push(ContentPart::Refusal {
                    reason: self.refusals[index].clone(),
                }),
                ContentOrder::Tool(index) => {
                    let call = completed_tools.get(&index).ok_or_else(|| {
                        protocol_error("Chat Completions tool assembly was incomplete")
                    })?;
                    content.push(ContentPart::ToolCall(call.clone()));
                }
            }
        }
        let response = build_response(
            self.response_id.clone(),
            response_model,
            content,
            finish_reason,
            self.usage.clone(),
            selected_response_metadata(&self.response_metadata)?,
        )?;
        events.push(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(response),
        }));
        Ok(events)
    }
}

impl LanguageStreamDecoder for ChatCompletionsStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.response_diagnostics = diagnostics;
    }

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

fn append_tool_input(buffer: &mut String, delta: &str) -> Result<(), Error> {
    if buffer
        .len()
        .checked_add(delta.len())
        .is_none_or(|total| total > DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Chat Completions streamed tool arguments exceeded the byte limit",
        ));
    }
    buffer.push_str(delta);
    Ok(())
}

fn merge_identity(
    current: &mut Option<String>,
    incoming: Option<String>,
    _field: &'static str,
) -> Result<(), Error> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    if incoming.is_empty()
        || current
            .as_ref()
            .is_some_and(|existing| existing != &incoming)
    {
        return Err(protocol_error(
            "Chat Completions tool stream changed or omitted an identity field",
        ));
    }
    current.get_or_insert(incoming);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use siumai_core::{
        ApiModeId, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId, UsageValue,
    };

    use super::*;
    use crate::chat_completions::WireFieldName;

    fn decoder() -> ChatCompletionsStreamDecoder {
        let scope = ProviderScope::new(ProviderId::new("deepseek").unwrap())
            .with_platform(PlatformId::new("public-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("deepseek-stream-test").unwrap(),
            ));
        ChatCompletionsStreamDecoder::new(
            scope,
            ModelId::new("deepseek-chat").unwrap(),
            ChatCompletionsDialect::generic()
                .with_reasoning_output_field(WireFieldName::new("reasoning_content").unwrap()),
        )
    }

    fn minimax_decoder() -> ChatCompletionsStreamDecoder {
        let scope = ProviderScope::new(ProviderId::new("minimax").unwrap())
            .with_platform(PlatformId::new("minimax-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("minimax-test").unwrap(),
            ));
        let reasoning = WireFieldName::new("reasoning_content").unwrap();
        let dialect = ChatCompletionsDialect::generic()
            .with_reasoning_input_field(reasoning.clone())
            .with_reasoning_output_field(reasoning)
            .with_replayable_reasoning_details_field(
                WireFieldName::new("reasoning_details").unwrap(),
            )
            .unwrap();
        ChatCompletionsStreamDecoder::new(scope, ModelId::new("MiniMax-M3").unwrap(), dialect)
    }

    #[test]
    fn assembles_lossless_text_reasoning_tool_usage_and_one_terminal() {
        let mut decoder = decoder();
        let first = decoder
            .decode(
                r#"{"id":"chat-1","model":"deepseek-chat","choices":[{"index":0,"delta":{"reasoning_content":"why","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
            )
            .unwrap();
        assert!(first.iter().any(|event| matches!(
            event,
            LanguageStreamEvent::ReasoningDelta { delta, .. } if delta == "why"
        )));

        let second = decoder
            .decode(
                r#"{"choices":[{"index":0,"delta":{"content":"","tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":0,"completion_tokens":4,"total_tokens":4}}"#,
            )
            .unwrap();
        assert!(second.iter().any(|event| matches!(
            event,
            LanguageStreamEvent::TextDelta { delta, .. } if delta.is_empty()
        )));
        assert!(second.iter().any(|event| matches!(
            event,
            LanguageStreamEvent::Usage(usage) if usage.input_tokens == UsageValue::Known(0)
        )));

        let terminal = decoder.decode("[DONE]").unwrap();
        assert_eq!(
            terminal
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })) =
            terminal.last()
        else {
            panic!("expected completed terminal")
        };
        assert_eq!(response.content().len(), 3);
        assert_eq!(response.usage().input_tokens, UsageValue::Known(0));
        assert!(decoder.decode("[DONE]").is_err());
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn done_without_finish_reason_is_a_protocol_error() {
        let mut decoder = decoder();
        decoder
            .decode(r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#)
            .unwrap();
        assert!(decoder.decode("[DONE]").is_err());
        assert!(!decoder.terminal_seen());
    }

    #[test]
    fn streamed_tool_input_is_bounded_before_json_normalization() {
        let mut decoder = decoder();
        decoder
            .decode(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{"}}]}}]}"#,
            )
            .unwrap();
        let oversized = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": " ".repeat(DEFAULT_TOOL_INPUT_BYTE_LIMIT)
                        }
                    }]
                }
            }]
        })
        .to_string();

        let error = decoder.decode(&oversized).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);
        assert_eq!(decoder.tools[&0].arguments, "{");
    }

    #[test]
    fn eof_before_done_is_unexpected_and_closes_the_decoder() {
        let mut decoder = decoder();
        decoder
            .decode(r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#)
            .unwrap();

        let error = decoder.finish().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
        assert!(!decoder.terminal_seen());
        assert!(decoder.decode("[DONE]").is_err());
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn preserves_cumulative_structured_reasoning_in_stream_terminal() {
        let mut decoder = minimax_decoder();
        let first = serde_json::json!({
            "id": "chat-minimax-stream",
            "model": "MiniMax-M3",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "inspect",
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "id": "reasoning-text-1",
                        "format": "MiniMax-response-v1",
                        "index": 0,
                        "text": "inspect"
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string();
        decoder.decode(&first).unwrap();

        let second = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": " the tool",
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "id": "reasoning-text-1",
                        "format": "MiniMax-response-v1",
                        "index": 0,
                        "text": "inspect the tool"
                    }],
                    "content": "done"
                },
                "finish_reason": "stop"
            }]
        })
        .to_string();
        decoder.decode(&second).unwrap();

        let terminal = decoder.decode("[DONE]").unwrap();
        let opaque_event = terminal.iter().find_map(|event| match event {
            LanguageStreamEvent::ProviderOpaque(item) => Some(item),
            _ => None,
        });
        assert!(matches!(
            opaque_event,
            Some(item)
                if item.kind() == crate::chat_completions::REASONING_DETAILS_OPAQUE_KIND
                    && item.data()[0]["text"] == "inspect the tool"
        ));
        let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })) =
            terminal.last()
        else {
            panic!("expected completed terminal")
        };
        assert!(response.content().iter().any(|part| matches!(
            part,
            ContentPart::ProviderOpaque(item)
                if item.data()[0]["text"] == "inspect the tool"
                    && item.provenance().provider().as_str() == "minimax"
        )));
    }

    #[test]
    fn stream_maps_root_cache_usage_only_for_an_explicit_dialect() {
        let scope = ProviderScope::new(ProviderId::new("moonshotai").unwrap())
            .with_platform(PlatformId::new("kimi-public-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("moonshotai-test").unwrap(),
            ));
        let mut decoder = ChatCompletionsStreamDecoder::new(
            scope,
            ModelId::new("kimi-k3").unwrap(),
            ChatCompletionsDialect::generic()
                .with_cache_read_tokens_field(WireFieldName::new("cached_tokens").unwrap()),
        );
        let events = decoder
            .decode(
                r#"{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cached_tokens":7}}"#,
            )
            .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            LanguageStreamEvent::Usage(usage)
                if usage.cache_read_tokens == UsageValue::Known(7)
        )));
        assert!(decoder.decode("[DONE]").is_ok());
    }

    #[test]
    fn choice_usage_requires_an_explicit_dialect() {
        let scope = ProviderScope::new(ProviderId::new("moonshotai").unwrap())
            .with_platform(PlatformId::new("kimi-public-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("moonshotai-test").unwrap(),
            ));
        let frame = r#"{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cached_tokens":7}}]}"#;

        let mut generic = ChatCompletionsStreamDecoder::new(
            scope.clone(),
            ModelId::new("kimi-k3").unwrap(),
            ChatCompletionsDialect::generic(),
        );
        assert!(
            !generic
                .decode(frame)
                .unwrap()
                .iter()
                .any(|event| matches!(event, LanguageStreamEvent::Usage(_)))
        );

        let mut kimi = ChatCompletionsStreamDecoder::new(
            scope,
            ModelId::new("kimi-k3").unwrap(),
            ChatCompletionsDialect::generic()
                .with_stream_choice_usage(true)
                .with_cache_read_tokens_field(WireFieldName::new("cached_tokens").unwrap()),
        );
        assert!(kimi.decode(frame).unwrap().iter().any(|event| matches!(
            event,
            LanguageStreamEvent::Usage(usage)
                if usage.cache_read_tokens == UsageValue::Known(7)
        )));
    }

    #[test]
    fn finish_reason_waits_for_trailing_usage_before_terminal() {
        let mut decoder = decoder();
        let finish = decoder
            .decode(r#"{"id":"chat-stream-1","model":"gpt-5.6-sol","system_fingerprint":"fp_current","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#)
            .unwrap();
        assert!(finish.iter().all(|event| event.terminal().is_none()));

        let usage = decoder
            .decode(
                r#"{"service_tier":"priority","future_scalar":true,"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":3,"audio_tokens":2},"completion_tokens_details":{"reasoning_tokens":5,"audio_tokens":1,"accepted_prediction_tokens":6,"rejected_prediction_tokens":7}}}"#,
            )
            .unwrap();
        assert!(usage.iter().any(|event| matches!(
            event,
            LanguageStreamEvent::Usage(usage)
                if usage.input_tokens == UsageValue::Known(10)
                    && usage.output_tokens == UsageValue::Known(8)
                    && usage.cache_write_tokens == UsageValue::Known(3)
        )));

        let terminal = decoder.decode("[DONE]").unwrap();
        assert!(matches!(
            terminal.as_slice(),
            [.., LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })]
                if response.usage().input_tokens == UsageValue::Known(10)
                    && response.usage().output_tokens == UsageValue::Known(8)
                    && response.usage().cache_read_tokens == UsageValue::Known(4)
                    && response.usage().cache_write_tokens == UsageValue::Known(3)
                    && response.usage().provider["accepted_prediction_tokens"] == 6
                    && response.provider_metadata()["openai"]["system_fingerprint"]
                        == "fp_current"
                    && response.provider_metadata()["openai"]["service_tier"] == "priority"
                    && response.provider_metadata()["openai"]["future_scalar"] == true
        ));
    }

    #[test]
    fn in_band_error_is_a_typed_failed_terminal() {
        let mut decoder = decoder().with_response_diagnostics(
            ResponseDiagnostics::default().with_retry_after(Duration::from_secs(3)),
        );
        let events = decoder
            .decode(
                r#"{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"stream-secret","retry_after":3}}"#,
            )
            .unwrap();

        assert!(matches!(
            events.as_slice(),
            [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, response: None })]
                if error.kind() == ErrorKind::RateLimited
                    && error.diagnostics().and_then(ResponseDiagnostics::retry_after)
                        == Some(Duration::from_secs(3))
        ));
        assert!(!format!("{events:?}").contains("stream-secret"));
    }
}
