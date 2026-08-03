use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::{
    ContentPart, Error, ErrorKind, ExecutionOwner, FinishReason, LanguageResponse,
    LanguageStreamEvent, ModelId, ProviderScope, StreamTerminal, ToolCall, Usage,
};

use super::ChatCompletionsDialect;
use super::response::{decode_finish_reason, decode_usage, parse_model, protocol_error};
use super::wire::{ChatStreamChunkWire, ToolCallDeltaWire};

/// Stateful Chat Completions stream decoder.
///
/// SSE framing remains transport-owned. This decoder owns protocol identity,
/// delta assembly, tool JSON finalization, usage, and the unique terminal event.
pub struct ChatCompletionsStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    dialect: ChatCompletionsDialect,
    started: bool,
    terminal: bool,
    response_id: Option<String>,
    response_model: Option<ModelId>,
    text: String,
    text_started: bool,
    reasoning: String,
    reasoning_started: bool,
    refusals: Vec<Option<String>>,
    tools: BTreeMap<u32, ToolAssembly>,
    order: Vec<ContentOrder>,
    usage: Usage,
    finish_reason: Option<FinishReason>,
}

impl std::fmt::Debug for ChatCompletionsStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatCompletionsStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("started", &self.started)
            .field("terminal", &self.terminal)
            .field("text_bytes", &self.text.len())
            .field("reasoning_bytes", &self.reasoning.len())
            .field("refusal_count", &self.refusals.len())
            .field("tool_count", &self.tools.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentOrder {
    Text,
    Reasoning,
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
            started: false,
            terminal: false,
            response_id: None,
            response_model: None,
            text: String::new(),
            text_started: false,
            reasoning: String::new(),
            reasoning_started: false,
            refusals: Vec::new(),
            tools: BTreeMap::new(),
            order: Vec::new(),
            usage: Usage::default(),
            finish_reason: None,
        }
    }

    /// Decode one framed SSE `data` value.
    pub fn decode(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.terminal {
            return Err(protocol_error(
                "Chat Completions stream emitted data after its terminal marker",
            ));
        }
        if data.trim() == "[DONE]" {
            return self.complete();
        }

        let chunk = serde_json::from_str::<ChatStreamChunkWire>(data).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed Chat Completions stream JSON",
            )
            .with_source(source)
        })?;
        let mut events = Vec::new();
        self.observe_identity(&chunk, &mut events)?;

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

        if let Some(usage) = chunk.usage {
            self.usage = decode_usage(usage);
            events.push(LanguageStreamEvent::Usage(self.usage.clone()));
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
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

    pub fn terminal_seen(&self) -> bool {
        self.terminal
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
                tool.arguments.push_str(&argument_delta);
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
            let call = ToolCall {
                id,
                name,
                arguments,
                owner: ExecutionOwner::Local,
            };
            events.push(LanguageStreamEvent::ToolCall(call.clone()));
            completed_tools.insert(*index, call);
        }

        let mut content = Vec::new();
        for part in &self.order {
            match *part {
                ContentOrder::Text => content.push(ContentPart::Text {
                    text: self.text.clone(),
                }),
                ContentOrder::Reasoning => content.push(ContentPart::Reasoning {
                    text: self.reasoning.clone(),
                }),
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
        let response = LanguageResponse {
            id: self.response_id.clone(),
            model: Some(
                self.response_model
                    .clone()
                    .unwrap_or_else(|| self.requested_model.clone()),
            ),
            content,
            finish_reason,
            usage: self.usage.clone(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        self.terminal = true;
        events.push(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(response),
        }));
        Ok(events)
    }
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
    use siumai_core::{ApiModeId, PlatformId, ProtocolId, ProviderId, UsageValue};

    use super::*;
    use crate::chat_completions::ReasoningField;

    fn decoder() -> ChatCompletionsStreamDecoder {
        let scope = ProviderScope::new(ProviderId::new("deepseek").unwrap())
            .with_platform(PlatformId::new("public-api").unwrap())
            .with_protocol(ProtocolId::new("openai").unwrap())
            .with_api_mode(ApiModeId::new("chat-completions").unwrap());
        ChatCompletionsStreamDecoder::new(
            scope,
            ModelId::new("deepseek-chat").unwrap(),
            ChatCompletionsDialect::generic()
                .with_reasoning_output_field(ReasoningField::new("reasoning_content").unwrap()),
        )
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
        assert_eq!(response.content.len(), 3);
        assert_eq!(response.usage.input_tokens, UsageValue::Known(0));
        assert!(decoder.decode("[DONE]").is_err());
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
}
