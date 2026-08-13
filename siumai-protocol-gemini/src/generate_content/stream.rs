use serde_json::{Map, Value, json};
use siumai_core::{
    ContentPart, DecoderLifecycle, Error, ErrorKind, LanguageCallError, LanguageResponse,
    LanguageStreamDecoder, LanguageStreamEvent, ModelId, PartialLanguageOutput, ProviderScope,
    PublicDiagnosticText, ResponseDiagnostics, StreamTerminal, UsageUpdate,
};

use super::language::{
    UsageMetadataWire, normalize_response_parts, partial_output, project_response_part,
    project_response_value, valid_bounded_text,
};

const MAX_STREAM_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_STREAM_PARTS: usize = 256;
const MAX_STREAM_METADATA_BYTES: usize = 1024 * 1024;
const MAX_STREAM_ID_BYTES: usize = 4 * 1024;

/// EOF-settled decoder for stable v1 Generate Content SSE data frames.
pub struct GenerateContentStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    lifecycle: DecoderLifecycle,
    diagnostics: ResponseDiagnostics,
    response_id: Option<String>,
    response_model: Option<ModelId>,
    model_status: Option<Value>,
    prompt_feedback: Option<Value>,
    usage_metadata: Option<Value>,
    candidate_metadata: Map<String, Value>,
    parts: Vec<Value>,
    finish_reason: Option<String>,
    prompt_blocked: bool,
    candidate_seen: bool,
    started: bool,
    open_text: Option<OpenText>,
}

impl std::fmt::Debug for GenerateContentStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerateContentStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("terminal_seen", &self.lifecycle.terminal_seen())
            .field("finish_seen", &self.lifecycle.finish_seen())
            .field("response_id", &self.response_id)
            .field("response_model", &self.response_model)
            .field("parts", &self.parts.len())
            .field("finish_reason", &self.finish_reason)
            .field("prompt_blocked", &self.prompt_blocked)
            .finish()
    }
}

impl GenerateContentStreamDecoder {
    pub fn new(scope: ProviderScope, requested_model: ModelId) -> Self {
        Self {
            scope,
            requested_model,
            lifecycle: DecoderLifecycle::default(),
            diagnostics: ResponseDiagnostics::default(),
            response_id: None,
            response_model: None,
            model_status: None,
            prompt_feedback: None,
            usage_metadata: None,
            candidate_metadata: Map::new(),
            parts: Vec::new(),
            finish_reason: None,
            prompt_blocked: false,
            candidate_seen: false,
            started: false,
            open_text: None,
        }
    }

    fn decode_frame(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if frame.len() > MAX_STREAM_FRAME_BYTES {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Generate Content SSE frame exceeded the protocol limit",
            ));
        }
        if frame.trim() == "[DONE]" {
            return Err(Error::protocol_violation(
                "stable Gemini Generate Content does not define a DONE sentinel",
            ));
        }
        let value = serde_json::from_str::<Value>(frame).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed Gemini Generate Content SSE JSON",
            )
            .with_source(source)
        })?;
        if value.get("error").is_some() {
            let error = self.in_band_error(&value);
            let partial = self.partial_output();
            let terminal = if error.kind() == ErrorKind::Cancelled {
                StreamTerminal::Cancelled {
                    reason: error.message().to_string(),
                    partial,
                }
            } else {
                StreamTerminal::Failed { error, partial }
            };
            return Ok(vec![LanguageStreamEvent::Terminal(terminal)]);
        }
        let object = value.as_object().ok_or_else(|| {
            Error::protocol_violation("Gemini Generate Content SSE frame was not an object")
        })?;

        self.update_identity(object)?;
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(LanguageStreamEvent::Started {
                id: self.response_id.clone(),
                model: Some(
                    self.response_model
                        .clone()
                        .unwrap_or_else(|| self.requested_model.clone()),
                ),
            });
        }

        if let Some(model_status) = object.get("modelStatus") {
            ensure_value_limit(model_status, MAX_STREAM_METADATA_BYTES)?;
            self.model_status = Some(model_status.clone());
        }
        if let Some(prompt_feedback) = object.get("promptFeedback") {
            self.record_prompt_feedback(prompt_feedback, &mut events)?;
        }
        if let Some(candidates) = object.get("candidates") {
            self.record_candidates(candidates, &mut events)?;
        }
        if let Some(usage_metadata) = object.get("usageMetadata") {
            let wire = serde_json::from_value::<UsageMetadataWire>(usage_metadata.clone())
                .map_err(|source| {
                    Error::protocol_violation(
                        "Gemini Generate Content streamed malformed usage metadata",
                    )
                    .with_source(source)
                })?;
            let usage = super::language::decode_usage(Some(wire))?;
            self.usage_metadata = Some(usage_metadata.clone());
            events.push(LanguageStreamEvent::Usage(UsageUpdate::snapshot(usage)));
        }
        Ok(events)
    }

    fn update_identity(&mut self, object: &Map<String, Value>) -> Result<(), Error> {
        if let Some(response_id) = object.get("responseId").and_then(Value::as_str) {
            if !valid_bounded_text(response_id, MAX_STREAM_ID_BYTES) {
                return Err(Error::protocol_violation(
                    "Gemini Generate Content stream returned an invalid response ID",
                ));
            }
            match &self.response_id {
                Some(existing) if existing != response_id => {
                    return Err(Error::protocol_violation(
                        "Gemini Generate Content stream changed its response ID",
                    ));
                }
                Some(_) => {}
                None => self.response_id = Some(response_id.to_string()),
            }
        }
        if let Some(model) = object.get("modelVersion").and_then(Value::as_str) {
            let model = ModelId::new(model).map_err(|source| {
                Error::protocol_violation(
                    "Gemini Generate Content stream returned an invalid model ID",
                )
                .with_source(source)
            })?;
            match &self.response_model {
                Some(existing) if existing != &model => {
                    return Err(Error::protocol_violation(
                        "Gemini Generate Content stream changed its response model",
                    ));
                }
                Some(_) => {}
                None => self.response_model = Some(model),
            }
        }
        Ok(())
    }

    fn record_prompt_feedback(
        &mut self,
        prompt_feedback: &Value,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        ensure_value_limit(prompt_feedback, MAX_STREAM_METADATA_BYTES)?;
        let block_reason = prompt_feedback.get("blockReason").and_then(Value::as_str);
        if block_reason.is_some_and(|reason| !valid_bounded_text(reason, 128)) {
            return Err(Error::protocol_violation(
                "Gemini streamed an invalid prompt block reason",
            ));
        }
        self.prompt_feedback = Some(prompt_feedback.clone());
        if block_reason.is_some_and(|reason| reason != "BLOCK_REASON_UNSPECIFIED") {
            if self.candidate_seen || self.finish_reason.is_some() {
                return Err(Error::protocol_violation(
                    "Gemini stream combined prompt blocking with a response candidate",
                ));
            }
            self.close_open_text(events);
            self.prompt_blocked = true;
        }
        Ok(())
    }

    fn record_candidates(
        &mut self,
        candidates: &Value,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        let candidates = candidates.as_array().ok_or_else(|| {
            Error::protocol_violation("Gemini streamed candidates were not an array")
        })?;
        if candidates.len() > 1 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "portable Gemini Generate Content streams support exactly one candidate",
            ));
        }
        let Some(candidate) = candidates.first() else {
            return Ok(());
        };
        if self.prompt_blocked {
            return Err(Error::protocol_violation(
                "Gemini stream emitted a candidate after prompt blocking",
            ));
        }
        let candidate = candidate.as_object().ok_or_else(|| {
            Error::protocol_violation("Gemini streamed candidate was not an object")
        })?;
        if candidate.get("index").and_then(Value::as_u64).unwrap_or(0) != 0 {
            return Err(Error::protocol_violation(
                "Gemini portable stream candidate had a non-zero index",
            ));
        }
        self.candidate_seen = true;

        if let Some(content) = candidate.get("content") {
            let content = content.as_object().ok_or_else(|| {
                Error::protocol_violation("Gemini streamed candidate content was not an object")
            })?;
            if content
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| role != "model")
            {
                return Err(Error::protocol_violation(
                    "Gemini streamed candidate used an invalid content role",
                ));
            }
            let parts = content
                .get("parts")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Error::protocol_violation("Gemini streamed candidate content omitted its parts")
                })?;
            if self.finish_reason.is_some() && !parts.is_empty() {
                return Err(Error::protocol_violation(
                    "Gemini stream emitted content after its finish reason",
                ));
            }
            for part in parts {
                self.record_part(part.clone(), events)?;
            }
        }

        for (key, value) in candidate {
            if !matches!(key.as_str(), "content" | "finishReason" | "index") {
                ensure_value_limit(value, MAX_STREAM_METADATA_BYTES)?;
                self.candidate_metadata.insert(key.clone(), value.clone());
            }
        }
        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            if !valid_bounded_text(reason, 128) {
                return Err(Error::protocol_violation(
                    "Gemini streamed an invalid finish reason",
                ));
            }
            if reason == "FINISH_REASON_UNSPECIFIED" {
                return Ok(());
            }
            match &self.finish_reason {
                Some(existing) if existing != reason => {
                    return Err(Error::protocol_violation(
                        "Gemini stream changed its finish reason",
                    ));
                }
                Some(_) => {}
                None => {
                    self.finish_reason = Some(reason.to_string());
                    self.close_open_text(events);
                }
            }
        }
        Ok(())
    }

    fn record_part(
        &mut self,
        part: Value,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        ensure_value_limit(&part, MAX_STREAM_METADATA_BYTES)?;
        let text = part.get("text").and_then(Value::as_str).map(str::to_string);
        let reasoning = part.get("thought").and_then(Value::as_bool) == Some(true);

        if let Some(text) = text {
            let part_index = self.append_part(part)?;
            let kind = if reasoning {
                TextKind::Reasoning
            } else {
                TextKind::Text
            };
            self.open_text(kind, part_index, events);
            if !text.is_empty() {
                let id = self
                    .open_text
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::Internal,
                            "Gemini stream text state was not initialized",
                        )
                    })?
                    .id
                    .clone();
                events.push(match kind {
                    TextKind::Text => LanguageStreamEvent::TextDelta { id, delta: text },
                    TextKind::Reasoning => LanguageStreamEvent::ReasoningDelta { id, delta: text },
                });
            }
            return Ok(());
        }

        self.close_open_text(events);
        if self.parts.len() >= MAX_STREAM_PARTS {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Generate Content stream exceeded the part limit",
            ));
        }
        let part_index = self.parts.len();
        self.parts.push(part.clone());

        let mut projected = Vec::new();
        let mut warnings = Vec::new();
        project_response_part(
            &part,
            &self.scope,
            &self.requested_model,
            self.response_id.as_deref(),
            0,
            part_index,
            &mut projected,
            &mut warnings,
        )?;
        if part.get("functionCall").is_some() {
            if let Some(call) = projected.into_iter().find_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call),
                _ => None,
            }) {
                events.push(LanguageStreamEvent::ToolCall(call));
            }
        } else if part.get("inlineData").is_none()
            && part.get("fileData").is_none()
            && let Some(item) = projected.into_iter().find_map(|part| match part {
                ContentPart::ProviderOpaque(item) => Some(item),
                _ => None,
            })
        {
            events.push(LanguageStreamEvent::ProviderOpaque(item));
        }
        Ok(())
    }

    fn append_part(&mut self, part: Value) -> Result<usize, Error> {
        if let Some(previous) = self.parts.last().cloned() {
            let normalized = normalize_response_parts(vec![previous, part.clone()])?;
            if normalized.len() == 1 {
                let index = self.parts.len() - 1;
                let normalized = normalized.into_iter().next().ok_or_else(|| {
                    Error::new(
                        ErrorKind::Internal,
                        "Gemini stream text normalization returned no part",
                    )
                })?;
                self.parts[index] = normalized;
                return Ok(index);
            }
        }
        if self.parts.len() >= MAX_STREAM_PARTS {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Generate Content stream exceeded the part limit",
            ));
        }
        let index = self.parts.len();
        self.parts.push(part);
        Ok(index)
    }

    fn open_text(
        &mut self,
        kind: TextKind,
        part_index: usize,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        if self
            .open_text
            .as_ref()
            .is_some_and(|open| open.kind == kind && open.part_index == part_index)
        {
            return;
        }
        self.close_open_text(events);
        let id = match kind {
            TextKind::Text => format!("gemini-text-0-{part_index}"),
            TextKind::Reasoning => format!("gemini-reasoning-0-{part_index}"),
        };
        events.push(match kind {
            TextKind::Text => LanguageStreamEvent::TextStart { id: id.clone() },
            TextKind::Reasoning => LanguageStreamEvent::ReasoningStart { id: id.clone() },
        });
        self.open_text = Some(OpenText {
            id,
            kind,
            part_index,
        });
    }

    fn close_open_text(&mut self, events: &mut Vec<LanguageStreamEvent>) {
        let Some(open) = self.open_text.take() else {
            return;
        };
        events.push(match open.kind {
            TextKind::Text => LanguageStreamEvent::TextEnd { id: open.id },
            TextKind::Reasoning => LanguageStreamEvent::ReasoningEnd { id: open.id },
        });
    }

    fn terminal_response(&self) -> Result<Result<LanguageResponse, LanguageCallError>, Error> {
        let mut root = Map::new();
        if let Some(response_id) = &self.response_id {
            root.insert("responseId".to_string(), Value::String(response_id.clone()));
        }
        if let Some(model) = &self.response_model {
            root.insert(
                "modelVersion".to_string(),
                Value::String(model.as_str().to_string()),
            );
        }
        if let Some(model_status) = &self.model_status {
            root.insert("modelStatus".to_string(), model_status.clone());
        }
        if let Some(prompt_feedback) = &self.prompt_feedback {
            root.insert("promptFeedback".to_string(), prompt_feedback.clone());
        }
        if let Some(usage_metadata) = &self.usage_metadata {
            root.insert("usageMetadata".to_string(), usage_metadata.clone());
        }

        if let Some(finish_reason) = &self.finish_reason {
            let mut candidate = self.candidate_metadata.clone();
            candidate.insert("index".to_string(), Value::from(0));
            candidate.insert(
                "content".to_string(),
                json!({"role": "model", "parts": self.parts}),
            );
            candidate.insert(
                "finishReason".to_string(),
                Value::String(finish_reason.clone()),
            );
            root.insert(
                "candidates".to_string(),
                Value::Array(vec![Value::Object(candidate)]),
            );
        }
        project_response_value(&Value::Object(root), &self.scope, &self.requested_model)
    }

    fn partial_output(&self) -> Option<PartialLanguageOutput> {
        let parts = normalize_response_parts(self.parts.clone()).ok()?;
        let model = self
            .response_model
            .as_ref()
            .unwrap_or(&self.requested_model);
        let mut content = Vec::new();
        let mut warnings = Vec::new();
        for (part_index, part) in parts.iter().enumerate() {
            project_response_part(
                part,
                &self.scope,
                model,
                self.response_id.as_deref(),
                0,
                part_index,
                &mut content,
                &mut warnings,
            )
            .ok()?;
        }
        let usage = self
            .usage_metadata
            .clone()
            .map(serde_json::from_value::<UsageMetadataWire>)
            .transpose()
            .ok()
            .flatten()
            .and_then(|wire| super::language::decode_usage(Some(wire)).ok())
            .unwrap_or_default();
        partial_output(&content, &usage)
    }

    fn in_band_error(&self, value: &Value) -> Error {
        let status = value
            .get("error")
            .and_then(|error| error.get("status"))
            .and_then(Value::as_str)
            .filter(|status| valid_bounded_text(status, 128));
        let kind = match status {
            Some("INVALID_ARGUMENT" | "NOT_FOUND" | "FAILED_PRECONDITION") => {
                ErrorKind::InvalidInput
            }
            Some("UNAUTHENTICATED") => ErrorKind::Authentication,
            Some("PERMISSION_DENIED") => ErrorKind::Authorization,
            Some("RESOURCE_EXHAUSTED") => ErrorKind::RateLimited,
            Some("DEADLINE_EXCEEDED") => ErrorKind::Timeout,
            Some("UNAVAILABLE") => ErrorKind::Unavailable,
            Some("CANCELLED") => ErrorKind::Cancelled,
            _ => ErrorKind::Provider,
        };
        let mut diagnostics = self.diagnostics.clone();
        if let Some(status) = status.and_then(|status| PublicDiagnosticText::new(status).ok()) {
            diagnostics = diagnostics.with_provider_code(status);
        }
        let message = match kind {
            ErrorKind::InvalidInput => "Gemini rejected the streaming request",
            ErrorKind::Authentication => "Gemini authentication failed during streaming",
            ErrorKind::Authorization => "Gemini authorization failed during streaming",
            ErrorKind::RateLimited => "Gemini rate limited the streaming request",
            ErrorKind::Timeout => "Gemini streaming request timed out",
            ErrorKind::Unavailable => "Gemini was unavailable during streaming",
            ErrorKind::Cancelled => "Gemini cancelled the streaming request",
            _ => "Gemini Generate Content streaming failed",
        };
        Error::new(kind, message).with_diagnostics(diagnostics)
    }
}

impl LanguageStreamDecoder for GenerateContentStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.diagnostics = diagnostics;
    }

    fn decode(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle.ensure_decode_allowed()?;
        let events = self.decode_frame(frame)?;
        self.lifecycle.record(&events)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish()? {
            return Ok(Vec::new());
        }
        if self.finish_reason.is_none() && !self.prompt_blocked {
            return Err(Error::unexpected_eof());
        }
        let response = self.terminal_response()?;
        let events = vec![LanguageStreamEvent::Terminal(terminal_event(response))];
        self.lifecycle.record(&events)?;
        Ok(events)
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Text,
    Reasoning,
}

#[derive(Debug)]
struct OpenText {
    id: String,
    kind: TextKind,
    part_index: usize,
}

fn terminal_event(response: Result<LanguageResponse, LanguageCallError>) -> StreamTerminal {
    match response {
        Ok(response) => StreamTerminal::Completed {
            response: Box::new(response),
        },
        Err(error) if error.kind() == ErrorKind::Cancelled => {
            let (error, partial) = error.into_parts();
            StreamTerminal::Cancelled {
                reason: error.message().to_string(),
                partial,
            }
        }
        Err(error) => {
            let (error, partial) = error.into_parts();
            StreamTerminal::Failed { error, partial }
        }
    }
}

fn ensure_value_limit(value: &Value, maximum: usize) -> Result<(), Error> {
    let length = serde_json::to_vec(value)
        .map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "Gemini stream value could not be measured",
            )
            .with_source(source)
        })?
        .len();
    if length > maximum {
        Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Generate Content stream value exceeded the protocol limit",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        ApiModeId, LanguageCompletionReason, LanguageTermination, PlatformId, ProtocolId,
        ProviderId, ReplayDomain, ReplayDomainId, UsageValue,
    };

    fn decoder() -> GenerateContentStreamDecoder {
        let scope = ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new("gemini-generate-content").unwrap())
            .with_api_mode(ApiModeId::new("generate-content").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ));
        GenerateContentStreamDecoder::new(scope, ModelId::new("gemini-3.6-flash").unwrap())
    }

    #[test]
    fn finish_reason_waits_for_trailing_usage_and_eof() {
        let mut decoder = decoder();
        let frames = [
            json!({
                "responseId": "response-1",
                "modelVersion": "gemini-3.6-flash",
                "candidates": [{
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "hello "}]}
                }]
            }),
            json!({
                "responseId": "response-1",
                "candidates": [{
                    "index": 0,
                    "content": {"role": "model", "parts": [
                        {"text": "world"},
                        {"functionCall": {"name": "lookup", "args": {"key": "x"}}}
                    ]},
                    "finishReason": "STOP"
                }]
            }),
            json!({
                "responseId": "response-1",
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 6
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
            0
        );
        let terminal_events = decoder.finish().unwrap();
        assert_eq!(terminal_events.len(), 1);
        let LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) =
            &terminal_events[0]
        else {
            panic!("expected completed terminal");
        };
        assert_eq!(
            response.termination(),
            &LanguageTermination::Completed(LanguageCompletionReason::ToolCalls)
        );
        assert_eq!(response.usage().total_tokens, UsageValue::Known(6));
        assert!(
            response.content().iter().any(|part| {
                matches!(part, ContentPart::Text { text } if text == "hello world")
            })
        );
        let streamed_call = events.iter().find_map(|event| match event {
            LanguageStreamEvent::ToolCall(call) => Some(call),
            _ => None,
        });
        assert_eq!(
            streamed_call.map(|call| call.id()),
            Some("gemini-call:10:response-1:0:1")
        );
        assert_eq!(
            streamed_call.map(|call| call.arguments()),
            Some(&json!({"key": "x"}))
        );
        assert!(response.content().iter().any(|part| {
            matches!(part, ContentPart::ToolCall(call) if call.id() == "gemini-call:10:response-1:0:1")
        }));
        assert!(matches!(decoder.finish(), Err(error) if error.kind() == ErrorKind::Protocol));
    }

    #[test]
    fn clean_eof_without_finish_reason_is_unexpected_eof() {
        let mut decoder = decoder();
        decoder
            .decode(
                &json!({
                    "responseId": "response-2",
                    "candidates": [{
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "partial"}]}
                    }]
                })
                .to_string(),
            )
            .unwrap();

        assert!(matches!(
            decoder.finish(),
            Err(error) if error.kind() == ErrorKind::UnexpectedEof
        ));
    }
}
