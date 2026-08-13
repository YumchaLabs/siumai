//! OpenAI Realtime conversation wire events and stateful server decoder.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde_json::{Map, Value};

use super::{
    common::{
        DecodedRealtimeEvent, OpenAiRealtimeServerError, ParsedEvent, RealtimeCodecLimits,
        UnknownRealtimeEvent, decode_audio, encode_audio, encode_object, event_id, event_object,
        optional_string, optional_u64, parse_frame, parse_server_error, required_string,
        required_value,
    },
    error::{RealtimeCodecError, RealtimeCodecResult},
    frame::{JsonTextFrame, RealtimeInputFrame},
};

/// A client event accepted by the OpenAI Realtime conversation endpoint.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpenAiRealtimeClientEvent {
    SessionUpdate {
        event_id: Option<String>,
        session: Value,
    },
    InputAudioBufferAppend {
        event_id: Option<String>,
        audio: Bytes,
    },
    InputAudioBufferCommit {
        event_id: Option<String>,
    },
    InputAudioBufferClear {
        event_id: Option<String>,
    },
    ConversationItemCreate {
        event_id: Option<String>,
        previous_item_id: Option<String>,
        item: Value,
    },
    ConversationItemRetrieve {
        event_id: Option<String>,
        item_id: String,
    },
    ConversationItemTruncate {
        event_id: Option<String>,
        item_id: String,
        content_index: u64,
        audio_end_ms: u64,
    },
    ConversationItemDelete {
        event_id: Option<String>,
        item_id: String,
    },
    ResponseCreate {
        event_id: Option<String>,
        response: Option<Value>,
    },
    ResponseCancel {
        event_id: Option<String>,
        response_id: Option<String>,
    },
    OutputAudioBufferClear {
        event_id: Option<String>,
    },
    /// A forward-compatible client event not yet modeled by this crate.
    Custom {
        event_type: String,
        event_id: Option<String>,
        fields: Map<String, Value>,
    },
}

impl OpenAiRealtimeClientEvent {
    pub fn event_type(&self) -> &str {
        match self {
            Self::SessionUpdate { .. } => "session.update",
            Self::InputAudioBufferAppend { .. } => "input_audio_buffer.append",
            Self::InputAudioBufferCommit { .. } => "input_audio_buffer.commit",
            Self::InputAudioBufferClear { .. } => "input_audio_buffer.clear",
            Self::ConversationItemCreate { .. } => "conversation.item.create",
            Self::ConversationItemRetrieve { .. } => "conversation.item.retrieve",
            Self::ConversationItemTruncate { .. } => "conversation.item.truncate",
            Self::ConversationItemDelete { .. } => "conversation.item.delete",
            Self::ResponseCreate { .. } => "response.create",
            Self::ResponseCancel { .. } => "response.cancel",
            Self::OutputAudioBufferClear { .. } => "output_audio_buffer.clear",
            Self::Custom { event_type, .. } => event_type.as_str(),
        }
    }

    /// Serializes the event as the JSON text frame required by OpenAI.
    pub fn encode(&self) -> RealtimeCodecResult<JsonTextFrame> {
        let object = match self {
            Self::SessionUpdate { event_id, session } => {
                let mut object = event_object("session.update", event_id);
                object.insert("session".to_owned(), session.clone());
                object
            }
            Self::InputAudioBufferAppend { event_id, audio } => {
                let mut object = event_object("input_audio_buffer.append", event_id);
                object.insert("audio".to_owned(), Value::String(encode_audio(audio)));
                object
            }
            Self::InputAudioBufferCommit { event_id } => {
                event_object("input_audio_buffer.commit", event_id)
            }
            Self::InputAudioBufferClear { event_id } => {
                event_object("input_audio_buffer.clear", event_id)
            }
            Self::ConversationItemCreate {
                event_id,
                previous_item_id,
                item,
            } => {
                let mut object = event_object("conversation.item.create", event_id);
                if let Some(previous_item_id) = previous_item_id {
                    object.insert(
                        "previous_item_id".to_owned(),
                        Value::String(previous_item_id.clone()),
                    );
                }
                object.insert("item".to_owned(), item.clone());
                object
            }
            Self::ConversationItemRetrieve { event_id, item_id } => {
                let mut object = event_object("conversation.item.retrieve", event_id);
                object.insert("item_id".to_owned(), Value::String(item_id.clone()));
                object
            }
            Self::ConversationItemTruncate {
                event_id,
                item_id,
                content_index,
                audio_end_ms,
            } => {
                let mut object = event_object("conversation.item.truncate", event_id);
                object.insert("item_id".to_owned(), Value::String(item_id.clone()));
                object.insert("content_index".to_owned(), Value::from(*content_index));
                object.insert("audio_end_ms".to_owned(), Value::from(*audio_end_ms));
                object
            }
            Self::ConversationItemDelete { event_id, item_id } => {
                let mut object = event_object("conversation.item.delete", event_id);
                object.insert("item_id".to_owned(), Value::String(item_id.clone()));
                object
            }
            Self::ResponseCreate { event_id, response } => {
                let mut object = event_object("response.create", event_id);
                if let Some(response) = response {
                    object.insert("response".to_owned(), response.clone());
                }
                object
            }
            Self::ResponseCancel {
                event_id,
                response_id,
            } => {
                let mut object = event_object("response.cancel", event_id);
                if let Some(response_id) = response_id {
                    object.insert("response_id".to_owned(), Value::String(response_id.clone()));
                }
                object
            }
            Self::OutputAudioBufferClear { event_id } => {
                event_object("output_audio_buffer.clear", event_id)
            }
            Self::Custom {
                event_type,
                event_id,
                fields,
            } => {
                if event_type.is_empty() {
                    return Err(RealtimeCodecError::MissingEventType);
                }
                let mut object = fields.clone();
                object.insert("type".to_owned(), Value::String(event_type.clone()));
                if let Some(event_id) = event_id {
                    object.insert("event_id".to_owned(), Value::String(event_id.clone()));
                } else {
                    object.remove("event_id");
                }
                object
            }
        };

        encode_object(object)
    }
}

/// The lifecycle status carried by a `response.done` event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealtimeResponseStatus {
    Completed,
    Cancelled,
    Failed,
    Incomplete,
    Other(String),
}

impl RealtimeResponseStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Incomplete => "incomplete",
            Self::Other(value) => value.as_str(),
        }
    }
}

impl From<String> for RealtimeResponseStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "completed" => Self::Completed,
            "cancelled" => Self::Cancelled,
            "failed" => Self::Failed,
            "incomplete" => Self::Incomplete,
            _ => Self::Other(value),
        }
    }
}

/// A function-call argument delta after it has been accepted into bounded state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallArgumentsDeltaEvent {
    pub event_id: Option<String>,
    pub response_id: String,
    pub item_id: String,
    pub output_index: Option<u64>,
    pub call_id: String,
    pub delta: String,
    pub accumulated_bytes: usize,
}

/// A completed function-call argument payload and its accumulated delta stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallArgumentsDoneEvent {
    pub event_id: Option<String>,
    pub response_id: String,
    pub item_id: String,
    pub output_index: Option<u64>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub accumulated_delta: Option<String>,
}

impl FunctionCallArgumentsDoneEvent {
    /// Whether the accumulated deltas exactly match OpenAI's final arguments.
    pub fn deltas_match_final_arguments(&self) -> Option<bool> {
        self.accumulated_delta
            .as_ref()
            .map(|delta| delta == &self.arguments)
    }
}

/// A partial function call drained when its enclosing response finishes first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteFunctionCall {
    pub response_id: String,
    pub item_id: String,
    pub call_id: String,
    pub arguments_delta: String,
}

/// The terminal event for one response, not for the Realtime session.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeResponseDoneEvent {
    pub event_id: Option<String>,
    pub response_id: String,
    pub status: RealtimeResponseStatus,
    pub response: Value,
    pub incomplete_function_calls: Vec<IncompleteFunctionCall>,
}

impl RealtimeResponseDoneEvent {
    /// `response.done` ends only this response; more session events may follow.
    pub const fn is_session_terminal(&self) -> bool {
        false
    }
}

/// A typed OpenAI Realtime server event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpenAiRealtimeServerEvent {
    SessionCreated {
        event_id: Option<String>,
        session: Value,
    },
    SessionUpdated {
        event_id: Option<String>,
        session: Value,
    },
    InputAudioBufferSpeechStarted {
        event_id: Option<String>,
        item_id: String,
        audio_start_ms: Option<u64>,
    },
    InputAudioBufferSpeechStopped {
        event_id: Option<String>,
        item_id: String,
        audio_end_ms: Option<u64>,
    },
    InputAudioBufferCommitted {
        event_id: Option<String>,
        item_id: String,
        previous_item_id: Option<String>,
    },
    InputAudioBufferCleared {
        event_id: Option<String>,
    },
    ConversationItemCreated {
        event_id: Option<String>,
        previous_item_id: Option<String>,
        item: Value,
    },
    ConversationItemRetrieved {
        event_id: Option<String>,
        item: Value,
    },
    ConversationItemDone {
        event_id: Option<String>,
        previous_item_id: Option<String>,
        item: Value,
    },
    ConversationItemTruncated {
        event_id: Option<String>,
        item_id: String,
        content_index: Option<u64>,
        audio_end_ms: Option<u64>,
    },
    ConversationItemDeleted {
        event_id: Option<String>,
        item_id: String,
    },
    InputAudioTranscriptionDelta {
        event_id: Option<String>,
        item_id: String,
        content_index: Option<u64>,
        delta: String,
    },
    InputAudioTranscriptionCompleted {
        event_id: Option<String>,
        item_id: String,
        content_index: Option<u64>,
        transcript: String,
    },
    InputAudioTranscriptionFailed {
        event_id: Option<String>,
        item_id: String,
        content_index: Option<u64>,
        error: Value,
    },
    ResponseCreated {
        event_id: Option<String>,
        response: Value,
    },
    ResponseDone(RealtimeResponseDoneEvent),
    ResponseOutputItemAdded {
        event_id: Option<String>,
        response_id: String,
        output_index: Option<u64>,
        item: Value,
    },
    ResponseOutputItemDone {
        event_id: Option<String>,
        response_id: String,
        output_index: Option<u64>,
        item: Value,
    },
    ResponseContentPartAdded {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        part: Value,
    },
    ResponseContentPartDone {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        part: Value,
    },
    ResponseOutputTextDelta {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        delta: String,
    },
    ResponseOutputTextDone {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        text: String,
    },
    ResponseOutputAudioTranscriptDelta {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        delta: String,
    },
    ResponseOutputAudioTranscriptDone {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        transcript: String,
    },
    ResponseOutputAudioDelta {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
        audio: Bytes,
    },
    ResponseOutputAudioDone {
        event_id: Option<String>,
        response_id: String,
        item_id: String,
        output_index: Option<u64>,
        content_index: Option<u64>,
    },
    ResponseFunctionCallArgumentsDelta(FunctionCallArgumentsDeltaEvent),
    ResponseFunctionCallArgumentsDone(FunctionCallArgumentsDoneEvent),
    Error(OpenAiRealtimeServerError),
    Unknown(UnknownRealtimeEvent),
}

impl OpenAiRealtimeServerEvent {
    /// Whether this event ends the WebSocket session.
    ///
    /// Neither `response.done` nor OpenAI `error` events are session-terminal.
    pub const fn is_session_terminal(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
struct PendingFunctionCall {
    response_id: String,
    item_id: String,
    arguments: String,
}

/// Stateful decoder for OpenAI Realtime conversation server events.
///
/// The decoder retains only bounded function-call argument deltas. It never
/// treats `response.done` or a server `error` event as a session terminal.
#[derive(Debug, Clone, Default)]
pub struct OpenAiRealtimeDecoder {
    limits: RealtimeCodecLimits,
    pending_function_calls: BTreeMap<String, PendingFunctionCall>,
    pending_argument_bytes: usize,
}

impl OpenAiRealtimeDecoder {
    /// Creates a decoder after validating that every resource limit is bounded.
    pub fn new(limits: RealtimeCodecLimits) -> RealtimeCodecResult<Self> {
        Ok(Self {
            limits: limits.validate()?,
            pending_function_calls: BTreeMap::new(),
            pending_argument_bytes: 0,
        })
    }

    pub fn limits(&self) -> RealtimeCodecLimits {
        self.limits
    }

    pub fn pending_function_calls(&self) -> usize {
        self.pending_function_calls.len()
    }

    pub fn pending_function_argument_bytes(&self) -> usize {
        self.pending_argument_bytes
    }

    /// Decodes one server frame while updating bounded function-call state.
    pub fn decode(
        &mut self,
        frame: RealtimeInputFrame<'_>,
    ) -> RealtimeCodecResult<DecodedRealtimeEvent<OpenAiRealtimeServerEvent>> {
        let parsed = parse_frame(frame, self.limits)?;
        let event = self.decode_parsed(&parsed)?;
        Ok(parsed.into_decoded(event))
    }

    fn decode_parsed(
        &mut self,
        parsed: &ParsedEvent,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        let event_type = parsed.event_type.as_str();
        let object = parsed.object();

        match event_type {
            "session.created" => Ok(OpenAiRealtimeServerEvent::SessionCreated {
                event_id: event_id(object, event_type)?,
                session: required_value(object, event_type, "session")?,
            }),
            "session.updated" => Ok(OpenAiRealtimeServerEvent::SessionUpdated {
                event_id: event_id(object, event_type)?,
                session: required_value(object, event_type, "session")?,
            }),
            "input_audio_buffer.speech_started" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioBufferSpeechStarted {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    audio_start_ms: optional_u64(object, event_type, "audio_start_ms")?,
                })
            }
            "input_audio_buffer.speech_stopped" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioBufferSpeechStopped {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    audio_end_ms: optional_u64(object, event_type, "audio_end_ms")?,
                })
            }
            "input_audio_buffer.committed" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioBufferCommitted {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    previous_item_id: optional_string(object, event_type, "previous_item_id")?,
                })
            }
            "input_audio_buffer.cleared" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioBufferCleared {
                    event_id: event_id(object, event_type)?,
                })
            }
            "conversation.item.created" | "conversation.item.added" => {
                Ok(OpenAiRealtimeServerEvent::ConversationItemCreated {
                    event_id: event_id(object, event_type)?,
                    previous_item_id: optional_string(object, event_type, "previous_item_id")?,
                    item: required_value(object, event_type, "item")?,
                })
            }
            "conversation.item.retrieved" => {
                Ok(OpenAiRealtimeServerEvent::ConversationItemRetrieved {
                    event_id: event_id(object, event_type)?,
                    item: required_value(object, event_type, "item")?,
                })
            }
            "conversation.item.done" => Ok(OpenAiRealtimeServerEvent::ConversationItemDone {
                event_id: event_id(object, event_type)?,
                previous_item_id: optional_string(object, event_type, "previous_item_id")?,
                item: required_value(object, event_type, "item")?,
            }),
            "conversation.item.truncated" => {
                Ok(OpenAiRealtimeServerEvent::ConversationItemTruncated {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    content_index: optional_u64(object, event_type, "content_index")?,
                    audio_end_ms: optional_u64(object, event_type, "audio_end_ms")?,
                })
            }
            "conversation.item.deleted" => Ok(OpenAiRealtimeServerEvent::ConversationItemDeleted {
                event_id: event_id(object, event_type)?,
                item_id: required_string(object, event_type, "item_id")?,
            }),
            "conversation.item.input_audio_transcription.delta" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioTranscriptionDelta {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    content_index: optional_u64(object, event_type, "content_index")?,
                    delta: required_string(object, event_type, "delta")?,
                })
            }
            "conversation.item.input_audio_transcription.completed" => Ok(
                OpenAiRealtimeServerEvent::InputAudioTranscriptionCompleted {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    content_index: optional_u64(object, event_type, "content_index")?,
                    transcript: required_string(object, event_type, "transcript")?,
                },
            ),
            "conversation.item.input_audio_transcription.failed" => {
                Ok(OpenAiRealtimeServerEvent::InputAudioTranscriptionFailed {
                    event_id: event_id(object, event_type)?,
                    item_id: required_string(object, event_type, "item_id")?,
                    content_index: optional_u64(object, event_type, "content_index")?,
                    error: required_value(object, event_type, "error")?,
                })
            }
            "response.created" => Ok(OpenAiRealtimeServerEvent::ResponseCreated {
                event_id: event_id(object, event_type)?,
                response: required_value(object, event_type, "response")?,
            }),
            "response.done" => self.decode_response_done(object, event_type),
            "response.output_item.added" => {
                Ok(OpenAiRealtimeServerEvent::ResponseOutputItemAdded {
                    event_id: event_id(object, event_type)?,
                    response_id: required_string(object, event_type, "response_id")?,
                    output_index: optional_u64(object, event_type, "output_index")?,
                    item: required_value(object, event_type, "item")?,
                })
            }
            "response.output_item.done" => Ok(OpenAiRealtimeServerEvent::ResponseOutputItemDone {
                event_id: event_id(object, event_type)?,
                response_id: required_string(object, event_type, "response_id")?,
                output_index: optional_u64(object, event_type, "output_index")?,
                item: required_value(object, event_type, "item")?,
            }),
            "response.content_part.added" => self.decode_content_part(object, event_type, true),
            "response.content_part.done" => self.decode_content_part(object, event_type, false),
            "response.output_text.delta" => self.decode_text_delta(object, event_type),
            "response.output_text.done" => self.decode_text_done(object, event_type),
            "response.output_audio_transcript.delta" | "response.audio_transcript.delta" => {
                self.decode_audio_transcript_delta(object, event_type)
            }
            "response.output_audio_transcript.done" | "response.audio_transcript.done" => {
                self.decode_audio_transcript_done(object, event_type)
            }
            "response.output_audio.delta" | "response.audio.delta" => {
                self.decode_output_audio_delta(object, event_type)
            }
            "response.output_audio.done" | "response.audio.done" => {
                self.decode_output_audio_done(object, event_type)
            }
            "response.function_call_arguments.delta" => {
                self.decode_function_arguments_delta(object, event_type)
            }
            "response.function_call_arguments.done" => {
                self.decode_function_arguments_done(object, event_type)
            }
            "error" => Ok(OpenAiRealtimeServerEvent::Error(parse_server_error(
                object, event_type,
            )?)),
            _ => Ok(OpenAiRealtimeServerEvent::Unknown(UnknownRealtimeEvent {
                event_type: parsed.event_type.clone(),
            })),
        }
    }

    fn decode_response_done(
        &mut self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        let response = required_value(object, event_type, "response")?;
        let response_object =
            response
                .as_object()
                .ok_or_else(|| RealtimeCodecError::InvalidField {
                    event_type: event_type.to_owned(),
                    field: "response",
                    reason: "expected an object".to_owned(),
                })?;
        let response_id = response_object
            .get("id")
            .and_then(Value::as_str)
            .or_else(|| object.get("response_id").and_then(Value::as_str))
            .ok_or_else(|| RealtimeCodecError::MissingField {
                event_type: event_type.to_owned(),
                field: "response.id",
            })?
            .to_owned();
        let status = response_object
            .get("status")
            .and_then(Value::as_str)
            .or_else(|| object.get("status").and_then(Value::as_str))
            .ok_or_else(|| RealtimeCodecError::MissingField {
                event_type: event_type.to_owned(),
                field: "response.status",
            })?
            .to_owned()
            .into();
        let incomplete_function_calls = self.take_incomplete_calls(&response_id);

        Ok(OpenAiRealtimeServerEvent::ResponseDone(
            RealtimeResponseDoneEvent {
                event_id: event_id(object, event_type)?,
                response_id,
                status,
                response,
                incomplete_function_calls,
            },
        ))
    }

    fn decode_content_part(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
        added: bool,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        let event_id = event_id(object, event_type)?;
        let response_id = required_string(object, event_type, "response_id")?;
        let item_id = required_string(object, event_type, "item_id")?;
        let output_index = optional_u64(object, event_type, "output_index")?;
        let content_index = optional_u64(object, event_type, "content_index")?;
        let part = required_value(object, event_type, "part")?;

        if added {
            Ok(OpenAiRealtimeServerEvent::ResponseContentPartAdded {
                event_id,
                response_id,
                item_id,
                output_index,
                content_index,
                part,
            })
        } else {
            Ok(OpenAiRealtimeServerEvent::ResponseContentPartDone {
                event_id,
                response_id,
                item_id,
                output_index,
                content_index,
                part,
            })
        }
    }

    fn decode_text_delta(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(OpenAiRealtimeServerEvent::ResponseOutputTextDelta {
            event_id: event_id(object, event_type)?,
            response_id: required_string(object, event_type, "response_id")?,
            item_id: required_string(object, event_type, "item_id")?,
            output_index: optional_u64(object, event_type, "output_index")?,
            content_index: optional_u64(object, event_type, "content_index")?,
            delta: required_string(object, event_type, "delta")?,
        })
    }

    fn decode_text_done(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(OpenAiRealtimeServerEvent::ResponseOutputTextDone {
            event_id: event_id(object, event_type)?,
            response_id: required_string(object, event_type, "response_id")?,
            item_id: required_string(object, event_type, "item_id")?,
            output_index: optional_u64(object, event_type, "output_index")?,
            content_index: optional_u64(object, event_type, "content_index")?,
            text: required_string(object, event_type, "text")?,
        })
    }

    fn decode_audio_transcript_delta(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(
            OpenAiRealtimeServerEvent::ResponseOutputAudioTranscriptDelta {
                event_id: event_id(object, event_type)?,
                response_id: required_string(object, event_type, "response_id")?,
                item_id: required_string(object, event_type, "item_id")?,
                output_index: optional_u64(object, event_type, "output_index")?,
                content_index: optional_u64(object, event_type, "content_index")?,
                delta: required_string(object, event_type, "delta")?,
            },
        )
    }

    fn decode_audio_transcript_done(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(
            OpenAiRealtimeServerEvent::ResponseOutputAudioTranscriptDone {
                event_id: event_id(object, event_type)?,
                response_id: required_string(object, event_type, "response_id")?,
                item_id: required_string(object, event_type, "item_id")?,
                output_index: optional_u64(object, event_type, "output_index")?,
                content_index: optional_u64(object, event_type, "content_index")?,
                transcript: required_string(object, event_type, "transcript")?,
            },
        )
    }

    fn decode_output_audio_delta(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(OpenAiRealtimeServerEvent::ResponseOutputAudioDelta {
            event_id: event_id(object, event_type)?,
            response_id: required_string(object, event_type, "response_id")?,
            item_id: required_string(object, event_type, "item_id")?,
            output_index: optional_u64(object, event_type, "output_index")?,
            content_index: optional_u64(object, event_type, "content_index")?,
            audio: decode_audio(object, event_type, "delta", self.limits)?,
        })
    }

    fn decode_output_audio_done(
        &self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        Ok(OpenAiRealtimeServerEvent::ResponseOutputAudioDone {
            event_id: event_id(object, event_type)?,
            response_id: required_string(object, event_type, "response_id")?,
            item_id: required_string(object, event_type, "item_id")?,
            output_index: optional_u64(object, event_type, "output_index")?,
            content_index: optional_u64(object, event_type, "content_index")?,
        })
    }

    fn decode_function_arguments_delta(
        &mut self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        let event_id = event_id(object, event_type)?;
        let response_id = required_string(object, event_type, "response_id")?;
        let item_id = required_string(object, event_type, "item_id")?;
        let output_index = optional_u64(object, event_type, "output_index")?;
        let call_id = required_string(object, event_type, "call_id")?;
        let delta = required_string(object, event_type, "delta")?;
        let accumulated_bytes =
            self.accumulate_function_delta(&response_id, &item_id, &call_id, &delta)?;

        Ok(
            OpenAiRealtimeServerEvent::ResponseFunctionCallArgumentsDelta(
                FunctionCallArgumentsDeltaEvent {
                    event_id,
                    response_id,
                    item_id,
                    output_index,
                    call_id,
                    delta,
                    accumulated_bytes,
                },
            ),
        )
    }

    fn decode_function_arguments_done(
        &mut self,
        object: &Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<OpenAiRealtimeServerEvent> {
        let event_id = event_id(object, event_type)?;
        let response_id = required_string(object, event_type, "response_id")?;
        let item_id = required_string(object, event_type, "item_id")?;
        let output_index = optional_u64(object, event_type, "output_index")?;
        let call_id = required_string(object, event_type, "call_id")?;
        let name = required_string(object, event_type, "name")?;
        let arguments = required_string(object, event_type, "arguments")?;
        let accumulated_delta = self
            .take_completed_call(&response_id, &item_id, &call_id, event_type)?
            .map(|pending| pending.arguments);

        Ok(
            OpenAiRealtimeServerEvent::ResponseFunctionCallArgumentsDone(
                FunctionCallArgumentsDoneEvent {
                    event_id,
                    response_id,
                    item_id,
                    output_index,
                    call_id,
                    name,
                    arguments,
                    accumulated_delta,
                },
            ),
        )
    }

    fn accumulate_function_delta(
        &mut self,
        response_id: &str,
        item_id: &str,
        call_id: &str,
        delta: &str,
    ) -> RealtimeCodecResult<usize> {
        if let Some(pending) = self.pending_function_calls.get(call_id) {
            if pending.response_id != response_id || pending.item_id != item_id {
                return Err(RealtimeCodecError::InvalidField {
                    event_type: "response.function_call_arguments.delta".to_owned(),
                    field: "call_id",
                    reason: "the call ID was already associated with a different response or item"
                        .to_owned(),
                });
            }
        } else if self.pending_function_calls.len() >= self.limits.max_active_function_calls {
            return Err(RealtimeCodecError::ActiveFunctionCallLimitExceeded {
                maximum: self.limits.max_active_function_calls,
            });
        }

        let current_bytes = self
            .pending_function_calls
            .get(call_id)
            .map_or(0, |pending| pending.arguments.len());
        let accumulated_bytes = current_bytes.checked_add(delta.len()).ok_or_else(|| {
            RealtimeCodecError::FunctionArgumentsTooLarge {
                call_id: call_id.to_owned(),
                actual: usize::MAX,
                maximum: self.limits.max_function_arguments_bytes_per_call,
            }
        })?;
        if accumulated_bytes > self.limits.max_function_arguments_bytes_per_call {
            return Err(RealtimeCodecError::FunctionArgumentsTooLarge {
                call_id: call_id.to_owned(),
                actual: accumulated_bytes,
                maximum: self.limits.max_function_arguments_bytes_per_call,
            });
        }

        let total_bytes = self.pending_argument_bytes.checked_add(delta.len()).ok_or(
            RealtimeCodecError::TotalFunctionArgumentsTooLarge {
                actual: usize::MAX,
                maximum: self.limits.max_function_arguments_bytes_total,
            },
        )?;
        if total_bytes > self.limits.max_function_arguments_bytes_total {
            return Err(RealtimeCodecError::TotalFunctionArgumentsTooLarge {
                actual: total_bytes,
                maximum: self.limits.max_function_arguments_bytes_total,
            });
        }

        let pending = self
            .pending_function_calls
            .entry(call_id.to_owned())
            .or_insert_with(|| PendingFunctionCall {
                response_id: response_id.to_owned(),
                item_id: item_id.to_owned(),
                arguments: String::new(),
            });
        pending.arguments.push_str(delta);
        self.pending_argument_bytes = total_bytes;
        Ok(accumulated_bytes)
    }

    fn remove_pending_call(&mut self, call_id: &str) -> Option<PendingFunctionCall> {
        let pending = self.pending_function_calls.remove(call_id)?;
        self.pending_argument_bytes = self
            .pending_argument_bytes
            .saturating_sub(pending.arguments.len());
        Some(pending)
    }

    fn take_completed_call(
        &mut self,
        response_id: &str,
        item_id: &str,
        call_id: &str,
        event_type: &str,
    ) -> RealtimeCodecResult<Option<PendingFunctionCall>> {
        if let Some(pending) = self.pending_function_calls.get(call_id)
            && (pending.response_id != response_id || pending.item_id != item_id)
        {
            return Err(RealtimeCodecError::InvalidField {
                event_type: event_type.to_owned(),
                field: "call_id",
                reason: "the call ID was accumulated for a different response or item".to_owned(),
            });
        }

        Ok(self.remove_pending_call(call_id))
    }

    fn take_incomplete_calls(&mut self, response_id: &str) -> Vec<IncompleteFunctionCall> {
        let call_ids = self
            .pending_function_calls
            .iter()
            .filter(|(_, pending)| pending.response_id == response_id)
            .map(|(call_id, _)| call_id.clone())
            .collect::<Vec<_>>();

        call_ids
            .into_iter()
            .filter_map(|call_id| {
                self.remove_pending_call(&call_id)
                    .map(|pending| IncompleteFunctionCall {
                        response_id: pending.response_id,
                        item_id: pending.item_id,
                        call_id,
                        arguments_delta: pending.arguments,
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn encodes_audio_as_base64_json_text() {
        let event = OpenAiRealtimeClientEvent::InputAudioBufferAppend {
            event_id: Some("evt_1".to_owned()),
            audio: Bytes::from_static(&[0, 1, 2, 255]),
        };

        let frame = event.encode().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(frame.as_str()).unwrap(),
            json!({
                "type": "input_audio_buffer.append",
                "event_id": "evt_1",
                "audio": "AAEC/w=="
            })
        );
    }

    #[test]
    fn decodes_audio_delta_into_bytes() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        let decoded = decoder
            .decode(
                r#"{"type":"response.output_audio.delta","response_id":"resp_1","item_id":"item_1","delta":"AAEC/w=="}"#
                    .into(),
            )
            .unwrap();

        match decoded.event {
            OpenAiRealtimeServerEvent::ResponseOutputAudioDelta { audio, .. } => {
                assert_eq!(audio, Bytes::from_static(&[0, 1, 2, 255]));
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn preserves_unknown_event_exactly() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        let source = r#"{ "type": "future.event", "large": 1e3, "new": {"x":true} }"#;
        let decoded = decoder.decode(source.into()).unwrap();

        assert_eq!(decoded.raw_json, source);
        assert_eq!(decoded.raw["large"], json!(1000.0));
        assert!(matches!(
            decoded.event,
            OpenAiRealtimeServerEvent::Unknown(UnknownRealtimeEvent { ref event_type })
                if event_type == "future.event"
        ));
    }

    #[test]
    fn server_error_is_recoverable_data() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        let error = decoder
            .decode(
                r#"{"type":"error","event_id":"evt_error","error":{"type":"invalid_request_error","code":"bad_audio","message":"Bad audio","event_id":"evt_input"}}"#
                    .into(),
            )
            .unwrap();

        match error.event {
            OpenAiRealtimeServerEvent::Error(error) => {
                assert_eq!(error.message, "Bad audio");
                assert_eq!(error.related_event_id.as_deref(), Some("evt_input"));
                assert!(!error.is_session_terminal());
            }
            event => panic!("unexpected event: {event:?}"),
        }

        let next = decoder
            .decode(r#"{"type":"session.updated","session":{"id":"sess_1"}}"#.into())
            .unwrap();
        assert!(matches!(
            next.event,
            OpenAiRealtimeServerEvent::SessionUpdated { .. }
        ));
    }

    #[test]
    fn response_done_drains_only_its_calls_and_does_not_end_session() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"{\"city\":"}"#
                    .into(),
            )
            .unwrap();
        decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"\"Paris\"}"}"#
                    .into(),
            )
            .unwrap();

        let done = decoder
            .decode(
                r#"{"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#
                    .into(),
            )
            .unwrap();
        match done.event {
            OpenAiRealtimeServerEvent::ResponseDone(done) => {
                assert_eq!(done.incomplete_function_calls.len(), 1);
                assert_eq!(
                    done.incomplete_function_calls[0].arguments_delta,
                    r#"{"city":"Paris"}"#
                );
                assert!(!done.is_session_terminal());
            }
            event => panic!("unexpected event: {event:?}"),
        }
        assert_eq!(decoder.pending_function_calls(), 0);

        let next = decoder
            .decode(r#"{"type":"response.created","response":{"id":"resp_2"}}"#.into())
            .unwrap();
        assert!(matches!(
            next.event,
            OpenAiRealtimeServerEvent::ResponseCreated { .. }
        ));
    }

    #[test]
    fn function_argument_accumulation_is_bounded_without_mutating_on_error() {
        let limits = RealtimeCodecLimits {
            max_function_arguments_bytes_per_call: 4,
            max_function_arguments_bytes_total: 4,
            ..RealtimeCodecLimits::default()
        };
        let mut decoder = OpenAiRealtimeDecoder::new(limits).unwrap();
        decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"1234"}"#
                    .into(),
            )
            .unwrap();

        let error = decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"5"}"#
                    .into(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            RealtimeCodecError::FunctionArgumentsTooLarge { actual: 5, .. }
        ));
        assert_eq!(decoder.pending_function_argument_bytes(), 4);
    }

    #[test]
    fn function_arguments_done_returns_and_releases_accumulated_deltas() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"{\"x\":"}"#
                    .into(),
            )
            .unwrap();
        decoder
            .decode(
                r#"{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"item_1","call_id":"call_1","delta":"1}"}"#
                    .into(),
            )
            .unwrap();

        let done = decoder
            .decode(
                r#"{"type":"response.function_call_arguments.done","response_id":"resp_1","item_id":"item_1","call_id":"call_1","name":"lookup","arguments":"{\"x\":1}"}"#
                    .into(),
            )
            .unwrap();
        match done.event {
            OpenAiRealtimeServerEvent::ResponseFunctionCallArgumentsDone(done) => {
                assert_eq!(done.accumulated_delta.as_deref(), Some(r#"{"x":1}"#));
                assert_eq!(done.deltas_match_final_arguments(), Some(true));
            }
            event => panic!("unexpected event: {event:?}"),
        }
        assert_eq!(decoder.pending_function_calls(), 0);
        assert_eq!(decoder.pending_function_argument_bytes(), 0);
    }

    #[test]
    fn decoded_audio_is_bounded() {
        let limits = RealtimeCodecLimits {
            max_decoded_audio_bytes_per_event: 2,
            ..RealtimeCodecLimits::default()
        };
        let mut decoder = OpenAiRealtimeDecoder::new(limits).unwrap();
        let error = decoder
            .decode(
                r#"{"type":"response.output_audio.delta","response_id":"resp_1","item_id":"item_1","delta":"AQID"}"#
                    .into(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            RealtimeCodecError::DecodedAudioTooLarge {
                actual: 3,
                maximum: 2,
                ..
            }
        ));
    }

    #[test]
    fn rejects_raw_binary_websocket_frames() {
        let mut decoder = OpenAiRealtimeDecoder::default();
        let error = decoder
            .decode(RealtimeInputFrame::Binary(&[1, 2, 3]))
            .unwrap_err();
        assert_eq!(
            error,
            RealtimeCodecError::RawBinaryFrameUnsupported { bytes: 3 }
        );
    }
}
