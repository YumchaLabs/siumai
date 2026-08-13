//! OpenAI Realtime Translation wire codec and close-state machine.

use bytes::Bytes;
use serde_json::Value;

use super::{
    common::{
        DecodedRealtimeEvent, OpenAiRealtimeServerError, RealtimeCodecLimits, UnknownRealtimeEvent,
        decode_audio, encode_audio, encode_object, event_id, event_object, optional_string,
        optional_u64, parse_frame, parse_server_error, required_string, required_value,
    },
    error::{RealtimeCodecError, RealtimeCodecResult},
    frame::{JsonTextFrame, RealtimeInputFrame},
};

/// A client event accepted by the OpenAI Realtime Translation endpoint.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpenAiTranslationClientEvent {
    SessionUpdate {
        event_id: Option<String>,
        session: Value,
    },
    InputAudioBufferAppend {
        event_id: Option<String>,
        audio: Bytes,
    },
    /// Flushes pending input and requests the server's terminal `session.closed`.
    SessionClose { event_id: Option<String> },
}

impl OpenAiTranslationClientEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::SessionUpdate { .. } => "session.update",
            Self::InputAudioBufferAppend { .. } => "session.input_audio_buffer.append",
            Self::SessionClose { .. } => "session.close",
        }
    }
}

/// Observable lifecycle state of an OpenAI Realtime Translation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranslationSessionState {
    AwaitingSessionCreated,
    Active,
    CloseRequested,
    Closed,
}

impl TranslationSessionState {
    fn label(self) -> &'static str {
        match self {
            Self::AwaitingSessionCreated => "awaiting session.created",
            Self::Active => "active",
            Self::CloseRequested => "waiting for session.closed",
            Self::Closed => "closed",
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// Decoded audio from a `session.output_audio.delta` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationAudioDeltaEvent {
    pub event_id: Option<String>,
    pub audio: Bytes,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub format: String,
    pub elapsed_ms: Option<u64>,
}

/// A typed server event from the OpenAI Realtime Translation endpoint.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OpenAiTranslationServerEvent {
    SessionCreated {
        event_id: Option<String>,
        session: Value,
    },
    SessionUpdated {
        event_id: Option<String>,
        session: Value,
    },
    SessionClosed {
        event_id: Option<String>,
        session: Option<Value>,
        close_was_requested: bool,
    },
    InputTranscriptDelta {
        event_id: Option<String>,
        delta: String,
        elapsed_ms: Option<u64>,
    },
    OutputTranscriptDelta {
        event_id: Option<String>,
        delta: String,
        elapsed_ms: Option<u64>,
    },
    OutputAudioDelta(TranslationAudioDeltaEvent),
    Error(OpenAiRealtimeServerError),
    Unknown(UnknownRealtimeEvent),
}

impl OpenAiTranslationServerEvent {
    /// Only `session.closed` terminates a Translation session.
    pub const fn is_session_terminal(&self) -> bool {
        matches!(self, Self::SessionClosed { .. })
    }
}

/// Bidirectional codec for an OpenAI Realtime Translation session.
///
/// Sending `session.close` is a half-close: the codec rejects further client
/// input but continues to accept output until the server emits
/// `session.closed`. A transport close before that event is an error.
#[derive(Debug, Clone, Default)]
pub struct OpenAiTranslationCodec {
    limits: RealtimeCodecLimits,
    session_created: bool,
    close_requested: bool,
    session_closed: bool,
}

impl OpenAiTranslationCodec {
    pub fn new(limits: RealtimeCodecLimits) -> RealtimeCodecResult<Self> {
        Ok(Self {
            limits: limits.validate()?,
            session_created: false,
            close_requested: false,
            session_closed: false,
        })
    }

    pub fn limits(&self) -> RealtimeCodecLimits {
        self.limits
    }

    pub fn state(&self) -> TranslationSessionState {
        if self.session_closed {
            TranslationSessionState::Closed
        } else if self.close_requested {
            TranslationSessionState::CloseRequested
        } else if self.session_created {
            TranslationSessionState::Active
        } else {
            TranslationSessionState::AwaitingSessionCreated
        }
    }

    /// Encodes one client event and advances the client half-close state.
    pub fn encode(
        &mut self,
        event: &OpenAiTranslationClientEvent,
    ) -> RealtimeCodecResult<JsonTextFrame> {
        let state = self.state();
        if self.session_closed || self.close_requested {
            return Err(RealtimeCodecError::TranslationClientStateViolation {
                event_type: event.event_type().to_owned(),
                state: state.label(),
            });
        }

        let object = match event {
            OpenAiTranslationClientEvent::SessionUpdate { event_id, session } => {
                let mut object = event_object("session.update", event_id);
                object.insert("session".to_owned(), session.clone());
                object
            }
            OpenAiTranslationClientEvent::InputAudioBufferAppend { event_id, audio } => {
                let mut object = event_object("session.input_audio_buffer.append", event_id);
                object.insert("audio".to_owned(), Value::String(encode_audio(audio)));
                object
            }
            OpenAiTranslationClientEvent::SessionClose { event_id } => {
                event_object("session.close", event_id)
            }
        };

        let frame = encode_object(object)?;
        if matches!(event, OpenAiTranslationClientEvent::SessionClose { .. }) {
            self.close_requested = true;
        }
        Ok(frame)
    }

    /// Decodes one server frame and advances the server lifecycle state.
    pub fn decode(
        &mut self,
        frame: RealtimeInputFrame<'_>,
    ) -> RealtimeCodecResult<DecodedRealtimeEvent<OpenAiTranslationServerEvent>> {
        let parsed = parse_frame(frame, self.limits)?;
        let event_type = parsed.event_type.as_str();
        let state = self.state();
        if self.session_closed {
            return Err(RealtimeCodecError::TranslationServerStateViolation {
                event_type: event_type.to_owned(),
                state: state.label(),
            });
        }

        let object = parsed.object();
        let event = match event_type {
            "session.created" => {
                if self.session_created {
                    return Err(RealtimeCodecError::TranslationServerStateViolation {
                        event_type: event_type.to_owned(),
                        state: state.label(),
                    });
                }
                let event = OpenAiTranslationServerEvent::SessionCreated {
                    event_id: event_id(object, event_type)?,
                    session: required_value(object, event_type, "session")?,
                };
                self.session_created = true;
                event
            }
            "session.updated" => {
                self.require_created(event_type)?;
                OpenAiTranslationServerEvent::SessionUpdated {
                    event_id: event_id(object, event_type)?,
                    session: required_value(object, event_type, "session")?,
                }
            }
            "session.input_transcript.delta" => {
                self.require_created(event_type)?;
                OpenAiTranslationServerEvent::InputTranscriptDelta {
                    event_id: event_id(object, event_type)?,
                    delta: required_string(object, event_type, "delta")?,
                    elapsed_ms: optional_u64(object, event_type, "elapsed_ms")?,
                }
            }
            "session.output_transcript.delta" => {
                self.require_created(event_type)?;
                OpenAiTranslationServerEvent::OutputTranscriptDelta {
                    event_id: event_id(object, event_type)?,
                    delta: required_string(object, event_type, "delta")?,
                    elapsed_ms: optional_u64(object, event_type, "elapsed_ms")?,
                }
            }
            "session.output_audio.delta" => {
                self.require_created(event_type)?;
                OpenAiTranslationServerEvent::OutputAudioDelta(
                    self.decode_audio_delta(object, event_type)?,
                )
            }
            "session.closed" => {
                self.require_created(event_type)?;
                let session = object.get("session").cloned();
                let event = OpenAiTranslationServerEvent::SessionClosed {
                    event_id: event_id(object, event_type)?,
                    session,
                    close_was_requested: self.close_requested,
                };
                self.session_closed = true;
                event
            }
            "error" => OpenAiTranslationServerEvent::Error(parse_server_error(object, event_type)?),
            _ => OpenAiTranslationServerEvent::Unknown(UnknownRealtimeEvent {
                event_type: parsed.event_type.clone(),
            }),
        };

        Ok(parsed.into_decoded(event))
    }

    /// Validates the terminal condition when the WebSocket transport closes.
    pub fn finish_transport(&self) -> RealtimeCodecResult<()> {
        if self.session_closed {
            Ok(())
        } else {
            Err(RealtimeCodecError::TranslationClosedBeforeSessionClosed)
        }
    }

    fn require_created(&self, event_type: &str) -> RealtimeCodecResult<()> {
        if self.session_created {
            Ok(())
        } else {
            Err(RealtimeCodecError::TranslationServerStateViolation {
                event_type: event_type.to_owned(),
                state: self.state().label(),
            })
        }
    }

    fn decode_audio_delta(
        &self,
        object: &serde_json::Map<String, Value>,
        event_type: &str,
    ) -> RealtimeCodecResult<TranslationAudioDeltaEvent> {
        let sample_rate = optional_u64(object, event_type, "sample_rate")?.unwrap_or(24_000);
        let sample_rate_hz =
            u32::try_from(sample_rate).map_err(|_| RealtimeCodecError::InvalidField {
                event_type: event_type.to_owned(),
                field: "sample_rate",
                reason: "value does not fit in u32".to_owned(),
            })?;
        let channel_count = optional_u64(object, event_type, "channels")?.unwrap_or(1);
        let channels =
            u16::try_from(channel_count).map_err(|_| RealtimeCodecError::InvalidField {
                event_type: event_type.to_owned(),
                field: "channels",
                reason: "value does not fit in u16".to_owned(),
            })?;

        Ok(TranslationAudioDeltaEvent {
            event_id: event_id(object, event_type)?,
            audio: decode_audio(object, event_type, "delta", self.limits)?,
            sample_rate_hz,
            channels,
            format: optional_string(object, event_type, "format")?
                .unwrap_or_else(|| "pcm16".to_owned()),
            elapsed_ms: optional_u64(object, event_type, "elapsed_ms")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn create_session(codec: &mut OpenAiTranslationCodec) {
        codec
            .decode(
                r#"{"type":"session.created","session":{"id":"sess_1","type":"translation"}}"#
                    .into(),
            )
            .unwrap();
    }

    #[test]
    fn encodes_translation_audio_as_base64_text() {
        let mut codec = OpenAiTranslationCodec::default();
        let frame = codec
            .encode(&OpenAiTranslationClientEvent::InputAudioBufferAppend {
                event_id: None,
                audio: Bytes::from_static(&[1, 2, 3]),
            })
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(frame.as_str()).unwrap(),
            json!({
                "type": "session.input_audio_buffer.append",
                "audio": "AQID"
            })
        );
    }

    #[test]
    fn close_is_a_half_close_until_session_closed() {
        let mut codec = OpenAiTranslationCodec::default();
        create_session(&mut codec);
        let close = codec
            .encode(&OpenAiTranslationClientEvent::SessionClose { event_id: None })
            .unwrap();
        assert_eq!(close.as_str(), r#"{"type":"session.close"}"#);
        assert_eq!(codec.state(), TranslationSessionState::CloseRequested);

        let client_error = codec
            .encode(&OpenAiTranslationClientEvent::InputAudioBufferAppend {
                event_id: None,
                audio: Bytes::from_static(&[4]),
            })
            .unwrap_err();
        assert!(matches!(
            client_error,
            RealtimeCodecError::TranslationClientStateViolation { .. }
        ));

        let trailing_output = codec
            .decode(r#"{"type":"session.output_transcript.delta","delta":"bonjour"}"#.into())
            .unwrap();
        assert!(matches!(
            trailing_output.event,
            OpenAiTranslationServerEvent::OutputTranscriptDelta { .. }
        ));

        let closed = codec
            .decode(r#"{"type":"session.closed","session":{"id":"sess_1"}}"#.into())
            .unwrap();
        assert!(closed.event.is_session_terminal());
        assert_eq!(codec.state(), TranslationSessionState::Closed);
        assert!(codec.finish_transport().is_ok());
    }

    #[test]
    fn transport_close_requires_session_closed() {
        let mut codec = OpenAiTranslationCodec::default();
        create_session(&mut codec);
        codec
            .encode(&OpenAiTranslationClientEvent::SessionClose { event_id: None })
            .unwrap();

        assert_eq!(
            codec.finish_transport(),
            Err(RealtimeCodecError::TranslationClosedBeforeSessionClosed)
        );
    }

    #[test]
    fn decodes_translation_audio_defaults() {
        let mut codec = OpenAiTranslationCodec::default();
        create_session(&mut codec);
        let event = codec
            .decode(r#"{"type":"session.output_audio.delta","delta":"AAE="}"#.into())
            .unwrap();

        match event.event {
            OpenAiTranslationServerEvent::OutputAudioDelta(audio) => {
                assert_eq!(audio.audio, Bytes::from_static(&[0, 1]));
                assert_eq!(audio.sample_rate_hz, 24_000);
                assert_eq!(audio.channels, 1);
                assert_eq!(audio.format, "pcm16");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn translation_error_is_recoverable() {
        let mut codec = OpenAiTranslationCodec::default();
        create_session(&mut codec);
        let event = codec
            .decode(r#"{"type":"error","error":{"message":"temporary"}}"#.into())
            .unwrap();
        assert!(matches!(
            event.event,
            OpenAiTranslationServerEvent::Error(_)
        ));
        assert_eq!(codec.state(), TranslationSessionState::Active);

        let next = codec
            .decode(r#"{"type":"session.input_transcript.delta","delta":"hello"}"#.into())
            .unwrap();
        assert!(matches!(
            next.event,
            OpenAiTranslationServerEvent::InputTranscriptDelta { .. }
        ));
    }

    #[test]
    fn translation_stream_data_requires_session_created() {
        let mut codec = OpenAiTranslationCodec::default();
        let error = codec
            .decode(r#"{"type":"session.output_transcript.delta","delta":"bonjour"}"#.into())
            .unwrap_err();

        assert!(matches!(
            error,
            RealtimeCodecError::TranslationServerStateViolation { .. }
        ));
        assert_eq!(
            codec.state(),
            TranslationSessionState::AwaitingSessionCreated
        );
    }

    #[test]
    fn rejects_events_after_session_closed() {
        let mut codec = OpenAiTranslationCodec::default();
        create_session(&mut codec);
        codec.decode(r#"{"type":"session.closed"}"#.into()).unwrap();

        let error = codec
            .decode(r#"{"type":"error","error":{"message":"late"}}"#.into())
            .unwrap_err();
        assert!(matches!(
            error,
            RealtimeCodecError::TranslationServerStateViolation { .. }
        ));
    }

    #[test]
    fn rejects_binary_translation_frames() {
        let mut codec = OpenAiTranslationCodec::default();
        let error = codec
            .decode(RealtimeInputFrame::Binary(&[0, 1]))
            .unwrap_err();
        assert_eq!(
            error,
            RealtimeCodecError::RawBinaryFrameUnsupported { bytes: 2 }
        );
    }
}
