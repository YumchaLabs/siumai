use siumai_core::{
    DecoderLifecycle, Error, ErrorKind, FinishReason, LanguageResponse, LanguageStreamDecoder,
    LanguageStreamEvent, StreamTerminal, Usage,
};

fn completed(reason: FinishReason) -> LanguageStreamEvent {
    LanguageStreamEvent::Terminal(StreamTerminal::Completed {
        response: Box::new(
            LanguageResponse::completed(Vec::new(), reason, Usage::default()).unwrap(),
        ),
    })
}

fn decode_frames<D>(
    decoder: &mut D,
    frames: &[D::ProtocolFrame],
) -> Result<Vec<LanguageStreamEvent>, Error>
where
    D: LanguageStreamDecoder,
    D::ProtocolFrame: Sized,
{
    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(frame)?);
    }
    Ok(events)
}

fn finish_once<D>(decoder: &mut D, events: &mut Vec<LanguageStreamEvent>) -> Result<(), Error>
where
    D: LanguageStreamDecoder,
{
    events.extend(decoder.finish()?);
    assert!(decoder.terminal_seen());
    assert_eq!(
        events
            .iter()
            .filter(|event| event.terminal().is_some())
            .count(),
        1
    );
    let duplicate = decoder.finish().expect_err("finish must run at most once");
    assert_eq!(duplicate.kind(), ErrorKind::Protocol);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ResponsesFrame {
    ResponseCreated,
    OutputTextDelta,
    ResponseCompleted,
}

#[derive(Debug, Default)]
struct ResponsesDecoder {
    lifecycle: DecoderLifecycle,
    created: bool,
}

impl LanguageStreamDecoder for ResponsesDecoder {
    type ProtocolFrame = ResponsesFrame;

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        let events = match frame {
            ResponsesFrame::ResponseCreated => {
                self.created = true;
                vec![LanguageStreamEvent::Started {
                    id: Some("resp-1".to_string()),
                    model: None,
                }]
            }
            ResponsesFrame::OutputTextDelta if self.created => {
                vec![LanguageStreamEvent::TextDelta {
                    id: "output-0".to_string(),
                    delta: "hello".to_string(),
                }]
            }
            ResponsesFrame::ResponseCompleted if self.created => {
                vec![completed(FinishReason::Stop)]
            }
            ResponsesFrame::OutputTextDelta | ResponsesFrame::ResponseCompleted => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "Responses event arrived before response.created",
                ));
            }
        };
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

#[derive(Debug, Clone, Copy)]
enum AnthropicFrame {
    MessageStart,
    ContentBlockStart,
    ContentBlockStop,
    MessageStop { refusal: bool },
}

#[derive(Debug, Default)]
struct AnthropicDecoder {
    lifecycle: DecoderLifecycle,
    message_started: bool,
    content_open: bool,
}

impl LanguageStreamDecoder for AnthropicDecoder {
    type ProtocolFrame = AnthropicFrame;

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        let events = match frame {
            AnthropicFrame::MessageStart if !self.message_started => {
                self.message_started = true;
                vec![LanguageStreamEvent::Started {
                    id: Some("msg-1".to_string()),
                    model: None,
                }]
            }
            AnthropicFrame::ContentBlockStart if self.message_started && !self.content_open => {
                self.content_open = true;
                vec![LanguageStreamEvent::TextStart {
                    id: "block-0".to_string(),
                }]
            }
            AnthropicFrame::ContentBlockStop if self.content_open => {
                self.content_open = false;
                vec![LanguageStreamEvent::TextEnd {
                    id: "block-0".to_string(),
                }]
            }
            AnthropicFrame::MessageStop { refusal }
                if self.message_started && !self.content_open =>
            {
                let mut events = Vec::new();
                if *refusal {
                    events.push(LanguageStreamEvent::Refusal {
                        reason: Some("safety refusal".to_string()),
                    });
                }
                events.push(completed(if *refusal {
                    FinishReason::Refusal
                } else {
                    FinishReason::Stop
                }));
                events
            }
            AnthropicFrame::MessageStop { .. } if self.content_open => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "Anthropic message_stop arrived before content_block_stop",
                ));
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "Anthropic Messages event violated block ordering",
                ));
            }
        };
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

#[derive(Debug, Clone, Copy)]
enum GeminiFrame {
    CandidateText,
    CandidateFinished,
}

#[derive(Debug, Default)]
struct GeminiDecoder {
    lifecycle: DecoderLifecycle,
    candidate_seen: bool,
    finish_reason: Option<FinishReason>,
}

impl LanguageStreamDecoder for GeminiDecoder {
    type ProtocolFrame = GeminiFrame;

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        let events = match frame {
            GeminiFrame::CandidateText => {
                self.candidate_seen = true;
                vec![LanguageStreamEvent::TextDelta {
                    id: "candidate-0".to_string(),
                    delta: "hello".to_string(),
                }]
            }
            GeminiFrame::CandidateFinished if self.candidate_seen => {
                self.finish_reason = Some(FinishReason::Stop);
                Vec::new()
            }
            GeminiFrame::CandidateFinished => {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "Gemini candidate finished before content",
                ));
            }
        };
        self.lifecycle.record(&events).map_err(Error::from)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish().map_err(Error::from)? {
            return Ok(Vec::new());
        }
        let reason = self
            .finish_reason
            .take()
            .ok_or_else(Error::unexpected_eof)?;
        let events = vec![completed(reason)];
        self.lifecycle.record(&events).map_err(Error::from)?;
        Ok(events)
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

#[test]
fn responses_uses_explicit_response_completed_terminal() {
    let mut decoder = ResponsesDecoder::default();
    let mut events = decode_frames(
        &mut decoder,
        &[
            ResponsesFrame::ResponseCreated,
            ResponsesFrame::OutputTextDelta,
            ResponsesFrame::ResponseCompleted,
        ],
    )
    .unwrap();

    assert!(decoder.terminal_seen());
    let after_terminal = decoder
        .decode(&ResponsesFrame::OutputTextDelta)
        .expect_err("frames after response.completed must fail");
    assert_eq!(after_terminal.kind(), ErrorKind::Protocol);
    finish_once(&mut decoder, &mut events).unwrap();
}

#[test]
fn anthropic_requires_closed_content_before_message_stop() {
    let mut invalid = AnthropicDecoder::default();
    decode_frames(
        &mut invalid,
        &[
            AnthropicFrame::MessageStart,
            AnthropicFrame::ContentBlockStart,
        ],
    )
    .unwrap();
    let ordering_error = invalid
        .decode(&AnthropicFrame::MessageStop { refusal: true })
        .expect_err("message_stop cannot close an open content block");
    assert_eq!(ordering_error.kind(), ErrorKind::Protocol);

    let mut decoder = AnthropicDecoder::default();
    let mut events = decode_frames(
        &mut decoder,
        &[
            AnthropicFrame::MessageStart,
            AnthropicFrame::ContentBlockStart,
            AnthropicFrame::ContentBlockStop,
            AnthropicFrame::MessageStop { refusal: true },
        ],
    )
    .unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Refusal { reason: Some(reason) } if reason == "safety refusal"
    )));
    finish_once(&mut decoder, &mut events).unwrap();
}

#[test]
fn gemini_emits_its_terminal_only_when_eof_finalizes_candidate() {
    let mut decoder = GeminiDecoder::default();
    let mut events = decode_frames(
        &mut decoder,
        &[GeminiFrame::CandidateText, GeminiFrame::CandidateFinished],
    )
    .unwrap();

    assert!(!decoder.terminal_seen());
    finish_once(&mut decoder, &mut events).unwrap();
}
