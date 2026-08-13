//! Incremental and bounded Server-Sent Events framing.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

use crate::TransportLimits;

/// One framed SSE event. Provider-specific terminal and payload semantics are
/// deliberately not interpreted here.
#[derive(Clone, PartialEq, Eq)]
pub struct SseEvent {
    event_type: String,
    data: String,
    id: Option<Arc<str>>,
    retry: Option<Duration>,
}

impl fmt::Debug for SseEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SseEvent")
            .field("event_type_bytes", &self.event_type.len())
            .field("data_bytes", &self.data.len())
            .field("id_bytes", &self.id.as_ref().map(|value| value.len()))
            .field("retry_present", &self.retry.is_some())
            .finish()
    }
}

impl SseEvent {
    pub fn event_type(&self) -> &str {
        &self.event_type
    }

    pub fn data(&self) -> &str {
        &self.data
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn retry(&self) -> Option<Duration> {
        self.retry
    }
}

/// Bounded SSE framing failure without raw payload disclosure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SseFrameError {
    #[error("SSE line exceeds the configured frame limit")]
    FrameTooLarge,
    #[error("SSE event exceeds the configured event limit")]
    EventTooLarge,
    #[error("SSE stream exceeds the configured event-count limit")]
    TooManyEvents,
    #[error("SSE stream contains invalid UTF-8")]
    InvalidUtf8,
    #[error("SSE stream ended with an incomplete event")]
    UnexpectedEof,
}

/// Stateful decoder for arbitrary byte-chunk boundaries.
pub struct SseDecoder {
    max_frame_bytes: usize,
    max_event_bytes: usize,
    max_events: usize,
    line: Vec<u8>,
    saw_carriage_return: bool,
    event_type: Option<String>,
    data: String,
    has_data: bool,
    last_event_id: Option<Arc<str>>,
    pending_retry: Option<Duration>,
    event_bytes: usize,
    emitted_events: usize,
}

impl fmt::Debug for SseDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SseDecoder")
            .field("max_frame_bytes", &self.max_frame_bytes)
            .field("max_event_bytes", &self.max_event_bytes)
            .field("max_events", &self.max_events)
            .field("buffered_line_bytes", &self.line.len())
            .field("saw_carriage_return", &self.saw_carriage_return)
            .field(
                "event_type_bytes",
                &self.event_type.as_ref().map(|value| value.len()),
            )
            .field("data_bytes", &self.data.len())
            .field("has_data", &self.has_data)
            .field(
                "last_event_id_bytes",
                &self.last_event_id.as_ref().map(|value| value.len()),
            )
            .field("pending_retry_present", &self.pending_retry.is_some())
            .field("event_bytes", &self.event_bytes)
            .field("emitted_events", &self.emitted_events)
            .finish()
    }
}

impl SseDecoder {
    pub fn new(limits: &TransportLimits) -> Self {
        Self {
            max_frame_bytes: limits.max_frame_bytes,
            max_event_bytes: limits.max_event_bytes,
            max_events: limits.max_events_per_stream,
            line: Vec::new(),
            saw_carriage_return: false,
            event_type: None,
            data: String::new(),
            has_data: false,
            last_event_id: None,
            pending_retry: None,
            event_bytes: 0,
            emitted_events: 0,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseFrameError> {
        let mut events = Vec::new();
        for byte in chunk {
            if self.saw_carriage_return {
                self.saw_carriage_return = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\r' => {
                    self.process_line(&mut events)?;
                    self.saw_carriage_return = true;
                }
                b'\n' => self.process_line(&mut events)?,
                byte => {
                    if self.line.len() >= self.max_frame_bytes {
                        return Err(SseFrameError::FrameTooLarge);
                    }
                    self.line.push(byte);
                }
            }
        }
        Ok(events)
    }

    pub fn finish(self) -> Result<(), SseFrameError> {
        if self.line.is_empty() && self.event_type.is_none() && !self.has_data {
            Ok(())
        } else {
            Err(SseFrameError::UnexpectedEof)
        }
    }

    fn process_line(&mut self, events: &mut Vec<SseEvent>) -> Result<(), SseFrameError> {
        let line = std::mem::take(&mut self.line);
        let line = std::str::from_utf8(&line).map_err(|_| SseFrameError::InvalidUtf8)?;
        if line.is_empty() {
            if let Some(event) = self.dispatch()? {
                events.push(event);
            }
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }

        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        self.add_event_bytes(field.len().saturating_add(value.len()))?;
        match field {
            "event" => self.event_type = Some(value.to_owned()),
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                self.has_data = true;
            }
            "id" if !value.contains('\0') => self.last_event_id = Some(Arc::from(value)),
            "retry" => {
                if !value.is_empty()
                    && value.bytes().all(|byte| byte.is_ascii_digit())
                    && let Ok(milliseconds) = value.parse::<u64>()
                {
                    self.pending_retry = Some(Duration::from_millis(milliseconds));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn add_event_bytes(&mut self, additional: usize) -> Result<(), SseFrameError> {
        self.event_bytes = self.event_bytes.saturating_add(additional);
        if self.event_bytes > self.max_event_bytes {
            Err(SseFrameError::EventTooLarge)
        } else {
            Ok(())
        }
    }

    fn dispatch(&mut self) -> Result<Option<SseEvent>, SseFrameError> {
        if !self.has_data {
            self.event_type = None;
            self.event_bytes = 0;
            return Ok(None);
        }
        self.emitted_events = self.emitted_events.saturating_add(1);
        if self.emitted_events > self.max_events {
            return Err(SseFrameError::TooManyEvents);
        }
        if self.data.ends_with('\n') {
            self.data.pop();
        }
        let event = SseEvent {
            event_type: self
                .event_type
                .take()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "message".to_owned()),
            data: std::mem::take(&mut self.data),
            id: self.last_event_id.clone(),
            retry: self.pending_retry.take(),
        };
        self.has_data = false;
        self.event_bytes = 0;
        Ok(Some(event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> TransportLimits {
        TransportLimits {
            max_frame_bytes: 32,
            max_event_bytes: 64,
            max_events_per_stream: 4,
            ..TransportLimits::default()
        }
    }

    #[test]
    fn handles_fragmented_utf8_crlf_comments_and_multiline_data() {
        let mut decoder = SseDecoder::new(&limits());
        let text = "你".as_bytes();
        assert!(
            decoder
                .push(b": ping\r\nid: 7\r\ndata: ")
                .unwrap()
                .is_empty()
        );
        assert!(decoder.push(&text[..2]).unwrap().is_empty());
        assert!(decoder.push(&text[2..]).unwrap().is_empty());
        let events = decoder.push(b"\r\ndata: ok\r\n\r\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data(), "你\nok");
        assert_eq!(events[0].id(), Some("7"));
        assert_eq!(events[0].event_type(), "message");
        decoder.finish().unwrap();
    }

    #[test]
    fn incomplete_event_is_not_fabricated_at_eof() {
        let mut decoder = SseDecoder::new(&limits());
        decoder.push(b"data: partial").unwrap();
        assert_eq!(decoder.finish(), Err(SseFrameError::UnexpectedEof));
    }

    #[test]
    fn retry_only_block_applies_to_the_next_data_event() {
        let mut decoder = SseDecoder::new(&limits());
        assert!(decoder.push(b"retry: 2500\n\n").unwrap().is_empty());
        let events = decoder.push(b"data: next\n\n").unwrap();
        assert_eq!(events[0].retry(), Some(Duration::from_millis(2500)));
        decoder.finish().unwrap();
    }

    #[test]
    fn oversized_line_and_event_count_fail_deterministically() {
        let mut decoder = SseDecoder::new(&limits());
        assert_eq!(decoder.push(&[b'a'; 33]), Err(SseFrameError::FrameTooLarge));

        let mut decoder = SseDecoder::new(&limits());
        for _ in 0..4 {
            decoder.push(b"data: x\n\n").unwrap();
        }
        assert_eq!(
            decoder.push(b"data: x\n\n"),
            Err(SseFrameError::TooManyEvents)
        );
    }

    #[test]
    fn persistent_large_event_ids_are_shared_across_dispatched_events() {
        let limits = TransportLimits {
            max_frame_bytes: 2 * 1024 * 1024,
            max_event_bytes: 2 * 1024 * 1024,
            max_events_per_stream: 1024,
            ..TransportLimits::default()
        };
        let id = "x".repeat(1024 * 1024);
        let mut input = format!("id: {id}\n").into_bytes();
        for _ in 0..1024 {
            input.extend_from_slice(b"data: x\n\n");
        }

        let mut decoder = SseDecoder::new(&limits);
        let events = decoder.push(&input).unwrap();
        assert_eq!(events.len(), 1024);
        let first = events[0].id.as_ref().unwrap();
        assert!(events.iter().all(|event| {
            event
                .id
                .as_ref()
                .is_some_and(|event_id| Arc::ptr_eq(first, event_id))
        }));
    }

    #[test]
    fn debug_redacts_event_and_decoder_payloads() {
        let mut decoder = SseDecoder::new(&limits());
        decoder
            .push(b"event: canary-event\nid: canary-id\ndata: canary-data")
            .unwrap();
        assert!(!format!("{decoder:?}").contains("canary"));

        let events = decoder.push(b"\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert!(!format!("{:?}", events[0]).contains("canary"));
        assert!(!format!("{decoder:?}").contains("canary"));
    }
}
