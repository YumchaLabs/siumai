//! Incremental and bounded JSON Lines framing.

use std::fmt;

use serde_json::Value;
use thiserror::Error;

use crate::TransportLimits;

/// Bounded JSONL framing failure without raw payload disclosure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JsonLinesError {
    #[error("JSONL line exceeds the configured frame limit")]
    FrameTooLarge,
    #[error("JSONL stream exceeds the configured event-count limit")]
    TooManyEvents,
    #[error("JSONL stream contains invalid UTF-8")]
    InvalidUtf8,
    #[error("JSONL stream contains invalid JSON")]
    InvalidJson,
}

/// Stateful JSONL decoder for arbitrary byte-chunk boundaries.
pub struct JsonLinesDecoder {
    max_frame_bytes: usize,
    max_events: usize,
    line: Vec<u8>,
    saw_carriage_return: bool,
    emitted_events: usize,
}

impl fmt::Debug for JsonLinesDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsonLinesDecoder")
            .field("max_frame_bytes", &self.max_frame_bytes)
            .field("max_events", &self.max_events)
            .field("buffered_line_bytes", &self.line.len())
            .field("saw_carriage_return", &self.saw_carriage_return)
            .field("emitted_events", &self.emitted_events)
            .finish()
    }
}

impl JsonLinesDecoder {
    pub fn new(limits: &TransportLimits) -> Self {
        Self {
            max_frame_bytes: limits.max_frame_bytes,
            max_events: limits.max_events_per_stream,
            line: Vec::new(),
            saw_carriage_return: false,
            emitted_events: 0,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Value>, JsonLinesError> {
        let mut values = Vec::new();
        for byte in chunk {
            if self.saw_carriage_return {
                self.saw_carriage_return = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\r' => {
                    self.process_line(&mut values)?;
                    self.saw_carriage_return = true;
                }
                b'\n' => self.process_line(&mut values)?,
                byte => {
                    if self.line.len() >= self.max_frame_bytes {
                        return Err(JsonLinesError::FrameTooLarge);
                    }
                    self.line.push(byte);
                }
            }
        }
        Ok(values)
    }

    pub fn finish(mut self) -> Result<Vec<Value>, JsonLinesError> {
        let mut values = Vec::new();
        if !self.line.is_empty() {
            self.process_line(&mut values)?;
        }
        Ok(values)
    }

    fn process_line(&mut self, values: &mut Vec<Value>) -> Result<(), JsonLinesError> {
        let line = std::mem::take(&mut self.line);
        let line = std::str::from_utf8(&line).map_err(|_| JsonLinesError::InvalidUtf8)?;
        if line.trim().is_empty() {
            return Ok(());
        }
        let value = serde_json::from_str(line).map_err(|_| JsonLinesError::InvalidJson)?;
        self.emitted_events = self.emitted_events.saturating_add(1);
        if self.emitted_events > self.max_events {
            return Err(JsonLinesError::TooManyEvents);
        }
        values.push(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> TransportLimits {
        TransportLimits {
            max_frame_bytes: 32,
            max_events_per_stream: 2,
            ..TransportLimits::default()
        }
    }

    #[test]
    fn parses_fragmented_multibyte_json_and_final_line() {
        let mut decoder = JsonLinesDecoder::new(&limits());
        let input = "{\"text\":\"你\"}\n{\"done\":true}".as_bytes();
        let split = input.iter().position(|byte| *byte >= 0x80).unwrap() + 1;
        assert!(decoder.push(&input[..split]).unwrap().is_empty());
        let mut values = decoder.push(&input[split..]).unwrap();
        values.extend(decoder.finish().unwrap());
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["text"], "你");
        assert_eq!(values[1]["done"], true);
    }

    #[test]
    fn rejects_invalid_utf8_json_and_excess_events() {
        let mut decoder = JsonLinesDecoder::new(&limits());
        assert_eq!(
            decoder.push(&[0xff, b'\n']),
            Err(JsonLinesError::InvalidUtf8)
        );

        let mut decoder = JsonLinesDecoder::new(&limits());
        assert_eq!(
            decoder.push(b"not-json\n"),
            Err(JsonLinesError::InvalidJson)
        );

        let mut decoder = JsonLinesDecoder::new(&limits());
        assert_eq!(
            decoder.push(b"{}\n{}\n{}\n"),
            Err(JsonLinesError::TooManyEvents)
        );
    }

    #[test]
    fn debug_redacts_buffered_payload() {
        let mut decoder = JsonLinesDecoder::new(&limits());
        decoder.push(br#"{"token":"canary"}"#).unwrap();

        assert!(!format!("{decoder:?}").contains("canary"));
    }
}
