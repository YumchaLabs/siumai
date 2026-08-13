//! Bounded WebSocket message framing independent of provider event semantics.

use std::fmt;

use bytes::Bytes;
use thiserror::Error;
use tokio_tungstenite::tungstenite::Message;

use crate::TransportLimits;

/// A validated WebSocket message for a provider protocol state machine.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebSocketFrame {
    Text(String),
    Binary(Bytes),
    Ping(Bytes),
    Pong(Bytes),
    Close { code: Option<u16>, reason: String },
}

impl fmt::Debug for WebSocketFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => formatter
                .debug_struct("WebSocketFrame::Text")
                .field("bytes", &text.len())
                .finish(),
            Self::Binary(bytes) => formatter
                .debug_struct("WebSocketFrame::Binary")
                .field("bytes", &bytes.len())
                .finish(),
            Self::Ping(bytes) => formatter
                .debug_struct("WebSocketFrame::Ping")
                .field("bytes", &bytes.len())
                .finish(),
            Self::Pong(bytes) => formatter
                .debug_struct("WebSocketFrame::Pong")
                .field("bytes", &bytes.len())
                .finish(),
            Self::Close { code, reason } => formatter
                .debug_struct("WebSocketFrame::Close")
                .field("code", code)
                .field("reason_bytes", &reason.len())
                .finish(),
        }
    }
}

/// Bounded WebSocket framing failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum WebSocketFrameError {
    #[error("WebSocket message exceeds the configured frame limit")]
    FrameTooLarge,
    #[error("WebSocket stream exceeds the configured event-count limit")]
    TooManyFrames,
    #[error("raw WebSocket frames are not a provider message boundary")]
    RawFrameUnsupported,
}

/// Stateful message-count and size guard.
#[derive(Debug)]
pub struct WebSocketFramer {
    max_frame_bytes: usize,
    max_frames: usize,
    frames: usize,
}

impl WebSocketFramer {
    pub fn new(limits: &TransportLimits) -> Self {
        Self {
            max_frame_bytes: limits.max_frame_bytes,
            max_frames: limits.max_events_per_stream,
            frames: 0,
        }
    }

    pub fn decode(&mut self, message: Message) -> Result<WebSocketFrame, WebSocketFrameError> {
        if message.len() > self.max_frame_bytes {
            return Err(WebSocketFrameError::FrameTooLarge);
        }
        self.frames = self.frames.saturating_add(1);
        if self.frames > self.max_frames {
            return Err(WebSocketFrameError::TooManyFrames);
        }
        match message {
            Message::Text(text) => Ok(WebSocketFrame::Text(text.to_string())),
            Message::Binary(bytes) => Ok(WebSocketFrame::Binary(bytes)),
            Message::Ping(bytes) => Ok(WebSocketFrame::Ping(bytes)),
            Message::Pong(bytes) => Ok(WebSocketFrame::Pong(bytes)),
            Message::Close(frame) => Ok(WebSocketFrame::Close {
                code: frame.as_ref().map(|frame| u16::from(frame.code)),
                reason: frame
                    .map(|frame| frame.reason.to_string())
                    .unwrap_or_default(),
            }),
            Message::Frame(_) => Err(WebSocketFrameError::RawFrameUnsupported),
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio_tungstenite::tungstenite::protocol::CloseFrame;
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

    use super::*;

    fn limits() -> TransportLimits {
        TransportLimits {
            max_frame_bytes: 8,
            max_events_per_stream: 2,
            ..TransportLimits::default()
        }
    }

    #[test]
    fn preserves_text_binary_control_and_close_boundaries() {
        let cases = [
            (
                Message::text("hello"),
                WebSocketFrame::Text("hello".to_owned()),
            ),
            (
                Message::binary(Bytes::from_static(&[1_u8, 2])),
                WebSocketFrame::Binary(Bytes::from_static(&[1, 2])),
            ),
            (
                Message::Ping(Bytes::from_static(b"p")),
                WebSocketFrame::Ping(Bytes::from_static(b"p")),
            ),
            (
                Message::Close(Some(CloseFrame {
                    code: CloseCode::Normal,
                    reason: "bye".into(),
                })),
                WebSocketFrame::Close {
                    code: Some(1000),
                    reason: "bye".to_owned(),
                },
            ),
        ];
        for (message, expected) in cases {
            assert_eq!(
                WebSocketFramer::new(&limits()).decode(message),
                Ok(expected)
            );
        }
    }

    #[test]
    fn frame_size_and_count_are_bounded() {
        let mut framer = WebSocketFramer::new(&limits());
        assert_eq!(
            framer.decode(Message::text("123456789")),
            Err(WebSocketFrameError::FrameTooLarge)
        );
        assert!(framer.decode(Message::text("one")).is_ok());
        assert!(framer.decode(Message::text("two")).is_ok());
        assert_eq!(
            framer.decode(Message::text("three")),
            Err(WebSocketFrameError::TooManyFrames)
        );
    }

    #[test]
    fn debug_redacts_all_frame_payloads() {
        let frames = [
            WebSocketFrame::Text("canary-text".to_owned()),
            WebSocketFrame::Binary(Bytes::from_static(b"canary-binary")),
            WebSocketFrame::Ping(Bytes::from_static(b"canary-ping")),
            WebSocketFrame::Pong(Bytes::from_static(b"canary-pong")),
            WebSocketFrame::Close {
                code: Some(1000),
                reason: "canary-reason".to_owned(),
            },
        ];

        for frame in frames {
            assert!(!format!("{frame:?}").contains("canary"));
        }
    }
}
