//! Bounded transport framing. Provider protocol state machines live elsewhere.

mod jsonl;
mod sse;
mod websocket;

pub use jsonl::{JsonLinesDecoder, JsonLinesError};
pub use sse::{SseDecoder, SseEvent, SseFrameError};
pub use websocket::{WebSocketFrame, WebSocketFrameError, WebSocketFramer};
