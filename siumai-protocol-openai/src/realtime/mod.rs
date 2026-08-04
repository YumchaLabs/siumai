//! Typed codecs for OpenAI Realtime and Realtime Translation WebSocket events.
//!
//! The module deliberately stops at the protocol boundary. It understands JSON
//! text frames and OpenAI's base64-encoded audio fields, but it does not own a
//! WebSocket connection, retries, authentication, or provider configuration.

mod common;
mod conversation;
mod error;
mod frame;
mod translation;

pub use common::{
    DecodedRealtimeEvent, OpenAiRealtimeServerError, RealtimeCodecLimits, UnknownRealtimeEvent,
};
pub use conversation::{
    FunctionCallArgumentsDeltaEvent, FunctionCallArgumentsDoneEvent, IncompleteFunctionCall,
    OpenAiRealtimeClientEvent, OpenAiRealtimeDecoder, OpenAiRealtimeServerEvent,
    RealtimeResponseDoneEvent, RealtimeResponseStatus,
};
pub use error::{RealtimeCodecError, RealtimeCodecResult};
pub use frame::{JsonTextFrame, RealtimeInputFrame};
pub use translation::{
    OpenAiTranslationClientEvent, OpenAiTranslationCodec, OpenAiTranslationServerEvent,
    TranslationAudioDeltaEvent, TranslationSessionState,
};
