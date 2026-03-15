//! MiniMaxi provider extension APIs (non-unified surface)
//!
//! These APIs expose MiniMaxi-specific helpers that are intentionally not part
//! of the Vercel-aligned unified model families.

pub mod music;
pub mod request_options;
pub mod structured_output;
pub mod thinking;
pub mod tts;
pub mod tts_options;
pub mod video;

pub use request_options::MinimaxiChatRequestExt;
