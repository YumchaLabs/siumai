//! MiniMax provider extension APIs (non-unified surface)
//!
//! These APIs expose MiniMax-specific helpers that are intentionally not part
//! of the Vercel-aligned unified model families.

pub mod music;
pub mod request_options;
pub mod thinking;
pub mod tts;
pub mod tts_options;
pub mod video;
pub mod video_options;

pub use request_options::MinimaxChatRequestExt;
pub use video_options::MinimaxVideoRequestExt;
