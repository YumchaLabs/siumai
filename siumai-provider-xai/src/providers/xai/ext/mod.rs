//! xAI provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need xAI-specific escape hatches.

pub mod image_options;
pub mod request_options;
pub mod tts_options;
pub mod video_options;

pub use image_options::XaiImageRequestExt;
pub use request_options::XaiChatRequestExt;
pub use tts_options::XaiTtsRequestExt;
pub use video_options::XaiVideoRequestExt;
