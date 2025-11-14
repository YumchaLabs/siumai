//! Image generation and processing types (re-export from core)
pub use siumai_core::types::image::*;

// Backward-compatible placeholders for legacy vision trait usage
pub type ImageGenRequest = ();
pub type ImageResponse = ();
pub type VisionRequest = ();
pub type VisionResponse = ();
