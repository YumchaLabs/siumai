//! MiniMaxi transformers for request/response conversion

pub mod audio;
pub mod image;

pub use audio::MinimaxiAudioTransformer;
pub use image::{MinimaxiImageAdapter, create_minimaxi_image_standard};
