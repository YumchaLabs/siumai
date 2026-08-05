//! MiniMax transformers for request/response conversion

pub mod audio;
pub mod image;

pub use audio::MinimaxAudioTransformer;
pub use image::{MinimaxImageAdapter, create_minimax_image_standard};
