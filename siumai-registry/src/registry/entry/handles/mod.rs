mod audio;
mod completion;
mod embedding;
mod image;
mod language;
mod rerank;
mod video;
mod video_support;

pub use audio::{SpeechModelHandle, TranscriptionModelHandle};
pub use completion::CompletionModelHandle;
pub use embedding::EmbeddingModelHandle;
pub use image::ImageModelHandle;
#[cfg(test)]
pub(in crate::registry::entry) use image::image_model_handle_max_images_per_call;
pub use language::LanguageModelHandle;
pub use rerank::RerankingModelHandle;
pub use video::VideoModelHandle;
#[cfg(test)]
pub(in crate::registry::entry) use video_support::video_model_handle_max_videos_per_call;
