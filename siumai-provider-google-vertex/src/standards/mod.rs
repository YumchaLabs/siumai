//! Provider-owned protocol mapping modules for Google Vertex AI.

pub mod vertex_embedding;
#[cfg(feature = "google-vertex")]
pub mod vertex_generative_ai;
pub mod vertex_imagen;
