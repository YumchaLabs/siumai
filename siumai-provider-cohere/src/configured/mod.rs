mod auth;
mod model;
mod options;
mod profile;
mod provider;
mod transcription;
mod wire;

pub use model::{CohereEmbeddingModel, CohereRerankModel};
pub use profile::{CohereProfile, CohereProfileError};
pub use provider::{CohereConfigError, CohereProvider, CohereProviderBuilder};
pub use transcription::{CohereTranscriptionRequest, CohereTranscriptions};

#[cfg(test)]
mod tests;
