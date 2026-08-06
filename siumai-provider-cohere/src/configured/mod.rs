mod auth;
mod model;
mod options;
mod profile;
mod provider;
mod wire;

pub use model::{CohereEmbeddingModel, CohereRerankModel};
pub use profile::{CohereProfile, CohereProfileError};
pub use provider::{CohereConfigError, CohereProvider, CohereProviderBuilder};

#[cfg(test)]
mod tests;
