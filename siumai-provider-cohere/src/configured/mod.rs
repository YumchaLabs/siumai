mod auth;
mod model;
mod options;
mod provider;
mod wire;

pub use model::{CohereEmbeddingModel, CohereRerankModel};
pub use provider::{CohereConfigError, CohereProvider, CohereProviderBuilder};

#[cfg(test)]
mod tests;
