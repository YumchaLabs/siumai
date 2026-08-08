//! Current MiniMax provider integration for Siumai.
//!
//! The provider composes three explicit language protocol modes under one
//! canonical `minimax` identity. Anthropic Messages is the recommended path;
//! OpenAI Chat Completions and the bounded Responses-compatible subset remain
//! available through provider-direct methods.
#![deny(unsafe_code)]

mod annotations;
mod credential;
mod language;
pub mod models;
mod options;
mod provider;
pub mod resources;

pub use annotations::{
    MinimaxAnnotationResolver, MinimaxContentCache, MinimaxMessageCache, MinimaxToolCache,
};
pub use credential::{MinimaxCredential, MinimaxCredentialError};
pub use language::MinimaxLanguageProfileError;
pub use models::{
    MINIMAX_M2, MINIMAX_M2_1, MINIMAX_M2_1_HIGHSPEED, MINIMAX_M2_5, MINIMAX_M2_5_HIGHSPEED,
    MINIMAX_M2_7, MINIMAX_M2_7_HIGHSPEED, MINIMAX_M3,
};
pub use options::{
    MinimaxChatCompletionsOptions, MinimaxMessagesOptions, MinimaxReasoningEffort,
    MinimaxResponsesOptions, MinimaxResponsesReasoning, MinimaxServiceTier, MinimaxThinking,
};
pub use provider::{
    MINIMAX_REPLAY_AUDIENCE, MinimaxConfigError, MinimaxLanguageApi, MinimaxLanguageModel,
    MinimaxProvider, MinimaxProviderBuilder,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests;
