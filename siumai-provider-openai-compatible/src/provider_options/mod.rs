//! Provider-owned typed option structs for OpenAI-compatible vendors.

pub mod openrouter;
pub mod perplexity;

pub use openrouter::{OpenRouterOptions, OpenRouterTransform};
pub use perplexity::{
    PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
    PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
};
