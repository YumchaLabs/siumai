//! OpenAI-compatible adapter + config + streaming protocol layer.

pub mod adapter;
pub mod alibaba_cache_control;
pub mod base_url;
pub mod metadata;
pub mod openai_config;
pub mod provider_registry;
pub mod spec;
pub mod streaming;
pub mod transformers;
pub mod types;
pub mod usage;

#[cfg(test)]
mod streaming_tests;

#[cfg(test)]
mod transformers_tests;
