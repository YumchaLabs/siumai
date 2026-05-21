//! OpenAI-compatible adapter + config + chat/completion conversion protocol layer.

pub mod adapter;
pub mod alibaba_cache_control;
pub mod base_url;
pub mod completion;
pub mod metadata;
pub mod openai_config;
pub mod provider_registry;
pub mod reasoning;
mod response_content;
pub mod spec;
pub mod streaming;
pub mod transformers;
pub mod types;
pub mod usage;

#[cfg(test)]
mod streaming_tests;

#[cfg(test)]
mod transformers_tests;
