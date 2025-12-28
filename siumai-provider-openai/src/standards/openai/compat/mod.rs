//! OpenAI-compatible adapter + config + streaming protocol layer
//!
//! This module hosts the reusable OpenAI-compatible protocol implementation
//! (adapters, request/response transformers, and streaming conversion).

pub mod adapter;
pub mod openai_config;
pub mod provider_registry;
pub mod spec;
pub mod streaming;
pub mod transformers;
pub mod types;
