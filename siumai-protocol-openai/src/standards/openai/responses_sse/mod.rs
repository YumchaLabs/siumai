//! OpenAI Responses SSE Event Converter (protocol layer)
//!
//! This module normalizes OpenAI Responses API SSE events into Siumai's unified
//! `ChatStreamEvent` sequence. It is intentionally part of the `standards::openai`
//! protocol implementation so that providers stay thin.
//!
//! Note: Providers may re-export this converter under historical module paths
//! (e.g. `providers::openai::responses::OpenAiResponsesEventConverter`).

mod converter;

#[cfg(test)]
mod tests;

pub use converter::{OpenAiResponsesEventConverter, StreamPartsStyle, WebSearchStreamMode};
