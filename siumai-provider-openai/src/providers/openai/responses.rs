//! OpenAI Responses SSE Event Converter (re-export)
//!
//! The canonical implementation lives in `siumai-core` under
//! `standards::openai::responses_sse`. This module preserves the historical
//! import path: `siumai::providers::openai::responses::OpenAiResponsesEventConverter`.

pub use crate::standards::openai::responses_sse::OpenAiResponsesEventConverter;
