//! Canonical provider-neutral language request and response content.
//!
//! The public module is intentionally a small assembly surface. Semantic
//! contracts live in focused private modules so callers keep the historical
//! `siumai_core::language::*` paths without inheriting a monolithic file.

mod message;
mod opaque;
mod output;
mod request;
mod response;

#[cfg(test)]
mod tests;

pub use message::{
    AssistantHistoryOmission, AssistantHistoryOmissionKind, AssistantHistoryProjection, Citation,
    ContentPart, MediaData, MediaPart, Message, MessagePart, MessageRole, MessageValidationError,
};
pub use opaque::{
    DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT, DEFAULT_OPAQUE_ITEM_COUNT_LIMIT,
    DEFAULT_OPAQUE_ITEM_LIMIT, OpaqueProviderBudget, OpaqueProviderItem, OpaqueProviderItemBuilder,
    OpaqueProviderItemError, ProviderItemRelation, ProviderProvenance, ProviderProvenanceError,
};
pub use output::{
    DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT,
    DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_COUNT_LIMIT,
    DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageTermination, PartialLanguageOutput,
    PartialLanguageOutputBudget, PartialLanguageOutputError, PartialLanguageOutputPart,
    PartialStructuredOutput,
};
pub use request::{
    GenerationConfig, GenerationConfigError, LanguageInput, LanguageRequest, LanguageRequestBudget,
    LanguageRequestError, StructuredOutputSpec, ToolChoice,
};
pub use response::{
    LanguageCallError, LanguageResponse, LanguageResponseError, Warning, WarningKind,
};
