//! Provider-facing utility seams for siumai.
//!
//! This crate mirrors the architectural role of Vercel AI SDK's
//! `@ai-sdk/provider-utils`: reusable helpers for provider/protocol adapters that are too
//! implementation-oriented for `siumai-spec`, but should not keep growing inside
//! `siumai-core::utils`.
//!
//! FCAB-090 started with URL composition, MIME detection, builder defaults, and chat request
//! normalization. FCAB-100 deepens the seam with the rest of the spec-only AI SDK-style helper set:
//! JSON parsing, schema/type validation, provider options/reference utilities, settings, downloads,
//! headers, IDs, media/data helpers, and small async/runtime helpers.
#![deny(unsafe_code)]

pub use siumai_spec::{error, types};

pub mod standards;

pub mod builder_helpers;
pub mod chat_request;
pub mod data;
pub mod download;
pub mod error_message;
pub mod headers;
pub mod id;
pub mod json_instruction;
pub mod json_parse;
pub mod mime;
pub mod option;
pub mod provider_options;
pub mod provider_reference;
pub mod reasoning;
pub mod runtime;
pub mod serial_job;
pub mod settings;
pub mod url;
pub mod utf8_decoder;
pub mod validate_types;

pub use builder_helpers::*;
pub use chat_request::*;
pub use data::*;
pub use download::*;
pub use error_message::*;
pub use headers::*;
pub use id::*;
pub use json_instruction::*;
pub use json_parse::*;
pub use mime::*;
pub use option::*;
pub use provider_options::*;
pub use provider_reference::*;
pub use reasoning::*;
pub use runtime::*;
pub use serial_job::*;
pub use settings::*;
pub use url::*;
pub use utf8_decoder::Utf8StreamDecoder;
pub use validate_types::*;
