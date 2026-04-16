//! Amazon Bedrock standards (Vercel-aligned).
#![deny(unsafe_code)]

pub mod chat;
pub mod embedding;
pub mod errors;
pub(crate) mod headers;
pub mod image;
pub mod rerank;
