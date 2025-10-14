//! Transformers layer (Phase 0 scaffolding)
//!
//! This module defines traits for request/response/stream transformation.
//! They are introduced behind the `new-transformers` feature and will be
//! gradually adopted by providers/executors during the refactor.

pub mod audio;
pub mod files;
pub mod request;
pub mod response;
pub mod stream;
