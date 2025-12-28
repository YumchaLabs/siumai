//! Provider builder core re-export.
//!
//! `ProviderCore` and its base snapshot type (`BuilderBase`) live in `siumai-core` so
//! provider implementations can be split into separate crates without depending on the
//! umbrella crate (`siumai-providers`).

pub use siumai_core::builder::{BuilderBase, ProviderCore};

