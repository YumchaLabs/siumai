//! Retry module (ergonomic namespace)
//! - policy.rs: generic policy-based retries
//! - backoff.rs: provider/HTTP-aware backoff retries

pub mod backoff;
pub mod policy;

pub use backoff::*;
pub use policy::*;
