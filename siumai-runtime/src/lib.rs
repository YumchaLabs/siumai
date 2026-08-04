//! Provider-neutral high-level execution for Siumai language models.
//!
//! Plain [`generate`] and [`stream`] perform exactly one model call. Explicit
//! multi-step tool execution is owned by the runtime's tool-loop APIs.

#![deny(unsafe_code)]

mod call;
mod options;

pub use call::{generate, stream};
pub use options::{ModelTarget, Runtime, RuntimeBuilder, RuntimeConfigError, StepOptions};
