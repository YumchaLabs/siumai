//! DeepSeek protocol mappings.
//!
//! DeepSeek is implemented via the OpenAI-compatible protocol family. This module is a
//! semantic alias so downstream code can depend on `siumai-provider-deepseek` without
//! reaching into `siumai-protocol-openai` directly.

pub mod compat {
    pub use siumai_protocol_openai::standards::openai::compat::*;
}
