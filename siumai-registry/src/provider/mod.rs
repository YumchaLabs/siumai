//! Legacy `Siumai` method-style provider interface.
//!
//! This module keeps the historical `Siumai` wrapper and builder available while the stable
//! architecture converges on registry/config-first construction plus model-family helpers.
//! New registry code should prefer `registry::*ModelHandle` and family traits over routing through
//! a boxed `LlmClient`.

mod siumai;
pub use siumai::{ProviderMetadata, Siumai};

mod proxies;
pub use proxies::{AudioCapabilityProxy, EmbeddingCapabilityProxy};

/// Compatibility builder for the historical method-style `Siumai` wrapper.
///
/// Prefer registry handles for stable family execution:
///
/// ```rust,no_run
/// # use siumai_registry::registry;
/// # fn main() -> Result<(), siumai_registry::LlmError> {
/// let model = registry::global().language_model("openai:gpt-4o-mini")?;
/// # Ok(())
/// # }
/// ```
///
/// Keep this builder for migration and provider-specific convenience paths only.
mod siumai_builder;
pub use siumai_builder::SiumaiBuilder;

pub(crate) mod ids;
pub(crate) mod resolver;

// Keep module slim by moving heavy build logic out
pub mod build;
