//! Centralized list of OpenAI-compatible providers for generating builder methods.
//!
//! This macro enumerates all supported OpenAI-compatible providers. It accepts
//! a callback macro `$mac` and invokes it as `$mac!(method_name, provider_id)`
//! for each provider entry. Use it to generate methods on different builders
//! without duplicating the list of providers.
//!
//! Example:
//!
//! ```ignore
//! macro_rules! gen_llmbuilder_method {
//!     ($name:ident, $id:expr) => {
//!         #[cfg(feature = "openai")]
//!         pub fn $name(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
//!             crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, $id)
//!         }
//!     };
//! }
//! siumai_for_each_openai_compatible_provider!(gen_llmbuilder_method);
//! ```
// Macro moved to crate root (src/macros.rs) to ensure availability
// regardless of feature flags and module inclusion order.
