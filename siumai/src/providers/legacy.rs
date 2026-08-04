//! Transitional access to provider packages awaiting Rust-first migration.
//!
//! U10-U13 migrate or delete these packages. New code should use a curated
//! provider namespace or depend on the provider package directly.

#[cfg(feature = "bedrock")]
pub use siumai_provider_amazon_bedrock as bedrock;
#[cfg(feature = "anthropic")]
pub use siumai_provider_anthropic as anthropic;
#[cfg(feature = "azure")]
pub use siumai_provider_azure as azure;
#[cfg(feature = "deepseek")]
pub use siumai_provider_deepseek as deepseek;
#[cfg(feature = "gateway")]
pub use siumai_provider_gateway as gateway;
#[cfg(feature = "google-vertex")]
pub use siumai_provider_google_vertex as google_vertex;
#[cfg(feature = "groq")]
pub use siumai_provider_groq as groq;
#[cfg(feature = "minimaxi")]
pub use siumai_provider_minimaxi as minimaxi;
#[cfg(feature = "ollama")]
pub use siumai_provider_ollama as ollama;
#[cfg(feature = "togetherai")]
pub use siumai_provider_togetherai as togetherai;
#[cfg(feature = "xai")]
pub use siumai_provider_xai as xai;
