//! Siumai LLM Interface
//!
//! This module provides the main siumai interface for calling different provider functionality,
//! similar to `llm_dart`'s approach. It uses dynamic dispatch to route calls to the
//! appropriate provider implementation.

mod siumai;
pub use siumai::{ProviderMetadata, Siumai};

mod proxies;
#[allow(deprecated)]
pub use proxies::VisionCapabilityProxy;
pub use proxies::{AudioCapabilityProxy, EmbeddingCapabilityProxy};

/// Unified Interface Builder - Provider Abstraction Layer
///
/// ## Core Responsibility: Unified Provider Interface
///
/// SiumaiBuilder provides a **unified interface** for creating LLM clients
/// across different providers while abstracting away provider-specific details.
///
/// ### What SiumaiBuilder Does:
/// - **Provider Abstraction**: Unified interface for all LLM providers
/// - **Parameter Unification**: Common parameter interface (temperature, max_tokens, etc.)
/// - **Reasoning Abstraction**: Unified reasoning interface across providers
/// - **Configuration Validation**: Validates configuration before client creation
/// - **Provider Selection**: Determines which provider to use based on configuration
/// - **Parameter Delegation**: Delegates to appropriate builders for actual construction
///
/// ### What SiumaiBuilder Does NOT Do:
/// - **Direct Client Creation**: Does not directly create HTTP clients
/// - **Parameter Mapping**: Does not handle provider-specific parameter mapping
/// - **HTTP Configuration**: Does not configure HTTP settings directly
///
/// ## Architecture Position
///
/// ```text
/// User Code
///     鈫?/// SiumaiBuilder (Unified Interface Layer) 鈫?YOU ARE HERE
///     鈫?/// 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?/// 鈫?                鈫?                                    鈫?/// LlmBuilder        RequestBuilder                Provider Clients
/// (Client Config)   (Parameter Management)        (Implementation)
/// ```
///
/// ## Delegation Pattern
///
/// SiumaiBuilder acts as a **coordinator** that delegates to specialized builders:
///
/// 1. **Parameter Validation**: Uses RequestBuilder for parameter validation
/// 2. **Client Construction**: Uses LlmBuilder or direct client constructors
/// 3. **Provider Selection**: Chooses appropriate implementation based on provider type
///
/// ### Example Flow:
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // 1. User configures through unified interface
///     let client = Siumai::builder()
///         .anthropic()                    // Provider selection
///         .api_key("your-api-key")        // Required API key
///         .model("claude-3-5-sonnet")     // Common parameter
///         .temperature(0.7)               // Common parameter
///         .reasoning(true)                // Unified reasoning
///         .build().await?;                // Delegation to appropriate builders
///     Ok(())
/// }
/// ```
///
/// This design allows users to switch providers with minimal code changes
/// while maintaining access to provider-specific features when needed.
mod siumai_builder;
pub use siumai_builder::SiumaiBuilder;

// Keep module slim by moving heavy build logic out
pub mod build;
