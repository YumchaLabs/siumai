//! Custom Provider Options Trait
//!
//! This module provides the `CustomProviderOptions` trait that allows users
//! to extend the system with their own provider-specific features.

use crate::error::LlmError;

/// Trait for custom provider options
///
/// Implement this trait to add support for custom provider features.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::{CustomProviderOptions, ProviderOptions};
///
/// #[derive(Debug, Clone)]
/// struct MyCustomFeature {
///     pub custom_param: String,
/// }
///
/// impl CustomProviderOptions for MyCustomFeature {
///     fn provider_id(&self) -> &str {
///         "my-provider"
///     }
///
///     fn to_json(&self) -> Result<serde_json::Value, LlmError> {
///         Ok(serde_json::json!({
///             "custom_param": self.custom_param
///         }))
///     }
/// }
///
/// // Usage
/// let feature = MyCustomFeature { custom_param: "value".to_string() };
/// let options = ProviderOptions::from_custom(feature)?;
/// ```
pub trait CustomProviderOptions {
    /// Get the provider ID for this custom feature
    fn provider_id(&self) -> &str;

    /// Convert the custom options to JSON
    fn to_json(&self) -> Result<serde_json::Value, LlmError>;
}
