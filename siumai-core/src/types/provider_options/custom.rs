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
/// use siumai::types::{ChatRequest, CustomProviderOptions};
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
/// let (provider_id, value) = feature.to_provider_options_map_entry()?;
/// let req = ChatRequest::new(messages).with_provider_option(provider_id, value);
/// ```
pub trait CustomProviderOptions {
    /// Get the provider ID for this custom feature
    fn provider_id(&self) -> &str;

    /// Convert the custom options to JSON
    fn to_json(&self) -> Result<serde_json::Value, LlmError>;

    /// Convert this custom options into a `(provider_id, json_value)` pair for `ProviderOptionsMap`.
    fn to_provider_options_map_entry(&self) -> Result<(String, serde_json::Value), LlmError> {
        Ok((self.provider_id().to_ascii_lowercase(), self.to_json()?))
    }
}
