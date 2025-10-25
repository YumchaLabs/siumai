//! Builder macros for reducing code duplication across provider builders
//!
//! This module provides macros to automatically generate common builder methods
//! that are shared across all provider builders.
//!
//! ## Purpose
//!
//! These macros are designed to reduce boilerplate code when creating new provider builders.
//! They generate common configuration methods (api_key, base_url, model, temperature, etc.)
//! and HTTP configuration delegation methods (timeout, retry, http_debug, etc.).
//!
//! ## When to Use
//!
//! - **New Provider Implementations**: Use these macros when adding a new provider to reduce boilerplate
//! - **Simple Providers**: Providers with standard configuration patterns benefit most from these macros
//! - **Prototyping**: Quickly scaffold a new provider builder
//!
//! ## When NOT to Use
//!
//! - **Existing Mature Providers**: Don't refactor existing, well-tested builders unless there's a strong reason
//! - **Custom Logic**: If a method needs custom logic (e.g., OpenAI's `model()` sets both `self.model` and `self.common_params.model`), implement it manually
//! - **Different Signatures**: If a provider has different parameter types or validation logic
//!
//! ## Example: Creating a New Provider Builder
//!
//! ```rust,ignore
//! use crate::core::builder_core::ProviderCore;
//! use crate::types::CommonParams;
//! use crate::LlmBuilder;
//!
//! pub struct MyProviderBuilder {
//!     pub(crate) core: ProviderCore,
//!     api_key: Option<String>,
//!     base_url: Option<String>,
//!     model: Option<String>,
//!     common_params: CommonParams,
//!     my_provider_params: MyProviderParams,
//! }
//!
//! impl MyProviderBuilder {
//!     pub fn new(base: LlmBuilder) -> Self {
//!         Self {
//!             core: ProviderCore::new(base),
//!             api_key: None,
//!             base_url: None,
//!             model: None,
//!             common_params: CommonParams::default(),
//!             my_provider_params: MyProviderParams::default(),
//!         }
//!     }
//!
//!     // Generate common builder methods
//!     crate::impl_common_builder_methods! {
//!         api_key: api_key,
//!         base_url: base_url,
//!         model: model,
//!         temperature: common_params.temperature,
//!         max_tokens: common_params.max_tokens,
//!         top_p: common_params.top_p,
//!     }
//!
//!     // Generate HTTP configuration delegation methods
//!     crate::impl_core_delegation_methods!();
//!
//!     // Add provider-specific methods here
//!     pub fn my_custom_param(mut self, value: String) -> Self {
//!         self.my_provider_params.custom = Some(value);
//!         self
//!     }
//!
//!     pub async fn build(self) -> Result<MyProviderClient, LlmError> {
//!         // Build logic here
//!     }
//! }
//! ```

/// Generate common configuration methods for provider builders
///
/// This macro generates the following methods:
/// - `api_key()` - Set the API key
/// - `base_url()` - Set the base URL
/// - `model()` - Set the model name
/// - `temperature()` - Set the temperature
/// - `max_tokens()` - Set max tokens
/// - `top_p()` - Set top-p value
///
/// # Usage
///
/// ```rust,ignore
/// impl MyProviderBuilder {
///     impl_common_builder_methods! {
///         api_key: api_key,
///         base_url: base_url,
///         model: model,
///         temperature: common_params.temperature,
///         max_tokens: common_params.max_tokens,
///         top_p: common_params.top_p,
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_common_builder_methods {
    (
        $(#[$meta:meta])*
        api_key: $api_key_field:ident,
        base_url: $base_url_field:ident,
        model: $model_field:ident,
        temperature: $($temp_path:ident).+,
        max_tokens: $($max_tokens_path:ident).+,
        top_p: $($top_p_path:ident).+
        $(,)?
    ) => {
        $(#[$meta])*
        /// Set the API key
        ///
        /// # Arguments
        /// * `key` - The API key to use
        pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
            self.$api_key_field = Some(key.into());
            self
        }

        $(#[$meta])*
        /// Set the base URL
        ///
        /// # Arguments
        /// * `url` - The base URL for the API endpoint
        pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
            self.$base_url_field = Some(url.into());
            self
        }

        $(#[$meta])*
        /// Set the model name
        ///
        /// # Arguments
        /// * `model` - The model identifier
        pub fn model<S: Into<String>>(mut self, model: S) -> Self {
            self.$model_field = Some(model.into());
            self
        }

        $(#[$meta])*
        /// Set the temperature for generation
        ///
        /// Controls randomness in the output. Higher values (e.g., 1.0) make output more random,
        /// lower values (e.g., 0.2) make it more focused and deterministic.
        ///
        /// # Arguments
        /// * `temperature` - Temperature value, typically between 0.0 and 2.0
        pub const fn temperature(mut self, temperature: f32) -> Self {
            self.$($temp_path).+ = Some(temperature);
            self
        }

        $(#[$meta])*
        /// Set the maximum number of tokens to generate
        ///
        /// # Arguments
        /// * `max_tokens` - Maximum number of tokens in the response
        pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
            self.$($max_tokens_path).+ = Some(max_tokens);
            self
        }

        $(#[$meta])*
        /// Set the top-p (nucleus sampling) value
        ///
        /// An alternative to temperature sampling. The model considers the results of tokens
        /// with top_p probability mass.
        ///
        /// # Arguments
        /// * `top_p` - Top-p value, typically between 0.0 and 1.0
        pub const fn top_p(mut self, top_p: f32) -> Self {
            self.$($top_p_path).+ = Some(top_p);
            self
        }
    };
}

/// Generate ProviderCore delegation methods
///
/// This macro generates methods that delegate to the `ProviderCore` for HTTP configuration:
/// - `timeout()` - Set request timeout
/// - `with_retry()` - Configure retry behavior
/// - `http_debug()` - Enable HTTP debugging
/// - `with_http_client()` - Use custom HTTP client
///
/// # Usage
///
/// ```rust,ignore
/// impl MyProviderBuilder {
///     impl_core_delegation_methods!();
/// }
/// ```
#[macro_export]
macro_rules! impl_core_delegation_methods {
    () => {
        /// Set the request timeout
        ///
        /// # Arguments
        /// * `timeout` - Duration to wait before timing out
        pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
            self.core = self.core.timeout(timeout);
            self
        }

        /// Configure retry behavior for failed requests
        ///
        /// # Arguments
        /// * `retry` - Retry options (e.g., `RetryOptions::backoff()`)
        ///
        /// # Example
        /// ```rust,ignore
        /// use siumai::retry_api::RetryOptions;
        ///
        /// let client = builder
        ///     .with_retry(RetryOptions::backoff())
        ///     .build()
        ///     .await?;
        /// ```
        pub fn with_retry(mut self, retry: $crate::retry_api::RetryOptions) -> Self {
            self.core = self.core.with_retry(retry);
            self
        }

        /// Enable HTTP request/response debugging
        ///
        /// When enabled, logs HTTP requests and responses (without sensitive data)
        ///
        /// # Arguments
        /// * `enabled` - Whether to enable debug logging
        pub fn http_debug(mut self, enabled: bool) -> Self {
            self.core = self.core.http_debug(enabled);
            self
        }

        /// Use a custom HTTP client
        ///
        /// # Arguments
        /// * `client` - Custom `reqwest::Client` instance
        pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
            self.core = self.core.with_http_client(client);
            self
        }
    };
}

/// Generate a complete provider builder implementation
///
/// This macro combines both common builder methods and core delegation methods.
///
/// # Usage
///
/// ```rust,ignore
/// impl MyProviderBuilder {
///     impl_provider_builder! {
///         api_key: api_key,
///         base_url: base_url,
///         model: model,
///         temperature: common_params.temperature,
///         max_tokens: common_params.max_tokens,
///         top_p: common_params.top_p,
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_provider_builder {
    (
        api_key: $api_key_field:ident,
        base_url: $base_url_field:ident,
        model: $model_field:ident,
        temperature: $($temp_path:ident).+,
        max_tokens: $($max_tokens_path:ident).+,
        top_p: $($top_p_path:ident).+
        $(,)?
    ) => {
        $crate::impl_common_builder_methods! {
            api_key: $api_key_field,
            base_url: $base_url_field,
            model: $model_field,
            temperature: $($temp_path).+,
            max_tokens: $($max_tokens_path).+,
            top_p: $($top_p_path).+,
        }

        $crate::impl_core_delegation_methods!();
    };
}

#[cfg(test)]
mod tests {
    use crate::core::builder_core::ProviderCore;
    use crate::types::CommonParams;
    use crate::LlmBuilder;

    struct TestBuilder {
        core: ProviderCore,
        api_key: Option<String>,
        base_url: Option<String>,
        model: Option<String>,
        common_params: CommonParams,
    }

    impl TestBuilder {
        fn new() -> Self {
            Self {
                core: ProviderCore::new(LlmBuilder::new()),
                api_key: None,
                base_url: None,
                model: None,
                common_params: CommonParams::default(),
            }
        }

        impl_provider_builder! {
            api_key: api_key,
            base_url: base_url,
            model: model,
            temperature: common_params.temperature,
            max_tokens: common_params.max_tokens,
            top_p: common_params.top_p,
        }
    }

    #[test]
    fn test_common_methods() {
        let builder = TestBuilder::new()
            .api_key("test-key")
            .base_url("https://test.com")
            .model("test-model")
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9);

        assert_eq!(builder.api_key, Some("test-key".to_string()));
        assert_eq!(builder.base_url, Some("https://test.com".to_string()));
        assert_eq!(builder.model, Some("test-model".to_string()));
        assert_eq!(builder.common_params.temperature, Some(0.7));
        assert_eq!(builder.common_params.max_tokens, Some(1000));
        assert_eq!(builder.common_params.top_p, Some(0.9));
    }

    #[test]
    fn test_core_delegation() {
        use std::time::Duration;

        let builder = TestBuilder::new().timeout(Duration::from_secs(30));

        assert_eq!(
            builder.core.http_config.timeout,
            Some(Duration::from_secs(30))
        );
    }
}

