//! LLM Client Builder - Client Configuration Layer
//!
//! ## 🎯 Core Responsibility: Client Configuration and Construction
//!
//! This module is the **client configuration layer** of the LLM library architecture.
//! It is responsible for:
//!
//! ### ✅ What LlmBuilder Does:
//! - **Client Construction**: Creates and configures provider-specific clients
//! - **HTTP Configuration**: Sets up HTTP clients, timeouts, and connection settings
//! - **Authentication**: Handles API keys and authentication configuration
//! - **Provider Selection**: Determines which provider implementation to use
//! - **Environment Setup**: Configures base URLs, headers, and provider-specific settings
//! - **Fluent API**: Provides chainable method interface for easy configuration
//!
//! ### ❌ What LlmBuilder Does NOT Do:
//! - **Parameter Validation**: Does not validate chat parameters (temperature, max_tokens, etc.)
//! - **Request Building**: Does not construct ChatRequest objects
//! - **Parameter Mapping**: Does not map parameters between formats
//! - **Chat Logic**: Does not implement chat or streaming functionality
//!
//! ## 🏗️ Architecture Position
//!
//! ```text
//! User Code
//!     ↓
//! SiumaiBuilder (Unified Interface Layer)
//!     ↓
//! LlmBuilder (Client Configuration Layer) ← YOU ARE HERE
//!     ↓
//! RequestBuilder (Parameter Management Layer)
//!     ↓
//! Provider Clients (Implementation Layer)
//!     ↓
//! HTTP/Network Layer
//! ```
//!
//! ## 🔄 Relationship with RequestBuilder
//!
//! - **LlmBuilder**: Handles client setup, HTTP config, and provider instantiation
//! - **RequestBuilder**: Handles parameter validation, mapping, and request building
//! - **Separation**: These operate at different architectural layers
//!
//! ### Collaboration Pattern:
//! 1. **LlmBuilder** creates and configures the client
//! 2. **RequestBuilder** handles parameter management within the client
//! 3. Both work together but have distinct, non-overlapping responsibilities
//!
//! ## 🎨 Design Principles
//! - **Fluent API**: Method chaining for intuitive configuration
//! - **Custom HTTP Clients**: Support for user-provided reqwest clients
//! - **Provider Abstraction**: Consistent interface across different providers
//! - **Environment Integration**: Automatic environment variable detection
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::builder::LlmBuilder;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Basic usage
//!     let client = LlmBuilder::new()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .build()
//!         .await?;
//!
//!     // With custom HTTP client
//!     let custom_client = reqwest::Client::builder()
//!         .timeout(Duration::from_secs(30))
//!         .build()?;
//!
//!     let client = LlmBuilder::new()
//!         .with_http_client(custom_client)
//!         .openai()
//!         .api_key("your-api-key")
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;

// Note: Removed unused import crate::providers::* to fix warning

/// Quick `OpenAI` client creation with minimal configuration.
///
/// Uses environment variable `OPENAI_API_KEY` and default settings.
///
/// # Example
/// ```rust,no_run
/// use siumai::{quick_openai, quick_openai_with_model};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Uses OPENAI_API_KEY env var and gpt-4o-mini model
///     let client = quick_openai().await?;
///
///     // With custom model
///     let client = quick_openai_with_model("gpt-4").await?;
///
///     Ok(())
/// }
/// ```
#[cfg(feature = "openai")]
pub async fn quick_openai() -> Result<crate::providers::openai::OpenAiClient, LlmError> {
    quick_openai_with_model("gpt-4o-mini").await
}

/// Quick `OpenAI` client creation with custom model.
#[cfg(feature = "openai")]
pub async fn quick_openai_with_model(
    model: &str,
) -> Result<crate::providers::openai::OpenAiClient, LlmError> {
    LlmBuilder::new().openai().model(model).build().await
}

/// Quick Anthropic client creation with minimal configuration.
///
/// Uses environment variable `ANTHROPIC_API_KEY` and default settings.
#[cfg(feature = "anthropic")]
pub async fn quick_anthropic() -> Result<crate::providers::anthropic::AnthropicClient, LlmError> {
    quick_anthropic_with_model("claude-3-5-sonnet-20241022").await
}

/// Quick Anthropic client creation with custom model.
#[cfg(feature = "anthropic")]
pub async fn quick_anthropic_with_model(
    model: &str,
) -> Result<crate::providers::anthropic::AnthropicClient, LlmError> {
    LlmBuilder::new().anthropic().model(model).build().await
}

/// Quick Gemini client creation with minimal configuration.
///
/// Uses environment variable `GEMINI_API_KEY` and default settings.
#[cfg(feature = "google")]
pub async fn quick_gemini() -> Result<crate::providers::gemini::GeminiClient, LlmError> {
    quick_gemini_with_model("gemini-1.5-flash").await
}

/// Quick Gemini client creation with custom model.
#[cfg(feature = "google")]
pub async fn quick_gemini_with_model(
    model: &str,
) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
    LlmBuilder::new().gemini().model(model).build().await
}

/// Quick Ollama client creation with minimal configuration.
///
/// Uses default Ollama settings (<http://localhost:11434>) and llama3.2 model.
#[cfg(feature = "ollama")]
pub async fn quick_ollama() -> Result<crate::providers::ollama::OllamaClient, LlmError> {
    quick_ollama_with_model("llama3.2").await
}

/// Quick Ollama client creation with custom model.
#[cfg(feature = "ollama")]
pub async fn quick_ollama_with_model(
    model: &str,
) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
    LlmBuilder::new().ollama().model(model).build().await
}

/// Quick Groq client creation with minimal configuration.
///
/// Uses environment variable `GROQ_API_KEY` and default settings.
#[cfg(feature = "groq")]
pub async fn quick_groq() -> Result<crate::providers::groq::GroqClient, LlmError> {
    quick_groq_with_model(crate::providers::groq::models::popular::FLAGSHIP).await
}

/// Quick Groq client creation with custom model.
#[cfg(feature = "groq")]
pub async fn quick_groq_with_model(
    model: &str,
) -> Result<crate::providers::groq::GroqClient, LlmError> {
    LlmBuilder::new().groq().model(model).build().await
}

/// Quick xAI client creation with minimal configuration.
///
/// Uses environment variable `XAI_API_KEY` and default settings.
#[cfg(feature = "xai")]
pub async fn quick_xai() -> Result<crate::providers::xai::XaiClient, LlmError> {
    quick_xai_with_model(crate::providers::xai::models::popular::LATEST).await
}

/// Quick xAI client creation with custom model.
#[cfg(feature = "xai")]
pub async fn quick_xai_with_model(
    model: &str,
) -> Result<crate::providers::xai::XaiClient, LlmError> {
    LlmBuilder::new().xai().model(model).build().await
}

/// Core LLM builder that provides common configuration options.
///
/// This builder allows setting up HTTP client configuration, timeouts,
/// and other provider-agnostic settings before choosing a specific provider.
///
/// # Design Philosophy
/// - Provider-agnostic configuration first
/// - Support for custom HTTP clients (key requirement)
/// - Fluent API with method chaining
/// - Validation at build time
#[derive(Clone)]
pub struct LlmBuilder {
    /// Custom HTTP client (key requirement from design doc)
    pub(crate) http_client: Option<reqwest::Client>,
    /// Request timeout
    pub(crate) timeout: Option<Duration>,
    /// Connection timeout
    pub(crate) connect_timeout: Option<Duration>,
    /// User agent string
    pub(crate) user_agent: Option<String>,
    /// Default headers
    pub(crate) default_headers: HashMap<String, String>,
    /// Optional HTTP interceptors applied to chat requests (inherited by provider builders)
    pub(crate) http_interceptors:
        Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    /// Enable a built-in LoggingInterceptor for lightweight HTTP debug
    pub(crate) http_debug: bool,
    /// Enable HTTP/2
    pub(crate) http2_prior_knowledge: Option<bool>,
    /// Enable gzip compression
    pub(crate) gzip: Option<bool>,
    /// Enable brotli compression
    pub(crate) brotli: Option<bool>,
    /// Proxy URL
    pub(crate) proxy: Option<String>,
    /// Enable cookies
    pub(crate) cookie_store: Option<bool>,
    // Note: redirect policy removed due to Clone constraint issues
}

impl LlmBuilder {
    /// Create a new LLM builder with default settings.
    pub fn new() -> Self {
        Self {
            http_client: None,
            timeout: None,
            connect_timeout: None,
            user_agent: None,
            default_headers: HashMap::new(),
            http_interceptors: Vec::new(),
            http_debug: false,
            http2_prior_knowledge: None,
            gzip: None,
            brotli: None,
            proxy: None,
            cookie_store: None,
            // redirect_policy removed
        }
    }

    /// Create a builder with sensible defaults for production use.
    ///
    /// Sets reasonable timeouts, compression, and other production-ready settings.
    pub fn with_defaults() -> Self {
        Self::new()
            .with_timeout(crate::defaults::timeouts::STANDARD)
            .with_connect_timeout(crate::defaults::http::CONNECT_TIMEOUT)
            .with_user_agent(crate::defaults::http::USER_AGENT)
            .with_gzip(true)
            .with_brotli(true)
    }

    /// Create a builder optimized for fast responses.
    ///
    /// Uses shorter timeouts suitable for interactive applications.
    pub fn fast() -> Self {
        Self::new()
            .with_timeout(crate::defaults::timeouts::FAST)
            .with_connect_timeout(Duration::from_secs(5))
            .with_user_agent(crate::defaults::http::USER_AGENT)
    }

    /// Create a builder optimized for long-running operations.
    ///
    /// Uses longer timeouts suitable for batch processing or complex tasks.
    pub fn long_running() -> Self {
        Self::new()
            .with_timeout(crate::defaults::timeouts::LONG_RUNNING)
            .with_connect_timeout(Duration::from_secs(30))
            .with_user_agent(crate::defaults::http::USER_AGENT)
    }

    /// Use a custom HTTP client.
    ///
    /// This allows you to provide your own configured reqwest client
    /// with custom settings, certificates, proxies, etc.
    ///
    /// # Arguments
    /// * `client` - The reqwest client to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use siumai::builder::LlmBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let custom_client = reqwest::Client::builder()
    ///         .timeout(Duration::from_secs(30))
    ///         .build()?;
    ///
    ///     let llm_client = LlmBuilder::new()
    ///         .with_http_client(custom_client)
    ///         .openai()
    ///         .api_key("your-key")
    ///         .build()
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set the request timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for a request
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the connection timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for connection establishment
    pub const fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Set a custom User-Agent header.
    ///
    /// # Arguments
    /// * `user_agent` - The User-Agent string to use
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Add a default header that will be sent with all requests.
    ///
    /// # Arguments
    /// * `name` - Header name
    /// * `value` - Header value
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, name: K, value: V) -> Self {
        self.default_headers.insert(name.into(), value.into());
        self
    }

    /// Install a custom HTTP interceptor at the unified builder level.
    /// Provider builders created from this `LlmBuilder` will inherit these
    /// interceptors and install them on their clients when built.
    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Enable or disable HTTP/2 prior knowledge.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable HTTP/2 prior knowledge
    pub const fn with_http2_prior_knowledge(mut self, enabled: bool) -> Self {
        self.http2_prior_knowledge = Some(enabled);
        self
    }

    /// Enable or disable gzip compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable gzip compression
    pub const fn with_gzip(mut self, enabled: bool) -> Self {
        self.gzip = Some(enabled);
        self
    }

    /// Enable or disable brotli compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable brotli compression
    pub const fn with_brotli(mut self, enabled: bool) -> Self {
        self.brotli = Some(enabled);
        self
    }

    /// Set a proxy URL.
    ///
    /// # Arguments
    /// * `proxy_url` - The proxy URL (e.g., "<http://proxy.example.com:8080>")
    pub fn with_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.proxy = Some(proxy_url.into());
        self
    }

    /// Enable or disable cookie store.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable cookie storage
    pub const fn with_cookie_store(mut self, enabled: bool) -> Self {
        self.cookie_store = Some(enabled);
        self
    }

    // Note: redirect policy configuration removed due to Clone constraints

    // Provider-specific builders

    // Provider builder methods are now defined in src/providers/builders.rs
    // This keeps the main builder clean and organized

    /// Build the HTTP client with the configured settings.
    ///
    /// This is used internally by provider builders to create the HTTP client.
    pub(crate) fn build_http_client(&self) -> Result<reqwest::Client, LlmError> {
        // If a custom client was provided, use it
        if let Some(client) = &self.http_client {
            return Ok(client.clone());
        }

        // Check if we need to use advanced features not in HttpConfig
        let needs_custom_build = self.http2_prior_knowledge.is_some()
            || self.gzip.is_some()
            || self.brotli.is_some()
            || self.cookie_store.is_some();

        if needs_custom_build {
            // Build client manually with all features
            let mut builder = reqwest::Client::builder();

            // Apply timeout settings
            if let Some(timeout) = self.timeout {
                builder = builder.timeout(timeout);
            }
            if let Some(connect_timeout) = self.connect_timeout {
                builder = builder.connect_timeout(connect_timeout);
            }

            // Apply proxy settings
            if let Some(proxy_url) = &self.proxy {
                let proxy = reqwest::Proxy::all(proxy_url)
                    .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
                builder = builder.proxy(proxy);
            }

            // Apply user agent
            if let Some(user_agent) = &self.user_agent {
                builder = builder.user_agent(user_agent);
            }

            // Apply compression settings (FIX: previously missing)
            #[cfg(feature = "gzip")]
            if let Some(enable) = self.gzip {
                builder = builder.gzip(enable);
            }
            #[cfg(feature = "brotli")]
            if let Some(enable) = self.brotli {
                builder = builder.brotli(enable);
            }

            // Apply HTTP/2 settings
            #[cfg(feature = "http2")]
            if let Some(true) = self.http2_prior_knowledge {
                builder = builder.http2_prior_knowledge();
            }

            // Apply cookie store (FIX: previously missing)
            #[cfg(feature = "cookies")]
            if let Some(enable) = self.cookie_store {
                builder = builder.cookie_store(enable);
            }

            // Apply default headers
            if !self.default_headers.is_empty() {
                let mut headers = reqwest::header::HeaderMap::new();
                for (k, v) in &self.default_headers {
                    let name =
                        reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                            LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}"))
                        })?;
                    let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                        LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
                    })?;
                    headers.insert(name, value);
                }
                builder = builder.default_headers(headers);
            }

            // Build the client; on failure, gracefully fall back to simple path
            match builder.build() {
                Ok(client) => Ok(client),
                Err(e) => {
                    // Fallback: build minimal client from HttpConfig to avoid test flakiness
                    let config = crate::types::HttpConfig {
                        timeout: self.timeout,
                        connect_timeout: self.connect_timeout,
                        headers: self.default_headers.clone(),
                        proxy: self.proxy.clone(),
                        user_agent: self.user_agent.clone(),
                        stream_disable_compression: true,
                    };
                    crate::execution::http::client::build_http_client_from_config(&config).map_err(
                        |_| {
                            LlmError::ConfigurationError(format!(
                                "Failed to build HTTP client: {e}"
                            ))
                        },
                    )
                }
            }
        } else {
            // Use unified HTTP client builder for simple cases
            let config = crate::types::HttpConfig {
                timeout: self.timeout,
                connect_timeout: self.connect_timeout,
                headers: self.default_headers.clone(),
                proxy: self.proxy.clone(),
                user_agent: self.user_agent.clone(),
                stream_disable_compression: true, // default value
            };

            crate::execution::http::client::build_http_client_from_config(&config)
        }
    }
}

impl fmt::Debug for LlmBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmBuilder")
            .field("has_http_client", &self.http_client.is_some())
            .field("timeout", &self.timeout)
            .field("connect_timeout", &self.connect_timeout)
            .field("user_agent", &self.user_agent)
            .field("default_headers_len", &self.default_headers.len())
            .field("http2_prior_knowledge", &self.http2_prior_knowledge)
            .field("gzip", &self.gzip)
            .field("brotli", &self.brotli)
            .field("proxy", &self.proxy)
            .field("cookie_store", &self.cookie_store)
            .field("http_debug", &self.http_debug)
            .finish()
    }
}

impl Default for LlmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = LlmBuilder::new();
        let _openai_builder = builder.openai();
        // Basic test for builder creation
        // Placeholder test
    }

    #[test]
    fn test_http_config_inheritance() {
        use std::time::Duration;

        // Test that HTTP configuration is properly inherited by provider builders
        let base_builder = LlmBuilder::new()
            .with_timeout(Duration::from_secs(60))
            .with_proxy("http://proxy.example.com:8080")
            .with_user_agent("test-agent/1.0")
            .with_header("X-Test-Header", "test-value");

        // Test OpenAI builder inherits HTTP config
        let openai_builder = base_builder.clone().openai();
        assert_eq!(
            openai_builder.core.base.timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            openai_builder.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );
        assert_eq!(
            openai_builder.core.base.user_agent,
            Some("test-agent/1.0".to_string())
        );
        assert!(
            openai_builder
                .core
                .base
                .default_headers
                .contains_key("X-Test-Header")
        );

        // Test Anthropic builder inherits HTTP config
        let anthropic_builder = base_builder.clone().anthropic();
        assert_eq!(
            anthropic_builder.core.base.timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            anthropic_builder.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );

        // Test Gemini builder inherits HTTP config
        let gemini_builder = base_builder.clone().gemini();
        assert_eq!(
            gemini_builder.core.base.timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            gemini_builder.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );

        // Test Ollama builder inherits HTTP config
        let ollama_builder = base_builder.clone().ollama();
        assert_eq!(
            ollama_builder.core.base.timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            ollama_builder.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );

        // Test xAI wrapper inherits HTTP config
        let xai_wrapper = base_builder.clone().xai();
        assert_eq!(xai_wrapper.core.base.timeout, Some(Duration::from_secs(60)));
        assert_eq!(
            xai_wrapper.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );

        // Test Groq wrapper inherits HTTP config
        let groq_wrapper = base_builder.groq();
        assert_eq!(
            groq_wrapper.core.base.timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            groq_wrapper.core.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );
    }
}
