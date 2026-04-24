//! `DeepSeek` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation.
//!
//! # Architecture
//! - `builder.rs` - Builder that delegates to `openai().compatible("deepseek")`
//! - `models.rs`  - Model constants (re-exported from OpenAI-compatible catalog)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Provider::deepseek()
//!         .api_key("your-api-key")
//!         .model(models::openai_compatible::deepseek::CHAT)
//!         .build()
//!         .await?;
//!
//!     let messages = vec![user!("Hello, DeepSeek!")];
//!     let response = client.chat(messages).await?;
//!     println!("{}", response.content_text().unwrap_or_default());
//!     Ok(())
//! }
//! ```

pub mod builder;
mod client;
pub mod config;
pub mod ext;
pub mod models;
pub mod settings;
pub mod spec;

pub use builder::DeepSeekBuilder;
pub use client::DeepSeekClient;
pub use config::DeepSeekConfig;
pub use settings::DeepSeekProviderSettings;
pub use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleErrorData as DeepSeekErrorData;

// Provider-owned typed options live at the crate root; re-export them under the provider path.
#[allow(deprecated)]
pub use crate::provider_options::{
    DeepSeekChatOptions, DeepSeekLanguageModelOptions, DeepSeekOptions,
};
pub use spec::DeepSeekSpec;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::DeepSeekErrorData;

    #[test]
    fn deepseek_error_data_deserializes_ai_sdk_shape() {
        let data: DeepSeekErrorData = serde_json::from_value(serde_json::json!({
            "error": {
                "message": "rate limit exceeded",
                "type": "rate_limit_error",
                "code": "too_many_requests"
            }
        }))
        .expect("deepseek error data should deserialize");

        assert_eq!(data.error.message, "rate limit exceeded");
        assert_eq!(data.error.error_type.as_deref(), Some("rate_limit_error"));
        assert_eq!(
            data.error.code,
            Some(serde_json::json!("too_many_requests"))
        );
    }
}
