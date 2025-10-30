//! Custom Provider Parameter Additions
//!
//! This example demonstrates how to add new parameters to providers without
//! waiting for library updates. When providers release new features, you can
//! use them immediately.
//!
//! # Use Cases
//!
//! - Provider releases a new feature before the library supports it
//! - Experimenting with beta/preview features
//! - Using private or customized provider extensions
//!
//! # Run
//!
//! ```bash
//! cargo run --example custom_provider_parameters --features xai
//! ```

use siumai::prelude::*;
use siumai::types::{CustomProviderOptions, ProviderOptions};

/// Example: Add new feature parameters for xAI
///
/// Assume xAI just released "deferred" mode and "parallel_function_calling".
/// The library may not support them yet, but we can use them immediately.
#[derive(Debug, Clone)]
pub struct XaiNewFeatures {
    /// Deferred mode — assumed new xAI feature
    pub deferred: Option<bool>,
    /// Parallel function calling — assumed new xAI feature
    pub parallel_function_calling: Option<bool>,
}

impl CustomProviderOptions for XaiNewFeatures {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut map = serde_json::Map::new();

        if let Some(deferred) = self.deferred {
            map.insert("deferred".to_string(), serde_json::Value::Bool(deferred));
        }

        if let Some(parallel) = self.parallel_function_calling {
            map.insert(
                "parallel_function_calling".to_string(),
                serde_json::Value::Bool(parallel),
            );
        }

        Ok(serde_json::Value::Object(map))
    }
}

/// Example: Add new feature parameters for OpenAI
#[derive(Debug, Clone)]
pub struct OpenAiNewFeatures {
    /// Custom parameter — assumed new OpenAI feature
    pub reasoning_effort: Option<String>,
    /// Experimental mode — assumed new OpenAI feature
    pub experimental_mode: Option<bool>,
}

impl CustomProviderOptions for OpenAiNewFeatures {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut map = serde_json::Map::new();

        if let Some(effort) = &self.reasoning_effort {
            map.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String(effort.clone()),
            );
        }

        if let Some(experimental) = self.experimental_mode {
            map.insert(
                "experimental_mode".to_string(),
                serde_json::Value::Bool(experimental),
            );
        }

        Ok(serde_json::Value::Object(map))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom Add Provider Parameters Example\n");

    // ============================================================
    // Example 1: Add new features to xAI
    // ============================================================
    println!("📝 Example 1: Add new features to xAI");
    println!("================================================\n");

    let xai_features = XaiNewFeatures {
        deferred: Some(true),
        parallel_function_calling: Some(true),
    };

    // Convert to ProviderOptions
    let options = ProviderOptions::from_custom(xai_features)?;

    println!("✅ Created custom xAI parameters:");
    println!("   {:?}\n", options);

    // ============================================================
    // Example 2: Use with ChatRequest
    // ============================================================
    println!("📝 Example 2: Use with ChatRequest");
    println!("=====================================\n");

    let openai_features = OpenAiNewFeatures {
        reasoning_effort: Some("high".to_string()),
        experimental_mode: Some(true),
    };

    let request = ChatRequest::new(vec![ChatMessage::user("Analyze this data").build()])
        .with_provider_options(ProviderOptions::from_custom(openai_features)?);

    println!("✅ Created ChatRequest with custom parameters");
    println!("   Provider: {:?}", request.provider_options.provider_id());
    println!();

    // ============================================================
    // Example 3: Direct HashMap usage (more flexible)
    // ============================================================
    println!("📝 Example 3: Direct HashMap usage (more flexible)");
    println!("====================================================\n");

    use std::collections::HashMap;

    let mut custom_params = HashMap::new();
    custom_params.insert("new_feature".to_string(), serde_json::json!("enabled"));
    custom_params.insert("priority_level".to_string(), serde_json::json!(5));

    let options = ProviderOptions::Custom {
        provider_id: "xai".to_string(),
        options: custom_params,
    };

    println!("✅ Created flexible custom parameters:");
    println!("   {:?}\n", options);

    // ============================================================
    // Best Practices
    // ============================================================
    println!("💡 Best Practices");
    println!("==============================\n");

    println!("✅ DO:");
    println!("   - Use custom parameters only for features not yet in the library");
    println!("   - Use for provider Beta/experimental features");
    println!();

    println!("❌ DON'T:");
    println!("   - Don't use custom parameters for features with built-in support");
    println!("   - Example: temperature, max_tokens should use CommonParams");
    println!();

    // ============================================================
    // How It Works
    // ============================================================
    println!("🔍 How It Works");
    println!("===========================\n");

    println!("1. Implement CustomProviderOptions trait");
    println!();

    println!("2. Convert using ProviderOptions::from_custom()");
    println!();

    println!("3. Library automatically injects parameters into API request");
    println!("   (via ProviderSpec::chat_before_send() hook)");
    println!();

    println!("4. Provider receives complete request parameters");
    println!();

    println!("🎉 Example complete!\n");
    println!("📚 Key Takeaways:");
    println!("   1. Use new provider features immediately without waiting for library updates");
    println!("   2. Type-safe way to extend functionality");
    println!("   3. All supported providers work with this feature");
    println!("   4. Smooth migration when the library adds built-in support");

    Ok(())
}
