//! Custom Provider Options Example
//!
//! This example demonstrates how to extend the library with custom provider features
//! using the `CustomProviderOptions` trait. This allows you to add support for new
//! provider-specific features without waiting for library updates.
//!
//! # Use Cases
//!
//! - Adding support for new provider features before they're built into the library
//! - Experimenting with beta/preview features
//! - Supporting custom/proprietary provider extensions
//!
//! # Run
//!
//! ```bash
//! cargo run --example custom-provider-options --features xai
//! ```

use siumai::prelude::*;
use siumai::types::{CustomProviderOptions, ProviderOptions};
use std::collections::HashMap;

/// Example: Custom xAI feature that isn't yet in the library
///
/// Let's say xAI releases a new feature called "deferred" mode and
/// "parallel_function_calling" that we want to use immediately.
#[derive(Debug, Clone)]
pub struct CustomXaiFeature {
    pub deferred: Option<bool>,
    pub parallel_function_calling: Option<bool>,
}

impl CustomProviderOptions for CustomXaiFeature {
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

/// Example: Custom OpenAI feature
#[derive(Debug, Clone)]
pub struct CustomOpenAiFeature {
    pub custom_param: String,
    pub experimental_mode: bool,
}

impl CustomProviderOptions for CustomOpenAiFeature {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        Ok(serde_json::json!({
            "custom_param": self.custom_param,
            "experimental_mode": self.experimental_mode,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom Provider Options Example\n");

    // Example 1: Using CustomProviderOptions trait
    println!("📝 Example 1: Using CustomProviderOptions trait");
    println!("================================================\n");

    let custom_feature = CustomXaiFeature {
        deferred: Some(true),
        parallel_function_calling: Some(true),
    };

    // Convert to ProviderOptions using the helper method
    let options = ProviderOptions::from_custom(custom_feature)?;

    println!("✅ Created custom options: {:?}\n", options);

    // Example 2: Direct Custom variant usage
    println!("📝 Example 2: Direct Custom variant usage");
    println!("==========================================\n");

    let mut custom_options = HashMap::new();
    custom_options.insert(
        "custom_feature".to_string(),
        serde_json::Value::String("enabled".to_string()),
    );
    custom_options.insert("beta_mode".to_string(), serde_json::Value::Bool(true));

    let options = ProviderOptions::Custom {
        provider_id: "my-custom-provider".to_string(),
        options: custom_options,
    };

    println!("✅ Created direct custom options: {:?}\n", options);

    // Example 3: Using with ChatRequest
    println!("📝 Example 3: Using with ChatRequest");
    println!("=====================================\n");

    let custom_openai = CustomOpenAiFeature {
        custom_param: "test-value".to_string(),
        experimental_mode: true,
    };

    let request = ChatRequest::new(vec![
        ChatMessage::user("Hello, this is a test with custom options").build(),
    ])
    .with_provider_options(ProviderOptions::from_custom(custom_openai)?);

    println!("✅ Created ChatRequest with custom options");
    println!("   Provider: {:?}", request.provider_options.provider_id());
    println!();

    // Example 4: Combining with existing provider options
    println!("📝 Example 4: Best Practice - Use built-in options when available");
    println!("==================================================================\n");

    println!("❌ DON'T do this (using Custom for built-in features):");
    println!("   let mut opts = HashMap::new();");
    println!("   opts.insert(\"search_parameters\", ...);  // This is already built-in!");
    println!();

    println!("✅ DO this instead (use type-safe built-in options):");
    println!("   let req = ChatRequest::new(messages)");
    println!("       .with_xai_options(");
    println!("           XaiOptions::new()");
    println!("               .with_search(XaiSearchParameters::default())");
    println!("       );");
    println!();

    println!("💡 Only use Custom variant for features that aren't yet built into the library!");
    println!();

    // Example 5: Real-world scenario
    println!("📝 Example 5: Real-world scenario - New xAI feature");
    println!("====================================================\n");

    println!("Scenario: xAI just released a new 'priority_mode' feature");
    println!("that isn't in the library yet. You can use it immediately:");
    println!();

    let priority_feature = CustomXaiFeature {
        deferred: Some(false),
        parallel_function_calling: Some(true),
    };

    let _request = ChatRequest::new(vec![
        ChatMessage::user("Analyze this data with priority processing").build(),
    ])
    .with_provider_options(ProviderOptions::from_custom(priority_feature)?);

    println!("✅ Request created with custom xAI features");
    println!("   These will be automatically injected into the API request");
    println!("   by the ProviderSpec::chat_before_send() hook");
    println!();

    println!("🎉 Custom Provider Options Example Complete!");
    println!();
    println!("📚 Key Takeaways:");
    println!("   1. Implement CustomProviderOptions trait for your custom features");
    println!("   2. Use ProviderOptions::from_custom() to convert to ProviderOptions");
    println!("   3. Custom options are automatically injected by ProviderSpec");
    println!("   4. Only use Custom variant for features not yet in the library");
    println!("   5. Prefer built-in type-safe options when available");

    Ok(())
}
