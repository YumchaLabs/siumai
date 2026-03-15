//! Custom Middleware - Transform requests globally
//!
//! This example demonstrates creating custom middleware for the registry
//! to transform requests, override models/providers, or add default parameters.
//!
//! Construction mode: registry-first.
//! Use this file for application-level routing and request shaping where the
//! registry owns middleware composition.
//!
//! ## Features Demonstrated
//! - `transform_params()` - Modify request parameters
//! - `override_model_id()` - Dynamic model routing (A/B testing, fallback)
//! - `override_provider_id()` - Dynamic provider routing
//! - Middleware chaining (first override wins)
//!
//! ## Run
//! ```bash
//! cargo run --example custom-middleware --features openai
//! ```

use siumai::experimental::execution::middleware::language_model::LanguageModelMiddleware;
use siumai::prelude::unified::*;
use siumai::registry::{RegistryOptions, create_provider_registry};
use std::collections::HashMap;
use std::sync::Arc;

// Middleware 1: Set default temperature
#[derive(Clone)]
struct DefaultTemperatureMiddleware {
    temperature: f64,
}

impl LanguageModelMiddleware for DefaultTemperatureMiddleware {
    fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
        // Set temperature if not already set
        if req.common_params.temperature.is_none() {
            req.common_params.temperature = Some(self.temperature);
            println!(
                "🌡️  Middleware: Set default temperature to {}",
                self.temperature
            );
        }
        req
    }
}

// Middleware 2: Override model for A/B testing or fallback
#[derive(Clone)]
struct ModelOverrideMiddleware {
    from_model: String,
    to_model: String,
}

impl LanguageModelMiddleware for ModelOverrideMiddleware {
    fn override_model_id(&self, current: &str) -> Option<String> {
        if current == self.from_model {
            println!(
                "🔄 Middleware: Overriding model {} → {}",
                current, self.to_model
            );
            Some(self.to_model.clone())
        } else {
            None
        }
    }
}

// Middleware 3: Override provider for testing or routing
#[derive(Clone)]
#[allow(dead_code)]
struct ProviderOverrideMiddleware {
    from_provider: String,
    to_provider: String,
}

impl LanguageModelMiddleware for ProviderOverrideMiddleware {
    fn override_provider_id(&self, current: &str) -> Option<String> {
        if current == self.from_provider {
            println!(
                "🔄 Middleware: Overriding provider {} → {}",
                current, self.to_provider
            );
            Some(self.to_provider.clone())
        } else {
            None
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom Middleware Example\n");

    // Create middlewares
    let temp_middleware = Arc::new(DefaultTemperatureMiddleware { temperature: 0.8 });
    let override_middleware = Arc::new(ModelOverrideMiddleware {
        from_model: "gpt-4o".to_string(),
        to_model: "gpt-4o-mini".to_string(),
    });

    // Create registry with custom middlewares
    let registry = create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: vec![temp_middleware, override_middleware],
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }),
    );

    if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("Requesting gpt-4o (will be overridden to gpt-4o-mini):");
        let lm = registry.language_model("openai:gpt-4o")?;
        let response = text::generate(
            &lm,
            ChatRequest::new(vec![user!("Hello!")]),
            text::GenerateOptions::default(),
        )
        .await?;
        println!("  {}\n", response.content_text().unwrap_or_default());

        println!("✅ Middleware applied:");
        println!("  - Default temperature set to 0.8");
        println!("  - Model overridden from gpt-4o to gpt-4o-mini");
    }

    Ok(())
}
