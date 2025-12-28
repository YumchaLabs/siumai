//! Example demonstrating the new middleware builder system.
//!
//! This example shows how to use the MiddlewareBuilder to:
//! - Automatically add middlewares based on provider and model
//! - Manually add, remove, and replace middlewares
//! - Use preset middlewares like ExtractReasoningMiddleware

use siumai::experimental::execution::middleware::{
    MiddlewareBuilder,
    auto::{MiddlewareConfig, build_auto_middlewares},
    presets::ExtractReasoningMiddleware,
};
use std::sync::Arc;

fn main() {
    println!("=== Middleware Builder Examples ===\n");

    // Example 1: Automatic middleware configuration
    println!("1. Automatic Middleware Configuration");
    println!("--------------------------------------");
    let config = MiddlewareConfig::new("openai", "o1-preview");
    let builder = build_auto_middlewares(&config);

    println!("Auto-configured middlewares for OpenAI o1-preview:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }
    println!();

    // Example 2: Manual middleware builder
    println!("2. Manual Middleware Builder");
    println!("----------------------------");
    let mut builder = MiddlewareBuilder::new();
    builder
        .add(
            "extract-reasoning",
            Arc::new(ExtractReasoningMiddleware::default()),
        )
        .add(
            "custom-middleware",
            Arc::new(ExtractReasoningMiddleware::default()),
        );

    println!("Manually configured middlewares:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }
    println!();

    // Example 3: Removing middlewares
    println!("3. Removing Middlewares");
    println!("-----------------------");
    let mut builder = build_auto_middlewares(&config);
    println!("Before removal:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }

    builder.remove("extract-reasoning");
    println!("\nAfter removing 'extract-reasoning':");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }
    println!();

    // Example 4: Replacing middlewares
    println!("4. Replacing Middlewares");
    println!("------------------------");
    let mut builder = build_auto_middlewares(&config);
    println!("Before replacement:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }

    // Replace with a custom configuration
    builder.replace(
        "extract-reasoning",
        Arc::new(ExtractReasoningMiddleware::for_model("gemini-2.5-pro")),
    );
    println!("\nAfter replacing with Gemini-specific middleware:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }
    println!();

    // Example 5: Inserting middlewares
    println!("5. Inserting Middlewares");
    println!("------------------------");
    let mut builder = build_auto_middlewares(&config);
    builder.insert_after(
        "extract-reasoning",
        "custom-post-processing",
        Arc::new(ExtractReasoningMiddleware::default()),
    );

    println!("After inserting middleware:");
    for named in builder.middlewares() {
        println!("  - {}", named.name);
    }
    println!();

    // Example 6: Different providers
    println!("6. Different Providers");
    println!("----------------------");
    let providers = vec![
        ("openai", "o1-preview"),
        ("anthropic", "claude-3-opus"),
        ("gemini", "gemini-2.5-pro"),
        ("xai", "grok-2"),
    ];

    for (provider, model) in providers {
        let config = MiddlewareConfig::new(provider, model);
        let builder = build_auto_middlewares(&config);
        println!("{} / {}:", provider, model);
        for named in builder.middlewares() {
            println!("  - {}", named.name);
        }
    }
    println!();

    // Example 7: Disabling reasoning extraction
    println!("7. Disabling Reasoning Extraction");
    println!("----------------------------------");
    let config = MiddlewareConfig::new("openai", "gpt-4").with_enable_reasoning(false);
    let builder = build_auto_middlewares(&config);

    println!("Middlewares with reasoning disabled:");
    if builder.is_empty() {
        println!("  (no middlewares)");
    } else {
        for named in builder.middlewares() {
            println!("  - {}", named.name);
        }
    }
    println!();

    println!("=== Examples Complete ===");
}
