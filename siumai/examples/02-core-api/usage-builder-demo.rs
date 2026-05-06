//! Usage builder and accessors.
//!
//! `Usage` stores AI SDK-style input/output token summaries as the canonical
//! shape. The legacy prompt/completion/total values are exposed through
//! accessors for compatibility, but new code should construct usage through
//! `Usage::new(...)` or `Usage::builder()`.

use serde_json::json;
use siumai::prelude::unified::{
    CompletionTokensDetails, PromptTokensDetails, Usage, UsageInputTokens, UsageOutputTokens,
};

fn main() {
    println!("=== Usage Builder Demo ===\n");

    // Example 1: Basic usage
    println!("1. Basic Usage:");

    let usage_basic = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .build();
    println!(
        "  prompt={:?}, completion={:?}, total={:?}",
        usage_basic.prompt_tokens(),
        usage_basic.completion_tokens(),
        usage_basic.total_tokens()
    );
    println!("  inputTokens={:?}", usage_basic.normalized_input_tokens());
    println!(
        "  outputTokens={:?}\n",
        usage_basic.normalized_output_tokens()
    );

    // Example 2: Usage with cached tokens
    println!("2. Usage with Cached Tokens:");

    let usage_cached = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .with_cached_tokens(20)
        .build();
    println!("  {:?}\n", usage_cached);

    // Example 3: Usage with reasoning tokens (for o1 models)
    println!("3. Usage with Reasoning Tokens:");

    let usage_reasoning = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .with_reasoning_tokens(30)
        .build();
    println!("  {:?}\n", usage_reasoning);

    // Example 4: Usage with audio tokens
    println!("4. Usage with Audio Tokens:");

    let usage_audio = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .with_prompt_audio_tokens(10)
        .with_completion_audio_tokens(15)
        .build();
    println!("  {:?}\n", usage_audio);

    // Example 5: Usage with prediction tokens (Predicted Outputs)
    println!("5. Usage with Prediction Tokens:");

    let usage_prediction = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .with_accepted_prediction_tokens(40)
        .with_rejected_prediction_tokens(5)
        .build();
    println!("  {:?}\n", usage_prediction);

    // Example 6: Complex usage with multiple detail types
    println!("6. Complex Usage (multiple details):");

    let usage_complex = Usage::builder()
        .prompt_tokens(200)
        .completion_tokens(100)
        .total_tokens(300)
        .with_cached_tokens(50)
        .with_prompt_audio_tokens(10)
        .with_reasoning_tokens(30)
        .with_completion_audio_tokens(20)
        .with_accepted_prediction_tokens(40)
        .with_rejected_prediction_tokens(10)
        .build();
    println!("  {:?}\n", usage_complex);

    // Example 7: Using convenience constructors for details
    println!("7. Using Detail Convenience Constructors:");

    let prompt_details = PromptTokensDetails::with_cached(50);
    let completion_details = CompletionTokensDetails::with_reasoning(30);

    let usage_with_details = Usage::builder()
        .prompt_tokens(200)
        .completion_tokens(100)
        .with_prompt_details(prompt_details)
        .with_completion_details(completion_details)
        .build();
    println!("  {:?}\n", usage_with_details);

    // Example 8: AI SDK-style input/output token summaries
    println!("8. AI SDK-style Input/Output Token Summaries:");

    let usage_ai_sdk = Usage::builder()
        .with_input_tokens(UsageInputTokens {
            total: Some(120),
            no_cache: Some(80),
            cache_read: Some(40),
            cache_write: None,
        })
        .with_output_tokens(UsageOutputTokens {
            total: Some(45),
            text: Some(30),
            reasoning: Some(15),
        })
        .with_raw_usage_value(json!({
            "input_tokens": 120,
            "output_tokens": 45,
            "provider_note": "provider-native payloads stay under raw"
        }))
        .build();
    println!("  inputTokens={:?}", usage_ai_sdk.normalized_input_tokens());
    println!(
        "  outputTokens={:?}",
        usage_ai_sdk.normalized_output_tokens()
    );
    println!("  raw={:?}\n", usage_ai_sdk.raw_usage_value());

    // Example 9: Typical provider usage patterns
    println!("9. Typical Provider Patterns:\n");

    // Anthropic pattern (with cached tokens)
    println!("  Anthropic (with cache):");
    let anthropic_usage = Usage::builder()
        .prompt_tokens(150)
        .completion_tokens(75)
        .with_cached_tokens(30)
        .build();
    println!("    {:?}\n", anthropic_usage);

    // Gemini pattern (with cached and reasoning)
    println!("  Gemini (with cache and reasoning):");
    let gemini_usage = Usage::builder()
        .prompt_tokens(180)
        .completion_tokens(90)
        .with_cached_tokens(40)
        .with_reasoning_tokens(25)
        .build();
    println!("    {:?}\n", gemini_usage);

    // OpenAI o1 pattern (with reasoning)
    println!("  OpenAI o1 (with reasoning):");
    let openai_o1_usage = Usage::builder()
        .prompt_tokens(120)
        .completion_tokens(60)
        .with_reasoning_tokens(45)
        .build();
    println!("    {:?}\n", openai_o1_usage);

    // OpenAI audio pattern
    println!("  OpenAI (with audio):");
    let openai_audio_usage = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .with_prompt_audio_tokens(20)
        .with_completion_audio_tokens(15)
        .build();
    println!("    {:?}\n", openai_audio_usage);

    // OpenAI Predicted Outputs pattern
    println!("  OpenAI (with predictions):");
    let openai_prediction_usage = Usage::builder()
        .prompt_tokens(150)
        .completion_tokens(75)
        .with_accepted_prediction_tokens(60)
        .with_rejected_prediction_tokens(15)
        .build();
    println!("    {:?}\n", openai_prediction_usage);

    println!("=== Demo Complete ===");
}
