/// Demonstration of the new Usage builder API
///
/// This example shows how to use the UsageBuilder to create Usage instances
/// with detailed token information in a clean and readable way.
use siumai::types::{CompletionTokensDetails, PromptTokensDetails, Usage};

fn main() {
    println!("=== Usage Builder Demo ===\n");

    // Example 1: Basic usage (old way vs new way)
    println!("1. Basic Usage:");

    // Old way - verbose and error-prone
    #[allow(deprecated)]
    let usage_old = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        cached_tokens: None,
        reasoning_tokens: None,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    };
    println!("  Old way: {:?}", usage_old);

    // New way - clean and simple
    let usage_new = Usage::builder()
        .prompt_tokens(100)
        .completion_tokens(50)
        .build();
    println!("  New way: {:?}\n", usage_new);

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

    // Example 8: Typical provider usage patterns
    println!("8. Typical Provider Patterns:\n");

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
