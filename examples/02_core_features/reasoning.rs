//! 🧠 Unified Reasoning Interface - Cross-provider thinking capabilities
//!
//! This example demonstrates the unified reasoning interface that works across
//! different AI providers, showcasing both streaming and non-streaming modes:
//! - DeepSeek reasoning with step-by-step problem solving
//! - Google Gemini thinking capabilities
//! - Unified `.reasoning(true)` and `.reasoning_budget()` interface
//! - Streaming vs non-streaming reasoning output handling
//!
//! Before running, set your API keys:
//! ```bash
//! export DEEPSEEK_API_KEY="your-deepseek-key"
//! export GEMINI_API_KEY="your-google-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example reasoning
//! ```

use futures::StreamExt;
use siumai::models;
use siumai::prelude::*;
use siumai::stream::ChatStreamEvent;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Unified Reasoning Interface - Cross-provider thinking capabilities\n");
    println!("{}", "=".repeat(80));

    // Demo 1: DeepSeek reasoning (non-streaming)
    println!("\n📋 Demo 1: DeepSeek Reasoning (Non-streaming)");
    demo_deepseek_non_streaming().await?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 2: DeepSeek reasoning (streaming)
    println!("📋 Demo 2: DeepSeek Reasoning (Streaming)");
    demo_deepseek_streaming().await?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 2.5: Enhanced DeepSeek streaming with better event handling
    println!("📋 Demo 2.5: Enhanced DeepSeek Streaming (Complete Event Handling)");
    demo_enhanced_deepseek_streaming().await?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 3: Gemini thinking (non-streaming)
    println!("📋 Demo 3: Gemini Thinking (Non-streaming)");
    demo_gemini_non_streaming().await?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 4: Gemini thinking (streaming)
    println!("📋 Demo 4: Gemini Thinking (Streaming)");
    demo_gemini_streaming().await?;

    println!("\n{}\n", "=".repeat(80));

    println!("\n✅ Unified reasoning interface examples completed!");
    Ok(())
}

/// Demo DeepSeek reasoning in non-streaming mode
async fn demo_deepseek_non_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🤖 DeepSeek Non-streaming Reasoning");

    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| {
        println!("   ⚠️  DEEPSEEK_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    // Create DeepSeek client with unified reasoning interface
    let client = Siumai::builder()
        .deepseek()
        .api_key(&api_key)
        .model("deepseek-reasoner")
        .reasoning(true) // ✅ Unified reasoning interface
        .reasoning_budget(8192) // ✅ Works across all providers
        .temperature(0.7)
        .max_tokens(2000)
        .build()
        .await?;

    let messages = vec![
        system!(
            "You are a mathematical problem solver. When given a problem, \
            think through it step by step and show your reasoning process."
        ),
        user!(
            "A train travels from City A to City B at 80 km/h and returns \
            at 120 km/h. If the total trip time is 5 hours, what is the \
            distance between the two cities? Show your reasoning step by step."
        ),
    ];

    println!("   📝 Problem: Train speed calculation with reasoning");
    println!("   🔄 Processing with DeepSeek reasoning...");

    match client.chat(messages).await {
        Ok(response) => {
            // Access the reasoning content
            if let Some(thinking) = &response.thinking {
                println!("\n   🧠 DeepSeek's Reasoning Process:");
                println!("   {}", "─".repeat(60));
                let thinking_lines: Vec<&str> = thinking.lines().take(10).collect();
                for (i, line) in thinking_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        println!("   {}: {}", i + 1, line.trim());
                    }
                }
                if thinking.lines().count() > 10 {
                    println!("   ... (showing first 10 lines of reasoning)");
                }
                println!(
                    "   📊 Total reasoning length: {} characters",
                    thinking.len()
                );
            } else {
                println!("\n   ℹ️  No reasoning content available");
                println!(
                    "   💡 Note: DeepSeek reasoning requires valid API key and reasoning model"
                );
            }

            // Access the final answer
            if let Some(answer) = response.content_text() {
                println!("\n   💬 Final Answer:");
                println!("   {}", "─".repeat(60));
                let answer_preview = if answer.len() > 300 {
                    format!("{}...", &answer[..300])
                } else {
                    answer.to_string()
                };
                println!("   {}", answer_preview);
            }

            println!("   ✅ DeepSeek non-streaming reasoning completed");
        }
        Err(e) => {
            println!("   ❌ DeepSeek reasoning failed: {}", e);
        }
    }

    Ok(())
}

/// Demo DeepSeek reasoning in streaming mode
async fn demo_deepseek_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🤖 DeepSeek Streaming Reasoning");
    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| {
        println!("   ⚠️  DEEPSEEK_API_KEY not set, using demo key");
        "demo-key".to_string()
    });
    // Create DeepSeek client with unified reasoning interface
    let client = Siumai::builder()
        .deepseek()
        .api_key(&api_key)
        .model("deepseek-reasoner")
        .reasoning(true) // ✅ Unified reasoning interface
        .reasoning_budget(4096) // ✅ Configurable reasoning budget
        .temperature(0.5)
        .max_tokens(1500)
        .build()
        .await?;

    let messages = vec![user!(
        "Solve this logic puzzle step by step: \
            \
            Three friends Alice, Bob, and Carol each have a different pet \
            (cat, dog, bird) and live in different colored houses (red, blue, green). \
            \
            Clues: \
            1. Alice doesn't live in the red house \
            2. The person with the cat lives in the blue house \
            3. Bob doesn't have the bird \
            4. Carol doesn't live in the green house \
            5. The person in the red house has the dog \
            \
            Who has which pet and lives in which house?"
    )];

    println!("   📝 Problem: Logic puzzle with streaming reasoning");
    println!("   🔄 Streaming DeepSeek reasoning...\n");

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut thinking_content = String::new();
            let mut response_content = String::new();
            let mut in_thinking_phase = false;
            let mut in_content_phase = false;
            let thinking_lines = 0;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                        if !in_thinking_phase {
                            println!("   🧠 Reasoning Process (streaming):");
                            println!("   {}", "─".repeat(60));
                            print!("   ");
                            in_thinking_phase = true;
                            in_content_phase = false;
                        }
                        thinking_content.push_str(&delta);

                        // Display reasoning content naturally (like normal text)
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        if !in_content_phase {
                            if in_thinking_phase {
                                println!("\n");
                            }
                            println!("   💬 Final Answer (streaming):");
                            println!("   {}", "─".repeat(60));
                            print!("   ");
                            in_content_phase = true;
                            in_thinking_phase = false;
                        }
                        response_content.push_str(&delta);
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::StreamStart { .. }) => {
                        println!("   🚀 Stream started...");
                    }
                    Ok(ChatStreamEvent::StreamEnd { .. }) => {
                        if in_content_phase {
                            println!();
                        }
                        println!("   ✅ DeepSeek streaming reasoning completed");
                        break;
                    }
                    Ok(ChatStreamEvent::Error { error }) => {
                        println!("\n   ❌ Stream error: {}", error);
                        break;
                    }
                    Err(e) => {
                        println!("\n   ❌ Stream processing error: {}", e);
                        break;
                    }
                    _ => {
                        // Handle other event types if needed
                    }
                }
            }

            // Summary statistics
            if !thinking_content.is_empty() {
                println!("   📊 Reasoning statistics:");
                println!(
                    "      - Reasoning length: {} characters",
                    thinking_content.len()
                );
                println!("      - Reasoning lines: {}", thinking_lines);
            }
            if !response_content.is_empty() {
                println!(
                    "      - Answer length: {} characters",
                    response_content.len()
                );
            }
        }
        Err(e) => {
            println!("   ❌ DeepSeek streaming failed: {}", e);
        }
    }

    Ok(())
}

/// Demo Gemini thinking in non-streaming mode
async fn demo_gemini_non_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🤖 Gemini Non-streaming Thinking");

    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| {
        println!("   ⚠️  GEMINI_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    // Create Gemini client with unified reasoning interface
    let client = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model(models::gemini::GEMINI_2_5_PRO)
        .reasoning(true) // ✅ Unified reasoning interface
        .reasoning_budget(2048) // ✅ Works across all providers
        .temperature(0.8)
        .max_tokens(1500)
        .build()
        .await?;

    let messages = vec![
        system!(
            "You are a creative problem solver. When given a challenge, \
            think through multiple approaches and explain your thought process."
        ),
        user!(
            "Design a sustainable urban transportation system for a city of \
            1 million people. Consider environmental impact, cost-effectiveness, \
            accessibility, and future scalability. Think through the key \
            components and trade-offs."
        ),
    ];

    println!("   📝 Problem: Urban transportation system design");
    println!("   🔄 Processing with Gemini thinking...");

    match client.chat(messages).await {
        Ok(response) => {
            // Access the thinking content
            if let Some(thinking) = &response.thinking {
                println!("\n   🧠 Gemini's Thinking Process:");
                println!("   {}", "─".repeat(60));
                let thinking_lines: Vec<&str> = thinking.lines().take(8).collect();
                for (i, line) in thinking_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        println!("   {}: {}", i + 1, line.trim());
                    }
                }
                if thinking.lines().count() > 8 {
                    println!("   ... (showing first 8 lines of thinking)");
                }
                println!("   📊 Total thinking length: {} characters", thinking.len());
            }

            // Access the final response
            if let Some(answer) = response.content_text() {
                println!("\n   💬 Final Design:");
                println!("   {}", "─".repeat(60));
                let answer_preview = if answer.len() > 400 {
                    format!("{}...", &answer[..400])
                } else {
                    answer.to_string()
                };
                println!("   {}", answer_preview);
            }

            println!("   ✅ Gemini non-streaming thinking completed");
        }
        Err(e) => {
            println!("   ❌ Gemini thinking failed: {}", e);
        }
    }

    Ok(())
}

/// Demo Gemini thinking in streaming mode
async fn demo_gemini_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🤖 Gemini Streaming Thinking");

    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| {
        println!("   ⚠️  GEMINI_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    // Create Gemini client with unified reasoning interface
    let client = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model(models::gemini::GEMINI_2_5_PRO)
        .reasoning(true) // ✅ Unified reasoning interface
        .reasoning_budget(1024) // ✅ Configurable thinking budget
        .temperature(0.6)
        .max_tokens(1200)
        .build()
        .await?;

    let messages = vec![user!(
        "Analyze this ethical dilemma step by step: \
            \
            A self-driving car's AI must choose between two unavoidable accidents: \
            1) Hit a group of 3 elderly people crossing legally \
            2) Swerve and hit 1 child who ran into the street illegally \
            \
            Consider the ethical frameworks, legal implications, and \
            societal values that should guide this decision."
    )];

    println!("   📝 Problem: Ethical dilemma analysis with streaming thinking");
    println!("   🔄 Streaming Gemini thinking...\n");

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut thinking_content = String::new();
            let mut response_content = String::new();
            let mut in_thinking_phase = false;
            let mut in_content_phase = false;
            let mut thinking_sections = 0;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                        if !in_thinking_phase {
                            println!("   🧠 Thinking Process (streaming):");
                            println!("   {}", "─".repeat(60));
                            print!("   ");
                            in_thinking_phase = true;
                            in_content_phase = false;
                        }
                        thinking_content.push_str(&delta);

                        // Display thinking content naturally (like normal text)
                        print!("{}", delta);
                        io::stdout().flush().unwrap();

                        // Count sections for statistics
                        thinking_sections += delta.matches("Framework").count()
                            + delta.matches("Consideration").count()
                            + delta.matches("Approach").count()
                            + delta.matches("Analysis").count();
                    }
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        if !in_content_phase {
                            if in_thinking_phase {
                                println!("\n");
                            }
                            println!("   💬 Ethical Analysis (streaming):");
                            println!("   {}", "─".repeat(60));
                            print!("   ");
                            in_content_phase = true;
                            in_thinking_phase = false;
                        }
                        response_content.push_str(&delta);
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::StreamStart { .. }) => {
                        println!("   🚀 Stream started...");
                    }
                    Ok(ChatStreamEvent::StreamEnd { .. }) => {
                        if in_content_phase {
                            println!();
                        }
                        println!("   ✅ Gemini streaming thinking completed");
                        break;
                    }
                    Ok(ChatStreamEvent::Error { error }) => {
                        println!("\n   ❌ Stream error: {}", error);
                        break;
                    }
                    Err(e) => {
                        println!("\n   ❌ Stream processing error: {}", e);
                        break;
                    }
                    _ => {
                        // Handle other event types if needed
                    }
                }
            }

            // Summary statistics
            if !thinking_content.is_empty() {
                println!("   📊 Thinking statistics:");
                println!(
                    "      - Thinking length: {} characters",
                    thinking_content.len()
                );
                println!("      - Thinking sections: {}", thinking_sections);
            }
            if !response_content.is_empty() {
                println!(
                    "      - Analysis length: {} characters",
                    response_content.len()
                );
            }
        }
        Err(e) => {
            println!("   ❌ Gemini streaming failed: {}", e);
        }
    }

    Ok(())
}

/// Demo cross-provider reasoning comparison (unused)
#[allow(dead_code)]
async fn demo_cross_provider_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🔄 Cross-provider Reasoning Comparison");

    let deepseek_key = std::env::var("DEEPSEEK_API_KEY").ok();
    let google_key = std::env::var("GEMINI_API_KEY").ok();

    if deepseek_key.is_none() && google_key.is_none() {
        println!("   ⚠️  No API keys available for comparison");
        return Ok(());
    }

    let test_problem = "Explain the concept of quantum entanglement in simple terms, \
                       thinking through the key principles and analogies that would \
                       help a non-physicist understand this phenomenon.";

    println!("   📝 Test Problem: Quantum entanglement explanation");
    println!("   🔄 Comparing reasoning approaches across providers...\n");

    // Test DeepSeek if available
    if let Some(api_key) = deepseek_key {
        println!("   🤖 DeepSeek Reasoning Approach:");
        println!("   {}", "─".repeat(50));

        let client = Siumai::builder()
            .deepseek()
            .api_key(&api_key)
            .model("deepseek-reasoner")
            .reasoning(true) // ✅ Unified interface
            .reasoning_budget(3072) // ✅ Consistent budget
            .temperature(0.4)
            .max_tokens(800)
            .build()
            .await?;

        let messages = vec![user!(test_problem)];

        match client.chat(messages).await {
            Ok(response) => {
                if let Some(thinking) = &response.thinking {
                    println!("   🧠 Reasoning style: Step-by-step logical breakdown");
                    println!("   📊 Reasoning length: {} chars", thinking.len());

                    // Analyze reasoning characteristics
                    let step_count =
                        thinking.matches("step").count() + thinking.matches("Step").count();
                    let analogy_count =
                        thinking.matches("like").count() + thinking.matches("analogy").count();

                    println!(
                        "   🔍 Analysis: {} steps, {} analogies",
                        step_count, analogy_count
                    );
                }

                if let Some(answer) = response.content_text() {
                    let preview = if answer.len() > 200 {
                        format!("{}...", &answer[..200])
                    } else {
                        answer.to_string()
                    };
                    println!("   💬 Answer preview: {}", preview);
                }
                println!("   ✅ DeepSeek comparison completed");
            }
            Err(e) => {
                println!("   ❌ DeepSeek failed: {}", e);
            }
        }
        println!();
    }

    // Test Gemini if available
    if let Some(api_key) = google_key {
        println!("   🤖 Gemini Thinking Approach:");
        println!("   {}", "─".repeat(50));

        let client = Siumai::builder()
            .gemini()
            .api_key(&api_key)
            .model(models::gemini::GEMINI_2_5_PRO)
            .reasoning(true) // ✅ Unified interface
            .reasoning_budget(3072) // ✅ Consistent budget
            .temperature(0.4)
            .max_tokens(800)
            .build()
            .await?;

        let messages = vec![user!(test_problem)];

        match client.chat(messages).await {
            Ok(response) => {
                if let Some(thinking) = &response.thinking {
                    println!("   🧠 Thinking style: Multi-perspective analysis");
                    println!("   📊 Thinking length: {} chars", thinking.len());

                    // Analyze thinking characteristics
                    let concept_count =
                        thinking.matches("concept").count() + thinking.matches("idea").count();
                    let example_count =
                        thinking.matches("example").count() + thinking.matches("instance").count();

                    println!(
                        "   🔍 Analysis: {} concepts, {} examples",
                        concept_count, example_count
                    );
                }

                if let Some(answer) = response.content_text() {
                    let preview = if answer.len() > 200 {
                        format!("{}...", &answer[..200])
                    } else {
                        answer.to_string()
                    };
                    println!("   💬 Answer preview: {}", preview);
                }
                println!("   ✅ Gemini comparison completed");
            }
            Err(e) => {
                println!("   ❌ Gemini failed: {}", e);
            }
        }
    }

    println!("\n   🎯 Key Insights:");
    println!("   • Both providers use the same unified .reasoning(true) interface");
    println!("   • Reasoning budgets are consistently configurable across providers");
    println!("   • Each provider has distinct reasoning/thinking styles");
    println!("   • The unified interface enables easy provider comparison");

    Ok(())
}

/// Enhanced DeepSeek streaming with complete event handling
async fn demo_enhanced_deepseek_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("   🤖 Enhanced DeepSeek Streaming with Complete Event Handling");

    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| {
        println!("   ⚠️  DEEPSEEK_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    // Use enhanced DeepSeek configuration
    let client = Siumai::builder()
        .deepseek() // Use standard DeepSeek adapter
        .api_key(&api_key)
        .model("deepseek-reasoner")
        .temperature(0.6)
        .max_tokens(4096)
        .build()
        .await?;

    let messages = vec![user!(
        "I need to plan a birthday party for 50 people with a budget of $1000. \
               Think through all the key considerations: venue, food, entertainment, \
               decorations, and logistics. Provide a detailed breakdown with reasoning \
               for each decision."
    )];

    println!("   📝 Problem: Complex party planning with budget constraints");
    println!("   🔄 Enhanced streaming with complete event handling...\n");

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut thinking_content = String::new();
            let mut response_content = String::new();
            let mut thinking_phase_active = false;
            let mut content_phase_active = false;
            let mut thinking_steps = 0;
            let mut content_words = 0;
            let mut stream_started = false;

            println!("   🎬 Stream Events Log:");
            println!("   {}", "═".repeat(60));

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::StreamStart { .. }) => {
                        stream_started = true;
                        println!("   🚀 [EVENT] Stream started");
                        println!("   ⏱️  Waiting for DeepSeek to begin reasoning...\n");
                    }

                    Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                        if !thinking_phase_active {
                            println!("   🧠 [PHASE] Reasoning Phase Started");
                            println!("   {}", "─".repeat(50));
                            thinking_phase_active = true;
                            content_phase_active = false;
                        }

                        thinking_content.push_str(&delta);

                        // Display reasoning content naturally (like normal text)
                        print!("{}", delta);
                        io::stdout().flush().unwrap();

                        // Count steps for statistics
                        thinking_steps += delta.matches("Step").count()
                            + delta.matches("1.").count()
                            + delta.matches("Consider").count()
                            + delta.matches("Analysis").count();
                    }

                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        if !content_phase_active {
                            if thinking_phase_active {
                                println!("\n   ✅ [PHASE] Reasoning Complete");
                                println!("   📝 [PHASE] Final Answer Started");
                                println!("   {}", "─".repeat(50));
                            }
                            content_phase_active = true;
                            thinking_phase_active = false;
                            print!("   💬 ");
                        }

                        response_content.push_str(&delta);
                        content_words += delta.split_whitespace().count();
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }

                    Ok(ChatStreamEvent::StreamEnd { .. }) => {
                        if content_phase_active {
                            println!();
                        }
                        println!("\n   🏁 [EVENT] Stream ended successfully");
                        break;
                    }

                    Ok(ChatStreamEvent::Error { error }) => {
                        println!("\n   ❌ [ERROR] Stream error: {}", error);
                        break;
                    }

                    Ok(ChatStreamEvent::ToolCallDelta { .. }) => {
                        println!("   🔧 [EVENT] Tool call delta received");
                    }

                    Err(e) => {
                        println!("\n   ❌ [ERROR] Stream processing error: {}", e);
                        break;
                    }

                    _ => {
                        // Handle any other event types
                        println!("   ℹ️  [EVENT] Other event received");
                    }
                }
            }

            // Comprehensive summary
            println!("\n   📊 Enhanced Streaming Summary:");
            println!("   {}", "═".repeat(50));
            println!(
                "   🚀 Stream started: {}",
                if stream_started { "✅" } else { "❌" }
            );
            println!(
                "   🧠 Reasoning phase: {}",
                if !thinking_content.is_empty() {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "   📝 Content phase: {}",
                if !response_content.is_empty() {
                    "✅"
                } else {
                    "❌"
                }
            );

            if !thinking_content.is_empty() {
                println!(
                    "   📏 Reasoning length: {} characters",
                    thinking_content.len()
                );
                println!("   🔢 Reasoning steps detected: {}", thinking_steps);
                println!(
                    "   📄 Reasoning lines: {}",
                    thinking_content.lines().count()
                );
            }

            if !response_content.is_empty() {
                println!("   📏 Answer length: {} characters", response_content.len());
                println!("   🔤 Answer words: {}", content_words);
                println!("   📄 Answer lines: {}", response_content.lines().count());
            }

            println!("   ⚡ Total processing: Real-time streaming with complete event handling");
            println!("   ✨ Enhanced features: Step detection, phase tracking, error handling");
        }
        Err(e) => {
            println!("   ❌ Enhanced DeepSeek streaming failed: {}", e);
            println!("   💡 Tip: Make sure DEEPSEEK_API_KEY is set and valid");
        }
    }

    Ok(())
}

/*
🧠 Unified Reasoning Interface - Key Features:

Unified API:
- `.reasoning(true)` - Enable reasoning/thinking mode across all providers
- `.reasoning_budget(n)` - Set thinking token budget consistently
- Same interface works with DeepSeek, Gemini, Anthropic, etc.

Provider-Specific Behavior:
- DeepSeek: Step-by-step logical reasoning with detailed breakdown
- Gemini: Multi-perspective thinking with creative analysis
- Anthropic: Structured thinking with clear reasoning chains

Streaming Support:
- `ChatStreamEvent::StreamStart` - Stream initialization event
- `ChatStreamEvent::ThinkingDelta { delta }` - Real-time thinking/reasoning content
- `ChatStreamEvent::ContentDelta { delta, .. }` - Final response content
- `ChatStreamEvent::StreamEnd` - Stream completion event
- `ChatStreamEvent::Error { error }` - Stream error handling
- `ChatStreamEvent::ToolCallDelta` - Tool usage events
- `ChatStreamEvent::UsageUpdate` - Token usage updates
- Proper phase detection and transition handling
- Real-time step detection and formatting

Response Access:
- `response.thinking` - Access the complete thinking/reasoning content
- `response.content_text()` - Access the final answer/response
- Both available in streaming and non-streaming modes

Best Practices:
1. Set appropriate reasoning budgets (1024-8192 tokens)
2. Use lower temperatures (0.4-0.7) for logical reasoning
3. Handle all stream events properly (Start, ThinkingDelta, ContentDelta, End, Error)
4. Implement phase detection for better UX (thinking → content transition)
5. Use real-time display with proper formatting and step detection
6. Provide clear, specific problems for better reasoning
7. Include comprehensive error handling for robust streaming
8. Compare providers to find best fit for your use case

Use Cases:
- Mathematical problem solving with step-by-step work
- Ethical dilemma analysis with multiple perspectives
- Complex planning with trade-off considerations
- Logic puzzles with systematic elimination
- Creative problem solving with ideation process

Temperature Guidelines:
- 0.0-0.4: Logical, mathematical, analytical reasoning
- 0.5-0.7: Balanced reasoning with some creativity
- 0.8-1.0: Creative thinking and brainstorming

Next Steps:
- Explore provider-specific reasoning optimizations
- Implement reasoning quality metrics
- Build reasoning templates for common problem types
- Create reasoning workflow automation
*/
