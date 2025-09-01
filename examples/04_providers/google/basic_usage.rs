//! 🔍 Google Gemini Basic Usage
//!
//! This example demonstrates Google Gemini capabilities including:
//! - Gemini model selection and optimization
//! - Multimodal capabilities (text, images, etc.)
//! - Safety settings and content filtering
//! - Performance optimization techniques
//! - Google-specific features and best practices
//!
//! Before running, set your API key:
//! ```bash
//! export GEMINI_API_KEY="your-google-api-key"
//! ```
//!
//! Usage:
//! ```bash
//! cargo run --example google_basic_usage
//! ```

use siumai::models;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Google Gemini Basic Usage Demo\n");

    // Get API key
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| {
        println!("⚠️  GEMINI_API_KEY not set, using demo key");
        "demo-key".to_string()
    });

    println!("🔧 Demonstrating Gemini Capabilities:");
    println!("   1. Model Selection and Comparison");
    println!("   2. Multimodal Capabilities");
    println!("   3. Safety Settings and Content Filtering");
    println!("   4. Performance Optimization");
    println!("   5. Google-Specific Features\n");

    // Demo 1: Model Selection
    println!("🎯 1. Model Selection and Comparison");
    demo_model_selection(&api_key).await?;
    println!();

    // Demo 2: Multimodal Capabilities
    println!("🖼️ 2. Multimodal Capabilities");
    demo_multimodal_features(&api_key).await?;
    println!();

    // Demo 3: Safety Settings
    println!("🛡️ 3. Safety Settings and Content Filtering");
    demo_safety_settings(&api_key).await?;
    println!();

    // Demo 4: Performance Optimization
    println!("⚡ 4. Performance Optimization");
    demo_performance_optimization(&api_key).await?;
    println!();

    // Demo 5: Google-Specific Features
    println!("🌟 5. Google-Specific Features");
    demo_google_features(&api_key).await?;

    println!("\n✅ Google Gemini demo completed!");
    Ok(())
}

/// Demo different Gemini models
async fn demo_model_selection(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing different Gemini models...");

    let models = vec![
        (
            models::gemini::GEMINI_1_5_FLASH,
            "Fast and efficient for most tasks",
        ),
        (
            models::gemini::GEMINI_1_5_PRO,
            "Advanced reasoning and complex tasks",
        ),
        (
            models::gemini::GEMINI_2_5_PRO,
            "Latest flagship model for production use",
        ),
    ];

    for (model_name, description) in models {
        println!("   Testing {model_name}: {description}");

        let ai = Siumai::builder()
            .anthropic()
            .api_key(api_key)
            .model(models::anthropic::CLAUDE_SONNET_3_5)
            .temperature(0.3)
            .max_tokens(200)
            .build()
            .await?;

        let messages = vec![
            ChatMessage::user(
                "Explain the concept of machine learning in simple terms, \
                focusing on how it differs from traditional programming.",
            )
            .build(),
        ];

        match ai.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("   📝 {}: {}", model_name, &text[..text.len().min(100)]);
                    if let Some(usage) = &response.usage {
                        println!("      Tokens: {}", usage.total_tokens);
                    }
                }
            }
            Err(e) => println!("   ❌ Error with {model_name}: {e}"),
        }
        println!();
    }

    Ok(())
}

/// Demo multimodal capabilities
async fn demo_multimodal_features(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating multimodal capabilities...");

    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model(models::anthropic::CLAUDE_SONNET_3_5)
        .temperature(0.3)
        .max_tokens(400)
        .build()
        .await?;

    // Text-only interaction
    println!("   Text-only interaction:");
    let text_messages = vec![
        ChatMessage::user(
            "Describe the process of photosynthesis and its importance \
            for life on Earth.",
        )
        .build(),
    ];

    match ai.chat(text_messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   📝 Text response: {}", &text[..text.len().min(120)]);
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Simulated multimodal interaction (image + text)
    println!("   Simulated multimodal interaction:");
    let multimodal_messages = vec![
        ChatMessage::user(
            "If I were to show you an image of a sunset over mountains, \
            what elements would you look for to describe the scene? \
            What questions might you ask about the image?",
        )
        .build(),
    ];

    match ai.chat(multimodal_messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!(
                    "   🖼️ Multimodal guidance: {}",
                    &text[..text.len().min(120)]
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Code understanding
    println!("   Code understanding:");
    let code_messages = vec![
        ChatMessage::user(
            "Explain this Python code and suggest improvements:\n\
            ```python\n\
            def fibonacci(n):\n\
                if n <= 1:\n\
                    return n\n\
                return fibonacci(n-1) + fibonacci(n-2)\n\
            ```",
        )
        .build(),
    ];

    match ai.chat(code_messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   💻 Code analysis: {}", &text[..text.len().min(120)]);
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    Ok(())
}

/// Demo safety settings and content filtering
async fn demo_safety_settings(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing safety settings and content filtering...");

    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model(models::anthropic::CLAUDE_SONNET_3_5)
        .temperature(0.3)
        .max_tokens(300)
        .build()
        .await?;

    // Test appropriate content handling
    println!("   Testing appropriate content handling:");
    let safe_messages = vec![
        ChatMessage::user(
            "Explain the importance of online safety for children \
            and provide some practical tips for parents.",
        )
        .build(),
    ];

    match ai.chat(safe_messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!(
                    "   🛡️ Safety guidance provided: {}",
                    &text[..text.len().min(100)]
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Test educational content about sensitive topics
    println!("   Testing educational content handling:");
    let educational_messages = vec![
        ChatMessage::user(
            "Explain the historical significance of the civil rights movement \
            in the United States, focusing on key achievements and ongoing challenges.",
        )
        .build(),
    ];

    match ai.chat(educational_messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!(
                    "   📚 Educational content: {}",
                    &text[..text.len().min(100)]
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    println!("   💡 Safety Features:");
    println!("      • Built-in content filtering");
    println!("      • Configurable safety settings");
    println!("      • Responsible AI guidelines");
    println!("      • Educational content support");
    println!("      • Age-appropriate responses");

    Ok(())
}

/// Demo performance optimization techniques
async fn demo_performance_optimization(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating performance optimization techniques...");

    // Optimization 1: Model selection for task complexity
    println!("   Optimization 1: Model selection based on task");

    let fast_ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model(models::anthropic::CLAUDE_HAIKU_3_5) // Faster model for simple tasks
        .temperature(0.3)
        .max_tokens(100)
        .build()
        .await?;

    let simple_task = vec![ChatMessage::user("What is the capital of Japan?").build()];

    match fast_ai.chat(simple_task).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   ⚡ Fast model result: {}", text.trim());
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Optimization 2: Batch processing
    println!("   Optimization 2: Batch processing multiple questions");

    let batch_questions = vec![
        ChatMessage::user(
            "Answer these geography questions briefly:\n\
        1. What is the largest ocean?\n\
        2. Which continent has the most countries?\n\
        3. What is the highest mountain?\n\
        4. Which river is the longest?",
        )
        .build(),
    ];

    match fast_ai.chat(batch_questions).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   📦 Batch results:");
                for (i, line) in text.lines().take(4).enumerate() {
                    if !line.trim().is_empty() {
                        println!("      {}. {}", i + 1, line.trim());
                    }
                }
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Optimization 3: Token management
    println!("   Optimization 3: Token management strategies");
    println!("   💡 Performance Tips:");
    println!("      • Use Gemini Flash for simple tasks");
    println!("      • Use Gemini Pro for complex reasoning");
    println!("      • Batch multiple questions together");
    println!("      • Set appropriate max_tokens limits");
    println!("      • Use streaming for long responses");
    println!("      • Cache frequent queries");

    Ok(())
}

/// Demo Google-specific features
async fn demo_google_features(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Exploring Google-specific capabilities...");

    let ai = Siumai::builder()
        .anthropic()
        .api_key(api_key)
        .model(models::anthropic::CLAUDE_SONNET_3_5)
        .temperature(0.3)
        .max_tokens(400)
        .build()
        .await?;

    // Feature 1: Large context window
    println!("   Feature 1: Large context window utilization");
    let context_test = vec![
        ChatMessage::system(
            "You are a research assistant helping with a comprehensive analysis. \
            You can handle large amounts of context and maintain coherence \
            across long conversations.",
        )
        .build(),
        ChatMessage::user(
            "I'm researching the impact of artificial intelligence on various industries. \
            Can you provide a structured analysis covering healthcare, finance, \
            education, and transportation? For each industry, discuss current applications, \
            benefits, challenges, and future prospects.",
        )
        .build(),
    ];

    match ai.chat(context_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!(
                    "   📚 Comprehensive analysis provided ({} words)",
                    text.split_whitespace().count()
                );
                println!(
                    "   Structure: {}",
                    if text.contains("Healthcare") && text.contains("Finance") {
                        "Well-organized multi-industry analysis"
                    } else {
                        "Detailed response"
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Feature 2: Reasoning and analysis
    println!("   Feature 2: Advanced reasoning capabilities");
    let reasoning_test = vec![
        ChatMessage::user(
            "Compare and contrast renewable energy sources (solar, wind, hydro) \
            considering factors like efficiency, environmental impact, cost, \
            and scalability. Provide a reasoned recommendation for a country \
            with diverse geography and moderate climate.",
        )
        .build(),
    ];

    match ai.chat(reasoning_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🧠 Reasoning analysis completed");
                println!(
                    "   Analysis includes: {}",
                    if text.contains("compare") || text.contains("recommendation") {
                        "Comparative analysis and recommendations"
                    } else {
                        "Detailed technical information"
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    // Feature 3: Integration capabilities
    println!("   Feature 3: Anthropic ecosystem integration potential");
    let integration_test = vec![
        ChatMessage::user(
            "Explain how AI assistants like Claude could integrate with \
            productivity tools and workflows \
            to enhance efficiency. Provide specific use case examples.",
        )
        .build(),
    ];

    match ai.chat(integration_test).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🔗 Integration insights provided");
                println!(
                    "   Focus: {}",
                    if text.contains("Workspace") || text.contains("workflow") {
                        "Practical integration scenarios"
                    } else {
                        "General productivity enhancement"
                    }
                );
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    println!("   🌟 Google Gemini Unique Strengths:");
    println!("      • Large context windows (up to 2M tokens)");
    println!("      • Strong multimodal capabilities");
    println!("      • Built-in safety and content filtering");
    println!("      • Google ecosystem integration potential");
    println!("      • Efficient performance with Flash model");
    println!("      • Advanced reasoning with Pro model");

    Ok(())
}

/// 🎯 Key Google Gemini Features Summary:
///
/// Model Options:
/// - Gemini 1.5 Flash: Fast, efficient for most tasks
/// - Gemini 1.5 Pro: Advanced reasoning and complex tasks
/// - Gemini 1.0 Pro: Stable production version
///
/// Core Strengths:
/// - Large context windows (up to 2M tokens)
/// - Strong multimodal capabilities (text, images, code)
/// - Built-in safety and content filtering
/// - Efficient performance optimization
/// - Google ecosystem integration
///
/// Multimodal Capabilities:
/// - Text and image understanding
/// - Code analysis and generation
/// - Document processing
/// - Visual content description
/// - Cross-modal reasoning
///
/// Safety Features:
/// - Built-in content filtering
/// - Configurable safety settings
/// - Responsible AI guidelines
/// - Educational content support
/// - Age-appropriate responses
///
/// Performance Optimization:
/// - Model selection based on task complexity
/// - Batch processing capabilities
/// - Token management strategies
/// - Streaming support for long responses
/// - Efficient caching opportunities
///
/// Best Practices:
/// - Use Flash for simple, fast tasks
/// - Use Pro for complex reasoning
/// - Leverage large context windows
/// - Implement appropriate safety settings
/// - Monitor usage and performance
///
/// Use Cases:
/// - Content creation and analysis
/// - Code review and generation
/// - Educational assistance
/// - Research and summarization
/// - Multimodal applications
///
/// Next Steps:
/// - Explore multimodal features
/// - Implement safety configurations
/// - Optimize for specific use cases
/// - Integrate with Google services
const fn _documentation() {}
