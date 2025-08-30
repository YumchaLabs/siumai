//! SiliconFlow Usage Examples
//!
//! This example demonstrates two ways to use SiliconFlow:
//! 1. Unified Interface (Siumai::builder) - Provider-agnostic code
//! 2. Provider-Specific Interface (LlmBuilder::new) - Direct provider access

use siumai::analysis::{PatternType, analyze_thinking_content};
use siumai::models::openai_compatible::siliconflow;
use siumai::prelude::*;
use siumai::traits::{ChatCapability, RerankCapability};
use siumai::types::RerankRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better debugging
    tracing_subscriber::fmt::init();

    // Get API key from environment
    let api_key = std::env::var("SILICONFLOW_API_KEY")
        .expect("SILICONFLOW_API_KEY environment variable not set");

    println!("🚀 SiliconFlow Usage Examples");
    println!("==============================\n");

    // =============================================================================
    // Method 1: Unified Interface (Recommended)
    // =============================================================================
    println!("📋 Method 1: Unified Interface (Siumai::builder)");
    println!("================================================");
    println!("✨ Benefits: Provider-agnostic, easy switching, consistent API\n");

    // Create client using unified interface
    let unified_client = Siumai::builder()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::DEEPSEEK_V3_1)
        .temperature(0.7)
        .build()
        .await?;

    println!("✅ Client created with unified interface");
    println!("   Provider: SiliconFlow");
    println!("   Model: {}", siliconflow::DEEPSEEK_V3_1);

    // Chat example with unified interface
    let messages = vec![user!("What is quantum computing? Keep it brief.")];
    let response = unified_client.chat(messages).await?;
    println!("\n💬 Chat Response (Unified):");
    println!(
        "   {}\n",
        response
            .content_text()
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>()
            + "..."
    );

    // =============================================================================
    // Method 2: Provider-Specific Interface
    // =============================================================================
    println!("📋 Method 2: Provider-Specific Interface (LlmBuilder::new)");
    println!("==========================================================");
    println!("✨ Benefits: Direct access, provider-specific optimizations, fine control\n");

    // Create client using provider-specific interface
    let provider_client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::QWEN3_235B_A22B) // Use base model (supports thinking)
        .temperature(0.8)
        .build()
        .await?;

    println!("✅ Client created with provider-specific interface");
    println!("   Provider: SiliconFlow (Direct)");
    println!("   Model: {}", siliconflow::QWEN3_235B_A22B);

    // Chat example with provider-specific interface
    let messages = vec![user!("Explain machine learning in one sentence.")];
    let response = provider_client.chat(messages).await?;
    println!("\n💬 Chat Response (Provider-Specific):");
    println!("   {}\n", response.content_text().unwrap_or_default());

    // =============================================================================
    // Advanced Example: Thinking Control
    // =============================================================================
    println!("🎛️ Advanced Example: Thinking Control");
    println!("======================================\n");

    // Example: Custom thinking budget for complex reasoning
    println!("💰 Testing Custom Thinking Budget (8K tokens):");
    let high_budget_client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::DEEPSEEK_V3_1)
        .with_thinking_budget(8192) // High budget for complex reasoning
        .temperature(0.1)
        .build()
        .await?;

    let complex_problem = "Design a scalable chat system for 1M users. Consider architecture, databases, and real-time messaging.";
    let messages = vec![
        system!(
            "Think through this system design problem step by step. Consider all technical aspects."
        ),
        user!(complex_problem),
    ];

    match high_budget_client.chat(messages).await {
        Ok(response) => {
            println!("   Problem: {}", complex_problem);

            if let Some(thinking) = &response.thinking {
                println!("\n   🧠 High-Budget Thinking Analysis:");
                display_thinking_analysis(thinking);
            }

            if let Some(text) = response.content_text() {
                let preview = if text.len() > 300 {
                    format!("{}...", &text[..300])
                } else {
                    text.to_string()
                };
                println!("\n   📝 System Design Solution:");
                println!("   {}", preview);
            }

            println!("   ✅ High-budget thinking completed");
        }
        Err(e) => {
            println!("   ❌ High-budget thinking failed: {}", e);
        }
    }

    println!();

    // =============================================================================
    // Advanced Example: Thinking Models (DeepSeek R1 & V3.1)
    // =============================================================================
    println!("🧠 Advanced Example: Thinking Models");
    println!("====================================\n");

    // Test DeepSeek R1 - Reasoning model with thinking process
    println!("🤔 Testing DeepSeek R1 Thinking Process:");
    let reasoning_client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::DEEPSEEK_R1)
        .temperature(0.3) // Lower temperature for consistent reasoning
        .build()
        .await?;

    let reasoning_messages = vec![
        system!("Think step by step about the problem. Show your reasoning process clearly."),
        user!(
            "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think through this carefully."
        ),
    ];

    match reasoning_client.chat(reasoning_messages).await {
        Ok(response) => {
            println!("   Problem: A farmer has 17 sheep. All but 9 die. How many sheep are left?");

            // Check if thinking process is available (should be in reasoning_content for DeepSeek)
            if let Some(thinking) = &response.thinking {
                println!("\n   🧠 DeepSeek R1's Thinking Process (from thinking field):");

                // Use advanced thinking analysis
                display_thinking_analysis(thinking);

                let thinking_preview = if thinking.len() > 300 {
                    format!("{}...", &thinking[..300])
                } else {
                    thinking.clone()
                };
                println!("\n   🔍 Thinking Preview:");
                println!("   {}", thinking_preview);

                println!("\n   📝 Final Answer:");
            } else {
                println!(
                    "\n   ℹ️  No thinking field found - checking response content for reasoning..."
                );
                println!("   📝 Response (may contain embedded reasoning):");
            }

            if let Some(text) = response.content_text() {
                println!("   {}", text);

                // If no dedicated thinking field, analyze the main content for reasoning patterns
                if response.thinking.is_none() {
                    let analysis = analyze_thinking_content(text);
                    if analysis.reasoning_steps > 0 || !analysis.patterns.is_empty() {
                        println!("\n   🔍 Embedded Reasoning Analysis:");
                        println!(
                            "      Steps: {}, Complexity: {:.2}",
                            analysis.reasoning_steps, analysis.complexity_score
                        );

                        let pattern_names: Vec<String> = analysis
                            .patterns
                            .iter()
                            .map(|p| {
                                match p.pattern_type {
                                    PatternType::Sequential => "Sequential",
                                    PatternType::Causal => "Causal",
                                    PatternType::ProblemSolving => "Problem-solving",
                                    _ => "Other",
                                }
                                .to_string()
                            })
                            .collect();

                        if !pattern_names.is_empty() {
                            println!("      Patterns: {}", pattern_names.join(", "));
                        }
                    }
                }
            }

            println!("   ✅ DeepSeek R1 thinking demonstration completed");
        }
        Err(e) => {
            println!("   ❌ DeepSeek R1 thinking failed: {}", e);
        }
    }
    println!();

    // Test DeepSeek V3.1 - Latest flagship model with explicit thinking parameters
    println!("🔬 Testing DeepSeek V3.1 Complex Reasoning:");
    let v31_client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::DEEPSEEK_V3_1)
        .temperature(0.2)
        .build()
        .await?;

    let complex_problem = "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water?";
    let v31_messages = vec![
        system!("Think through this problem step by step. Show your reasoning process clearly."),
        user!(complex_problem),
    ];

    match v31_client.chat(v31_messages).await {
        Ok(response) => {
            println!("   Problem: {}", complex_problem);

            // Check for thinking content in both possible fields
            let mut thinking_found = false;

            if let Some(thinking) = &response.thinking {
                println!("\n   🧠 DeepSeek V3.1's Thinking Process (from thinking field):");

                // Analyze thinking process using advanced analysis
                display_thinking_analysis(thinking);

                // Show thinking preview
                let thinking_preview = if thinking.len() > 400 {
                    format!("{}...", &thinking[..400])
                } else {
                    thinking.clone()
                };
                println!("\n   🔍 Thinking Preview:");
                println!("   {}", thinking_preview);

                thinking_found = true;
                println!("\n   📝 Final Solution:");
            }

            if let Some(text) = response.content_text() {
                if !thinking_found {
                    println!("\n   ℹ️  Checking response content for reasoning patterns...");

                    // Look for reasoning patterns in the main content
                    let reasoning_patterns = [
                        "step 1",
                        "step 2",
                        "first",
                        "then",
                        "next",
                        "finally",
                        "let me think",
                        "i need to",
                        "the solution is",
                        "approach",
                    ];

                    let pattern_matches = reasoning_patterns
                        .iter()
                        .map(|&pattern| text.to_lowercase().matches(pattern).count())
                        .sum::<usize>();

                    if pattern_matches > 0 {
                        println!(
                            "   💡 Found {} reasoning patterns in response",
                            pattern_matches
                        );
                        println!("   🧠 Response appears to contain embedded reasoning:");
                    } else {
                        println!("   📝 Standard response:");
                    }
                }

                println!("   {}", text);
            }

            println!("   ✅ DeepSeek V3.1 reasoning demonstration completed");
        }
        Err(e) => {
            println!("   ❌ DeepSeek V3.1 reasoning failed: {}", e);
        }
    }
    println!();

    // =============================================================================
    // Quick Document Reranking Example
    // =============================================================================
    println!("🔄 Quick Document Reranking Example:");
    let rerank_client = LlmBuilder::new()
        .siliconflow()
        .api_key(&api_key)
        .model(siliconflow::BGE_RERANKER_V2_M3)
        .build()
        .await?;

    let query = "artificial intelligence applications";
    let documents = vec![
        "AI is used in healthcare for medical diagnosis".to_string(),
        "Machine learning helps in financial fraud detection".to_string(),
        "Computer vision enables autonomous vehicle navigation".to_string(),
    ];

    let rerank_request = RerankRequest::new(
        siliconflow::BGE_RERANKER_V2_M3.to_string(),
        query.to_string(),
        documents,
    )
    .with_top_n(3);

    match rerank_client.rerank(rerank_request).await {
        Ok(rerank_response) => {
            println!("   Top {} documents:", rerank_response.results.len());
            for (i, result) in rerank_response.results.iter().enumerate() {
                println!(
                    "     {}. [Score: {:.4}] Document {}",
                    i + 1,
                    result.relevance_score,
                    result.index
                );
            }
            println!("   ✅ Document reranking successful");
        }
        Err(e) => {
            println!("   ❌ Document reranking failed: {}", e);
        }
    }

    // =============================================================================
    // Summary and Recommendations
    // =============================================================================
    println!("\n📊 Summary and Recommendations");
    println!("==============================");
    println!("🎯 Use Unified Interface (Siumai::builder) when:");
    println!("   • Building applications that might switch providers");
    println!("   • Want provider-agnostic code");
    println!("   • Need consistent API across different providers");
    println!();
    println!("🎯 Use Provider-Specific Interface (LlmBuilder::new) when:");
    println!("   • Need direct access to provider-specific features");
    println!("   • Building provider-specific integrations");
    println!("   • Want fine-grained control over configuration");
    println!();
    println!("🧠 SiliconFlow Thinking Models:");
    println!("   • DeepSeek R1: Specialized reasoning model with thinking process");
    println!("   • DeepSeek V3.1: Latest flagship with advanced reasoning capabilities");
    println!("   • Both models can expose their internal thinking process");
    println!("   • Use lower temperature (0.1-0.3) for consistent reasoning");
    println!();
    println!("💡 SiliconFlow capabilities demonstrated:");
    println!("   • Chat completions with thinking models");
    println!("   • Access to internal reasoning processes");
    println!("   • Document reranking with BGE models");
    println!("   • Multiple model types (reasoning, chat, embedding, rerank)");

    println!("\n✅ SiliconFlow usage examples completed!");
    Ok(())
}

/// Display thinking analysis results
fn display_thinking_analysis(thinking: &str) {
    let analysis = analyze_thinking_content(thinking);

    println!("      📊 Thinking Analysis:");
    println!(
        "         Length: {} characters, {} words",
        analysis.length, analysis.word_count
    );
    println!("         Reasoning steps: {}", analysis.reasoning_steps);
    println!("         Questions asked: {}", analysis.questions_count);
    println!(
        "         Complexity score: {:.2}/1.0",
        analysis.complexity_score
    );

    if !analysis.patterns.is_empty() {
        println!("         Reasoning patterns found:");
        for pattern in &analysis.patterns {
            let pattern_name = match pattern.pattern_type {
                PatternType::Sequential => "Sequential",
                PatternType::Causal => "Causal",
                PatternType::Comparative => "Comparative",
                PatternType::Hypothetical => "Hypothetical",
                PatternType::Analytical => "Analytical",
                PatternType::ProblemSolving => "Problem-solving",
                PatternType::SelfCorrection => "Self-correction",
                PatternType::Uncertainty => "Uncertainty",
            };
            println!(
                "           • {}: {} occurrences (confidence: {:.2})",
                pattern_name, pattern.count, pattern.confidence
            );
        }
    }

    if !analysis.insights.is_empty() {
        println!("         Key insights:");
        for insight in &analysis.insights {
            println!("           • {}", insight);
        }
    }
}
