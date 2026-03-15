//! Moonshot AI - Long Context Processing Example
//!
//! This example demonstrates Moonshot AI's exceptional long-context capabilities.
//! Moonshot models can handle up to 256K tokens (Kimi K2) or 128K tokens (V1 models),
//! making them ideal for processing long documents, research papers, and books.
//!
//! Package tier:
//! - compat preset example on the shared OpenAI-compatible runtime
//! - preferred path in this file: config-first built-in compat construction
//! - use `moonshot-siumai-builder.rs` only when you specifically want a builder convenience comparison
//!
//! ## Features
//! - Process extremely long documents (up to 256K tokens)
//! - Multi-document analysis
//! - Long conversation history
//! - Document summarization and Q&A
//!
//! ## Run
//! ```bash
//! # Set your Moonshot API key
//! export MOONSHOT_API_KEY="your-api-key-here"
//!
//! cargo run --example moonshot-long-context --features openai
//! ```

use siumai::models;
use siumai::prelude::*;
use siumai::provider_ext::openai_compatible::OpenAiCompatibleClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use Kimi K2 for maximum context window (256K tokens).
    // Note: API key is automatically read from `MOONSHOT_API_KEY`.
    let client = OpenAiCompatibleClient::from_builtin_env(
        "moonshot",
        Some(models::openai_compatible::moonshot::KIMI_K2_0905_PREVIEW),
    )
    .await?;

    // Example 1: Long document summarization
    println!("Example 1: Long document summarization");

    let long_document = r#"
# The History of Artificial Intelligence

Artificial Intelligence (AI) has a rich and fascinating history spanning over seven decades...

## Early Beginnings (1950s-1960s)
The field of AI was officially founded in 1956 at the Dartmouth Conference, where John McCarthy 
coined the term "Artificial Intelligence." Early pioneers like Alan Turing, Marvin Minsky, and 
Herbert Simon laid the groundwork for what would become one of the most transformative 
technologies of our time.

The Turing Test, proposed by Alan Turing in 1950, became a fundamental benchmark for machine 
intelligence. Early AI programs like the Logic Theorist (1956) and the General Problem Solver 
(1957) demonstrated that machines could perform tasks requiring human-like reasoning.

## The First AI Winter (1970s-1980s)
Despite initial optimism, AI research faced significant challenges in the 1970s. The limitations 
of early computers, lack of data, and overpromised expectations led to reduced funding and 
interest—a period known as the "AI Winter."

However, this period also saw important developments in expert systems, which used rule-based 
approaches to solve specific problems in domains like medical diagnosis and chemical analysis.

## Renaissance and Machine Learning (1990s-2000s)
The 1990s brought renewed interest in AI, driven by increased computational power and the 
emergence of machine learning techniques. The victory of IBM's Deep Blue over chess champion 
Garry Kasparov in 1997 marked a significant milestone.

Statistical approaches and probabilistic methods began to dominate AI research, moving away 
from purely symbolic reasoning. Support Vector Machines, Random Forests, and other machine 
learning algorithms showed promising results across various applications.

## Deep Learning Revolution (2010s)
The 2010s witnessed an AI revolution powered by deep learning. The availability of large 
datasets, powerful GPUs, and algorithmic innovations led to breakthrough achievements:

- ImageNet competition (2012): AlexNet demonstrated the power of deep convolutional neural networks
- AlphaGo (2016): DeepMind's AI defeated world champion Go player Lee Sedol
- Transformer architecture (2017): "Attention is All You Need" paper revolutionized NLP
- GPT series: Large language models showed unprecedented language understanding capabilities

## Modern Era (2020s)
Today, AI has become ubiquitous in our daily lives. Large Language Models like GPT-4, Claude, 
and others have demonstrated remarkable capabilities in natural language understanding, 
generation, and reasoning.

Key developments include:
- Multimodal AI systems that can process text, images, audio, and video
- AI-powered tools for code generation, scientific research, and creative work
- Increased focus on AI safety, ethics, and alignment
- Edge AI and efficient models for mobile and IoT devices

## Future Directions
The future of AI holds immense promise and challenges:
- Artificial General Intelligence (AGI): Creating AI systems with human-level intelligence
- AI Safety and Alignment: Ensuring AI systems behave in accordance with human values
- Explainable AI: Making AI decisions more transparent and interpretable
- Quantum AI: Leveraging quantum computing for AI applications
- Neuromorphic Computing: Brain-inspired computing architectures

As we continue to advance AI technology, it's crucial to address ethical considerations, 
ensure responsible development, and work towards AI systems that benefit all of humanity.
"#;

    let response = text::generate(
        &client,
        ChatRequest::new(vec![user!(format!(
            "请用中文总结以下英文文档的主要内容，包括关键时间节点和重要事件：\n\n{}",
            long_document
        ))]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!(
        "Summary:\n{}\n",
        response.content_text().unwrap_or_default()
    );

    // Example 2: Multi-turn conversation with long context
    println!("Example 2: Multi-turn long-context follow-up");

    let mut conversation = vec![
        user!(format!(
            "Here's a long document about AI history:\n\n{}",
            long_document
        )),
        assistant!(
            "I've read the document about AI history. It covers the evolution of AI from the 1950s to the present day, including key milestones and future directions. How can I help you with this information?"
        ),
        user!("What were the main challenges during the first AI winter?"),
    ];

    let response = text::generate(
        &client,
        ChatRequest::new(conversation.clone()),
        text::GenerateOptions::default(),
    )
    .await?;
    println!(
        "Answer 1:\n{}\n",
        response.content_text().unwrap_or_default()
    );

    // Continue the conversation
    conversation.extend(response.to_messages());
    conversation.push(user!(
        "How did deep learning change the field in the 2010s?"
    ));

    let response = text::generate(
        &client,
        ChatRequest::new(conversation.clone()),
        text::GenerateOptions::default(),
    )
    .await?;
    println!(
        "Answer 2:\n{}\n",
        response.content_text().unwrap_or_default()
    );

    // Example 3: Comparing different context window models
    println!("Example 3: Context window comparison");
    println!("Model context windows:");
    println!("- Kimi K2 (0905): 256K tokens (~200K words)");
    println!("- Moonshot V1 128K: 128K tokens (~100K words)");
    println!("- Moonshot V1 32K: 32K tokens (~25K words)");
    println!("- Moonshot V1 8K: 8K tokens (~6K words)");
    println!();

    println!("Use case recommendations:");
    println!("- Books / research papers: use Kimi K2 or V1 128K");
    println!("- Long articles: use V1 32K");
    println!("- Short conversations: use V1 8K");
    println!();

    // Example 4: Document Q&A
    println!("Example 4: Document Q&A");

    let qa_response = text::generate(
        &client,
        ChatRequest::new(vec![user!(format!(
            "Document:\n{}\n\nQuestion: When was the term 'Artificial Intelligence' coined and by whom?",
            long_document
        ))]),
        text::GenerateOptions::default(),
    )
    .await?;

    println!("Question: When was the term 'Artificial Intelligence' coined and by whom?");
    println!(
        "Answer:\n{}\n",
        qa_response.content_text().unwrap_or_default()
    );
    println!("Notes:");
    println!("- Moonshot handles long Chinese and English documents well");
    println!("- Use Kimi K2 for the maximum context window");
    println!("- Long context helps with multi-document analysis and follow-up Q&A");
    println!("- This pattern works well for research or technical documentation");

    Ok(())
}
