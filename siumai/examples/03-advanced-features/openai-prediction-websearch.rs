//! OpenAI Predicted Outputs and Web Search Options Example
//!
//! This example demonstrates:
//! 1. Using Predicted Outputs to speed up response times when regenerating content
//! 2. Configuring web search options with context size and user location
//!
//! Learn more:
//! - Predicted Outputs: https://platform.openai.com/docs/guides/predicted-outputs
//! - Web Search: https://platform.openai.com/docs/guides/tools-web-search

use siumai::{
    chat::ChatRequestBuilder,
    types::{
        provider_options::openai::{
            OpenAiOptions, OpenAiWebSearchOptions, PredictionContent, WebSearchLocation,
        },
        ContentPart, Message, Role,
    },
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the client
    let client = Client::builder()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    println!("=== OpenAI Predicted Outputs and Web Search Options Demo ===\n");

    // Example 1: Predicted Outputs
    // When you're regenerating a file with minor changes, you can provide the original
    // content as a prediction to significantly speed up the response time.
    println!("1. Predicted Outputs Example");
    println!("   Regenerating a file with minor changes...\n");

    let original_content = r#"
# Project Documentation

## Overview
This is a sample project that demonstrates various features.

## Installation
```bash
npm install my-package
```

## Usage
Import the package and use it in your code:
```javascript
import { myFunction } from 'my-package';
myFunction();
```

## License
MIT License
"#;

    let request = ChatRequestBuilder::new("gpt-4o")
        .message(Message::user(vec![ContentPart::text(format!(
            "Update the installation section to use 'pnpm' instead of 'npm'. \
             Here's the original content:\n\n{}",
            original_content
        ))]))
        .openai_options(
            OpenAiOptions::new().with_prediction(PredictionContent::text(original_content)),
        )
        .build()?;

    match client.chat(request).await {
        Ok(response) => {
            println!("✅ Response (with prediction):");
            println!("{}\n", response.content);
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    // Example 2: Web Search Options with Context Size
    println!("2. Web Search Options - Context Size");
    println!("   Searching with high context size...\n");

    let request = ChatRequestBuilder::new("gpt-4o")
        .message(Message::user("What are the latest developments in Rust async runtime?"))
        .openai_options(
            OpenAiOptions::new()
                .with_web_search_options(OpenAiWebSearchOptions::new().with_context_size("high")),
        )
        .build()?;

    match client.chat(request).await {
        Ok(response) => {
            println!("✅ Response (high context):");
            println!("{}\n", response.content);
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    // Example 3: Web Search Options with User Location
    println!("3. Web Search Options - User Location");
    println!("   Searching with specific location (San Francisco, US)...\n");

    let request = ChatRequestBuilder::new("gpt-4o")
        .message(Message::user("What's the weather like today?"))
        .openai_options(
            OpenAiOptions::new().with_web_search_options(
                OpenAiWebSearchOptions::new()
                    .with_context_size("medium")
                    .with_location(
                        WebSearchLocation::new()
                            .with_country("US")
                            .with_region("California")
                            .with_city("San Francisco")
                            .with_timezone("America/Los_Angeles"),
                    ),
            ),
        )
        .build()?;

    match client.chat(request).await {
        Ok(response) => {
            println!("✅ Response (with location):");
            println!("{}\n", response.content);
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    // Example 4: Combined - Prediction + Web Search
    println!("4. Combined Example - Prediction + Web Search");
    println!("   Using both predicted outputs and web search...\n");

    let template_content = r#"
# Weekly Tech News Summary

## AI & Machine Learning
[Latest developments will be inserted here]

## Programming Languages
[Latest developments will be inserted here]

## Cloud & Infrastructure
[Latest developments will be inserted here]
"#;

    let request = ChatRequestBuilder::new("gpt-4o")
        .message(Message::user(vec![ContentPart::text(format!(
            "Fill in the latest tech news for this week. Use the template:\n\n{}",
            template_content
        ))]))
        .openai_options(
            OpenAiOptions::new()
                .with_prediction(PredictionContent::text(template_content))
                .with_web_search_options(
                    OpenAiWebSearchOptions::new()
                        .with_context_size("high")
                        .with_location(WebSearchLocation::new().with_country("US")),
                ),
        )
        .build()?;

    match client.chat(request).await {
        Ok(response) => {
            println!("✅ Response (prediction + web search):");
            println!("{}\n", response.content);
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    // Example 5: Prediction with Content Parts
    println!("5. Prediction with Content Parts");
    println!("   Using multimodal content parts for prediction...\n");

    let request = ChatRequestBuilder::new("gpt-4o")
        .message(Message::user("Add a conclusion section to this document"))
        .openai_options(
            OpenAiOptions::new().with_prediction(PredictionContent::parts(vec![
                ContentPart::text("# Introduction\n\nThis is the introduction."),
                ContentPart::text("# Main Content\n\nThis is the main content."),
            ])),
        )
        .build()?;

    match client.chat(request).await {
        Ok(response) => {
            println!("✅ Response (with content parts):");
            println!("{}\n", response.content);
        }
        Err(e) => {
            println!("❌ Error: {}\n", e);
        }
    }

    println!("=== Demo Complete ===");

    Ok(())
}

