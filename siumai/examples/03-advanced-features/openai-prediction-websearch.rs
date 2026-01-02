//! OpenAI Predicted Outputs + Web Search Options
//!
//! This example demonstrates:
//! 1) Predicted outputs (provide prior content to speed up regeneration)
//! 2) Web search via OpenAI provider-defined tools (Responses API)
//!
//! Learn more:
//! - Predicted Outputs: https://platform.openai.com/docs/guides/predicted-outputs
//! - Web Search: https://platform.openai.com/docs/guides/tools-web-search
//!
//! Run:
//! ```bash
//! cargo run --example openai-prediction-websearch --features openai
//! ```

use siumai::hosted_tools::openai as openai_tools;
use siumai::prelude::*;
use siumai::provider_ext::openai::{
    OpenAiChatRequestExt, OpenAiOptions, PredictionContent, ResponsesApiConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o")
        .build()
        .await?;

    println!("OpenAI Predicted Outputs + Web Search Options Demo\n");

    predicted_outputs(&client).await?;
    web_search_context_size(&client).await?;
    web_search_location(&client).await?;
    two_phase_workflow(&client).await?;

    Ok(())
}

async fn predicted_outputs(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("1) Predicted Outputs\n");

    let original_content = r#"
# Project Documentation
#
## Overview
This is a sample project that demonstrates various features.
#
## Installation
```bash
npm install my-package
```
#
## Usage
Import the package and use it in your code:
```javascript
import { myFunction } from 'my-package';
myFunction();
```
#
## License
MIT License
"#;

    let req = ChatRequest::new(vec![user!(format!(
        "Update the installation section to use 'pnpm' instead of 'npm'. \
Here is the original content:\n\n{}",
        original_content
    ))])
    .with_openai_options(
        OpenAiOptions::new().with_prediction(PredictionContent::text(original_content)),
    );

    let resp = client.chat_request(req).await?;
    println!("Text:\n{}\n", resp.content_text().unwrap_or_default());
    Ok(())
}

async fn web_search_context_size(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("2) Web Search (Responses API): context size\n");

    let req = ChatRequest::new(vec![user!(
        "What are the latest developments in Rust async runtime?"
    )])
    .with_tools(vec![
        openai_tools::web_search()
            .with_search_context_size("high")
            .build(),
    ])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let resp = client.chat_request(req).await?;
    println!("Text:\n{}\n", resp.content_text().unwrap_or_default());
    Ok(())
}

async fn web_search_location(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("3) Web Search (Responses API): user location\n");

    let req = ChatRequest::new(vec![user!("What's the weather like today?")])
        .with_tools(vec![
            openai_tools::web_search()
                .with_search_context_size("medium")
                .with_user_location(
                    openai_tools::UserLocation::new("approximate")
                        .with_country("US")
                        .with_region("California")
                        .with_city("San Francisco")
                        .with_timezone("America/Los_Angeles"),
                )
                .build(),
        ])
        .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let resp = client.chat_request(req).await?;
    println!("Text:\n{}\n", resp.content_text().unwrap_or_default());
    Ok(())
}

async fn two_phase_workflow(client: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("4) Two-phase workflow: prediction (Chat Completions) + web search (Responses)\n");

    let template_content = r#"
# Weekly Tech News Summary
#
## AI & Machine Learning
[Latest developments will be inserted here]
#
## Programming Languages
[Latest developments will be inserted here]
#
## Cloud & Infrastructure
[Latest developments will be inserted here]
"#;

    // Phase 1: predicted outputs are currently supported on the Chat Completions path.
    let req1 = ChatRequest::new(vec![user!(format!(
        "Fill in the latest tech news for this week using the template:\n\n{}",
        template_content
    ))])
    .with_openai_options(
        OpenAiOptions::new().with_prediction(PredictionContent::text(template_content)),
    );

    let resp1 = client.chat_request(req1).await?;
    println!(
        "Phase 1 (prediction) text:\n{}\n",
        resp1.content_text().unwrap_or_default()
    );

    // Phase 2: web search is configured as a provider-defined tool on the Responses API path.
    let req2 = ChatRequest::new(vec![user!(
        "Summarize the latest tech news for this week (AI/ML, Programming Languages, Cloud/Infra)."
    )])
    .with_tools(vec![
        openai_tools::web_search()
            .with_search_context_size("high")
            .build(),
    ])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let resp2 = client.chat_request(req2).await?;
    println!(
        "Phase 2 (web search) text:\n{}\n",
        resp2.content_text().unwrap_or_default()
    );
    Ok(())
}
