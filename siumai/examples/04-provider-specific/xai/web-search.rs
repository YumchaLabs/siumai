//! xAI web search with typed provider options and metadata helpers.
//!
//! Package tier:
//! - provider-owned wrapper package
//! - preferred path in this example: config-first `XaiClient`, because search controls
//!   and metadata helpers are xAI-specific provider surface
//!
//! Credentials:
//! - set `XAI_API_KEY` before running this example
//!
//! Run:
//! ```bash
//! cargo run --example xai-web-search --features xai
//! ```

use siumai::models;
use siumai::prelude::unified::*;
use siumai::provider_ext::xai::{
    SearchMode, SearchSource, SearchSourceType, XaiChatRequestExt, XaiChatResponseExt, XaiClient,
    XaiOptions, XaiSearchParameters,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = XaiClient::from_builtin_env(Some(models::xai::grok_4::GROK_4_LATEST)).await?;

    let search = XaiSearchParameters {
        mode: SearchMode::On,
        return_citations: Some(true),
        max_search_results: Some(5),
        from_date: None,
        to_date: None,
        sources: Some(vec![SearchSource {
            source_type: SearchSourceType::Web,
            country: Some("US".to_string()),
            allowed_websites: Some(vec![
                "blog.rust-lang.org".to_string(),
                "this-week-in-rust.org".to_string(),
            ]),
            excluded_websites: None,
            safe_search: Some(true),
        }]),
    };

    let request = ChatRequest::new(vec![user!(
        "Find recent Rust language or ecosystem news and summarize the three most important developments."
    )])
    .with_xai_options(XaiOptions::new().with_search(search));

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    if let Some(metadata) = response.xai_metadata()
        && let Some(sources) = metadata.sources
        && !sources.is_empty()
    {
        println!("Sources:");
        for source in sources {
            let title = source.title.unwrap_or_else(|| "Untitled".to_string());
            println!("- {} ({})", title, source.url);
        }
    }

    Ok(())
}
