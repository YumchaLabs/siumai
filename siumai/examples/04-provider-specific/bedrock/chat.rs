//! Amazon Bedrock config-first chat example.
//!
//! This example demonstrates the provider-owned Bedrock surface:
//! - config-first construction via `BedrockConfig` / `BedrockClient`
//! - split endpoint ownership (`runtime` for chat, `agent-runtime` for rerank)
//! - request-level Bedrock typed options via `BedrockChatOptions`
//!
//! Authentication notes:
//! - Real AWS Bedrock usually requires SigV4-signed headers.
//! - Siumai intentionally does not sign Bedrock requests internally.
//! - You can inject pre-signed headers via `HttpConfig.headers`.
//! - For proxy / gateway compatibility, this example also accepts `BEDROCK_API_KEY`
//!   as a Bearer token.
//!
//! Run:
//! ```bash
//! cargo run --example bedrock-chat --features bedrock
//! ```
//!
//! Common environment variables:
//! - `AWS_REGION` / `AWS_DEFAULT_REGION`
//! - `BEDROCK_MODEL`
//! - `BEDROCK_BASE_URL` (optional runtime URL override)
//! - `BEDROCK_API_KEY` (optional Bearer/proxy compatibility)
//! - `BEDROCK_AUTHORIZATION` (optional pre-signed `Authorization` header)
//! - `BEDROCK_X_AMZ_DATE`
//! - `BEDROCK_X_AMZ_SECURITY_TOKEN`
//! - `BEDROCK_X_AMZ_CONTENT_SHA256`

use siumai::prelude::unified::*;
use siumai::provider_ext::bedrock::{
    BedrockChatOptions, BedrockChatRequestExt, BedrockClient, BedrockConfig,
};

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn apply_env_header(http: &mut HttpConfig, header_name: &str, env_name: &str) {
    if let Some(value) = read_non_empty_env(env_name) {
        http.headers.insert(header_name.to_string(), value);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let region = read_non_empty_env("AWS_REGION")
        .or_else(|| read_non_empty_env("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|| "us-east-1".to_string());
    let model = read_non_empty_env("BEDROCK_MODEL")
        .unwrap_or_else(|| "anthropic.claude-3-haiku-20240307-v1:0".to_string());

    let mut http_config = HttpConfig::default();
    apply_env_header(&mut http_config, "Authorization", "BEDROCK_AUTHORIZATION");
    apply_env_header(&mut http_config, "X-Amz-Date", "BEDROCK_X_AMZ_DATE");
    apply_env_header(
        &mut http_config,
        "X-Amz-Security-Token",
        "BEDROCK_X_AMZ_SECURITY_TOKEN",
    );
    apply_env_header(
        &mut http_config,
        "X-Amz-Content-Sha256",
        "BEDROCK_X_AMZ_CONTENT_SHA256",
    );

    let mut config = BedrockConfig::new()
        .with_region(region)
        .with_model(model)
        .with_http_config(http_config);

    if let Some(base_url) = read_non_empty_env("BEDROCK_BASE_URL") {
        config = config.with_base_url(base_url);
    }
    if let Some(api_key) = read_non_empty_env("BEDROCK_API_KEY") {
        config = config.with_api_key(api_key);
    }

    let has_auth = config.api_key.is_some()
        || config
            .http_config
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("authorization"));
    if !has_auth {
        eprintln!(
            "Missing Bedrock auth. Set BEDROCK_API_KEY for a proxy/bearer flow, or inject SigV4 headers via BEDROCK_AUTHORIZATION / BEDROCK_X_AMZ_DATE / BEDROCK_X_AMZ_SECURITY_TOKEN."
        );
        std::process::exit(2);
    }

    let client = BedrockClient::from_config(config)?;

    let request = ChatRequest::new(vec![user!(
        "Explain why provider-owned config/client surfaces are useful in a Rust SDK, in three concise bullet points."
    )])
    .with_bedrock_chat_options(
        BedrockChatOptions::new()
            .with_additional_model_request_fields(serde_json::json!({ "topK": 32 })),
    );

    let response = text::generate(&client, request, text::GenerateOptions::default()).await?;

    println!("{}", response.content_text().unwrap_or_default());
    Ok(())
}
