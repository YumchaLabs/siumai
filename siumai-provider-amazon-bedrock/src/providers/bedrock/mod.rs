//! `Bedrock` provider module.
//!
//! This module exposes a provider-owned config/client/builder surface for the
//! Amazon Bedrock Converse and rerank integrations.

pub mod builder;
pub mod client;
pub mod config;
pub mod ext;

pub use builder::BedrockBuilder;
pub use client::BedrockClient;
pub use config::BedrockConfig;
pub use ext::{BedrockChatRequestExt, BedrockRerankRequestExt};
