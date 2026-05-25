mod builder;
mod client;
mod config;
mod ext;

pub use builder::GatewayBuilder;
pub use client::GatewayClient;
pub use config::GatewayConfig;
pub use ext::{GatewayChatRequestExt, GatewayEmbeddingRequestExt};

pub mod model_constants {
    pub const GPT_5_MINI: &str = "openai/gpt-5-mini";
    pub const TEXT_EMBEDDING_3_SMALL: &str = "openai/text-embedding-3-small";
}
