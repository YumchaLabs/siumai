//! Siumai Anthropic Standard
//!
//! 本 crate 承载 Anthropic Messages API 的标准请求/响应/流式转换与适配器，
//! 作为 `siumai-core` 之上的标准层，不依赖具体 provider 实现。

pub const VERSION: &str = "0.0.1";

pub mod anthropic;

pub use anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
