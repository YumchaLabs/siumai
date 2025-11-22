//! Siumai Gemini Provider (extracted)
//!
//! 该 crate 承载 Gemini provider 的核心实现，包括：
//! - 基于 `siumai-core` / `siumai-std-gemini` 的 `CoreProviderSpec`
//! - 与 Gemini API 对齐的 headers 构建策略
//!
//! 聚合层 (`siumai` crate) 通过 feature gate
//! `provider-gemini-external` 来接入这里的实现。

/// 版本标记（scaffolding 阶段）
pub const VERSION: &str = "0.0.1-scaffolding";

/// Marker 类型，可用于测试或调试。
#[derive(Debug, Clone, Default)]
pub struct GeminiProviderMarker;

/// HTTP helpers for Gemini API.
pub mod headers;
/// Core-level provider spec implementation.
pub mod spec;

pub use headers::build_gemini_json_headers;
pub use spec::GeminiCoreSpec;
