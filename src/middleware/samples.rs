//! Sample middlewares for demonstration and testing
//!
//! English-only comments in code as requested.

use std::sync::Arc;

use crate::middleware::language_model::LanguageModelMiddleware;
use crate::types::ChatRequest;

/// Normalize temperature/top_p defaults (demo only).
#[derive(Clone, Default)]
pub struct DefaultParamsMiddleware;

impl LanguageModelMiddleware for DefaultParamsMiddleware {
    fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
        // If neither temperature nor top_p is set, set a safe temperature default.
        if req.common_params.temperature.is_none() && req.common_params.top_p.is_none() {
            req.common_params.temperature = Some(0.7);
        }
        req
    }
}

/// Clamp top_p to [0.0, 1.0] (demo only).
#[derive(Clone, Default)]
pub struct ClampTopPMiddleware;

impl LanguageModelMiddleware for ClampTopPMiddleware {
    fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
        if let Some(tp) = req.common_params.top_p {
            let clamped = if tp < 0.0 {
                0.0
            } else if tp > 1.0 {
                1.0
            } else {
                tp
            };
            req.common_params.top_p = Some(clamped);
        }
        req
    }
}

/// Helper to build a vector of middlewares from simple types.
pub fn chain_default_and_clamp() -> Vec<Arc<dyn LanguageModelMiddleware>> {
    vec![
        Arc::new(DefaultParamsMiddleware::default()),
        Arc::new(ClampTopPMiddleware::default()),
    ]
}
