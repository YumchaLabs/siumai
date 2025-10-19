//! Language-model-level middleware
//!
//! English-only comments in code as requested.
//!
//! This layer allows transforming high-level `ChatRequest` before provider
//! mapping, and (in future iterations) wrapping non-stream/stream calls.

use std::sync::Arc;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, ChatResponse};
use futures::future::BoxFuture;

// Wrapper function types for future extensions (not used in Iteration A)
pub type GenerateFn = dyn Fn(ChatRequest) -> Result<ChatResponse, LlmError> + Send + Sync;
pub type StreamFn =
    dyn Fn(ChatRequest) -> Result<crate::stream::ChatStream, LlmError> + Send + Sync;
/// Async wrapper function types for future around-style middleware.
pub type GenerateAsyncFn =
    dyn Fn(ChatRequest) -> BoxFuture<'static, Result<ChatResponse, LlmError>> + Send + Sync;
pub type StreamAsyncFn =
    dyn Fn(ChatRequest) -> BoxFuture<'static, Result<ChatStream, LlmError>> + Send + Sync;

/// Model-level middleware.
///
/// Iteration A: only `transform_params` is consumed by executors.
/// `wrap_generate` and `wrap_stream` are placeholders for future iterations.
pub trait LanguageModelMiddleware: Send + Sync {
    /// Transform high-level request before provider-specific mapping.
    fn transform_params(&self, req: ChatRequest) -> ChatRequest {
        req
    }

    /// Pre-generate short-circuit. If returns Some(Ok(resp)) or Some(Err(e)),
    /// the executor will short-circuit and skip HTTP completely.
    /// If returns None, the executor continues normally.
    fn pre_generate(&self, _req: &ChatRequest) -> Option<Result<ChatResponse, LlmError>> {
        None
    }

    /// Wrap non-stream generate call (not used in Iteration A).
    fn wrap_generate(
        &self,
        _do_generate: &GenerateFn,
    ) -> Option<Box<dyn Fn(ChatRequest) -> Result<ChatResponse, LlmError> + Send + Sync>> {
        None
    }

    /// Wrap stream call (not used in Iteration A).
    fn wrap_stream(
        &self,
        _do_stream: &StreamFn,
    ) -> Option<Box<dyn Fn(ChatRequest) -> Result<crate::stream::ChatStream, LlmError> + Send + Sync>>
    {
        None
    }

    /// Pre-stream short-circuit. If returns Some(Ok(stream)) or Some(Err(e)),
    /// the executor will short-circuit and skip HTTP completely.
    /// If returns None, the executor continues normally.
    fn pre_stream(&self, _req: &ChatRequest) -> Option<Result<ChatStream, LlmError>> {
        None
    }

    /// Post-process non-stream response. Runs after HTTP response is parsed.
    /// Default: no-op.
    fn post_generate(
        &self,
        _req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(resp)
    }

    /// Intercept a single ChatStreamEvent. May return zero or more events.
    /// Default: pass-through.
    fn on_stream_event(
        &self,
        _req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        Ok(vec![ev])
    }

    /// Around-style async wrappers (future use). Default: passthrough.
    fn wrap_generate_async(&self, next: Arc<GenerateAsyncFn>) -> Arc<GenerateAsyncFn> {
        next
    }

    fn wrap_stream_async(&self, next: Arc<StreamAsyncFn>) -> Arc<StreamAsyncFn> {
        next
    }

    /// Optional provider override (not used in Iteration A).
    fn override_provider_id(&self, _current: &str) -> Option<String> {
        None
    }

    /// Optional model override (not used in Iteration A).
    fn override_model_id(&self, _current: &str) -> Option<String> {
        None
    }
}

/// Apply `transform_params` across middlewares in order.
pub fn apply_transform_chain(
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    mut req: ChatRequest,
) -> ChatRequest {
    for mw in middlewares {
        req = mw.transform_params(req);
    }
    req
}

/// Try pre-generate short-circuit middlewares in reverse order (last registered runs first).
/// Returns Some(Result<ChatResponse, LlmError>) if short-circuited; None to continue.
pub fn try_pre_generate(
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    req: &ChatRequest,
) -> Option<Result<ChatResponse, LlmError>> {
    for mw in middlewares.iter().rev() {
        if let Some(out) = mw.pre_generate(req) {
            return Some(out);
        }
    }
    None
}

/// Try pre-stream short-circuit middlewares in reverse order (last registered runs first).
/// Returns Some(Result<ChatStream, LlmError>) if short-circuited; None to continue.
pub fn try_pre_stream(
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    req: &ChatRequest,
) -> Option<Result<ChatStream, LlmError>> {
    for mw in middlewares.iter().rev() {
        if let Some(out) = mw.pre_stream(req) {
            return Some(out);
        }
    }
    None
}

/// Apply post-generate processors in registration order.
pub fn apply_post_generate_chain(
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    req: &ChatRequest,
    mut resp: ChatResponse,
) -> Result<ChatResponse, LlmError> {
    for mw in middlewares {
        resp = mw.post_generate(req, resp)?;
    }
    Ok(resp)
}

/// Apply stream event processors in registration order.
pub fn apply_stream_event_chain(
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    req: &ChatRequest,
    ev: ChatStreamEvent,
) -> Result<Vec<ChatStreamEvent>, LlmError> {
    let mut events = vec![ev];
    for mw in middlewares {
        let mut next_batch = Vec::new();
        for e in events.into_iter() {
            let mut produced = mw.on_stream_event(req, e)?;
            next_batch.append(&mut produced);
        }
        events = next_batch;
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AppendModelSuffix(&'static str);
    impl LanguageModelMiddleware for AppendModelSuffix {
        fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
            req.common_params.model.push_str(self.0);
            req
        }
    }

    struct PreGenOnce;
    impl LanguageModelMiddleware for PreGenOnce {
        fn pre_generate(&self, req: &ChatRequest) -> Option<Result<ChatResponse, LlmError>> {
            if req.common_params.model == "hit" {
                Some(Ok(ChatResponse::new(crate::types::MessageContent::Text(
                    "short-circuit".to_string(),
                ))))
            } else {
                None
            }
        }
    }

    #[test]
    fn transform_chain_applies_in_order() {
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "base".to_string();
        let mws: Vec<Arc<dyn LanguageModelMiddleware>> = vec![
            Arc::new(AppendModelSuffix("-a")),
            Arc::new(AppendModelSuffix("-b")),
        ];
        let out = apply_transform_chain(&mws, req);
        assert!(out.common_params.model.ends_with("base-a-b"));
    }

    #[test]
    fn pre_generate_short_circuit_and_order() {
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "hit".to_string();
        let mws: Vec<Arc<dyn LanguageModelMiddleware>> = vec![Arc::new(PreGenOnce)];
        let out = try_pre_generate(&mws, &req).unwrap().unwrap();
        assert_eq!(out.content_text().unwrap_or_default(), "short-circuit");
    }

    struct PostAppendSuffix(&'static str);
    impl LanguageModelMiddleware for PostAppendSuffix {
        fn post_generate(
            &self,
            _req: &ChatRequest,
            mut resp: ChatResponse,
        ) -> Result<ChatResponse, LlmError> {
            if let Some(t) = resp.content_text() {
                let mut s = t.to_string();
                s.push_str(self.0);
                resp.content = crate::types::MessageContent::Text(s);
            }
            Ok(resp)
        }
    }

    #[test]
    fn post_generate_chain_applies_in_order() {
        let req = ChatRequest::new(vec![]);
        let base = ChatResponse::new(crate::types::MessageContent::Text("x".into()));
        let mws: Vec<Arc<dyn LanguageModelMiddleware>> = vec![
            Arc::new(PostAppendSuffix("-a")),
            Arc::new(PostAppendSuffix("-b")),
        ];
        let out = apply_post_generate_chain(&mws, &req, base).unwrap();
        assert_eq!(out.content_text().unwrap_or_default(), "x-a-b");
    }
}
