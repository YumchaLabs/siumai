//! Sample middlewares for demonstration and testing
//!
//! English-only comments in code as requested.

use std::sync::Arc;

use crate::middleware::language_model::LanguageModelMiddleware;
use crate::stream::ChatStreamEvent;
use crate::types::ChatRequest;
use futures::StreamExt;

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

/// Simulate streaming by chunking final text when provider emits no deltas.
///
/// - If the underlying stream already emits `ContentDelta`, events pass through unchanged.
/// - If the stream only emits a final `StreamEnd` with text, this middleware will
///   synthesize `ContentDelta` events by splitting the text, optionally inserting delays,
///   and then forward the original `StreamEnd`.
#[derive(Clone)]
pub struct SimulateStreamingMiddleware {
    chunk_size: usize,
    delay_ms: Option<u64>,
}

impl SimulateStreamingMiddleware {
    pub fn new(chunk_size: usize, delay_ms: Option<u64>) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            delay_ms,
        }
    }
}

impl LanguageModelMiddleware for SimulateStreamingMiddleware {
    fn wrap_stream_async(
        &self,
        next: Arc<crate::middleware::language_model::StreamAsyncFn>,
    ) -> Arc<crate::middleware::language_model::StreamAsyncFn> {
        let chunk = self.chunk_size;
        let delay = self.delay_ms;
        Arc::new(move |req: ChatRequest| {
            let next = next.clone();
            Box::pin(async move {
                let s = next(req).await?;
                let mut saw_delta = false;
                let stream = async_stream::try_stream! {
                    use crate::types::ChatResponse;
                    let mut final_resp: Option<ChatResponse> = None;
                    let mut inner = s;
                    while let Some(item) = inner.next().await {
                        match item? {
                            ChatStreamEvent::ContentDelta { delta, index } => {
                                saw_delta = true;
                                yield ChatStreamEvent::ContentDelta { delta, index };
                            }
                            ChatStreamEvent::UsageUpdate { usage } => {
                                yield ChatStreamEvent::UsageUpdate { usage };
                            }
                            ChatStreamEvent::StreamStart { .. } => {
                                // pass through
                            }
                            ChatStreamEvent::StreamEnd { response } => {
                                final_resp = Some(response);
                                break;
                            }
                            other => {
                                // Pass through all other events
                                yield other;
                            }
                        }
                    }
                    if let Some(resp) = final_resp {
                        if !saw_delta {
                            let text = resp.content_text().map(|s| s.to_string()).unwrap_or_default();
                            if !text.is_empty() {
                                let mut i = 0;
                                while i < text.len() {
                                    let end = (i + chunk).min(text.len());
                                    let piece = text[i..end].to_string();
                                    yield ChatStreamEvent::ContentDelta { delta: piece, index: None };
                                    if let Some(ms) = delay { tokio::time::sleep(std::time::Duration::from_millis(ms)).await; }
                                    i = end;
                                }
                            }
                        }
                        // Always forward the original StreamEnd
                        yield ChatStreamEvent::StreamEnd { response: resp };
                    }
                };
                Ok(Box::pin(stream) as crate::stream::ChatStream)
            })
        })
    }
}

/// Convenience constructor.
pub fn simulate_streaming_middleware(
    chunk_size: usize,
    delay_ms: Option<u64>,
) -> Arc<dyn LanguageModelMiddleware> {
    Arc::new(SimulateStreamingMiddleware::new(chunk_size, delay_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::language_model::StreamAsyncFn;
    use futures::StreamExt;

    #[tokio::test]
    async fn simulate_streaming_chunks_when_no_deltas() {
        // Base stream: only StreamEnd with text "hello"
        let base: Arc<StreamAsyncFn> = Arc::new(|_req: ChatRequest| {
            Box::pin(async move {
                let s = async_stream::try_stream! {
                    yield ChatStreamEvent::StreamEnd { response: crate::types::ChatResponse::new(crate::types::MessageContent::Text("hello".into())) };
                };
                Ok(Box::pin(s) as crate::stream::ChatStream)
            })
        });
        let mw = SimulateStreamingMiddleware::new(2, None);
        let wrapped = mw.wrap_stream_async(base);
        let req = ChatRequest::new(vec![]);
        let s = wrapped(req).await.expect("stream");
        let evs: Vec<_> = s.collect().await;
        // Expect deltas: he, ll, o then StreamEnd
        let deltas: Vec<String> = evs
            .iter()
            .filter_map(|e| match e {
                Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["he", "ll", "o"]);
        assert!(
            evs.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        );
    }

    #[tokio::test]
    async fn simulate_streaming_passthrough_when_deltas_exist() {
        // Base stream: already has deltas, middleware should not synthesize
        let base: Arc<StreamAsyncFn> = Arc::new(|_req: ChatRequest| {
            Box::pin(async move {
                let s = async_stream::try_stream! {
                    yield ChatStreamEvent::ContentDelta{ delta: "a".into(), index: None };
                    yield ChatStreamEvent::StreamEnd { response: crate::types::ChatResponse::new(crate::types::MessageContent::Text("ab".into())) };
                };
                Ok(Box::pin(s) as crate::stream::ChatStream)
            })
        });
        let mw = SimulateStreamingMiddleware::new(1, None);
        let wrapped = mw.wrap_stream_async(base);
        let req = ChatRequest::new(vec![]);
        let s = wrapped(req).await.expect("stream");
        let evs: Vec<_> = s.collect().await;
        // Should only see original one delta, not split into more
        let delta_count = evs
            .iter()
            .filter(|e| matches!(e, Ok(ChatStreamEvent::ContentDelta { .. })))
            .count();
        assert_eq!(delta_count, 1);
    }
}
