//! Streaming adapter wrappers
//!
//! Reusable wrappers to layer middleware and HTTP interceptors around
//! SSE and JSON event converters. This removes duplication from the
//! chat executor and makes the pipeline composable for other
//! capabilities that stream.

use std::sync::Arc;

use crate::error::LlmError;
use crate::streaming::SseEventConverter;

/// Wraps a `StreamChunkTransformer` as an `SseEventConverter`.
///
/// This is a thin adapter so higher-level code can work uniformly
/// with `SseEventConverter` regardless of transformer implementation.
#[derive(Clone)]
pub struct TransformerConverter(
    pub Arc<dyn crate::execution::transformers::stream::StreamChunkTransformer>,
);

impl SseEventConverter for TransformerConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        self.0.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.0.handle_stream_end()
    }
}

/// Wrap SSE conversion with language model middlewares applied to each event.
#[derive(Clone)]
pub struct MiddlewareConverter<C> {
    pub middlewares:
        Vec<Arc<dyn crate::execution::middleware::language_model::LanguageModelMiddleware>>,
    pub req: crate::types::ChatRequest,
    pub convert: C,
}

impl<C> SseEventConverter for MiddlewareConverter<C>
where
    C: SseEventConverter + Clone + Send + Sync + 'static,
{
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        let mws = self.middlewares.clone();
        let req = self.req.clone();
        let inner = self.convert.clone();
        Box::pin(async move {
            let raw = inner.convert_event(event).await;
            let mut out = Vec::new();
            for item in raw.into_iter() {
                match item {
                    Ok(ev) => {
                        match crate::execution::middleware::language_model::apply_stream_event_chain(
                            &mws, &req, ev,
                        ) {
                            Ok(list) => out.extend(list.into_iter().map(Ok)),
                            Err(e) => out.push(Err(e)),
                        }
                    }
                    Err(e) => out.push(Err(e)),
                }
            }
            out
        })
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        match self.convert.handle_stream_end() {
            Some(Ok(ev)) => {
                match crate::execution::middleware::language_model::apply_stream_event_chain(
                    &self.middlewares,
                    &self.req,
                    ev,
                )
                .map(|mut v| v.pop())
                {
                    Ok(Some(last)) => Some(Ok(last)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            }
            other => other,
        }
    }
}

/// Wrap SSE conversion with HTTP interceptors for each raw SSE event.
#[derive(Clone)]
pub struct InterceptingConverter<C> {
    pub interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    pub ctx: crate::execution::http::interceptor::HttpRequestContext,
    pub convert: C,
}

impl<C> SseEventConverter for InterceptingConverter<C>
where
    C: SseEventConverter + Clone + Send + Sync + 'static,
{
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        let interceptors = self.interceptors.clone();
        let ctx = self.ctx.clone();
        let inner = self.convert.clone();
        Box::pin(async move {
            for it in &interceptors {
                if let Err(e) = it.on_sse_event(&ctx, &event) {
                    return vec![Err(e)];
                }
            }
            inner.convert_event(event).await
        })
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.convert.handle_stream_end()
    }
}

/// Wrap a `JsonEventConverter` with language model middlewares.
#[derive(Clone)]
pub struct MiddlewareJsonConverter {
    pub middlewares:
        Vec<Arc<dyn crate::execution::middleware::language_model::LanguageModelMiddleware>>,
    pub req: crate::types::ChatRequest,
    pub convert: Arc<dyn crate::streaming::JsonEventConverter>,
}

impl crate::streaming::JsonEventConverter for MiddlewareJsonConverter {
    fn convert_json<'a>(
        &'a self,
        json_data: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>,
                > + Send
                + Sync
                + 'a,
        >,
    > {
        let mws = self.middlewares.clone();
        let req = self.req.clone();
        let inner = self.convert.clone();
        Box::pin(async move {
            let raw = inner.convert_json(json_data).await;
            let mut out = Vec::new();
            for item in raw.into_iter() {
                match item {
                    Ok(ev) => {
                        match crate::execution::middleware::language_model::apply_stream_event_chain(
                            &mws, &req, ev,
                        ) {
                            Ok(list) => out.extend(list.into_iter().map(Ok)),
                            Err(e) => out.push(Err(e)),
                        }
                    }
                    Err(e) => out.push(Err(e)),
                }
            }
            out
        })
    }
}

