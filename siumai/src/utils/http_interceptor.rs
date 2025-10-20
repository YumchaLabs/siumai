//! HTTP Interceptor interfaces
//!
//! This module defines a small, ergonomic interceptor API inspired by
//! middleware patterns in HTTP clients. Interceptors can observe and tweak
//! request builders before send, observe responses, be notified of errors, and
//! receive streaming SSE events. The hooks are best-effort and should avoid
//! expensive work by default.

use crate::error::LlmError;
use reqwest::header::HeaderMap;

/// Context passed to interceptors describing the request.
#[derive(Clone, Debug)]
pub struct HttpRequestContext {
    pub provider_id: String,
    pub url: String,
    pub stream: bool,
}

/// HTTP interceptor trait
pub trait HttpInterceptor: Send + Sync {
    /// Called before sending a request. Interceptors may add headers or modify
    /// attributes on the request builder. Return the (possibly modified)
    /// builder or an error to short-circuit the request.
    fn on_before_send(
        &self,
        _ctx: &HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        _headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        Ok(builder)
    }

    /// Called after a response is received (only for successful responses).
    fn on_response(
        &self,
        _ctx: &HttpRequestContext,
        _response: &reqwest::Response,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    /// Called when an error occurs during sending or classification.
    fn on_error(&self, _ctx: &HttpRequestContext, _error: &LlmError) {}

    /// Called when an SSE event is received in a streaming request.
    fn on_sse_event(
        &self,
        _ctx: &HttpRequestContext,
        _event: &eventsource_stream::Event,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// A simple logging interceptor backed by `tracing` (no sensitive data).
#[derive(Clone, Default)]
pub struct LoggingInterceptor;

impl HttpInterceptor for LoggingInterceptor {
    fn on_before_send(
        &self,
        ctx: &HttpRequestContext,
        builder: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        _headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        tracing::debug!(target: "siumai::http", provider=%ctx.provider_id, url=%ctx.url, stream=%ctx.stream, "sending request");
        Ok(builder)
    }

    fn on_response(
        &self,
        ctx: &HttpRequestContext,
        response: &reqwest::Response,
    ) -> Result<(), LlmError> {
        tracing::debug!(target: "siumai::http", provider=%ctx.provider_id, url=%ctx.url, status=%response.status().as_u16(), "response received");
        Ok(())
    }

    fn on_error(&self, ctx: &HttpRequestContext, error: &LlmError) {
        tracing::debug!(target: "siumai::http", provider=%ctx.provider_id, url=%ctx.url, stream=%ctx.stream, err=%error, "request error");
    }

    fn on_sse_event(
        &self,
        ctx: &HttpRequestContext,
        event: &eventsource_stream::Event,
    ) -> Result<(), LlmError> {
        tracing::trace!(target: "siumai::http", provider=%ctx.provider_id, url=%ctx.url, event_name=%event.event, "sse event");
        Ok(())
    }
}
