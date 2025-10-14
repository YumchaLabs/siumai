//! HTTP Request/Response Tracing
//!
//! This module provides tracing capabilities for HTTP requests and responses.

use super::events::{HttpEvent, HttpRequestInfo, HttpResponseInfo, NetworkTiming, TracingEvent};
use super::{SpanId, TraceId};
use crate::error::LlmError;
use reqwest::{Request, Response};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, error, info};

/// HTTP request tracing context
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Trace ID
    pub trace_id: TraceId,
    /// Span ID
    pub span_id: SpanId,
    /// Request start time
    pub start_time: Instant,
    /// Request information
    pub request_info: HttpRequestInfo,
    /// Network timing
    pub timing: NetworkTiming,
}

/// HTTP response tracing context
#[derive(Debug, Clone)]
pub struct ResponseContext {
    /// Request context
    pub request: RequestContext,
    /// Response information
    pub response_info: Option<HttpResponseInfo>,
    /// Total duration
    pub duration: Duration,
    /// Error information
    pub error: Option<String>,
}

/// HTTP tracer for monitoring HTTP requests and responses
#[derive(Debug, Clone)]
pub struct HttpTracer {
    /// Whether to include request/response bodies
    include_bodies: bool,
    /// Maximum body size to log
    max_body_size: usize,
    /// Whether to include sensitive headers
    include_sensitive_headers: bool,
}

impl HttpTracer {
    /// Create a new HTTP tracer
    pub fn new(
        include_bodies: bool,
        max_body_size: usize,
        include_sensitive_headers: bool,
    ) -> Self {
        Self {
            include_bodies,
            max_body_size,
            include_sensitive_headers,
        }
    }

    /// Start tracing an HTTP request
    pub async fn start_request(
        &self,
        trace_id: TraceId,
        request: &Request,
    ) -> Result<RequestContext, LlmError> {
        let span_id = SpanId::new();
        let start_time = Instant::now();

        // Extract request information
        let request_info = self.extract_request_info(request).await?;

        // Log request start
        info!(
            trace_id = %trace_id,
            span_id = %span_id,
            method = %request.method(),
            url = %request.url(),
            "HTTP request started"
        );

        debug!(
            trace_id = %trace_id,
            span_id = %span_id,
            headers = ?request_info.headers,
            body_size = request_info.body_size,
            "HTTP request details"
        );

        Ok(RequestContext {
            trace_id,
            span_id,
            start_time,
            request_info,
            timing: NetworkTiming {
                dns_lookup: None,
                tcp_connect: None,
                tls_handshake: None,
                time_to_first_byte: None,
                content_download: None,
            },
        })
    }

    /// End tracing an HTTP request with response
    pub async fn end_request(
        &self,
        context: RequestContext,
        response: Result<Response, reqwest::Error>,
    ) -> ResponseContext {
        let duration = context.start_time.elapsed();

        let (response_info, error) = match response {
            Ok(response) => {
                let response_info = self.extract_response_info(&response).await;
                match response_info {
                    Ok(info) => {
                        info!(
                            trace_id = %context.trace_id,
                            span_id = %context.span_id,
                            status_code = info.status_code,
                            duration_ms = duration.as_millis(),
                            "HTTP request completed successfully"
                        );
                        (Some(info), None)
                    }
                    Err(e) => {
                        error!(
                            trace_id = %context.trace_id,
                            span_id = %context.span_id,
                            error = %e,
                            duration_ms = duration.as_millis(),
                            "Failed to extract response info"
                        );
                        (None, Some(e.to_string()))
                    }
                }
            }
            Err(e) => {
                error!(
                    trace_id = %context.trace_id,
                    span_id = %context.span_id,
                    error = %e,
                    duration_ms = duration.as_millis(),
                    "HTTP request failed"
                );
                (None, Some(e.to_string()))
            }
        };

        ResponseContext {
            request: context,
            response_info,
            duration,
            error,
        }
    }

    /// Create a tracing event from response context
    pub fn create_event(&self, context: &ResponseContext) -> TracingEvent {
        TracingEvent::Http(HttpEvent {
            timestamp: SystemTime::now(),
            request: context.request.request_info.clone(),
            response: context.response_info.clone(),
            duration: Some(context.duration),
            error: context.error.clone(),
            timing: Some(context.request.timing.clone()),
        })
    }

    /// Extract request information from reqwest::Request
    async fn extract_request_info(&self, request: &Request) -> Result<HttpRequestInfo, LlmError> {
        let method = request.method().to_string();
        let url = request.url().to_string();
        let headers = self.extract_headers(request.headers());

        // Extract body if configured
        let (body, body_size) = if self.include_bodies {
            // Note: This is tricky with reqwest::Request as the body is consumed
            // In practice, we'd need to clone the body before sending the request
            (None, 0)
        } else {
            (None, 0)
        };

        let content_type = request
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Ok(HttpRequestInfo {
            method,
            url,
            headers,
            body,
            body_size,
            content_type,
        })
    }

    /// Extract response information from reqwest::Response
    async fn extract_response_info(
        &self,
        response: &Response,
    ) -> Result<HttpResponseInfo, LlmError> {
        let status_code = response.status().as_u16();
        let headers = self.extract_headers(response.headers());

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Extract body if configured
        let (body, body_size) = if self.include_bodies {
            // Note: This consumes the response body, so in practice we'd need to
            // clone the response or use a different approach
            (None, 0)
        } else {
            (None, 0)
        };

        Ok(HttpResponseInfo {
            status_code,
            headers,
            body,
            body_size,
            content_type,
        })
    }

    /// Extract headers from reqwest::HeaderMap
    fn extract_headers(&self, headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
        let mut result = HashMap::new();

        for (name, value) in headers {
            let name_str = name.as_str();

            // Skip sensitive headers if not configured to include them
            if !self.include_sensitive_headers && self.is_sensitive_header(name_str) {
                result.insert(name_str.to_string(), "[REDACTED]".to_string());
                continue;
            }

            if let Ok(value_str) = value.to_str() {
                result.insert(name_str.to_string(), value_str.to_string());
            } else {
                result.insert(name_str.to_string(), "[BINARY]".to_string());
            }
        }

        result
    }

    /// Check if a header is considered sensitive
    fn is_sensitive_header(&self, name: &str) -> bool {
        matches!(
            name.to_lowercase().as_str(),
            "authorization" | "cookie" | "set-cookie" | "x-api-key" | "api-key"
        )
    }

    /// Truncate body content if it exceeds max size
    #[allow(dead_code)] // Used by optional debug/pretty print paths
    fn truncate_body(&self, body: &str) -> String {
        if body.len() <= self.max_body_size {
            body.to_string()
        } else {
            format!(
                "{}... [TRUNCATED: {} bytes total]",
                &body[..self.max_body_size],
                body.len()
            )
        }
    }

    /// Try to format body as pretty JSON if possible
    #[allow(dead_code)] // Used by optional debug/pretty print paths
    fn format_body(&self, body: &str, content_type: Option<&str>) -> String {
        if let Some(ct) = content_type
            && ct.contains("application/json")
            && let Ok(json) = serde_json::from_str::<Value>(body)
            && let Ok(pretty) = serde_json::to_string_pretty(&json)
        {
            return self.truncate_body(&pretty);
        }
        self.truncate_body(body)
    }
}

impl Default for HttpTracer {
    fn default() -> Self {
        Self::new(false, 1024, false)
    }
}
