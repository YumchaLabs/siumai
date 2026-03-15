//! Text model family APIs.
//!
//! This is the recommended Rust-first surface for text generation:
//! - `generate` for non-streaming
//! - `stream` for streaming
//! - `stream_with_cancel` for streaming with cancellation

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::text::{
    LanguageModel, TextModelV3, TextRequest, TextResponse, TextStream, TextStreamHandle,
};
use siumai_core::types::{HttpConfig, Tool, ToolChoice};

/// Options for `text::generate`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `ChatRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `ChatRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// Optional tools to add to the request for this call.
    ///
    /// When the request already has tools, these are appended.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice override for this call.
    pub tool_choice: Option<ToolChoice>,
    /// Optional telemetry config for this call.
    ///
    /// This is applied to `ChatRequest.telemetry` (runtime-only; not serialized).
    pub telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
}

/// Options for `text::stream`.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Optional retry policy applied when establishing the stream.
    ///
    /// Note: this retries stream *creation* only. It does not retry mid-stream failures.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    pub headers: HashMap<String, String>,
    /// Optional tools to add to the request for this call.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice override for this call.
    pub tool_choice: Option<ToolChoice>,
    /// Optional telemetry config for this call (runtime-only).
    pub telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
}

fn apply_text_call_options(
    mut request: TextRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
) -> TextRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }

    if let Some(ts) = tools {
        match request.tools.as_mut() {
            Some(existing) => existing.extend(ts),
            None => request.tools = Some(ts),
        }
    }

    if let Some(choice) = tool_choice {
        request.tool_choice = Some(choice);
    }

    if let Some(tel) = telemetry {
        request.telemetry = Some(tel);
    }

    request
}

/// Generate a non-streaming text response.
pub async fn generate<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: GenerateOptions,
) -> Result<TextResponse, LlmError> {
    let request = apply_text_call_options(
        request,
        options.timeout,
        options.headers,
        options.tools,
        options.tool_choice,
        options.telemetry,
    );

    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.generate(req).await }
            },
            retry,
        )
        .await
    } else {
        model.generate(request).await
    }
}

/// Generate a streaming text response.
pub async fn stream<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStream, LlmError> {
    let request = apply_text_call_options(
        request,
        options.timeout,
        options.headers,
        options.tools,
        options.tool_choice,
        options.telemetry,
    );

    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.stream(req).await }
            },
            retry,
        )
        .await
    } else {
        model.stream(request).await
    }
}

/// Generate a streaming text response with cancellation support.
pub async fn stream_with_cancel<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStreamHandle, LlmError> {
    let request = apply_text_call_options(
        request,
        options.timeout,
        options.headers,
        options.tools,
        options.tool_choice,
        options.telemetry,
    );

    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.stream_with_cancel(req).await }
            },
            retry,
        )
        .await
    } else {
        model.stream_with_cancel(request).await
    }
}
