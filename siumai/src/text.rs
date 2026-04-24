//! Text model family APIs.
//!
//! This is the recommended Rust-first surface for text generation:
//! - `generate` for non-streaming
//! - `stream` for streaming
//! - `stream_with_cancel` for streaming with cancellation

use crate::request_options::{
    EffectiveRequestOptions, link_stream_handle_abort, retry_or_call_with_abort,
    wrap_stream_with_abort,
};
use crate::retry_api::RetryOptions;
use siumai_core::error::LlmError;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::text::{
    LanguageModel, TextModelV3, TextRequest, TextResponse, TextStream, TextStreamHandle,
};
pub use siumai_core::types::StreamRequestOptions;
use siumai_core::types::{HttpConfig, RequestOptions, Tool, ToolChoice};

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
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
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
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
    /// Optional tools to add to the request for this call.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice override for this call.
    pub tool_choice: Option<ToolChoice>,
    /// Optional telemetry config for this call (runtime-only).
    pub telemetry: Option<siumai_core::observability::telemetry::TelemetryConfig>,
    /// Include provider raw chunks on the stream part lane.
    pub include_raw_chunks: bool,
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

pub(crate) fn prepare_generate_request(
    request: TextRequest,
    options: GenerateOptions,
) -> (TextRequest, EffectiveRequestOptions) {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    (request, effective)
}

pub(crate) async fn generate_prepared<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    effective: EffectiveRequestOptions,
) -> Result<TextResponse, LlmError> {
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let req = request.clone();
        async move { model.generate(req).await }
    })
    .await
}

/// Generate a non-streaming text response.
pub async fn generate<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: GenerateOptions,
) -> Result<TextResponse, LlmError> {
    let (request, effective) = prepare_generate_request(request, options);
    generate_prepared(model, request, effective).await
}

/// Generate a streaming text response.
pub async fn stream<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStream, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }

    let abort_signal = effective.abort_signal();
    let stream = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let req = request.clone();
        async move { model.stream(req).await }
    })
    .await?;
    Ok(wrap_stream_with_abort(stream, abort_signal))
}

/// Generate a streaming text response with cancellation support.
pub async fn stream_with_cancel<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStreamHandle, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request = apply_text_call_options(
        request,
        effective.timeout(),
        effective.headers(),
        options.tools,
        options.tool_choice,
        options.telemetry,
    );
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }

    let abort_signal = effective.abort_signal();
    let handle = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let req = request.clone();
        async move { model.stream_with_cancel(req).await }
    })
    .await?;
    Ok(link_stream_handle_abort(handle, abort_signal))
}
