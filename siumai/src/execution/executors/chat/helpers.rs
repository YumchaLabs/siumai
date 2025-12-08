use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::execution::transformers::request::RequestTransformer;
use crate::streaming::adapters::{
    InterceptingConverter, MiddlewareConverter, MiddlewareJsonConverter, TransformerConverter,
};

// -----------------------------------------------------------------------------
// Helper builders for SSE/JSON streaming sources (no behavior change)
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) async fn create_sse_stream_with_middlewares(
    provider_id: String,
    url: String,
    http: reqwest::Client,
    headers_base: reqwest::header::HeaderMap,
    transformed: serde_json::Value,
    sse_tx: Arc<dyn crate::execution::transformers::stream::StreamChunkTransformer>,
    interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    req_in: crate::types::ChatRequest,
    disable_compression: bool,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> Result<crate::streaming::ChatStream, LlmError> {
    let converter = TransformerConverter(sse_tx.clone());
    let mw_wrapped = MiddlewareConverter {
        middlewares: middlewares.clone(),
        req: req_in.clone(),
        convert: converter,
    };
    let intercepting = InterceptingConverter {
        interceptors: interceptors.clone(),
        ctx: crate::execution::http::interceptor::HttpRequestContext {
            provider_id: provider_id.clone(),
            url: url.clone(),
            stream: true,
        },
        convert: mw_wrapped,
    };

    // Prepare body (OpenAI stream flags) before calling common helper
    let mut body_for_send = transformed.clone();
    if provider_id.starts_with("openai") {
        body_for_send["stream"] = serde_json::Value::Bool(true);
        if body_for_send.get("stream_options").is_none() {
            body_for_send["stream_options"] = serde_json::json!({ "include_usage": true });
        } else if let Some(obj) = body_for_send["stream_options"].as_object_mut() {
            obj.entry("include_usage")
                .or_insert(serde_json::Value::Bool(true));
        }
    }

    crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers(
        &http,
        &provider_id,
        &url,
        headers_base,
        body_for_send,
        &interceptors,
        retry_options,
        req_in.http_config.as_ref().map(|hc| hc.headers.clone()),
        intercepting,
        disable_compression,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn create_json_stream_with_middlewares(
    provider_id: String,
    url: String,
    http: reqwest::Client,
    headers_base: reqwest::header::HeaderMap,
    transformed: serde_json::Value,
    json_conv: Arc<dyn crate::streaming::JsonEventConverter>,
    interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    req_in: crate::types::ChatRequest,
    disable_compression: bool,
) -> Result<crate::streaming::ChatStream, LlmError> {
    let mw = MiddlewareJsonConverter {
        middlewares: middlewares.clone(),
        req: req_in.clone(),
        convert: json_conv.clone(),
    };
    let per_req_headers = req_in.http_config.as_ref().map(|hc| &hc.headers);
    crate::execution::executors::stream_json::execute_json_stream_request_with_headers(
        &http,
        &provider_id,
        &url,
        headers_base,
        transformed,
        &interceptors,
        None,
        per_req_headers,
        mw,
        disable_compression,
    )
    .await
}

// -----------------------------------------------------------------------------
// Small helper to build chat JSON body with middlewares + before_send
// -----------------------------------------------------------------------------

pub(super) fn build_chat_body(
    request_tx: &Arc<dyn RequestTransformer>,
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    before_send: &Option<crate::execution::executors::BeforeSendHook>,
    req_in: &crate::types::ChatRequest,
) -> Result<serde_json::Value, LlmError> {
    let mut body = request_tx.transform_chat(req_in)?;
    crate::execution::middleware::language_model::apply_json_body_transform_chain(
        middlewares,
        req_in,
        &mut body,
    )?;
    let out = if let Some(cb) = before_send {
        cb(&body)?
    } else {
        body
    };
    Ok(out)
}
