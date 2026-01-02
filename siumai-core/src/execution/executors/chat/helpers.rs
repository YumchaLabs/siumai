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
    provider_spec: Arc<dyn crate::core::ProviderSpec>,
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
    let request_id = crate::execution::http::interceptor::generate_request_id();
    let intercepting = InterceptingConverter {
        interceptors: interceptors.clone(),
        ctx: crate::execution::http::interceptor::HttpRequestContext {
            request_id: request_id.clone(),
            provider_id: provider_id.clone(),
            url: url.clone(),
            stream: true,
        },
        convert: mw_wrapped,
    };

    crate::execution::executors::stream_sse::execute_sse_stream_request_with_headers(
        &http,
        &provider_id,
        Some(provider_spec.as_ref()),
        &url,
        request_id,
        headers_base,
        transformed,
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
    provider_spec: Arc<dyn crate::core::ProviderSpec>,
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
        Some(provider_spec.as_ref()),
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
    provider_spec: &Arc<dyn crate::core::ProviderSpec>,
    provider_context: &crate::core::ProviderContext,
    before_send: &Option<crate::execution::executors::BeforeSendHook>,
    req_in: &crate::types::ChatRequest,
) -> Result<serde_json::Value, LlmError> {
    let mut body = request_tx.transform_chat(req_in)?;
    crate::execution::middleware::language_model::apply_json_body_transform_chain(
        middlewares,
        req_in,
        &mut body,
    )?;
    if let Some(cb) = provider_spec.chat_before_send(req_in, provider_context) {
        body = cb(&body)?;
    }
    if let Some(cb) = before_send {
        body = cb(&body)?;
    }
    Ok(body)
}
