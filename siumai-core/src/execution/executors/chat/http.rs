use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::middleware::language_model::{
    GenerateAsyncFn, LanguageModelMiddleware, StreamAsyncFn, apply_post_generate_chain,
    apply_transform_chain, try_pre_generate, try_pre_stream,
};
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::ChatStream;
use crate::types::{ChatRequest, ChatResponse};

use super::ChatExecutor;
use super::helpers::{
    build_chat_body, build_effective_chat_request_headers, create_json_stream_with_middlewares,
    create_sse_stream_with_middlewares,
};

/// Generic HTTP-based ChatExecutor that wires transformers and HTTP
pub struct HttpChatExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RequestTransformer>,
    pub response_transformer: Arc<dyn ResponseTransformer>,
    pub stream_transformer: Option<Arc<dyn StreamChunkTransformer>>,
    /// Optional JSON streaming converter for providers that emit JSON lines
    pub json_stream_converter: Option<Arc<dyn crate::streaming::JsonEventConverter>>,
    /// Execution policy (interceptors/retry/before_send/stream flags)
    pub policy: crate::execution::ExecutionPolicy,
    /// Optional model-level middlewares (transform ChatRequest before mapping)
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Provider spec for building headers and URLs
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    /// Provider context for header/URL construction
    pub provider_context: crate::core::ProviderContext,
}

#[async_trait::async_trait]
impl ChatExecutor for HttpChatExecutor {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Initialize telemetry if enabled
        let trace_id = uuid::Uuid::new_v4().to_string();
        let span_id = uuid::Uuid::new_v4().to_string();
        let start_time = std::time::SystemTime::now();
        let telemetry_config = req.telemetry.clone();

        crate::execution::telemetry::chat::span_start(
            telemetry_config.as_ref(),
            &trace_id,
            &span_id,
            &self.provider_id,
            &req.common_params.model,
            false,
        )
        .await;

        // Apply model-level parameter transforms
        let req = apply_transform_chain(&self.middlewares, req);
        // Try pre-generate short-circuit
        if let Some(decision) = try_pre_generate(&self.middlewares, &req) {
            // Emit telemetry span end for short-circuit
            crate::execution::telemetry::chat::span_end_ok(
                telemetry_config.as_ref(),
                &trace_id,
                &span_id,
                true,
                None,
            )
            .await;
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let client = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let response_tx = self.response_transformer.clone();
        let interceptors = self.policy.interceptors.clone();
        let transport = self.policy.transport.clone();
        let before_send = self.policy.before_send.clone();
        let middlewares = self.middlewares.clone();
        // Pre-compute URL (provider/base-level). Request-level headers are merged later per-request.
        let url = self
            .provider_spec
            .chat_url(false, &req, &self.provider_context);
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let retry_options = self.policy.retry_options.clone();

        // Base async generator (no post_generate here)
        let base: Arc<GenerateAsyncFn> = Arc::new(move |req_in: ChatRequest| {
            let url = url.clone();
            let client = client.clone();
            let request_tx = request_tx.clone();
            let response_tx = response_tx.clone();
            let interceptors = interceptors.clone();
            let transport = transport.clone();
            let before_send = before_send.clone();
            let provider_id = provider_id.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let middlewares = middlewares.clone();
            let retry_options = retry_options.clone();
            Box::pin({
                async move {
                    let retry_wrapper_opts = retry_options.clone();
                    let run_once = move || {
                        let req_in = req_in.clone();
                        let url = url.clone();
                        let client = client.clone();
                        let request_tx = request_tx.clone();
                        let response_tx = response_tx.clone();
                        let interceptors = interceptors.clone();
                        let transport = transport.clone();
                        let before_send = before_send.clone();
                        let provider_id = provider_id.clone();
                        let provider_spec = provider_spec.clone();
                        let provider_context = provider_context.clone();
                        let middlewares = middlewares.clone();
                        let retry_options = retry_options.clone();

                        async move {
                            let json_body = build_chat_body(
                                &request_tx,
                                &middlewares,
                                &provider_spec,
                                &provider_context,
                                &before_send,
                                &req_in,
                            )?;

                            let config = crate::execution::executors::common::HttpExecutionConfig {
                                provider_id: provider_id.clone(),
                                http_client: client.clone(),
                                transport: transport.clone(),
                                provider_spec: provider_spec.clone(),
                                provider_context: provider_context.clone(),
                                interceptors: interceptors.clone(),
                                retry_options: retry_options.clone(),
                            };
                            let owned_headers = build_effective_chat_request_headers(
                                &provider_spec,
                                &provider_context,
                                false,
                                &req_in,
                            );
                            let per_request_headers = owned_headers.as_ref();
                            let result = crate::execution::executors::common::execute_json_request(
                                &config,
                                &url,
                                crate::execution::executors::common::HttpBody::Json(json_body),
                                per_request_headers,
                                false,
                            )
                            .await?;
                            let resp = response_tx.transform_chat_response(&result.json)?;
                            Ok(resp)
                        }
                    };

                    if let Some(opts) = retry_wrapper_opts {
                        crate::retry_api::retry_with(run_once, opts).await
                    } else {
                        run_once().await
                    }
                }
            })
        });

        // Build around-style async wrappers in order (first registered becomes outermost)
        let wrapped = self
            .middlewares
            .iter()
            .rev()
            .fold(base, |next, mw| mw.wrap_generate_async(next));

        // Execute wrapped pipeline
        let result = wrapped(req.clone()).await;

        // Emit telemetry events
        match &result {
            Ok(response) => {
                crate::execution::telemetry::chat::span_end_ok(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &span_id,
                    false,
                    response.finish_reason.as_ref(),
                )
                .await;
                crate::execution::telemetry::chat::generation(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &provider_id_for_telemetry,
                    &req.common_params.model,
                    &req,
                    response,
                    start_time,
                )
                .await;
            }
            Err(error) => {
                crate::execution::telemetry::chat::span_end_err(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &span_id,
                    error,
                )
                .await;
            }
        }

        // Apply post-generate processors
        let resp = result?;
        apply_post_generate_chain(&self.middlewares, &req, resp)
    }

    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError> {
        // Initialize telemetry if enabled
        let trace_id = uuid::Uuid::new_v4().to_string();
        let span_id = uuid::Uuid::new_v4().to_string();
        let _start_time = std::time::SystemTime::now();
        let telemetry_config = req.telemetry.clone();

        crate::execution::telemetry::chat::span_start_stream(
            telemetry_config.as_ref(),
            &trace_id,
            &span_id,
            &self.provider_id,
            &req.common_params.model,
        )
        .await;

        let sse_tx_opt = self.stream_transformer.clone();
        let json_tx_opt = self.json_stream_converter.clone();
        if sse_tx_opt.is_none() && json_tx_opt.is_none() {
            return Err(LlmError::UnsupportedOperation(
                "Streaming not supported by this executor".into(),
            ));
        }
        // Apply model-level parameter transforms
        let req = apply_transform_chain(&self.middlewares, req);
        // Try pre-stream short-circuit
        if let Some(decision) = try_pre_stream(&self.middlewares, &req) {
            // Emit telemetry span end for short-circuit
            crate::execution::telemetry::chat::span_end_ok_stream(
                telemetry_config.as_ref(),
                &trace_id,
                &span_id,
                true,
                false,
            )
            .await;
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let http = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let sse_tx = sse_tx_opt.clone();
        let json_tx = json_tx_opt.clone();
        let interceptors = self.policy.interceptors.clone();
        let before_send = self.policy.before_send.clone();
        let url = self
            .provider_spec
            .chat_url(true, &req, &self.provider_context);
        let headers_base = self.provider_spec.build_headers(&self.provider_context)?;
        let disable_compression = self.policy.stream_disable_compression;
        let middlewares = self.middlewares.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let retry_options = self.policy.retry_options.clone();

        // Base async stream builder
        let base: Arc<StreamAsyncFn> = Arc::new(move |req_in: ChatRequest| {
            let provider_id = provider_id.clone();
            let http = http.clone();
            let request_tx = request_tx.clone();
            let sse_tx = sse_tx.clone();
            let json_tx = json_tx.clone();
            let interceptors = interceptors.clone();
            let before_send = before_send.clone();
            let url = url.clone();
            let headers_base = headers_base.clone();
            let middlewares = middlewares.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let retry_options = retry_options.clone();
            Box::pin(async move {
                let transformed = build_chat_body(
                    &request_tx,
                    &middlewares,
                    &provider_spec,
                    &provider_context,
                    &before_send,
                    &req_in,
                )?;

                // Build and send streaming via helpers below (SSE or JSON)

                // Converters are module-scoped; call unified helpers to build the stream
                if let Some(stream_tx) = sse_tx {
                    create_sse_stream_with_middlewares(
                        provider_id.clone(),
                        provider_spec.clone(),
                        provider_context.clone(),
                        url.clone(),
                        http.clone(),
                        headers_base.clone(),
                        transformed.clone(),
                        stream_tx.clone(),
                        interceptors.clone(),
                        middlewares.clone(),
                        req_in.clone(),
                        disable_compression,
                        retry_options.clone(),
                    )
                    .await
                } else if let Some(jsonc) = json_tx {
                    create_json_stream_with_middlewares(
                        provider_id.clone(),
                        provider_spec.clone(),
                        provider_context.clone(),
                        url.clone(),
                        http.clone(),
                        headers_base.clone(),
                        transformed.clone(),
                        jsonc.clone(),
                        interceptors.clone(),
                        middlewares.clone(),
                        req_in.clone(),
                        disable_compression,
                    )
                    .await
                } else {
                    Err(LlmError::UnsupportedOperation(
                        "No stream transformer".into(),
                    ))
                }
            })
        });

        // Wrap with around-style async middlewares in order
        let wrapped = self
            .middlewares
            .iter()
            .rev()
            .fold(base, |next, mw| mw.wrap_stream_async(next));

        let result = wrapped(req.clone()).await;

        // Emit telemetry span end event and wrap stream with telemetry
        if let Some(ref telemetry) = telemetry_config {
            if telemetry.enabled {
                match result {
                    Ok(stream) => {
                        crate::execution::telemetry::chat::span_end_ok_stream(
                            telemetry_config.as_ref(),
                            &trace_id,
                            &span_id,
                            false,
                            true,
                        )
                        .await;

                        // Wrap the stream with telemetry tracking
                        let wrapped_stream = crate::streaming::wrap_stream_with_telemetry(
                            stream,
                            std::sync::Arc::new(telemetry.clone()),
                            trace_id.clone(),
                            provider_id_for_telemetry.clone(),
                            req.common_params.model.clone(),
                            req.messages.clone(),
                        );

                        Ok(wrapped_stream)
                    }
                    Err(error) => {
                        crate::execution::telemetry::chat::span_end_err_stream(
                            telemetry_config.as_ref(),
                            &trace_id,
                            &span_id,
                            &error,
                        )
                        .await;

                        Err(error)
                    }
                }
            } else {
                result
            }
        } else {
            result
        }
    }
}
