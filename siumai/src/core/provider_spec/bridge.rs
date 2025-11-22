//! Bridge helpers between core transformers and aggregator transformers.
//!
//! 该模块负责将 `siumai-core` 中的标准化 transformers
//!（chat / embedding / image / rerank / streaming）桥接到聚合层的
//! transformers 与事件类型上。

use crate::error::LlmError;
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, EmbeddingRequest, ImageGenerationRequest};
use std::collections::HashMap;
use std::sync::Arc;

use super::{ChatTransformers, EmbeddingTransformers, ImageTransformers, RerankTransformers};

/// Bridge core-only embedding transformers into aggregator-level transformers.
///
/// 用于将 `siumai-core` 的 embedding transformers 适配为聚合层的
/// `RequestTransformer` / `ResponseTransformer`。
pub fn bridge_core_embedding_transformers(
    core_request: Arc<dyn siumai_core::execution::embedding::EmbeddingRequestTransformer>,
    core_response: Arc<dyn siumai_core::execution::embedding::EmbeddingResponseTransformer>,
) -> EmbeddingTransformers {
    use siumai_core::execution::embedding::EmbeddingInput;

    struct EmbRequestBridge {
        inner: Arc<dyn siumai_core::execution::embedding::EmbeddingRequestTransformer>,
    }

    impl RequestTransformer for EmbRequestBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat(&self, _req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "Chat is not supported by embedding transformer".to_string(),
            ))
        }

        fn transform_embedding(
            &self,
            req: &EmbeddingRequest,
        ) -> Result<serde_json::Value, LlmError> {
            let encoding_format = req.encoding_format.as_ref().map(|f| match f {
                crate::types::embedding::EmbeddingFormat::Float => "float".to_string(),
                crate::types::embedding::EmbeddingFormat::Base64 => "base64".to_string(),
            });
            let input = EmbeddingInput {
                input: req.input.clone(),
                model: req.model.clone(),
                dimensions: req.dimensions,
                encoding_format,
                user: req.user.clone(),
                title: req.title.clone(),
            };
            self.inner.transform_embedding(&input)
        }
    }

    struct EmbResponseBridge {
        inner: Arc<dyn siumai_core::execution::embedding::EmbeddingResponseTransformer>,
    }

    impl ResponseTransformer for EmbResponseBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_embedding_response(
            &self,
            raw: &serde_json::Value,
        ) -> Result<crate::types::EmbeddingResponse, LlmError> {
            let res = self.inner.transform_embedding_response(raw)?;
            let mut out = crate::types::EmbeddingResponse::new(res.embeddings, res.model);
            if let Some(u) = res.usage {
                out = out.with_usage(crate::types::embedding::EmbeddingUsage::new(
                    u.prompt_tokens,
                    u.total_tokens,
                ));
            }
            Ok(out)
        }
    }

    EmbeddingTransformers {
        request: Arc::new(EmbRequestBridge {
            inner: core_request,
        }),
        response: Arc::new(EmbResponseBridge {
            inner: core_response,
        }),
    }
}

/// Bridge core-only image transformers into aggregator-level transformers.
///
/// 用于将 `siumai-core` 的 image transformers 适配为聚合层的
/// `RequestTransformer` / `ResponseTransformer`。
pub fn bridge_core_image_transformers(
    core_request: Arc<dyn siumai_core::execution::image::ImageRequestTransformer>,
    core_response: Arc<dyn siumai_core::execution::image::ImageResponseTransformer>,
) -> ImageTransformers {
    struct ImageRequestBridge {
        inner: Arc<dyn siumai_core::execution::image::ImageRequestTransformer>,
    }

    impl crate::execution::transformers::request::RequestTransformer for ImageRequestBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat(
            &self,
            _req: &crate::types::ChatRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "Chat is not supported by image transformer".to_string(),
            ))
        }

        fn transform_image(
            &self,
            req: &ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            self.inner.transform_image(req)
        }

        fn transform_image_edit(
            &self,
            req: &crate::types::ImageEditRequest,
        ) -> Result<crate::execution::transformers::request::ImageHttpBody, LlmError> {
            match self.inner.transform_image_edit(req)? {
                siumai_core::execution::image::ImageHttpBody::Json(v) => {
                    Ok(crate::execution::transformers::request::ImageHttpBody::Json(v))
                }
                siumai_core::execution::image::ImageHttpBody::Multipart(f) => {
                    Ok(crate::execution::transformers::request::ImageHttpBody::Multipart(f))
                }
            }
        }

        fn transform_image_variation(
            &self,
            req: &crate::types::ImageVariationRequest,
        ) -> Result<crate::execution::transformers::request::ImageHttpBody, LlmError> {
            match self.inner.transform_image_variation(req)? {
                siumai_core::execution::image::ImageHttpBody::Json(v) => {
                    Ok(crate::execution::transformers::request::ImageHttpBody::Json(v))
                }
                siumai_core::execution::image::ImageHttpBody::Multipart(f) => {
                    Ok(crate::execution::transformers::request::ImageHttpBody::Multipart(f))
                }
            }
        }
    }

    struct ImageResponseBridge {
        inner: Arc<dyn siumai_core::execution::image::ImageResponseTransformer>,
    }

    impl crate::execution::transformers::response::ResponseTransformer for ImageResponseBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_image_response(
            &self,
            raw: &serde_json::Value,
        ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
            self.inner.transform_image_response(raw)
        }
    }

    ImageTransformers {
        request: Arc::new(ImageRequestBridge {
            inner: core_request,
        }),
        response: Arc::new(ImageResponseBridge {
            inner: core_response,
        }),
    }
}

/// Bridge core-only rerank transformers into aggregator-level transformers.
///
/// 用于将 `siumai-core` 的 rerank transformers 适配为聚合层的
/// rerank transformers。
pub fn bridge_core_rerank_transformers(
    core_request: Arc<dyn siumai_core::execution::rerank::RerankRequestTransformer>,
    core_response: Arc<dyn siumai_core::execution::rerank::RerankResponseTransformer>,
) -> RerankTransformers {
    struct ReqBridge {
        inner: Arc<dyn siumai_core::execution::rerank::RerankRequestTransformer>,
    }

    impl crate::execution::transformers::rerank_request::RerankRequestTransformer for ReqBridge {
        fn transform(
            &self,
            req: &crate::types::RerankRequest,
        ) -> Result<serde_json::Value, LlmError> {
            let input = siumai_core::execution::rerank::RerankInput {
                model: Some(req.model.clone()),
                query: req.query.clone(),
                documents: req.documents.clone(),
                top_n: req.top_n,
                return_documents: req.return_documents,
                extra: Default::default(),
            };
            self.inner.transform(&input)
        }
    }

    struct RespBridge {
        inner: Arc<dyn siumai_core::execution::rerank::RerankResponseTransformer>,
    }

    impl crate::execution::transformers::rerank_response::RerankResponseTransformer for RespBridge {
        fn transform(
            &self,
            raw: serde_json::Value,
        ) -> Result<crate::types::RerankResponse, LlmError> {
            let out = self.inner.transform_response(&raw)?;
            let results = out
                .results
                .into_iter()
                .map(|i| crate::types::RerankResult {
                    document: i.document.map(|text| crate::types::RerankDocument { text }),
                    index: i.index,
                    relevance_score: i.relevance_score,
                })
                .collect::<Vec<_>>();
            Ok(crate::types::RerankResponse {
                id: out.id.unwrap_or_default(),
                results,
                tokens: crate::types::RerankTokenUsage {
                    input_tokens: out.input_tokens,
                    output_tokens: out.output_tokens,
                },
            })
        }
    }

    RerankTransformers {
        request: Arc::new(ReqBridge {
            inner: core_request,
        }),
        response: Arc::new(RespBridge {
            inner: core_response,
        }),
    }
}

/// Bridge a core-level chat transformers bundle into aggregator-level transformers.
///
/// 将核心 Chat transformers（request / response / stream）桥接到聚合层，
/// 通过 `to_core_input` 与 `map_stream_event` 完成请求/流事件的映射。
pub fn bridge_core_chat_transformers<F, G>(
    core_txs: siumai_core::provider_spec::CoreChatTransformers,
    to_core_input: F,
    map_stream_event: G,
) -> ChatTransformers
where
    F: Fn(&ChatRequest) -> siumai_core::execution::chat::ChatInput + Send + Sync + 'static,
    G: Fn(
            siumai_core::execution::streaming::ChatStreamEventCore,
        ) -> crate::streaming::ChatStreamEvent
        + Send
        + Sync
        + 'static,
{
    use siumai_core::execution::chat::{
        ChatInput, ChatRequestTransformer, ChatResponseTransformer,
    };

    struct ChatRequestBridge<F> {
        inner: Arc<dyn ChatRequestTransformer>,
        to_core: F,
    }

    impl<F> RequestTransformer for ChatRequestBridge<F>
    where
        F: Fn(&ChatRequest) -> ChatInput + Send + Sync + 'static,
    {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
            let input = (self.to_core)(req);
            self.inner.transform_chat(&input)
        }
    }

    struct ChatResponseBridge {
        inner: Arc<dyn ChatResponseTransformer>,
    }

    impl ResponseTransformer for ChatResponseBridge {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn transform_chat_response(
            &self,
            raw: &serde_json::Value,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            use crate::types::{ContentPart, FinishReason, MessageContent, Usage};
            use siumai_core::types::FinishReasonCore;

            let provider_id = self.inner.provider_id().to_string();
            let core_res = self.inner.transform_chat_response(raw)?;

            // Preserve thinking content for potential provider-specific metadata.
            let mut thinking_text: Option<String> = None;

            // Prefer the structured parsed_content view when available, so that
            // higher layers can get text + tool calls + thinking without
            // re-parsing provider JSON.
            let content = if let Some(parsed) = &core_res.parsed_content {
                let mut parts = Vec::new();

                if !parsed.text.is_empty() {
                    parts.push(ContentPart::text(&parsed.text));
                }

                for tc in &parsed.tool_calls {
                    parts.push(ContentPart::tool_call(
                        tc.id.clone().unwrap_or_else(|| "".to_string()),
                        tc.name.clone(),
                        tc.arguments.clone(),
                        None,
                    ));
                }

                if let Some(thinking) = &parsed.thinking {
                    if !thinking.is_empty() {
                        thinking_text = Some(thinking.clone());
                        parts.push(ContentPart::reasoning(thinking));
                    }
                }

                if parts.is_empty() {
                    MessageContent::Text(core_res.content)
                } else if parts.len() == 1 && parts[0].is_text() {
                    // Keep simple Text when only plain text is present.
                    MessageContent::Text(parsed.text.clone())
                } else {
                    MessageContent::MultiModal(parts)
                }
            } else {
                MessageContent::Text(core_res.content)
            };

            let usage = core_res
                .usage
                .map(|u| Usage::new(u.prompt_tokens, u.completion_tokens));

            let finish_reason = core_res.finish_reason.map(|fr| match fr {
                FinishReasonCore::Stop => FinishReason::Stop,
                FinishReasonCore::Length => FinishReason::Length,
                FinishReasonCore::ContentFilter => FinishReason::ContentFilter,
                FinishReasonCore::ToolCalls => FinishReason::ToolCalls,
                FinishReasonCore::Other(s) => FinishReason::Other(s),
            });

            let mut response = crate::types::ChatResponse {
                id: None,
                model: None,
                content,
                usage,
                finish_reason,
                system_fingerprint: None,
                service_tier: None,
                audio: None,
                warnings: None,
                provider_metadata: None,
            };

            // For Anthropic, expose thinking content via provider_metadata["anthropic"]
            // so callers can use response.anthropic_metadata().thinking even when
            // going through the std/core pipeline.
            if provider_id == "anthropic" {
                if let Some(thinking) = thinking_text {
                    if !thinking.is_empty() {
                        let mut inner: HashMap<String, serde_json::Value> = HashMap::new();
                        inner.insert("thinking".to_string(), serde_json::Value::String(thinking));
                        let mut outer: HashMap<String, HashMap<String, serde_json::Value>> =
                            HashMap::new();
                        outer.insert("anthropic".to_string(), inner);
                        response.provider_metadata = Some(outer);
                    }
                }
            }

            Ok(response)
        }
    }

    struct StreamBridge<G> {
        inner: Arc<dyn siumai_core::execution::streaming::ChatStreamEventConverterCore>,
        map_evt: G,
    }

    impl<G> StreamChunkTransformer for StreamBridge<G>
    where
        G: Fn(
                siumai_core::execution::streaming::ChatStreamEventCore,
            ) -> crate::streaming::ChatStreamEvent
            + Send
            + Sync
            + 'static,
    {
        fn provider_id(&self) -> &str {
            self.inner.provider_id()
        }

        fn convert_event(
            &self,
            event: eventsource_stream::Event,
        ) -> crate::execution::transformers::stream::StreamEventFuture<'_> {
            let inner = Arc::clone(&self.inner);
            let map_evt = &self.map_evt;
            Box::pin(async move {
                inner
                    .convert_event(event)
                    .into_iter()
                    .map(|res| res.map(|e| map_evt(e)))
                    .collect()
            })
        }

        fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
            self.inner
                .handle_stream_end()
                .map(|res| res.map(|e| (self.map_evt)(e)))
        }
    }

    let request = Arc::new(ChatRequestBridge {
        inner: core_txs.request,
        to_core: to_core_input,
    });

    let response = Arc::new(ChatResponseBridge {
        inner: core_txs.response,
    });

    let stream = core_txs.stream.map(|inner| {
        Arc::new(StreamBridge {
            inner,
            map_evt: map_stream_event,
        }) as Arc<dyn StreamChunkTransformer>
    });

    ChatTransformers {
        request,
        response,
        stream,
        json: None,
    }
}

/// Helper: map a core-level stream event into the aggregator's
/// `ChatStreamEvent`, injecting the given provider id into the
/// StreamStart metadata.
///
/// 所有复用 core streaming 事件模型的 provider 共享此映射逻辑。
pub fn map_core_stream_event_with_provider(
    provider: &str,
    evt: siumai_core::execution::streaming::ChatStreamEventCore,
) -> crate::streaming::ChatStreamEvent {
    use crate::streaming::ChatStreamEvent;
    use crate::types::{ChatResponse, FinishReason};
    use siumai_core::execution::streaming::ChatStreamEventCore;
    use siumai_core::types::FinishReasonCore;

    match evt {
        ChatStreamEventCore::ContentDelta { delta, index } => {
            ChatStreamEvent::ContentDelta { delta, index }
        }
        ChatStreamEventCore::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        } => ChatStreamEvent::ToolCallDelta {
            id: id.unwrap_or_default(),
            function_name,
            arguments_delta,
            index,
        },
        ChatStreamEventCore::ThinkingDelta { delta } => ChatStreamEvent::ThinkingDelta { delta },
        ChatStreamEventCore::UsageUpdate {
            prompt_tokens,
            completion_tokens,
            ..
        } => {
            let usage = crate::types::Usage::new(prompt_tokens, completion_tokens);
            ChatStreamEvent::UsageUpdate { usage }
        }
        ChatStreamEventCore::StreamStart {} => ChatStreamEvent::StreamStart {
            metadata: crate::types::ResponseMetadata {
                id: None,
                model: None,
                created: None,
                provider: provider.to_string(),
                request_id: None,
            },
        },
        ChatStreamEventCore::StreamEnd { finish_reason } => {
            let mapped = match finish_reason {
                Some(FinishReasonCore::Stop) => FinishReason::Stop,
                Some(FinishReasonCore::Length) => FinishReason::Length,
                Some(FinishReasonCore::ContentFilter) => FinishReason::ContentFilter,
                Some(FinishReasonCore::ToolCalls) => FinishReason::ToolCalls,
                Some(FinishReasonCore::Other(s)) => FinishReason::Other(s),
                None => FinishReason::Unknown,
            };
            let response = ChatResponse::empty_with_finish_reason(mapped);
            ChatStreamEvent::StreamEnd { response }
        }
        ChatStreamEventCore::Custom { event_type, data } => {
            ChatStreamEvent::Custom { event_type, data }
        }
        ChatStreamEventCore::Error { error } => ChatStreamEvent::Error { error },
    }
}

/// Anthropic 风格流事件映射助手。
///
/// 兼容旧代码，内部委托给 [`map_core_stream_event_with_provider`]。
pub fn anthropic_like_map_core_stream_event(
    provider: &'static str,
    evt: siumai_core::execution::streaming::ChatStreamEventCore,
) -> crate::streaming::ChatStreamEvent {
    map_core_stream_event_with_provider(provider, evt)
}
