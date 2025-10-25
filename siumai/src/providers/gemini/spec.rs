use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// Gemini ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct GeminiSpec;

impl ProviderSpec for GeminiSpec {
    fn id(&self) -> &'static str {
        "gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_vision()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // Gemini header rules are encapsulated in ProviderHeaders::gemini:
        // - If Authorization exists (e.g., Vertex token), do not inject x-goog-api-key.
        // - Otherwise use x-goog-api-key when api_key is provided.
        let api_key = ctx.api_key.as_deref().unwrap_or("");
        ProviderHeaders::gemini(api_key, &ctx.http_extra_headers)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = &req.common_params.model;
        if stream {
            format!("{}/models/{}:streamGenerateContent?alt=sse", base, model)
        } else {
            format!("{}/models/{}:generateContent", base, model)
        }
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let req_tx = crate::providers::gemini::transformers::GeminiRequestTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        let resp_tx = crate::providers::gemini::transformers::GeminiResponseTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        let inner = crate::providers::gemini::streaming::GeminiEventConverter::new(
            crate::providers::gemini::types::GeminiConfig::default(),
        );
        let stream_tx = crate::providers::gemini::transformers::GeminiStreamChunkTransformer {
            provider_id: "gemini".to_string(),
            inner,
        };
        ChatTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
            stream: Some(Arc::new(stream_tx)),
            json: None,
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = crate::core::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle Gemini-specific options (code_execution, search_grounding)
        // 🎯 Extract Gemini-specific options from provider_options
        let (code_execution, search_grounding) =
            if let ProviderOptions::Gemini(ref options) = req.provider_options {
                (
                    options.code_execution.clone(),
                    options.search_grounding.clone(),
                )
            } else {
                return None;
            };

        // Check if we have anything to inject
        if code_execution.is_none() && search_grounding.is_none() {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject code execution tool
            // According to Gemini API, code execution is enabled via tools array
            if let Some(ref code_exec) = code_execution {
                if code_exec.enabled {
                    let mut tools = out
                        .get("tools")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();

                    // Add code execution tool
                    tools.push(serde_json::json!({
                        "code_execution": {}
                    }));

                    out["tools"] = serde_json::Value::Array(tools);
                }
            }

            // 🎯 Inject search grounding (Google Search)
            // According to Gemini API, search grounding is enabled via tools array
            if let Some(ref search) = search_grounding {
                if search.enabled {
                    let mut tools = out
                        .get("tools")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();

                    let mut google_search_tool = serde_json::json!({
                        "google_search": {}
                    });

                    // Add dynamic retrieval config if specified
                    if let Some(ref dynamic_config) = search.dynamic_retrieval_config {
                        if let Ok(config_json) = serde_json::to_value(dynamic_config) {
                            google_search_tool["google_search"]["dynamic_retrieval_config"] =
                                config_json;
                        }
                    }

                    tools.push(google_search_tool);
                    out["tools"] = serde_json::Value::Array(tools);
                }
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        let base = ctx.base_url.trim_end_matches('/');
        let model = req.model.as_deref().unwrap_or("");
        if req.input.len() == 1 {
            format!("{}/models/{}:embedContent", base, model)
        } else {
            format!("{}/models/{}:batchEmbedContents", base, model)
        }
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        let req_tx = crate::providers::gemini::transformers::GeminiRequestTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        let resp_tx = crate::providers::gemini::transformers::GeminiResponseTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        EmbeddingTransformers {
            request: std::sync::Arc::new(req_tx),
            response: std::sync::Arc::new(resp_tx),
        }
    }

    fn image_url(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        // Gemini image generation uses generateContent
        let base = ctx.base_url.trim_end_matches('/');
        // Model is supplied in request common params; default to empty handled by executor
        format!("{}/models/{}:generateContent", base, "")
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        let req_tx = crate::providers::gemini::transformers::GeminiRequestTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        let resp_tx = crate::providers::gemini::transformers::GeminiResponseTransformer {
            config: crate::providers::gemini::types::GeminiConfig::default(),
        };
        ImageTransformers {
            request: std::sync::Arc::new(req_tx),
            response: std::sync::Arc::new(resp_tx),
        }
    }

    fn files_base_url(&self, ctx: &ProviderContext) -> String {
        ctx.base_url.trim_end_matches('/').to_string()
    }

    fn choose_files_transformer(&self, _ctx: &ProviderContext) -> crate::core::FilesTransformer {
        crate::core::FilesTransformer {
            transformer: Arc::new(
                crate::providers::gemini::transformers::GeminiFilesTransformer {
                    config: crate::providers::gemini::types::GeminiConfig::default(),
                },
            ),
        }
    }
}
