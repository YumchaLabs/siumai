use crate::core::{
    ChatTransformers, EmbeddingTransformers, ImageTransformers, ProviderContext, ProviderSpec,
};
use crate::error::LlmError;
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
            .with_custom_feature("image_generation", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        // Delegate to standard headers with adapter hook capability
        let spec = crate::standards::gemini::GeminiChatStandard::new().create_spec("gemini");
        spec.build_headers(ctx)
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        // Delegate to standard spec for URL decision
        let spec = crate::standards::gemini::GeminiChatStandard::new().create_spec("gemini");
        spec.chat_url(stream, req, ctx)
    }

    fn choose_chat_transformers(
        &self,
        _req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        crate::standards::gemini::GeminiChatStandard::new().create_transformers(&ctx.provider_id)
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
        let spec = crate::standards::gemini::GeminiEmbeddingStandard::new().create_spec("gemini");
        spec.embedding_url(req, ctx)
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        crate::standards::gemini::GeminiEmbeddingStandard::new().create_transformers()
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        let spec = crate::standards::gemini::GeminiImageStandard::new().create_spec("gemini");
        spec.image_url(req, ctx)
    }

    fn choose_image_transformers(
        &self,
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        crate::standards::gemini::GeminiImageStandard::new().create_transformers()
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
