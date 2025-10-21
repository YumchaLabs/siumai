use crate::error::LlmError;
use crate::provider_core::{
    ChatTransformers, EmbeddingTransformers, ProviderContext, ProviderSpec,
};
// (no additional trait imports required)
use crate::traits::ProviderCapabilities;
use crate::types::{ChatRequest, ProviderOptions};
use crate::utils::http_headers::{ProviderHeaders, inject_tracing_headers};
use reqwest::header::HeaderMap;
use std::sync::Arc;

/// OpenAI ProviderSpec implementation
#[derive(Clone, Copy, Default)]
pub struct OpenAiSpec;

impl OpenAiSpec {
    fn use_responses_api(&self, req: &ChatRequest, _ctx: &ProviderContext) -> bool {
        // Check if Responses API is configured in provider_options
        if let ProviderOptions::OpenAi(ref options) = req.provider_options {
            options.responses_api.is_some()
        } else {
            false
        }
    }
}

impl ProviderSpec for OpenAiSpec {
    fn id(&self) -> &'static str {
        "openai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_ref()
            .ok_or_else(|| LlmError::MissingApiKey("OpenAI API key not provided".into()))?;
        let mut headers = ProviderHeaders::openai(
            api_key,
            ctx.organization.as_deref(),
            ctx.project.as_deref(),
            &ctx.http_extra_headers,
        )?;
        inject_tracing_headers(&mut headers);
        Ok(headers)
    }

    fn chat_url(&self, _stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let use_responses = self.use_responses_api(req, ctx);
        let suffix = if use_responses {
            "/responses"
        } else {
            "/chat/completions"
        };
        format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix)
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        if self.use_responses_api(req, ctx) {
            // Responses API transformers
            let req_tx = crate::providers::openai::transformers::OpenAiResponsesRequestTransformer;
            let resp_tx =
                crate::providers::openai::transformers::OpenAiResponsesResponseTransformer;
            let converter =
                crate::providers::openai::responses::OpenAiResponsesEventConverter::new();
            let stream_tx =
                crate::providers::openai::transformers::OpenAiResponsesStreamChunkTransformer {
                    provider_id: "openai_responses".to_string(),
                    inner: converter,
                };
            ChatTransformers {
                request: Arc::new(req_tx),
                response: Arc::new(resp_tx),
                stream: Some(Arc::new(stream_tx)),
                json: None,
            }
        } else {
            let req_tx = crate::providers::openai::transformers::OpenAiRequestTransformer;
            let resp_tx = crate::providers::openai::transformers::OpenAiResponseTransformer;
            // Provide a stream transformer via OpenAI-compatible streaming adapter
            let adapter: std::sync::Arc<
                dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
            > = std::sync::Arc::new(crate::providers::openai::adapter::OpenAiStandardAdapter {
                base_url: ctx.base_url.clone(),
            });
            let compat_cfg =
                crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    ctx.api_key.as_deref().unwrap_or_default(),
                    &ctx.base_url,
                    adapter.clone(),
                )
                .with_model(&req.common_params.model);
            let inner =
                crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
                    compat_cfg, adapter,
                );
            let stream_tx = crate::providers::openai::transformers::OpenAiStreamChunkTransformer {
                provider_id: "openai".to_string(),
                inner,
            };
            ChatTransformers {
                request: Arc::new(req_tx),
                response: Arc::new(resp_tx),
                stream: Some(Arc::new(stream_tx)),
                json: None,
            }
        }
    }

    fn chat_before_send(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> Option<crate::executors::BeforeSendHook> {
        // 1. First check for CustomProviderOptions (using default implementation)
        if let Some(hook) = Self::default_custom_options_hook(self.id(), req) {
            return Some(hook);
        }

        // 2. Handle OpenAI-specific options (built_in_tools, responses_api, etc.)
        // Extract options from provider_options
        let (builtins, responses_api_config, reasoning_effort, service_tier) =
            if let ProviderOptions::OpenAi(ref options) = req.provider_options {
                let builtins = if !options.built_in_tools.is_empty() {
                    // 🎯 Call to_json() on each tool to get proper JSON representation
                    let tools_json: Vec<serde_json::Value> =
                        options.built_in_tools.iter().map(|t| t.to_json()).collect();
                    Some(serde_json::Value::Array(tools_json))
                } else {
                    None
                };
                (
                    builtins,
                    options.responses_api.clone(),
                    options.reasoning_effort.clone(),
                    options.service_tier.clone(),
                )
            } else {
                return None;
            };

        let builtins = builtins.unwrap_or(serde_json::Value::Array(vec![]));

        // Extract all Responses API config fields
        let prev_id = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.previous_response_id.clone());
        let response_format = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.response_format.clone())
            .and_then(|fmt| serde_json::to_value(fmt).ok());
        let background = responses_api_config.as_ref().and_then(|cfg| cfg.background);
        let include = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.include.clone());
        let instructions = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.instructions.clone());
        let max_tool_calls = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.max_tool_calls);
        let store = responses_api_config.as_ref().and_then(|cfg| cfg.store);
        let truncation = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.truncation.clone());
        let text_verbosity = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.text_verbosity.clone());
        let metadata = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.metadata.clone());
        let parallel_tool_calls = responses_api_config
            .as_ref()
            .and_then(|cfg| cfg.parallel_tool_calls);

        // Check if we need to inject anything
        let has_builtins = matches!(&builtins, serde_json::Value::Array(arr) if !arr.is_empty());
        let has_prev_id = prev_id.is_some();
        let has_response_format = response_format.is_some();
        let has_reasoning_effort = reasoning_effort.is_some();
        let has_service_tier = service_tier.is_some();
        let has_background = background.is_some();
        let has_include = include.is_some();
        let has_instructions = instructions.is_some();
        let has_max_tool_calls = max_tool_calls.is_some();
        let has_store = store.is_some();
        let has_truncation = truncation.is_some();
        let has_text_verbosity = text_verbosity.is_some();
        let has_metadata = metadata.is_some();
        let has_parallel_tool_calls = parallel_tool_calls.is_some();

        if !has_builtins
            && !has_prev_id
            && !has_response_format
            && !has_reasoning_effort
            && !has_service_tier
            && !has_background
            && !has_include
            && !has_instructions
            && !has_max_tool_calls
            && !has_store
            && !has_truncation
            && !has_text_verbosity
            && !has_metadata
            && !has_parallel_tool_calls
        {
            return None;
        }

        let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();

            // 🎯 Inject built-in tools (merge with existing tools)
            if let serde_json::Value::Array(bi) = &builtins {
                if !bi.is_empty() {
                    let mut arr = out
                        .get("tools")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();
                    for t in bi.iter() {
                        arr.push(t.clone());
                    }
                    // De-dup logic:
                    // - Keep all "function" tools (they have unique names)
                    // - For built-in tools (file_search, web_search, computer_use):
                    //   - Keep all if they have different configurations
                    //   - Only de-dup exact duplicates
                    let mut dedup = Vec::new();
                    let mut seen_builtins: Vec<serde_json::Value> = Vec::new();

                    for item in arr.into_iter() {
                        let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

                        if typ == "function" {
                            // Always keep function tools (they have unique names)
                            dedup.push(item);
                        } else {
                            // For built-in tools, check if we've seen an exact duplicate
                            if !seen_builtins.contains(&item) {
                                seen_builtins.push(item.clone());
                                dedup.push(item);
                            }
                        }
                    }
                    out["tools"] = serde_json::Value::Array(dedup);
                }
            }

            // 🎯 Inject Responses API fields
            if let Some(pid) = &prev_id {
                out["previous_response_id"] = serde_json::Value::String(pid.clone());
            }
            if let Some(fmt) = &response_format {
                out["response_format"] = fmt.clone();
            }
            if let Some(bg) = background {
                out["background"] = serde_json::Value::Bool(bg);
            }
            if let Some(ref inc) = include {
                if let Ok(val) = serde_json::to_value(inc) {
                    out["include"] = val;
                }
            }
            if let Some(ref instr) = instructions {
                out["instructions"] = serde_json::Value::String(instr.clone());
            }
            if let Some(mtc) = max_tool_calls {
                out["max_tool_calls"] = serde_json::Value::Number(mtc.into());
            }
            if let Some(st) = store {
                out["store"] = serde_json::Value::Bool(st);
            }
            if let Some(ref trunc) = truncation {
                if let Ok(val) = serde_json::to_value(trunc) {
                    out["truncation"] = val;
                }
            }
            if let Some(ref verb) = text_verbosity {
                if let Ok(val) = serde_json::to_value(verb) {
                    // text_verbosity should be nested under "text.verbosity" in Responses API
                    if let Some(text_obj) = out.get_mut("text") {
                        if let Some(text_map) = text_obj.as_object_mut() {
                            text_map.insert("verbosity".to_string(), val);
                        }
                    } else {
                        // If no "text" field exists, create one with verbosity
                        out["text"] = serde_json::json!({
                            "verbosity": val
                        });
                    }
                }
            }
            if let Some(ref meta) = metadata {
                if let Ok(val) = serde_json::to_value(meta) {
                    out["metadata"] = val;
                }
            }
            if let Some(ptc) = parallel_tool_calls {
                out["parallel_tool_calls"] = serde_json::Value::Bool(ptc);
            }

            // 🎯 Inject reasoning_effort
            if let Some(ref effort) = reasoning_effort {
                if let Ok(val) = serde_json::to_value(effort) {
                    out["reasoning_effort"] = val;
                }
            }

            // 🎯 Inject service_tier
            if let Some(ref tier) = service_tier {
                if let Ok(val) = serde_json::to_value(tier) {
                    out["service_tier"] = val;
                }
            }

            Ok(out)
        };
        Some(Arc::new(hook))
    }

    fn choose_embedding_transformers(
        &self,
        _req: &crate::types::EmbeddingRequest,
        _ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        let req_tx = crate::providers::openai::transformers::OpenAiRequestTransformer;
        let resp_tx = crate::providers::openai::transformers::OpenAiResponseTransformer;
        EmbeddingTransformers {
            request: Arc::new(req_tx),
            response: Arc::new(resp_tx),
        }
    }
}
