//! Response transformers for OpenAI (Chat/Embedding/Image) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::response::ResponseTransformer;
#[cfg(feature = "std-openai-external")]
use crate::std_openai::openai::responses::{OpenAiResponsesStandard, parse_responses_output};
use crate::types::{
    ChatResponse, EmbeddingResponse, EmbeddingUsage, GeneratedImage, ImageGenerationResponse,
};

#[derive(Clone)]
pub struct OpenAiResponseTransformer;

impl ResponseTransformer for OpenAiResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        #[cfg(all(feature = "openai-compatible", feature = "std-openai-external"))]
        {
            let model = raw
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let adapter: std::sync::Arc<
                dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
            > = std::sync::Arc::new(crate::providers::openai::adapter::OpenAiStandardAdapter {
                base_url: String::new(),
            });
            let cfg =
                crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    "",
                    "",
                    adapter.clone(),
                )
                .with_model(&model);
            let compat =
                crate::providers::openai_compatible::transformers::CompatResponseTransformer {
                    config: cfg,
                    adapter,
                };
            return compat.transform_chat_response(raw);
        }

        #[cfg(not(all(feature = "openai-compatible", feature = "std-openai-external")))]
        {
            // Fallback: minimal native mapping without compat dependency
            #[derive(serde::Deserialize)]
            struct OpenAiMessage {
                role: String,
                content: Option<serde_json::Value>,
            }
            #[derive(serde::Deserialize)]
            struct OpenAiChoice {
                message: OpenAiMessage,
                finish_reason: Option<String>,
            }
            #[derive(serde::Deserialize)]
            struct OpenAiUsage {
                prompt_tokens: Option<u32>,
                completion_tokens: Option<u32>,
                total_tokens: Option<u32>,
            }
            #[derive(serde::Deserialize)]
            struct Root {
                id: Option<String>,
                model: Option<String>,
                choices: Vec<OpenAiChoice>,
                usage: Option<OpenAiUsage>,
            }

            let r: Root = serde_json::from_value(raw.clone())
                .map_err(|e| LlmError::ParseError(format!("Invalid OpenAI chat response: {e}")))?;

            let mut parts = Vec::new();
            if let Some(first) = r.choices.first() {
                if let Some(c) = &first.message.content {
                    match c {
                        serde_json::Value::String(s) => {
                            parts.push(crate::types::ContentPart::text(s))
                        }
                        serde_json::Value::Array(arr) => {
                            for el in arr {
                                if let Some(text) = el.get("text").and_then(|v| v.as_str()) {
                                    parts.push(crate::types::ContentPart::text(text));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut resp = ChatResponse::new(crate::types::MessageContent::MultiModal(parts));
            resp.id = r.id;
            resp.model = r.model;
            if let Some(u) = r.usage {
                resp.usage = Some(crate::types::Usage::new(
                    u.prompt_tokens.unwrap_or(0),
                    u.completion_tokens.unwrap_or(0),
                ));
            }
            resp.finish_reason = crate::providers::openai::utils::parse_finish_reason(
                r.choices.first().and_then(|c| c.finish_reason.as_deref()),
            );
            Ok(resp)
        }
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingObject {
            embedding: Vec<f32>,
            index: usize,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingUsage {
            prompt_tokens: u32,
            total_tokens: u32,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingResponse {
            data: Vec<OpenAiEmbeddingObject>,
            model: String,
            usage: OpenAiEmbeddingUsage,
        }

        let mut r: OpenAiEmbeddingResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid OpenAI embedding response: {e}")))?;
        r.data.sort_by_key(|o| o.index);
        let vectors = r.data.into_iter().map(|o| o.embedding).collect();
        let mut resp = EmbeddingResponse::new(vectors, r.model);
        resp.usage = Some(EmbeddingUsage::new(
            r.usage.prompt_tokens,
            r.usage.total_tokens,
        ));
        Ok(resp)
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiImageData {
            url: Option<String>,
            b64_json: Option<String>,
            revised_prompt: Option<String>,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiImageResponse {
            created: u64,
            data: Vec<OpenAiImageData>,
        }

        let r: OpenAiImageResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid OpenAI image response: {e}")))?;
        let images: Vec<GeneratedImage> = r
            .data
            .into_iter()
            .map(|img| GeneratedImage {
                url: img.url,
                b64_json: img.b64_json,
                format: None,
                width: None,
                height: None,
                revised_prompt: img.revised_prompt,
                metadata: std::collections::HashMap::new(),
            })
            .collect();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("created".to_string(), serde_json::json!(r.created));
        Ok(ImageGenerationResponse { images, metadata })
    }
}

/// Extract thinking content from multiple possible field names with priority order
/// Priority order: reasoning_content > thinking > reasoning
pub fn extract_thinking_from_multiple_fields(value: &serde_json::Value) -> Option<String> {
    let field_names = ["reasoning_content", "thinking", "reasoning"];
    for field in field_names {
        if let Some(s) = value.get(field).and_then(|v| v.as_str()) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Response transformer for OpenAI Responses API
#[derive(Clone)]
pub struct OpenAiResponsesResponseTransformer;

impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use crate::types::{ContentPart, FinishReason, MessageContent, Usage};
        #[cfg(feature = "std-openai-external")]
        let (root, core_usage, core_finish) = {
            let standard = OpenAiResponsesStandard::new();
            let tx = standard.create_response_transformer("openai_responses");
            match tx.transform_responses_response(raw) {
                Ok(res) => (res.output, res.usage, res.finish_reason),
                Err(_) => (
                    raw.get("response").cloned().unwrap_or_else(|| raw.clone()),
                    None,
                    None,
                ),
            }
        };

        #[cfg(not(feature = "std-openai-external"))]
        let (root, core_usage, core_finish) = (
            raw.get("response").cloned().unwrap_or_else(|| raw.clone()),
            None,
            None,
        );

        let root_ref = &root;

        // Build content parts (text + tool calls)
        let (text_content, content_parts) = {
            #[cfg(feature = "std-openai-external")]
            {
                let parsed = parse_responses_output(root_ref);
                let mut parts = Vec::new();
                if !parsed.text.is_empty() {
                    parts.push(ContentPart::text(&parsed.text));
                }
                for tc in parsed.tool_calls {
                    parts.push(ContentPart::tool_call(tc.id, tc.name, tc.arguments, None));
                }
                (parsed.text, parts)
            }
            #[cfg(not(feature = "std-openai-external"))]
            {
                let mut parts = Vec::new();
                let mut text = String::new();
                if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
                    for item in output {
                        if let Some(content_parts_arr) =
                            item.get("content").and_then(|c| c.as_array())
                        {
                            for p in content_parts_arr {
                                if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                }
                            }
                        }
                    }
                }
                if !text.is_empty() {
                    parts.push(ContentPart::text(&text));
                }
                if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
                    for item in output {
                        if let Some(calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                            for call in calls {
                                let id = call
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let (name, arguments) = if let Some(f) = call.get("function") {
                                    (
                                        f.get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        f.get("arguments")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                } else {
                                    (
                                        call.get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        call.get("arguments")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                    )
                                };
                                if !name.is_empty() {
                                    let args_value = serde_json::from_str(&arguments)
                                        .unwrap_or(serde_json::Value::String(arguments));
                                    parts.push(ContentPart::tool_call(id, name, args_value, None));
                                }
                            }
                        }
                    }
                }
                (text, parts)
            }
        };

        // Usage
        fn parse_usage_from_root(root: &serde_json::Value) -> Option<Usage> {
            let u = root.get("usage")?;

            let prompt_tokens = u
                .get("input_tokens")
                .or_else(|| u.get("prompt_tokens"))
                .or_else(|| u.get("inputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let completion_tokens = u
                .get("output_tokens")
                .or_else(|| u.get("completion_tokens"))
                .or_else(|| u.get("outputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let total_tokens = u
                .get("total_tokens")
                .or_else(|| u.get("totalTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let reasoning_tokens = u
                .get("reasoning_tokens")
                .or_else(|| u.get("reasoningTokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

            let mut builder = Usage::builder()
                .prompt_tokens(prompt_tokens)
                .completion_tokens(completion_tokens)
                .total_tokens(total_tokens);

            if let Some(reasoning) = reasoning_tokens {
                builder = builder.with_reasoning_tokens(reasoning);
            }

            Some(builder.build())
        }

        let usage = {
            #[cfg(feature = "std-openai-external")]
            {
                core_usage
                    .map(|u| {
                        let mut builder = Usage::builder()
                            .prompt_tokens(u.prompt_tokens)
                            .completion_tokens(u.completion_tokens)
                            .total_tokens(u.total_tokens);

                        // Preserve reasoning tokens from the original payload if present.
                        let reasoning_tokens = root_ref
                            .get("usage")
                            .and_then(|u| {
                                u.get("reasoning_tokens")
                                    .or_else(|| u.get("reasoningTokens"))
                            })
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32);

                        if let Some(reasoning) = reasoning_tokens {
                            builder = builder.with_reasoning_tokens(reasoning);
                        }

                        builder.build()
                    })
                    .or_else(|| parse_usage_from_root(root_ref))
            }
            #[cfg(not(feature = "std-openai-external"))]
            {
                parse_usage_from_root(root_ref)
            }
        };

        // Finish reason
        let finish_reason = {
            #[cfg(feature = "std-openai-external")]
            {
                use siumai_core::types::FinishReasonCore;

                core_finish
                    .map(|fr| match fr {
                        FinishReasonCore::Stop => FinishReason::Stop,
                        FinishReasonCore::Length => FinishReason::Length,
                        FinishReasonCore::ContentFilter => FinishReason::ContentFilter,
                        FinishReasonCore::ToolCalls => FinishReason::ToolCalls,
                        FinishReasonCore::Other(s) => FinishReason::Other(s),
                    })
                    .or_else(|| {
                        root_ref
                            .get("finish_reason")
                            .or_else(|| root_ref.get("stop_reason"))
                            .and_then(|v| v.as_str())
                            .map(|s| match s {
                                "stop" => FinishReason::Stop,
                                "length" | "max_tokens" => FinishReason::Length,
                                "tool_calls" | "tool_use" | "function_call" => {
                                    FinishReason::ToolCalls
                                }
                                "content_filter" | "safety" => FinishReason::ContentFilter,
                                other => FinishReason::Other(other.to_string()),
                            })
                    })
            }
            #[cfg(not(feature = "std-openai-external"))]
            {
                root_ref
                    .get("finish_reason")
                    .or_else(|| root_ref.get("stop_reason"))
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "stop" => FinishReason::Stop,
                        "length" | "max_tokens" => FinishReason::Length,
                        "tool_calls" | "tool_use" | "function_call" => FinishReason::ToolCalls,
                        "content_filter" | "safety" => FinishReason::ContentFilter,
                        other => FinishReason::Other(other.to_string()),
                    })
            }
        };

        // Determine final content
        let content = if content_parts.is_empty() {
            MessageContent::Text(String::new())
        } else if content_parts.len() == 1 && content_parts[0].is_text() {
            MessageContent::Text(text_content)
        } else {
            MessageContent::MultiModal(content_parts)
        };

        // Extract warnings and provider metadata if present
        Ok(ChatResponse {
            id: root
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            content,
            model: root
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            usage,
            finish_reason,
            audio: None, // Responses API doesn't support audio output yet
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        })
    }
}
