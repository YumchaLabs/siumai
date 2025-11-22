//! Response transformers for OpenAI (Chat/Embedding/Image) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::response::ResponseTransformer;
#[cfg(feature = "std-openai-external")]
use crate::std_openai::openai::responses::{OpenAiResponsesStandard, parse_responses_output};
use crate::types::{
    ChatResponse, ContentPart, EmbeddingResponse, EmbeddingUsage, FinishReason, GeneratedImage,
    ImageGenerationResponse, MessageContent, Usage,
};
use serde_json::Value;

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
            let mut resp = compat.transform_chat_response(raw)?;
            enhance_chat_response_with_openai_metadata(&mut resp, raw);
            return Ok(resp);
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
            enhance_chat_response_with_openai_metadata(&mut resp, raw);
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

/// Responses API transformer (non-streaming) for OpenAI
///
/// When `std-openai-external` is enabled this delegates shape/usage/finish_reason
/// parsing to `siumai-std-openai` and only performs the final mapping into
/// `ChatResponse`. When the standard crate is disabled it falls back to a minimal
/// in-crate implementation to preserve compatibility.
#[derive(Clone)]
pub struct OpenAiResponsesResponseTransformer;

impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat_response(&self, raw: &Value) -> Result<ChatResponse, LlmError> {
        #[cfg(feature = "std-openai-external")]
        {
            use siumai_core::types::FinishReasonCore;

            // Use OpenAI Responses standard to normalize output/usage/finish_reason
            let std = OpenAiResponsesStandard::new();
            let tx = std.create_response_transformer("openai");
            let result = tx.transform_responses_response(raw)?;

            // The standard normalizes `output` to the nested `response` object when
            // present; keep using the raw JSON for metadata extraction so that
            // we have access to the original `usage` fields.
            let root = raw.get("response").unwrap_or(raw);

            // Parse output (text + tool calls) using the shared helper so that
            // Responses semantics remain consistent across languages/bindings.
            let parsed = parse_responses_output(&result.output);

            let mut parts: Vec<ContentPart> = Vec::new();
            if !parsed.text.is_empty() {
                parts.push(ContentPart::text(parsed.text));
            }
            for call in parsed.tool_calls {
                parts.push(ContentPart::tool_call(
                    call.id,
                    call.name,
                    call.arguments,
                    None,
                ));
            }

            let content = if parts.len() == 1 && parts[0].is_text() {
                MessageContent::Text(parts[0].as_text().unwrap_or_default().to_string())
            } else if !parts.is_empty() {
                MessageContent::MultiModal(parts)
            } else {
                MessageContent::Text(String::new())
            };

            // Map usage to aggregator Usage and enrich with detailed fields
            let usage = result.usage.map(|u| {
                let mut builder = Usage::builder()
                    .prompt_tokens(u.prompt_tokens)
                    .completion_tokens(u.completion_tokens)
                    .total_tokens(u.total_tokens);

                if let Some(usage_json) = root.get("usage") {
                    // Top-level reasoning tokens (legacy fields)
                    if let Some(reasoning) = usage_json
                        .get("reasoning_tokens")
                        .or_else(|| usage_json.get("reasoningTokens"))
                        .and_then(Value::as_u64)
                    {
                        builder = builder.with_reasoning_tokens(reasoning as u32);
                    }

                    // prompt_tokens_details.cached_tokens
                    if let Some(cached) = usage_json
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(Value::as_u64)
                    {
                        builder = builder.with_cached_tokens(cached as u32);
                    }

                    // completion_tokens_details: reasoning / accepted / rejected prediction tokens
                    if let Some(details) = usage_json.get("completion_tokens_details") {
                        if let Some(reasoning) =
                            details.get("reasoning_tokens").and_then(Value::as_u64)
                        {
                            builder = builder.with_reasoning_tokens(reasoning as u32);
                        }
                        if let Some(accepted) = details
                            .get("accepted_prediction_tokens")
                            .and_then(Value::as_u64)
                        {
                            builder = builder.with_accepted_prediction_tokens(accepted as u32);
                        }
                        if let Some(rejected) = details
                            .get("rejected_prediction_tokens")
                            .and_then(Value::as_u64)
                        {
                            builder = builder.with_rejected_prediction_tokens(rejected as u32);
                        }
                    }
                }

                builder.build()
            });

            let finish_reason = match result.finish_reason {
                Some(FinishReasonCore::Stop) => Some(FinishReason::Stop),
                Some(FinishReasonCore::Length) => Some(FinishReason::Length),
                Some(FinishReasonCore::ContentFilter) => Some(FinishReason::ContentFilter),
                Some(FinishReasonCore::ToolCalls) => Some(FinishReason::ToolCalls),
                Some(FinishReasonCore::Other(s)) => Some(FinishReason::Other(s)),
                None => None,
            };

            let mut resp = ChatResponse {
                id: root
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(std::string::ToString::to_string),
                model: root
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(std::string::ToString::to_string),
                content,
                usage,
                finish_reason,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            };

            enhance_chat_response_with_openai_metadata(&mut resp, root);
            Ok(resp)
        }

        #[cfg(not(feature = "std-openai-external"))]
        {
            // Fallback implementation when std-openai is not available.
            // This mirrors the behaviour of the original aggregator transformer.
            let root = raw.get("response").unwrap_or(raw);

            // Collect text and tool calls from the output array
            let mut text_content = String::new();
            let mut parts: Vec<ContentPart> = Vec::new();

            if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
                for item in output {
                    if let Some(content_arr) = item.get("content").and_then(|c| c.as_array()) {
                        for c in content_arr {
                            if let Some(t) = c.get("text").and_then(|v| v.as_str()) {
                                if !text_content.is_empty() {
                                    text_content.push('\n');
                                }
                                text_content.push_str(t);
                            }
                        }
                    }

                    if let Some(tool_calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for call in tool_calls {
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
                                    .unwrap_or(Value::String(arguments));
                                parts.push(ContentPart::tool_call(id, name, args_value, None));
                            }
                        }
                    }
                }
            }

            if !text_content.is_empty() {
                parts.insert(0, ContentPart::text(text_content));
            }

            let content = if parts.len() == 1 && parts[0].is_text() {
                MessageContent::Text(parts[0].as_text().unwrap_or_default().to_string())
            } else if !parts.is_empty() {
                MessageContent::MultiModal(parts)
            } else {
                MessageContent::Text(String::new())
            };

            // Usage and finish_reason parsing in the fallback path mirrors the
            // SSE converter logic.
            let usage = root.get("usage").map(|usage| {
                let prompt_tokens = usage
                    .get("prompt_tokens")
                    .or_else(|| usage.get("input_tokens"))
                    .or_else(|| usage.get("inputTokens"))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0);
                let completion_tokens = usage
                    .get("completion_tokens")
                    .or_else(|| usage.get("output_tokens"))
                    .or_else(|| usage.get("outputTokens"))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0);
                let total_tokens = usage
                    .get("total_tokens")
                    .or_else(|| usage.get("totalTokens"))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(prompt_tokens.saturating_add(completion_tokens));

                let mut builder = Usage::builder()
                    .prompt_tokens(prompt_tokens)
                    .completion_tokens(completion_tokens)
                    .total_tokens(total_tokens);

                if let Some(reasoning) = usage
                    .get("reasoning_tokens")
                    .or_else(|| usage.get("reasoningTokens"))
                    .and_then(Value::as_u64)
                {
                    builder = builder.with_reasoning_tokens(reasoning as u32);
                }

                if let Some(cached) = usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(Value::as_u64)
                {
                    builder = builder.with_cached_tokens(cached as u32);
                }

                if let Some(details) = usage.get("completion_tokens_details") {
                    if let Some(reasoning) = details.get("reasoning_tokens").and_then(Value::as_u64)
                    {
                        builder = builder.with_reasoning_tokens(reasoning as u32);
                    }
                    if let Some(accepted) = details
                        .get("accepted_prediction_tokens")
                        .and_then(Value::as_u64)
                    {
                        builder = builder.with_accepted_prediction_tokens(accepted as u32);
                    }
                    if let Some(rejected) = details
                        .get("rejected_prediction_tokens")
                        .and_then(Value::as_u64)
                    {
                        builder = builder.with_rejected_prediction_tokens(rejected as u32);
                    }
                }

                builder.build()
            });

            let raw_reason = root
                .get("stop_reason")
                .or_else(|| root.get("finish_reason"))
                .and_then(|v| v.as_str());

            let finish_reason = raw_reason.map(|reason| match reason {
                "max_tokens" => FinishReason::Length,
                "tool_use" | "function_call" => FinishReason::ToolCalls,
                "safety" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            });

            let mut resp = ChatResponse {
                id: root
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(std::string::ToString::to_string),
                model: root
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(std::string::ToString::to_string),
                content,
                usage,
                finish_reason,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            };

            enhance_chat_response_with_openai_metadata(&mut resp, root);
            Ok(resp)
        }
    }

    fn transform_embedding_response(&self, _raw: &Value) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "OpenAI Responses transformer does not support embeddings".to_string(),
        ))
    }

    fn transform_image_response(&self, _raw: &Value) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "OpenAI Responses transformer does not support images".to_string(),
        ))
    }
}

/// Build OpenAI-specific metadata from a raw response root object.
///
/// This helper is intentionally conservative and only extracts fields that are
/// explicitly documented and stable:
/// - `usage.completion_tokens_details.reasoning_tokens` (o1/o3 reasoning tokens)
/// - legacy `usage.reasoning_tokens` / `usage.reasoningTokens`
/// - `system_fingerprint`
/// - `service_tier`
fn build_openai_metadata_from_root(root: &Value) -> Option<crate::types::OpenAiMetadata> {
    use crate::types::OpenAiMetadata;

    let mut meta = OpenAiMetadata::default();

    // Reasoning tokens from completion details (preferred)
    if let Some(usage) = root.get("usage") {
        if let Some(details) = usage
            .get("completion_tokens_details")
            .or_else(|| usage.get("output_tokens_details"))
        {
            if let Some(reasoning) = details.get("reasoning_tokens").and_then(Value::as_u64) {
                meta.reasoning_tokens = Some(reasoning as u32);
            }
        }

        // Fallback: legacy top-level reasoning_tokens / reasoningTokens
        if meta.reasoning_tokens.is_none() {
            if let Some(reasoning) = usage
                .get("reasoning_tokens")
                .or_else(|| usage.get("reasoningTokens"))
                .and_then(Value::as_u64)
            {
                meta.reasoning_tokens = Some(reasoning as u32);
            }
        }
    }

    // System fingerprint and service tier from the response root
    if let Some(fp) = root
        .get("system_fingerprint")
        .or_else(|| root.get("systemFingerprint"))
        .and_then(Value::as_str)
    {
        meta.system_fingerprint = Some(fp.to_string());
    }

    if let Some(tier) = root
        .get("service_tier")
        .or_else(|| root.get("serviceTier"))
        .and_then(Value::as_str)
    {
        meta.service_tier = Some(tier.to_string());
    }

    if meta.reasoning_tokens.is_none()
        && meta.system_fingerprint.is_none()
        && meta.service_tier.is_none()
        && meta.revised_prompt.is_none()
    {
        return None;
    }

    Some(meta)
}

/// Enhance a ChatResponse with OpenAI-specific provider metadata and top-level
/// fields (system_fingerprint/service_tier) if present in the raw JSON.
fn enhance_chat_response_with_openai_metadata(resp: &mut ChatResponse, root: &Value) {
    if let Some(meta) = build_openai_metadata_from_root(root) {
        // Populate top-level fields if not already set
        if resp.system_fingerprint.is_none() {
            resp.system_fingerprint = meta.system_fingerprint.clone();
        }
        if resp.service_tier.is_none() {
            resp.service_tier = meta.service_tier.clone();
        }

        if let Ok(value) = serde_json::to_value(meta) {
            if let Value::Object(obj) = value {
                // Merge into existing provider_metadata["openai"] namespace
                let mut provider_metadata = resp.provider_metadata.take().unwrap_or_default();
                let mut openai_meta = provider_metadata.remove("openai").unwrap_or_default();

                for (k, v) in obj {
                    openai_meta.insert(k, v);
                }

                provider_metadata.insert("openai".to_string(), openai_meta);
                resp.provider_metadata = Some(provider_metadata);
            }
        }
    }
}
