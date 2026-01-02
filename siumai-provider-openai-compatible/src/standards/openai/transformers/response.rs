//! Response transformers for OpenAI-compatible protocol (Chat/Embedding/Image) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{
    ChatResponse, EmbeddingResponse, EmbeddingUsage, GeneratedImage, ImageGenerationResponse,
    ModerationResponse, ModerationResult,
};

#[derive(Clone)]
pub struct OpenAiResponseTransformer;

impl ResponseTransformer for OpenAiResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Delegate to OpenAI-compatible response transformer for robust mapping
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let adapter: std::sync::Arc<
            dyn crate::standards::openai::compat::adapter::ProviderAdapter,
        > = std::sync::Arc::new(
            crate::standards::openai::compat::adapter::OpenAiStandardAdapter {
                base_url: String::new(),
            },
        );
        let cfg = crate::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
            "openai",
            "",
            "",
            adapter.clone(),
        )
        .with_model(&model);
        let compat = crate::standards::openai::compat::transformers::CompatResponseTransformer {
            config: cfg,
            adapter,
        };
        compat.transform_chat_response(raw)
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

    fn transform_moderation_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ModerationResponse, LlmError> {
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("Missing OpenAI moderation response model".into()))?
            .to_string();

        let results = raw
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LlmError::ParseError("Missing OpenAI moderation response results".into())
            })?;

        let mut out: Vec<ModerationResult> = Vec::with_capacity(results.len());
        for item in results {
            let flagged = item
                .get("flagged")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let categories_obj = item
                .get("categories")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    LlmError::ParseError("Missing OpenAI moderation response categories".into())
                })?;
            let mut categories: std::collections::HashMap<String, bool> =
                std::collections::HashMap::new();
            for (k, v) in categories_obj {
                categories.insert(k.clone(), v.as_bool().unwrap_or(false));
            }

            let scores_obj = item
                .get("category_scores")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    LlmError::ParseError(
                        "Missing OpenAI moderation response category_scores".into(),
                    )
                })?;
            let mut category_scores: std::collections::HashMap<String, f32> =
                std::collections::HashMap::new();
            for (k, v) in scores_obj {
                let score = v.as_f64().unwrap_or(0.0) as f32;
                category_scores.insert(k.clone(), score);
            }

            out.push(ModerationResult {
                flagged,
                categories,
                category_scores,
            });
        }

        Ok(ModerationResponse {
            results: out,
            model,
        })
    }
}

#[cfg(test)]
mod moderation_tests {
    use super::*;
    use crate::execution::transformers::response::ResponseTransformer;

    #[test]
    fn openai_moderation_response_maps_dynamic_category_keys() {
        let raw = serde_json::json!({
            "id": "modr_123",
            "model": "text-moderation-latest",
            "results": [
                {
                    "flagged": true,
                    "categories": {
                        "hate": false,
                        "hate/threatening": true
                    },
                    "category_scores": {
                        "hate": 0.01,
                        "hate/threatening": 0.99
                    }
                }
            ]
        });

        let tx = OpenAiResponseTransformer;
        let resp = tx
            .transform_moderation_response(&raw)
            .expect("moderation response");
        assert_eq!(resp.model, "text-moderation-latest");
        assert_eq!(resp.results.len(), 1);
        assert!(resp.results[0].flagged);
        assert!(!resp.results[0].categories["hate"]);
        assert!(resp.results[0].categories["hate/threatening"]);
        assert!(resp.results[0].category_scores["hate/threatening"] > 0.9);
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

#[cfg(feature = "openai-responses")]
/// Response transformer for OpenAI Responses API
#[derive(Clone)]
pub struct OpenAiResponsesResponseTransformer;

#[cfg(feature = "openai-responses")]
impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use crate::provider_metadata::openai::OpenAiSource;
        use crate::types::{ContentPart, FinishReason, MessageContent, Usage};
        let root = raw.get("response").unwrap_or(raw);

        // Build content parts (tool calls/results + text).
        //
        // Vercel alignment: represent provider-executed tools as ToolCall + ToolResult parts
        // in the unified stream/content surface (without introducing new unified traits).
        let mut content_parts: Vec<ContentPart> = Vec::new();

        // Provider-executed tools in Responses API appear as output items:
        // - web_search_call
        // - file_search_call
        // - computer_call
        //
        // We translate them into `ContentPart::ToolCall` + `ContentPart::ToolResult` with
        // `provider_executed = Some(true)`.
        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let tool_call_id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("call_id").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();

                let (tool_name, args, result) = match item_type {
                    "web_search_call" => {
                        let args = serde_json::json!({
                            "query": item.get("query").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        let result = serde_json::json!({
                            "results": item.get("results").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        ("web_search", args, result)
                    }
                    "file_search_call" => {
                        let args = serde_json::json!({
                            "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        let result = serde_json::json!({
                            "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                            "results": item.get("results").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        ("file_search", args, result)
                    }
                    "computer_call" => {
                        let status = item
                            .get("status")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!("completed"));
                        let result = serde_json::json!({
                            "type": "computer_use_tool_result",
                            "status": status,
                        });
                        ("computer_use", serde_json::json!({}), result)
                    }
                    "function_call" => {
                        // User-defined function call (tool calling).
                        //
                        // OpenAI Responses encodes arguments as a JSON string. Parse into JSON when possible.
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }

                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        if name.is_empty() {
                            continue;
                        }

                        let args_str = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}");
                        let args_json = serde_json::from_str::<serde_json::Value>(args_str)
                            .unwrap_or_else(|_| serde_json::Value::String(args_str.to_string()));

                        content_parts.push(ContentPart::tool_call(
                            call_id.to_string(),
                            name,
                            args_json,
                            None,
                        ));

                        // Function calls are not provider-executed; no synthetic ToolResult is emitted here.
                        continue;
                    }
                    _ => continue,
                };

                if tool_call_id.is_empty() {
                    continue;
                }

                content_parts.push(ContentPart::tool_call(
                    tool_call_id.clone(),
                    tool_name,
                    args,
                    Some(true),
                ));
                content_parts.push(ContentPart::ToolResult {
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    output: crate::types::ToolResultOutput::json(result),
                    provider_executed: Some(true),
                });
            }
        }

        // Extract text content from output[*].content[*].text
        let mut text_content = String::new();
        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(t);
                        }
                    }
                }
            }
        }

        // Add text content if present
        if !text_content.is_empty() {
            content_parts.push(ContentPart::text(&text_content));
        }

        // Tool calls (support nested function object or flattened)
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
                            // Parse arguments string to JSON Value
                            let args_value = serde_json::from_str(&arguments)
                                .unwrap_or(serde_json::Value::String(arguments));
                            content_parts.push(ContentPart::tool_call(id, name, args_value, None));
                        }
                    }
                }
            }
        }

        // Usage
        let usage = root.get("usage").map(|u| {
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

            builder.build()
        });

        // Finish reason
        let finish_reason = root
            .get("finish_reason")
            .or_else(|| root.get("stop_reason"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "stop" => FinishReason::Stop,
                "length" | "max_tokens" => FinishReason::Length,
                "tool_calls" | "tool_use" | "function_call" => FinishReason::ToolCalls,
                "content_filter" | "safety" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            });

        // Determine final content
        let content = if content_parts.is_empty() {
            MessageContent::Text(String::new())
        } else if content_parts.len() == 1 && content_parts[0].is_text() {
            MessageContent::Text(text_content)
        } else {
            MessageContent::MultiModal(content_parts)
        };

        // Provider metadata (Vercel-aligned): sources extracted from web_search_call results.
        let provider_metadata = {
            let mut sources: Vec<OpenAiSource> = Vec::new();
            let mut seen_urls: std::collections::HashSet<String> = std::collections::HashSet::new();
            if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
                for item in output {
                    if item.get("type").and_then(|v| v.as_str()) != Some("web_search_call") {
                        continue;
                    }
                    let tool_call_id = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("call_id").and_then(|v| v.as_str()))
                        .unwrap_or("")
                        .to_string();
                    if tool_call_id.is_empty() {
                        continue;
                    }
                    let Some(arr) = item.get("results").and_then(|v| v.as_array()) else {
                        continue;
                    };
                    for (i, r) in arr.iter().enumerate() {
                        let Some(obj) = r.as_object() else {
                            continue;
                        };
                        let url = obj.get("url").and_then(|v| v.as_str()).unwrap_or("");
                        if url.is_empty() {
                            continue;
                        }
                        if !seen_urls.insert(url.to_string()) {
                            continue;
                        }
                        let title = obj
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let snippet = obj
                            .get("snippet")
                            .or_else(|| obj.get("text"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());

                        sources.push(OpenAiSource {
                            id: format!("{tool_call_id}:{i}"),
                            source_type: "url".to_string(),
                            url: url.to_string(),
                            title,
                            tool_call_id: Some(tool_call_id.clone()),
                            media_type: None,
                            filename: None,
                            provider_metadata: None,
                            snippet,
                        });
                    }
                }

                // Additionally, OpenAI Responses may include URL citations as annotations
                // on message output parts (Vercel alignment: expand to sources).
                let mut ann_idx: u64 = 0;
                for item in output {
                    if item.get("type").and_then(|v| v.as_str()) != Some("message") {
                        continue;
                    }
                    let Some(content_parts) = item.get("content").and_then(|v| v.as_array()) else {
                        continue;
                    };
                    for cp in content_parts {
                        let Some(annotations) = cp.get("annotations").and_then(|v| v.as_array())
                        else {
                            continue;
                        };
                        for ann in annotations {
                            let ann_type = ann.get("type").and_then(|v| v.as_str()).unwrap_or("");

                            if ann_type == "url_citation" {
                                let url = ann.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                if url.is_empty() {
                                    continue;
                                }
                                if !seen_urls.insert(url.to_string()) {
                                    continue;
                                }
                                let title = ann
                                    .get("title")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                sources.push(OpenAiSource {
                                    id: format!("ann:{ann_idx}"),
                                    source_type: "url".to_string(),
                                    url: url.to_string(),
                                    title,
                                    tool_call_id: None,
                                    media_type: None,
                                    filename: None,
                                    provider_metadata: None,
                                    snippet: None,
                                });
                                ann_idx += 1;
                                continue;
                            }

                            // Document sources
                            if matches!(
                                ann_type,
                                "file_citation" | "container_file_citation" | "file_path"
                            ) {
                                let file_id = ann
                                    .get("file_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if file_id.is_empty() {
                                    continue;
                                }

                                // A pseudo-url key to dedup document sources in the same list.
                                let doc_key = format!("doc:{ann_type}:{file_id}");
                                if !seen_urls.insert(doc_key) {
                                    continue;
                                }

                                let filename = ann
                                    .get("filename")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| Some(file_id.clone()));

                                let title = ann
                                    .get("quote")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| filename.clone())
                                    .or_else(|| Some("Document".to_string()));

                                let media_type = if ann_type == "file_path" {
                                    Some("application/octet-stream".to_string())
                                } else {
                                    Some("text/plain".to_string())
                                };

                                let provider_metadata = match ann_type {
                                    "file_citation" => Some(serde_json::json!({
                                        "openai": { "fileId": file_id }
                                    })),
                                    "container_file_citation" => Some(serde_json::json!({
                                        "openai": {
                                            "fileId": file_id,
                                            "containerId": ann.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                                            "index": ann.get("index").cloned().unwrap_or(serde_json::Value::Null),
                                        }
                                    })),
                                    "file_path" => Some(serde_json::json!({
                                        "openai": {
                                            "fileId": file_id,
                                            "index": ann.get("index").cloned().unwrap_or(serde_json::Value::Null),
                                        }
                                    })),
                                    _ => None,
                                };

                                sources.push(OpenAiSource {
                                    id: format!("ann:{ann_idx}"),
                                    source_type: "document".to_string(),
                                    url: file_id.clone(),
                                    title,
                                    tool_call_id: None,
                                    media_type,
                                    filename,
                                    provider_metadata,
                                    snippet: None,
                                });
                                ann_idx += 1;
                            }
                        }
                    }
                }
            }

            if sources.is_empty() {
                None
            } else {
                let mut openai_meta: std::collections::HashMap<String, serde_json::Value> =
                    std::collections::HashMap::new();
                if let Ok(v) = serde_json::to_value(sources) {
                    openai_meta.insert("sources".to_string(), v);
                }

                if openai_meta.is_empty() {
                    None
                } else {
                    let mut all = std::collections::HashMap::new();
                    all.insert("openai".to_string(), openai_meta);
                    Some(all)
                }
            }
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
            provider_metadata,
        })
    }
}

#[cfg(all(test, feature = "openai-responses"))]
mod tests {
    use super::*;
    use crate::execution::transformers::response::ResponseTransformer;

    #[test]
    fn responses_provider_tools_are_exposed_as_tool_parts() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_1",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "query": "rust 1.85 release notes",
                        "results": [
                            {"url":"https://blog.rust-lang.org/","title":"Rust Blog","snippet":"..."}
                        ]
                    },
                    {
                        "type": "file_search_call",
                        "id": "fs_1",
                        "queries": ["rust 1.85 release notes"],
                        "results": [
                            {"file_id": "file_1", "filename": "notes.md", "score": 0.9, "text": "..." }
                        ]
                    },
                    {
                        "type": "computer_call",
                        "id": "cp_1",
                        "status": "completed"
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Done.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://www.rust-lang.org",
                                        "title": "Rust"
                                    },
                                    {
                                        "type": "file_citation",
                                        "file_id": "file_123",
                                        "filename": "notes.txt",
                                        "quote": "Document"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 },
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer;
        let resp = tx.transform_chat_response(&raw).unwrap();

        // Text should still be accessible even when content is multimodal.
        assert_eq!(resp.content_text(), Some("Done."));

        let parts = resp.content.as_multimodal().expect("expected multimodal");
        assert!(parts.iter().any(|p| p.is_tool_call()));
        assert!(parts.iter().any(|p| p.is_tool_result()));

        let meta = crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(&resp)
            .expect("openai metadata present");
        let sources = meta.sources.expect("sources present");
        assert_eq!(sources.len(), 3);
        assert!(
            sources
                .iter()
                .any(|s| s.url == "https://blog.rust-lang.org/")
        );
        assert!(sources.iter().any(|s| s.url == "https://www.rust-lang.org"));
        assert!(sources.iter().any(|s| s.source_type == "document"));
    }
}
