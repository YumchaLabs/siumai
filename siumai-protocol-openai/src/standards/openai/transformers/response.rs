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
        Ok(ImageGenerationResponse {
            images,
            metadata,
            warnings: None,
            response: None,
        })
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
pub struct OpenAiResponsesResponseTransformer {
    style: ResponsesTransformStyle,
    provider_metadata_key: String,
}

#[cfg(feature = "openai-responses")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesTransformStyle {
    OpenAi,
    Xai,
}

#[cfg(feature = "openai-responses")]
impl Default for OpenAiResponsesResponseTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "openai-responses")]
impl OpenAiResponsesResponseTransformer {
    pub fn new() -> Self {
        Self {
            style: ResponsesTransformStyle::OpenAi,
            provider_metadata_key: "openai".to_string(),
        }
    }

    pub fn with_style(mut self, style: ResponsesTransformStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.provider_metadata_key = key.into();
        self
    }

    fn single_provider_metadata_map(
        &self,
        value: serde_json::Value,
    ) -> std::collections::HashMap<String, serde_json::Value> {
        let mut out = std::collections::HashMap::new();
        out.insert(self.provider_metadata_key.clone(), value);
        out
    }

    fn single_provider_metadata_value(&self, value: serde_json::Value) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert(self.provider_metadata_key.clone(), value);
        serde_json::Value::Object(out)
    }
}

#[cfg(feature = "openai-responses")]
pub(crate) fn extract_responses_output_text_logprobs(
    root: &serde_json::Value,
) -> Option<serde_json::Value> {
    let output = root.get("output")?.as_array()?;

    let mut outer: Vec<serde_json::Value> = Vec::new();
    for item in output {
        if item.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }

        let content = item.get("content").and_then(|v| v.as_array());
        let Some(content) = content else { continue };

        for part in content {
            if part.get("type").and_then(|v| v.as_str()) != Some("output_text") {
                continue;
            }

            let logprobs = part.get("logprobs").and_then(|v| v.as_array());
            let Some(logprobs) = logprobs else { continue };

            let mut inner: Vec<serde_json::Value> = Vec::new();
            for entry in logprobs {
                let token = entry.get("token").and_then(|v| v.as_str()).unwrap_or("");
                if token.is_empty() {
                    continue;
                }

                let logprob = entry
                    .get("logprob")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let mut out_entry = serde_json::Map::new();
                out_entry.insert(
                    "token".to_string(),
                    serde_json::Value::String(token.to_string()),
                );
                out_entry.insert("logprob".to_string(), logprob);

                let top = entry.get("top_logprobs").and_then(|v| v.as_array());
                if let Some(top) = top {
                    let mut tops: Vec<serde_json::Value> = Vec::new();
                    for t in top {
                        let t_token = t.get("token").and_then(|v| v.as_str()).unwrap_or("");
                        if t_token.is_empty() {
                            continue;
                        }
                        let t_logprob =
                            t.get("logprob").cloned().unwrap_or(serde_json::Value::Null);
                        tops.push(serde_json::json!({
                            "token": t_token,
                            "logprob": t_logprob,
                        }));
                    }
                    out_entry.insert("top_logprobs".to_string(), serde_json::Value::Array(tops));
                } else {
                    out_entry.insert("top_logprobs".to_string(), serde_json::Value::Array(vec![]));
                }

                inner.push(serde_json::Value::Object(out_entry));
            }

            if !inner.is_empty() {
                outer.push(serde_json::Value::Array(inner));
            }
        }
    }

    if outer.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(outer))
    }
}

#[cfg(feature = "openai-responses")]
impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use crate::types::{ContentPart, FinishReason, MessageContent, Usage};
        let root = raw.get("response").unwrap_or(raw);
        let xai_style = self.style == ResponsesTransformStyle::Xai;

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
        let mut mcp_approval_tool_call_id_by_approval_id: std::collections::HashMap<
            String,
            String,
        > = std::collections::HashMap::new();
        let mut next_mcp_approval_tool_call_index: usize = 0;

        let mut mcp_approval_tool_call_id = |approval_id: &str| -> String {
            if approval_id.is_empty() {
                return "id-0".to_string();
            }

            if let Some(id) = mcp_approval_tool_call_id_by_approval_id.get(approval_id) {
                return id.clone();
            }

            let idx = next_mcp_approval_tool_call_index;
            next_mcp_approval_tool_call_index = next_mcp_approval_tool_call_index.saturating_add(1);

            let id = format!("id-{idx}");
            mcp_approval_tool_call_id_by_approval_id.insert(approval_id.to_string(), id.clone());
            id
        };

        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let item_id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("call_id").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();

                // Vercel alignment: MCP listTools items are internal and should not be surfaced.
                if item_type == "mcp_list_tools" {
                    continue;
                }

                // Vercel alignment: MCP approval requests are emitted as a tool-call plus a
                // separate tool-approval-request part, with a deterministic synthetic toolCallId.
                if item_type == "mcp_approval_request" {
                    let approval_id = item
                        .get("approval_request_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if approval_id.is_empty() {
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

                    let tool_call_id = mcp_approval_tool_call_id(approval_id);
                    let tool_name = format!("mcp.{name}");

                    content_parts.push(ContentPart::tool_call(
                        tool_call_id.clone(),
                        tool_name,
                        serde_json::Value::String(args_str.to_string()),
                        Some(true),
                    ));
                    content_parts.push(ContentPart::tool_approval_request(
                        approval_id.to_string(),
                        tool_call_id,
                    ));
                    continue;
                }

                // Vercel alignment: MCP calls are surfaced as dynamic provider-executed tool
                // calls/results, and results carry itemId provider metadata.
                if item_type == "mcp_call" {
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if name.is_empty() {
                        continue;
                    }

                    let server_label = item
                        .get("server_label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let args_str = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");

                    let tool_call_id = if let Some(approval_id) =
                        item.get("approval_request_id").and_then(|v| v.as_str())
                        && !approval_id.is_empty()
                    {
                        mcp_approval_tool_call_id(approval_id)
                    } else {
                        item_id.clone()
                    };

                    if tool_call_id.is_empty() {
                        continue;
                    }

                    let tool_name = format!("mcp.{name}");

                    let mut result_obj = serde_json::Map::new();
                    result_obj.insert("type".to_string(), serde_json::json!("call"));
                    result_obj.insert(
                        "serverLabel".to_string(),
                        serde_json::Value::String(server_label.to_string()),
                    );
                    result_obj.insert(
                        "name".to_string(),
                        serde_json::Value::String(name.to_string()),
                    );
                    result_obj.insert(
                        "arguments".to_string(),
                        serde_json::Value::String(args_str.to_string()),
                    );

                    if let Some(output) = item.get("output")
                        && !output.is_null()
                    {
                        result_obj.insert("output".to_string(), output.clone());
                    }
                    if let Some(error) = item.get("error")
                        && !error.is_null()
                    {
                        result_obj.insert("error".to_string(), error.clone());
                    }

                    content_parts.push(ContentPart::tool_call(
                        tool_call_id.clone(),
                        tool_name.clone(),
                        serde_json::Value::String(args_str.to_string()),
                        Some(true),
                    ));

                    let provider_metadata = if item_id.is_empty() {
                        None
                    } else {
                        Some(self.single_provider_metadata_map(serde_json::json!({
                            "itemId": item_id,
                        })))
                    };

                    content_parts.push(ContentPart::ToolResult {
                        tool_call_id,
                        tool_name,
                        output: crate::types::ToolResultOutput::json(serde_json::Value::Object(
                            result_obj,
                        )),
                        provider_executed: Some(true),
                        provider_metadata,
                    });

                    continue;
                }

                let tool_call_id = item_id;

                // Reasoning items (o1/o3/gpt-5 reasoning models).
                //
                // Vercel alignment:
                // - Each `summary_text` block becomes a separate `type: "reasoning"` content part.
                // - Empty summary still yields a single reasoning part with empty text.
                // - Surface item id and optional encrypted content via `providerMetadata.openai`.
                if item_type == "reasoning" {
                    if tool_call_id.is_empty() {
                        continue;
                    }

                    let encrypted_content = item
                        .get("encrypted_content")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    let mut openai_meta = serde_json::Map::new();
                    openai_meta.insert(
                        "itemId".to_string(),
                        serde_json::Value::String(tool_call_id.clone()),
                    );
                    openai_meta.insert(
                        "reasoningEncryptedContent".to_string(),
                        encrypted_content
                            .as_ref()
                            .map(|s| serde_json::Value::String(s.clone()))
                            .unwrap_or(serde_json::Value::Null),
                    );

                    let provider_metadata = Some(
                        self.single_provider_metadata_map(serde_json::Value::Object(openai_meta)),
                    );

                    let mut emitted = 0usize;
                    if let Some(summary) = item.get("summary").and_then(|v| v.as_array()) {
                        for s in summary {
                            if s.get("type").and_then(|v| v.as_str()) != Some("summary_text") {
                                continue;
                            }
                            let text = s.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            content_parts.push(ContentPart::Reasoning {
                                text: text.to_string(),
                                provider_metadata: provider_metadata.clone(),
                            });
                            emitted += 1;
                        }
                    }

                    if emitted == 0 {
                        content_parts.push(ContentPart::Reasoning {
                            text: String::new(),
                            provider_metadata,
                        });
                    }

                    continue;
                }

                let mut emit_tool_result = true;
                let (tool_name, args, result) = match item_type {
                    "web_search_call" => {
                        if xai_style {
                            // xAI Vercel alignment:
                            // - toolName: `web_search` (snake_case)
                            // - tool input: keep `arguments` as-is (JSON string)
                            // - no tool-result part; citations are surfaced as `source` parts
                            let Some(arguments) = item.get("arguments").and_then(|v| v.as_str())
                            else {
                                continue;
                            };
                            emit_tool_result = false;

                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let tool_name = if name.is_empty() { "web_search" } else { name };
                            (
                                tool_name,
                                serde_json::Value::String(arguments.to_string()),
                                serde_json::Value::Null,
                            )
                        } else {
                            // Vercel alignment: provider-executed web search tool call uses empty input (`{}`),
                            // while the tool result contains an `action` object and optional `sources`.
                            let args = serde_json::Value::String("{}".to_string());

                            let action = item.get("action").unwrap_or(&serde_json::Value::Null);
                            let action_type_raw = action
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("search");
                            let action_type = match action_type_raw {
                                "open_page" => "openPage",
                                "find_in_page" => "findInPage",
                                other => other,
                            };

                            let mut action_obj = serde_json::Map::new();
                            action_obj.insert(
                                "type".to_string(),
                                serde_json::Value::String(action_type.to_string()),
                            );

                            if action_type_raw == "search" {
                                let query = action
                                    .get("query")
                                    .or_else(|| item.get("query"))
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                if !query.is_null() {
                                    action_obj.insert("query".to_string(), query);
                                }
                            }

                            if matches!(action_type_raw, "open_page" | "find_in_page") {
                                let url = action
                                    .get("url")
                                    .or_else(|| item.get("url"))
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                if !url.is_null() {
                                    action_obj.insert("url".to_string(), url);
                                }
                            }

                            if action_type_raw == "find_in_page" {
                                let pattern = action
                                    .get("pattern")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                if !pattern.is_null() {
                                    action_obj.insert("pattern".to_string(), pattern);
                                }
                            }

                            let mut result_obj = serde_json::Map::new();
                            result_obj.insert(
                                "action".to_string(),
                                serde_json::Value::Object(action_obj),
                            );

                            let sources = action.get("sources").and_then(|v| v.as_array());
                            if let Some(sources) = sources
                                && !sources.is_empty()
                            {
                                result_obj.insert(
                                    "sources".to_string(),
                                    serde_json::Value::Array(sources.to_vec()),
                                );
                            }

                            // Fallback: older response variants may surface results directly.
                            if !result_obj.contains_key("sources")
                                && let Some(results) =
                                    item.get("results").and_then(|v| v.as_array())
                                && !results.is_empty()
                            {
                                let mut sources_out: Vec<serde_json::Value> =
                                    Vec::with_capacity(results.len());
                                for r in results {
                                    let Some(url) = r.get("url").and_then(|v| v.as_str()) else {
                                        continue;
                                    };
                                    sources_out
                                        .push(serde_json::json!({ "type": "url", "url": url }));
                                }
                                if !sources_out.is_empty() {
                                    result_obj.insert(
                                        "sources".to_string(),
                                        serde_json::Value::Array(sources_out),
                                    );
                                }
                            }

                            ("webSearch", args, serde_json::Value::Object(result_obj))
                        }
                    }
                    "file_search_call" => {
                        // Vercel alignment: provider-executed file search tool call uses empty input (`{}`),
                        // and returns queries/results in the tool result.
                        let args = serde_json::Value::String("{}".to_string());
                        let result = serde_json::json!({
                            "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                            "results": item.get("results").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        ("fileSearch", args, result)
                    }
                    "code_interpreter_call" => {
                        if xai_style {
                            // xAI Vercel alignment:
                            // - toolName: `code_execution` (snake_case)
                            // - tool input: keep `arguments` as-is (JSON string)
                            // - no tool-result part; output is surfaced as normal text message
                            let Some(arguments) = item.get("arguments").and_then(|v| v.as_str())
                            else {
                                continue;
                            };
                            emit_tool_result = false;

                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let tool_name = if name.is_empty() {
                                "code_execution"
                            } else {
                                name
                            };

                            (
                                tool_name,
                                serde_json::Value::String(arguments.to_string()),
                                serde_json::Value::Null,
                            )
                        } else {
                            // Vercel alignment (OpenAI):
                            // - toolName: `codeExecution`
                            // - tool input: JSON string `{ code, containerId }` (preserve key order)
                            // - tool result: `{ outputs }`
                            let code = item.get("code").and_then(|v| v.as_str()).unwrap_or("");
                            let container_id = item
                                .get("container_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");

                            let code_json =
                                serde_json::to_string(&code).unwrap_or_else(|_| "\"\"".to_string());
                            let container_json = serde_json::to_string(&container_id)
                                .unwrap_or_else(|_| "\"\"".to_string());
                            let input_str = format!(
                                "{{\"code\":{code_json},\"containerId\":{container_json}}}"
                            );

                            let outputs = item
                                .get("outputs")
                                .cloned()
                                .unwrap_or_else(|| serde_json::Value::Array(vec![]));
                            let result = serde_json::json!({ "outputs": outputs });

                            (
                                "codeExecution",
                                serde_json::Value::String(input_str),
                                result,
                            )
                        }
                    }
                    "image_generation_call" => {
                        // Vercel alignment:
                        // - toolName: `generateImage`
                        // - tool input: `{}` (provider executed)
                        // - tool result: `{ result }` (base64 string)
                        let args = serde_json::Value::String("{}".to_string());
                        let result = serde_json::json!({
                            "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
                        });
                        ("generateImage", args, result)
                    }
                    "x_search_call" => {
                        if !xai_style {
                            continue;
                        }
                        let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        // xAI Vercel alignment:
                        // - toolName: `x_search` (public tool name)
                        // - tool input: keep `arguments` as-is (JSON string)
                        // - no tool-result part; citations are surfaced as `source` parts
                        emit_tool_result = false;
                        (
                            "x_search",
                            serde_json::Value::String(arguments.to_string()),
                            serde_json::Value::Null,
                        )
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
                        // Vercel alignment: computer use tool calls have an empty input string.
                        (
                            "computer_use",
                            serde_json::Value::String(String::new()),
                            result,
                        )
                    }
                    "local_shell_call" => {
                        // Vercel alignment: map OpenAI hosted local shell tool call to `toolName: "shell"`,
                        // keep `input` as a JSON string, and surface the output item id via providerMetadata.
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }

                        let action = item.get("action").unwrap_or(&serde_json::Value::Null);
                        // NOTE: serde_json serializes object keys in sorted order by default, but Vercel's
                        // JSON.stringify preserves insertion order. Build the JSON string manually to match.
                        let action_type = serde_json::to_string(
                            action.get("type").unwrap_or(&serde_json::json!("exec")),
                        )
                        .unwrap_or_else(|_| "\"exec\"".to_string());
                        let command = serde_json::to_string(
                            action.get("command").unwrap_or(&serde_json::json!([])),
                        )
                        .unwrap_or_else(|_| "[]".to_string());
                        let working_directory = serde_json::to_string(
                            action
                                .get("working_directory")
                                .unwrap_or(&serde_json::json!("/")),
                        )
                        .unwrap_or_else(|_| "\"/\"".to_string());
                        let env = serde_json::to_string(
                            action.get("env").unwrap_or(&serde_json::json!({})),
                        )
                        .unwrap_or_else(|_| "{}".to_string());
                        let input_str = format!(
                            "{{\"action\":{{\"type\":{action_type},\"command\":{command},\"working_directory\":{working_directory},\"env\":{env}}}}}"
                        );

                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(ContentPart::ToolCall {
                            tool_call_id: call_id.to_string(),
                            tool_name: "shell".to_string(),
                            arguments: serde_json::Value::String(input_str),
                            provider_executed: None,
                            provider_metadata,
                        });
                        continue;
                    }
                    "shell_call" => {
                        // Vercel alignment: map OpenAI hosted shell tool call to `toolName: "shell"` and
                        // stringify `{ action: { commands } }` (omit other action fields like timeouts).
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }

                        let action = item.get("action").unwrap_or(&serde_json::Value::Null);
                        let commands = serde_json::to_string(
                            action.get("commands").unwrap_or(&serde_json::json!([])),
                        )
                        .unwrap_or_else(|_| "[]".to_string());
                        let input_str = format!("{{\"action\":{{\"commands\":{commands}}}}}");

                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(ContentPart::ToolCall {
                            tool_call_id: call_id.to_string(),
                            tool_name: "shell".to_string(),
                            arguments: serde_json::Value::String(input_str),
                            provider_executed: None,
                            provider_metadata,
                        });
                        continue;
                    }
                    "apply_patch_call" => {
                        // Vercel alignment: map hosted apply_patch tool call to `toolName: "apply_patch"`,
                        // stringify `{ callId, operation }`, and surface item id via providerMetadata.
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }

                        let op = item.get("operation").unwrap_or(&serde_json::Value::Null);
                        let call_id_json =
                            serde_json::to_string(&call_id).unwrap_or_else(|_| "\"\"".to_string());
                        let op_type = serde_json::to_string(
                            op.get("type").unwrap_or(&serde_json::Value::Null),
                        )
                        .unwrap_or_else(|_| "null".to_string());
                        let op_path = serde_json::to_string(
                            op.get("path").unwrap_or(&serde_json::Value::Null),
                        )
                        .unwrap_or_else(|_| "null".to_string());
                        let op_diff = serde_json::to_string(
                            op.get("diff").unwrap_or(&serde_json::Value::Null),
                        )
                        .unwrap_or_else(|_| "null".to_string());
                        let input_str = format!(
                            "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type},\"path\":{op_path},\"diff\":{op_diff}}}}}"
                        );

                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(ContentPart::ToolCall {
                            tool_call_id: call_id.to_string(),
                            tool_name: "apply_patch".to_string(),
                            arguments: serde_json::Value::String(input_str),
                            provider_executed: None,
                            provider_metadata,
                        });
                        continue;
                    }
                    "function_call" => {
                        // User-defined function call (tool calling).
                        //
                        // OpenAI Responses encodes arguments as a JSON string.
                        //
                        // Vercel alignment: keep the raw JSON string as `input` instead of parsing, and
                        // surface the output item id via `providerMetadata.openai.itemId`.
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
                        let mut provider_metadata: Option<
                            std::collections::HashMap<String, serde_json::Value>,
                        > = None;
                        if let Some(item_id) = item.get("id").and_then(|v| v.as_str()) {
                            provider_metadata =
                                Some(self.single_provider_metadata_map(serde_json::json!({
                                    "itemId": item_id
                                })));
                        }

                        content_parts.push(ContentPart::ToolCall {
                            tool_call_id: call_id.to_string(),
                            tool_name: name.to_string(),
                            arguments: serde_json::Value::String(args_str.to_string()),
                            provider_executed: None,
                            provider_metadata,
                        });

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

                if emit_tool_result {
                    content_parts.push(ContentPart::ToolResult {
                        tool_call_id,
                        tool_name: tool_name.to_string(),
                        output: crate::types::ToolResultOutput::json(result),
                        provider_executed: Some(true),
                        provider_metadata: None,
                    });
                }
            }
        }

        // Extract text content from output[*].content[*].text
        let mut text_content = String::new();
        let mut xai_source_urls: Vec<String> = Vec::new();
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

                        if xai_style {
                            let Some(annotations) = p.get("annotations").and_then(|v| v.as_array())
                            else {
                                continue;
                            };
                            for ann in annotations {
                                if ann.get("type").and_then(|v| v.as_str()) != Some("url_citation")
                                {
                                    continue;
                                }
                                let url = ann.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                if url.is_empty() {
                                    continue;
                                }
                                xai_source_urls.push(url.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Add text content if present
        if !text_content.is_empty() {
            content_parts.push(ContentPart::text(&text_content));
        }

        if xai_style && !xai_source_urls.is_empty() {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut idx = 0usize;
            for url in xai_source_urls {
                if !seen.insert(url.clone()) {
                    continue;
                }
                let id = format!("id-{idx}");
                content_parts.push(ContentPart::source(&id, "url", &url, &url));
                idx += 1;
            }
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

            let cached_tokens = u
                .get("input_tokens_details")
                .or_else(|| u.get("prompt_tokens_details"))
                .and_then(|v| v.get("cached_tokens").or_else(|| v.get("cachedTokens")))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

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

            let reasoning_tokens = reasoning_tokens.or_else(|| {
                u.get("output_tokens_details")
                    .or_else(|| u.get("completion_tokens_details"))
                    .and_then(|v| {
                        v.get("reasoning_tokens")
                            .or_else(|| v.get("reasoningTokens"))
                    })
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
            });

            let mut builder = Usage::builder()
                .prompt_tokens(prompt_tokens)
                .completion_tokens(completion_tokens)
                .total_tokens(total_tokens);

            if let Some(cached) = cached_tokens {
                builder = builder.with_cached_tokens(cached);
            }

            if let Some(reasoning) = reasoning_tokens {
                builder = builder.with_reasoning_tokens(reasoning);
            }

            builder.build()
        });

        // Finish reason
        //
        // Vercel alignment (OpenAI Responses):
        // - `finishReason.raw` comes from `response.incomplete_details?.reason`
        // - `finishReason.unified` is derived from:
        //   - `incomplete_details.reason` (e.g. `max_output_tokens`, `content_filter`)
        //   - whether there was a *client-side* function call (`function_call`)
        //
        // Note: Provider-executed tools (e.g. `web_search_call`, `file_search_call`, `code_interpreter_call`,
        // `image_generation_call`, `computer_call`, `mcp_call`) do NOT make the finish reason `tool-calls`.
        let explicit_finish_reason = root
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

        let has_function_call = root
            .get("output")
            .and_then(|v| v.as_array())
            .is_some_and(|out| {
                out.iter()
                    .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("function_call"))
            });

        let incomplete_reason = root
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(|v| v.as_str());

        let status = root.get("status").and_then(|v| v.as_str());
        let inferred_finish_reason = match status {
            Some("failed") => Some(FinishReason::Error),
            _ => match incomplete_reason {
                None => Some(if has_function_call {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                }),
                Some("max_output_tokens") => Some(FinishReason::Length),
                Some("content_filter") => Some(FinishReason::ContentFilter),
                Some(_) => Some(if has_function_call {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Other("other".to_string())
                }),
            },
        };

        let finish_reason = explicit_finish_reason.or(inferred_finish_reason);

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
            let mut item_id: Option<String> = None;
            if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
                for item in output {
                    if item.get("type").and_then(|v| v.as_str()) != Some("message") {
                        continue;
                    }
                    let role = item.get("role").and_then(|v| v.as_str());
                    if role != Some("assistant") {
                        continue;
                    }
                    if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                        item_id = Some(id.to_string());
                        break;
                    }
                }
            }

            let mut sources: Vec<serde_json::Value> = Vec::new();
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

                        let mut src = serde_json::Map::new();
                        src.insert(
                            "id".to_string(),
                            serde_json::Value::String(format!("{tool_call_id}:{i}")),
                        );
                        src.insert(
                            "source_type".to_string(),
                            serde_json::Value::String("url".to_string()),
                        );
                        src.insert(
                            "url".to_string(),
                            serde_json::Value::String(url.to_string()),
                        );
                        if let Some(t) = title {
                            src.insert("title".to_string(), serde_json::Value::String(t));
                        }
                        src.insert(
                            "tool_call_id".to_string(),
                            serde_json::Value::String(tool_call_id.clone()),
                        );
                        if let Some(s) = snippet {
                            src.insert("snippet".to_string(), serde_json::Value::String(s));
                        }
                        sources.push(serde_json::Value::Object(src));
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
                                if xai_style {
                                    // xAI Vercel alignment: citations are exposed as `source` parts,
                                    // so keep provider metadata minimal.
                                    continue;
                                }
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
                                let mut src = serde_json::Map::new();
                                src.insert(
                                    "id".to_string(),
                                    serde_json::Value::String(format!("ann:{ann_idx}")),
                                );
                                src.insert(
                                    "source_type".to_string(),
                                    serde_json::Value::String("url".to_string()),
                                );
                                src.insert(
                                    "url".to_string(),
                                    serde_json::Value::String(url.to_string()),
                                );
                                if let Some(t) = title {
                                    src.insert("title".to_string(), serde_json::Value::String(t));
                                }
                                sources.push(serde_json::Value::Object(src));
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
                                    "file_citation" => Some(self.single_provider_metadata_value(
                                        serde_json::json!({ "fileId": file_id }),
                                    )),
                                    "container_file_citation" => Some(self.single_provider_metadata_value(
                                        serde_json::json!({
                                            "fileId": file_id,
                                            "containerId": ann.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                                            "index": ann.get("index").cloned().unwrap_or(serde_json::Value::Null),
                                        }),
                                    )),
                                    "file_path" => Some(self.single_provider_metadata_value(
                                        serde_json::json!({
                                            "fileId": file_id,
                                            "index": ann.get("index").cloned().unwrap_or(serde_json::Value::Null),
                                        }),
                                    )),
                                    _ => None,
                                };

                                let mut src = serde_json::Map::new();
                                src.insert(
                                    "id".to_string(),
                                    serde_json::Value::String(format!("ann:{ann_idx}")),
                                );
                                src.insert(
                                    "source_type".to_string(),
                                    serde_json::Value::String("document".to_string()),
                                );
                                src.insert("url".to_string(), serde_json::Value::String(file_id));
                                if let Some(t) = title {
                                    src.insert("title".to_string(), serde_json::Value::String(t));
                                }
                                if let Some(mt) = media_type {
                                    src.insert(
                                        "media_type".to_string(),
                                        serde_json::Value::String(mt),
                                    );
                                }
                                if let Some(fn_) = filename {
                                    src.insert(
                                        "filename".to_string(),
                                        serde_json::Value::String(fn_),
                                    );
                                }
                                if let Some(pm) = provider_metadata {
                                    src.insert("provider_metadata".to_string(), pm);
                                }
                                sources.push(serde_json::Value::Object(src));
                                ann_idx += 1;
                            }
                        }
                    }
                }
            }

            let mut openai_meta: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            if let Some(id) = item_id {
                openai_meta.insert("itemId".to_string(), serde_json::Value::String(id));
            }

            if !sources.is_empty() {
                openai_meta.insert("sources".to_string(), serde_json::Value::Array(sources));
            }

            if let Some(logprobs) = extract_responses_output_text_logprobs(root) {
                openai_meta.insert("logprobs".to_string(), logprobs);
            }

            if openai_meta.is_empty() {
                None
            } else {
                let mut all = std::collections::HashMap::new();
                all.insert(self.provider_metadata_key.clone(), openai_meta);
                Some(all)
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
            system_fingerprint: root
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            service_tier: root
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
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

        let tx = OpenAiResponsesResponseTransformer::new();
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
