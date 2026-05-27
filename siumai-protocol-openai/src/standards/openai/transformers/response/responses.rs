use crate::error::LlmError;
use crate::execution::transformers::response::ResponseTransformer;
use crate::standards::openai::compat::usage::{
    parse_xai_responses_usage_value, xai_responses_zero_usage,
};
use crate::standards::openai::utils::parse_openai_usage_value;
use crate::types::ChatResponse;
#[cfg(feature = "openai-responses")]
use crate::types::ToolExecutionOwner;

#[cfg(feature = "openai-responses")]
mod hosted_tools;

#[cfg(feature = "openai-responses")]
mod metadata;

#[cfg(feature = "openai-responses")]
mod response_content;

#[cfg(feature = "openai-responses")]
pub(crate) use metadata::output_text_logprobs as extract_responses_output_text_logprobs;

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
const fn provider_executed_flag(is_provider_owned: bool) -> Option<bool> {
    ToolExecutionOwner::from_is_provider_executed(is_provider_owned).to_provider_executed_flag()
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

    fn custom_tool_name_for_call_name(&self, call_name: &str) -> String {
        if call_name.is_empty() {
            return String::new();
        }

        match (self.style, call_name) {
            (ResponsesTransformStyle::Xai, "x_keyword_search") => "x_search".to_string(),
            _ => call_name.to_string(),
        }
    }

    fn parse_usage_value(&self, value: &serde_json::Value) -> Option<crate::types::Usage> {
        match self.style {
            ResponsesTransformStyle::OpenAi => parse_openai_usage_value(value),
            ResponsesTransformStyle::Xai => parse_xai_responses_usage_value(value),
        }
    }
}

#[cfg(feature = "openai-responses")]
impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use crate::types::FinishReason;
        let root = raw.get("response").unwrap_or(raw);
        let xai_style = self.style == ResponsesTransformStyle::Xai;
        let shell_call_provider_executed =
            hosted_tools::response_shell_call_provider_executed(root);

        // Build content parts (tool calls/results + text).
        //
        // Vercel alignment: represent provider-executed tools as ToolCall + ToolResult parts
        // in the unified stream/content surface (without introducing new unified traits).
        let mut content_parts: Vec<crate::types::ContentPart> = Vec::new();

        // Provider-executed tools in Responses API appear as output items:
        // - web_search_call
        // - file_search_call
        // - computer_call
        // - code_interpreter_call
        // - image_generation_call
        // - mcp_call
        //
        // We translate them into legacy tool-call + tool-result compatibility parts with
        // `provider_executed = Some(true)`.
        let mut mcp_approval_tool_call_id_by_approval_id: std::collections::HashMap<
            String,
            String,
        > = std::collections::HashMap::new();
        let mut next_mcp_approval_tool_call_index: usize = 0;
        let mut hosted_tool_search_call_ids: std::collections::VecDeque<String> =
            std::collections::VecDeque::new();

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
                    .unwrap_or("")
                    .to_string();
                let output_call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
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

                    content_parts.push(
                        response_content::tool_call(
                            tool_call_id.clone(),
                            tool_name,
                            serde_json::Value::String(args_str.to_string()),
                            Some(true),
                        )
                        .with_tool_dynamic(true),
                    );
                    content_parts.push(response_content::tool_approval_request(
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
                    } else if !output_call_id.is_empty() {
                        output_call_id.clone()
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

                    let input = serde_json::Value::String(args_str.to_string());
                    content_parts.push(
                        response_content::tool_call(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            input.clone(),
                            Some(true),
                        )
                        .with_tool_dynamic(true),
                    );

                    let provider_metadata = if item_id.is_empty() {
                        None
                    } else {
                        Some(self.single_provider_metadata_map(serde_json::json!({
                            "itemId": item_id,
                        })))
                    };

                    content_parts.push(
                        response_content::tool_result_with_provider_executed(
                            tool_call_id,
                            tool_name,
                            provider_metadata,
                            crate::types::ToolResultOutput::json(serde_json::Value::Object(
                                result_obj,
                            )),
                            provider_executed_flag(true),
                        )
                        .with_tool_result_input(input)
                        .with_tool_dynamic(true),
                    );

                    continue;
                }

                // Reasoning items (o1/o3/gpt-5 reasoning models).
                //
                // Vercel alignment:
                // - Each `summary_text` block becomes a separate `type: "reasoning"` content part.
                // - Empty summary still yields a single reasoning part with empty text.
                // - Surface item id and optional encrypted content via `providerMetadata.openai`.
                if item_type == "reasoning" {
                    if item_id.is_empty() {
                        continue;
                    }

                    let encrypted_content = item
                        .get("encrypted_content")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    let mut openai_meta = serde_json::Map::new();
                    openai_meta.insert(
                        "itemId".to_string(),
                        serde_json::Value::String(item_id.clone()),
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
                            content_parts.push(response_content::reasoning(
                                text.to_string(),
                                provider_metadata.clone(),
                            ));
                            emitted += 1;
                        }
                    }

                    if emitted == 0 {
                        content_parts.push(response_content::reasoning(
                            String::new(),
                            provider_metadata,
                        ));
                    }

                    continue;
                }

                if item_type == "compaction" {
                    if item_id.is_empty() {
                        continue;
                    }

                    let mut openai_meta = serde_json::Map::new();
                    openai_meta.insert(
                        "itemId".to_string(),
                        serde_json::Value::String(item_id.clone()),
                    );
                    openai_meta.insert("type".to_string(), serde_json::json!("compaction"));
                    if let Some(encrypted_content) = item.get("encrypted_content")
                        && !encrypted_content.is_null()
                    {
                        openai_meta
                            .insert("encryptedContent".to_string(), encrypted_content.clone());
                    }

                    content_parts.push(response_content::custom(
                        "openai.compaction",
                        Some(
                            self.single_provider_metadata_map(serde_json::Value::Object(
                                openai_meta,
                            )),
                        ),
                    ));

                    continue;
                }

                let tool_call_id = if output_call_id.is_empty() {
                    item_id.clone()
                } else {
                    output_call_id.clone()
                };

                if item_type == "custom_tool_call" {
                    if tool_call_id.is_empty() {
                        continue;
                    }

                    let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let tool_name = self.custom_tool_name_for_call_name(call_name);
                    if tool_name.is_empty() {
                        continue;
                    }

                    let arguments = item
                        .get("input")
                        .cloned()
                        .unwrap_or_else(|| serde_json::Value::String("{}".to_string()));
                    let provider_item_id = if item_id.is_empty() {
                        tool_call_id.clone()
                    } else {
                        item_id.clone()
                    };
                    let provider_metadata =
                        Some(self.single_provider_metadata_map(serde_json::json!({
                            "itemId": provider_item_id,
                        })));

                    content_parts.push(
                        response_content::tool_call_with_metadata(
                            tool_call_id.clone(),
                            tool_name.clone(),
                            arguments.clone(),
                            Some(true),
                            provider_metadata.clone(),
                        )
                        .with_tool_dynamic(true),
                    );

                    if let Some(output) = item.get("output") {
                        let is_error = item
                            .get("is_error")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);

                        content_parts.push(
                            response_content::tool_result_with_provider_executed(
                                tool_call_id,
                                tool_name,
                                provider_metadata,
                                custom_tool_output_to_result_output(output, is_error),
                                Some(true),
                            )
                            .with_tool_result_input(arguments.clone())
                            .with_tool_dynamic(true),
                        );
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
                        let args = if xai_style {
                            serde_json::Value::String(String::new())
                        } else {
                            serde_json::Value::String("{}".to_string())
                        };
                        let result = if xai_style {
                            serde_json::json!({
                                "queries": hosted_tools::xai_file_search_queries(item),
                                "results": hosted_tools::file_search_results(item),
                            })
                        } else {
                            serde_json::json!({
                                "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                                "results": hosted_tools::file_search_results(item),
                            })
                        };
                        let tool_name = if xai_style {
                            "file_search"
                        } else {
                            "fileSearch"
                        };
                        (tool_name, args, result)
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
                    "tool_search_call" => {
                        let tool_call_id = if output_call_id.is_empty() {
                            item_id.clone()
                        } else {
                            output_call_id.clone()
                        };
                        if tool_call_id.is_empty() {
                            continue;
                        }

                        let is_hosted = item
                            .get("execution")
                            .and_then(|v| v.as_str())
                            .unwrap_or("server")
                            == "server";
                        if is_hosted {
                            hosted_tool_search_call_ids.push_back(tool_call_id.clone());
                        }

                        let arguments_json = serde_json::to_string(
                            item.get("arguments").unwrap_or(&serde_json::Value::Null),
                        )
                        .unwrap_or_else(|_| "null".to_string());
                        let call_id_json = if output_call_id.is_empty() {
                            "null".to_string()
                        } else {
                            serde_json::to_string(&output_call_id)
                                .unwrap_or_else(|_| "null".to_string())
                        };
                        let input = format!(
                            "{{\"arguments\":{arguments_json},\"call_id\":{call_id_json}}}"
                        );
                        let provider_metadata = (!item_id.is_empty()).then(|| {
                            self.single_provider_metadata_map(serde_json::json!({
                                "itemId": item_id,
                            }))
                        });

                        content_parts.push(response_content::tool_call_with_metadata(
                            tool_call_id,
                            "toolSearch",
                            serde_json::Value::String(input),
                            provider_executed_flag(is_hosted),
                            provider_metadata,
                        ));
                        continue;
                    }
                    "tool_search_output" => {
                        let tool_call_id = if output_call_id.is_empty() {
                            hosted_tool_search_call_ids
                                .pop_front()
                                .unwrap_or_else(|| item_id.clone())
                        } else {
                            output_call_id.clone()
                        };
                        if tool_call_id.is_empty() {
                            continue;
                        }

                        let provider_metadata = (!item_id.is_empty()).then(|| {
                            self.single_provider_metadata_map(serde_json::json!({
                                "itemId": item_id,
                            }))
                        });

                        content_parts.push(response_content::tool_result(
                            tool_call_id,
                            "toolSearch",
                            provider_metadata,
                            crate::types::ToolResultOutput::json(serde_json::json!({
                                "tools": item.get("tools").cloned().unwrap_or_else(|| {
                                    serde_json::Value::Array(Vec::new())
                                }),
                            })),
                        ));
                        continue;
                    }
                    "local_shell_call_output" => {
                        if output_call_id.is_empty() {
                            continue;
                        }

                        content_parts.push(
                            response_content::tool_result(
                                output_call_id.clone(),
                                "shell",
                                None,
                                crate::types::ToolResultOutput::json(serde_json::json!({
                                    "output": item
                                        .get("output")
                                        .cloned()
                                        .unwrap_or_else(|| serde_json::json!("")),
                                })),
                            )
                            .with_tool_dynamic(true),
                        );
                        continue;
                    }
                    "shell_call_output" => {
                        if output_call_id.is_empty() {
                            continue;
                        }

                        let output = item
                            .get("output")
                            .and_then(|value| value.as_array())
                            .map(|items| {
                                items
                                    .iter()
                                    .map(|entry| {
                                        let stdout = entry
                                            .get("stdout")
                                            .cloned()
                                            .unwrap_or_else(|| serde_json::json!(""));
                                        let stderr = entry
                                            .get("stderr")
                                            .cloned()
                                            .unwrap_or_else(|| serde_json::json!(""));
                                        let outcome = entry
                                            .get("outcome")
                                            .and_then(|value| value.as_object())
                                            .map(|outcome| {
                                                match outcome
                                                    .get("type")
                                                    .and_then(|value| value.as_str())
                                                    .unwrap_or("")
                                                {
                                                    "timeout" => {
                                                        serde_json::json!({ "type": "timeout" })
                                                    }
                                                    "exit" => serde_json::json!({
                                                        "type": "exit",
                                                        "exitCode": outcome
                                                            .get("exit_code")
                                                            .or_else(|| outcome.get("exitCode"))
                                                            .cloned()
                                                            .unwrap_or_else(|| serde_json::json!(0)),
                                                    }),
                                                    _ => serde_json::Value::Object(outcome.clone()),
                                                }
                                            })
                                            .unwrap_or_else(|| {
                                                serde_json::json!({ "type": "exit", "exitCode": 0 })
                                            });

                                        serde_json::json!({
                                            "stdout": stdout,
                                            "stderr": stderr,
                                            "outcome": outcome,
                                        })
                                    })
                                    .collect::<Vec<serde_json::Value>>()
                            })
                            .unwrap_or_default();

                        content_parts.push(
                            response_content::tool_result(
                                output_call_id.clone(),
                                "shell",
                                None,
                                crate::types::ToolResultOutput::json(serde_json::json!({
                                    "output": output,
                                })),
                            )
                            .with_tool_dynamic(true),
                        );
                        continue;
                    }
                    "apply_patch_call_output" => {
                        if output_call_id.is_empty() {
                            continue;
                        }

                        let mut result = serde_json::Map::new();
                        result.insert(
                            "status".to_string(),
                            item.get("status")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!("completed")),
                        );
                        if let Some(output) = item.get("output").cloned() {
                            result.insert("output".to_string(), output);
                        }

                        content_parts.push(
                            response_content::tool_result(
                                output_call_id.clone(),
                                "apply_patch",
                                None,
                                crate::types::ToolResultOutput::json(serde_json::Value::Object(
                                    result,
                                )),
                            )
                            .with_tool_dynamic(true),
                        );
                        continue;
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

                        // AI SDK doGenerate preserves OpenAI's raw action shape for local_shell_call.
                        let input_str = hosted_tools::local_shell_generate_input(item);

                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(
                            response_content::tool_call_with_metadata(
                                call_id.to_string(),
                                "shell",
                                serde_json::Value::String(input_str),
                                None,
                                provider_metadata,
                            )
                            .with_tool_dynamic(true),
                        );
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

                        content_parts.push(
                            response_content::tool_call_with_metadata(
                                call_id.to_string(),
                                "shell",
                                serde_json::Value::String(input_str),
                                provider_executed_flag(shell_call_provider_executed),
                                provider_metadata,
                            )
                            .with_tool_dynamic(true),
                        );
                        continue;
                    }
                    "apply_patch_call" => {
                        // Vercel alignment: map hosted apply_patch tool call to `toolName: "apply_patch"`,
                        // stringify `{ callId, operation }`, and surface item id via providerMetadata.
                        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }

                        let input_str = hosted_tools::apply_patch_generate_input(call_id, item);

                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(
                            response_content::tool_call_with_metadata(
                                call_id.to_string(),
                                "apply_patch",
                                serde_json::Value::String(input_str),
                                None,
                                provider_metadata,
                            )
                            .with_tool_dynamic(true),
                        );
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
                        let provider_metadata = item.get("id").and_then(|v| v.as_str()).map(|id| {
                            self.single_provider_metadata_map(serde_json::json!({ "itemId": id }))
                        });

                        content_parts.push(response_content::tool_call_with_metadata(
                            call_id.to_string(),
                            name.to_string(),
                            serde_json::Value::String(args_str.to_string()),
                            None,
                            provider_metadata,
                        ));

                        // Function calls are not provider-executed; no synthetic ToolResult is emitted here.
                        continue;
                    }
                    _ => continue,
                };

                if tool_call_id.is_empty() {
                    continue;
                }

                let result_input = args.clone();
                content_parts.push(
                    response_content::tool_call(tool_call_id.clone(), tool_name, args, Some(true))
                        .with_tool_dynamic(true),
                );

                if emit_tool_result {
                    content_parts.push(
                        response_content::tool_result_with_provider_executed(
                            tool_call_id,
                            tool_name.to_string(),
                            None,
                            crate::types::ToolResultOutput::json(result),
                            provider_executed_flag(true),
                        )
                        .with_tool_result_input(result_input)
                        .with_tool_dynamic(true),
                    );
                }
            }
        }

        // Extract text content from output[*].content[*].text.
        //
        // Vercel AI SDK always surfaces OpenAI Responses `message.content[*].output_text`
        // as structured text parts carrying `providerMetadata.{provider}.itemId`.
        // Preserve that typed part structure for standard OpenAI/Azure Responses items,
        // instead of collapsing plain single-text messages back to bare text content.
        let mut text_content = String::new();
        let mut structured_text_parts: Vec<crate::types::ContentPart> = Vec::new();
        let mut xai_source_urls: Vec<String> = Vec::new();
        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                let item_id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let item_phase = item
                    .get("phase")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        let annotations = p
                            .get("annotations")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();

                        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(t);

                            if !xai_style {
                                let mut openai_meta = serde_json::Map::new();
                                if !item_id.is_empty() {
                                    openai_meta.insert(
                                        "itemId".to_string(),
                                        serde_json::Value::String(item_id.clone()),
                                    );
                                }
                                if let Some(phase) = &item_phase {
                                    openai_meta.insert(
                                        "phase".to_string(),
                                        serde_json::Value::String(phase.clone()),
                                    );
                                }
                                if !annotations.is_empty() {
                                    openai_meta.insert(
                                        "annotations".to_string(),
                                        serde_json::Value::Array(annotations.clone()),
                                    );
                                }

                                let provider_metadata = (!openai_meta.is_empty()).then(|| {
                                    self.single_provider_metadata_map(serde_json::Value::Object(
                                        openai_meta,
                                    ))
                                });

                                structured_text_parts
                                    .push(response_content::text(t.to_string(), provider_metadata));
                            }
                        }

                        if xai_style {
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

        if !structured_text_parts.is_empty() {
            content_parts.extend(structured_text_parts);
        } else if !text_content.is_empty() {
            // Add text content if present.
            content_parts.push(response_content::text(text_content.clone(), None));
        }

        if xai_style && !xai_source_urls.is_empty() {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut idx = 0usize;
            for url in xai_source_urls {
                if !seen.insert(url.clone()) {
                    continue;
                }
                let id = format!("id-{idx}");
                content_parts.push(response_content::source_url(&id, &url, &url));
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
                            content_parts
                                .push(response_content::tool_call(id, name, args_value, None));
                        }
                    }
                }
            }
        }

        // Usage
        let usage = root
            .get("usage")
            .and_then(|usage| self.parse_usage_value(usage))
            .or_else(|| xai_style.then(xai_responses_zero_usage));

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
        let raw_finish_reason = incomplete_reason.map(ToString::to_string);

        // Determine final content
        let content = response_content::message_content_from_parts(content_parts, text_content);

        // Provider metadata (Vercel-aligned): sources extracted from provider tool results and
        // message annotations.
        let provider_metadata =
            metadata::response_provider_metadata(root, self.style, &self.provider_metadata_key);

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
            raw_finish_reason,
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
            request: None,
            response: Some(crate::types::HttpResponseInfo {
                timestamp: root
                    .get("created_at")
                    .and_then(|value| value.as_i64())
                    .and_then(|created_at| chrono::DateTime::from_timestamp(created_at, 0))
                    .unwrap_or_else(chrono::Utc::now),
                model_id: root
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                headers: std::collections::HashMap::new(),
                body: Some(raw.clone()),
            }),
            provider_metadata,
        })
    }
}

#[cfg(all(test, feature = "openai-responses"))]
mod tests {
    use super::*;
    use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
    use crate::execution::transformers::response::ResponseTransformer;
    use crate::standards::openai::json_response::OpenAiResponsesJsonResponseConverter;

    fn transformer_source() -> &'static str {
        include_str!("responses.rs")
            .split_once("#[cfg(all(test, feature = \"openai-responses\"))]")
            .expect("test marker should exist")
            .0
    }

    fn response_content_source() -> &'static str {
        include_str!("responses/response_content.rs")
    }

    #[test]
    fn responses_response_transformer_source_does_not_emit_request_provider_options() {
        let source = transformer_source();

        assert!(
            !source.contains("providerOptions"),
            "OpenAI Responses response parsing must not emit request-side providerOptions"
        );
        assert!(
            !source.contains(".provider_options"),
            "OpenAI Responses response parsing must not read request provider_options fields"
        );
        assert!(
            !source.contains("provider_options_map"),
            "OpenAI Responses response parsing must not read request provider option maps"
        );

        for line in source
            .lines()
            .filter(|line| line.contains("provider_options"))
            .filter(|line| !line.trim_start().starts_with("//"))
        {
            let trimmed = line.trim();
            assert!(
                trimmed == "provider_options: crate::types::ProviderOptionsMap::default(),"
                    || trimmed == "provider_options,"
                    || trimmed.contains("provider_options.is_empty()"),
                "OpenAI Responses response provider_options must stay empty defaults or final empty checks: {line}"
            );
        }
    }

    #[test]
    fn responses_response_transformer_delegates_legacy_content_construction_to_adapter() {
        let source = transformer_source();

        assert!(
            source.contains("mod response_content;"),
            "OpenAI Responses response transformer must declare the parser-local response content adapter"
        );
        assert!(
            source.contains("response_content::"),
            "OpenAI Responses response transformer must delegate legacy ContentPart construction"
        );

        for forbidden in [
            "ContentPart::ToolCall {",
            "ContentPart::ToolResult {",
            "ContentPart::Reasoning {",
            "ContentPart::Custom {",
            "ContentPart::Source {",
            "ContentPart::Text {",
            "ContentPart::tool_call(",
            "ContentPart::tool_approval_request(",
            "ContentPart::text(",
            "ContentPart::source(",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI Responses response transformer should not construct legacy content directly; use response_content adapter instead of {forbidden}"
            );
        }
    }

    #[test]
    fn responses_response_content_adapter_owns_empty_request_option_defaults() {
        let source = response_content_source();

        assert!(
            source.contains("ContentPart::"),
            "response content adapter should own legacy ContentPart construction"
        );
        assert!(
            !source.contains("providerOptions"),
            "response content adapter must not emit request-side providerOptions"
        );
        assert!(
            !source.contains(".provider_options"),
            "response content adapter must not read request provider_options fields"
        );
        assert!(
            !source.contains("provider_options_map"),
            "response content adapter must not read request provider option maps"
        );

        for line in source
            .lines()
            .filter(|line| line.contains("provider_options"))
            .filter(|line| !line.trim_start().starts_with("//"))
        {
            let trimmed = line.trim();
            assert!(
                trimmed == "provider_options: ProviderOptionsMap::default(),"
                    || trimmed == "provider_options,"
                    || trimmed.contains("provider_options.is_empty()"),
                "response content adapter may only initialize empty legacy request provider options: {line}"
            );
        }
    }

    #[test]
    fn responses_response_transformer_does_not_force_generated_output_projection() {
        let transformer_source = transformer_source();
        let adapter_source = response_content_source();

        for source in [transformer_source, adapter_source] {
            for forbidden in [
                "GenerateTextContentPart",
                "project_chat_response_to_generate_text_content_parts",
                "project_response_content_to_generate_text_content_parts",
                "project_response_content_part_to_generate_text_content_part",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "OpenAI Responses response parsing must keep generated-output projection as a later lossiness-classified boundary, not call `{forbidden}` directly"
                );
            }
        }
    }

    #[test]
    fn responses_generated_output_projection_accepts_lossless_text_reasoning_and_function_call() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_lossless_1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "enc_reasoning",
                        "summary": [
                            { "type": "summary_text", "text": "Thinking" }
                        ]
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_weather",
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo\"}"
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "content": [
                            { "type": "output_text", "text": "Sunny" }
                        ]
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let projected = crate::types::project_chat_response_to_generate_text_content_parts(&resp)
            .expect("text, reasoning, and function-call response parts should project losslessly");
        let projected_json = serde_json::to_value(&projected).expect("serialize projected parts");

        assert_eq!(projected_json.as_array().expect("array").len(), 3);
        assert_eq!(projected_json[0]["type"], serde_json::json!("reasoning"));
        assert_eq!(
            projected_json[0]["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("rs_1")
        );
        assert_eq!(projected_json[1]["type"], serde_json::json!("tool-call"));
        assert_eq!(
            projected_json[1]["toolCallId"],
            serde_json::json!("call_weather")
        );
        assert_eq!(projected_json[2]["type"], serde_json::json!("text"));
        assert_eq!(
            projected_json[2]["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("msg_1")
        );
        assert_json_has_no_key(&projected_json, "providerOptions");
        assert_json_has_no_key(&projected_json, "provider_options");
    }

    #[test]
    fn responses_generated_output_projection_rejects_mcp_approval_requests() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_mcp_approval_1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "mcp_approval_request",
                        "id": "approval_1",
                        "name": "read_file",
                        "arguments": "{\"path\":\"Cargo.toml\"}"
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let err = crate::types::project_chat_response_to_generate_text_content_parts(&resp)
            .expect_err("MCP approval requests require surrounding approval/tool context");

        assert_eq!(
            err,
            crate::types::GenerateTextContentPartProjectionError::UnsupportedContentPart {
                part_type: "tool-approval-request",
                reason: "tool approval request output requires the original tool call",
            }
        );
    }

    #[test]
    fn responses_generated_output_projection_rejects_output_only_hosted_tool_results() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_hosted_output_1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "local_shell_call_output",
                        "call_id": "call_local",
                        "output": "ok\n"
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let err = crate::types::project_chat_response_to_generate_text_content_parts(&resp)
            .expect_err("hosted tool output-only payloads lack the original tool input");

        assert_eq!(
            err,
            crate::types::GenerateTextContentPartProjectionError::UnsupportedContentPart {
                part_type: "tool-result",
                reason: "tool-result generated output requires original input",
            }
        );
    }

    fn assert_json_has_no_key(value: &serde_json::Value, key: &str) {
        match value {
            serde_json::Value::Object(map) => {
                assert!(
                    !map.contains_key(key),
                    "projected generated output must not contain `{key}`"
                );
                for nested in map.values() {
                    assert_json_has_no_key(nested, key);
                }
            }
            serde_json::Value::Array(values) => {
                for nested in values {
                    assert_json_has_no_key(nested, key);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn responses_chat_response_preserves_raw_response_body() {
        let raw = serde_json::json!({
            "id": "resp_1",
            "created_at": 1710000000,
            "model": "gpt-4.1-mini",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [
                        { "type": "output_text", "text": "ok" }
                    ]
                }
            ],
            "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let response_info = resp.response.expect("response metadata");

        assert_eq!(response_info.model_id.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(response_info.timestamp.timestamp(), 1710000000);
        assert!(response_info.headers.is_empty());
        assert_eq!(
            response_info.body.as_ref().expect("raw body")["id"],
            "resp_1"
        );
    }

    #[test]
    fn responses_provider_tools_are_exposed_as_tool_parts() {
        let raw = serde_json::json!({
                "response": {
                    "id": "resp_1",
                    "model": "gpt-4o-mini",
                    "service_tier": "default",
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
                            {"file_id": "file_1", "filename": "notes.md", "attributes": {"topic": "rust"}, "score": 0.9, "text": "..." }
                        ]
                    },
                    {
                        "type": "computer_call",
                        "id": "cp_1",
                        "status": "completed"
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
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
        assert_eq!(meta.response_id.as_deref(), Some("resp_1"));
        assert_eq!(meta.service_tier.as_deref(), Some("default"));
        assert!(
            resp.provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("openai"))
                .and_then(|metadata| metadata.get("itemId"))
                .is_none()
        );
        let sources = meta.sources.expect("sources present");
        assert_eq!(sources.len(), 4);
        assert!(
            sources
                .iter()
                .any(|s| s.url == "https://blog.rust-lang.org/")
        );
        assert!(sources.iter().any(|s| s.url == "https://www.rust-lang.org"));
        let text = resp
            .content
            .as_multimodal()
            .expect("expected multimodal")
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Text { .. }))
            .expect("text part");
        let text_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(text)
                .expect("typed text metadata");
        assert_eq!(text_meta.item_id.as_deref(), Some("msg_1"));
        assert_eq!(
            text_meta
                .annotations
                .as_ref()
                .map(|annotations| annotations.len()),
            Some(2)
        );
        let document = sources
            .iter()
            .find(|s| s.tool_call_id.as_deref() == Some("fs_1"))
            .expect("file search document source present");
        let source_meta =
            crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(document)
                .expect("typed source metadata");
        assert_eq!(document.filename.as_deref(), Some("notes.md"));
        assert_eq!(document.snippet.as_deref(), Some("..."));
        assert_eq!(document.tool_call_id.as_deref(), Some("fs_1"));
        assert_eq!(source_meta.file_id.as_deref(), Some("file_1"));
        assert!(source_meta.container_id.is_none());
        assert!(source_meta.index.is_none());

        let file_search_result = parts
            .iter()
            .find(|part| {
                matches!(
                    part,
                    crate::types::ContentPart::ToolResult { tool_name, .. }
                        if tool_name == "fileSearch"
                )
            })
            .expect("file search tool result");
        let crate::types::ContentPart::ToolResult { output, .. } = file_search_result else {
            unreachable!("expected file search tool result");
        };
        let crate::types::ToolResultOutput::Json { value, .. } = output else {
            panic!("expected JSON file search output");
        };
        assert_eq!(value["results"][0]["fileId"], serde_json::json!("file_1"));
        assert_eq!(
            value["results"][0]["attributes"]["topic"],
            serde_json::json!("rust")
        );
        assert!(value["results"][0].get("file_id").is_none());

        let cited_document = sources
            .iter()
            .find(|s| s.url == "file_123")
            .expect("message annotation document source present");
        let cited_source_meta =
            crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(cited_document)
                .expect("typed cited source metadata");
        assert_eq!(
            cited_source_meta.metadata_type.as_deref(),
            Some("file_citation")
        );
        assert_eq!(cited_source_meta.file_id.as_deref(), Some("file_123"));
        assert!(cited_source_meta.container_id.is_none());
        assert!(cited_source_meta.index.is_none());
    }

    #[test]
    fn responses_transformer_keeps_plain_output_text_as_structured_text_part() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_plain_text_1",
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_plain_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Plain answer.",
                                "annotations": []
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
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        let text = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Text { .. }))
            .expect("text part");
        let text_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(text)
                .expect("typed text metadata");

        let crate::types::ContentPart::Text { text, .. } = text else {
            unreachable!("expected text part");
        };
        assert_eq!(text, "Plain answer.");
        assert_eq!(text_meta.item_id.as_deref(), Some("msg_plain_1"));
        assert!(text_meta.phase.is_none());
        assert!(text_meta.annotations.is_none());
        assert!(
            resp.provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("openai"))
                .and_then(|metadata| metadata.get("itemId"))
                .is_none()
        );
    }

    #[test]
    fn xai_responses_transformer_uses_xai_usage_semantics() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_xai_usage_1",
                "model": "grok-4",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_xai_usage_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Done."
                            }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 4142,
                    "output_tokens": 254,
                    "total_tokens": 4396,
                    "cost_in_usd_ticks": 113500,
                    "input_tokens_details": {
                        "cached_tokens": 4328
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 10
                    }
                },
                "status": "completed"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new()
            .with_style(ResponsesTransformStyle::Xai)
            .with_provider_metadata_key("xai");
        let resp = tx.transform_chat_response(&raw).unwrap();
        let usage = resp.usage.as_ref().expect("usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(8470));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
        assert_eq!(usage.normalized_output_tokens().total, Some(254));
        assert_eq!(usage.normalized_output_tokens().text, Some(244));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["input_tokens"],
            serde_json::json!(4142)
        );
        let metadata = resp.provider_metadata.as_ref().expect("provider metadata");
        assert_eq!(metadata["xai"]["costInUsdTicks"], serde_json::json!(113500));
    }

    #[test]
    fn xai_responses_transformer_defaults_missing_usage_to_zero() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_xai_no_usage_1",
                "model": "grok-4",
                "output": [],
                "status": "completed"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new()
            .with_style(ResponsesTransformStyle::Xai)
            .with_provider_metadata_key("xai");
        let resp = tx.transform_chat_response(&raw).unwrap();
        let usage = resp.usage.as_ref().expect("zero usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(0));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(0));
        assert_eq!(usage.normalized_output_tokens().text, Some(0));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
        assert!(usage.raw_usage_value().is_none());
        assert!(resp.provider_metadata.is_none());
    }

    #[test]
    fn responses_transformer_surfaces_reasoning_content_part_metadata() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_reasoning_1",
                "model": "o4-mini",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "enc_payload_123",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Let me think."
                            }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "output_tokens_details": {
                        "reasoning_tokens": 1
                    },
                    "total_tokens": 3
                },
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        let reasoning = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Reasoning { .. }))
            .expect("expected reasoning part");

        let meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(reasoning)
                .expect("openai content part metadata");
        assert_eq!(meta.item_id.as_deref(), Some("rs_1"));
        assert_eq!(
            meta.reasoning_encrypted_content.as_deref(),
            Some("enc_payload_123")
        );
    }

    #[test]
    fn responses_transformer_surfaces_typed_source_metadata_for_container_and_file_path() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_sources_2",
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "See attached files.",
                                "annotations": [
                                    {
                                        "type": "container_file_citation",
                                        "file_id": "file_container_1",
                                        "container_id": "container_42",
                                        "index": 3,
                                        "filename": "bundle.txt",
                                        "quote": "Bundle"
                                    },
                                    {
                                        "type": "file_path",
                                        "file_id": "file_path_9",
                                        "index": 5,
                                        "filename": "artifact.bin"
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
        let meta = crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(&resp)
            .expect("openai metadata present");
        let sources = meta.sources.expect("sources present");

        let container_source = sources
            .iter()
            .find(|source| source.url == "file_container_1")
            .expect("container citation source present");
        let container_meta =
            crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(container_source)
                .expect("typed container source metadata");
        assert_eq!(
            container_meta.metadata_type.as_deref(),
            Some("container_file_citation")
        );
        assert_eq!(container_meta.file_id.as_deref(), Some("file_container_1"));
        assert_eq!(container_meta.container_id.as_deref(), Some("container_42"));
        assert_eq!(container_meta.index, Some(3));

        let file_path_source = sources
            .iter()
            .find(|source| source.url == "file_path_9")
            .expect("file path source present");
        let file_path_meta =
            crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(file_path_source)
                .expect("typed file path source metadata");
        assert_eq!(file_path_meta.metadata_type.as_deref(), Some("file_path"));
        assert_eq!(file_path_meta.file_id.as_deref(), Some("file_path_9"));
        assert!(file_path_meta.container_id.is_none());
        assert_eq!(file_path_meta.index, Some(5));
        assert_eq!(
            file_path_source.media_type.as_deref(),
            Some("application/octet-stream")
        );
    }

    #[test]
    fn responses_transformer_keeps_distinct_file_citation_sources_per_index() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_sources_dup",
                "model": "gpt-5",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_dup",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Annotated answer.",
                                "annotations": [
                                    {
                                        "type": "file_citation",
                                        "file_id": "file_shared",
                                        "filename": "resource1.json",
                                        "index": 145
                                    },
                                    {
                                        "type": "file_citation",
                                        "file_id": "file_shared",
                                        "filename": "resource1.json",
                                        "index": 192
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let meta = crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(&resp)
            .expect("openai metadata");
        let sources = meta.sources.expect("sources present");

        assert_eq!(sources.len(), 2);
        let indices: std::collections::BTreeSet<u32> = sources
            .iter()
            .filter_map(|source| {
                crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(source)
                    .and_then(|meta| meta.index)
            })
            .collect();
        assert_eq!(indices, std::collections::BTreeSet::from([145, 192]));
    }

    #[test]
    fn responses_transformer_maps_compaction_items_to_custom_content_parts() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_compaction_1",
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "compaction",
                        "id": "cmp_1",
                        "encrypted_content": "enc_compaction"
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        let custom = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::Custom { .. }))
            .expect("expected custom part");

        match custom {
            crate::types::ContentPart::Custom { kind, .. } => {
                assert_eq!(kind, "openai.compaction");
                let meta =
                    crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(custom)
                        .expect("typed compaction metadata");
                assert_eq!(meta.metadata_type.as_deref(), Some("compaction"));
                assert_eq!(meta.item_id.as_deref(), Some("cmp_1"));
                assert_eq!(meta.encrypted_content.as_deref(), Some("enc_compaction"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn responses_transformer_roundtrips_custom_tool_call_items() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_custom_1",
                "model": "grok-4",
                "output": [
                    {
                        "type": "custom_tool_call",
                        "id": "ct_1",
                        "name": "browser_agent",
                        "input": "{\"url\":\"https://example.com\"}",
                        "output": {
                            "message": "blocked"
                        },
                        "is_error": true,
                        "status": "completed"
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");

        let tool_call = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::ToolCall { .. }))
            .expect("tool call part");
        let tool_result = parts
            .iter()
            .find(|part| matches!(part, crate::types::ContentPart::ToolResult { .. }))
            .expect("tool result part");

        match tool_call {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "ct_1");
                assert_eq!(tool_name, "browser_agent");
                assert_eq!(
                    arguments,
                    &serde_json::Value::String("{\"url\":\"https://example.com\"}".to_string())
                );
                assert_eq!(*provider_executed, Some(true));
            }
            _ => unreachable!(),
        }

        let result_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(tool_result)
                .expect("tool result metadata");
        assert_eq!(result_meta.item_id.as_deref(), Some("ct_1"));

        match tool_result {
            crate::types::ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "ct_1");
                assert_eq!(tool_name, "browser_agent");
                assert_eq!(*provider_executed, Some(true));
                assert_eq!(
                    output,
                    &crate::types::ToolResultOutput::error_json(serde_json::json!({
                        "message": "blocked"
                    }))
                );
            }
            _ => unreachable!(),
        }

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&resp, &mut out, JsonEncodeOptions::default())
            .expect("serialize");
        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("custom_tool_call")
        );
        assert_eq!(value["output"][0]["id"], serde_json::json!("ct_1"));
        assert_eq!(value["output"][0]["is_error"], serde_json::json!(true));
    }

    #[test]
    fn responses_transformer_maps_tool_search_call_and_output() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_tool_search_1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "tool_search_call",
                        "id": "tsc_1",
                        "execution": "server",
                        "call_id": null,
                        "status": "completed",
                        "arguments": { "paths": ["get_weather"] }
                    },
                    {
                        "type": "tool_search_output",
                        "id": "tso_1",
                        "execution": "server",
                        "call_id": null,
                        "status": "completed",
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_weather",
                                "parameters": { "type": "object" }
                            }
                        ]
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        assert_eq!(parts.len(), 2);

        match &parts[0] {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "tsc_1");
                assert_eq!(tool_name, "toolSearch");
                assert_eq!(*provider_executed, Some(true));
                assert_eq!(
                    arguments,
                    &serde_json::Value::String(
                        "{\"arguments\":{\"paths\":[\"get_weather\"]},\"call_id\":null}"
                            .to_string()
                    )
                );
            }
            other => panic!("expected tool call, got {other:?}"),
        }

        match &parts[1] {
            crate::types::ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "tsc_1");
                assert_eq!(tool_name, "toolSearch");
                assert_eq!(*provider_executed, None);
                assert_eq!(
                    output,
                    &crate::types::ToolResultOutput::json(serde_json::json!({
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_weather",
                                "parameters": { "type": "object" }
                            }
                        ]
                    }))
                );
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn responses_transformer_maps_hosted_dynamic_call_outputs() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_hosted_dynamic_1",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "local_shell_call",
                        "id": "lsh_1",
                        "call_id": "call_local",
                        "status": "completed",
                        "action": {
                            "type": "exec",
                            "command": ["pwd"],
                            "timeout_ms": 1000,
                            "user": "coder"
                        }
                    },
                    {
                        "type": "local_shell_call_output",
                        "call_id": "call_local",
                        "output": "ok\n"
                    },
                    {
                        "type": "shell_call",
                        "id": "sh_1",
                        "call_id": "call_shell",
                        "status": "completed",
                        "action": {
                            "commands": ["echo ok"],
                            "timeout_ms": null,
                            "max_output_length": null
                        }
                    },
                    {
                        "type": "shell_call_output",
                        "id": "sho_1",
                        "call_id": "call_shell",
                        "status": "completed",
                        "output": [{
                            "stdout": "ok\n",
                            "stderr": "",
                            "outcome": { "type": "exit", "exit_code": 7 }
                        }]
                    },
                    {
                        "type": "apply_patch_call",
                        "id": "apc_1",
                        "call_id": "call_apply",
                        "status": "completed",
                        "operation": {
                            "type": "delete_file",
                            "path": "src/lib.rs"
                        }
                    },
                    {
                        "type": "apply_patch_call_output",
                        "call_id": "call_apply",
                        "status": "failed",
                        "output": "conflict"
                    }
                ],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        assert_eq!(parts.len(), 6);

        match &parts[0] {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed,
                dynamic,
                provider_metadata,
                ..
            } => {
                assert_eq!(tool_call_id, "call_local");
                assert_eq!(tool_name, "shell");
                assert_eq!(*provider_executed, None);
                assert_eq!(*dynamic, Some(true));
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("openai"))
                        .and_then(|metadata| metadata.get("itemId")),
                    Some(&serde_json::json!("lsh_1"))
                );
                let input: serde_json::Value =
                    serde_json::from_str(arguments.as_str().expect("local shell input string"))
                        .expect("local shell input JSON");
                assert_eq!(input["action"]["type"], serde_json::json!("exec"));
                assert_eq!(input["action"]["command"], serde_json::json!(["pwd"]));
                assert_eq!(input["action"]["timeout_ms"], serde_json::json!(1000));
                assert_eq!(input["action"]["user"], serde_json::json!("coder"));
                assert!(input["action"].get("timeoutMs").is_none());
                assert!(input["action"].get("working_directory").is_none());
                assert!(input["action"].get("workingDirectory").is_none());
                assert!(input["action"].get("env").is_none());
            }
            other => panic!("expected local shell tool call, got {other:?}"),
        }

        match &parts[2] {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "call_shell");
                assert_eq!(tool_name, "shell");
                assert_eq!(*provider_executed, None);
            }
            other => panic!("expected shell tool call, got {other:?}"),
        }

        match &parts[1] {
            crate::types::ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output,
                dynamic,
                ..
            } => {
                assert_eq!(tool_call_id, "call_local");
                assert_eq!(tool_name, "shell");
                assert_eq!(*dynamic, Some(true));
                assert_eq!(
                    output,
                    &crate::types::ToolResultOutput::json(serde_json::json!({
                        "output": "ok\n"
                    }))
                );
            }
            other => panic!("expected local shell tool result, got {other:?}"),
        }

        match &parts[3] {
            crate::types::ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output,
                dynamic,
                ..
            } => {
                assert_eq!(tool_call_id, "call_shell");
                assert_eq!(tool_name, "shell");
                assert_eq!(*dynamic, Some(true));
                assert_eq!(
                    output,
                    &crate::types::ToolResultOutput::json(serde_json::json!({
                        "output": [{
                            "stdout": "ok\n",
                            "stderr": "",
                            "outcome": { "type": "exit", "exitCode": 7 }
                        }]
                    }))
                );
            }
            other => panic!("expected shell tool result, got {other:?}"),
        }

        match &parts[4] {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_metadata,
                ..
            } => {
                assert_eq!(tool_call_id, "call_apply");
                assert_eq!(tool_name, "apply_patch");
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("openai"))
                        .and_then(|metadata| metadata.get("itemId")),
                    Some(&serde_json::json!("apc_1"))
                );
                let input: serde_json::Value =
                    serde_json::from_str(arguments.as_str().expect("apply_patch input string"))
                        .expect("apply_patch input JSON");
                assert_eq!(input["callId"], serde_json::json!("call_apply"));
                assert_eq!(input["operation"]["type"], serde_json::json!("delete_file"));
                assert_eq!(input["operation"]["path"], serde_json::json!("src/lib.rs"));
                assert!(input["operation"].get("diff").is_none());
            }
            other => panic!("expected apply_patch tool call, got {other:?}"),
        }

        match &parts[5] {
            crate::types::ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                output,
                dynamic,
                ..
            } => {
                assert_eq!(tool_call_id, "call_apply");
                assert_eq!(tool_name, "apply_patch");
                assert_eq!(*dynamic, Some(true));
                assert_eq!(
                    output,
                    &crate::types::ToolResultOutput::json(serde_json::json!({
                        "status": "failed",
                        "output": "conflict"
                    }))
                );
            }
            other => panic!("expected apply_patch tool result, got {other:?}"),
        }
    }

    #[test]
    fn responses_transformer_marks_container_shell_provider_executed() {
        let raw = serde_json::json!({
            "response": {
                "id": "resp_shell_container_1",
                "model": "gpt-5.4",
                "tools": [{
                    "type": "shell",
                    "environment": { "type": "container_auto" }
                }],
                "output": [{
                    "type": "shell_call",
                    "id": "sh_1",
                    "call_id": "call_shell",
                    "status": "completed",
                    "action": { "commands": ["echo ok"] }
                }],
                "finish_reason": "stop"
            }
        });

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let parts = resp.content.as_multimodal().expect("expected multimodal");
        assert_eq!(parts.len(), 1);

        match &parts[0] {
            crate::types::ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                provider_executed,
                ..
            } => {
                assert_eq!(tool_call_id, "call_shell");
                assert_eq!(tool_name, "shell");
                assert_eq!(*provider_executed, Some(true));
            }
            other => panic!("expected shell tool call, got {other:?}"),
        }
    }

    #[test]
    fn responses_file_search_sources_roundtrip_with_tool_scoped_metadata() {
        let mut response =
            crate::types::ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
                crate::types::ContentPart::ToolCall {
                    tool_call_id: "fs_1".to_string(),
                    tool_name: "fileSearch".to_string(),
                    arguments: serde_json::json!({ "query": "bridge design" }),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    invalid: None,
                    error: None,
                    title: None,
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: Some(std::collections::HashMap::from([(
                        "openai".to_string(),
                        serde_json::json!({ "itemId": "fs_item_1" }),
                    )])),
                },
                crate::types::ContentPart::ToolResult {
                    tool_call_id: "fs_1".to_string(),
                    tool_name: "fileSearch".to_string(),
                    output: crate::types::ToolResultOutput::json(serde_json::json!({
                        "queries": ["bridge design"]
                    })),
                    input: Some(serde_json::json!({ "query": "bridge design" })),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    preliminary: None,
                    title: None,
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
                crate::types::ContentPart::ToolCall {
                    tool_call_id: "fs_2".to_string(),
                    tool_name: "fileSearch".to_string(),
                    arguments: serde_json::json!({ "query": "bridge design" }),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    invalid: None,
                    error: None,
                    title: None,
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: Some(std::collections::HashMap::from([(
                        "openai".to_string(),
                        serde_json::json!({ "itemId": "fs_item_2" }),
                    )])),
                },
                crate::types::ContentPart::ToolResult {
                    tool_call_id: "fs_2".to_string(),
                    tool_name: "fileSearch".to_string(),
                    output: crate::types::ToolResultOutput::json(serde_json::json!({
                        "queries": ["bridge design"]
                    })),
                    input: Some(serde_json::json!({ "query": "bridge design" })),
                    provider_executed: Some(true),
                    dynamic: Some(true),
                    preliminary: None,
                    title: None,
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
            ]));
        response.id = Some("resp_fs_roundtrip".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.provider_metadata = Some(std::collections::HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "sources": [
                    {
                        "id": "src_fs_1",
                        "source_type": "document",
                        "url": "file_shared",
                        "filename": "design-a.md",
                        "title": "Design A",
                        "snippet": "first hit",
                        "media_type": "text/markdown",
                        "tool_call_id": "fs_1",
                        "provider_metadata": {
                            "openai": {
                                "fileId": "file_shared",
                                "containerId": "container_a",
                                "index": 1
                            }
                        }
                    },
                    {
                        "id": "src_fs_2",
                        "source_type": "document",
                        "url": "file_shared",
                        "filename": "design-b.md",
                        "title": "Design B",
                        "snippet": "second hit",
                        "media_type": "text/markdown",
                        "tool_call_id": "fs_2",
                        "provider_metadata": {
                            "openai": {
                                "fileId": "file_shared",
                                "containerId": "container_b",
                                "index": 2
                            }
                        }
                    }
                ]
            }),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");
        let raw: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(raw["output"][0]["id"], serde_json::json!("fs_item_1"));
        assert_eq!(raw["output"][0]["call_id"], serde_json::json!("fs_1"));
        assert_eq!(
            raw["output"][0]["results"][0]["id"],
            serde_json::json!("src_fs_1")
        );
        assert_eq!(raw["output"][1]["id"], serde_json::json!("fs_item_2"));
        assert_eq!(raw["output"][1]["call_id"], serde_json::json!("fs_2"));
        assert_eq!(
            raw["output"][1]["results"][0]["id"],
            serde_json::json!("src_fs_2")
        );

        let tx = OpenAiResponsesResponseTransformer::new();
        let resp = tx.transform_chat_response(&raw).unwrap();
        let meta = crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(&resp)
            .expect("openai metadata present");
        let sources = meta.sources.expect("sources present");

        assert_eq!(sources.len(), 2);

        let source_a = sources
            .iter()
            .find(|source| source.tool_call_id.as_deref() == Some("fs_1"))
            .expect("tool scoped source a");
        let meta_a = crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(source_a)
            .expect("typed source metadata a");
        assert_eq!(source_a.id, "src_fs_1");
        assert_eq!(meta_a.file_id.as_deref(), Some("file_shared"));
        assert_eq!(meta_a.container_id.as_deref(), Some("container_a"));
        assert_eq!(meta_a.index, Some(1));
        assert_eq!(source_a.filename.as_deref(), Some("design-a.md"));
        assert_eq!(source_a.media_type.as_deref(), Some("text/markdown"));
        assert_eq!(source_a.snippet.as_deref(), Some("first hit"));

        let source_b = sources
            .iter()
            .find(|source| source.tool_call_id.as_deref() == Some("fs_2"))
            .expect("tool scoped source b");
        let meta_b = crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(source_b)
            .expect("typed source metadata b");
        assert_eq!(source_b.id, "src_fs_2");
        assert_eq!(meta_b.file_id.as_deref(), Some("file_shared"));
        assert_eq!(meta_b.container_id.as_deref(), Some("container_b"));
        assert_eq!(meta_b.index, Some(2));
        assert_eq!(source_b.filename.as_deref(), Some("design-b.md"));
        assert_eq!(source_b.media_type.as_deref(), Some("text/markdown"));
        assert_eq!(source_b.snippet.as_deref(), Some("second hit"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn openai_responses_response_transformer_has_hosted_tool_output_module() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/standards/openai/transformers/response/responses.rs"
        ));
        let implementation = source
            .lines()
            .take_while(|line| !line.contains("mod tests {"))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            implementation.contains("mod hosted_tools;"),
            "OpenAI Responses hosted/dynamic output item policy should live behind a narrow response module"
        );

        for forbidden in [
            "fn xai_file_search_queries(",
            "fn file_search_results(",
            "fn local_shell_generate_input(",
            "fn apply_patch_generate_input(",
            "fn response_shell_call_provider_executed(",
        ] {
            assert!(
                !implementation.contains(forbidden),
                "hosted/dynamic output item helper `{forbidden}` should not live in the monolithic response transformer"
            );
        }
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn openai_responses_response_transformer_has_metadata_module() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/standards/openai/transformers/response/responses.rs"
        ));
        let implementation = source
            .lines()
            .take_while(|line| !line.contains("mod tests {"))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            implementation.contains("mod metadata;"),
            "OpenAI Responses provider metadata/source/logprobs aggregation should live behind a narrow response module"
        );

        for forbidden in [
            "fn extract_responses_output_text_logprobs(",
            "let provider_metadata = if xai_style",
            "let mut sources: Vec<serde_json::Value>",
            "seen_source_keys",
        ] {
            assert!(
                !implementation.contains(forbidden),
                "provider metadata/source/logprobs helper `{forbidden}` should not live in the monolithic response transformer"
            );
        }
    }
}

#[cfg(feature = "openai-responses")]
fn custom_tool_output_to_result_output(
    output: &serde_json::Value,
    is_error: bool,
) -> crate::types::ToolResultOutput {
    if is_error {
        if let Some(text) = output.as_str() {
            crate::types::ToolResultOutput::error_text(text.to_string())
        } else {
            crate::types::ToolResultOutput::error_json(output.clone())
        }
    } else if let Some(text) = output.as_str() {
        crate::types::ToolResultOutput::text(text.to_string())
    } else {
        crate::types::ToolResultOutput::json(output.clone())
    }
}
