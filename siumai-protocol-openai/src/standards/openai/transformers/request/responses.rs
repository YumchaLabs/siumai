use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::types::{
    ChatRequest, FilePartSource, MediaSource, ModerationRequest, ProviderOptionsMap,
    ProviderReference, RerankRequest,
};
use base64::Engine;

/// Request transformer for OpenAI Responses API
#[derive(Clone)]
#[cfg(feature = "openai-responses")]
pub struct OpenAiResponsesRequestTransformer;

#[cfg(feature = "openai-responses")]
#[derive(Debug, Default)]
struct ResponsesInputConversionState {
    // Vercel parity: reasoning parts are merged by `itemId` across the entire prompt.
    // For store=false we keep the index of the first emitted reasoning item so we can
    // append subsequent summary parts in-place.
    reasoning_item_index: std::collections::HashMap<String, usize>,
    // For store=true we only emit a single item_reference per reasoning id.
    reasoning_item_seen: std::collections::HashSet<String>,
}

#[cfg(feature = "openai-responses")]
fn provider_option_object<'a>(
    provider_options: Option<&'a ProviderOptionsMap>,
    provider_name: &str,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    provider_options?.get_object(provider_name)
}

#[cfg(feature = "openai-responses")]
fn openai_or_azure_provider_option_object<'a>(
    provider_options: Option<&'a ProviderOptionsMap>,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    provider_option_object(provider_options, "openai")
        .or_else(|| provider_option_object(provider_options, "azure"))
}

#[cfg(feature = "openai-responses")]
fn xai_provider_option_object<'a>(
    provider_options: Option<&'a ProviderOptionsMap>,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    provider_option_object(provider_options, "xai")
}

#[cfg(feature = "openai-responses")]
fn openai_or_azure_provider_option_item_id(
    provider_options: Option<&ProviderOptionsMap>,
) -> Option<String> {
    openai_or_azure_provider_option_object(provider_options)
        .and_then(|options| options.get("itemId").or_else(|| options.get("item_id")))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

#[cfg(feature = "openai-responses")]
fn xai_provider_option_item_id(provider_options: Option<&ProviderOptionsMap>) -> Option<String> {
    xai_provider_option_object(provider_options)
        .and_then(|options| options.get("itemId").or_else(|| options.get("item_id")))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

#[cfg(feature = "openai-responses")]
fn openai_or_azure_assistant_tool_call_item_id(
    provider_options: Option<&ProviderOptionsMap>,
) -> Option<String> {
    openai_or_azure_provider_option_item_id(provider_options)
}

#[cfg(feature = "openai-responses")]
fn openai_approval_id_from_tool_result(output: &crate::types::ToolResultOutput) -> Option<String> {
    output
        .provider_options()
        .get_object("openai")
        .and_then(|options| {
            options
                .get("approvalId")
                .or_else(|| options.get("approval_id"))
        })
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

#[cfg(feature = "openai-responses")]
fn openai_image_detail(provider_options: Option<&ProviderOptionsMap>) -> Option<String> {
    openai_or_azure_provider_option_object(provider_options)
        .and_then(|options| {
            options
                .get("imageDetail")
                .or_else(|| options.get("image_detail"))
        })
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

#[cfg(feature = "openai-responses")]
fn openai_tool_result_file_id(file_id: &crate::types::ToolResultFileId) -> Option<String> {
    file_id
        .preferred_value(&["openai", "azure"])
        .map(|value| value.to_string())
}

#[cfg(feature = "openai-responses")]
fn openai_or_azure_provider_reference_value(
    provider_reference: &ProviderReference,
) -> Result<String, LlmError> {
    provider_reference
        .preferred_value(&["openai", "azure"])
        .map(|value| value.to_string())
        .ok_or_else(|| {
            let available = provider_reference.available_providers();
            let available = if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            };
            LlmError::InvalidParameter(format!(
                "No provider reference found for provider 'openai' or 'azure'. Available providers: {available}"
            ))
        })
}

#[cfg(feature = "openai-responses")]
impl OpenAiResponsesRequestTransformer {
    fn is_xai_request(req: &ChatRequest) -> bool {
        req.provider_options_map.get_object("xai").is_some()
    }

    fn force_reasoning(req: &ChatRequest) -> bool {
        req.provider_options_map
            .get_object("openai")
            .and_then(|m| m.get("forceReasoning").or_else(|| m.get("force_reasoning")))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn is_reasoning_model_id(model: &str) -> bool {
        let m = model.trim().to_ascii_lowercase();
        if m.is_empty() {
            return false;
        }

        m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4")
            || m.starts_with("gpt-5")
            || m.contains("codex")
            || m.contains("computer-use-preview")
    }

    fn system_message_mode(req: &ChatRequest) -> Option<&str> {
        let explicit = req
            .provider_options_map
            .get_object("openai")
            .and_then(|m| {
                m.get("systemMessageMode")
                    .or_else(|| m.get("system_message_mode"))
            })
            .and_then(|v| v.as_str());

        if explicit.is_some() {
            return explicit;
        }

        // Vercel alignment: reasoning models default to developer system messages.
        if Self::force_reasoning(req) || Self::is_reasoning_model_id(&req.common_params.model) {
            return Some("developer");
        }

        None
    }

    fn file_id_prefixes(req: &ChatRequest) -> Option<Vec<String>> {
        req.provider_options_map
            .get_object("openai")
            .and_then(|m| {
                m.get("fileIdPrefixes")
                    .or_else(|| m.get("file_id_prefixes"))
            })
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
    }

    fn is_file_id(data: &str, prefixes: Option<&[String]>) -> bool {
        let Some(prefixes) = prefixes else {
            return false;
        };
        if prefixes.is_empty() {
            return false;
        }
        prefixes.iter().any(|p| data.starts_with(p))
    }

    fn should_include_item_reference(req: &ChatRequest) -> bool {
        // Vercel alignment:
        // - `convertToOpenAIResponsesInput` takes `store` as a parameter.
        // - In Siumai, `store` lives in `providerOptions.openai` (Responses API config).
        // - Default to true when unspecified (matches Vercel fixtures expectations).
        let openai = req.provider_options_map.get_object("openai");

        let store = openai
            .and_then(|m| m.get("store"))
            .and_then(|v| v.as_bool())
            .or_else(|| {
                openai
                    .and_then(|m| m.get("responsesApi").or_else(|| m.get("responses_api")))
                    .and_then(|v| v.as_object())
                    .and_then(|m| m.get("store"))
                    .and_then(|v| v.as_bool())
            });

        store != Some(false)
    }

    fn extend_message(
        req: &ChatRequest,
        msg: &crate::types::ChatMessage,
        state: &mut ResponsesInputConversionState,
        input: &mut Vec<serde_json::Value>,
    ) -> Result<(), LlmError> {
        use crate::types::{ContentPart, MessageContent, MessageRole};
        use siumai_core::standards::tool_name_mapping::create_tool_name_mapping;

        // Tool role message becomes one or many `function_call_output` items (one per tool result).
        if matches!(msg.role, MessageRole::Tool) {
            let store = Self::should_include_item_reference(req);
            let tool_name_mapping = req.tools.as_deref().map(|tools| {
                create_tool_name_mapping(tools, siumai_core::tools::openai::PROVIDER_TOOL_NAMES)
            });
            let tool_name_mapping = tool_name_mapping.unwrap_or_default();

            let mut items: Vec<serde_json::Value> = Vec::new();
            let mut processed_approval_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            if let MessageContent::MultiModal(parts) = &msg.content {
                for part in parts {
                    match part {
                        ContentPart::ToolApprovalResponse {
                            approval_id,
                            approved,
                            reason,
                            ..
                        } => {
                            if !processed_approval_ids.insert(approval_id.clone()) {
                                continue;
                            }

                            if store {
                                items.push(serde_json::json!({
                                    "type": "item_reference",
                                    "id": approval_id,
                                }));
                            }

                            items.push(serde_json::json!({
                                "type": "mcp_approval_response",
                                "approval_request_id": approval_id,
                                "approve": approved,
                                "reason": reason,
                            }));
                        }
                        ContentPart::ToolResult {
                            tool_call_id,
                            tool_name,
                            output,
                            provider_metadata: _,
                            ..
                        } => {
                            // Vercel parity: skip execution-denied tool results that carry an approval id.
                            if matches!(
                                output,
                                crate::types::ToolResultOutput::ExecutionDenied { .. }
                            ) && openai_approval_id_from_tool_result(output).is_some()
                            {
                                continue;
                            }

                            let resolved_tool_name =
                                tool_name_mapping.to_provider_tool_name(tool_name);

                            // Vercel parity: provider tool outputs use dedicated output item types.
                            if resolved_tool_name == "local_shell"
                                && let crate::types::ToolResultOutput::Json { value, .. } = output
                                && let Some(s) = value.get("output").and_then(|v| v.as_str())
                            {
                                items.push(serde_json::json!({
                                    "type": "local_shell_call_output",
                                    "call_id": tool_call_id,
                                    "output": s,
                                }));
                                continue;
                            }

                            if resolved_tool_name == "shell"
                                && let crate::types::ToolResultOutput::Json { value, .. } = output
                                && let Some(arr) = value.get("output").and_then(|v| v.as_array())
                            {
                                let mapped: Vec<serde_json::Value> = arr
                                    .iter()
                                    .filter_map(|item| {
                                        let stdout = item.get("stdout")?.clone();
                                        let stderr = item.get("stderr")?.clone();
                                        let outcome = item.get("outcome")?.as_object()?;
                                        let outcome_type = outcome.get("type")?.as_str()?;
                                        let mapped_outcome = match outcome_type {
                                            "timeout" => serde_json::json!({ "type": "timeout" }),
                                            "exit" => serde_json::json!({
                                                "type": "exit",
                                                "exit_code": outcome.get("exitCode").or_else(|| outcome.get("exit_code"))?.clone()
                                            }),
                                            _ => return None,
                                        };
                                        Some(serde_json::json!({
                                            "stdout": stdout,
                                            "stderr": stderr,
                                            "outcome": mapped_outcome,
                                        }))
                                    })
                                    .collect();

                                items.push(serde_json::json!({
                                    "type": "shell_call_output",
                                    "call_id": tool_call_id,
                                    "output": mapped,
                                }));
                                continue;
                            }

                            if resolved_tool_name == "apply_patch"
                                && let crate::types::ToolResultOutput::Json { value, .. } = output
                                && let Some(status) = value.get("status")
                            {
                                let output_text = value
                                    .get("output")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                items.push(serde_json::json!({
                                    "type": "apply_patch_call_output",
                                    "call_id": tool_call_id,
                                    "status": status,
                                    "output": output_text,
                                }));
                                continue;
                            }

                            // OpenAI Responses expects `output` (string or output content list). Keep it stable by
                            // sending a string form for simple outputs and a list for multipart outputs.
                            let output_value: serde_json::Value = match output {
                                crate::types::ToolResultOutput::Text { value, .. } => {
                                    serde_json::Value::String(value.clone())
                                }
                                crate::types::ToolResultOutput::Json { value, .. } => {
                                    serde_json::Value::String(
                                        serde_json::to_string(value).unwrap_or_default(),
                                    )
                                }
                                crate::types::ToolResultOutput::ErrorText { value, .. } => {
                                    serde_json::Value::String(value.clone())
                                }
                                crate::types::ToolResultOutput::ErrorJson { value, .. } => {
                                    serde_json::Value::String(
                                        serde_json::to_string(value).unwrap_or_default(),
                                    )
                                }
                                crate::types::ToolResultOutput::ExecutionDenied {
                                    reason, ..
                                } => serde_json::Value::String(
                                    reason
                                        .clone()
                                        .unwrap_or_else(|| "Execution denied".to_string()),
                                ),
                                crate::types::ToolResultOutput::Content { value, .. } => {
                                    let mut out: Vec<serde_json::Value> = Vec::new();
                                    for part in value {
                                        match part {
                                            crate::types::ToolResultContentPart::Text {
                                                text,
                                                ..
                                            } => {
                                                out.push(serde_json::json!({
                                                    "type": "input_text",
                                                    "text": text,
                                                }));
                                            }
                                            crate::types::ToolResultContentPart::ImageData {
                                                data,
                                                media_type,
                                                provider_options,
                                            } => {
                                                let media_type = if media_type == "image/*" {
                                                    "image/jpeg"
                                                } else {
                                                    media_type.as_str()
                                                };
                                                let mut image = serde_json::json!({
                                                    "type": "input_image",
                                                    "image_url": format!(
                                                        "data:{};base64,{}",
                                                        media_type, data
                                                    ),
                                                });
                                                if let Some(provider_detail) =
                                                    openai_image_detail(Some(provider_options))
                                                {
                                                    image["detail"] =
                                                        serde_json::json!(provider_detail);
                                                }
                                                out.push(image);
                                            }
                                            crate::types::ToolResultContentPart::ImageUrl {
                                                url,
                                                provider_options,
                                            } => {
                                                let mut image = serde_json::json!({
                                                    "type": "input_image",
                                                    "image_url": url,
                                                });
                                                if let Some(provider_detail) =
                                                    openai_image_detail(Some(provider_options))
                                                {
                                                    image["detail"] =
                                                        serde_json::json!(provider_detail);
                                                }
                                                out.push(image);
                                            }
                                            crate::types::ToolResultContentPart::ImageFileId {
                                                file_id,
                                                provider_options,
                                            } => {
                                                let Some(file_id) =
                                                    openai_tool_result_file_id(file_id)
                                                else {
                                                    continue;
                                                };
                                                let mut image = serde_json::json!({
                                                    "type": "input_image",
                                                    "file_id": file_id,
                                                });
                                                if let Some(provider_detail) =
                                                    openai_image_detail(Some(provider_options))
                                                {
                                                    image["detail"] =
                                                        serde_json::json!(provider_detail);
                                                }
                                                out.push(image);
                                            }
                                            crate::types::ToolResultContentPart::FileData {
                                                data,
                                                media_type,
                                                filename,
                                                ..
                                            } => {
                                                let filename = filename
                                                    .clone()
                                                    .unwrap_or_else(|| "data".to_string());
                                                out.push(serde_json::json!({
                                                    "type": "input_file",
                                                    "filename": filename,
                                                    "file_data": format!(
                                                        "data:{};base64,{}",
                                                        media_type, data
                                                    ),
                                                }));
                                            }
                                            crate::types::ToolResultContentPart::FileUrl {
                                                url,
                                                ..
                                            } => {
                                                out.push(serde_json::json!({
                                                    "type": "input_file",
                                                    "file_url": url,
                                                }));
                                            }
                                            crate::types::ToolResultContentPart::FileId {
                                                file_id,
                                                ..
                                            } => {
                                                let Some(file_id) =
                                                    openai_tool_result_file_id(file_id)
                                                else {
                                                    continue;
                                                };
                                                out.push(serde_json::json!({
                                                    "type": "input_file",
                                                    "file_id": file_id,
                                                }));
                                            }
                                            crate::types::ToolResultContentPart::Custom {
                                                ..
                                            } => {
                                                out.push(serde_json::json!({
                                                    "type": "input_text",
                                                    "text": "[Custom tool content]",
                                                }));
                                            }
                                        }
                                    }

                                    serde_json::Value::Array(out)
                                }
                            };

                            items.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_call_id,
                                "output": output_value,
                            }));
                        }
                        _ => {}
                    }
                }
            }

            if items.is_empty() {
                return Err(LlmError::InvalidInput(
                    "Tool message missing tool result".into(),
                ));
            }

            input.extend(items);
            return Ok(());
        }

        let store = Self::should_include_item_reference(req);
        let file_id_prefixes = Self::file_id_prefixes(req);
        let file_id_prefixes = file_id_prefixes.as_deref();

        // Vercel alignment: optionally remove system messages for some models.
        if matches!(msg.role, MessageRole::System)
            && matches!(Self::system_message_mode(req), Some("remove"))
        {
            return Ok(());
        }

        // Assistant messages (Vercel-aligned: expand to message + tool call items).
        if matches!(msg.role, MessageRole::Assistant) {
            let tool_name_mapping = req.tools.as_deref().map(|tools| {
                create_tool_name_mapping(tools, siumai_core::tools::openai::PROVIDER_TOOL_NAMES)
            });
            let tool_name_mapping = tool_name_mapping.unwrap_or_default();

            #[allow(unreachable_patterns)]
            match &msg.content {
                MessageContent::Text(text) => {
                    let openai_message_item_id =
                        openai_or_azure_provider_option_item_id(Some(&msg.provider_options));
                    let xai_message_item_id = if Self::is_xai_request(req) {
                        xai_provider_option_item_id(Some(&msg.provider_options))
                    } else {
                        None
                    };
                    let message_item_id = openai_message_item_id
                        .clone()
                        .or_else(|| xai_message_item_id.clone())
                        .or_else(|| msg.metadata.id.clone());
                    let message_phase =
                        provider_option_object(Some(&msg.provider_options), "openai")
                            .or_else(|| {
                                provider_option_object(Some(&msg.provider_options), "azure")
                            })
                            .and_then(|options| options.get("phase"))
                            .and_then(|value| value.as_str());

                    if store
                        && (openai_message_item_id.is_some()
                            || (xai_message_item_id.is_none() && msg.metadata.id.is_some()))
                    {
                        input.push(serde_json::json!({
                            "type": "item_reference",
                            "id": message_item_id.clone().unwrap(),
                        }));
                        return Ok(());
                    }

                    let mut api_message = if Self::is_xai_request(req) {
                        serde_json::json!({
                            "role": "assistant",
                            "content": text,
                        })
                    } else {
                        serde_json::json!({
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": text }],
                        })
                    };
                    if let Some(id) = message_item_id {
                        api_message["id"] = serde_json::json!(id);
                    }
                    if let Some(phase) = message_phase {
                        api_message["phase"] = serde_json::json!(phase);
                    }
                    input.push(api_message);
                    return Ok(());
                }
                MessageContent::MultiModal(parts) => {
                    if Self::is_xai_request(req) {
                        let message_item_id =
                            xai_provider_option_item_id(Some(&msg.provider_options));

                        for part in parts {
                            match part {
                                ContentPart::Text {
                                    text,
                                    provider_options,
                                    ..
                                } => {
                                    let item_id =
                                        xai_provider_option_item_id(Some(provider_options))
                                            .or_else(|| message_item_id.clone());
                                    let mut api_message = serde_json::json!({
                                        "role": "assistant",
                                        "content": text,
                                    });
                                    if let Some(id) = item_id {
                                        api_message["id"] = serde_json::json!(id);
                                    }
                                    input.push(api_message);
                                }
                                ContentPart::ToolCall {
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    provider_executed,
                                    provider_options,
                                    ..
                                } => {
                                    if provider_executed == &Some(true) {
                                        continue;
                                    }

                                    let item_id =
                                        xai_provider_option_item_id(Some(provider_options))
                                            .unwrap_or_else(|| tool_call_id.clone());
                                    input.push(serde_json::json!({
                                        "type": "function_call",
                                        "id": item_id,
                                        "call_id": tool_call_id,
                                        "name": tool_name,
                                        "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                                        "status": "completed",
                                    }));
                                }
                                ContentPart::ToolResult { .. }
                                | ContentPart::Reasoning { .. }
                                | ContentPart::ToolApprovalResponse { .. }
                                | ContentPart::ToolApprovalRequest { .. }
                                | ContentPart::ReasoningFile { .. }
                                | ContentPart::Image { .. }
                                | ContentPart::Audio { .. }
                                | ContentPart::File { .. }
                                | ContentPart::Source { .. }
                                | ContentPart::Custom { .. } => {}
                            }
                        }

                        return Ok(());
                    }

                    let has_openai_referenceable_parts = parts.iter().any(|part| match part {
                        ContentPart::ToolCall {
                            provider_options, ..
                        } => openai_or_azure_assistant_tool_call_item_id(Some(provider_options))
                            .is_some(),
                        ContentPart::ToolResult {
                            provider_options, ..
                        } => openai_or_azure_provider_option_item_id(Some(provider_options))
                            .is_some(),
                        _ => false,
                    });

                    // Vercel parity: prefer item references when IDs are provided and store is enabled.
                    if store && (msg.metadata.id.is_some() || has_openai_referenceable_parts) {
                        let mut refs: Vec<serde_json::Value> = Vec::new();
                        if let Some(id) = msg.metadata.id.clone() {
                            refs.push(serde_json::json!({ "type": "item_reference", "id": id }));
                        }

                        for part in parts {
                            let item_id = match part {
                                ContentPart::ToolCall {
                                    provider_options, ..
                                } => openai_or_azure_assistant_tool_call_item_id(Some(
                                    provider_options,
                                )),
                                ContentPart::ToolResult {
                                    provider_options, ..
                                } => {
                                    openai_or_azure_provider_option_item_id(Some(provider_options))
                                }
                                _ => None,
                            };
                            if let Some(id) = item_id {
                                refs.push(
                                    serde_json::json!({ "type": "item_reference", "id": id }),
                                );
                            }
                        }

                        if !refs.is_empty() {
                            input.extend(refs);
                            return Ok(());
                        }
                    }

                    let mut content_parts: Vec<serde_json::Value> = Vec::new();

                    let flush_assistant =
                        |out: &mut Vec<serde_json::Value>,
                         content_parts: &mut Vec<serde_json::Value>| {
                            if content_parts.is_empty() {
                                return;
                            }
                            let parts = std::mem::take(content_parts);
                            out.push(serde_json::json!({
                                "role": "assistant",
                                "content": parts,
                            }));
                        };

                    for part in parts {
                        match part {
                            ContentPart::Text { text, .. } => {
                                content_parts.push(
                                    serde_json::json!({ "type": "output_text", "text": text }),
                                );
                            }
                            ContentPart::ToolCall {
                                tool_call_id,
                                tool_name,
                                arguments,
                                provider_executed,
                                provider_options,
                                ..
                            } => {
                                if provider_executed == &Some(true) {
                                    // Provider-executed tool calls are not sent back to the API.
                                    // Flush any accumulated assistant text to preserve ordering.
                                    flush_assistant(input, &mut content_parts);
                                    continue;
                                }

                                flush_assistant(input, &mut content_parts);

                                let resolved_tool_name =
                                    tool_name_mapping.to_provider_tool_name(tool_name);
                                let xai_item_id = if Self::is_xai_request(req) {
                                    xai_provider_option_item_id(Some(provider_options))
                                } else {
                                    None
                                };
                                let item_id = openai_or_azure_assistant_tool_call_item_id(Some(
                                    provider_options,
                                ))
                                .or_else(|| xai_item_id.clone());

                                if resolved_tool_name == "local_shell" {
                                    let action =
                                        arguments.get("action").cloned().unwrap_or_default();
                                    let mut mapped_action = serde_json::Map::new();
                                    if let Some(obj) = action.as_object() {
                                        if let Some(v) = obj.get("type") {
                                            mapped_action.insert("type".to_string(), v.clone());
                                        }
                                        if let Some(v) = obj.get("command") {
                                            mapped_action.insert("command".to_string(), v.clone());
                                        }
                                        if let Some(v) =
                                            obj.get("timeoutMs").or_else(|| obj.get("timeout_ms"))
                                        {
                                            mapped_action
                                                .insert("timeout_ms".to_string(), v.clone());
                                        }
                                        if let Some(v) = obj.get("user") {
                                            mapped_action.insert("user".to_string(), v.clone());
                                        }
                                        if let Some(v) = obj
                                            .get("workingDirectory")
                                            .or_else(|| obj.get("working_directory"))
                                        {
                                            mapped_action
                                                .insert("working_directory".to_string(), v.clone());
                                        }
                                        if let Some(v) = obj.get("env") {
                                            mapped_action.insert("env".to_string(), v.clone());
                                        }
                                    }

                                    let mut call = serde_json::json!({
                                        "type": "local_shell_call",
                                        "call_id": tool_call_id,
                                        "action": serde_json::Value::Object(mapped_action),
                                    });
                                    if let Some(id) = item_id.as_deref() {
                                        call["id"] = serde_json::json!(id);
                                    } else if Self::is_xai_request(req) {
                                        call["id"] = serde_json::json!(tool_call_id);
                                    }
                                    input.push(call);
                                    continue;
                                }

                                if resolved_tool_name == "shell" {
                                    let action =
                                        arguments.get("action").cloned().unwrap_or_default();
                                    let mut mapped_action = serde_json::Map::new();
                                    if let Some(obj) = action.as_object() {
                                        if let Some(v) = obj.get("commands") {
                                            mapped_action.insert("commands".to_string(), v.clone());
                                        }
                                        if let Some(v) =
                                            obj.get("timeoutMs").or_else(|| obj.get("timeout_ms"))
                                        {
                                            mapped_action
                                                .insert("timeout_ms".to_string(), v.clone());
                                        }
                                        if let Some(v) = obj
                                            .get("maxOutputLength")
                                            .or_else(|| obj.get("max_output_length"))
                                        {
                                            mapped_action
                                                .insert("max_output_length".to_string(), v.clone());
                                        }
                                    }

                                    let mut call = serde_json::json!({
                                        "type": "shell_call",
                                        "call_id": tool_call_id,
                                        "status": "completed",
                                        "action": serde_json::Value::Object(mapped_action),
                                    });
                                    if let Some(id) = item_id.as_deref() {
                                        call["id"] = serde_json::json!(id);
                                    } else if Self::is_xai_request(req) {
                                        call["id"] = serde_json::json!(tool_call_id);
                                    }
                                    input.push(call);
                                    continue;
                                }

                                let mut call = serde_json::json!({
                                    "type": "function_call",
                                    "call_id": tool_call_id,
                                    "name": resolved_tool_name,
                                    "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                                });
                                if let Some(id) = item_id.as_deref() {
                                    call["id"] = serde_json::json!(id);
                                } else if Self::is_xai_request(req) {
                                    call["id"] = serde_json::json!(tool_call_id);
                                }
                                if Self::is_xai_request(req) {
                                    call["status"] = serde_json::json!("completed");
                                }
                                input.push(call);
                            }
                            ContentPart::ToolResult {
                                tool_call_id,
                                provider_options,
                                provider_metadata: _,
                                provider_executed,
                                ..
                            } => {
                                flush_assistant(input, &mut content_parts);

                                // Assistant tool results are typically provider-executed and stored.
                                if store
                                    && (provider_executed == &Some(true)
                                        || provider_executed.is_none())
                                {
                                    let item_id = openai_or_azure_provider_option_item_id(Some(
                                        provider_options,
                                    ))
                                    .unwrap_or_else(|| tool_call_id.clone());

                                    input.push(serde_json::json!({
                                        "type": "item_reference",
                                        "id": item_id,
                                    }));
                                }
                            }
                            ContentPart::Reasoning {
                                text,
                                provider_options,
                                ..
                            } => {
                                flush_assistant(input, &mut content_parts);

                                let openai_options =
                                    openai_or_azure_provider_option_object(Some(provider_options));

                                let item_id = openai_options
                                    .and_then(|m| m.get("itemId").or_else(|| m.get("item_id")))
                                    .and_then(|v| v.as_str());

                                let encrypted = openai_options.and_then(|m| {
                                    m.get("reasoningEncryptedContent")
                                        .or_else(|| m.get("reasoning_encrypted_content"))
                                });

                                if store {
                                    let Some(id) = item_id else {
                                        // Vercel parity: non-OpenAI reasoning parts are not supported.
                                        continue;
                                    };

                                    if state.reasoning_item_seen.insert(id.to_string()) {
                                        input.push(serde_json::json!({
                                            "type": "item_reference",
                                            "id": id,
                                        }));
                                    }
                                    continue;
                                }

                                let Some(id) = item_id else {
                                    if let Some(enc) = encrypted
                                        && !enc.is_null()
                                    {
                                        let summary = if text.is_empty() {
                                            Vec::new()
                                        } else {
                                            vec![serde_json::json!({
                                                "type": "summary_text",
                                                "text": text,
                                            })]
                                        };
                                        input.push(serde_json::json!({
                                            "type": "reasoning",
                                            "encrypted_content": enc,
                                            "summary": summary,
                                        }));
                                    }
                                    continue;
                                };

                                let idx = state.reasoning_item_index.get(id).copied();
                                if let Some(idx) = idx {
                                    if !text.is_empty()
                                        && let Some(obj) =
                                            input.get_mut(idx).and_then(|v| v.as_object_mut())
                                    {
                                        let arr = obj
                                            .entry("summary")
                                            .or_insert_with(|| serde_json::Value::Array(vec![]))
                                            .as_array_mut();
                                        if let Some(arr) = arr {
                                            arr.push(serde_json::json!({
                                                "type": "summary_text",
                                                "text": text,
                                            }));
                                        }
                                    }

                                    // Vercel parity: only overwrite when the provider option is not nullish.
                                    if let Some(enc) = encrypted
                                        && !enc.is_null()
                                        && let Some(obj) =
                                            input.get_mut(idx).and_then(|v| v.as_object_mut())
                                    {
                                        obj.insert("encrypted_content".to_string(), enc.clone());
                                    }

                                    continue;
                                }

                                let mut obj = serde_json::Map::new();
                                obj.insert("type".to_string(), serde_json::json!("reasoning"));
                                obj.insert("id".to_string(), serde_json::json!(id));

                                if let Some(enc) = encrypted {
                                    obj.insert("encrypted_content".to_string(), enc.clone());
                                }

                                let summary = if text.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![serde_json::json!({
                                        "type": "summary_text",
                                        "text": text,
                                    })]
                                };
                                obj.insert(
                                    "summary".to_string(),
                                    serde_json::Value::Array(summary),
                                );

                                let idx = input.len();
                                input.push(serde_json::Value::Object(obj));
                                state.reasoning_item_index.insert(id.to_string(), idx);
                            }
                            ContentPart::Custom {
                                kind,
                                provider_options,
                                ..
                            } => {
                                flush_assistant(input, &mut content_parts);

                                if kind != "openai.compaction" {
                                    continue;
                                }

                                let openai_options =
                                    openai_or_azure_provider_option_object(Some(provider_options));

                                let item_id = openai_options
                                    .and_then(|options| {
                                        options.get("itemId").or_else(|| options.get("item_id"))
                                    })
                                    .and_then(|value| value.as_str());

                                if store {
                                    if let Some(id) = item_id {
                                        input.push(serde_json::json!({
                                            "type": "item_reference",
                                            "id": id,
                                        }));
                                    }
                                    continue;
                                }

                                let Some(id) = item_id else {
                                    continue;
                                };

                                let encrypted_content = openai_options.and_then(|options| {
                                    options
                                        .get("encryptedContent")
                                        .or_else(|| options.get("encrypted_content"))
                                });

                                let mut compaction = serde_json::Map::new();
                                compaction
                                    .insert("type".to_string(), serde_json::json!("compaction"));
                                compaction.insert("id".to_string(), serde_json::json!(id));
                                if let Some(encrypted_content) = encrypted_content {
                                    compaction.insert(
                                        "encrypted_content".to_string(),
                                        encrypted_content.clone(),
                                    );
                                }
                                input.push(serde_json::Value::Object(compaction));
                            }
                            ContentPart::ToolApprovalResponse { .. }
                            | ContentPart::ToolApprovalRequest { .. }
                            | ContentPart::ReasoningFile { .. }
                            | ContentPart::Image { .. }
                            | ContentPart::Audio { .. }
                            | ContentPart::File { .. }
                            | ContentPart::Source { .. } => {}
                        }
                    }

                    flush_assistant(input, &mut content_parts);
                    return Ok(());
                }
                _ => {
                    input.push(serde_json::json!({
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": msg.content.all_text() }],
                    }));
                    return Ok(());
                }
            }
        }

        // Base message with role (system/user/developer)
        let role = match msg.role {
            MessageRole::System => match Self::system_message_mode(req) {
                Some("developer") => "developer",
                _ => "system",
            },
            MessageRole::Developer => "developer",
            MessageRole::User => "user",
            MessageRole::Tool => "user",
            MessageRole::Assistant => "assistant",
        };
        let mut api_message = serde_json::json!({ "role": role });

        // Default content handling
        #[allow(unreachable_patterns)]
        match &msg.content {
            MessageContent::Text(text) => {
                if role == "system" || role == "developer" {
                    api_message["content"] = serde_json::Value::String(text.clone());
                } else {
                    api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                        "type": "input_text",
                        "text": text
                    })]);
                }
            }
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for (part_index, part) in parts.iter().enumerate() {
                    match part {
                        ContentPart::Text { text, .. } => {
                            content_parts
                                .push(serde_json::json!({ "type": "input_text", "text": text }));
                        }
                        ContentPart::Image {
                            source,
                            detail,
                            provider_options,
                            ..
                        } => {
                            // Responses API prefers `input_image` items.
                            let mut image_part = match source {
                                FilePartSource::Media(MediaSource::Url { url }) => {
                                    serde_json::json!({
                                        "type": "input_image",
                                        "image_url": url,
                                    })
                                }
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    serde_json::json!({
                                        "type": "input_image",
                                        "image_url": format!("data:image/jpeg;base64,{}", data),
                                    })
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    serde_json::json!({
                                        "type": "input_image",
                                        "image_url": format!("data:image/jpeg;base64,{}", encoded),
                                    })
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    serde_json::json!({
                                        "type": "input_image",
                                        "file_id": openai_or_azure_provider_reference_value(provider_reference)?,
                                    })
                                }
                            };

                            let provider_detail = openai_image_detail(Some(provider_options));
                            if let Some(detail) = provider_detail {
                                image_part["detail"] = serde_json::json!(detail);
                            } else if matches!(
                                detail,
                                Some(
                                    crate::types::ImageDetail::Low
                                        | crate::types::ImageDetail::High
                                )
                            ) {
                                image_part["detail"] = serde_json::json!(detail.clone().unwrap());
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio {
                            source, media_type, ..
                        } => {
                            // Responses API input does not currently accept audio inside message content.
                            // Keep a stable fallback representation.
                            let hint = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    format!("[Audio: {}]", url)
                                }
                                crate::types::chat::MediaSource::Base64 { .. }
                                | crate::types::chat::MediaSource::Binary { .. } => {
                                    let mt =
                                        media_type.clone().unwrap_or_else(|| "audio".to_string());
                                    format!("[Audio: {mt}]")
                                }
                            };
                            content_parts
                                .push(serde_json::json!({ "type": "input_text", "text": hint }));
                        }
                        ContentPart::File {
                            source,
                            media_type,
                            filename,
                            provider_options,
                            ..
                        } => {
                            // Responses API file support
                            if media_type.starts_with("image/") {
                                let media_type = if media_type == "image/*" {
                                    "image/jpeg"
                                } else {
                                    media_type.as_str()
                                };

                                match source {
                                    FilePartSource::Media(MediaSource::Url { url }) => {
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "image_url": url,
                                        });

                                        // Vercel parity: image detail can be specified via provider metadata.
                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                    FilePartSource::Media(MediaSource::Base64 { data }) => {
                                        if Self::is_file_id(data, file_id_prefixes) {
                                            let mut image_part = serde_json::json!({
                                                "type": "input_image",
                                                "file_id": data,
                                            });

                                            let provider_detail =
                                                openai_image_detail(Some(provider_options));
                                            if let Some(provider_detail) = provider_detail {
                                                image_part["detail"] =
                                                    serde_json::json!(provider_detail);
                                            }

                                            content_parts.push(image_part);
                                        } else {
                                            let mut image_part = serde_json::json!({
                                                "type": "input_image",
                                                "image_url": format!("data:{};base64,{}", media_type, data),
                                            });

                                            let provider_detail =
                                                openai_image_detail(Some(provider_options));
                                            if let Some(provider_detail) = provider_detail {
                                                image_part["detail"] =
                                                    serde_json::json!(provider_detail);
                                            }

                                            content_parts.push(image_part);
                                        }
                                    }
                                    FilePartSource::Media(MediaSource::Binary { data }) => {
                                        let encoded =
                                            base64::engine::general_purpose::STANDARD.encode(data);
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "image_url": format!("data:{};base64,{}", media_type, encoded),
                                        });

                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                    FilePartSource::ProviderReference { provider_reference } => {
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "file_id": openai_or_azure_provider_reference_value(provider_reference)?,
                                        });

                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                }
                            } else if media_type == "application/pdf" {
                                match source {
                                    FilePartSource::Media(MediaSource::Url { url }) => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "file_url": url,
                                        }));
                                    }
                                    FilePartSource::Media(MediaSource::Base64 { data }) => {
                                        if Self::is_file_id(data, file_id_prefixes) {
                                            content_parts.push(serde_json::json!({
                                                "type": "input_file",
                                                "file_id": data,
                                            }));
                                        } else {
                                            let filename = filename.clone().unwrap_or_else(|| {
                                                format!("part-{}.pdf", part_index)
                                            });
                                            content_parts.push(serde_json::json!({
                                                "type": "input_file",
                                                "filename": filename,
                                                "file_data": format!("data:application/pdf;base64,{}", data),
                                            }));
                                        }
                                    }
                                    FilePartSource::Media(MediaSource::Binary { data }) => {
                                        let encoded =
                                            base64::engine::general_purpose::STANDARD.encode(data);
                                        let filename = filename
                                            .clone()
                                            .unwrap_or_else(|| format!("part-{}.pdf", part_index));
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": filename,
                                            "file_data": format!("data:application/pdf;base64,{}", encoded),
                                        }));
                                    }
                                    FilePartSource::ProviderReference { provider_reference } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "file_id": openai_or_azure_provider_reference_value(provider_reference)?,
                                        }));
                                    }
                                }
                            } else {
                                return Err(LlmError::InvalidParameter(format!(
                                    "file part media type {}",
                                    media_type
                                )));
                            }
                        }
                        ContentPart::ReasoningFile {
                            source,
                            media_type,
                            provider_options,
                            ..
                        } => {
                            if media_type.starts_with("image/") {
                                let media_type = if media_type == "image/*" {
                                    "image/jpeg"
                                } else {
                                    media_type.as_str()
                                };

                                match source {
                                    crate::types::chat::MediaSource::Url { url } => {
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "image_url": url,
                                        });

                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                    crate::types::chat::MediaSource::Base64 { data } => {
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "image_url": format!("data:{};base64,{}", media_type, data),
                                        });

                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                    crate::types::chat::MediaSource::Binary { data } => {
                                        let encoded =
                                            base64::engine::general_purpose::STANDARD.encode(data);
                                        let mut image_part = serde_json::json!({
                                            "type": "input_image",
                                            "image_url": format!("data:{};base64,{}", media_type, encoded),
                                        });

                                        let provider_detail =
                                            openai_image_detail(Some(provider_options));
                                        if let Some(provider_detail) = provider_detail {
                                            image_part["detail"] =
                                                serde_json::json!(provider_detail);
                                        }

                                        content_parts.push(image_part);
                                    }
                                }
                            } else if media_type == "application/pdf" {
                                match source {
                                    crate::types::chat::MediaSource::Url { url } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "file_url": url,
                                        }));
                                    }
                                    crate::types::chat::MediaSource::Base64 { data } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": format!("part-{}.pdf", part_index),
                                            "file_data": format!("data:application/pdf;base64,{}", data),
                                        }));
                                    }
                                    crate::types::chat::MediaSource::Binary { data } => {
                                        let encoded =
                                            base64::engine::general_purpose::STANDARD.encode(data);
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": format!("part-{}.pdf", part_index),
                                            "file_data": format!("data:application/pdf;base64,{}", encoded),
                                        }));
                                    }
                                }
                            } else {
                                return Err(LlmError::InvalidParameter(format!(
                                    "reasoning file part media type {}",
                                    media_type
                                )));
                            }
                        }
                        ContentPart::ToolCall { .. } => {
                            if let ContentPart::ToolCall {
                                provider_executed, ..
                            } = part
                                && provider_executed == &Some(true)
                            {
                                continue;
                            }

                            // Assistant tool calls are represented as `tool_use` content parts with
                            // structured `input` (aligned with the official Responses API semantics).
                            if let ContentPart::ToolCall {
                                tool_call_id,
                                tool_name,
                                arguments,
                                ..
                            } = part
                            {
                                content_parts.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": tool_call_id,
                                    "name": tool_name,
                                    "input": arguments,
                                }));
                            }
                        }
                        ContentPart::ToolResult { .. } => {}
                        ContentPart::Reasoning { text, .. } => {
                            // Reasoning content as text
                            content_parts.push(serde_json::json!({
                                "type": "input_text",
                                "text": format!("<thinking>{}</thinking>", text)
                            }));
                        }
                        ContentPart::Custom { .. } => {}
                        ContentPart::ToolApprovalResponse { .. } => {}
                        ContentPart::ToolApprovalRequest { .. } => {}
                        ContentPart::Source { .. } => {}
                    }
                }

                // Vercel alignment: if a message only contained provider-executed tool calls
                // (or other skipped parts), omit the message entirely.
                if content_parts.is_empty() {
                    return Ok(());
                }

                api_message["content"] = serde_json::Value::Array(content_parts);
            }
            _ => {
                // Responses API does not define an `input_json` content part; serialize as text.
                let s = msg.content.all_text();
                if role == "system" || role == "developer" {
                    api_message["content"] = serde_json::Value::String(s);
                } else {
                    api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                        "type": "input_text",
                        "text": s
                    })]);
                }
            }
        }

        input.push(api_message);
        Ok(())
    }
}

#[cfg(feature = "openai-responses")]
impl RequestTransformer for OpenAiResponsesRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        struct ResponsesHooks;
        impl crate::execution::transformers::request::ProviderRequestHooks for ResponsesHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Build base body
                let mut body = serde_json::json!({
                    "model": req.common_params.model,
                });

                if req.stream {
                    body["stream"] = serde_json::Value::Bool(true);
                }

                // input
                let mut input_items: Vec<serde_json::Value> = Vec::new();
                let mut state = ResponsesInputConversionState::default();
                for m in &req.messages {
                    OpenAiResponsesRequestTransformer::extend_message(
                        req,
                        m,
                        &mut state,
                        &mut input_items,
                    )?;
                }
                body["input"] = serde_json::Value::Array(input_items);

                // tools (flattened)
                if let Some(tools) = &req.tools {
                    let openai_tools =
                        crate::standards::openai::utils::convert_tools_to_responses_format(tools)?;
                    if !openai_tools.is_empty() {
                        body["tools"] = serde_json::Value::Array(openai_tools);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice {
                            if let Some(tool_choice) =
                                crate::standards::openai::utils::convert_responses_tool_choice(
                                    choice,
                                    req.tools.as_deref(),
                                )
                            {
                                body["tool_choice"] = tool_choice;
                            }
                        }
                    }
                }

                // temperature
                if let Some(temp) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(temp);
                }

                // top_p
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }

                // max_output_tokens (prefer max_completion_tokens, fallback to max_tokens)
                if let Some(max_tokens) = req.common_params.max_completion_tokens {
                    body["max_output_tokens"] = serde_json::json!(max_tokens);
                } else if let Some(max_tokens) = req.common_params.max_tokens {
                    body["max_output_tokens"] = serde_json::json!(max_tokens);
                }

                if let Some(fmt) = &req.response_format {
                    let text = body
                        .as_object_mut()
                        .expect("responses request body must be an object")
                        .entry("text".to_string())
                        .or_insert_with(|| serde_json::json!({}));

                    if !text.is_object() {
                        *text = serde_json::json!({});
                    }

                    text.as_object_mut()
                        .expect("responses text entry was normalized to an object")
                        .insert(
                            "format".to_string(),
                            crate::standards::openai::utils::convert_responses_response_format(fmt),
                        );
                }

                Ok(body)
            }

            fn post_process_chat(
                &self,
                req: &crate::types::ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                let Some(xai_options) = req.provider_options_map.get_object("xai") else {
                    return Ok(());
                };

                let Some(body_obj) = body.as_object_mut() else {
                    return Ok(());
                };

                let get_option = |camel_case: &str, snake_case: &str| {
                    xai_options
                        .get(camel_case)
                        .or_else(|| xai_options.get(snake_case))
                };

                let reasoning_effort = get_option("reasoningEffort", "reasoning_effort")
                    .and_then(|value| value.as_str());
                let reasoning_summary = get_option("reasoningSummary", "reasoning_summary")
                    .and_then(|value| value.as_str());

                if reasoning_effort.is_some() || reasoning_summary.is_some() {
                    let reasoning = body_obj
                        .entry("reasoning".to_string())
                        .or_insert_with(|| serde_json::json!({}));
                    if !reasoning.is_object() {
                        *reasoning = serde_json::json!({});
                    }
                    let reasoning_obj = reasoning
                        .as_object_mut()
                        .expect("xai reasoning body was normalized to an object");
                    if let Some(effort) = reasoning_effort {
                        reasoning_obj.insert("effort".to_string(), serde_json::json!(effort));
                    }
                    if let Some(summary) = reasoning_summary {
                        reasoning_obj.insert("summary".to_string(), serde_json::json!(summary));
                    }
                }

                let top_logprobs = get_option("topLogprobs", "top_logprobs").cloned();
                let logprobs = get_option("logprobs", "logprobs").and_then(|value| value.as_bool());
                if let Some(top_logprobs) = top_logprobs {
                    body_obj.insert("top_logprobs".to_string(), top_logprobs);
                    body_obj.insert("logprobs".to_string(), serde_json::json!(true));
                } else if let Some(logprobs) = logprobs {
                    body_obj.insert("logprobs".to_string(), serde_json::json!(logprobs));
                }

                let store = get_option("store", "store").and_then(|value| value.as_bool());
                if store == Some(false) {
                    body_obj.insert("store".to_string(), serde_json::json!(false));
                }

                if let Some(previous_response_id) =
                    get_option("previousResponseId", "previous_response_id")
                        .and_then(|value| value.as_str())
                {
                    body_obj.insert(
                        "previous_response_id".to_string(),
                        serde_json::json!(previous_response_id),
                    );
                }

                let include_value = get_option("include", "include");
                let include_was_explicit_array =
                    include_value.is_some_and(|value| value.is_array());
                let mut include = include_value
                    .and_then(|value| value.as_array())
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.as_str().map(|value| value.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                if store == Some(false)
                    && !include
                        .iter()
                        .any(|value| value == "reasoning.encrypted_content")
                {
                    include.push("reasoning.encrypted_content".to_string());
                }

                if include_was_explicit_array || !include.is_empty() {
                    body_obj.insert("include".to_string(), serde_json::json!(include));
                }

                Ok(())
            }
        }
        let hooks = ResponsesHooks;
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "openai_responses",
            rules: vec![crate::execution::transformers::request::Rule::Range {
                field: "temperature",
                min: 0.0,
                max: 2.0,
                mode: crate::execution::transformers::request::RangeMode::Error,
                message: None,
            }],
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }

    fn transform_rerank(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let mut payload = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents,
        });
        if let Some(instr) = &req.instruction {
            payload["instruction"] = serde_json::json!(instr);
        }
        if let Some(top_n) = req.top_n {
            payload["top_n"] = serde_json::json!(top_n);
        }
        if let Some(rd) = req.return_documents {
            payload["return_documents"] = serde_json::json!(rd);
        }
        if let Some(maxc) = req.max_chunks_per_doc {
            payload["max_chunks_per_doc"] = serde_json::json!(maxc);
        }
        if let Some(over) = req.overlap_tokens {
            payload["overlap_tokens"] = serde_json::json!(over);
        }
        Ok(payload)
    }

    fn transform_moderation(&self, req: &ModerationRequest) -> Result<serde_json::Value, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "omni-moderation-latest".to_string());

        // OpenAI Moderations accepts either string or string[] for `input`.
        // Prefer array when provided in request.
        let input_value = if let Some(arr) = &req.inputs {
            serde_json::Value::Array(
                arr.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            )
        } else {
            serde_json::Value::String(req.input.clone())
        };

        let json = serde_json::json!({ "model": model, "input": input_value });
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use super::super::OpenAiRequestTransformer;
    use super::*;

    #[test]
    fn transform_chat_stream_includes_stream_options_include_usage() {
        use crate::types::ChatMessage;

        let tx = OpenAiRequestTransformer;
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .model("gpt-4o-mini")
            .stream(true)
            .build();

        let body = tx.transform_chat(&req).expect("transform chat");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_stream_omits_chat_completions_stream_options() {
        use crate::types::ChatMessage;

        let tx = OpenAiResponsesRequestTransformer;
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .model("gpt-4.1-mini")
            .stream(true)
            .build();

        let body = tx.transform_chat(&req).expect("transform chat");
        assert_eq!(body["stream"], true);
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_system_message_defaults_to_developer_for_reasoning_models() {
        use crate::types::ChatMessage;

        let tx = OpenAiResponsesRequestTransformer;
        let req = ChatRequest::builder()
            .message(ChatMessage::system("sys").build())
            .message(ChatMessage::user("hi").build())
            .model("o1")
            .stream(false)
            .build();

        let body = tx.transform_chat(&req).expect("transform chat");
        let input = body.get("input").and_then(|v| v.as_array()).expect("input");
        assert_eq!(
            input[0].get("role").and_then(|v| v.as_str()),
            Some("developer")
        );
        assert_eq!(
            input[0].get("content").and_then(|v| v.as_str()),
            Some("sys")
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_structured_output_to_text_format() {
        use crate::types::{ChatMessage, ResponseFormat};

        let tx = OpenAiResponsesRequestTransformer;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
            "additionalProperties": false
        });
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .response_format(
                ResponseFormat::json_schema(schema.clone())
                    .with_name("result")
                    .with_strict(true),
            )
            .model("gpt-4.1-mini")
            .build();

        let body = tx.transform_chat(&req).expect("transform chat");
        assert_eq!(
            body["text"]["format"]["type"],
            serde_json::json!("json_schema")
        );
        assert_eq!(body["text"]["format"]["name"], serde_json::json!("result"));
        assert_eq!(body["text"]["format"]["strict"], serde_json::json!(true));
        assert_eq!(body["text"]["format"]["schema"], schema);
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_preserves_tool_approval_reason() {
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };

        let tx = OpenAiResponsesRequestTransformer;
        let request = ChatRequest::builder()
            .message(ChatMessage {
                role: MessageRole::Tool,
                content: MessageContent::MultiModal(vec![
                    ContentPart::tool_approval_response_with_reason(
                        "apr_1",
                        false,
                        Some("need manual review".to_string()),
                    ),
                ]),
                metadata: MessageMetadata::default(),
                provider_options: crate::types::ProviderOptionsMap::default(),
            })
            .model("gpt-4.1")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        let approval = input
            .iter()
            .find(|item| item.get("type") == Some(&serde_json::json!("mcp_approval_response")))
            .expect("approval response item");
        assert_eq!(approval["approval_request_id"], serde_json::json!("apr_1"));
        assert_eq!(approval["approve"], serde_json::json!(false));
        assert_eq!(approval["reason"], serde_json::json!("need manual review"));
    }

    #[test]
    fn transform_rerank_includes_optional_fields() {
        use crate::types::RerankRequest;

        let tx = OpenAiRequestTransformer;
        let req = RerankRequest::new(
            "bge-reranker".to_string(),
            "rust async".to_string(),
            vec!["doc1".into(), "doc2".into()],
        )
        .with_instruction("rank by semantic relevance".to_string())
        .with_top_n(3)
        .with_return_documents(true)
        .with_max_chunks_per_doc(8)
        .with_overlap_tokens(16);

        let body = tx.transform_rerank(&req).expect("transform rerank");

        assert_eq!(body["model"], "bge-reranker");
        assert_eq!(body["query"], "rust async");
        let docs = body["documents"].as_array().expect("documents array");
        assert_eq!(docs.len(), 2);
        assert_eq!(body["top_n"], 3);
        assert_eq!(body["return_documents"], true);
        assert_eq!(body["max_chunks_per_doc"], 8);
        assert_eq!(body["overlap_tokens"], 16);
        // instruction is optional and may be adapted per provider; if present, assert matches
        assert_eq!(body["instruction"], "rank by semantic relevance");
    }

    #[cfg(all(feature = "structured-messages", feature = "openai-responses"))]
    #[test]
    fn convert_message_json_maps_to_input_text_json_string() {
        use crate::types::ChatRequest;
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Json(serde_json::json!({"a":1})),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };
        let req = ChatRequest::new(vec![]);
        let mut state = super::ResponsesInputConversionState::default();
        let mut items = Vec::new();
        super::OpenAiResponsesRequestTransformer::extend_message(
            &req, &msg, &mut state, &mut items,
        )
        .expect("convert");
        assert_eq!(items.len(), 1);
        let v = &items[0];
        // Expect content array with input_text JSON string
        let content = v
            .get("content")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap();
        assert!(!content.is_empty());
        let first = &content[0];
        assert_eq!(
            first.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "input_text"
        );
        let s = first.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed.get("a").and_then(|x| x.as_i64()).unwrap_or(0), 1);
    }

    #[cfg(all(feature = "structured-messages", feature = "openai-responses"))]
    #[test]
    fn convert_tool_output_json_maps_to_output_string() {
        use crate::types::{
            ChatMessage, ChatRequest, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };
        let msg = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_json(
                "call-1",
                "test_tool",
                serde_json::json!({"r":42}),
            )]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };
        let req = ChatRequest::new(vec![]);
        let mut state = super::ResponsesInputConversionState::default();
        let mut items = Vec::new();
        super::OpenAiResponsesRequestTransformer::extend_message(
            &req, &msg, &mut state, &mut items,
        )
        .expect("convert");
        assert_eq!(items.len(), 1);
        let v = &items[0];
        assert_eq!(
            v.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "function_call_output"
        );
        assert_eq!(
            v.get("call_id").and_then(|x| x.as_str()).unwrap_or(""),
            "call-1"
        );
        let s = v.get("output").and_then(|o| o.as_str()).unwrap_or("");
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed.get("r").and_then(|x| x.as_i64()).unwrap_or(0), 42);

        // Fallback to output text for Text output
        let msg2 = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_text(
                "call-2",
                "test_tool",
                "ok",
            )]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };
        let mut items2 = Vec::new();
        super::OpenAiResponsesRequestTransformer::extend_message(
            &req,
            &msg2,
            &mut state,
            &mut items2,
        )
        .expect("convert");
        assert_eq!(items2.len(), 1);
        let v2 = &items2[0];
        assert_eq!(
            v2.get("output").and_then(|x| x.as_str()).unwrap_or(""),
            "ok"
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_prefers_message_provider_options_over_legacy_metadata_id() {
        let tx = OpenAiResponsesRequestTransformer;
        let mut msg = crate::types::ChatMessage::assistant("done").build();
        msg.metadata.id = Some("legacy_msg".to_string());
        msg.provider_options.insert(
            "openai",
            serde_json::json!({
                "itemId": "provider_msg",
                "phase": "output"
            }),
        );

        let request = ChatRequest::builder()
            .message(msg)
            .provider_option("openai", serde_json::json!({ "store": false }))
            .model("gpt-4.1")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["id"], serde_json::json!("provider_msg"));
        assert_eq!(input[0]["phase"], serde_json::json!("output"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_prefers_image_provider_options_over_legacy_provider_metadata() {
        use crate::types::{
            ChatMessage, ContentPart, ImageDetail, MessageContent, MessageMetadata, MessageRole,
        };
        use std::collections::HashMap;

        let tx = OpenAiResponsesRequestTransformer;

        let mut image_provider_options = crate::types::ProviderOptionsMap::default();
        image_provider_options.insert("openai", serde_json::json!({ "imageDetail": "high" }));

        let mut provider_metadata = HashMap::new();
        provider_metadata.insert(
            "openai".to_string(),
            serde_json::json!({ "imageDetail": "low" }),
        );

        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::Image {
                source: crate::types::chat::FilePartSource::url("https://example.com/image.png"),
                detail: Some(ImageDetail::Low),
                provider_options: image_provider_options,
                provider_metadata: Some(provider_metadata),
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder().message(msg).model("gpt-4.1").build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["detail"], serde_json::json!("high"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_ignores_legacy_image_provider_metadata_without_provider_options() {
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };
        use std::collections::HashMap;

        let tx = OpenAiResponsesRequestTransformer;

        let mut provider_metadata = HashMap::new();
        provider_metadata.insert(
            "openai".to_string(),
            serde_json::json!({ "imageDetail": "low" }),
        );

        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: crate::types::chat::FilePartSource::base64("AAECAw=="),
                media_type: "image/png".to_string(),
                filename: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(provider_metadata),
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder().message(msg).model("gpt-4.1").build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert!(content[0].get("detail").is_none());
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_image_provider_reference_to_file_id() {
        use crate::types::{
            ChatMessage, ContentPart, FilePartSource, MessageContent, MessageMetadata, MessageRole,
            ProviderOptionsMap, ProviderReference,
        };

        let tx = OpenAiResponsesRequestTransformer;
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::Image {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "openai",
                    "file-image",
                )),
                detail: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder().message(msg).model("gpt-4.1").build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["type"], serde_json::json!("input_image"));
        assert_eq!(content[0]["file_id"], serde_json::json!("file-image"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_pdf_provider_reference_to_file_id() {
        use crate::types::{
            ChatMessage, ContentPart, FilePartSource, MessageContent, MessageMetadata, MessageRole,
            ProviderOptionsMap, ProviderReference,
        };

        let tx = OpenAiResponsesRequestTransformer;
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "azure", "file-pdf",
                )),
                media_type: "application/pdf".to_string(),
                filename: Some("doc.pdf".to_string()),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder().message(msg).model("gpt-4.1").build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["type"], serde_json::json!("input_file"));
        assert_eq!(content[0]["file_id"], serde_json::json!("file-pdf"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_ignores_assistant_tool_call_legacy_metadata_item_id() {
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };
        use std::collections::HashMap;

        let tx = OpenAiResponsesRequestTransformer;

        let msg = ChatMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![ContentPart::ToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "weather".to_string(),
                arguments: serde_json::json!({ "city": "Tokyo" }),
                provider_executed: None,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({ "itemId": "fc_legacy_1" }),
                )])),
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder()
            .message(msg)
            .provider_option("openai", serde_json::json!({ "store": true }))
            .model("gpt-4.1")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], serde_json::json!("function_call"));
        assert!(input[0].get("id").is_none());
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_keeps_reasoning_without_item_id_when_encrypted_content_present() {
        let tx = OpenAiResponsesRequestTransformer;
        let mut provider_options = crate::types::ProviderOptionsMap::default();
        provider_options.insert(
            "openai",
            serde_json::json!({
                "reasoningEncryptedContent": "encrypted_content_001"
            }),
        );
        let request = ChatRequest::builder()
            .message(
                crate::types::ChatMessage::assistant_with_content(vec![
                    crate::types::ContentPart::Reasoning {
                        text: "Analyzing the problem step by step".to_string(),
                        provider_options,
                        provider_metadata: None,
                    },
                ])
                .build(),
            )
            .provider_option("openai", serde_json::json!({ "store": false }))
            .model("gpt-4.1-mini")
            .build();

        let body = tx.transform_chat(&request).expect("transform");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], serde_json::json!("reasoning"));
        assert!(input[0].get("id").is_none());
        assert_eq!(
            input[0]["encrypted_content"],
            serde_json::json!("encrypted_content_001")
        );
        assert_eq!(
            input[0]["summary"],
            serde_json::json!([
                {
                    "type": "summary_text",
                    "text": "Analyzing the problem step by step"
                }
            ])
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_custom_compaction_parts() {
        let tx = OpenAiResponsesRequestTransformer;
        let request = ChatRequest::builder()
            .message(
                crate::types::ChatMessage::assistant_with_content(vec![
                    crate::types::ContentPart::custom("openai.compaction").with_provider_option(
                        "openai",
                        serde_json::json!({
                            "type": "compaction",
                            "itemId": "cmp_123",
                            "encryptedContent": "encrypted_state"
                        }),
                    ),
                ])
                .build(),
            )
            .provider_option("openai", serde_json::json!({ "store": false }))
            .model("gpt-4.1-mini")
            .build();

        let body = tx.transform_chat(&request).expect("transform");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], serde_json::json!("compaction"));
        assert_eq!(input[0]["id"], serde_json::json!("cmp_123"));
        assert_eq!(
            input[0]["encrypted_content"],
            serde_json::json!("encrypted_state")
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_custom_compaction_parts_to_item_references_when_store_enabled()
    {
        let tx = OpenAiResponsesRequestTransformer;
        let request = ChatRequest::builder()
            .message(
                crate::types::ChatMessage::assistant_with_content(vec![
                    crate::types::ContentPart::custom("openai.compaction").with_provider_option(
                        "openai",
                        serde_json::json!({
                            "type": "compaction",
                            "itemId": "cmp_456",
                            "encryptedContent": "encrypted_state"
                        }),
                    ),
                ])
                .build(),
            )
            .provider_option("openai", serde_json::json!({ "store": true }))
            .model("gpt-4.1-mini")
            .build();

        let body = tx.transform_chat(&request).expect("transform");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], serde_json::json!("item_reference"));
        assert_eq!(input[0]["id"], serde_json::json!("cmp_456"));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_xai_responses_options() {
        let tx = OpenAiResponsesRequestTransformer;
        let request = ChatRequest::builder()
            .message(crate::types::ChatMessage::user("hi").build())
            .provider_option(
                "xai",
                serde_json::json!({
                    "reasoningEffort": "high",
                    "reasoningSummary": "detailed",
                    "topLogprobs": 3,
                    "previousResponseId": "resp_prev_123",
                    "include": ["file_search_call.results"],
                    "store": false
                }),
            )
            .model("grok-4")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        assert_eq!(
            body["reasoning"],
            serde_json::json!({
                "effort": "high",
                "summary": "detailed"
            })
        );
        assert_eq!(body["top_logprobs"], serde_json::json!(3));
        assert_eq!(body["logprobs"], serde_json::json!(true));
        assert_eq!(
            body["previous_response_id"],
            serde_json::json!("resp_prev_123")
        );
        assert_eq!(body["store"], serde_json::json!(false));
        let include = body["include"].as_array().expect("include array");
        assert!(include.contains(&serde_json::json!("file_search_call.results")));
        assert!(include.contains(&serde_json::json!("reasoning.encrypted_content")));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_xai_store_false_appends_reasoning_encrypted_content_once() {
        let tx = OpenAiResponsesRequestTransformer;
        let request = ChatRequest::builder()
            .message(crate::types::ChatMessage::user("hi").build())
            .provider_option(
                "xai",
                serde_json::json!({
                    "include": [
                        "file_search_call.results",
                        "reasoning.encrypted_content"
                    ],
                    "store": false
                }),
            )
            .model("grok-4")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let include = body["include"].as_array().expect("include array");
        let encrypted_count = include
            .iter()
            .filter(|value| **value == serde_json::json!("reasoning.encrypted_content"))
            .count();
        assert_eq!(encrypted_count, 1);
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_xai_assistant_text_item_id_without_item_reference() {
        let tx = OpenAiResponsesRequestTransformer;
        let mut msg = crate::types::ChatMessage::assistant("done").build();
        msg.provider_options.insert(
            "xai",
            serde_json::json!({
                "itemId": "xai_msg_1"
            }),
        );

        let request = ChatRequest::builder()
            .message(msg)
            .provider_option("xai", serde_json::json!({ "store": true }))
            .model("grok-4")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], serde_json::json!("assistant"));
        assert_eq!(input[0]["content"], serde_json::json!("done"));
        assert_eq!(input[0]["id"], serde_json::json!("xai_msg_1"));
        assert!(input[0].get("type").is_none());
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn responses_transform_chat_maps_xai_assistant_tool_call_ids_and_status() {
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };

        let tx = OpenAiResponsesRequestTransformer;

        let mut tool_call_options = crate::types::ProviderOptionsMap::default();
        tool_call_options.insert(
            "xai",
            serde_json::json!({
                "itemId": "xai_tool_call_1"
            }),
        );

        let msg = ChatMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![ContentPart::ToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "weather".to_string(),
                arguments: serde_json::json!({ "city": "Tokyo" }),
                provider_executed: None,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: tool_call_options,
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let request = ChatRequest::builder()
            .message(msg)
            .provider_option("xai", serde_json::json!({ "store": false }))
            .model("grok-4")
            .build();

        let body = tx.transform_chat(&request).expect("transform chat");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], serde_json::json!("function_call"));
        assert_eq!(input[0]["id"], serde_json::json!("xai_tool_call_1"));
        assert_eq!(input[0]["call_id"], serde_json::json!("call_1"));
        assert_eq!(input[0]["status"], serde_json::json!("completed"));
        assert_eq!(input[0]["name"], serde_json::json!("weather"));
        assert_eq!(
            input[0]["arguments"],
            serde_json::json!("{\"city\":\"Tokyo\"}")
        );
    }
}
