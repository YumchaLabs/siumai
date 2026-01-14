//! Amazon Bedrock Chat Standard (Vercel-aligned).
//!
//! Vercel reference: `repo-ref/ai/packages/amazon-bedrock/src/bedrock-chat-language-model.ts`

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::streaming::{ChatStreamEvent, EventBuilder, JsonEventConverter, StreamStateTracker};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ContentPart, FinishReason, MessageContent,
    ResponseFormat, ResponseMetadata, Tool, ToolChoice, Usage, Warning,
};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct BedrockChatStandard;

impl BedrockChatStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> BedrockChatSpec {
        BedrockChatSpec { provider_id }
    }

    pub fn create_transformers(
        &self,
        provider_id: &str,
        uses_json_response_tool: bool,
    ) -> ChatTransformers {
        ChatTransformers {
            request: Arc::new(BedrockChatRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(BedrockChatResponseTransformer {
                provider_id: provider_id.to_string(),
                uses_json_response_tool,
            }),
            stream: None,
            json: Some(Arc::new(BedrockEventConverter::new(
                provider_id,
                uses_json_response_tool,
            ))),
        }
    }
}

/// ProviderSpec implementation for Amazon Bedrock Converse.
pub struct BedrockChatSpec {
    provider_id: &'static str,
}

impl ProviderSpec for BedrockChatSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        // NOTE: Bedrock normally requires AWS SigV4 signing (or bearer token auth).
        // This crate keeps auth lightweight for fixture alignment. Users can inject
        // signed headers via `ProviderContext.http_extra_headers`.
        if let Some(api_key) = ctx.api_key.as_deref().filter(|v| !v.trim().is_empty()) {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid Bedrock bearer token: {e}"))
                })?,
            );
        }

        for (k, v) in &ctx.http_extra_headers {
            if let (Ok(name), Ok(value)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                HeaderValue::from_str(v),
            ) {
                headers.insert(name, value);
            }
        }

        Ok(headers)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::bedrock::errors::classify_bedrock_http_error(
            self.provider_id,
            status,
            body_text,
            headers,
        )
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        let model = urlencoding::encode(&req.common_params.model);
        let suffix = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        crate::utils::url::join_url(&ctx.base_url, &format!("/model/{model}/{suffix}"))
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let uses_json_response_tool = matches!(
            req.response_format.as_ref(),
            Some(ResponseFormat::Json { .. })
        );
        BedrockChatStandard::new().create_transformers(self.provider_id, uses_json_response_tool)
    }
}

struct BedrockChatRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

impl BedrockChatRequestTransformer {
    fn uses_json_response_tool(req: &ChatRequest) -> bool {
        matches!(
            req.response_format.as_ref(),
            Some(ResponseFormat::Json { .. })
        )
    }

    fn build_tool_config(req: &ChatRequest) -> (Option<serde_json::Value>, Vec<Warning>) {
        let mut warnings: Vec<Warning> = Vec::new();

        let uses_json_tool = Self::uses_json_response_tool(req);
        let mut tools: Vec<Tool> = req.tools.clone().unwrap_or_default();

        if uses_json_tool {
            let schema = match req.response_format.as_ref() {
                Some(ResponseFormat::Json { schema }) => schema.clone(),
                _ => serde_json::json!({ "type": "object" }),
            };
            tools.push(Tool::function(
                "json",
                "Respond with a JSON object.",
                schema,
            ));
        }

        if tools.is_empty() {
            return (None, warnings);
        }

        let tool_choice = if uses_json_tool {
            Some(ToolChoice::Required)
        } else {
            req.tool_choice.clone()
        };

        if matches!(tool_choice, Some(ToolChoice::None)) {
            return (None, warnings);
        }

        let mut bedrock_tools: Vec<serde_json::Value> = Vec::new();
        for t in tools {
            match t {
                Tool::Function { function } => {
                    let mut tool_spec = serde_json::json!({
                        "name": function.name,
                        "inputSchema": { "json": function.parameters },
                    });
                    if !function.description.trim().is_empty() {
                        tool_spec["description"] = serde_json::Value::String(function.description);
                    }
                    bedrock_tools.push(serde_json::json!({ "toolSpec": tool_spec }));
                }
                Tool::ProviderDefined(tool) => {
                    warnings.push(Warning::unsupported_tool(
                        tool.id,
                        Some("provider-defined tools are not yet supported for Amazon Bedrock"),
                    ));
                }
            }
        }

        if bedrock_tools.is_empty() {
            return (None, warnings);
        }

        let mut tool_config = serde_json::json!({ "tools": bedrock_tools });

        if let Some(tc) = tool_choice {
            let mapped = match tc {
                ToolChoice::Auto => serde_json::json!({ "auto": {} }),
                ToolChoice::Required => serde_json::json!({ "any": {} }),
                ToolChoice::Tool { name } => serde_json::json!({ "tool": { "name": name } }),
                ToolChoice::None => serde_json::Value::Null,
            };
            if !mapped.is_null() {
                tool_config["toolChoice"] = mapped;
            }
        }

        (Some(tool_config), warnings)
    }

    fn split_system_messages(messages: &[ChatMessage]) -> (Vec<serde_json::Value>, &[ChatMessage]) {
        let mut system: Vec<serde_json::Value> = Vec::new();
        let mut idx = 0usize;
        while idx < messages.len() {
            match messages[idx].role {
                crate::types::MessageRole::System | crate::types::MessageRole::Developer => {
                    let text = messages[idx].content_text().unwrap_or_default().to_string();
                    system.push(serde_json::json!({ "text": text }));
                    idx += 1;
                }
                _ => break,
            }
        }
        (system, &messages[idx..])
    }

    fn content_parts(message: &ChatMessage) -> Vec<ContentPart> {
        #[allow(unreachable_patterns)]
        match &message.content {
            MessageContent::Text(t) => vec![ContentPart::text(t.clone())],
            MessageContent::MultiModal(parts) => parts.clone(),
            _ => vec![ContentPart::text(message.content.all_text())],
        }
    }

    fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out: Vec<serde_json::Value> = Vec::new();
        let mut i = 0usize;

        while i < messages.len() {
            match messages[i].role {
                crate::types::MessageRole::User | crate::types::MessageRole::Tool => {
                    let mut content: Vec<serde_json::Value> = Vec::new();
                    while i < messages.len()
                        && matches!(
                            messages[i].role,
                            crate::types::MessageRole::User | crate::types::MessageRole::Tool
                        )
                    {
                        let msg = &messages[i];
                        match msg.role {
                            crate::types::MessageRole::User => {
                                for part in Self::content_parts(msg) {
                                    match part {
                                        ContentPart::Text { text, .. } => {
                                            content.push(serde_json::json!({ "text": text }));
                                        }
                                        _ => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request currently supports text only"
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                }
                            }
                            crate::types::MessageRole::Tool => {
                                for part in Self::content_parts(msg) {
                                    if matches!(part, ContentPart::ToolApprovalResponse { .. }) {
                                        continue;
                                    }
                                    if let Some(tr) = part.as_tool_result() {
                                        let text = tr.output.to_string_lossy();
                                        content.push(serde_json::json!({
                                            "toolResult": {
                                                "toolUseId": tr.tool_call_id,
                                                "content": [{ "text": text }],
                                            }
                                        }));
                                    }
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    out.push(serde_json::json!({ "role": "user", "content": content }));
                }
                crate::types::MessageRole::Assistant => {
                    let mut content: Vec<serde_json::Value> = Vec::new();
                    while i < messages.len()
                        && matches!(messages[i].role, crate::types::MessageRole::Assistant)
                    {
                        let msg = &messages[i];
                        for part in Self::content_parts(msg) {
                            match part {
                                ContentPart::Text { text, .. } => {
                                    if !text.trim().is_empty() {
                                        content.push(serde_json::json!({ "text": text }));
                                    }
                                }
                                ContentPart::ToolCall {
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    ..
                                } => {
                                    content.push(serde_json::json!({
                                        "toolUse": {
                                            "toolUseId": tool_call_id,
                                            "name": tool_name,
                                            "input": arguments,
                                        }
                                    }));
                                }
                                ContentPart::Reasoning { text, .. } => {
                                    content.push(serde_json::json!({
                                        "reasoningContent": { "reasoningText": { "text": text } }
                                    }));
                                }
                                _ => {}
                            }
                        }
                        i += 1;
                    }
                    out.push(serde_json::json!({ "role": "assistant", "content": content }));
                }
                _ => {
                    return Err(LlmError::UnsupportedOperation(
                        "System/developer messages must be placed at the beginning for Bedrock"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(out)
    }
}

impl RequestTransformer for BedrockChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let (tool_config, _warnings) = Self::build_tool_config(req);
        let (system, rest) = Self::split_system_messages(&req.messages);
        let messages = Self::convert_messages(rest)?;

        let mut body = serde_json::json!({
            "system": system,
            "messages": messages,
        });
        if let Some(cfg) = tool_config {
            body["toolConfig"] = cfg;
        }

        let max_tokens = req
            .common_params
            .max_completion_tokens
            .or(req.common_params.max_tokens);
        let mut inference: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        if let Some(v) = max_tokens {
            inference.insert("maxTokens".to_string(), serde_json::json!(v));
        }
        if let Some(v) = req.common_params.temperature {
            inference.insert("temperature".to_string(), serde_json::json!(v));
        }
        if let Some(v) = req.common_params.top_p {
            inference.insert("topP".to_string(), serde_json::json!(v));
        }
        if let Some(v) = req.common_params.top_k {
            inference.insert("topK".to_string(), serde_json::json!(v));
        }
        if let Some(v) = req.common_params.stop_sequences.as_ref() {
            inference.insert("stopSequences".to_string(), serde_json::json!(v));
        }
        if !inference.is_empty() {
            body["inferenceConfig"] = serde_json::Value::Object(inference);
        }

        if let Some(opts) = req.provider_options_map.get_object("bedrock")
            && let Some(fields) = opts
                .get("additionalModelRequestFields")
                .or_else(|| opts.get("additional_model_request_fields"))
        {
            body["additionalModelRequestFields"] = fields.clone();
        }

        Ok(body)
    }
}

#[derive(Clone)]
struct BedrockChatResponseTransformer {
    provider_id: String,
    uses_json_response_tool: bool,
}

impl BedrockChatResponseTransformer {
    fn map_finish_reason(
        raw: Option<&str>,
        is_json_response_from_tool: bool,
    ) -> Option<FinishReason> {
        let raw = raw?;
        Some(match raw {
            "stop" | "stop_sequence" | "end_turn" => FinishReason::Stop,
            "max_tokens" | "length" => FinishReason::Length,
            "content_filtered" | "guardrail_intervened" | "content-filter" | "content_filter" => {
                FinishReason::ContentFilter
            }
            "tool_use" | "tool-calls" | "tool_calls" => {
                if is_json_response_from_tool {
                    FinishReason::Stop
                } else {
                    FinishReason::ToolCalls
                }
            }
            other => FinishReason::Other(other.to_string()),
        })
    }

    fn set_bedrock_metadata(
        resp: &mut ChatResponse,
        is_json_response_from_tool: bool,
        stop_sequence: Option<serde_json::Value>,
    ) {
        if !is_json_response_from_tool && stop_sequence.is_none() {
            return;
        }

        let mut bedrock: HashMap<String, serde_json::Value> = HashMap::new();
        if is_json_response_from_tool {
            bedrock.insert(
                "isJsonResponseFromTool".to_string(),
                serde_json::Value::Bool(true),
            );
        }
        if let Some(v) = stop_sequence {
            bedrock.insert("stopSequence".to_string(), v);
        }

        let mut root = resp.provider_metadata.take().unwrap_or_default();
        root.insert("bedrock".to_string(), bedrock);
        resp.provider_metadata = Some(root);
    }
}

impl ResponseTransformer for BedrockChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let content_arr = raw
            .get("output")
            .and_then(|o| o.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmError::ParseError("Missing Bedrock output.message.content".into()))?;

        let mut parts: Vec<ContentPart> = Vec::new();
        let mut is_json_response_from_tool = false;

        for item in content_arr {
            if let Some(text) = item.get("text").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                parts.push(ContentPart::text(text.to_string()));
            }

            if let Some(tool_use) = item.get("toolUse") {
                let tool_use_id = tool_use
                    .get("toolUseId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool-use-id")
                    .to_string();
                let name = tool_use
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool")
                    .to_string();
                let input = tool_use
                    .get("input")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if self.uses_json_response_tool && name == "json" {
                    is_json_response_from_tool = true;
                    parts.push(ContentPart::text(
                        serde_json::to_string(&input).unwrap_or_default(),
                    ));
                } else {
                    parts.push(ContentPart::tool_call(tool_use_id, name, input, None));
                }
            }
        }

        let usage = raw.get("usage").and_then(|u| {
            let prompt = u.get("inputTokens")?.as_u64()? as u32;
            let completion = u.get("outputTokens")?.as_u64()? as u32;
            let total = u.get("totalTokens")?.as_u64()? as u32;
            Some(
                Usage::builder()
                    .prompt_tokens(prompt)
                    .completion_tokens(completion)
                    .total_tokens(total)
                    .build(),
            )
        });

        let mut resp = ChatResponse::new(MessageContent::MultiModal(parts));
        resp.usage = usage;

        let raw_reason = raw.get("stopReason").and_then(|v| v.as_str());
        resp.finish_reason = Self::map_finish_reason(raw_reason, is_json_response_from_tool);

        let stop_sequence = raw
            .get("additionalModelResponseFields")
            .and_then(|v| v.get("delta"))
            .and_then(|v| v.get("stop_sequence"))
            .cloned();
        Self::set_bedrock_metadata(&mut resp, is_json_response_from_tool, stop_sequence);

        Ok(resp)
    }
}

// ---------------------------------------------------------------------------
// Streaming (JSON lines)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockMessageStop {
    stop_reason: Option<String>,
    #[allow(dead_code)]
    additional_model_response_fields: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockUsageInfo {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct BedrockMetadata {
    usage: Option<BedrockUsageInfo>,
    #[allow(dead_code)]
    trace: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolUseStart {
    tool_use_id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BedrockContentBlockStartInner {
    #[serde(default, rename = "toolUse")]
    tool_use: Option<BedrockToolUseStart>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockContentBlockStart {
    content_block_index: Option<u32>,
    start: Option<BedrockContentBlockStartInner>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockContentBlockStop {
    content_block_index: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockContentBlockDelta {
    content_block_index: Option<u32>,
    delta: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockStreamChunk {
    #[serde(default)]
    content_block_start: Option<BedrockContentBlockStart>,
    #[serde(default)]
    content_block_delta: Option<BedrockContentBlockDelta>,
    #[serde(default)]
    content_block_stop: Option<BedrockContentBlockStop>,
    #[serde(default)]
    metadata: Option<BedrockMetadata>,
    #[serde(default)]
    message_stop: Option<BedrockMessageStop>,
}

#[derive(Debug, Clone)]
struct ToolAcc {
    id: String,
    name: String,
    args: String,
    is_json: bool,
}

#[derive(Debug, Default)]
struct BedrockStreamAcc {
    content: String,
    tool_by_block: HashMap<u32, ToolAcc>,
    tool_by_id: HashMap<String, ToolAcc>,
    usage: Option<Usage>,
    finish_reason_raw: Option<String>,
    is_json_response_from_tool: bool,
}

#[derive(Clone)]
pub struct BedrockEventConverter {
    provider_id: String,
    uses_json_response_tool: bool,
    tracker: StreamStateTracker,
    acc: Arc<Mutex<BedrockStreamAcc>>,
}

impl BedrockEventConverter {
    pub fn new(provider_id: &str, uses_json_response_tool: bool) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            uses_json_response_tool,
            tracker: StreamStateTracker::new(),
            acc: Arc::new(Mutex::new(BedrockStreamAcc::default())),
        }
    }

    fn stream_start_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,
            model: None,
            created: Some(chrono::Utc::now()),
            provider: self.provider_id.clone(),
            request_id: None,
        }
    }

    fn finalize_response(&self) -> ChatResponse {
        let acc = self.acc.lock().expect("lock");

        let mut parts: Vec<ContentPart> = Vec::new();
        if !acc.content.is_empty() {
            parts.push(ContentPart::text(acc.content.clone()));
        }

        for tool in acc.tool_by_id.values() {
            if tool.is_json {
                continue;
            }
            let args_value: serde_json::Value =
                serde_json::from_str(&tool.args).unwrap_or_else(|_| serde_json::json!({}));
            parts.push(ContentPart::tool_call(
                tool.id.clone(),
                tool.name.clone(),
                args_value,
                None,
            ));
        }

        let mut resp = ChatResponse::new(MessageContent::MultiModal(parts));
        resp.usage = acc.usage.clone();
        resp.finish_reason = BedrockChatResponseTransformer::map_finish_reason(
            acc.finish_reason_raw.as_deref(),
            acc.is_json_response_from_tool,
        );

        if acc.is_json_response_from_tool {
            let mut bedrock: HashMap<String, serde_json::Value> = HashMap::new();
            bedrock.insert(
                "isJsonResponseFromTool".to_string(),
                serde_json::Value::Bool(true),
            );
            let mut root = HashMap::new();
            root.insert("bedrock".to_string(), bedrock);
            resp.provider_metadata = Some(root);
        }

        resp
    }
}

impl JsonEventConverter for BedrockEventConverter {
    fn convert_json<'a>(
        &'a self,
        json_data: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>
    {
        Box::pin(async move {
            let chunk: BedrockStreamChunk =
                match crate::streaming::parse_json_with_repair(json_data) {
                    Ok(v) => v,
                    Err(e) => {
                        return vec![Err(LlmError::ParseError(format!(
                            "Failed to parse Bedrock JSON chunk: {e}"
                        )))];
                    }
                };

            let mut builder = EventBuilder::new();

            if self.tracker.needs_stream_start() {
                builder = builder.add_stream_start(self.stream_start_metadata());
            }

            // Text deltas.
            if let Some(delta) = chunk
                .content_block_delta
                .as_ref()
                .and_then(|d| d.delta.as_ref())
            {
                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                    let mut acc = self.acc.lock().expect("lock");
                    acc.content.push_str(text);
                    drop(acc);
                    builder = builder.add_content_delta(text.to_string(), None);
                }

                if let Some(rc) = delta.get("reasoningContent")
                    && let Some(text) = rc.get("text").and_then(|v| v.as_str())
                {
                    builder = builder.add_thinking_delta(text.to_string());
                }
            }

            // Tool starts.
            if let Some(start) = chunk.content_block_start.as_ref()
                && let (Some(block), Some(tool)) = (
                    start.content_block_index,
                    start.start.as_ref().and_then(|s| s.tool_use.as_ref()),
                )
            {
                let id = tool
                    .tool_use_id
                    .clone()
                    .unwrap_or_else(|| "tool-use-id".to_string());
                let name = tool.name.clone().unwrap_or_else(|| "tool".to_string());
                let is_json = self.uses_json_response_tool && name == "json";
                let tool_acc = ToolAcc {
                    id: id.clone(),
                    name: name.clone(),
                    args: String::new(),
                    is_json,
                };

                let mut acc = self.acc.lock().expect("lock");
                acc.tool_by_block.insert(block, tool_acc.clone());
                acc.tool_by_id.insert(id.clone(), tool_acc);
                drop(acc);

                if !is_json {
                    builder = builder.add_tool_call_delta(id, Some(name), None, None);
                }
            }

            // Tool input deltas.
            if let Some(delta) = chunk
                .content_block_delta
                .as_ref()
                .and_then(|d| d.delta.as_ref())
                && let (Some(block), Some(tool_use)) = (
                    chunk
                        .content_block_delta
                        .as_ref()
                        .and_then(|d| d.content_block_index),
                    delta.get("toolUse"),
                )
                && let Some(input) = tool_use.get("input").and_then(|v| v.as_str())
            {
                let mut acc = self.acc.lock().expect("lock");
                let mut tool_id: Option<String> = None;
                let mut is_json = false;
                if let Some(t) = acc.tool_by_block.get_mut(&block) {
                    t.args.push_str(input);
                    tool_id = Some(t.id.clone());
                    is_json = t.is_json;
                }

                if let Some(id) = tool_id {
                    if let Some(t2) = acc.tool_by_id.get_mut(&id) {
                        t2.args.push_str(input);
                    }
                    if !is_json {
                        drop(acc);
                        builder =
                            builder.add_tool_call_delta(id, None, Some(input.to_string()), None);
                    }
                }
            }

            // Tool stop: emit json tool input as text.
            if let Some(stop) = chunk
                .content_block_stop
                .as_ref()
                .and_then(|s| s.content_block_index)
            {
                let mut acc = self.acc.lock().expect("lock");
                if let Some(tool) = acc.tool_by_block.remove(&stop)
                    && tool.is_json
                    && !tool.args.is_empty()
                {
                    acc.is_json_response_from_tool = true;
                    acc.content.push_str(&tool.args);
                    let text = tool.args;
                    drop(acc);
                    builder = builder.add_content_delta(text, None);
                }
            }

            // Usage.
            if let Some(meta) = chunk.metadata.as_ref().and_then(|m| m.usage.as_ref())
                && let (Some(prompt), Some(completion), Some(total)) =
                    (meta.input_tokens, meta.output_tokens, meta.total_tokens)
            {
                let usage = Usage::builder()
                    .prompt_tokens(prompt)
                    .completion_tokens(completion)
                    .total_tokens(total)
                    .build();
                let mut acc = self.acc.lock().expect("lock");
                acc.usage = Some(usage.clone());
                drop(acc);
                builder = builder.add_usage_update(usage);
            }

            // Finish.
            if let Some(stop) = chunk.message_stop.as_ref() {
                let mut acc = self.acc.lock().expect("lock");
                acc.finish_reason_raw = stop.stop_reason.clone();
                drop(acc);

                self.tracker.mark_stream_ended();
                let resp = self.finalize_response();
                builder = builder.add_stream_end(resp);
            }

            builder.build().into_iter().map(Ok).collect()
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        if !self.tracker.needs_stream_end() {
            return None;
        }

        let resp = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text(String::new()),
            usage: None,
            finish_reason: Some(FinishReason::Unknown),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response: resp }))
    }
}
