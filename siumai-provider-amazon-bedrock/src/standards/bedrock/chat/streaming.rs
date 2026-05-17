use super::{
    BedrockChatResponseTransformer, BedrockUsageInfo, bedrock_reasoning_part_metadata,
    bedrock_usage_metadata_fragment, build_bedrock_usage_from_info_with_raw, is_mistral_model,
    merge_bedrock_metadata_root, normalize_tool_call_id,
    response_provider_metadata_to_stream_provider_metadata,
};
use crate::error::LlmError;
use crate::streaming::{
    ChatStreamEvent, ChatStreamPart, EventBuilder, JsonEventConverter, StreamStateTracker,
};
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, ChatStreamToolCall, ContentPart, FinishReason,
    MessageContent, ResponseMetadata, Usage, Warning,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockMessageStop {
    stop_reason: Option<String>,
    #[allow(dead_code)]
    additional_model_response_fields: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockMetadata {
    usage: Option<BedrockUsageInfo>,
    trace: Option<serde_json::Value>,
    performance_config: Option<serde_json::Value>,
    service_tier: Option<serde_json::Value>,
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
    internal_server_exception: Option<serde_json::Value>,
    #[serde(default)]
    model_stream_error_exception: Option<serde_json::Value>,
    #[serde(default)]
    throttling_exception: Option<serde_json::Value>,
    #[serde(default)]
    validation_exception: Option<serde_json::Value>,
    #[serde(default)]
    message_stop: Option<BedrockMessageStop>,
}

#[derive(Debug, Clone)]
struct ToolAcc {
    id: String,
    name: String,
    json_text: String,
    is_json: bool,
}

#[derive(Debug, Clone)]
struct TextBlockAcc {
    text: String,
    started_emitted: bool,
}

#[derive(Debug, Clone)]
struct ReasoningBlockAcc {
    text: String,
    provider_metadata: Option<HashMap<String, serde_json::Value>>,
    started_emitted: bool,
}

#[derive(Debug, Clone)]
enum BedrockBlockAcc {
    Text(TextBlockAcc),
    Reasoning(ReasoningBlockAcc),
    Tool(ToolAcc),
}

#[derive(Debug, Default)]
struct BedrockStreamAcc {
    active_blocks: HashMap<u32, BedrockBlockAcc>,
    final_parts: Vec<ContentPart>,
    usage: Option<Usage>,
    provider_metadata: serde_json::Map<String, serde_json::Value>,
    finish_reason_raw: Option<String>,
    stop_sequence: Option<serde_json::Value>,
    is_json_response_from_tool: bool,
    stream_start_part_emitted: bool,
    response_metadata_emitted: bool,
}

#[derive(Clone)]
pub struct BedrockEventConverter {
    provider_id: String,
    uses_json_response_tool: bool,
    default_model: Option<String>,
    warnings: Vec<Warning>,
    include_raw_chunks: bool,
    created_at: chrono::DateTime<chrono::Utc>,
    tracker: StreamStateTracker,
    acc: Arc<Mutex<BedrockStreamAcc>>,
}

impl BedrockEventConverter {
    pub fn new(
        provider_id: &str,
        uses_json_response_tool: bool,
        default_model: Option<String>,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            uses_json_response_tool,
            default_model: default_model.filter(|model| !model.trim().is_empty()),
            warnings,
            include_raw_chunks,
            created_at: chrono::Utc::now(),
            tracker: StreamStateTracker::new(),
            acc: Arc::new(Mutex::new(BedrockStreamAcc::default())),
        }
    }

    fn is_mistral_model(&self) -> bool {
        is_mistral_model(self.default_model.as_deref())
    }

    fn response_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,
            model: self.default_model.clone(),
            created: Some(self.created_at),
            provider: self.provider_id.clone(),
            request_id: None,
            headers: None,
            body: None,
        }
    }

    fn append_stream_preamble(&self, out: &mut Vec<Result<ChatStreamEvent, LlmError>>) {
        let metadata = self.response_metadata();
        let (emit_stream_start_part, emit_response_metadata) = {
            let mut acc = self.acc.lock().expect("lock");
            let emit_stream_start_part = !acc.stream_start_part_emitted;
            if emit_stream_start_part {
                acc.stream_start_part_emitted = true;
            }
            let emit_response_metadata = !acc.response_metadata_emitted;
            if emit_response_metadata {
                acc.response_metadata_emitted = true;
            }
            (emit_stream_start_part, emit_response_metadata)
        };

        if self.tracker.needs_stream_start() {
            out.push(Ok(ChatStreamEvent::StreamStart {
                metadata: metadata.clone(),
            }));
        }
        if emit_stream_start_part {
            out.push(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart {
                    warnings: self.warnings.clone(),
                },
            }));
        }
        if emit_response_metadata {
            out.push(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::ResponseMetadata(metadata),
            }));
        }
    }

    fn append_raw_chunk(
        &self,
        out: &mut Vec<Result<ChatStreamEvent, LlmError>>,
        raw_value: &serde_json::Value,
    ) {
        if !self.include_raw_chunks {
            return;
        }

        out.push(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw {
                raw_value: raw_value.clone(),
            },
        }));
    }

    fn stream_error_part(chunk: &BedrockStreamChunk) -> Option<serde_json::Value> {
        chunk
            .internal_server_exception
            .clone()
            .or_else(|| chunk.model_stream_error_exception.clone())
            .or_else(|| chunk.throttling_exception.clone())
            .or_else(|| chunk.validation_exception.clone())
    }

    fn stop_sequence(message_stop: &BedrockMessageStop) -> Option<serde_json::Value> {
        message_stop
            .additional_model_response_fields
            .as_ref()
            .and_then(|value| value.get("delta"))
            .and_then(|value| value.get("stop_sequence"))
            .cloned()
    }

    fn append_terminal_events(
        &self,
        out: &mut Vec<Result<ChatStreamEvent, LlmError>>,
        use_unknown_finish_reason: bool,
    ) {
        let mut response = self.finalize_response();
        if use_unknown_finish_reason && response.finish_reason.is_none() {
            response.finish_reason = Some(FinishReason::Unknown);
        }

        out.push(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage: response.usage.clone().unwrap_or_else(Usage::unknown),
                finish_reason: ChatStreamFinishInfo {
                    unified: response
                        .finish_reason
                        .clone()
                        .unwrap_or(FinishReason::Unknown),
                    raw: response.raw_finish_reason.clone(),
                },
                provider_metadata: response_provider_metadata_to_stream_provider_metadata(
                    response.provider_metadata.as_ref(),
                ),
            },
        }));
        out.push(Ok(ChatStreamEvent::StreamEnd { response }));
    }

    fn flush_active_blocks(acc: &mut BedrockStreamAcc) {
        let mut block_indexes: Vec<u32> = acc.active_blocks.keys().copied().collect();
        block_indexes.sort_unstable();

        for block_index in block_indexes {
            if let Some(block) = acc.active_blocks.remove(&block_index) {
                Self::push_final_part(acc, block);
            }
        }
    }

    fn push_final_part(acc: &mut BedrockStreamAcc, block: BedrockBlockAcc) {
        match block {
            BedrockBlockAcc::Text(block) => {
                if block.started_emitted || !block.text.is_empty() {
                    acc.final_parts.push(ContentPart::text(block.text));
                }
            }
            BedrockBlockAcc::Reasoning(block) => {
                if block.started_emitted
                    || block.provider_metadata.is_some()
                    || !block.text.is_empty()
                {
                    acc.final_parts.push(ContentPart::Reasoning {
                        text: block.text,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata: block.provider_metadata,
                    });
                }
            }
            BedrockBlockAcc::Tool(tool) => {
                if tool.is_json {
                    acc.is_json_response_from_tool = true;
                    acc.final_parts.push(ContentPart::text(tool.json_text));
                } else {
                    let input = if tool.json_text.is_empty() {
                        "{}"
                    } else {
                        tool.json_text.as_str()
                    };
                    let arguments =
                        serde_json::from_str(input).unwrap_or_else(|_| serde_json::json!({}));
                    acc.final_parts
                        .push(ContentPart::tool_call(tool.id, tool.name, arguments, None));
                }
            }
        }
    }

    fn finalize_response(&self) -> ChatResponse {
        let mut acc = self.acc.lock().expect("lock");
        Self::flush_active_blocks(&mut acc);

        let parts = acc.final_parts.clone();
        let usage = acc.usage.clone();
        let finish_reason_raw = acc.finish_reason_raw.clone();
        let is_json_response_from_tool = acc.is_json_response_from_tool;
        let stop_sequence = acc.stop_sequence.clone();
        let mut provider_metadata = acc.provider_metadata.clone();
        drop(acc);

        let mut resp = ChatResponse::new(MessageContent::MultiModal(parts));
        resp.model = self.default_model.clone();
        resp.usage = usage;
        resp.finish_reason = BedrockChatResponseTransformer::map_finish_reason(
            finish_reason_raw.as_deref(),
            is_json_response_from_tool,
        );
        resp.raw_finish_reason = finish_reason_raw;
        if !self.warnings.is_empty() {
            resp.warnings = Some(self.warnings.clone());
        }
        BedrockChatResponseTransformer::set_bedrock_metadata(
            &mut resp,
            is_json_response_from_tool,
            stop_sequence,
        );
        if !provider_metadata.is_empty() {
            let root = resp.provider_metadata.get_or_insert_with(HashMap::new);
            merge_bedrock_metadata_root(root, &mut provider_metadata);
        }

        resp
    }
}

fn merge_json_object(
    target: &mut serde_json::Map<String, serde_json::Value>,
    incoming: &mut serde_json::Map<String, serde_json::Value>,
) {
    for (key, value) in std::mem::take(incoming) {
        match (target.get_mut(&key), value) {
            (Some(serde_json::Value::Object(existing)), serde_json::Value::Object(mut inner)) => {
                existing.append(&mut inner);
            }
            (_, value) => {
                target.insert(key, value);
            }
        }
    }
}

fn merge_provider_metadata_maps(
    target: &mut Option<HashMap<String, serde_json::Value>>,
    incoming: HashMap<String, serde_json::Value>,
) {
    let Some(target) = target.as_mut() else {
        *target = Some(incoming);
        return;
    };

    for (key, value) in incoming {
        match (target.get_mut(&key), value) {
            (Some(serde_json::Value::Object(existing)), serde_json::Value::Object(mut inner)) => {
                existing.append(&mut inner);
            }
            (_, value) => {
                target.insert(key, value);
            }
        }
    }
}

impl JsonEventConverter for BedrockEventConverter {
    fn convert_json<'a>(
        &'a self,
        json_data: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>
    {
        Box::pin(async move {
            let raw_value: serde_json::Value = match serde_json::from_str(json_data) {
                Ok(value) => value,
                Err(e) => {
                    let mut out = Vec::new();
                    self.append_stream_preamble(&mut out);
                    self.append_raw_chunk(
                        &mut out,
                        &serde_json::Value::String(json_data.to_string()),
                    );
                    out.push(Err(LlmError::ParseError(format!(
                        "Failed to parse Bedrock JSON chunk: {e}"
                    ))));
                    return out;
                }
            };

            let chunk: BedrockStreamChunk = match serde_json::from_value(raw_value.clone()) {
                Ok(chunk) => chunk,
                Err(e) => {
                    let mut out = Vec::new();
                    self.append_stream_preamble(&mut out);
                    self.append_raw_chunk(&mut out, &raw_value);
                    out.push(Err(LlmError::ParseError(format!(
                        "Failed to parse Bedrock JSON chunk: {e}"
                    ))));
                    return out;
                }
            };

            let mut out = Vec::new();
            self.append_stream_preamble(&mut out);
            self.append_raw_chunk(&mut out, &raw_value);

            if let Some(error) = Self::stream_error_part(&chunk) {
                self.tracker.mark_stream_ended();
                out.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::Error { error },
                }));
                return out;
            }

            let mut builder = EventBuilder::new();
            let delta = chunk
                .content_block_delta
                .as_ref()
                .and_then(|chunk| chunk.delta.as_ref());

            if let Some(start) = chunk.content_block_start.as_ref()
                && let Some(block_index) = start.content_block_index
            {
                if let Some(tool_use) = start
                    .start
                    .as_ref()
                    .and_then(|start| start.tool_use.as_ref())
                {
                    let name = tool_use.name.clone().unwrap_or_else(|| "tool".to_string());
                    let raw_id = tool_use
                        .tool_use_id
                        .clone()
                        .unwrap_or_else(|| "tool-use-id".to_string());
                    let normalized_id = normalize_tool_call_id(&raw_id, self.is_mistral_model());
                    let is_json = self.uses_json_response_tool && name == "json";

                    let mut acc = self.acc.lock().expect("lock");
                    acc.active_blocks.insert(
                        block_index,
                        BedrockBlockAcc::Tool(ToolAcc {
                            id: normalized_id.clone(),
                            name: name.clone(),
                            json_text: String::new(),
                            is_json,
                        }),
                    );
                    drop(acc);

                    if !is_json {
                        builder = builder.add_tool_input_start(normalized_id, name);
                    }
                } else {
                    let emit_text_start = {
                        let mut acc = self.acc.lock().expect("lock");
                        match acc.active_blocks.get_mut(&block_index) {
                            Some(BedrockBlockAcc::Text(block)) => {
                                if block.started_emitted {
                                    false
                                } else {
                                    block.started_emitted = true;
                                    true
                                }
                            }
                            _ => {
                                acc.active_blocks.insert(
                                    block_index,
                                    BedrockBlockAcc::Text(TextBlockAcc {
                                        text: String::new(),
                                        started_emitted: true,
                                    }),
                                );
                                true
                            }
                        }
                    };

                    if emit_text_start {
                        builder = builder.add_part(ChatStreamPart::TextStart {
                            id: block_index.to_string(),
                            provider_metadata: None,
                        });
                    }
                }
            }

            if let Some(text) = delta
                .and_then(|delta| delta.get("text"))
                .and_then(|value| value.as_str())
            {
                let block_index = chunk
                    .content_block_delta
                    .as_ref()
                    .and_then(|delta| delta.content_block_index)
                    .unwrap_or(0);

                let emit_text_start = {
                    let mut acc = self.acc.lock().expect("lock");
                    match acc.active_blocks.get_mut(&block_index) {
                        Some(BedrockBlockAcc::Text(block)) => {
                            let emit_text_start = !block.started_emitted;
                            block.started_emitted = true;
                            block.text.push_str(text);
                            emit_text_start
                        }
                        _ => {
                            acc.active_blocks.insert(
                                block_index,
                                BedrockBlockAcc::Text(TextBlockAcc {
                                    text: text.to_string(),
                                    started_emitted: true,
                                }),
                            );
                            true
                        }
                    }
                };

                if emit_text_start {
                    builder = builder.add_part(ChatStreamPart::TextStart {
                        id: block_index.to_string(),
                        provider_metadata: None,
                    });
                }

                builder = builder.add_text_delta(block_index.to_string(), text);
            }

            if let Some(reasoning_content) = delta.and_then(|delta| delta.get("reasoningContent")) {
                let block_index = chunk
                    .content_block_delta
                    .as_ref()
                    .and_then(|delta| delta.content_block_index)
                    .unwrap_or(0);

                if let Some(text) = reasoning_content
                    .get("text")
                    .and_then(|value| value.as_str())
                {
                    let emit_reasoning_start = {
                        let mut acc = self.acc.lock().expect("lock");
                        match acc.active_blocks.get_mut(&block_index) {
                            Some(BedrockBlockAcc::Reasoning(block)) => {
                                let emit_reasoning_start = !block.started_emitted;
                                block.started_emitted = true;
                                block.text.push_str(text);
                                emit_reasoning_start
                            }
                            _ => {
                                acc.active_blocks.insert(
                                    block_index,
                                    BedrockBlockAcc::Reasoning(ReasoningBlockAcc {
                                        text: text.to_string(),
                                        provider_metadata: None,
                                        started_emitted: true,
                                    }),
                                );
                                true
                            }
                        }
                    };

                    if emit_reasoning_start {
                        builder = builder.add_part(ChatStreamPart::ReasoningStart {
                            id: block_index.to_string(),
                            provider_metadata: None,
                        });
                    }

                    builder = builder.add_reasoning_delta(block_index.to_string(), text);
                } else if let Some(provider_metadata) = bedrock_reasoning_part_metadata(
                    reasoning_content
                        .get("signature")
                        .and_then(|value| value.as_str()),
                    reasoning_content
                        .get("data")
                        .and_then(|value| value.as_str()),
                ) {
                    let mut acc = self.acc.lock().expect("lock");
                    match acc.active_blocks.get_mut(&block_index) {
                        Some(BedrockBlockAcc::Reasoning(block)) => {
                            merge_provider_metadata_maps(
                                &mut block.provider_metadata,
                                provider_metadata.clone(),
                            );
                        }
                        _ => {
                            acc.active_blocks.insert(
                                block_index,
                                BedrockBlockAcc::Reasoning(ReasoningBlockAcc {
                                    text: String::new(),
                                    provider_metadata: Some(provider_metadata.clone()),
                                    started_emitted: false,
                                }),
                            );
                        }
                    }
                    drop(acc);

                    builder = builder.add_part(ChatStreamPart::ReasoningDelta {
                        id: block_index.to_string(),
                        delta: String::new(),
                        provider_metadata: Some(provider_metadata),
                    });
                }
            }

            if let Some(tool_use) = delta.and_then(|delta| delta.get("toolUse"))
                && let Some(input) = tool_use.get("input").and_then(|value| value.as_str())
            {
                let block_index = chunk
                    .content_block_delta
                    .as_ref()
                    .and_then(|delta| delta.content_block_index)
                    .unwrap_or(0);

                let mut tool_id = None;
                let mut is_json = false;
                {
                    let mut acc = self.acc.lock().expect("lock");
                    if let Some(BedrockBlockAcc::Tool(tool)) =
                        acc.active_blocks.get_mut(&block_index)
                    {
                        tool.json_text.push_str(input);
                        tool_id = Some(tool.id.clone());
                        is_json = tool.is_json;
                    }
                }

                if let Some(tool_id) = tool_id
                    && !is_json
                {
                    builder = builder.add_tool_input_delta(tool_id, input);
                }
            }

            if let Some(block_index) = chunk
                .content_block_stop
                .as_ref()
                .and_then(|stop| stop.content_block_index)
            {
                let stopped_block = {
                    let mut acc = self.acc.lock().expect("lock");
                    acc.active_blocks.remove(&block_index)
                };

                if let Some(stopped_block) = stopped_block {
                    match &stopped_block {
                        BedrockBlockAcc::Text(block) => {
                            if block.started_emitted {
                                builder = builder.add_part(ChatStreamPart::TextEnd {
                                    id: block_index.to_string(),
                                    provider_metadata: None,
                                });
                            }
                        }
                        BedrockBlockAcc::Reasoning(block) => {
                            if block.started_emitted {
                                builder = builder.add_part(ChatStreamPart::ReasoningEnd {
                                    id: block_index.to_string(),
                                    provider_metadata: None,
                                });
                            }
                        }
                        BedrockBlockAcc::Tool(tool) => {
                            if tool.is_json {
                                let text = tool.json_text.clone();
                                builder = builder.add_part(ChatStreamPart::TextStart {
                                    id: block_index.to_string(),
                                    provider_metadata: None,
                                });
                                if !text.is_empty() {
                                    builder = builder.add_text_delta(block_index.to_string(), text);
                                }
                                builder = builder.add_part(ChatStreamPart::TextEnd {
                                    id: block_index.to_string(),
                                    provider_metadata: None,
                                });
                            } else {
                                let input = if tool.json_text.is_empty() {
                                    "{}".to_string()
                                } else {
                                    tool.json_text.clone()
                                };
                                builder = builder
                                    .add_part(ChatStreamPart::ToolInputEnd {
                                        id: tool.id.clone(),
                                        provider_metadata: None,
                                    })
                                    .add_part(ChatStreamPart::ToolCall(ChatStreamToolCall {
                                        tool_call_id: tool.id.clone(),
                                        tool_name: tool.name.clone(),
                                        input,
                                        provider_executed: None,
                                        dynamic: None,
                                        provider_metadata: None,
                                    }));
                            }
                        }
                    }

                    let mut acc = self.acc.lock().expect("lock");
                    Self::push_final_part(&mut acc, stopped_block);
                }
            }

            if let Some(metadata) = chunk.metadata.as_ref() {
                if let Some(usage_info) = metadata.usage.as_ref() {
                    let raw_usage = raw_value
                        .get("metadata")
                        .and_then(|metadata| metadata.get("usage"))
                        .cloned();
                    let usage = build_bedrock_usage_from_info_with_raw(usage_info, raw_usage);
                    let mut acc = self.acc.lock().expect("lock");
                    acc.usage = Some(usage.clone());

                    let mut provider_metadata = serde_json::Map::new();
                    if let Some((key, value)) = bedrock_usage_metadata_fragment(usage_info) {
                        provider_metadata.insert(key, value);
                    }
                    merge_json_object(&mut acc.provider_metadata, &mut provider_metadata);
                    drop(acc);
                }

                let mut provider_metadata = serde_json::Map::new();
                if let Some(trace) = metadata.trace.clone() {
                    provider_metadata.insert("trace".to_string(), trace);
                }
                if let Some(performance_config) = metadata.performance_config.clone() {
                    provider_metadata.insert("performanceConfig".to_string(), performance_config);
                }
                if let Some(service_tier) = metadata.service_tier.clone() {
                    provider_metadata.insert("serviceTier".to_string(), service_tier);
                }
                if !provider_metadata.is_empty() {
                    let mut acc = self.acc.lock().expect("lock");
                    merge_json_object(&mut acc.provider_metadata, &mut provider_metadata);
                }
            }

            out.extend(builder.build().into_iter().map(Ok));

            if let Some(stop) = chunk.message_stop.as_ref() {
                let mut acc = self.acc.lock().expect("lock");
                acc.finish_reason_raw = stop.stop_reason.clone();
                acc.stop_sequence = Self::stop_sequence(stop);
                drop(acc);

                self.tracker.mark_stream_ended();
                self.append_terminal_events(&mut out, false);
            }

            out
        })
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        if !self.tracker.needs_stream_end() {
            return Vec::new();
        }

        let mut out = Vec::new();
        self.append_terminal_events(&mut out, true);
        out
    }
}
