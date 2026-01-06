//! Anthropic streaming implementation using eventsource-stream
//!
//! Provides SSE event conversion for Anthropic streaming responses.
//! The legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.

use super::params::AnthropicParams;
use super::provider_metadata::AnthropicSource;
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{ChatStreamEvent, StreamStateTracker};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use eventsource_stream::Event;
use serde::Deserialize;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

/// Citation document metadata extracted from the prompt (Vercel-aligned).
///
/// Anthropic streaming citation deltas reference documents by index, where the index is the
/// position of the cited document in the prompt's user file parts that have citations enabled.
#[derive(Debug, Clone)]
pub struct AnthropicCitationDocument {
    pub title: String,
    pub filename: Option<String>,
    pub media_type: String,
}

/// Anthropic stream event structure
/// This structure is flexible to handle different event types from Anthropic's SSE stream
#[derive(Debug, Clone, Deserialize)]
struct AnthropicStreamEvent {
    r#type: String,
    #[serde(default)]
    message: Option<AnthropicMessage>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    error: Option<serde_json::Value>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    #[allow(dead_code)]
    // Kept for forward-compatibility with Anthropic SSE payloads (serde needs the field even if unused)
    index: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    // Some Anthropic events provide a content_block object we don't consume yet; retained to avoid parse failures
    content_block: Option<serde_json::Value>,
}

/// Anthropic message structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessage {
    id: Option<String>,
    model: Option<String>,
    #[allow(dead_code)]
    // Role is not consumed by our unified event model, but appears in message_start payloads
    role: Option<String>,
    #[allow(dead_code)]
    // Raw content blocks not needed for our delta-based pipeline; retain for serde compatibility
    content: Option<Vec<AnthropicContent>>,
    #[allow(dead_code)]
    // Final stop reason may appear on message events; parsing handled elsewhere
    stop_reason: Option<String>,
}

/// Anthropic content structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    // Different content block types exist; we only consume text via deltas
    content_type: String,
    #[allow(dead_code)]
    // Some events carry full text here; our converter aggregates from deltas instead
    text: Option<String>,
}

/// Anthropic delta structure
/// Supports different delta types: text_delta, input_json_delta, thinking_delta, etc.
#[derive(Debug, Clone, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    #[serde(default)]
    #[allow(dead_code)]
    // Delta subtype (text_delta, input_json_delta, etc.); not required for our current transformations
    delta_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Partial JSON chunks for tool inputs; not emitted as separate events in our model yet
    partial_json: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    citation: Option<serde_json::Value>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    // Stop sequence token for deltas; usage is reflected via finish events elsewhere
    stop_sequence: Option<String>,
}

/// Anthropic usage structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Anthropic event converter
#[derive(Clone)]
pub struct AnthropicEventConverter {
    #[allow(dead_code)]
    // Retained for potential future behavior toggles; not read in the current converter
    config: AnthropicParams,
    state_tracker: StreamStateTracker,
    tool_use_ids_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    content_block_type_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    thinking_signature_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    redacted_thinking_data_by_index: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    citation_documents: Vec<AnthropicCitationDocument>,
    sources_by_id: Arc<Mutex<std::collections::HashMap<String, AnthropicSource>>>,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self {
            config,
            state_tracker: StreamStateTracker::new(),
            tool_use_ids_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            content_block_type_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            thinking_signature_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            redacted_thinking_data_by_index: Arc::new(Mutex::new(std::collections::HashMap::new())),
            citation_documents: Vec::new(),
            sources_by_id: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    pub fn with_citation_documents(mut self, docs: Vec<AnthropicCitationDocument>) -> Self {
        self.citation_documents = docs;
        self
    }

    fn record_source(&self, source: AnthropicSource) {
        if let Ok(mut map) = self.sources_by_id.lock() {
            map.insert(source.id.clone(), source);
        }
    }

    fn record_content_block_type(&self, index: usize, block_type: String) {
        if let Ok(mut map) = self.content_block_type_by_index.lock() {
            map.insert(index, block_type);
        }
    }

    fn get_content_block_type(&self, index: usize) -> Option<String> {
        self.content_block_type_by_index
            .lock()
            .ok()
            .and_then(|map| map.get(&index).cloned())
    }

    fn append_thinking_signature(&self, index: usize, delta: String) {
        if delta.is_empty() {
            return;
        }
        if let Ok(mut map) = self.thinking_signature_by_index.lock() {
            let entry = map.entry(index).or_default();
            entry.push_str(&delta);
        }
    }

    fn record_redacted_thinking_data(&self, index: usize, data: String) {
        if data.is_empty() {
            return;
        }
        if let Ok(mut map) = self.redacted_thinking_data_by_index.lock() {
            map.insert(index, data);
        }
    }

    fn build_stream_provider_metadata(
        &self,
    ) -> Option<
        std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
    > {
        let mut anthropic = std::collections::HashMap::new();

        if let Ok(map) = self.thinking_signature_by_index.lock()
            && let Some((_, sig)) = map.iter().min_by_key(|(k, _)| *k)
            && !sig.is_empty()
        {
            anthropic.insert("thinking_signature".to_string(), serde_json::json!(sig));
        }

        if let Ok(map) = self.redacted_thinking_data_by_index.lock()
            && let Some((_, data)) = map.iter().min_by_key(|(k, _)| *k)
            && !data.is_empty()
        {
            anthropic.insert(
                "redacted_thinking_data".to_string(),
                serde_json::json!(data),
            );
        }

        if let Ok(map) = self.sources_by_id.lock()
            && !map.is_empty()
        {
            let mut sources: Vec<_> = map.values().cloned().collect();
            sources.sort_by(|a, b| a.id.cmp(&b.id));
            if let Ok(v) = serde_json::to_value(sources) {
                anthropic.insert("sources".to_string(), v);
            }
        }

        if anthropic.is_empty() {
            None
        } else {
            let mut all = std::collections::HashMap::new();
            all.insert("anthropic".to_string(), anthropic);
            Some(all)
        }
    }

    /// Convert Anthropic stream event to one or more ChatStreamEvents
    fn convert_anthropic_event(&self, event: AnthropicStreamEvent) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        match event.r#type.as_str() {
            "error" => {
                let msg = event
                    .error
                    .as_ref()
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        event.error.as_ref().and_then(|e| {
                            serde_json::to_string(e)
                                .ok()
                                .map(|json| format!("Anthropic streaming error: {json}"))
                        })
                    })
                    .unwrap_or_else(|| "Anthropic streaming error (unknown)".to_string());

                vec![ChatStreamEvent::Error { error: msg }]
            }
            "message_start" => {
                if let Some(message) = event.message {
                    let metadata = ResponseMetadata {
                        id: message.id,
                        model: message.model,
                        created: Some(chrono::Utc::now()),
                        provider: "anthropic".to_string(),
                        request_id: None,
                    };
                    EventBuilder::new().add_stream_start(metadata).build()
                } else {
                    vec![]
                }
            }
            "content_block_start" => {
                let Some(content_block) = event.content_block else {
                    return vec![];
                };

                let block_type = content_block
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if let Some(idx) = event.index {
                    self.record_content_block_type(idx, block_type.to_string());
                }

                match block_type {
                    "thinking" => {
                        if let Some(idx) = event.index {
                            vec![ChatStreamEvent::Custom {
                                event_type: "anthropic:reasoning-start".to_string(),
                                data: serde_json::json!({
                                    "type": "reasoning-start",
                                    "contentBlockIndex": idx as u64,
                                }),
                            }]
                        } else {
                            vec![]
                        }
                    }
                    "redacted_thinking" => {
                        if let Some(idx) = event.index {
                            let data = content_block
                                .get("data")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            if !data.is_empty() {
                                self.record_redacted_thinking_data(idx, data.clone());
                            }

                            vec![ChatStreamEvent::Custom {
                                event_type: "anthropic:reasoning-start".to_string(),
                                data: serde_json::json!({
                                    "type": "reasoning-start",
                                    "contentBlockIndex": idx as u64,
                                    "redactedData": if data.is_empty() { serde_json::Value::Null } else { serde_json::json!(data) },
                                }),
                            }]
                        } else {
                            vec![]
                        }
                    }
                    "tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tool_name = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if tool_call_id.is_empty() || tool_name.is_empty() {
                            return vec![];
                        }

                        if let Some(idx) = event.index
                            && let Ok(mut map) = self.tool_use_ids_by_index.lock()
                        {
                            map.insert(idx, tool_call_id.clone());
                        }

                        vec![ChatStreamEvent::ToolCallDelta {
                            id: tool_call_id,
                            function_name: Some(tool_name),
                            arguments_delta: None,
                            index: event.index,
                        }]
                    }
                    "server_tool_use" => {
                        let tool_call_id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tool_name_raw = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = content_block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));

                        if tool_call_id.is_empty() || tool_name_raw.is_empty() {
                            return vec![];
                        }

                        // Vercel-aligned: map provider tool names back to stable custom names.
                        let tool_name = match tool_name_raw.as_str() {
                            "tool_search_tool_regex" | "tool_search_tool_bm25" => "tool_search",
                            other => other,
                        }
                        .to_string();

                        vec![ChatStreamEvent::Custom {
                            event_type: "anthropic:tool-call".to_string(),
                            data: serde_json::json!({
                                "type": "tool-call",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": input,
                                "providerExecuted": true,
                                "contentBlockIndex": event.index.map(|i| i as u64),
                                "rawContentBlock": content_block,
                            }),
                        }]
                    }
                    t if t.ends_with("_tool_result") => {
                        let tool_call_id = content_block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if tool_call_id.is_empty() {
                            return vec![];
                        }

                        let tool_name = match t {
                            "tool_search_tool_result" => "tool_search".to_string(),
                            _ => t.strip_suffix("_tool_result").unwrap_or(t).to_string(),
                        };
                        let raw_result = content_block
                            .get("content")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);

                        // Vercel-aligned: normalize provider-hosted tool results when possible
                        // (keep `rawContentBlock` for lossless access).
                        let (result, is_error) = if t == "web_search_tool_result" {
                            if let Some(arr) = raw_result.as_array() {
                                let normalized = arr
                                    .iter()
                                    .filter_map(|item| item.as_object())
                                    .map(|obj| {
                                        serde_json::json!({
                                            "type": obj.get("type").cloned().unwrap_or_else(|| serde_json::json!("web_search_result")),
                                            "url": obj.get("url").cloned().unwrap_or(serde_json::json!(null)),
                                            "title": obj.get("title").cloned().unwrap_or(serde_json::json!(null)),
                                            "pageAge": obj.get("page_age").cloned().unwrap_or(serde_json::json!(null)),
                                            "encryptedContent": obj.get("encrypted_content").cloned().unwrap_or(serde_json::json!(null)),
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                (serde_json::Value::Array(normalized), false)
                            } else if let Some(obj) = raw_result.as_object() {
                                let error_code = obj
                                    .get("error_code")
                                    .cloned()
                                    .unwrap_or(serde_json::json!(null));
                                (
                                    serde_json::json!({
                                        "type": "web_search_tool_result_error",
                                        "errorCode": error_code,
                                    }),
                                    true,
                                )
                            } else {
                                (
                                    serde_json::json!({
                                        "type": "web_search_tool_result_error",
                                        "errorCode": serde_json::Value::Null,
                                    }),
                                    true,
                                )
                            }
                        } else if t == "web_fetch_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "web_fetch_result" {
                                    let url =
                                        obj.get("url").cloned().unwrap_or(serde_json::json!(null));
                                    let retrieved_at = obj
                                        .get("retrieved_at")
                                        .cloned()
                                        .unwrap_or(serde_json::json!(null));
                                    let content = obj.get("content").and_then(|v| v.as_object());

                                    let mut out_content = serde_json::Map::new();
                                    if let Some(content) = content {
                                        if let Some(v) = content.get("type") {
                                            out_content.insert("type".to_string(), v.clone());
                                        }
                                        if let Some(v) = content.get("title") {
                                            out_content.insert("title".to_string(), v.clone());
                                        }
                                        if let Some(v) = content.get("citations") {
                                            out_content.insert("citations".to_string(), v.clone());
                                        }
                                        if let Some(source) =
                                            content.get("source").and_then(|v| v.as_object())
                                        {
                                            let mut out_source = serde_json::Map::new();
                                            if let Some(v) = source.get("type") {
                                                out_source.insert("type".to_string(), v.clone());
                                            }
                                            if let Some(v) = source
                                                .get("media_type")
                                                .or_else(|| source.get("mediaType"))
                                            {
                                                out_source
                                                    .insert("mediaType".to_string(), v.clone());
                                            }
                                            if let Some(v) = source.get("data") {
                                                out_source.insert("data".to_string(), v.clone());
                                            }
                                            out_content.insert(
                                                "source".to_string(),
                                                serde_json::Value::Object(out_source),
                                            );
                                        }
                                    }

                                    (
                                        serde_json::json!({
                                            "type": "web_fetch_result",
                                            "url": url,
                                            "retrievedAt": retrieved_at,
                                            "content": serde_json::Value::Object(out_content),
                                        }),
                                        false,
                                    )
                                } else if tpe == "web_fetch_tool_result_error" {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::json!(null));
                                    (
                                        serde_json::json!({
                                            "type": "web_fetch_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                } else {
                                    (raw_result.clone(), false)
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else if t == "tool_search_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "tool_search_tool_search_result" {
                                    let refs = obj
                                        .get("tool_references")
                                        .and_then(|v| v.as_array())
                                        .cloned()
                                        .unwrap_or_default()
                                        .into_iter()
                                        .filter_map(|v| v.as_object().cloned())
                                        .map(|ref_obj| {
                                            serde_json::json!({
                                                "type": ref_obj.get("type").cloned().unwrap_or_else(|| serde_json::json!("tool_reference")),
                                                "toolName": ref_obj.get("tool_name").cloned().unwrap_or(serde_json::Value::Null),
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    (serde_json::Value::Array(refs), false)
                                } else {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    (
                                        serde_json::json!({
                                            "type": "tool_search_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else if t == "code_execution_tool_result" {
                            if let Some(obj) = raw_result.as_object() {
                                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

                                if tpe == "code_execution_result" {
                                    (
                                        serde_json::json!({
                                            "type": "code_execution_result",
                                            "stdout": obj.get("stdout").cloned().unwrap_or(serde_json::Value::Null),
                                            "stderr": obj.get("stderr").cloned().unwrap_or(serde_json::Value::Null),
                                            "return_code": obj.get("return_code").cloned().unwrap_or(serde_json::Value::Null),
                                        }),
                                        false,
                                    )
                                } else if tpe == "code_execution_tool_result_error" {
                                    let error_code = obj
                                        .get("error_code")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    (
                                        serde_json::json!({
                                            "type": "code_execution_tool_result_error",
                                            "errorCode": error_code,
                                        }),
                                        true,
                                    )
                                } else {
                                    (raw_result.clone(), false)
                                }
                            } else {
                                (raw_result.clone(), false)
                            }
                        } else {
                            (raw_result.clone(), false)
                        };

                        let mut events = vec![ChatStreamEvent::Custom {
                            event_type: "anthropic:tool-result".to_string(),
                            data: serde_json::json!({
                                "type": "tool-result",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "result": result,
                                "providerExecuted": true,
                                "isError": is_error,
                                "contentBlockIndex": event.index.map(|i| i as u64),
                                "rawContentBlock": content_block,
                            }),
                        }];

                        // Vercel-aligned: emit sources for web search results
                        if t == "web_search_tool_result"
                            && let Some(arr) =
                                content_block.get("content").and_then(|v| v.as_array())
                        {
                            for (i, item) in arr.iter().enumerate() {
                                let Some(obj) = item.as_object() else {
                                    continue;
                                };
                                let Some(url) = obj.get("url").and_then(|v| v.as_str()) else {
                                    continue;
                                };
                                let title = obj.get("title").and_then(|v| v.as_str());
                                let page_age = obj.get("page_age").and_then(|v| v.as_str());
                                let encrypted_content =
                                    obj.get("encrypted_content").and_then(|v| v.as_str());

                                self.record_source(AnthropicSource {
                                    id: format!("{tool_call_id}:{i}"),
                                    source_type: "url".to_string(),
                                    url: Some(url.to_string()),
                                    title: title.map(|s| s.to_string()),
                                    media_type: None,
                                    filename: None,
                                    page_age: page_age.map(|s| s.to_string()),
                                    encrypted_content: encrypted_content.map(|s| s.to_string()),
                                    tool_call_id: Some(tool_call_id.clone()),
                                    provider_metadata: None,
                                });

                                events.push(ChatStreamEvent::Custom {
                                    event_type: "anthropic:source".to_string(),
                                    data: serde_json::json!({
                                        "type": "source",
                                        "sourceType": "url",
                                        "id": format!("{tool_call_id}:{i}"),
                                        "url": url,
                                        "title": title,
                                        "toolCallId": tool_call_id,
                                        "providerMetadata": {
                                            "anthropic": {
                                                "pageAge": page_age,
                                                "encryptedContent": encrypted_content,
                                            }
                                        }
                                    }),
                                });
                            }
                        }

                        events
                    }
                    _ => vec![],
                }
            }
            "content_block_delta" => {
                let mut builder = EventBuilder::new();
                if let Some(delta) = event.delta {
                    match delta.delta_type.as_deref() {
                        Some("text_delta") => {
                            if let Some(text) = delta.text {
                                builder = builder.add_content_delta(text, None);
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(thinking) = delta.thinking {
                                builder = builder.add_thinking_delta(thinking);
                            }
                        }
                        Some("signature_delta") => {
                            if let (Some(idx), Some(sig_delta)) = (event.index, delta.signature)
                                && self
                                    .get_content_block_type(idx)
                                    .is_some_and(|t| t == "thinking")
                            {
                                self.append_thinking_signature(idx, sig_delta.clone());
                                builder = builder.add_custom_event(
                                    "anthropic:thinking-signature-delta".to_string(),
                                    serde_json::json!({
                                        "type": "thinking-signature-delta",
                                        "contentBlockIndex": idx as u64,
                                        "signatureDelta": sig_delta,
                                    }),
                                );
                            }
                        }
                        Some("citations_delta") => {
                            if let (Some(idx), Some(citation)) = (event.index, delta.citation) {
                                // Vercel-aligned: citations deltas are converted into `source` events when possible.
                                let Some(obj) = citation.as_object() else {
                                    return builder.build();
                                };
                                let Some(kind) = obj.get("type").and_then(|v| v.as_str()) else {
                                    return builder.build();
                                };

                                match kind {
                                    "page_location" => {
                                        let doc_index = obj
                                            .get("document_index")
                                            .and_then(|v| v.as_u64())
                                            .map(|u| u as usize);
                                        let Some(doc_index) = doc_index else {
                                            return builder.build();
                                        };
                                        let Some(doc) = self.citation_documents.get(doc_index)
                                        else {
                                            return builder.build();
                                        };

                                        let cited_text =
                                            obj.get("cited_text").and_then(|v| v.as_str());
                                        let start_page =
                                            obj.get("start_page_number").and_then(|v| v.as_u64());
                                        let end_page =
                                            obj.get("end_page_number").and_then(|v| v.as_u64());
                                        let title = obj
                                            .get("document_title")
                                            .and_then(|v| v.as_str())
                                            .filter(|s| !s.is_empty())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| doc.title.clone());

                                        let id = format!(
                                            "doc:{doc_index}:page:{}-{}",
                                            start_page.unwrap_or(0),
                                            end_page.unwrap_or(0)
                                        );

                                        builder = builder.add_custom_event(
                                            "anthropic:source".to_string(),
                                            serde_json::json!({
                                                "type": "source",
                                                "sourceType": "document",
                                                "id": id,
                                                "mediaType": doc.media_type,
                                                "title": title,
                                                "filename": doc.filename,
                                                "providerMetadata": {
                                                    "anthropic": {
                                                        "citedText": cited_text,
                                                        "startPageNumber": start_page,
                                                        "endPageNumber": end_page,
                                                    }
                                                }
                                            }),
                                        );

                                        self.record_source(AnthropicSource {
                                            id,
                                            source_type: "document".to_string(),
                                            url: None,
                                            title: Some(title),
                                            media_type: Some(doc.media_type.clone()),
                                            filename: doc.filename.clone(),
                                            page_age: None,
                                            encrypted_content: None,
                                            tool_call_id: None,
                                            provider_metadata: Some(serde_json::json!({
                                                "citedText": cited_text,
                                                "startPageNumber": start_page,
                                                "endPageNumber": end_page,
                                            })),
                                        });
                                    }
                                    "char_location" => {
                                        let doc_index = obj
                                            .get("document_index")
                                            .and_then(|v| v.as_u64())
                                            .map(|u| u as usize);
                                        let Some(doc_index) = doc_index else {
                                            return builder.build();
                                        };
                                        let Some(doc) = self.citation_documents.get(doc_index)
                                        else {
                                            return builder.build();
                                        };

                                        let cited_text =
                                            obj.get("cited_text").and_then(|v| v.as_str());
                                        let start_char =
                                            obj.get("start_char_index").and_then(|v| v.as_u64());
                                        let end_char =
                                            obj.get("end_char_index").and_then(|v| v.as_u64());
                                        let title = obj
                                            .get("document_title")
                                            .and_then(|v| v.as_str())
                                            .filter(|s| !s.is_empty())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| doc.title.clone());

                                        let id = format!(
                                            "doc:{doc_index}:char:{}-{}",
                                            start_char.unwrap_or(0),
                                            end_char.unwrap_or(0)
                                        );

                                        builder = builder.add_custom_event(
                                            "anthropic:source".to_string(),
                                            serde_json::json!({
                                                "type": "source",
                                                "sourceType": "document",
                                                "id": id,
                                                "mediaType": doc.media_type,
                                                "title": title,
                                                "filename": doc.filename,
                                                "providerMetadata": {
                                                    "anthropic": {
                                                        "citedText": cited_text,
                                                        "startCharIndex": start_char,
                                                        "endCharIndex": end_char,
                                                    }
                                                }
                                            }),
                                        );

                                        self.record_source(AnthropicSource {
                                            id,
                                            source_type: "document".to_string(),
                                            url: None,
                                            title: Some(title),
                                            media_type: Some(doc.media_type.clone()),
                                            filename: doc.filename.clone(),
                                            page_age: None,
                                            encrypted_content: None,
                                            tool_call_id: None,
                                            provider_metadata: Some(serde_json::json!({
                                                "citedText": cited_text,
                                                "startCharIndex": start_char,
                                                "endCharIndex": end_char,
                                            })),
                                        });
                                    }
                                    _ => {
                                        // Vercel-aligned: ignore other citation kinds in the stream converter.
                                        let _ = idx;
                                    }
                                }
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(partial_json) = delta.partial_json
                                && !partial_json.is_empty()
                                && let Some(idx) = event.index
                                && let Ok(map) = self.tool_use_ids_by_index.lock()
                                && let Some(tool_call_id) = map.get(&idx)
                            {
                                builder = builder.add_tool_call_delta(
                                    tool_call_id.clone(),
                                    None,
                                    Some(partial_json),
                                    Some(idx),
                                );
                            }
                        }
                        _ => {
                            if let Some(text) = delta.text {
                                builder = builder.add_content_delta(text, None);
                            }
                            if let Some(thinking) = delta.thinking {
                                builder = builder.add_thinking_delta(thinking);
                            }
                            if let Some(partial_json) = delta.partial_json
                                && !partial_json.is_empty()
                                && let Some(idx) = event.index
                                && let Ok(map) = self.tool_use_ids_by_index.lock()
                                && let Some(tool_call_id) = map.get(&idx)
                            {
                                builder = builder.add_tool_call_delta(
                                    tool_call_id.clone(),
                                    None,
                                    Some(partial_json),
                                    Some(idx),
                                );
                            }
                        }
                    };
                }
                builder.build()
            }
            "message_delta" => {
                let mut builder = EventBuilder::new();

                // Thinking (if present)
                if let Some(delta) = &event.delta
                    && let Some(thinking) = &delta.thinking
                    && !thinking.is_empty()
                {
                    builder = builder.add_thinking_delta(thinking.clone());
                }

                // Usage update
                if let Some(usage) = &event.usage {
                    let usage_info = Usage {
                        prompt_tokens: usage.input_tokens.unwrap_or(0),
                        completion_tokens: usage.output_tokens.unwrap_or(0),
                        total_tokens: usage.input_tokens.unwrap_or(0)
                            + usage.output_tokens.unwrap_or(0),
                        #[allow(deprecated)]
                        cached_tokens: None,
                        #[allow(deprecated)]
                        reasoning_tokens: None,
                        prompt_tokens_details: None,
                        completion_tokens_details: None,
                    };
                    builder = builder.add_usage_update(usage_info);
                }

                // Finish reason -> StreamEnd
                if let Some(delta) = &event.delta
                    && let Some(stop_reason) = &delta.stop_reason
                {
                    let reason = match stop_reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "stop_sequence" => FinishReason::Stop,
                        "tool_use" => FinishReason::ToolCalls,
                        "refusal" => FinishReason::ContentFilter,
                        _ => FinishReason::Stop,
                    };

                    // Mark that StreamEnd is being emitted
                    self.state_tracker.mark_stream_ended();

                    let response = ChatResponse {
                        id: None,
                        model: None,
                        content: MessageContent::Text("".to_string()),
                        usage: None, // usage already emitted as UsageUpdate above if present
                        finish_reason: Some(reason),
                        audio: None,
                        system_fingerprint: None,
                        service_tier: None,
                        warnings: None,
                        provider_metadata: self.build_stream_provider_metadata(),
                    };
                    builder = builder.add_stream_end(response);
                }

                builder.build()
            }
            "message_stop" => {
                // Mark that StreamEnd is being emitted
                self.state_tracker.mark_stream_ended();

                let response = ChatResponse {
                    id: None,
                    model: None,
                    content: MessageContent::Text("".to_string()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: self.build_stream_provider_metadata(),
                };
                EventBuilder::new().add_stream_end(response).build()
            }
            _ => vec![],
        }
    }
}

impl SseEventConverter for AnthropicEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Log the raw event data for debugging
            tracing::debug!("Anthropic SSE event: {}", event.data);

            // Handle special cases first
            if event.data.trim() == "[DONE]" {
                return vec![];
            }

            // Try to parse as standard Anthropic event
            match crate::streaming::parse_json_with_repair::<AnthropicStreamEvent>(&event.data) {
                Ok(anthropic_event) => self
                    .convert_anthropic_event(anthropic_event)
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    // Enhanced error reporting with event data
                    tracing::warn!("Failed to parse Anthropic SSE event: {}", e);
                    tracing::warn!("Raw event data: {}", event.data);

                    // Try to parse as a generic JSON to see if it's a different format
                    if let Ok(generic_json) =
                        crate::streaming::parse_json_with_repair::<serde_json::Value>(&event.data)
                    {
                        tracing::warn!("Event parsed as generic JSON: {:#}", generic_json);

                        // Check if this looks like an error response
                        if let Some(error_obj) = generic_json.get("error") {
                            let error_message = error_obj
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown error");

                            return vec![Ok(ChatStreamEvent::Error {
                                error: format!("Anthropic API error: {}", error_message),
                            })];
                        }
                    }

                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Anthropic event: {}. Raw data: {}",
                        e, event.data
                    )))]
                }
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Anthropic normally emits StreamEnd via message_stop event in convert_event.
        // If we reach here without seeing message_stop, the model has not transmitted
        // a finish reason (e.g., connection lost, server error, client cancelled).
        // Always emit StreamEnd with Unknown reason so users can detect this.

        // Check if StreamEnd was already emitted
        if !self.state_tracker.needs_stream_end() {
            return None; // StreamEnd already emitted
        }

        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text("".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Unknown),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: self.build_stream_provider_metadata(),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

// Legacy AnthropicStreaming client has been removed in favor of the unified HttpChatExecutor.
// The AnthropicEventConverter is still used for SSE event conversion in tests.

#[cfg(test)]
mod tests {
    use super::super::params::AnthropicParams;
    use super::*;
    use eventsource_stream::Event;

    fn create_test_config() -> AnthropicParams {
        AnthropicParams::default()
    }

    #[tokio::test]
    async fn test_anthropic_streaming_conversion() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        // Test content delta conversion
        let event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(!result.is_empty());

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result.first() {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event");
        }
    }

    #[tokio::test]
    async fn test_anthropic_streaming_error_event_is_exposed() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let event = Event {
            event: "".to_string(),
            data:
                r#"{"type":"error","error":{"type":"overloaded_error","message":"rate limited"}}"#
                    .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        let err = result
            .iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::Error { .. })));
        match err {
            Some(Ok(ChatStreamEvent::Error { error })) => {
                assert!(error.contains("rate limited"));
            }
            other => panic!("Expected Error event, got: {other:?}"),
        }
    }

    // Removed legacy merge-provider-params test; behavior now covered by transformers

    #[tokio::test]
    async fn test_anthropic_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let result = converter.handle_stream_end();
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::StreamEnd { .. })) = result {
            // Success
        } else {
            panic!("Expected StreamEnd event");
        }
    }

    #[tokio::test]
    async fn emits_custom_events_for_server_tool_use_and_results() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let tool_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
            }
            other => panic!("Expected Custom event, got {:?}", other),
        }

        let tool_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Rust","url":"https://www.rust-lang.org","encrypted_content":"..."}]}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_result_event).await;
        assert_eq!(evs.len(), 2);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
                assert!(data["result"].is_array());
            }
            other => panic!("Expected Custom event, got {:?}", other),
        }

        match evs.get(1).unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:source");
                assert_eq!(data["sourceType"], serde_json::json!("url"));
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
            }
            other => panic!("Expected source Custom event, got {:?}", other),
        }

        let web_fetch_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":2,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_2","content":{"type":"web_fetch_result","url":"https://example.com","retrieved_at":"2025-01-01T00:00:00Z","content":{"type":"document","title":"Example","citations":{"enabled":true},"source":{"type":"text","media_type":"text/plain","data":"hello"}}}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(web_fetch_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_2"));
                assert_eq!(data["toolName"], serde_json::json!("web_fetch"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert_eq!(
                    data["result"]["type"],
                    serde_json::json!("web_fetch_result")
                );
                assert_eq!(
                    data["result"]["retrievedAt"],
                    serde_json::json!("2025-01-01T00:00:00Z")
                );
                assert_eq!(
                    data["result"]["content"]["source"]["mediaType"],
                    serde_json::json!("text/plain")
                );
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }

        let tool_search_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":3,"content_block":{"type":"server_tool_use","id":"srvtoolu_3","name":"tool_search_tool_regex","input":{"pattern":"weather","limit":2}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_search_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
                assert_eq!(data["toolName"], serde_json::json!("tool_search"));
                assert_eq!(data["input"]["pattern"], serde_json::json!("weather"));
            }
            other => panic!("Expected tool-call Custom event, got {:?}", other),
        }

        let tool_search_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":4,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_3","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(tool_search_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
                assert_eq!(data["toolName"], serde_json::json!("tool_search"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert!(data["result"].is_array());
                assert_eq!(
                    data["result"][0]["toolName"],
                    serde_json::json!("get_weather")
                );
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }

        let code_exec_call_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":5,"content_block":{"type":"server_tool_use","id":"srvtoolu_4","name":"code_execution","input":{"code":"print(1+1)"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(code_exec_call_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
                assert_eq!(data["toolName"], serde_json::json!("code_execution"));
            }
            other => panic!("Expected tool-call Custom event, got {:?}", other),
        }

        let code_exec_result_event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":6,"content_block":{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_4","content":{"type":"code_execution_result","stdout":"2\n","stderr":"","return_code":0}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(code_exec_result_event).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
                assert_eq!(data["toolName"], serde_json::json!("code_execution"));
                assert_eq!(data["isError"], serde_json::json!(false));
                assert_eq!(
                    data["result"]["type"],
                    serde_json::json!("code_execution_result")
                );
                assert_eq!(data["result"]["return_code"], serde_json::json!(0));
            }
            other => panic!("Expected tool-result Custom event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn emits_tool_call_delta_for_local_tool_use_input_json_delta() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"tokyo"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(start).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(function_name.as_deref(), Some("get_weather"));
                assert!(arguments_delta.is_none());
                assert_eq!(*index, Some(0));
            }
            other => panic!("Expected ToolCallDelta, got {:?}", other),
        }

        let delta = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"unit\":\"c\"}"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let evs = converter.convert_event(delta).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                assert_eq!(id, "toolu_1");
                assert!(function_name.is_none());
                assert_eq!(arguments_delta.as_deref(), Some("{\"unit\":\"c\"}"));
                assert_eq!(*index, Some(0));
            }
            other => panic!("Expected ToolCallDelta, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn captures_thinking_signature_delta_and_exposes_in_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let thinking_start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let _ = converter.convert_event(thinking_start).await;

        let sig_delta = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(sig_delta).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:thinking-signature-delta");
                assert_eq!(data["signatureDelta"], serde_json::json!("sig-1"));
            }
            other => panic!("Expected signature delta Custom event, got {:?}", other),
        }

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(stop).await;
        let end = evs
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                assert_eq!(
                    meta.get("anthropic")
                        .unwrap()
                        .get("thinking_signature")
                        .unwrap(),
                    &serde_json::json!("sig-1")
                );
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn captures_redacted_thinking_data_in_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let redacted_start = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"abc123"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(redacted_start).await;
        assert_eq!(evs.len(), 1);
        match evs.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:reasoning-start");
                assert_eq!(data["redactedData"], serde_json::json!("abc123"));
            }
            other => panic!("Expected reasoning-start Custom event, got {:?}", other),
        }

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let evs = converter.convert_event(stop).await;
        let end = evs
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                assert_eq!(
                    meta.get("anthropic")
                        .unwrap()
                        .get("redacted_thinking_data")
                        .unwrap(),
                    &serde_json::json!("abc123")
                );
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn emits_source_event_for_citations_delta_with_document_location() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
            AnthropicCitationDocument {
                title: "Doc A".to_string(),
                filename: Some("a.pdf".to_string()),
                media_type: "application/pdf".to_string(),
            },
        ]);

        let ev = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let out = converter.convert_event(ev).await;
        assert_eq!(out.len(), 1);
        match out.first().unwrap().as_ref().unwrap() {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:source");
                assert_eq!(data["sourceType"], serde_json::json!("document"));
                assert_eq!(data["mediaType"], serde_json::json!("application/pdf"));
                assert_eq!(data["title"], serde_json::json!("Doc A"));
                assert_eq!(data["filename"], serde_json::json!("a.pdf"));
                assert_eq!(
                    data["providerMetadata"]["anthropic"]["startPageNumber"],
                    serde_json::json!(1)
                );
            }
            other => panic!("Expected source Custom event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn accumulates_sources_into_stream_end_provider_metadata() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
            AnthropicCitationDocument {
                title: "Doc A".to_string(),
                filename: Some("a.pdf".to_string()),
                media_type: "application/pdf".to_string(),
            },
        ]);

        let ev = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };
        let _ = converter.convert_event(ev).await;

        let stop = Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(stop).await;
        let end = out
            .into_iter()
            .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
            .expect("stream end");
        match end.unwrap() {
            ChatStreamEvent::StreamEnd { response } => {
                let meta = response.provider_metadata.expect("provider_metadata");
                let anthropic = meta.get("anthropic").expect("anthropic");
                let sources = anthropic
                    .get("sources")
                    .and_then(|v| v.as_array())
                    .expect("sources array");
                assert_eq!(sources.len(), 1);
                assert_eq!(sources[0]["source_type"], serde_json::json!("document"));
                assert_eq!(
                    sources[0]["media_type"],
                    serde_json::json!("application/pdf")
                );
                assert_eq!(sources[0]["filename"], serde_json::json!("a.pdf"));
            }
            _ => unreachable!(),
        }
    }
}
