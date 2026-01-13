//! Gemini streaming implementation using eventsource-stream (protocol layer)
//!
//! Provides SSE event conversion for Gemini streaming responses.
//! The legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.

use super::types::GeminiConfig;
use crate::error::LlmError;
use crate::streaming::SseEventConverter;
use crate::streaming::{
    ChatStreamEvent, LanguageModelV3Source, LanguageModelV3StreamPart, StreamStateTracker,
    V3UnsupportedPartBehavior,
};
use crate::types::Usage;
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default, Clone)]
struct GeminiFunctionCallSerializeState {
    name: Option<String>,
    arguments: String,
    last_emitted_args_json: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct GeminiSerializeState {
    function_calls_by_id: std::collections::HashMap<String, GeminiFunctionCallSerializeState>,
}

/// Gemini stream response structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiStreamResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<super::types::PromptFeedback>,
    /// Gateway-only metadata used for stable tool call id propagation in re-serialized streams.
    #[serde(default)]
    siumai: Option<GeminiSiumaiMetadata>,
}

/// Gemini candidate structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    grounding_metadata: Option<super::types::GroundingMetadata>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlContextMetadata")]
    url_context_metadata: Option<super::types::UrlContextMetadata>,
    #[serde(default, rename = "safetyRatings")]
    safety_ratings: Vec<super::types::SafetyRating>,
}

/// Gemini content structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    #[allow(dead_code)]
    // Role appears in some responses but is not required by our unified event model
    role: Option<String>,
}

/// Gemini part structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    /// Optional. Whether this is a thought summary (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    /// Optional. An opaque signature for the thought so it can be reused in subsequent requests.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
    thought_signature: Option<String>,
    #[serde(rename = "executableCode", skip_serializing_if = "Option::is_none")]
    executable_code: Option<GeminiExecutableCode>,
    #[serde(
        rename = "codeExecutionResult",
        skip_serializing_if = "Option::is_none"
    )]
    code_execution_result: Option<GeminiCodeExecutionResult>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiExecutableCode {
    language: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiCodeExecutionResult {
    outcome: Option<String>,
    output: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
struct GeminiSiumaiMetadata {
    #[serde(rename = "toolCallId")]
    tool_call_id: Option<String>,
}

/// Gemini usage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
    /// Number of tokens used for thinking (only for thinking models)
    #[serde(rename = "thoughtsTokenCount")]
    thoughts_token_count: Option<u32>,
}

/// Gemini event converter
#[derive(Clone)]
pub struct GeminiEventConverter {
    config: GeminiConfig,
    /// Track if StreamStart has been emitted
    state_tracker: StreamStateTracker,
    /// Deduplicate sources across stream chunks
    seen_source_keys: Arc<Mutex<std::collections::HashSet<String>>>,
    /// Monotonic id counter for emitted `gemini:source` events
    next_source_id: Arc<AtomicU64>,
    /// Monotonic id counter for emitted text/reasoning blocks (Vercel-aligned custom events)
    next_block_id: Arc<AtomicU64>,
    /// Monotonic id counter for emitted tool calls (client-executed functionCall parts)
    next_tool_call_id: Arc<AtomicU64>,
    /// Pair executableCode -> codeExecutionResult across chunks
    pending_code_execution_id: Arc<Mutex<Option<String>>>,
    /// Track the active reasoning block id (for Vercel-aligned custom reasoning events)
    current_reasoning_block_id: Arc<Mutex<Option<String>>>,

    serialize_state: Arc<Mutex<GeminiSerializeState>>,
    v3_unsupported_part_behavior: V3UnsupportedPartBehavior,
    emit_v3_tool_call_parts: bool,
    emit_function_response_tool_results: bool,
}

impl GeminiEventConverter {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            state_tracker: StreamStateTracker::new(),
            seen_source_keys: Arc::new(Mutex::new(std::collections::HashSet::new())),
            next_source_id: Arc::new(AtomicU64::new(0)),
            next_block_id: Arc::new(AtomicU64::new(0)),
            next_tool_call_id: Arc::new(AtomicU64::new(0)),
            pending_code_execution_id: Arc::new(Mutex::new(None)),
            current_reasoning_block_id: Arc::new(Mutex::new(None)),
            serialize_state: Arc::new(Mutex::new(GeminiSerializeState::default())),
            v3_unsupported_part_behavior: V3UnsupportedPartBehavior::default(),
            emit_v3_tool_call_parts: false,
            emit_function_response_tool_results: false,
        }
    }

    pub fn with_v3_unsupported_part_behavior(
        mut self,
        behavior: V3UnsupportedPartBehavior,
    ) -> Self {
        self.v3_unsupported_part_behavior = behavior;
        self
    }

    /// Emit Vercel-aligned v3 `tool-call` parts (as `ChatStreamEvent::Custom`) for `functionCall`
    /// stream chunks.
    ///
    /// Notes:
    /// - When enabled, the converter emits `ChatStreamEvent::Custom` events instead of
    ///   `ChatStreamEvent::ToolCallDelta` for `functionCall` chunks.
    /// - This is useful for gateways/proxies that want to preserve Gemini-only metadata such as
    ///   `thoughtSignature` across transcoding.
    pub fn with_emit_v3_tool_call_parts(mut self, enabled: bool) -> Self {
        self.emit_v3_tool_call_parts = enabled;
        self
    }

    /// Emit Gemini `functionResponse` frames for v3 `tool-result` parts when serializing.
    ///
    /// Notes:
    /// - This is primarily a gateway/proxy feature: Gemini responses usually don't contain
    ///   functionResponse parts; they are commonly sent in the *next request*.
    /// - When enabled, Siumai will include a `siumai.toolCallId` field in the JSON frame so the
    ///   decoder can associate results with the original tool call id.
    pub fn with_emit_function_response_tool_results(mut self, enabled: bool) -> Self {
        self.emit_function_response_tool_results = enabled;
        self
    }

    fn provider_metadata_key(&self) -> &'static str {
        if let Some(override_key) = self.config.provider_metadata_key.as_deref() {
            let key = override_key.trim().to_ascii_lowercase();
            if key.contains("vertex") {
                return "vertex";
            }
            if key.contains("google") {
                return "google";
            }
        }

        if self.config.base_url.contains("aiplatform.googleapis.com")
            || self.config.base_url.contains("vertex")
        {
            "vertex"
        } else {
            "google"
        }
    }

    fn thought_signature_provider_metadata_value(
        &self,
        sig: Option<&String>,
    ) -> Option<serde_json::Value> {
        let sig = sig?;
        if sig.trim().is_empty() {
            return None;
        }
        let key = self.provider_metadata_key();
        Some(serde_json::json!({ key: { "thoughtSignature": sig } }))
    }

    fn take_reasoning_block_id(&self) -> Option<String> {
        let mut lock = self
            .current_reasoning_block_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        lock.take()
    }

    /// Convert Gemini stream response to multiple ChatStreamEvents
    async fn convert_gemini_response_async(
        &self,
        response: GeminiStreamResponse,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
            builder = builder.add_stream_start(self.create_stream_start_metadata());
        }

        // Process content - support multiple candidates/parts per chunk
        let texts = self.extract_all_texts(&response);
        if !texts.is_empty() {
            for t in texts {
                builder = builder.add_content_delta(t, None);
            }
        }

        // Process thinking content (if supported).
        // Also emit Vercel-aligned custom reasoning events with providerMetadata.thoughtSignature.
        for (thinking, sig) in self.extract_thinking_parts(&response) {
            if thinking.is_empty() {
                continue;
            }

            let provider_metadata = self.thought_signature_provider_metadata_value(sig.as_ref());

            // reasoning-start
            let id = {
                let mut lock = self
                    .current_reasoning_block_id
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(id) = lock.as_ref() {
                    id.clone()
                } else {
                    let next = self.next_block_id.fetch_add(1, Ordering::Relaxed);
                    let id = next.to_string();
                    *lock = Some(id.clone());
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "type".to_string(),
                        serde_json::Value::String("reasoning-start".to_string()),
                    );
                    obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
                    if let Some(pm) = provider_metadata.clone() {
                        obj.insert("providerMetadata".to_string(), pm);
                    }
                    builder = builder.add_custom_event(
                        "gemini:reasoning".to_string(),
                        serde_json::Value::Object(obj),
                    );
                    id
                }
            };

            // reasoning-delta
            let mut obj = serde_json::Map::new();
            obj.insert(
                "type".to_string(),
                serde_json::Value::String("reasoning-delta".to_string()),
            );
            obj.insert("id".to_string(), serde_json::Value::String(id));
            obj.insert(
                "delta".to_string(),
                serde_json::Value::String(thinking.clone()),
            );
            if let Some(pm) = provider_metadata {
                obj.insert("providerMetadata".to_string(), pm);
            }
            builder = builder.add_custom_event(
                "gemini:reasoning".to_string(),
                serde_json::Value::Object(obj),
            );

            builder = builder.add_thinking_delta(thinking);
        }

        // Process provider-executed tool parts (Vercel-aligned tool-call/tool-result events).
        for data in self.extract_code_execution_events(&response) {
            builder = builder.add_custom_event("gemini:tool".to_string(), data);
        }

        // Process client-executed tool calls (functionCall parts).
        for (tool_name, args_json, thought_sig) in self.extract_function_call_events(&response) {
            let id_num = self.next_tool_call_id.fetch_add(1, Ordering::Relaxed);
            let call_id = format!("call_{id_num}");

            if self.emit_v3_tool_call_parts {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "type".to_string(),
                    serde_json::Value::String("tool-call".to_string()),
                );
                obj.insert("toolCallId".to_string(), serde_json::Value::String(call_id));
                obj.insert("toolName".to_string(), serde_json::Value::String(tool_name));
                obj.insert("input".to_string(), serde_json::Value::String(args_json));
                if let Some(pm) =
                    self.thought_signature_provider_metadata_value(thought_sig.as_ref())
                {
                    obj.insert("providerMetadata".to_string(), pm);
                }
                builder = builder
                    .add_custom_event("gemini:tool".to_string(), serde_json::Value::Object(obj));
            } else {
                builder =
                    builder.add_tool_call_delta(call_id, Some(tool_name), Some(args_json), None);
            }
        }

        // Process client-provided tool results (functionResponse parts) as v3 tool-result events.
        for (tool_name, result, thought_sig) in self.extract_function_response_events(&response) {
            let id_num = self.next_tool_call_id.fetch_add(1, Ordering::Relaxed);
            let tool_call_id = response
                .siumai
                .as_ref()
                .and_then(|m| m.tool_call_id.clone())
                .unwrap_or_else(|| format!("call_{id_num}"));

            let mut obj = serde_json::Map::new();
            obj.insert(
                "type".to_string(),
                serde_json::Value::String("tool-result".to_string()),
            );
            obj.insert(
                "toolCallId".to_string(),
                serde_json::Value::String(tool_call_id),
            );
            obj.insert("toolName".to_string(), serde_json::Value::String(tool_name));
            obj.insert("result".to_string(), result);
            if let Some(pm) = self.thought_signature_provider_metadata_value(thought_sig.as_ref()) {
                obj.insert("providerMetadata".to_string(), pm);
            }

            builder =
                builder.add_custom_event("gemini:tool".to_string(), serde_json::Value::Object(obj));
        }

        // Process usage update if available
        if let Some(usage) = self.extract_usage(&response) {
            builder = builder.add_usage_update(usage);
        }

        // Emit normalized sources (Vercel-aligned) if grounding chunks are present.
        for data in self.extract_source_events(&response) {
            builder = builder.add_custom_event("gemini:source".to_string(), data);
        }

        // Handle completion/finish reason
        if let Some(end_response) = self.extract_completion(&response) {
            if let Some(id) = self.take_reasoning_block_id() {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "type".to_string(),
                    serde_json::Value::String("reasoning-end".to_string()),
                );
                obj.insert("id".to_string(), serde_json::Value::String(id));
                builder = builder.add_custom_event(
                    "gemini:reasoning".to_string(),
                    serde_json::Value::Object(obj),
                );
            }
            builder = builder.add_stream_end(end_response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
    }

    /// Extract content from Gemini response
    #[allow(dead_code)]
    fn extract_content(&self, response: &GeminiStreamResponse) -> Option<String> {
        let candidates = response.candidates.as_ref()?;
        for cand in candidates {
            if let Some(content) = &cand.content
                && let Some(parts) = &content.parts
            {
                for part in parts {
                    if let Some(text) = &part.text
                        && !text.is_empty()
                    {
                        return Some(text.clone());
                    }
                }
            }
        }
        None
    }

    /// Extract all non-empty texts across candidates/parts (for multi-candidate streams)
    fn extract_all_texts(&self, response: &GeminiStreamResponse) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(candidates) = &response.candidates {
            for cand in candidates {
                if let Some(content) = &cand.content
                    && let Some(parts) = &content.parts
                {
                    for part in parts {
                        if let Some(text) = &part.text
                            && !text.is_empty()
                            && !part.thought.unwrap_or(false)
                        {
                            out.push(text.clone());
                        }
                    }
                }
            }
        }
        out
    }

    fn extract_function_call_events(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<(String, String, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for cand in candidates {
            let Some(content) = cand.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                let Some(call) = part.function_call.as_ref() else {
                    continue;
                };

                let args = call.args.clone().unwrap_or_else(|| serde_json::json!({}));
                let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                out.push((call.name.clone(), args_json, part.thought_signature.clone()));
            }
        }

        out
    }

    fn extract_function_response_events(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<(String, serde_json::Value, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for cand in candidates {
            let Some(content) = cand.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                let Some(res) = part.function_response.as_ref() else {
                    continue;
                };

                out.push((
                    res.name.clone(),
                    res.response.clone(),
                    part.thought_signature.clone(),
                ));
            }
        }

        out
    }

    fn has_function_call_parts(&self, response: &GeminiStreamResponse) -> bool {
        let Some(candidates) = response.candidates.as_ref() else {
            return false;
        };
        for cand in candidates {
            if let Some(content) = cand.content.as_ref()
                && let Some(parts) = content.parts.as_ref()
                && parts.iter().any(|p| p.function_call.is_some())
            {
                return true;
            }
        }
        false
    }

    fn extract_code_execution_events(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<serde_json::Value> {
        let mut out: Vec<serde_json::Value> = Vec::new();

        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        for candidate in candidates {
            let Some(content) = candidate.content.as_ref() else {
                continue;
            };
            let Some(parts) = content.parts.as_ref() else {
                continue;
            };

            for part in parts {
                if let Some(exec) = part.executable_code.as_ref() {
                    let id = {
                        let id = format!("call_{}", uuid::Uuid::new_v4());
                        if let Ok(mut lock) = self.pending_code_execution_id.lock() {
                            *lock = Some(id.clone());
                        }
                        id
                    };

                    let input = serde_json::json!({
                        "language": exec.language.clone().unwrap_or_else(|| "PYTHON".to_string()),
                        "code": exec.code.clone().unwrap_or_default()
                    });

                    out.push(serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": id,
                        "toolName": "code_execution",
                        "providerExecuted": true,
                        "input": input
                    }));
                }

                if let Some(res) = part.code_execution_result.as_ref() {
                    let id = if let Ok(mut lock) = self.pending_code_execution_id.lock() {
                        lock.take()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()))
                    } else {
                        format!("call_{}", uuid::Uuid::new_v4())
                    };

                    out.push(serde_json::json!({
                        "type": "tool-result",
                        "toolCallId": id,
                        "toolName": "code_execution",
                        "providerExecuted": true,
                        "result": {
                            "outcome": res.outcome.clone().unwrap_or_else(|| "OUTCOME_OK".to_string()),
                            "output": res.output.clone().unwrap_or_default()
                        }
                    }));
                }
            }
        }

        out
    }

    /// Extract thinking parts (text + thoughtSignature) from Gemini response.
    fn extract_thinking_parts(
        &self,
        response: &GeminiStreamResponse,
    ) -> Vec<(String, Option<String>)> {
        let mut out = Vec::new();
        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };
        for cand in candidates {
            if let Some(content) = cand.content.as_ref()
                && let Some(parts) = content.parts.as_ref()
            {
                for part in parts {
                    if let Some(text) = part.text.as_ref()
                        && !text.is_empty()
                        && part.thought.unwrap_or(false)
                    {
                        out.push((text.clone(), part.thought_signature.clone()));
                    }
                }
            }
        }
        out
    }

    /// Extract Vercel-aligned source events from grounding metadata (deduplicated across stream).
    fn extract_source_events(&self, response: &GeminiStreamResponse) -> Vec<serde_json::Value> {
        let mut out: Vec<serde_json::Value> = Vec::new();

        let Some(candidates) = response.candidates.as_ref() else {
            return out;
        };

        let mut seen = self
            .seen_source_keys
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        for cand in candidates {
            let sources = super::sources::extract_sources(cand.grounding_metadata.as_ref());
            for source in sources {
                let key = super::sources::source_key(&source);
                if !seen.insert(key) {
                    continue;
                }

                let id_num = self.next_source_id.fetch_add(1, Ordering::Relaxed);
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "type".to_string(),
                    serde_json::Value::String("source".to_string()),
                );
                obj.insert(
                    "sourceType".to_string(),
                    serde_json::Value::String(source.source_type.clone()),
                );
                obj.insert(
                    "id".to_string(),
                    serde_json::Value::String(format!("src_{id_num}")),
                );

                if let Some(url) = source.url {
                    obj.insert("url".to_string(), serde_json::Value::String(url));
                }
                if let Some(title) = source.title {
                    obj.insert("title".to_string(), serde_json::Value::String(title));
                }
                if let Some(media_type) = source.media_type {
                    obj.insert(
                        "mediaType".to_string(),
                        serde_json::Value::String(media_type),
                    );
                }
                if let Some(filename) = source.filename {
                    obj.insert("filename".to_string(), serde_json::Value::String(filename));
                }

                out.push(serde_json::Value::Object(obj));
            }
        }

        out
    }

    /// Extract completion information
    fn extract_completion(&self, response: &GeminiStreamResponse) -> Option<ChatResponse> {
        let candidate = response.candidates.as_ref()?.first()?;

        if let Some(finish_reason) = &candidate.finish_reason {
            let finish_reason = match finish_reason.as_str() {
                "STOP" => {
                    if self.has_function_call_parts(response) {
                        FinishReason::ToolCalls
                    } else {
                        FinishReason::Stop
                    }
                }
                "MAX_TOKENS" => FinishReason::Length,
                "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT"
                | "SPII" => FinishReason::ContentFilter,
                "MALFORMED_FUNCTION_CALL" => FinishReason::Error,
                other => FinishReason::Other(other.to_string()),
            };

            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let provider_metadata = {
                let provider_key = self.provider_metadata_key();
                let mut meta: std::collections::HashMap<String, serde_json::Value> =
                    std::collections::HashMap::new();

                if let Some(m) = &candidate.grounding_metadata
                    && let Ok(v) = serde_json::to_value(m)
                {
                    meta.insert("groundingMetadata".to_string(), v);
                }
                if let Some(m) = &candidate.url_context_metadata
                    && let Ok(v) = serde_json::to_value(m)
                {
                    meta.insert("urlContextMetadata".to_string(), v);
                }
                if !candidate.safety_ratings.is_empty()
                    && let Ok(v) = serde_json::to_value(&candidate.safety_ratings)
                {
                    meta.insert("safetyRatings".to_string(), v);
                }
                if let Some(m) = response.prompt_feedback.as_ref()
                    && let Ok(v) = serde_json::to_value(m)
                {
                    meta.insert("promptFeedback".to_string(), v);
                }

                if let Some(m) = response.usage_metadata.as_ref()
                    && let Ok(v) = serde_json::to_value(m)
                {
                    meta.insert("usageMetadata".to_string(), v);
                }

                let sources =
                    super::sources::extract_sources(candidate.grounding_metadata.as_ref());
                if !sources.is_empty()
                    && let Ok(v) = serde_json::to_value(sources)
                {
                    meta.insert("sources".to_string(), v);
                }

                if meta.is_empty() {
                    None
                } else {
                    let mut all = std::collections::HashMap::new();
                    all.insert(provider_key.to_string(), meta);
                    Some(all)
                }
            };

            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: None,
                finish_reason: Some(finish_reason),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata,
            };

            Some(response)
        } else {
            None
        }
    }

    /// Extract usage information
    fn extract_usage(&self, response: &GeminiStreamResponse) -> Option<Usage> {
        if let Some(meta) = &response.usage_metadata {
            return Some(Usage {
                prompt_tokens: meta.prompt_token_count.unwrap_or(0),
                completion_tokens: meta.candidates_token_count.unwrap_or(0),
                total_tokens: meta.total_token_count.unwrap_or(
                    meta.prompt_token_count.unwrap_or(0) + meta.candidates_token_count.unwrap_or(0),
                ),
                #[allow(deprecated)]
                cached_tokens: None,
                #[allow(deprecated)]
                reasoning_tokens: meta.thoughts_token_count,
                prompt_tokens_details: None,
                completion_tokens_details: meta.thoughts_token_count.map(|reasoning| {
                    crate::types::CompletionTokensDetails {
                        reasoning_tokens: Some(reasoning),
                        audio_tokens: None,
                        accepted_prediction_tokens: None,
                        rejected_prediction_tokens: None,
                    }
                }),
            });
        }
        None
    }

    /// Create StreamStart metadata
    fn create_stream_start_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,                               // Gemini doesn't provide ID in stream events
            model: Some(self.config.model.clone()), // Use model from config
            created: Some(chrono::Utc::now()),
            provider: "gemini".to_string(),
            request_id: None,
        }
    }
}

impl SseEventConverter for GeminiEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Skip done marker or empty events
            if event.data.trim() == "[DONE]" || event.data.trim().is_empty() {
                return vec![];
            }

            // Parse the JSON data from the SSE event
            // Feature-gated behavior:
            // - Without `json-repair`: strict parsing to surface errors
            // - With `json-repair`: tolerant parsing using jsonrepair
            #[cfg(not(feature = "json-repair"))]
            let parsed: Result<GeminiStreamResponse, _> = serde_json::from_str(&event.data);
            #[cfg(feature = "json-repair")]
            let parsed: Result<GeminiStreamResponse, _> =
                crate::streaming::parse_json_with_repair::<GeminiStreamResponse>(&event.data);

            match parsed {
                Ok(gemini_response) => self
                    .convert_gemini_response_async(gemini_response)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Gemini SSE JSON: {e}"
                    )))]
                }
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Gemini normally emits finish_reason in the stream (handled in extract_completion).
        // If we reach here without seeing finish_reason, the model has not transmitted
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
            provider_metadata: None,
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }

    fn serialize_event(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        self.serialize_event_impl(event)
    }
}

// Legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.
// The GeminiEventConverter is still used for SSE event conversion in tests.

mod serialize;

impl GeminiEventConverter {
    fn serialize_event_impl(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        serialize::serialize_event(self, event)
    }
}

#[cfg(test)]
mod tests;
