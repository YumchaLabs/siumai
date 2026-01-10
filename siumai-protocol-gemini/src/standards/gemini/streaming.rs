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
use serde::Deserialize;
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

/// Gemini usage metadata
#[derive(Debug, Clone, Deserialize)]
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
        }
    }

    pub fn with_v3_unsupported_part_behavior(
        mut self,
        behavior: V3UnsupportedPartBehavior,
    ) -> Self {
        self.v3_unsupported_part_behavior = behavior;
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

        // Process client-executed tool calls (functionCall parts) as unified ToolCallDelta events.
        for (tool_name, args_json) in self.extract_function_call_events(&response) {
            let id_num = self.next_tool_call_id.fetch_add(1, Ordering::Relaxed);
            builder = builder.add_tool_call_delta(
                format!("call_{id_num}"),
                Some(tool_name),
                Some(args_json),
                None,
            );
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
    ) -> Vec<(String, String)> {
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
                out.push((call.name.clone(), args_json));
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
                "SAFETY" => FinishReason::ContentFilter,
                "RECITATION" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
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
        fn sse_data_frame(value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
            let data = serde_json::to_vec(value).map_err(|e| {
                LlmError::JsonError(format!("Failed to serialize Gemini SSE JSON: {e}"))
            })?;
            let mut out = Vec::with_capacity(data.len() + 8);
            out.extend_from_slice(b"data: ");
            out.extend_from_slice(&data);
            out.extend_from_slice(b"\n\n");
            Ok(out)
        }

        fn map_finish_reason(reason: &FinishReason) -> &'static str {
            match reason {
                FinishReason::Stop | FinishReason::StopSequence => "STOP",
                FinishReason::Length => "MAX_TOKENS",
                FinishReason::ContentFilter => "SAFETY",
                FinishReason::ToolCalls => "STOP",
                FinishReason::Error => "STOP",
                FinishReason::Unknown => "STOP",
                FinishReason::Other(_) => "STOP",
            }
        }

        match event {
            // Gemini streaming does not have an explicit "start" frame; the first chunk carries data.
            ChatStreamEvent::StreamStart { .. } => Ok(Vec::new()),
            ChatStreamEvent::ContentDelta { delta, .. } => {
                let payload = serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    { "text": delta }
                                ]
                            }
                        }
                    ]
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                // Gemini thinking chunks use `thought: true` on the part.
                let payload = serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    { "text": delta, "thought": true }
                                ]
                            }
                        }
                    ]
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                let thoughts = usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens)
                    .or({
                        #[allow(deprecated)]
                        {
                            usage.reasoning_tokens
                        }
                    });

                let payload = serde_json::json!({
                    "usageMetadata": {
                        "promptTokenCount": usage.prompt_tokens,
                        "candidatesTokenCount": usage.completion_tokens,
                        "totalTokenCount": usage.total_tokens,
                        "thoughtsTokenCount": thoughts,
                    }
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                let mut state = self.serialize_state.lock().map_err(|_| {
                    LlmError::InternalError("serialize_state lock poisoned".to_string())
                })?;

                let call = state.function_calls_by_id.entry(id.clone()).or_default();

                if let Some(name) = function_name.clone()
                    && !name.trim().is_empty()
                {
                    call.name = Some(name);
                }

                if let Some(delta) = arguments_delta.clone() {
                    call.arguments.push_str(&delta);
                }

                let Some(name) = call.name.clone() else {
                    return Ok(Vec::new());
                };

                if call.arguments.trim().is_empty() {
                    return Ok(Vec::new());
                }

                let parsed: serde_json::Value =
                    crate::streaming::parse_json_with_repair(&call.arguments).map_err(|e| {
                        LlmError::ParseError(format!(
                            "Failed to parse Gemini tool call arguments as JSON object: {e}"
                        ))
                    })?;

                let Some(obj) = parsed.as_object() else {
                    return Ok(Vec::new());
                };

                let args_json = serde_json::to_string(obj).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to serialize Gemini tool call args JSON object: {e}"
                    ))
                })?;

                if call
                    .last_emitted_args_json
                    .as_ref()
                    .is_some_and(|v| v == &args_json)
                {
                    return Ok(Vec::new());
                }
                call.last_emitted_args_json = Some(args_json);

                let payload = serde_json::json!({
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": name,
                                            "args": serde_json::Value::Object(obj.clone())
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::StreamEnd { response } => {
                let reason = response
                    .finish_reason
                    .as_ref()
                    .map(map_finish_reason)
                    .unwrap_or("STOP");

                let thoughts = response
                    .usage
                    .as_ref()
                    .and_then(|u| u.completion_tokens_details.as_ref())
                    .and_then(|d| d.reasoning_tokens)
                    .or_else(|| {
                        #[allow(deprecated)]
                        {
                            response.usage.as_ref().and_then(|u| u.reasoning_tokens)
                        }
                    });

                let usage = response.usage.as_ref().map(|u| {
                    serde_json::json!({
                        "promptTokenCount": u.prompt_tokens,
                        "candidatesTokenCount": u.completion_tokens,
                        "totalTokenCount": u.total_tokens,
                        "thoughtsTokenCount": thoughts,
                    })
                });

                let mut out = Vec::new();

                // Flush pending tool calls (best-effort) before finish chunk.
                if let Ok(mut state) = self.serialize_state.lock() {
                    for (call_id, call) in state.function_calls_by_id.iter_mut() {
                        let Some(name) = call.name.clone() else {
                            continue;
                        };
                        if call.arguments.trim().is_empty() {
                            continue;
                        }

                        let parsed: Result<serde_json::Value, _> =
                            crate::streaming::parse_json_with_repair(&call.arguments);
                        let Ok(parsed) = parsed else {
                            continue;
                        };
                        let Some(obj) = parsed.as_object() else {
                            continue;
                        };

                        let args_json = match serde_json::to_string(obj) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if call
                            .last_emitted_args_json
                            .as_ref()
                            .is_some_and(|v| v == &args_json)
                        {
                            continue;
                        }
                        call.last_emitted_args_json = Some(args_json);

                        let payload = serde_json::json!({
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "functionCall": {
                                                    "name": name,
                                                    "args": serde_json::Value::Object(obj.clone())
                                                }
                                            }
                                        ]
                                    }
                                }
                            ],
                            "siumai": { "toolCallId": call_id }
                        });
                        if let Ok(frame) = sse_data_frame(&payload) {
                            out.extend_from_slice(&frame);
                        }
                    }
                }

                let payload = serde_json::json!({
                    "candidates": [
                        { "finishReason": reason }
                    ],
                    "usageMetadata": usage.unwrap_or(serde_json::Value::Null),
                });
                out.extend_from_slice(&sse_data_frame(&payload)?);
                Ok(out)
            }
            ChatStreamEvent::Error { error } => {
                // Gemini SSE errors do not have a stable in-band frame; emit a best-effort JSON payload.
                let payload = serde_json::json!({
                    "error": { "message": error }
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::Custom { data, .. } => {
                let Some(part) = LanguageModelV3StreamPart::parse_loose_json(data) else {
                    return Ok(Vec::new());
                };

                match part {
                    LanguageModelV3StreamPart::Source(source) => {
                        let chunk = match source {
                            LanguageModelV3Source::Url { url, title, .. } => {
                                let mut web = serde_json::Map::new();
                                web.insert("uri".to_string(), serde_json::Value::String(url));
                                if let Some(title) = title {
                                    web.insert(
                                        "title".to_string(),
                                        serde_json::Value::String(title),
                                    );
                                }
                                serde_json::json!({ "web": serde_json::Value::Object(web) })
                            }
                            LanguageModelV3Source::Document {
                                title,
                                filename,
                                media_type: _,
                                ..
                            } => {
                                let mut retrieved = serde_json::Map::new();
                                retrieved.insert(
                                    "title".to_string(),
                                    serde_json::Value::String(match filename {
                                        Some(f) => format!("{title} ({f})"),
                                        None => title,
                                    }),
                                );
                                serde_json::json!({
                                    "retrievedContext": serde_json::Value::Object(retrieved)
                                })
                            }
                        };

                        let payload = serde_json::json!({
                            "candidates": [
                                {
                                    "groundingMetadata": {
                                        "groundingChunks": [chunk]
                                    }
                                }
                            ]
                        });
                        sse_data_frame(&payload)
                    }
                    LanguageModelV3StreamPart::Finish {
                        usage,
                        finish_reason,
                        ..
                    } => {
                        let unified = finish_reason.unified.to_ascii_lowercase();
                        let reason = if unified.contains("length") || unified.contains("max") {
                            "MAX_TOKENS"
                        } else if unified.contains("safety") || unified.contains("content") {
                            "SAFETY"
                        } else {
                            "STOP"
                        };

                        let prompt =
                            usage.input_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                        let completion =
                            usage.output_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                        let total = prompt.saturating_add(completion);

                        let payload = serde_json::json!({
                            "candidates": [
                                { "finishReason": reason }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": prompt,
                                "candidatesTokenCount": completion,
                                "totalTokenCount": total,
                                "thoughtsTokenCount": usage.output_tokens.reasoning.map(|v| v.min(u32::MAX as u64) as u32),
                            }
                        });
                        sse_data_frame(&payload)
                    }
                    LanguageModelV3StreamPart::ToolResult(tr) => {
                        if tr.tool_name == "code_execution" {
                            let outcome = tr
                                .result
                                .get("outcome")
                                .and_then(|v| v.as_str())
                                .unwrap_or("OUTCOME_OK");
                            let output = tr.result.get("output").and_then(|v| v.as_str());

                            let mut res = serde_json::Map::new();
                            res.insert(
                                "outcome".to_string(),
                                serde_json::Value::String(outcome.to_string()),
                            );
                            if let Some(out) = output {
                                res.insert(
                                    "output".to_string(),
                                    serde_json::Value::String(out.to_string()),
                                );
                            }

                            let payload = serde_json::json!({
                                "candidates": [
                                    {
                                        "content": {
                                            "parts": [
                                                { "codeExecutionResult": serde_json::Value::Object(res) }
                                            ]
                                        }
                                    }
                                ]
                            });
                            return sse_data_frame(&payload);
                        }

                        if self.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                            && let Some(text) =
                                LanguageModelV3StreamPart::ToolResult(tr).to_lossy_text()
                        {
                            return self.serialize_event(&ChatStreamEvent::ContentDelta {
                                delta: text,
                                index: None,
                            });
                        }

                        Ok(Vec::new())
                    }
                    other => {
                        let mut out = Vec::new();
                        for ev in other.to_best_effort_chat_events() {
                            out.extend_from_slice(&self.serialize_event(&ev)?);
                        }

                        if out.is_empty()
                            && self.v3_unsupported_part_behavior
                                == V3UnsupportedPartBehavior::AsText
                            && let Some(text) = other.to_lossy_text()
                        {
                            out.extend_from_slice(&self.serialize_event(
                                &ChatStreamEvent::ContentDelta {
                                    delta: text,
                                    index: None,
                                },
                            )?);
                        }

                        Ok(out)
                    }
                }
            }
        }
    }
}

// Legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.
// The GeminiEventConverter is still used for SSE event conversion in tests.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standards::gemini::types::GeminiConfig;
    use crate::streaming::SseEventConverter;
    use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};

    fn create_test_config() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("test-key".to_string()),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_gemini_streaming_conversion() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test content delta conversion
        let json_data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(!result.is_empty());

        // In the new architecture, we might get StreamStart + ContentDelta
        let content_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_event {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event in results: {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_gemini_streaming_emits_source_events_and_dedups() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let json_data = r#"{"candidates":[{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://www.rust-lang.org/","title":"Rust"}}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let r1 = converter.convert_event(event).await;
        assert!(
            r1.iter().any(|e| matches!(e, Ok(ChatStreamEvent::Custom { event_type, .. }) if event_type == "gemini:source")),
            "expected gemini:source in first chunk: {r1:?}"
        );

        // Same payload again should not emit a duplicate source event.
        let event2 = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };
        let r2 = converter.convert_event(event2).await;
        assert!(
            !r2.iter().any(|e| matches!(e, Ok(ChatStreamEvent::Custom { event_type, .. }) if event_type == "gemini:source")),
            "expected no duplicate gemini:source in second chunk: {r2:?}"
        );
    }

    #[tokio::test]
    async fn test_gemini_finish_reason() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test finish reason conversion
        let json_data = r#"{"candidates":[{"finishReason":"STOP"}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(!result.is_empty());

        // In the new architecture, first event might be StreamStart, look for StreamEnd
        let stream_end_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

        if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        } else {
            panic!("Expected StreamEnd event in results: {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_empty_event_is_ignored() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let event = eventsource_stream::Event {
            event: "".into(),
            data: "".into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert!(result.is_empty(), "Empty SSE event should be ignored");
    }

    // In strict mode (no json-repair), invalid JSON should produce a parse error
    #[cfg(not(feature = "json-repair"))]
    #[tokio::test]
    async fn test_invalid_json_emits_error() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let event = eventsource_stream::Event {
            event: "".into(),
            data: "{ not json".into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], Err(LlmError::ParseError(_))));
    }

    // In tolerant mode (json-repair enabled), invalid JSON should not error with ParseError
    #[cfg(feature = "json-repair")]
    #[tokio::test]
    async fn test_invalid_json_is_tolerated_with_repair() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let event = eventsource_stream::Event {
            event: "".into(),
            data: "{ not json".into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert_eq!(result.len(), 1);
        assert!(!matches!(result[0], Err(LlmError::ParseError(_))));
    }

    #[tokio::test]
    async fn test_stream_start_emitted_once_across_events() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let mk_event = |text: &str| eventsource_stream::Event {
            event: "".into(),
            data: format!(
                "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{}\"}}]}}}}]}}",
                text
            ),
            id: "".into(),
            retry: None,
        };

        let r1 = converter.convert_event(mk_event("first")).await;
        let r2 = converter.convert_event(mk_event("second")).await;

        // First batch should contain a StreamStart
        assert!(
            r1.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
        );
        // Second batch should not contain StreamStart
        assert!(
            !r2.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
        );
    }

    #[tokio::test]
    async fn test_multi_parts_emit_multiple_deltas_in_order() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let json = r#"{"candidates":[{"content":{"parts":[{"text":"A"},{"text":"B"}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".into(),
            data: json.into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        let deltas: Vec<_> = result
            .into_iter()
            .filter_map(|e| match e {
                Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(delta),
                _ => None,
            })
            .collect();
        assert!(deltas.contains(&"A".to_string()));
        assert!(deltas.contains(&"B".to_string()));
        // Order is preserved within a single event
        let a_pos = deltas.iter().position(|d| d == "A").unwrap();
        let b_pos = deltas.iter().position(|d| d == "B").unwrap();
        assert!(a_pos < b_pos);
    }

    #[tokio::test]
    async fn test_thinking_delta_extraction() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let json =
            r#"{"candidates":[{"content":{"parts":[{"text":"thinking..","thought":true}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".into(),
            data: json.into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert!(
            result
                .iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::ThinkingDelta { .. })))
        );
    }

    fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        let text = String::from_utf8_lossy(bytes);
        text.split("\n\n")
            .filter_map(|chunk| {
                let line = chunk
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .map(str::trim)?;
                if line.is_empty() || line == "[DONE]" {
                    return None;
                }
                serde_json::from_str::<serde_json::Value>(line).ok()
            })
            .collect()
    }

    #[tokio::test]
    async fn gemini_stream_proxy_serializes_content_delta() {
        let converter = GeminiEventConverter::new(create_test_config());

        let bytes = converter
            .serialize_event(&ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            })
            .expect("serialize ok");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["text"],
            serde_json::json!("Hello")
        );

        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: serde_json::to_string(&frames[0]).expect("json"),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(event).await;
        assert!(out.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "Hello"
        )));
    }

    #[tokio::test]
    async fn gemini_stream_proxy_serializes_thinking_delta() {
        let converter = GeminiEventConverter::new(create_test_config());

        let bytes = converter
            .serialize_event(&ChatStreamEvent::ThinkingDelta {
                delta: "think".to_string(),
            })
            .expect("serialize ok");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["thought"],
            serde_json::json!(true)
        );

        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: serde_json::to_string(&frames[0]).expect("json"),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(event).await;
        assert!(out.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ThinkingDelta { delta }) if delta == "think"
        )));
    }

    #[tokio::test]
    async fn gemini_stream_proxy_serializes_usage_update() {
        let converter = GeminiEventConverter::new(create_test_config());

        let bytes = converter
            .serialize_event(&ChatStreamEvent::UsageUpdate {
                usage: Usage::builder()
                    .prompt_tokens(3)
                    .completion_tokens(5)
                    .total_tokens(8)
                    .build(),
            })
            .expect("serialize ok");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["usageMetadata"]["promptTokenCount"],
            serde_json::json!(3)
        );

        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: serde_json::to_string(&frames[0]).expect("json"),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(event).await;
        assert!(out.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::UsageUpdate { usage }) if usage.prompt_tokens == 3
        )));
    }

    #[tokio::test]
    async fn gemini_stream_proxy_serializes_stream_end_finish_reason() {
        let converter = GeminiEventConverter::new(create_test_config());

        let bytes = converter
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: None,
                    content: MessageContent::Text(String::new()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(3)
                            .completion_tokens(5)
                            .total_tokens(8)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize ok");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0]["candidates"][0]["finishReason"],
            serde_json::json!("STOP")
        );

        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: serde_json::to_string(&frames[0]).expect("json"),
            id: "".to_string(),
            retry: None,
        };
        let out = converter.convert_event(event).await;
        assert!(
            out.iter()
                .any(|e| matches!(
                    e,
                    Ok(ChatStreamEvent::StreamEnd { response }) if response.finish_reason == Some(FinishReason::Stop)
                ))
        );
    }

    #[test]
    fn gemini_serializes_tool_call_delta_as_function_call_part() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: Some("get_weather".to_string()),
                arguments_delta: Some(r#"{"city":"Tokyo"}"#.to_string()),
                index: None,
            })
            .expect("serialize tool call delta");

        let text = String::from_utf8(bytes).expect("utf8");
        let json_line = text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("data line");

        let v: serde_json::Value = serde_json::from_str(json_line).expect("json");
        assert_eq!(
            v["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            v["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
            serde_json::json!("Tokyo")
        );
    }

    #[test]
    fn gemini_serializes_v3_custom_parts_best_effort() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:text-delta".to_string(),
                data: serde_json::json!({
                    "type": "text-delta",
                    "id": "0",
                    "delta": "Hello",
                }),
            })
            .expect("serialize custom text-delta");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["text"],
            serde_json::json!("Hello")
        );

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:tool-call".to_string(),
                data: serde_json::json!({
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": r#"{"city":"Tokyo"}"#,
                }),
            })
            .expect("serialize custom tool-call");
        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
            serde_json::json!("Tokyo")
        );
    }

    #[test]
    fn gemini_serializes_v3_source_part_as_grounding_chunk() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "anthropic:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "url",
                    "id": "src_1",
                    "url": "https://www.rust-lang.org/",
                    "title": "Rust",
                }),
            })
            .expect("serialize v3 source");

        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(
            frames[0]["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["web"]["uri"],
            serde_json::json!("https://www.rust-lang.org/")
        );
    }

    #[test]
    fn gemini_serializes_v3_finish_part_as_finish_reason_chunk() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "usage": {
                        "inputTokens": { "total": 3 },
                        "outputTokens": { "total": 5 }
                    },
                    "finishReason": { "unified": "stop" }
                }),
            })
            .expect("serialize v3 finish");

        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(
            frames[0]["candidates"][0]["finishReason"],
            serde_json::json!("STOP")
        );
        assert_eq!(
            frames[0]["usageMetadata"]["totalTokenCount"],
            serde_json::json!(8)
        );
    }

    #[test]
    fn gemini_serializes_v3_code_execution_tool_result_as_code_execution_result_part() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        let bytes = converter
            .serialize_event(&ChatStreamEvent::Custom {
                event_type: "openai:tool-result".to_string(),
                data: serde_json::json!({
                    "type": "tool-result",
                    "toolCallId": "call_1",
                    "toolName": "code_execution",
                    "result": { "outcome": "OUTCOME_OK", "output": "1" }
                }),
            })
            .expect("serialize v3 tool-result");

        let frames = parse_sse_json_frames(&bytes);
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["outcome"],
            serde_json::json!("OUTCOME_OK")
        );
        assert_eq!(
            frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["output"],
            serde_json::json!("1")
        );
    }
}
