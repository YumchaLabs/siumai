//! Stream Processor
//!
//! Processes and transforms streaming events, accumulating content, tool calls,
//! and thinking buffers with configurable limits and overflow handling.

use crate::error::LlmError;
use crate::types::{
    ChatResponse, ChatStreamEvent, ChatStreamFileData, ChatStreamPart, ContentPart, FinishReason,
    MessageContent, ProviderMetadataMap, ResponseMetadata, Usage, Warning, merge_provider_metadata,
    provider_metadata_from_object,
};
use std::collections::HashMap;

/// Overflow handler callback type
///
/// Called when a buffer exceeds its configured limit.
/// Parameters: (buffer_name, attempted_size)
pub type OverflowHandler = Box<dyn FnMut(&str, usize) + Send + Sync>;

/// Stream Processor Configuration
///
/// Controls buffer limits and overflow behavior for stream processing.
pub struct StreamProcessorConfig {
    /// Maximum size for content buffer (in bytes)
    pub max_content_buffer_size: Option<usize>,
    /// Maximum size for thinking buffer (in bytes)  
    pub max_thinking_buffer_size: Option<usize>,
    /// Maximum number of tool calls to track
    pub max_tool_calls: Option<usize>,
    /// Maximum accumulated size for a single tool call's arguments (in bytes)
    pub max_tool_arguments_size: Option<usize>,
    /// Handler for buffer overflow
    pub overflow_handler: Option<OverflowHandler>,
}

impl std::fmt::Debug for StreamProcessorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessorConfig")
            .field("max_content_buffer_size", &self.max_content_buffer_size)
            .field("max_thinking_buffer_size", &self.max_thinking_buffer_size)
            .field("max_tool_calls", &self.max_tool_calls)
            .field("max_tool_arguments_size", &self.max_tool_arguments_size)
            .field(
                "has_overflow_handler",
                &self
                    .overflow_handler
                    .as_ref()
                    .map(|_| true)
                    .unwrap_or(false),
            )
            .finish()
    }
}

impl Default for StreamProcessorConfig {
    fn default() -> Self {
        Self {
            max_content_buffer_size: Some(10 * 1024 * 1024), // 10MB default
            max_thinking_buffer_size: Some(5 * 1024 * 1024), // 5MB default
            max_tool_calls: Some(100),                       // 100 tool calls max
            max_tool_arguments_size: None,                   // default: no truncation for args
            overflow_handler: None,
        }
    }
}

/// Stream Processor
///
/// Processes streaming events and accumulates content, tool calls, and thinking buffers.
/// Provides buffer overflow protection and incremental state tracking.
pub struct StreamProcessor {
    buffer: String,
    tool_calls: HashMap<String, ToolCallBuilder>, // Use ID as key to handle duplicate indices
    tool_call_order: Vec<String>,                 // Track order of tool calls for consistent output
    tool_call_sources: HashMap<String, ToolCallStreamSource>,
    thinking_buffer: String,
    stream_parts: Vec<ContentPart>,
    stream_warnings: Vec<Warning>,
    start_metadata: Option<ResponseMetadata>,
    terminal_response: Option<ChatResponse>,
    current_usage: Option<Usage>,
    stream_finish_reason: Option<FinishReason>,
    stream_raw_finish_reason: Option<String>,
    final_provider_metadata: Option<ProviderMetadataMap>,
    config: StreamProcessorConfig,
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProcessor {
    /// Create a new stream processor with default configuration
    pub fn new() -> Self {
        Self::with_config(StreamProcessorConfig::default())
    }

    /// Create a new stream processor with custom configuration
    pub fn with_config(config: StreamProcessorConfig) -> Self {
        Self {
            buffer: String::new(),
            tool_calls: HashMap::new(),
            tool_call_order: Vec::new(),
            tool_call_sources: HashMap::new(),
            thinking_buffer: String::new(),
            stream_parts: Vec::new(),
            stream_warnings: Vec::new(),
            start_metadata: None,
            terminal_response: None,
            current_usage: None,
            stream_finish_reason: None,
            stream_raw_finish_reason: None,
            final_provider_metadata: None,
            config,
        }
    }

    /// Process a stream event and return the processed result
    pub fn process_event(&mut self, event: ChatStreamEvent) -> ProcessedEvent {
        match event {
            ChatStreamEvent::ContentDelta { delta, index } => {
                self.process_content_delta(delta, index)
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => self.process_tool_call_delta(
                id,
                function_name,
                arguments_delta,
                index,
                ToolCallStreamSource::LegacyDelta,
            ),
            ChatStreamEvent::ThinkingDelta { delta } => self.process_thinking_delta(delta),
            ChatStreamEvent::UsageUpdate { usage } => self.process_usage_update(usage),
            ChatStreamEvent::StreamStart { metadata } => {
                self.start_metadata = Some(metadata.clone());
                ProcessedEvent::StreamStart { metadata }
            }
            ChatStreamEvent::StreamEnd { response } => {
                if let Some(usage) = response.usage.clone() {
                    self.current_usage = Some(usage);
                }
                if let Some(provider_metadata) = response.provider_metadata.clone() {
                    if let Some(current) = self.final_provider_metadata.as_mut() {
                        merge_provider_metadata(current, provider_metadata);
                    } else {
                        self.final_provider_metadata = Some(provider_metadata);
                    }
                }
                self.terminal_response = Some(response.clone());
                ProcessedEvent::StreamEnd {
                    response: Box::new(response),
                }
            }
            ChatStreamEvent::Part { part } | ChatStreamEvent::PartWithReplay { part, .. } => {
                self.process_stream_part(part)
            }
            ChatStreamEvent::Error { error } => ProcessedEvent::Error {
                error: LlmError::InternalError(error),
            },
            ChatStreamEvent::Custom { event_type, data } => {
                ProcessedEvent::Custom { event_type, data }
            }
        }
    }

    fn process_stream_part(&mut self, part: ChatStreamPart) -> ProcessedEvent {
        match &part {
            ChatStreamPart::TextDelta { delta, .. } => {
                return self.process_content_delta(delta.clone(), None);
            }
            ChatStreamPart::ReasoningDelta { delta, .. } => {
                return self.process_thinking_delta(delta.clone());
            }
            ChatStreamPart::ToolInputStart { id, tool_name, .. } => {
                let mut event = self.process_tool_call_delta(
                    id.clone(),
                    Some(tool_name.clone()),
                    None,
                    None,
                    ToolCallStreamSource::StablePart,
                );

                if let ChatStreamPart::ToolInputStart {
                    provider_metadata,
                    provider_executed,
                    dynamic,
                    title,
                    ..
                } = &part
                {
                    let tool_id = match &event {
                        ProcessedEvent::ToolCallUpdate { id, .. } => id.as_str(),
                        _ => id.as_str(),
                    };
                    self.merge_tool_call_builder_metadata(
                        tool_id,
                        *provider_executed,
                        *dynamic,
                        title.clone(),
                        provider_metadata.clone(),
                    );
                    self.refresh_tool_call_update(&mut event);
                }

                return event;
            }
            ChatStreamPart::ToolInputDelta { id, delta, .. } => {
                return self.process_tool_call_delta(
                    id.clone(),
                    None,
                    Some(delta.clone()),
                    None,
                    ToolCallStreamSource::StablePart,
                );
            }
            ChatStreamPart::StreamStart { warnings } => {
                self.stream_warnings.extend(warnings.clone());
            }
            ChatStreamPart::ResponseMetadata(metadata) => {
                self.start_metadata = Some(metadata.clone());
            }
            ChatStreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => {
                let _ = self.process_usage_update(usage.clone());
                self.stream_finish_reason = Some(finish_reason.unified.clone());
                self.stream_raw_finish_reason = finish_reason.raw.clone();
                if let Some(provider_metadata) = provider_metadata.clone() {
                    self.merge_shared_provider_metadata(provider_metadata);
                }
            }
            ChatStreamPart::ToolApprovalRequest(request) => {
                self.stream_parts
                    .push(ContentPart::tool_approval_request_with_metadata(
                        request.approval_id.clone(),
                        request.tool_call_id.clone(),
                        request.provider_metadata.clone().unwrap_or_default(),
                    ));
            }
            ChatStreamPart::ToolCall(call) => {
                let builder = self.tool_calls.get(&call.tool_call_id);
                self.stream_parts.push(ContentPart::ToolCall {
                    tool_call_id: call.tool_call_id.clone(),
                    tool_name: call.tool_name.clone(),
                    arguments: serde_json::from_str(&call.input)
                        .unwrap_or_else(|_| serde_json::Value::String(call.input.clone())),
                    provider_executed: call
                        .provider_executed
                        .or_else(|| builder.and_then(|builder| builder.provider_executed)),
                    dynamic: call
                        .dynamic
                        .or_else(|| builder.and_then(|builder| builder.dynamic)),
                    invalid: None,
                    error: None,
                    title: builder.and_then(|builder| builder.title.clone()),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: call
                        .provider_metadata
                        .clone()
                        .or_else(|| builder.and_then(|builder| builder.provider_metadata.clone())),
                });
            }
            ChatStreamPart::ToolResult(result) => {
                let builder = self.tool_calls.get(&result.tool_call_id);
                self.stream_parts.push(ContentPart::ToolResult {
                    tool_call_id: result.tool_call_id.clone(),
                    tool_name: result.tool_name.clone(),
                    output: if result.is_error.unwrap_or(false) {
                        crate::types::ToolResultOutput::error_json(result.result.clone())
                    } else {
                        crate::types::ToolResultOutput::json(result.result.clone())
                    },
                    input: builder.map(tool_input_from_builder),
                    provider_executed: builder.and_then(|builder| builder.provider_executed),
                    dynamic: result
                        .dynamic
                        .or_else(|| builder.and_then(|builder| builder.dynamic)),
                    preliminary: result.preliminary,
                    title: builder.and_then(|builder| builder.title.clone()),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: result.provider_metadata.clone(),
                });
            }
            ChatStreamPart::Custom(custom) => {
                self.stream_parts.push(ContentPart::Custom {
                    kind: custom.kind.clone(),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: custom.provider_metadata.clone(),
                });
            }
            ChatStreamPart::File(file) => {
                self.stream_parts
                    .push(stream_file_part_to_content_part(file, false));
            }
            ChatStreamPart::ReasoningFile(file) => {
                self.stream_parts
                    .push(stream_file_part_to_content_part(file, true));
            }
            ChatStreamPart::Source {
                id,
                source,
                provider_metadata,
            } => {
                self.stream_parts.push(ContentPart::Source {
                    id: id.clone(),
                    source: source.clone(),
                    provider_metadata: provider_metadata.clone(),
                });
            }
            ChatStreamPart::Error { error } => {
                return ProcessedEvent::Error {
                    error: LlmError::InternalError(
                        serde_json::to_string(error).unwrap_or_else(|_| "stream part error".into()),
                    ),
                };
            }
            ChatStreamPart::TextStart { .. }
            | ChatStreamPart::TextEnd { .. }
            | ChatStreamPart::ReasoningStart { .. }
            | ChatStreamPart::ReasoningEnd { .. }
            | ChatStreamPart::ToolInputEnd { .. }
            | ChatStreamPart::Raw { .. } => {}
        }

        ProcessedEvent::Part {
            part: Box::new(part),
        }
    }

    /// Process content delta
    fn process_content_delta(&mut self, delta: String, index: Option<usize>) -> ProcessedEvent {
        // Check buffer size limit before appending
        if let Some(max_size) = self.config.max_content_buffer_size {
            let new_size = self.buffer.len() + delta.len();
            if new_size > max_size {
                // Call overflow handler if provided
                if let Some(handler) = self.config.overflow_handler.as_mut() {
                    (handler)("content_buffer", new_size);
                }
                // Truncate buffer to keep within limits
                let available = max_size.saturating_sub(self.buffer.len());
                let truncated_delta = if available > 0 {
                    delta.chars().take(available).collect()
                } else {
                    String::new()
                };
                self.buffer.push_str(&truncated_delta);
                return ProcessedEvent::ContentUpdate {
                    delta: truncated_delta,
                    accumulated: self.buffer.clone(),
                    index,
                };
            }
        }

        self.buffer.push_str(&delta);
        ProcessedEvent::ContentUpdate {
            delta,
            accumulated: self.buffer.clone(),
            index,
        }
    }

    /// Process tool call delta
    fn process_tool_call_delta(
        &mut self,
        id: String,
        function_name: Option<String>,
        arguments_delta: Option<String>,
        index: Option<usize>,
        source: ToolCallStreamSource,
    ) -> ProcessedEvent {
        tracing::debug!("Tool call delta - ID: '{}', Index: {:?}", id, index);

        // Use tool call ID as the primary key to handle duplicate indices
        let tool_id = if !id.is_empty() {
            id.clone()
        } else {
            // If no ID, use the most recent tool call
            if let Some(last_id) = self.tool_call_order.last() {
                last_id.clone()
            } else {
                // Fallback: create a temporary ID based on order
                format!("temp_tool_call_{}", self.tool_call_order.len())
            }
        };

        match self.tool_call_sources.entry(tool_id.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(source);
            }
            std::collections::hash_map::Entry::Occupied(entry) if *entry.get() != source => {
                return ProcessedEvent::ToolCallUpdate {
                    id: tool_id.clone(),
                    current_state: self.tool_calls.get(&tool_id).cloned().unwrap_or_default(),
                    index,
                };
            }
            std::collections::hash_map::Entry::Occupied(_) => {}
        }

        // Get or create the tool call builder
        let is_new_tool_call = !self.tool_calls.contains_key(&tool_id);

        // Check tool call limit
        if let Some(max_tool_calls) = self.config.max_tool_calls
            && is_new_tool_call
            && self.tool_calls.len() >= max_tool_calls
        {
            // Too many tool calls, skip this one
            if let Some(handler) = self.config.overflow_handler.as_mut() {
                (handler)("tool_calls", self.tool_calls.len() + 1);
            }
            return ProcessedEvent::ToolCallUpdate {
                id: tool_id,
                current_state: ToolCallBuilder::new(),
                index,
            };
        }

        let builder = self.tool_calls.entry(tool_id.clone()).or_insert_with(|| {
            let mut builder = ToolCallBuilder::new();
            if !id.is_empty() {
                builder.id = id.clone();
            } else {
                builder.id = tool_id.clone();
            }
            builder
        });

        // Track order of tool calls for consistent output
        if is_new_tool_call && !id.is_empty() {
            self.tool_call_order.push(tool_id.clone());
        }

        // Accumulate function name
        if let Some(name) = function_name {
            if builder.name.is_empty() {
                builder.name = name;
            } else {
                builder.name.push_str(&name);
            }
        }

        // Accumulate arguments
        if let Some(args) = arguments_delta {
            if let Some(max_args) = self.config.max_tool_arguments_size {
                let new_size = builder.arguments.len() + args.len();
                if new_size > max_args {
                    if let Some(handler) = self.config.overflow_handler.as_mut() {
                        (handler)("tool_arguments", new_size);
                    }
                    let available = max_args.saturating_sub(builder.arguments.len());
                    if available > 0 {
                        let truncated: String = args.chars().take(available).collect();
                        builder.arguments.push_str(&truncated);
                    }
                } else {
                    builder.arguments.push_str(&args);
                }
            } else {
                builder.arguments.push_str(&args);
            }
        }

        ProcessedEvent::ToolCallUpdate {
            id: builder.id.clone(),
            current_state: builder.clone(),
            index,
        }
    }

    /// Process thinking delta
    fn process_thinking_delta(&mut self, delta: String) -> ProcessedEvent {
        // Check thinking buffer size limit
        if let Some(max_size) = self.config.max_thinking_buffer_size {
            let new_size = self.thinking_buffer.len() + delta.len();
            if new_size > max_size {
                // Call overflow handler if provided
                if let Some(handler) = self.config.overflow_handler.as_mut() {
                    (handler)("thinking_buffer", new_size);
                }
                // Truncate buffer to keep within limits
                let available = max_size.saturating_sub(self.thinking_buffer.len());
                let truncated_delta = if available > 0 {
                    delta.chars().take(available).collect()
                } else {
                    String::new()
                };
                self.thinking_buffer.push_str(&truncated_delta);
                return ProcessedEvent::ThinkingUpdate {
                    delta: truncated_delta,
                    accumulated: self.thinking_buffer.clone(),
                };
            }
        }

        self.thinking_buffer.push_str(&delta);
        ProcessedEvent::ThinkingUpdate {
            delta,
            accumulated: self.thinking_buffer.clone(),
        }
    }

    /// Process usage update
    fn process_usage_update(&mut self, usage: Usage) -> ProcessedEvent {
        if let Some(ref mut current) = self.current_usage {
            current.merge(&usage);
        } else {
            self.current_usage = Some(usage.clone());
        }
        ProcessedEvent::UsageUpdate {
            usage: self.current_usage.clone().unwrap(),
        }
    }

    fn merge_tool_call_builder_metadata(
        &mut self,
        tool_id: &str,
        provider_executed: Option<bool>,
        dynamic: Option<bool>,
        title: Option<String>,
        provider_metadata: Option<ProviderMetadataMap>,
    ) {
        let Some(builder) = self.tool_calls.get_mut(tool_id) else {
            return;
        };

        if provider_executed.is_some() {
            builder.provider_executed = provider_executed;
        }
        if dynamic.is_some() {
            builder.dynamic = dynamic;
        }
        if title.is_some() {
            builder.title = title;
        }
        if let Some(provider_metadata) = provider_metadata {
            if let Some(current) = builder.provider_metadata.as_mut() {
                merge_provider_metadata(current, provider_metadata);
            } else {
                builder.provider_metadata = Some(provider_metadata);
            }
        }
    }

    fn refresh_tool_call_update(&self, event: &mut ProcessedEvent) {
        if let ProcessedEvent::ToolCallUpdate {
            id, current_state, ..
        } = event
            && let Some(builder) = self.tool_calls.get(id)
        {
            *current_state = builder.clone();
        }
    }

    fn merge_shared_provider_metadata(&mut self, source: ProviderMetadataMap) {
        if let Some(current) = self.final_provider_metadata.as_mut() {
            merge_provider_metadata(current, source);
        } else {
            self.final_provider_metadata = Some(source);
        }
    }

    /// Build the final response
    pub fn build_final_response(&self) -> ChatResponse {
        self.build_final_response_with_finish_reason(None)
    }

    /// Build the final response with finish reason
    pub fn build_final_response_with_finish_reason(
        &self,
        finish_reason: Option<FinishReason>,
    ) -> ChatResponse {
        let terminal_response = self.terminal_response.as_ref();
        let mut stream_metadata = HashMap::new();

        if !self.thinking_buffer.is_empty() {
            stream_metadata.insert(
                "thinking".to_string(),
                serde_json::Value::String(self.thinking_buffer.clone()),
            );
        }

        let content = self.build_final_content(terminal_response);

        // Convert to nested provider_metadata structure
        let mut provider_metadata = self.final_provider_metadata.clone().unwrap_or_default();
        if !stream_metadata.is_empty() {
            merge_provider_metadata(
                &mut provider_metadata,
                provider_metadata_from_object("stream", stream_metadata),
            );
        }
        let provider_metadata = if provider_metadata.is_empty() {
            None
        } else {
            Some(provider_metadata)
        };

        ChatResponse {
            id: terminal_response
                .and_then(|response| response.id.clone())
                .or_else(|| {
                    self.start_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.id.clone())
                }),
            content,
            model: terminal_response
                .and_then(|response| response.model.clone())
                .or_else(|| {
                    self.start_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.model.clone())
                }),
            usage: self
                .current_usage
                .clone()
                .or_else(|| terminal_response.and_then(|response| response.usage.clone())),
            finish_reason: finish_reason
                .or_else(|| self.stream_finish_reason.clone())
                .or_else(|| terminal_response.and_then(|response| response.finish_reason.clone())),
            raw_finish_reason: terminal_response
                .and_then(|response| response.raw_finish_reason.clone())
                .or_else(|| self.stream_raw_finish_reason.clone()),
            audio: terminal_response.and_then(|response| response.audio.clone()),
            system_fingerprint: terminal_response
                .and_then(|response| response.system_fingerprint.clone()),
            service_tier: terminal_response.and_then(|response| response.service_tier.clone()),
            warnings: terminal_response
                .and_then(|response| response.warnings.clone())
                .or_else(|| {
                    (!self.stream_warnings.is_empty()).then(|| self.stream_warnings.clone())
                }),
            provider_metadata,
        }
    }

    fn build_final_content(&self, terminal_response: Option<&ChatResponse>) -> MessageContent {
        let has_accumulated_content = !self.buffer.is_empty()
            || !self.tool_calls.is_empty()
            || !self.thinking_buffer.is_empty()
            || !self.stream_parts.is_empty();

        if !has_accumulated_content {
            return terminal_response
                .map(|response| response.content.clone())
                .unwrap_or_else(|| MessageContent::Text(String::new()));
        }

        #[cfg(feature = "structured-messages")]
        if matches!(
            terminal_response.map(|response| &response.content),
            Some(MessageContent::Json(_))
        ) {
            return terminal_response
                .map(|response| response.content.clone())
                .unwrap_or_else(|| MessageContent::Text(String::new()));
        }

        let mut parts = if !self.buffer.is_empty() {
            vec![build_text_part(&self.buffer, terminal_response)]
        } else {
            terminal_response
                .map(|response| extract_terminal_text_parts(&response.content))
                .unwrap_or_default()
        };

        if !self.tool_calls.is_empty() {
            parts.extend(self.build_accumulated_tool_call_parts(terminal_response));
        } else if let Some(response) = terminal_response {
            parts.extend(extract_terminal_tool_call_parts(&response.content));
        }

        if !self.thinking_buffer.is_empty() {
            parts.push(build_reasoning_part(
                &self.thinking_buffer,
                terminal_response,
            ));
        } else if let Some(response) = terminal_response {
            parts.extend(extract_terminal_reasoning_parts(&response.content));
        }

        if let Some(response) = terminal_response {
            parts.extend(extract_terminal_extra_parts(&response.content));
        }

        parts.extend(extract_stream_tool_call_parts(
            &self.stream_parts,
            &self.tool_call_order,
        ));
        parts.extend(extract_stream_reasoning_extra_parts(&self.stream_parts));
        parts.extend(extract_stream_extra_parts(&self.stream_parts));

        message_content_from_parts(parts)
    }

    fn build_accumulated_tool_call_parts(
        &self,
        terminal_response: Option<&ChatResponse>,
    ) -> Vec<ContentPart> {
        let mut parts = Vec::new();

        for (tool_index, id) in self.tool_call_order.iter().enumerate() {
            if self
                .stream_parts
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall { tool_call_id, .. } if tool_call_id == id))
            {
                continue;
            }

            if let Some(builder) = self.tool_calls.get(id)
                && !builder.name.is_empty()
            {
                let arguments = serde_json::from_str(&builder.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(builder.arguments.clone()));

                let terminal_match = terminal_response.and_then(|response| {
                    find_terminal_tool_call_part(&response.content, builder, tool_index)
                });

                parts.push(build_tool_call_part(builder, arguments, terminal_match));
            }
        }

        parts
    }
}

fn stream_file_part_to_content_part(
    file: &crate::types::ChatStreamFilePart,
    reasoning: bool,
) -> ContentPart {
    let source = match &file.data {
        ChatStreamFileData::Base64(data) => crate::types::MediaSource::base64(data.clone()),
        ChatStreamFileData::Bytes(data) => crate::types::MediaSource::binary(data.clone()),
    };

    if reasoning {
        ContentPart::ReasoningFile {
            source,
            media_type: file.media_type.clone(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: file.provider_metadata.clone(),
        }
    } else {
        ContentPart::File {
            source: crate::types::FilePartSource::from(source),
            media_type: file.media_type.clone(),
            filename: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: file.provider_metadata.clone(),
        }
    }
}

fn build_text_part(text: &str, terminal_response: Option<&ChatResponse>) -> ContentPart {
    let provider_metadata =
        terminal_response.and_then(|response| first_terminal_text_metadata(&response.content));

    ContentPart::Text {
        text: text.to_string(),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata,
    }
}

fn build_reasoning_part(text: &str, terminal_response: Option<&ChatResponse>) -> ContentPart {
    let provider_metadata =
        terminal_response.and_then(|response| first_terminal_reasoning_metadata(&response.content));

    ContentPart::Reasoning {
        text: text.to_string(),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata,
    }
}

fn build_tool_call_part(
    builder: &ToolCallBuilder,
    arguments: serde_json::Value,
    terminal_part: Option<&ContentPart>,
) -> ContentPart {
    let (provider_executed, dynamic, title, provider_metadata) = match terminal_part {
        Some(ContentPart::ToolCall {
            provider_executed,
            dynamic,
            title,
            provider_metadata,
            ..
        }) => (
            *provider_executed,
            *dynamic,
            title.clone(),
            provider_metadata.clone(),
        ),
        _ => (None, None, None, None),
    };

    ContentPart::ToolCall {
        tool_call_id: builder.id.clone(),
        tool_name: builder.name.clone(),
        arguments,
        provider_executed: builder.provider_executed.or(provider_executed),
        dynamic: builder.dynamic.or(dynamic),
        invalid: None,
        error: None,
        title: builder.title.clone().or(title),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata: builder.provider_metadata.clone().or(provider_metadata),
    }
}

fn tool_input_from_builder(builder: &ToolCallBuilder) -> serde_json::Value {
    serde_json::from_str(&builder.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(builder.arguments.clone()))
}

fn find_terminal_tool_call_part<'a>(
    content: &'a MessageContent,
    builder: &ToolCallBuilder,
    tool_index: usize,
) -> Option<&'a ContentPart> {
    let terminal_parts = match content {
        MessageContent::MultiModal(parts) => parts,
        _ => return None,
    };

    terminal_parts
        .iter()
        .find(|part| matches!(part, ContentPart::ToolCall { tool_call_id, .. } if tool_call_id == &builder.id))
        .or_else(|| {
            terminal_parts
                .iter()
                .filter(|part| part.is_tool_call())
                .nth(tool_index)
        })
}

fn first_terminal_text_metadata(content: &MessageContent) -> Option<ProviderMetadataMap> {
    match content {
        MessageContent::MultiModal(parts) => parts.iter().find_map(|part| match part {
            ContentPart::Text {
                provider_metadata, ..
            } => provider_metadata.clone(),
            _ => None,
        }),
        _ => None,
    }
}

fn first_terminal_reasoning_metadata(content: &MessageContent) -> Option<ProviderMetadataMap> {
    match content {
        MessageContent::MultiModal(parts) => parts.iter().find_map(|part| match part {
            ContentPart::Reasoning {
                provider_metadata, ..
            } => provider_metadata.clone(),
            _ => None,
        }),
        _ => None,
    }
}

fn extract_terminal_text_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::Text(text) if !text.is_empty() => vec![ContentPart::text(text.clone())],
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_text())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_tool_call_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_tool_call())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_reasoning_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_reasoning())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_extra_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| !part.is_text() && !part.is_tool_call() && !part.is_reasoning())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_stream_tool_call_parts(
    parts: &[ContentPart],
    _accumulated_tool_call_ids: &[String],
) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| matches!(part, ContentPart::ToolCall { .. }))
        .cloned()
        .collect()
}

fn extract_stream_reasoning_extra_parts(parts: &[ContentPart]) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| matches!(part, ContentPart::ReasoningFile { .. }))
        .cloned()
        .collect()
}

fn extract_stream_extra_parts(parts: &[ContentPart]) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| !part.is_text() && !part.is_reasoning() && !part.is_tool_call())
        .cloned()
        .collect()
}

fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    match parts.as_slice() {
        [
            ContentPart::Text {
                text,
                provider_options,
                provider_metadata: None,
            },
        ] if provider_options.is_empty() => MessageContent::Text(text.clone()),
        [] => MessageContent::Text(String::new()),
        _ => MessageContent::MultiModal(parts),
    }
}

/// Processed Event
///
/// Result of processing a stream event, containing accumulated state.
#[derive(Debug, Clone)]
pub enum ProcessedEvent {
    /// Content update with delta and accumulated content
    ContentUpdate {
        delta: String,
        accumulated: String,
        index: Option<usize>,
    },
    /// Tool call update with current state
    ToolCallUpdate {
        id: String,
        current_state: ToolCallBuilder,
        index: Option<usize>,
    },
    /// Thinking update with delta and accumulated thinking
    ThinkingUpdate { delta: String, accumulated: String },
    /// Usage update
    UsageUpdate { usage: Usage },
    /// Stream start event
    StreamStart { metadata: ResponseMetadata },
    /// Stream end event
    StreamEnd { response: Box<ChatResponse> },
    /// Typed stream part passed through after state updates.
    Part { part: Box<ChatStreamPart> },
    /// Error event
    Error { error: LlmError },
    /// Custom provider-specific event (passed through without processing)
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
}

/// Tool Call Builder
///
/// Accumulates tool call information incrementally during streaming.
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    /// Tool call ID
    pub id: String,
    /// Tool type (deprecated, kept for compatibility)
    #[allow(dead_code)]
    pub r#type: Option<String>,
    /// Function name
    pub name: String,
    /// Function arguments (JSON string)
    pub arguments: String,
    /// Whether the provider executes the tool directly.
    pub provider_executed: Option<bool>,
    /// Whether the tool call is dynamic/runtime-defined.
    pub dynamic: Option<bool>,
    /// Optional human-readable tool title.
    pub title: Option<String>,
    /// Provider metadata carried by stable tool-input parts.
    pub provider_metadata: Option<ProviderMetadataMap>,
}

impl Default for ToolCallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallBuilder {
    /// Create a new empty tool call builder
    pub const fn new() -> Self {
        Self {
            id: String::new(),
            r#type: None,
            name: String::new(),
            arguments: String::new(),
            provider_executed: None,
            dynamic: None,
            title: None,
            provider_metadata: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallStreamSource {
    LegacyDelta,
    StablePart,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioOutput, PromptTokensDetails, Warning};

    #[test]
    fn tool_arguments_respect_max_size() {
        let mut cfg = StreamProcessorConfig {
            max_tool_arguments_size: Some(8),
            ..Default::default()
        };
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_for_cb = called.clone();
        cfg.overflow_handler = Some(Box::new(move |name, size| {
            assert_eq!(name, "tool_arguments");
            assert!(size > 8);
            called_for_cb.store(true, std::sync::atomic::Ordering::Relaxed);
        }));
        let mut sp = StreamProcessor::with_config(cfg);
        let ev = ChatStreamEvent::ToolCallDelta {
            id: "id1".into(),
            function_name: Some("fn".into()),
            arguments_delta: Some("abcdefghijk".into()),
            index: Some(0),
        };
        let _ = sp.process_event(ev);
        // Ensure builder exists and arguments have been truncated
        let b = sp.tool_calls.get("id1").unwrap();
        assert!(b.arguments.len() <= 8);
        assert!(called.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn mixed_legacy_and_stable_tool_streams_deduplicate_by_first_source() {
        let mut sp = StreamProcessor::new();

        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: "call_1".to_string(),
                tool_name: "search".to_string(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            },
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputDelta {
                id: "call_1".to_string(),
                delta: "{\"query\":\"rust\"}".to_string(),
                provider_metadata: None,
            },
        });

        // The equivalent legacy delta should be ignored once the stable lane
        // has already claimed this tool-call id.
        let _ = sp.process_event(ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("search".to_string()),
            arguments_delta: Some("{\"query\":\"rust\"}".to_string()),
            index: Some(0),
        });

        let builder = sp.tool_calls.get("call_1").expect("tool call builder");
        assert_eq!(builder.name, "search");
        assert_eq!(builder.arguments, "{\"query\":\"rust\"}");
    }

    #[test]
    fn stable_tool_call_parts_survive_final_response_build() {
        let mut sp = StreamProcessor::new();

        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: "call_1".to_string(),
                tool_name: "search".to_string(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            },
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputDelta {
                id: "call_1".to_string(),
                delta: "{\"query\":\"rust\"}".to_string(),
                provider_metadata: None,
            },
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                input: "{\"query\":\"rust\"}".to_string(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: None,
            }),
        });

        let final_resp = sp.build_final_response_with_finish_reason(Some(FinishReason::ToolCalls));
        let tool_calls = final_resp.tool_calls();

        assert_eq!(tool_calls.len(), 1);
        let tool_call = tool_calls[0].as_tool_call().expect("tool call");
        assert_eq!(tool_call.tool_call_id, "call_1");
        assert_eq!(tool_call.tool_name, "search");
        assert_eq!(*tool_call.arguments, serde_json::json!({ "query": "rust" }));
    }

    #[test]
    fn stream_end_response_updates_final_usage_and_provider_metadata() {
        let mut sp = StreamProcessor::new();
        let response = ChatResponse {
            id: None,
            content: MessageContent::Text("hello".to_string()),
            model: None,
            usage: Some(Usage::new(3, 5)),
            finish_reason: Some(FinishReason::Stop),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: Some(HashMap::from([(
                "perplexity".to_string(),
                serde_json::json!({
                    "usage": { "citation_tokens": 1 }
                }),
            )])),
        };

        let _ = sp.process_event(ChatStreamEvent::StreamEnd { response });
        let final_resp = sp.build_final_response_with_finish_reason(Some(FinishReason::Stop));

        assert_eq!(
            final_resp
                .usage
                .as_ref()
                .and_then(|usage| usage.total_tokens()),
            Some(8)
        );
        let metadata = final_resp.provider_metadata.expect("provider metadata");
        assert_eq!(
            metadata["perplexity"]["usage"]["citation_tokens"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn aggregates_usage_updates() {
        let mut sp = StreamProcessor::new();
        let u1 = Usage::new(3, 5);
        let mut u2 = Usage::new(2, 4);
        u2.prompt_tokens_details = Some(PromptTokensDetails::with_cached(7));
        let _ = sp.process_event(ChatStreamEvent::UsageUpdate { usage: u1 });
        let _ = sp.process_event(ChatStreamEvent::UsageUpdate { usage: u2 });
        let final_resp = sp.build_final_response_with_finish_reason(Some(FinishReason::Stop));
        let usage = final_resp.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens(), Some(5));
        assert_eq!(usage.completion_tokens(), Some(9));
        assert_eq!(usage.total_tokens(), Some(14));
        assert_eq!(
            usage.prompt_tokens_details.unwrap().cached_tokens.unwrap(),
            7
        );
    }

    #[test]
    fn stream_end_response_preserves_terminal_envelope_fields() {
        let mut sp = StreamProcessor::new();
        let response = ChatResponse {
            id: Some("resp_123".to_string()),
            content: MessageContent::Text("terminal text".to_string()),
            model: Some("claude-3-7-sonnet".to_string()),
            usage: Some(Usage::new(11, 7)),
            finish_reason: Some(FinishReason::Stop),
            raw_finish_reason: Some("end_turn".to_string()),
            audio: Some(AudioOutput {
                id: "aud_123".to_string(),
                expires_at: 1_744_000_000,
                data: "ZmFrZQ==".to_string(),
                transcript: "hello".to_string(),
            }),
            system_fingerprint: Some("fp_terminal".to_string()),
            service_tier: Some("priority".to_string()),
            warnings: Some(vec![Warning::other("watch settings")]),
            provider_metadata: None,
        };

        let _ = sp.process_event(ChatStreamEvent::StreamEnd { response });
        let final_resp = sp.build_final_response();

        assert_eq!(final_resp.id.as_deref(), Some("resp_123"));
        assert_eq!(final_resp.model.as_deref(), Some("claude-3-7-sonnet"));
        assert_eq!(
            final_resp
                .usage
                .as_ref()
                .and_then(|usage| usage.total_tokens()),
            Some(18)
        );
        assert_eq!(final_resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(final_resp.raw_finish_reason.as_deref(), Some("end_turn"));
        assert_eq!(
            final_resp.system_fingerprint.as_deref(),
            Some("fp_terminal")
        );
        assert_eq!(final_resp.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            final_resp.audio.as_ref().map(|audio| audio.id.as_str()),
            Some("aud_123")
        );
        assert_eq!(
            final_resp.warnings,
            Some(vec![Warning::other("watch settings")])
        );
        assert_eq!(
            final_resp.content,
            MessageContent::Text("terminal text".to_string())
        );
    }

    #[test]
    fn stream_start_metadata_falls_back_when_stream_end_missing() {
        let mut sp = StreamProcessor::new();
        let _ = sp.process_event(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("start_resp".to_string()),
                model: Some("gpt-4o-mini".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: Some("req_123".to_string()),
                headers: None,
            },
        });
        let _ = sp.process_event(ChatStreamEvent::ContentDelta {
            delta: "hello world".to_string(),
            index: Some(0),
        });

        let final_resp = sp.build_final_response_with_finish_reason(Some(FinishReason::Stop));

        assert_eq!(final_resp.id.as_deref(), Some("start_resp"));
        assert_eq!(final_resp.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(final_resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            final_resp.content,
            MessageContent::Text("hello world".to_string())
        );
    }

    #[test]
    fn stream_end_content_preserves_extra_parts_when_accumulated_text_exists() {
        let mut sp = StreamProcessor::new();
        let _ = sp.process_event(ChatStreamEvent::ContentDelta {
            delta: "Hello from stream".to_string(),
            index: Some(0),
        });

        let response = ChatResponse {
            id: Some("resp_source".to_string()),
            content: MessageContent::MultiModal(vec![
                ContentPart::Text {
                    text: "terminal fallback".to_string(),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: Some(HashMap::from([(
                        "openai".to_string(),
                        serde_json::json!({ "annotations": ["citation"] }),
                    )])),
                },
                ContentPart::source("source-1", "url", "https://example.com", "Example Source"),
            ]),
            model: Some("gpt-4.1".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Stop),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        };

        let _ = sp.process_event(ChatStreamEvent::StreamEnd { response });
        let final_resp = sp.build_final_response();

        match final_resp.content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);
                assert_eq!(
                    parts[0],
                    ContentPart::Text {
                        text: "Hello from stream".to_string(),
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata: Some(HashMap::from([(
                            "openai".to_string(),
                            serde_json::json!({ "annotations": ["citation"] }),
                        )])),
                    }
                );
                assert_eq!(
                    parts[1],
                    ContentPart::source("source-1", "url", "https://example.com", "Example Source",)
                );
            }
            other => panic!("expected multimodal content, got {:?}", other),
        }
    }

    #[test]
    fn accumulated_tool_calls_preserve_terminal_provider_fields() {
        let mut sp = StreamProcessor::new();
        let _ = sp.process_event(ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("search".to_string()),
            arguments_delta: Some("{\"query\":\"rust\"}".to_string()),
            index: Some(0),
        });

        let response = ChatResponse {
            id: Some("resp_tool".to_string()),
            content: MessageContent::MultiModal(vec![ContentPart::ToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                arguments: serde_json::json!({ "query": "rust" }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({ "server_tool_use": true }),
                )])),
            }]),
            model: Some("claude-3-7-sonnet".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::ToolCalls),
            raw_finish_reason: None,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        };

        let _ = sp.process_event(ChatStreamEvent::StreamEnd { response });
        let final_resp = sp.build_final_response();

        match final_resp.content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(
                    parts[0],
                    ContentPart::ToolCall {
                        tool_call_id: "call_1".to_string(),
                        tool_name: "search".to_string(),
                        arguments: serde_json::json!({ "query": "rust" }),
                        provider_executed: Some(true),
                        dynamic: None,
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata: Some(HashMap::from([(
                            "anthropic".to_string(),
                            serde_json::json!({ "server_tool_use": true }),
                        )])),
                    }
                );
            }
            other => panic!("expected multimodal content, got {:?}", other),
        }
    }

    #[test]
    fn stable_tool_input_start_metadata_flows_to_stable_parts() {
        let mut sp = StreamProcessor::new();

        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: "call_1".to_string(),
                tool_name: "search".to_string(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({ "itemId": "item_1" }),
                )])),
                provider_executed: Some(true),
                dynamic: Some(true),
                title: Some("Web Search".to_string()),
            },
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputDelta {
                id: "call_1".to_string(),
                delta: "{\"query\":\"rust\"}".to_string(),
                provider_metadata: None,
            },
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(crate::types::ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                input: "{\"query\":\"rust\"}".to_string(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: None,
            }),
        });
        let _ = sp.process_event(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolResult(crate::types::ChatStreamToolResult {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                result: serde_json::json!({ "hits": 3 }),
                is_error: None,
                preliminary: Some(true),
                dynamic: None,
                provider_metadata: None,
            }),
        });

        let final_resp = sp.build_final_response_with_finish_reason(Some(FinishReason::ToolCalls));
        let parts = final_resp
            .content
            .as_multimodal()
            .expect("expected multimodal");

        let tool_call = parts
            .iter()
            .find_map(|part| part.as_tool_call())
            .expect("tool call");
        assert_eq!(tool_call.input, &serde_json::json!({ "query": "rust" }));
        assert_eq!(tool_call.provider_executed.copied(), Some(true));
        assert_eq!(tool_call.dynamic.copied(), Some(true));
        assert_eq!(tool_call.title, Some("Web Search"));

        let tool_result = parts
            .iter()
            .find_map(|part| part.as_tool_result())
            .expect("tool result");
        assert_eq!(
            tool_result.input,
            Some(&serde_json::json!({ "query": "rust" }))
        );
        assert_eq!(tool_result.provider_executed.copied(), Some(true));
        assert_eq!(tool_result.dynamic.copied(), Some(true));
        assert_eq!(tool_result.preliminary.copied(), Some(true));
        assert_eq!(tool_result.title, Some("Web Search"));
    }
}
