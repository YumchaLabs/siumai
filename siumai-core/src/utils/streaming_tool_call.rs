//! AI SDK-style streaming tool-call tracker.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::streaming::{
    LanguageModelV4StreamPart, LanguageModelV4StreamToolCall, SharedV4ProviderMetadata,
};
use crate::types::JSONValue;

use super::is_parsable_json;

/// Function payload inside an OpenAI-compatible streaming tool-call delta.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamingToolCallFunctionDelta {
    /// Function/tool name when present.
    pub name: Option<String>,
    /// JSON argument text delta when present.
    pub arguments: Option<String>,
}

/// Minimal OpenAI-compatible streaming tool-call delta.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamingToolCallDelta {
    /// Tool-call index in the provider stream.
    pub index: Option<usize>,
    /// Provider tool-call id.
    pub id: Option<String>,
    /// Provider type field, usually `function`.
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    /// Function/tool delta payload.
    pub function: Option<StreamingToolCallFunctionDelta>,
}

impl StreamingToolCallDelta {
    /// Create a new streaming tool-call delta.
    pub fn new(index: Option<usize>) -> Self {
        Self {
            index,
            ..Self::default()
        }
    }

    /// Attach a provider tool-call id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Attach the provider type field.
    pub fn with_type(mut self, r#type: impl Into<String>) -> Self {
        self.r#type = Some(r#type.into());
        self
    }

    /// Attach a function/tool name.
    pub fn with_function_name(mut self, name: impl Into<String>) -> Self {
        let function = self.function.get_or_insert_with(Default::default);
        function.name = Some(name.into());
        self
    }

    /// Attach a function/tool argument text delta.
    pub fn with_arguments(mut self, arguments: impl Into<String>) -> Self {
        let function = self.function.get_or_insert_with(Default::default);
        function.arguments = Some(arguments.into());
        self
    }
}

/// Type-validation mode for new streaming tool-call deltas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamingToolCallTypeValidation {
    /// Do not validate the delta `type` field.
    #[default]
    None,
    /// Validate `type` only when present.
    IfPresent,
    /// Require `type == "function"`.
    Required,
}

type GenerateToolCallIdFn = Arc<dyn Fn() -> String + Send + Sync + 'static>;
type ExtractToolCallMetadataFn = Arc<
    dyn Fn(&StreamingToolCallDelta) -> Option<SharedV4ProviderMetadata> + Send + Sync + 'static,
>;
type BuildToolCallProviderMetadataFn = Arc<
    dyn Fn(Option<&SharedV4ProviderMetadata>) -> Option<SharedV4ProviderMetadata>
        + Send
        + Sync
        + 'static,
>;

/// Options for [`StreamingToolCallTracker`].
#[derive(Clone, Default)]
pub struct StreamingToolCallTrackerOptions {
    generate_id: Option<GenerateToolCallIdFn>,
    type_validation: StreamingToolCallTypeValidation,
    extract_metadata: Option<ExtractToolCallMetadataFn>,
    build_tool_call_provider_metadata: Option<BuildToolCallProviderMetadataFn>,
}

impl std::fmt::Debug for StreamingToolCallTrackerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingToolCallTrackerOptions")
            .field("has_generate_id", &self.generate_id.is_some())
            .field("type_validation", &self.type_validation)
            .field("has_extract_metadata", &self.extract_metadata.is_some())
            .field(
                "has_build_tool_call_provider_metadata",
                &self.build_tool_call_provider_metadata.is_some(),
            )
            .finish()
    }
}

impl StreamingToolCallTrackerOptions {
    /// Create default tracker options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the upstream-compatible generated-id function option.
    ///
    /// The current OpenAI-compatible delta contract still requires the first
    /// chunk to provide `id`, so this option is retained for shape parity.
    pub fn with_generate_id_fn<F>(mut self, generate_id: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(generate_id));
        self
    }

    /// Set tool-call type-validation mode.
    pub fn with_type_validation(
        mut self,
        type_validation: StreamingToolCallTypeValidation,
    ) -> Self {
        self.type_validation = type_validation;
        self
    }

    /// Extract provider metadata from a newly observed tool-call delta.
    pub fn with_extract_metadata_fn<F>(mut self, extract_metadata: F) -> Self
    where
        F: Fn(&StreamingToolCallDelta) -> Option<SharedV4ProviderMetadata> + Send + Sync + 'static,
    {
        self.extract_metadata = Some(Arc::new(extract_metadata));
        self
    }

    /// Build provider metadata for the final `tool-call` stream part.
    pub fn with_build_tool_call_provider_metadata_fn<F>(
        mut self,
        build_tool_call_provider_metadata: F,
    ) -> Self
    where
        F: Fn(Option<&SharedV4ProviderMetadata>) -> Option<SharedV4ProviderMetadata>
            + Send
            + Sync
            + 'static,
    {
        self.build_tool_call_provider_metadata = Some(Arc::new(build_tool_call_provider_metadata));
        self
    }
}

#[derive(Debug, Clone)]
struct TrackedToolCall {
    id: String,
    name: String,
    arguments: String,
    has_finished: bool,
    metadata: Option<SharedV4ProviderMetadata>,
}

/// Tracks streaming tool-call state across OpenAI-compatible deltas.
#[derive(Debug, Clone, Default)]
pub struct StreamingToolCallTracker {
    tool_calls: Vec<Option<TrackedToolCall>>,
    options: StreamingToolCallTrackerOptions,
}

impl StreamingToolCallTracker {
    /// Create a tracker with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a tracker with explicit options.
    pub fn with_options(options: StreamingToolCallTrackerOptions) -> Self {
        Self {
            tool_calls: Vec::new(),
            options,
        }
    }

    /// Process a streaming tool-call delta and enqueue generated stream parts.
    pub fn process_delta(
        &mut self,
        tool_call_delta: StreamingToolCallDelta,
        mut enqueue: impl FnMut(LanguageModelV4StreamPart),
    ) -> Result<(), LlmError> {
        let index = tool_call_delta.index.unwrap_or(self.tool_calls.len());
        if self.tool_calls.len() <= index {
            self.tool_calls.resize_with(index + 1, || None);
        }

        if self.tool_calls[index].is_none() {
            self.process_new_tool_call(index, tool_call_delta, &mut enqueue)
        } else {
            self.process_existing_tool_call(index, tool_call_delta, &mut enqueue);
            Ok(())
        }
    }

    /// Finalize any unfinished tool calls.
    pub fn flush(&mut self, mut enqueue: impl FnMut(LanguageModelV4StreamPart)) {
        let pending_indexes = self
            .tool_calls
            .iter()
            .enumerate()
            .filter_map(|(index, tool_call)| {
                tool_call
                    .as_ref()
                    .is_some_and(|tool_call| !tool_call.has_finished)
                    .then_some(index)
            })
            .collect::<Vec<_>>();

        for index in pending_indexes {
            self.finish_tool_call(index, &mut enqueue);
        }
    }

    fn process_new_tool_call(
        &mut self,
        index: usize,
        tool_call_delta: StreamingToolCallDelta,
        enqueue: &mut impl FnMut(LanguageModelV4StreamPart),
    ) -> Result<(), LlmError> {
        self.validate_type(&tool_call_delta)?;

        let id = tool_call_delta.id.clone().ok_or_else(|| {
            invalid_streaming_tool_call_delta("Expected 'id' to be a string.", &tool_call_delta)
        })?;
        let name = tool_call_delta
            .function
            .as_ref()
            .and_then(|function| function.name.clone())
            .ok_or_else(|| {
                invalid_streaming_tool_call_delta(
                    "Expected 'function.name' to be a string.",
                    &tool_call_delta,
                )
            })?;

        enqueue(LanguageModelV4StreamPart::ToolInputStart {
            id: id.clone(),
            tool_name: name.clone(),
            provider_metadata: None,
            provider_executed: None,
            dynamic: None,
            title: None,
        });

        let metadata = self
            .options
            .extract_metadata
            .as_ref()
            .and_then(|extract_metadata| extract_metadata(&tool_call_delta));
        let arguments = tool_call_delta
            .function
            .and_then(|function| function.arguments)
            .unwrap_or_default();

        self.tool_calls[index] = Some(TrackedToolCall {
            id,
            name,
            arguments,
            has_finished: false,
            metadata,
        });

        if let Some(tool_call) = self.tool_calls[index].as_ref()
            && !tool_call.arguments.is_empty()
        {
            enqueue(LanguageModelV4StreamPart::ToolInputDelta {
                id: tool_call.id.clone(),
                delta: tool_call.arguments.clone(),
                provider_metadata: None,
            });
        }

        if self.tool_calls[index]
            .as_ref()
            .is_some_and(|tool_call| is_parsable_json(&tool_call.arguments))
        {
            self.finish_tool_call(index, enqueue);
        }

        Ok(())
    }

    fn process_existing_tool_call(
        &mut self,
        index: usize,
        tool_call_delta: StreamingToolCallDelta,
        enqueue: &mut impl FnMut(LanguageModelV4StreamPart),
    ) {
        let Some(tool_call) = self.tool_calls[index].as_mut() else {
            return;
        };

        if tool_call.has_finished {
            return;
        }

        if let Some(arguments) = tool_call_delta
            .function
            .and_then(|function| function.arguments)
        {
            tool_call.arguments.push_str(&arguments);
            enqueue(LanguageModelV4StreamPart::ToolInputDelta {
                id: tool_call.id.clone(),
                delta: arguments,
                provider_metadata: None,
            });
        }

        if is_parsable_json(&tool_call.arguments) {
            self.finish_tool_call(index, enqueue);
        }
    }

    fn finish_tool_call(
        &mut self,
        index: usize,
        enqueue: &mut impl FnMut(LanguageModelV4StreamPart),
    ) {
        let Some(tool_call) = self.tool_calls.get_mut(index).and_then(Option::as_mut) else {
            return;
        };

        if tool_call.has_finished {
            return;
        }

        enqueue(LanguageModelV4StreamPart::ToolInputEnd {
            id: tool_call.id.clone(),
            provider_metadata: None,
        });

        let provider_metadata = self
            .options
            .build_tool_call_provider_metadata
            .as_ref()
            .and_then(|build_provider_metadata| {
                build_provider_metadata(tool_call.metadata.as_ref())
            });
        enqueue(LanguageModelV4StreamPart::ToolCall(
            LanguageModelV4StreamToolCall {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                input: tool_call.arguments.clone(),
                provider_executed: None,
                dynamic: None,
                provider_metadata,
            },
        ));

        tool_call.has_finished = true;
    }

    fn validate_type(&self, tool_call_delta: &StreamingToolCallDelta) -> Result<(), LlmError> {
        match self.options.type_validation {
            StreamingToolCallTypeValidation::None => Ok(()),
            StreamingToolCallTypeValidation::IfPresent => {
                if tool_call_delta
                    .r#type
                    .as_deref()
                    .is_some_and(|ty| ty != "function")
                {
                    Err(invalid_streaming_tool_call_delta(
                        "Expected 'function' type.",
                        tool_call_delta,
                    ))
                } else {
                    Ok(())
                }
            }
            StreamingToolCallTypeValidation::Required => {
                if tool_call_delta.r#type.as_deref() == Some("function") {
                    Ok(())
                } else {
                    Err(invalid_streaming_tool_call_delta(
                        "Expected 'function' type.",
                        tool_call_delta,
                    ))
                }
            }
        }
    }
}

fn invalid_streaming_tool_call_delta(message: &str, delta: &StreamingToolCallDelta) -> LlmError {
    let data = serde_json::to_value(delta).unwrap_or(JSONValue::Null);
    LlmError::InvalidInput(format!("{message} Delta: {data}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_emits_parts_when_arguments_become_parsable_json() {
        let mut tracker = StreamingToolCallTracker::new();
        let mut parts = Vec::new();

        tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0))
                    .with_id("call_1")
                    .with_type("function")
                    .with_function_name("search")
                    .with_arguments("{\"query\""),
                |part| parts.push(part),
            )
            .expect("first delta");
        tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0)).with_arguments(":\"tea\"}"),
                |part| parts.push(part),
            )
            .expect("second delta");

        assert_eq!(parts.len(), 5);
        assert!(matches!(
            &parts[0],
            LanguageModelV4StreamPart::ToolInputStart { id, tool_name, .. }
                if id == "call_1" && tool_name == "search"
        ));
        assert!(matches!(
            &parts[1],
            LanguageModelV4StreamPart::ToolInputDelta { delta, .. }
                if delta == "{\"query\""
        ));
        assert!(matches!(
            &parts[2],
            LanguageModelV4StreamPart::ToolInputDelta { delta, .. }
                if delta == ":\"tea\"}"
        ));
        assert!(matches!(
            &parts[3],
            LanguageModelV4StreamPart::ToolInputEnd { id, .. } if id == "call_1"
        ));
        assert!(matches!(
            &parts[4],
            LanguageModelV4StreamPart::ToolCall(call)
                if call.tool_call_id == "call_1"
                    && call.tool_name == "search"
                    && call.input == "{\"query\":\"tea\"}"
        ));
    }

    #[test]
    fn tracker_flushes_incomplete_tool_calls() {
        let mut tracker = StreamingToolCallTracker::new();
        let mut parts = Vec::new();

        tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0))
                    .with_id("call_1")
                    .with_function_name("search")
                    .with_arguments("{\"query\""),
                |part| parts.push(part),
            )
            .expect("delta");
        tracker.flush(|part| parts.push(part));

        assert!(matches!(
            parts.last(),
            Some(LanguageModelV4StreamPart::ToolCall(call))
                if call.input == "{\"query\""
        ));
    }

    #[test]
    fn tracker_validates_new_tool_call_type() {
        let mut tracker = StreamingToolCallTracker::with_options(
            StreamingToolCallTrackerOptions::new()
                .with_type_validation(StreamingToolCallTypeValidation::Required),
        );

        let error = tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0))
                    .with_id("call_1")
                    .with_function_name("search"),
                |_| {},
            )
            .expect_err("type required");

        assert!(matches!(error, LlmError::InvalidInput(_)));
    }

    #[test]
    fn tracker_requires_id_on_new_tool_call() {
        let mut tracker = StreamingToolCallTracker::new();

        let error = tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0)).with_function_name("search"),
                |_| {},
            )
            .expect_err("id required");

        assert!(matches!(error, LlmError::InvalidInput(_)));
    }

    #[test]
    fn tracker_preserves_empty_provider_id() {
        let mut tracker = StreamingToolCallTracker::with_options(
            StreamingToolCallTrackerOptions::new().with_generate_id_fn(|| "generated".to_string()),
        );
        let mut parts = Vec::new();

        tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0))
                    .with_id("")
                    .with_function_name("search")
                    .with_arguments("{}"),
                |part| parts.push(part),
            )
            .expect("delta");

        assert!(matches!(
            parts.last(),
            Some(LanguageModelV4StreamPart::ToolCall(call)) if call.tool_call_id.is_empty()
        ));
    }

    #[test]
    fn tracker_builds_provider_metadata() {
        let mut tracker = StreamingToolCallTracker::with_options(
            StreamingToolCallTrackerOptions::new()
                .with_extract_metadata_fn(|_| {
                    Some(SharedV4ProviderMetadata::from([(
                        "openai".to_string(),
                        serde_json::json!({ "trace": "1" }),
                    )]))
                })
                .with_build_tool_call_provider_metadata_fn(|metadata| metadata.cloned()),
        );
        let mut parts = Vec::new();

        tracker
            .process_delta(
                StreamingToolCallDelta::new(Some(0))
                    .with_id("call_1")
                    .with_function_name("search")
                    .with_arguments("{}"),
                |part| parts.push(part),
            )
            .expect("delta");

        assert!(matches!(
            parts.last(),
            Some(LanguageModelV4StreamPart::ToolCall(call))
                if call.provider_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("openai"))
                    .and_then(|metadata| metadata.get("trace"))
                    == Some(&serde_json::json!("1"))
        ));
    }
}
