//! Structured output helpers.
//!
//! This module provides provider-agnostic JSON extraction utilities for responses produced
//! with `ChatRequest.response_format`.

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use siumai_core::error::LlmError;
use siumai_core::streaming::ChatStream;
use siumai_core::types::{
    CallWarning, ChatResponse, FinishReason, FlexibleSchema, JSONSchema7,
    LanguageModelRequestMetadata, LanguageModelResponseMetadata, LanguageModelUsage,
    ProviderMetadata, ResponseFormat, Schema, ValidationResult,
};
use siumai_core::utils::generate_id;

use crate::text::{GenerateOptions, LanguageModel, TextRequest};

/// Schema input accepted by `generate_object`.
#[derive(Clone)]
pub enum GenerateObjectSchema<T = serde_json::Value> {
    /// Concrete or lazy typed schema.
    Flexible(FlexibleSchema<T>),
    /// Plain JSON Schema without a runtime validator.
    Json(JSONSchema7),
}

impl<T> std::fmt::Debug for GenerateObjectSchema<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flexible(schema) => f.debug_tuple("Flexible").field(schema).finish(),
            Self::Json(schema) => f.debug_tuple("Json").field(schema).finish(),
        }
    }
}

impl<T> GenerateObjectSchema<T> {
    fn into_schema(self) -> Schema<T> {
        match self {
            Self::Flexible(schema) => schema.into_schema(),
            Self::Json(schema) => Schema::new(schema),
        }
    }
}

impl<T> From<Schema<T>> for GenerateObjectSchema<T> {
    fn from(value: Schema<T>) -> Self {
        Self::Flexible(value.into())
    }
}

impl<T> From<siumai_core::types::LazySchema<T>> for GenerateObjectSchema<T> {
    fn from(value: siumai_core::types::LazySchema<T>) -> Self {
        Self::Flexible(value.into())
    }
}

impl<T> From<FlexibleSchema<T>> for GenerateObjectSchema<T> {
    fn from(value: FlexibleSchema<T>) -> Self {
        Self::Flexible(value)
    }
}

impl<T> From<JSONSchema7> for GenerateObjectSchema<T> {
    fn from(value: JSONSchema7) -> Self {
        Self::Json(value)
    }
}

/// Options for `structured_output::generate_object`.
#[derive(Debug, Clone, Default)]
pub struct GenerateObjectOptions {
    /// Text generation options used for the underlying model call.
    pub generate_options: GenerateOptions,
    /// Optional schema name used by providers that support named JSON Schema output.
    pub schema_name: Option<String>,
    /// Optional schema description used by providers that support it.
    pub schema_description: Option<String>,
    /// Optional strictness hint.
    pub strict: Option<bool>,
}

impl GenerateObjectOptions {
    /// Create default object generation options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set underlying text generation options.
    pub fn with_generate_options(mut self, generate_options: GenerateOptions) -> Self {
        self.generate_options = generate_options;
        self
    }

    /// Set schema name.
    pub fn with_schema_name(mut self, schema_name: impl Into<String>) -> Self {
        self.schema_name = Some(schema_name.into());
        self
    }

    /// Set schema description.
    pub fn with_schema_description(mut self, schema_description: impl Into<String>) -> Self {
        self.schema_description = Some(schema_description.into());
        self
    }

    /// Set strictness hint.
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

/// Result of a structured object generation call.
#[derive(Debug, Clone)]
pub struct GenerateObjectResult<T> {
    /// Generated object.
    pub object: T,
    /// Concatenated reasoning text when available.
    pub reasoning: Option<String>,
    /// Finish reason from the underlying model response.
    pub finish_reason: FinishReason,
    /// Token usage projected onto the AI SDK language-model usage shape.
    pub usage: LanguageModelUsage,
    /// Warnings emitted by the model provider.
    pub warnings: Option<Vec<CallWarning>>,
    /// Best-effort request metadata from the stable Rust request shape.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata. Missing provider ids fall back to an SDK-generated id.
    pub response: LanguageModelResponseMetadata,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Raw underlying text/chat response for callers that need fields not projected above.
    pub raw_response: ChatResponse,
}

/// Generate a typed object from a language model using a JSON Schema response format.
///
/// This is the Rust equivalent of AI SDK `generateObject` for the non-streaming path. It does not
/// fabricate provider HTTP bodies; metadata is populated from stable model and response data.
pub async fn generate_object<M, T, S>(
    model: &M,
    request: TextRequest,
    schema: S,
    options: GenerateObjectOptions,
) -> Result<GenerateObjectResult<T>, LlmError>
where
    M: LanguageModel + ?Sized,
    T: DeserializeOwned,
    S: Into<GenerateObjectSchema<T>>,
{
    let schema = schema.into().into_schema();
    let mut response_format = ResponseFormat::json_schema(schema.json_schema().clone());

    if let Some(name) = options.schema_name {
        response_format = response_format.with_name(name);
    }
    if let Some(description) = options.schema_description {
        response_format = response_format.with_description(description);
    }
    if let Some(strict) = options.strict {
        response_format = response_format.with_strict(strict);
    }

    let request = request.with_response_format(response_format);
    let (request, effective_options) =
        crate::text::prepare_generate_request(request, options.generate_options);
    let request_metadata = LanguageModelRequestMetadata {
        body: serde_json::to_value(&request).ok(),
    };
    let response_timestamp = Utc::now();
    let model_id = model.model_id().to_string();

    let response = crate::text::generate_prepared(model, request, effective_options).await?;
    let value = extract_json_value_from_response(&response)?;
    let object = parse_generated_object(&schema, value)?;

    Ok(GenerateObjectResult::from_response(
        object,
        request_metadata,
        response_timestamp,
        model_id,
        response,
    ))
}

/// Extract a `serde_json::Value` from a model output string.
pub fn extract_json_value(text: &str) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value(text)
}

/// Extract a `serde_json::Value` from a unified chat response.
pub fn extract_json_value_from_response(
    response: &ChatResponse,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value_from_response(response)
}

/// Extract a `serde_json::Value` from a streaming response.
pub async fn extract_json_value_from_stream(
    stream: ChatStream,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value_from_stream(stream).await
}

/// Extract and deserialize structured output into a typed value.
pub fn extract_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json(text)
}

/// Extract and deserialize structured output from a stream into a typed value.
pub async fn extract_json_from_stream<T: serde::de::DeserializeOwned>(
    stream: ChatStream,
) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json_from_stream(stream).await
}

/// Extract and deserialize structured output from a response into a typed value.
pub fn extract_json_from_response<T: serde::de::DeserializeOwned>(
    response: &ChatResponse,
) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json_from_response(response)
}

fn parse_generated_object<T>(schema: &Schema<T>, value: serde_json::Value) -> Result<T, LlmError>
where
    T: DeserializeOwned,
{
    match schema.validate(&value) {
        Some(ValidationResult::Success { value }) => Ok(value),
        Some(ValidationResult::Failure { error }) => Err(error),
        None => serde_json::from_value(value).map_err(|error| {
            LlmError::ParseError(format!(
                "Failed to deserialize generated object into target type: {error}"
            ))
        }),
    }
}

impl<T> GenerateObjectResult<T> {
    fn from_response(
        object: T,
        request: LanguageModelRequestMetadata,
        response_timestamp: DateTime<Utc>,
        model_id: String,
        raw_response: ChatResponse,
    ) -> Self {
        let reasoning = raw_response
            .reasoning()
            .into_iter()
            .collect::<Vec<_>>()
            .join("\n");
        let response = language_model_response_metadata_from_chat_response(
            &raw_response,
            response_timestamp,
            model_id,
        );

        Self {
            object,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            finish_reason: raw_response
                .finish_reason
                .clone()
                .unwrap_or(FinishReason::Unknown),
            usage: raw_response
                .usage
                .as_ref()
                .map(LanguageModelUsage::from)
                .unwrap_or_default(),
            warnings: raw_response.warnings.clone(),
            request,
            response,
            provider_metadata: raw_response.provider_metadata.clone(),
            raw_response,
        }
    }
}

fn language_model_response_metadata_from_chat_response(
    response: &ChatResponse,
    timestamp: DateTime<Utc>,
    model_id: String,
) -> LanguageModelResponseMetadata {
    LanguageModelResponseMetadata {
        id: response.id.clone().unwrap_or_else(generate_id),
        timestamp,
        model_id: response.model.clone().unwrap_or(model_id),
        headers: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::TextModelV3;
    use async_trait::async_trait;
    use serde::Deserialize;
    use std::sync::{Arc, Mutex};

    struct FakeObjectModel {
        response: ChatResponse,
        seen_request: Arc<Mutex<Option<TextRequest>>>,
    }

    impl siumai_core::traits::ModelMetadata for FakeObjectModel {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-model"
        }
    }

    #[async_trait]
    impl TextModelV3 for FakeObjectModel {
        async fn generate(&self, request: TextRequest) -> Result<ChatResponse, LlmError> {
            *self.seen_request.lock().expect("request lock") = Some(request);
            Ok(self.response.clone())
        }

        async fn stream(&self, _request: TextRequest) -> Result<crate::text::TextStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream not used in generate_object tests".to_string(),
            ))
        }

        async fn stream_with_cancel(
            &self,
            _request: TextRequest,
        ) -> Result<crate::text::TextStreamHandle, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream_with_cancel not used in generate_object tests".to_string(),
            ))
        }
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
    }

    #[tokio::test]
    async fn generate_object_sets_response_format_and_projects_result() {
        let seen_request = Arc::new(Mutex::new(None));
        let mut response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"name\":\"Ada\"}".to_string(),
        ));
        response.id = Some("resp_1".to_string());
        response.model = Some("fake-model".to_string());
        response.finish_reason = Some(FinishReason::Stop);
        response.usage = Some(siumai_core::types::Usage::new(3, 5));
        response.provider_metadata = Some(std::collections::HashMap::from([(
            "fake".to_string(),
            serde_json::json!({ "traceId": "trace_1" }),
        )]));

        let model = FakeObjectModel {
            response,
            seen_request: seen_request.clone(),
        };

        let result: GenerateObjectResult<Person> = generate_object(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }),
            GenerateObjectOptions::new()
                .with_schema_name("person")
                .with_schema_description("Person payload")
                .with_strict(true),
        )
        .await
        .expect("generate object");

        assert_eq!(
            result.object,
            Person {
                name: "Ada".to_string()
            }
        );
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.usage.input_tokens, Some(3));
        assert_eq!(result.response.model_id, "fake-model");
        assert_eq!(
            result
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("fake"))
                .and_then(|metadata| metadata.get("traceId"))
                .and_then(serde_json::Value::as_str),
            Some("trace_1")
        );
        assert_eq!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("responseFormat"))
                .and_then(|format| format.get("type"))
                .and_then(serde_json::Value::as_str),
            Some("json")
        );

        let request = seen_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("request should be captured");
        let response_format = request
            .response_format
            .as_ref()
            .expect("response format should be set");
        match response_format {
            ResponseFormat::Json {
                name,
                description,
                strict,
                ..
            } => {
                assert_eq!(name.as_deref(), Some("person"));
                assert_eq!(description.as_deref(), Some("Person payload"));
                assert_eq!(*strict, Some(true));
            }
        }
    }

    #[tokio::test]
    async fn generate_object_uses_typed_schema_validator_when_present() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"name\":\"Ada\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request,
        };
        let schema = siumai_core::types::json_schema_with_validator(
            serde_json::json!({ "type": "object" }),
            |value| {
                value
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| {
                        ValidationResult::success(Person {
                            name: name.to_uppercase(),
                        })
                    })
                    .unwrap_or_else(|| {
                        ValidationResult::failure(LlmError::ParseError("missing name".to_string()))
                    })
            },
        );

        let result = generate_object(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            schema,
            GenerateObjectOptions::default(),
        )
        .await
        .expect("generate object");

        assert_eq!(
            result.object,
            Person {
                name: "ADA".to_string()
            }
        );
    }
}
