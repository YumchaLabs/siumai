//! Structured output helpers.
//!
//! This module provides provider-agnostic JSON extraction utilities for responses produced
//! with `ChatRequest.response_format`.

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use siumai_core::error::LlmError;
use siumai_core::streaming::ChatStream;
use siumai_core::types::{
    CallWarning, ChatResponse, FinishReason, FlexibleSchema, GenerateObjectResponseMetadata,
    JSONSchema7, LanguageModelRequestMetadata, LanguageModelResponseMetadata, LanguageModelUsage,
    ProviderMetadata, ResponseFormat, Schema, ValidationResult,
};
use siumai_core::utils::generate_id;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::text::{GenerateOptions, LanguageModel, TextRequest};

pub use siumai_core::structured_output::{
    PartialJsonParseResult, PartialJsonParseState, PartialJsonValueStream,
    PartialJsonValueStreamEvent,
};

/// Context passed to a structured-output repair callback.
#[derive(Debug, Clone)]
pub struct RepairTextContext {
    /// Raw text extracted from the model response.
    pub text: String,
    /// Parse or validation error that triggered repair.
    pub error: LlmError,
}

/// Future returned by a structured-output repair callback.
pub type RepairTextFuture = Pin<Box<dyn Future<Output = Result<Option<String>, LlmError>> + Send>>;

/// Async callback that can repair malformed or schema-invalid model output text.
pub type RepairTextFunction = Arc<dyn Fn(RepairTextContext) -> RepairTextFuture + Send + Sync>;

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
#[derive(Clone, Default)]
pub struct GenerateObjectOptions {
    /// Text generation options used for the underlying model call.
    pub generate_options: GenerateOptions,
    /// Optional schema name used by providers that support named JSON Schema output.
    pub schema_name: Option<String>,
    /// Optional schema description used by providers that support it.
    pub schema_description: Option<String>,
    /// Optional strictness hint.
    pub strict: Option<bool>,
    /// Optional repair callback invoked after initial JSON parsing or validation fails.
    pub repair_text: Option<RepairTextFunction>,
}

impl std::fmt::Debug for GenerateObjectOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateObjectOptions")
            .field("generate_options", &self.generate_options)
            .field("schema_name", &self.schema_name)
            .field("schema_description", &self.schema_description)
            .field("strict", &self.strict)
            .field("has_repair_text", &self.repair_text.is_some())
            .finish()
    }
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

    /// Set an async repair callback.
    pub fn with_repair_text(mut self, repair_text: RepairTextFunction) -> Self {
        self.repair_text = Some(repair_text);
        self
    }

    /// Set an async repair callback from a Rust closure.
    pub fn with_repair_text_fn<F, Fut>(mut self, repair_text: F) -> Self
    where
        F: Fn(RepairTextContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>, LlmError>> + Send + 'static,
    {
        self.repair_text = Some(Arc::new(move |context| Box::pin(repair_text(context))));
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
    pub response: GenerateObjectResponseMetadata,
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
    let response_schema = schema.json_schema().clone();

    generate_with_response_schema(model, request, response_schema, options, move |value| {
        parse_generated_object(&schema, value)
    })
    .await
}

/// Generate a typed array from a language model using AI SDK's wrapped array output strategy.
///
/// Providers receive a JSON Schema for `{ "elements": [...] }`, matching AI SDK's `output:
/// "array"` strategy. The returned Rust object is the extracted `Vec<T>`.
pub async fn generate_array<M, T, S>(
    model: &M,
    request: TextRequest,
    element_schema: S,
    options: GenerateObjectOptions,
) -> Result<GenerateObjectResult<Vec<T>>, LlmError>
where
    M: LanguageModel + ?Sized,
    T: DeserializeOwned,
    S: Into<GenerateObjectSchema<T>>,
{
    let schema = element_schema.into().into_schema();
    let response_schema = array_output_json_schema(schema.json_schema().clone());

    generate_with_response_schema(model, request, response_schema, options, move |value| {
        parse_generated_array(&schema, value)
    })
    .await
}

/// Generate an arbitrary JSON value without a provider-facing JSON Schema.
///
/// This is the Rust equivalent of AI SDK `generateText({ output: json() })` for the
/// non-streaming path. Providers receive a schema-less JSON response format when they support it.
pub async fn generate_json<M>(
    model: &M,
    request: TextRequest,
    options: GenerateObjectOptions,
) -> Result<GenerateObjectResult<serde_json::Value>, LlmError>
where
    M: LanguageModel + ?Sized,
{
    let response_format = response_format_from_json_object(&options)?;

    generate_with_response_format(model, request, response_format, options, Ok).await
}

/// Generate one string from a fixed choice set using AI SDK's wrapped choice output strategy.
///
/// This mirrors AI SDK `generateText({ output: choice(...) })`. Unlike `generate_enum`, this
/// helper accepts `schema_name` and `schema_description` because the newer `Output.choice()`
/// surface forwards those labels on the JSON response format.
pub async fn generate_choice<M, I, V>(
    model: &M,
    request: TextRequest,
    choices: I,
    options: GenerateObjectOptions,
) -> Result<GenerateObjectResult<String>, LlmError>
where
    M: LanguageModel + ?Sized,
    I: IntoIterator<Item = V>,
    V: Into<String>,
{
    let choices = choices.into_iter().map(Into::into).collect::<Vec<_>>();
    let response_schema = enum_output_json_schema(&choices);

    generate_with_response_schema(model, request, response_schema, options, move |value| {
        parse_generated_enum(&choices, value)
    })
    .await
}

/// Generate one string from a fixed enum set using AI SDK's wrapped enum output strategy.
///
/// Providers receive a JSON Schema for `{ "result": "..." }`, matching AI SDK's `output:
/// "enum"` strategy. Schema names and descriptions are rejected for this helper to preserve the
/// upstream contract.
pub async fn generate_enum<M, I, V>(
    model: &M,
    request: TextRequest,
    enum_values: I,
    options: GenerateObjectOptions,
) -> Result<GenerateObjectResult<String>, LlmError>
where
    M: LanguageModel + ?Sized,
    I: IntoIterator<Item = V>,
    V: Into<String>,
{
    validate_enum_options(&options)?;
    let enum_values = enum_values.into_iter().map(Into::into).collect::<Vec<_>>();
    let response_schema = enum_output_json_schema(&enum_values);

    generate_with_response_schema(model, request, response_schema, options, move |value| {
        parse_generated_enum(&enum_values, value)
    })
    .await
}

async fn generate_with_response_schema<M, T, P>(
    model: &M,
    request: TextRequest,
    response_schema: JSONSchema7,
    options: GenerateObjectOptions,
    parse: P,
) -> Result<GenerateObjectResult<T>, LlmError>
where
    M: LanguageModel + ?Sized,
    P: Fn(serde_json::Value) -> Result<T, LlmError>,
{
    let response_format = response_format_from_schema(response_schema, &options);
    generate_with_response_format(model, request, response_format, options, parse).await
}

async fn generate_with_response_format<M, T, P>(
    model: &M,
    request: TextRequest,
    response_format: ResponseFormat,
    options: GenerateObjectOptions,
    parse: P,
) -> Result<GenerateObjectResult<T>, LlmError>
where
    M: LanguageModel + ?Sized,
    P: Fn(serde_json::Value) -> Result<T, LlmError>,
{
    let repair_text = options.repair_text.clone();
    let request = request.with_response_format(response_format);
    let (request, effective_options) =
        crate::text::prepare_generate_request(request, options.generate_options);
    let response_timestamp = Utc::now();
    let model_id = model.model_id().to_string();

    let response = crate::text::generate_prepared(model, request, effective_options).await?;
    let request_metadata = response
        .request
        .as_ref()
        .map(LanguageModelRequestMetadata::from)
        .unwrap_or_default();
    let response_metadata = language_model_response_metadata_from_chat_response(
        &response,
        response_timestamp,
        model_id,
    );
    let finish_reason = response
        .finish_reason
        .clone()
        .unwrap_or(FinishReason::Unknown);
    let usage = response
        .usage
        .as_ref()
        .map(LanguageModelUsage::from)
        .unwrap_or_default();
    let object = parse_generated_value_with_optional_repair(&response, parse, repair_text)
        .await
        .map_err(|error| {
            no_object_generated_error(
                &response,
                response_metadata.clone(),
                usage.clone(),
                finish_reason.clone(),
                error,
            )
        })?;

    Ok(GenerateObjectResult::from_response(
        object,
        request_metadata,
        response_metadata,
        response,
    ))
}

async fn parse_generated_value_with_optional_repair<T, P>(
    response: &ChatResponse,
    parse: P,
    repair_text: Option<RepairTextFunction>,
) -> Result<T, LlmError>
where
    P: Fn(serde_json::Value) -> Result<T, LlmError>,
{
    let initial = extract_json_value_from_response(response).and_then(&parse);

    match initial {
        Ok(object) => Ok(object),
        Err(error) => {
            let Some(repair_text) = repair_text else {
                return Err(error);
            };
            let context = RepairTextContext {
                text: response.text().unwrap_or_default(),
                error: error.clone(),
            };
            let Some(repaired_text) = repair_text(context).await? else {
                return Err(error);
            };
            let repaired_value = extract_json_value(&repaired_text)?;
            parse(repaired_value)
        }
    }
}

fn no_object_generated_error(
    response: &ChatResponse,
    response_metadata: GenerateObjectResponseMetadata,
    usage: LanguageModelUsage,
    finish_reason: FinishReason,
    error: LlmError,
) -> LlmError {
    LlmError::NoObjectGenerated {
        message: "No object generated: response did not match the requested structured output."
            .to_string(),
        text: response.text(),
        response: Some(Box::new(response_metadata)),
        usage: Some(Box::new(usage)),
        finish_reason: Some(finish_reason),
        cause: Some(Box::new(error)),
    }
}

fn response_format_from_schema(
    schema: JSONSchema7,
    options: &GenerateObjectOptions,
) -> ResponseFormat {
    let mut response_format = ResponseFormat::json_schema(schema);

    if let Some(name) = options.schema_name.clone() {
        response_format = response_format.with_name(name);
    }
    if let Some(description) = options.schema_description.clone() {
        response_format = response_format.with_description(description);
    }
    if let Some(strict) = options.strict {
        response_format = response_format.with_strict(strict);
    }

    response_format
}

fn response_format_from_json_object(
    options: &GenerateObjectOptions,
) -> Result<ResponseFormat, LlmError> {
    if options.strict.is_some() {
        return Err(LlmError::InvalidParameter(
            "strict is not supported for schema-less JSON output".to_string(),
        ));
    }

    let mut response_format = ResponseFormat::json_object();

    if let Some(name) = options.schema_name.clone() {
        response_format = response_format.with_name(name);
    }
    if let Some(description) = options.schema_description.clone() {
        response_format = response_format.with_description(description);
    }

    Ok(response_format)
}

fn array_output_json_schema(mut item_schema: JSONSchema7) -> JSONSchema7 {
    if let Some(object) = item_schema.as_object_mut() {
        object.remove("$schema");
    }

    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "items": item_schema,
            },
        },
        "required": ["elements"],
        "additionalProperties": false,
    })
}

fn enum_output_json_schema(enum_values: &[String]) -> JSONSchema7 {
    serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "enum": enum_values,
            },
        },
        "required": ["result"],
        "additionalProperties": false,
    })
}

fn validate_enum_options(options: &GenerateObjectOptions) -> Result<(), LlmError> {
    if options.schema_name.is_some() {
        return Err(LlmError::InvalidParameter(
            "schema_name is not supported for enum output".to_string(),
        ));
    }
    if options.schema_description.is_some() {
        return Err(LlmError::InvalidParameter(
            "schema_description is not supported for enum output".to_string(),
        ));
    }
    Ok(())
}

/// Extract a `serde_json::Value` from a model output string.
pub fn extract_json_value(text: &str) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value(text)
}

/// Repair incomplete JSON using AI SDK `fixJson` semantics.
pub fn fix_partial_json(input: &str) -> String {
    siumai_core::structured_output::fix_partial_json(input)
}

/// Parse a partial JSON string using AI SDK `parsePartialJson` semantics.
pub fn parse_partial_json(json_text: Option<&str>) -> PartialJsonParseResult {
    siumai_core::structured_output::parse_partial_json(json_text)
}

/// Parse partial and final JSON values from an existing chat stream.
pub fn partial_json_value_stream(stream: ChatStream) -> PartialJsonValueStream {
    siumai_core::structured_output::partial_json_value_stream(stream)
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

fn parse_generated_array<T>(
    schema: &Schema<T>,
    value: serde_json::Value,
) -> Result<Vec<T>, LlmError>
where
    T: DeserializeOwned,
{
    let elements = value
        .get("elements")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            LlmError::ParseError(
                "Generated array output must be an object with an elements array".to_string(),
            )
        })?;

    elements
        .iter()
        .cloned()
        .map(|element| parse_generated_object(schema, element))
        .collect()
}

fn parse_generated_enum(
    enum_values: &[String],
    value: serde_json::Value,
) -> Result<String, LlmError> {
    let result = value
        .get("result")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            LlmError::ParseError(
                "Generated enum output must be an object with a string result".to_string(),
            )
        })?;

    if enum_values.iter().any(|value| value == result) {
        Ok(result.to_string())
    } else {
        Err(LlmError::ParseError(format!(
            "Generated enum output {result:?} is not one of the allowed values"
        )))
    }
}

impl<T> GenerateObjectResult<T> {
    fn from_response(
        object: T,
        request: LanguageModelRequestMetadata,
        response: GenerateObjectResponseMetadata,
        raw_response: ChatResponse,
    ) -> Self {
        let reasoning = raw_response
            .reasoning()
            .into_iter()
            .collect::<Vec<_>>()
            .join("\n");

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
            warnings: raw_response
                .warnings
                .clone()
                .map(|warnings| warnings.into_iter().map(CallWarning::from).collect()),
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
) -> GenerateObjectResponseMetadata {
    let http_response = response.response.as_ref();
    let mut metadata = GenerateObjectResponseMetadata::new(LanguageModelResponseMetadata {
        id: response.id.clone().unwrap_or_else(generate_id),
        timestamp: http_response
            .map(|response| response.timestamp)
            .unwrap_or(timestamp),
        model_id: http_response
            .and_then(|response| response.model_id.clone())
            .or_else(|| response.model.clone())
            .unwrap_or(model_id),
        headers: http_response
            .map(|response| response.headers.clone())
            .filter(|headers| !headers.is_empty()),
    });

    if let Some(body) = http_response.and_then(|response| response.body.clone()) {
        metadata = metadata.with_body(body);
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::TextModelV3;
    use async_trait::async_trait;
    use serde::Deserialize;
    use siumai_core::types::{HttpRequestInfo, HttpResponseInfo};
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
        response.warnings = Some(vec![siumai_core::types::Warning::UnsupportedTool {
            tool_name: "legacy-tool".to_string(),
            details: Some("not available".to_string()),
        }]);
        response.request = Some(HttpRequestInfo {
            body: Some(
                serde_json::json!({
                    "model": "provider-model",
                    "messages": [{ "role": "user", "content": "json" }],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "person",
                            "strict": true
                        }
                    }
                })
                .to_string(),
            ),
        });
        response.response = Some(HttpResponseInfo {
            timestamp: chrono::DateTime::parse_from_rfc3339("2026-04-30T00:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&chrono::Utc),
            model_id: Some("fake-model".to_string()),
            headers: std::collections::HashMap::from([(
                "x-request-id".to_string(),
                "obj_req_123".to_string(),
            )]),
            body: Some(serde_json::json!({ "id": "resp_1", "raw": true })),
        });

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
        assert_eq!(result.response.metadata.model_id, "fake-model");
        assert_eq!(
            result
                .response
                .metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("obj_req_123")
        );
        assert_eq!(
            result.response.body,
            Some(serde_json::json!({ "id": "resp_1", "raw": true }))
        );
        assert_eq!(
            serde_json::to_value(result.warnings.as_ref().expect("warnings"))
                .expect("serialize warnings"),
            serde_json::json!([{
                "type": "unsupported",
                "feature": "legacy-tool",
                "details": "not available"
            }])
        );
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
                .and_then(|body| body.get("model")),
            Some(&serde_json::json!("provider-model"))
        );
        assert_eq!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("response_format"))
                .and_then(|format| format.get("type"))
                .and_then(serde_json::Value::as_str),
            Some("json_schema")
        );
        assert!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("responseFormat"))
                .is_none()
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
            ResponseFormat::JsonObject { .. } => {
                panic!("expected schema-backed JSON response format");
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

    #[tokio::test]
    async fn generate_object_repairs_validation_failure_when_callback_returns_text() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"title\":\"Ada\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request,
        };
        let seen_repair_text = Arc::new(Mutex::new(None));
        let seen_repair_text_for_callback = seen_repair_text.clone();

        let result: GenerateObjectResult<Person> = generate_object(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            GenerateObjectOptions::new().with_repair_text_fn(move |context| {
                let seen_repair_text = seen_repair_text_for_callback.clone();
                async move {
                    *seen_repair_text.lock().expect("repair text lock") = Some(context.text);
                    Ok(Some("{\"name\":\"Ada\"}".to_string()))
                }
            }),
        )
        .await
        .expect("generate object with repair");

        assert_eq!(
            result.object,
            Person {
                name: "Ada".to_string()
            }
        );
        assert_eq!(
            seen_repair_text
                .lock()
                .expect("repair text lock")
                .as_deref(),
            Some("{\"title\":\"Ada\"}")
        );
    }

    #[tokio::test]
    async fn generate_object_wraps_final_parse_failure_as_no_object_generated() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"title\":\"Ada\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request,
        };

        let error = generate_object::<_, Person, _>(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            GenerateObjectOptions::default(),
        )
        .await
        .expect_err("invalid object should use structured error");

        let LlmError::NoObjectGenerated {
            text,
            response,
            finish_reason,
            cause,
            ..
        } = error
        else {
            panic!("expected no object generated error");
        };

        assert_eq!(text.as_deref(), Some("{\"title\":\"Ada\"}"));
        assert_eq!(
            response.map(|response| response.metadata.model_id),
            Some("fake-model".to_string())
        );
        assert_eq!(finish_reason, Some(FinishReason::Unknown));
        assert!(matches!(cause.as_deref(), Some(LlmError::ParseError(_))));
    }

    #[tokio::test]
    async fn generate_array_wraps_schema_and_extracts_elements() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"elements\":[{\"name\":\"Ada\"},{\"name\":\"Grace\"}]}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request: seen_request.clone(),
        };
        let schema = siumai_core::types::json_schema_with_validator(
            serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            |value| {
                value
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| {
                        ValidationResult::success(Person {
                            name: name.to_string(),
                        })
                    })
                    .unwrap_or_else(|| {
                        ValidationResult::failure(LlmError::ParseError("missing name".to_string()))
                    })
            },
        );

        let result = generate_array(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            schema,
            GenerateObjectOptions::default(),
        )
        .await
        .expect("generate array");

        assert_eq!(
            result.object,
            vec![
                Person {
                    name: "Ada".to_string()
                },
                Person {
                    name: "Grace".to_string()
                }
            ]
        );

        let request = seen_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("request should be captured");
        let ResponseFormat::Json { schema, .. } = request
            .response_format
            .as_ref()
            .expect("response format should be set")
        else {
            panic!("expected schema-backed JSON response format");
        };
        assert_eq!(schema["properties"]["elements"]["type"], "array");
        assert!(
            schema["properties"]["elements"]["items"]
                .get("$schema")
                .is_none()
        );
    }

    #[tokio::test]
    async fn generate_enum_wraps_values_and_extracts_result() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"result\":\"green\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request: seen_request.clone(),
        };

        let result = generate_enum(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            ["red", "green", "blue"],
            GenerateObjectOptions::default(),
        )
        .await
        .expect("generate enum");

        assert_eq!(result.object, "green");

        let request = seen_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("request should be captured");
        let ResponseFormat::Json { schema, .. } = request
            .response_format
            .as_ref()
            .expect("response format should be set")
        else {
            panic!("expected schema-backed JSON response format");
        };
        assert_eq!(
            schema["properties"]["result"]["enum"],
            serde_json::json!(["red", "green", "blue"])
        );
    }

    #[tokio::test]
    async fn generate_choice_allows_schema_labels_and_extracts_result() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"result\":\"green\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request: seen_request.clone(),
        };

        let result = generate_choice(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            ["red", "green", "blue"],
            GenerateObjectOptions::new()
                .with_schema_name("color")
                .with_schema_description("Color choice"),
        )
        .await
        .expect("generate choice");

        assert_eq!(result.object, "green");

        let request = seen_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("request should be captured");
        let ResponseFormat::Json {
            schema,
            name,
            description,
            ..
        } = request
            .response_format
            .as_ref()
            .expect("response format should be set")
        else {
            panic!("expected schema-backed JSON response format");
        };
        assert_eq!(name.as_deref(), Some("color"));
        assert_eq!(description.as_deref(), Some("Color choice"));
        assert_eq!(
            schema["properties"]["result"]["enum"],
            serde_json::json!(["red", "green", "blue"])
        );
    }

    #[tokio::test]
    async fn generate_enum_rejects_schema_labels() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"result\":\"green\"}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request,
        };

        let error = generate_enum(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            ["red", "green"],
            GenerateObjectOptions::new().with_schema_name("color"),
        )
        .await
        .expect_err("schema name should be rejected for enum output");

        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }

    #[tokio::test]
    async fn generate_json_sets_schema_less_response_format_and_parses_value() {
        let seen_request = Arc::new(Mutex::new(None));
        let mut response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"answer\":42}".to_string(),
        ));
        response.request = Some(HttpRequestInfo {
            body: Some(
                serde_json::json!({
                    "model": "provider-model",
                    "response_format": {
                        "type": "json_object"
                    }
                })
                .to_string(),
            ),
        });
        let model = FakeObjectModel {
            response,
            seen_request: seen_request.clone(),
        };

        let result = generate_json(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            GenerateObjectOptions::new()
                .with_schema_name("payload")
                .with_schema_description("Any JSON payload"),
        )
        .await
        .expect("generate json");

        assert_eq!(result.object, serde_json::json!({ "answer": 42 }));

        let request = seen_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("request should be captured");
        let response_format = request
            .response_format
            .as_ref()
            .expect("response format should be set");
        assert_eq!(
            response_format,
            &ResponseFormat::json_object()
                .with_name("payload")
                .with_description("Any JSON payload")
        );
        assert_eq!(
            result
                .request
                .body
                .as_ref()
                .and_then(|body| body.get("response_format")),
            Some(&serde_json::json!({
                "type": "json_object"
            }))
        );
    }

    #[tokio::test]
    async fn generate_json_rejects_strict_schema_option() {
        let seen_request = Arc::new(Mutex::new(None));
        let response = ChatResponse::new(siumai_core::types::MessageContent::Text(
            "{\"answer\":42}".to_string(),
        ));
        let model = FakeObjectModel {
            response,
            seen_request,
        };

        let error = generate_json(
            &model,
            TextRequest::new(vec![siumai_core::types::ChatMessage::user("json").build()]),
            GenerateObjectOptions::new().with_strict(true),
        )
        .await
        .expect_err("strict is schema-only");

        assert!(matches!(error, LlmError::InvalidParameter(_)));
    }
}
