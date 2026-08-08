//! Provider-native OpenAI Responses resource lifecycle operations.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorContext, ErrorKind, ModelId, Warning};
use siumai_protocol_openai::responses::{ResponseWire, decode_response_resource};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget, TransportResponse,
};

use super::mode::OpenAiApiMode;
use super::model::{request_build_error, response_error};
use super::options::{OpenAiReasoning, OpenAiResponseInclude, OpenAiTruncation};
use super::provider::OpenAiRuntime;
use super::tools::OpenAiResponsesTool;

const MAX_RESOURCE_ID_BYTES: usize = 512;

/// A newly created background response plus provider-policy advisories.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiBackgroundResponse {
    resource: ResponseWire,
    warnings: Vec<Warning>,
}

impl OpenAiBackgroundResponse {
    pub(crate) fn new(resource: ResponseWire, warnings: Vec<Warning>) -> Self {
        Self { resource, warnings }
    }

    pub fn resource(&self) -> &ResponseWire {
        &self.resource
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    pub fn into_parts(self) -> (ResponseWire, Vec<Warning>) {
        (self.resource, self.warnings)
    }
}

/// Provider-native operations for stored and background Responses resources.
#[derive(Clone)]
pub struct OpenAiResponsesResource {
    runtime: Arc<OpenAiRuntime>,
}

impl OpenAiResponsesResource {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self { runtime }
    }

    /// Retrieve a stored response without requiring it to be terminal.
    pub async fn retrieve(
        &self,
        response_id: &str,
        options: OpenAiResponsesRetrieveOptions,
        call_options: CallOptions,
    ) -> Result<ResponseWire, Error> {
        validate_resource_id("response_id", response_id)?;
        options.validate()?;
        let pairs = options
            .include
            .into_iter()
            .map(|include| ("include", include.as_str().to_string()))
            .collect::<Vec<_>>();
        let target = target_with_query(&format!("responses/{response_id}"), pairs)?;
        let response = self
            .execute(
                request_plan(Method::GET, target, ReplaySafety::SemanticallyIdempotent)?,
                call_options,
            )
            .await?;
        decode_response_resource(response.body()).map_err(|error| self.contextualize(error))
    }

    /// Delete a stored response. The operation is not replayed after dispatch.
    pub async fn delete(
        &self,
        response_id: &str,
        call_options: CallOptions,
    ) -> Result<OpenAiDeletedResponse, Error> {
        validate_resource_id("response_id", response_id)?;
        let target = target(&format!("responses/{response_id}"))?;
        let response = self
            .execute(
                request_plan(Method::DELETE, target, ReplaySafety::Never)?,
                call_options,
            )
            .await?;
        self.decode_json(
            response.body(),
            "OpenAI returned a malformed deleted response",
        )
    }

    /// Cancel a queued or in-progress background response.
    pub async fn cancel(
        &self,
        response_id: &str,
        call_options: CallOptions,
    ) -> Result<ResponseWire, Error> {
        validate_resource_id("response_id", response_id)?;
        let target = target(&format!("responses/{response_id}/cancel"))?;
        let response = self
            .execute(
                json_request_plan(
                    Method::POST,
                    target,
                    &Value::Object(Default::default()),
                    ReplaySafety::Never,
                )?,
                call_options,
            )
            .await?;
        decode_response_resource(response.body()).map_err(|error| self.contextualize(error))
    }

    /// List the lossless input items associated with a response.
    pub async fn list_input_items(
        &self,
        response_id: &str,
        options: OpenAiResponsesInputItemsOptions,
        call_options: CallOptions,
    ) -> Result<OpenAiResponsesInputItemsPage, Error> {
        validate_resource_id("response_id", response_id)?;
        options.validate()?;
        let mut pairs = Vec::new();
        if let Some(limit) = options.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = options.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        if let Some(after) = options.after {
            pairs.push(("after", after));
        }
        pairs.extend(
            options
                .include
                .into_iter()
                .map(|include| ("include", include.as_str().to_string())),
        );
        let target = target_with_query(&format!("responses/{response_id}/input_items"), pairs)?;
        let response = self
            .execute(
                request_plan(Method::GET, target, ReplaySafety::SemanticallyIdempotent)?,
                call_options,
            )
            .await?;
        self.decode_json(
            response.body(),
            "OpenAI returned malformed Responses input items",
        )
    }

    /// Compact a Responses conversation into provider-native continuation items.
    pub async fn compact(
        &self,
        request: OpenAiResponsesCompactRequest,
        call_options: CallOptions,
    ) -> Result<OpenAiResponsesCompaction, Error> {
        request.validate()?;
        let target = target("responses/compact")?;
        let response = self
            .execute(
                json_request_plan(Method::POST, target, &request, ReplaySafety::Never)?,
                call_options,
            )
            .await?;
        self.decode_json(
            response.body(),
            "OpenAI returned a malformed Responses compaction",
        )
    }

    /// Count the exact input tokens for a provider-native Responses request shape.
    pub async fn count_input_tokens(
        &self,
        request: OpenAiResponsesInputTokenCountRequest,
        call_options: CallOptions,
    ) -> Result<OpenAiResponsesInputTokenCount, Error> {
        request.validate()?;
        let response = self
            .execute(
                json_request_plan(
                    Method::POST,
                    target("responses/input_tokens")?,
                    &request,
                    ReplaySafety::SemanticallyIdempotent,
                )?,
                call_options,
            )
            .await?;
        let decoded: OpenAiResponsesInputTokenCount = self.decode_json(
            response.body(),
            "OpenAI returned a malformed Responses input-token count",
        )?;
        if decoded.object != "response.input_tokens" {
            return Err(self.contextualize(Error::new(
                ErrorKind::Protocol,
                "OpenAI returned an unexpected Responses input-token object",
            )));
        }
        Ok(decoded)
    }

    async fn execute(
        &self,
        plan: RequestPlan,
        call_options: CallOptions,
    ) -> Result<TransportResponse, Error> {
        let response = self
            .runtime
            .transport
            .execute(plan, call_options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(self.contextualize(response_error(OpenAiApiMode::Responses, response)))
        }
    }

    fn decode_json<T: DeserializeOwned>(
        &self,
        body: &[u8],
        message: &'static str,
    ) -> Result<T, Error> {
        serde_json::from_slice(body).map_err(|source| {
            self.contextualize(Error::new(ErrorKind::Protocol, message).with_source(source))
        })
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(
                self.runtime
                    .scope(OpenAiApiMode::Responses)
                    .provider_id()
                    .clone(),
            ),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for OpenAiResponsesResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesResource")
            .field("scope", self.runtime.scope(OpenAiApiMode::Responses))
            .field("transport", &"shared")
            .finish()
    }
}

/// Typed query options for retrieving a stored response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiResponsesRetrieveOptions {
    pub include: Vec<OpenAiResponseInclude>,
}

impl OpenAiResponsesRetrieveOptions {
    pub fn with_include(mut self, include: OpenAiResponseInclude) -> Self {
        if !self.include.contains(&include) {
            self.include.push(include);
        }
        self
    }

    fn validate(&self) -> Result<(), Error> {
        validate_unique_includes(&self.include)
    }
}

/// Input-item ordering for the Responses resource API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiResponsesInputItemsOrder {
    Asc,
    Desc,
}

impl OpenAiResponsesInputItemsOrder {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}

/// Typed query options for listing response input items.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiResponsesInputItemsOptions {
    pub limit: Option<u32>,
    pub order: Option<OpenAiResponsesInputItemsOrder>,
    pub after: Option<String>,
    pub include: Vec<OpenAiResponseInclude>,
}

impl OpenAiResponsesInputItemsOptions {
    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_order(mut self, order: OpenAiResponsesInputItemsOrder) -> Self {
        self.order = Some(order);
        self
    }

    pub fn with_after(mut self, after: impl Into<String>) -> Self {
        self.after = Some(after.into());
        self
    }

    pub fn with_include(mut self, include: OpenAiResponseInclude) -> Self {
        if !self.include.contains(&include) {
            self.include.push(include);
        }
        self
    }

    fn validate(&self) -> Result<(), Error> {
        if self.limit == Some(0) {
            return Err(invalid_input(
                "Responses input item limit must be greater than zero",
            ));
        }
        if let Some(after) = &self.after {
            validate_resource_id("after", after)?;
        }
        validate_unique_includes(&self.include)
    }
}

/// One lossless page returned by `responses/{id}/input_items`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiResponsesInputItemsPage {
    pub object: String,
    #[serde(default)]
    pub data: Vec<Value>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Request body for provider-native Responses compaction.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiResponsesCompactRequest {
    pub model: ModelId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl OpenAiResponsesCompactRequest {
    pub fn new(model: ModelId) -> Self {
        Self {
            model,
            input: None,
            previous_response_id: None,
            instructions: None,
        }
    }

    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    pub fn with_previous_response_id(mut self, response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(response_id.into());
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    fn validate(&self) -> Result<(), Error> {
        if let Some(response_id) = &self.previous_response_id {
            validate_resource_id("previous_response_id", response_id)?;
        }
        if self
            .instructions
            .as_deref()
            .is_some_and(|instructions| instructions.trim().is_empty())
        {
            return Err(invalid_input(
                "Responses compaction instructions cannot be empty",
            ));
        }
        if self.input.is_none() && self.previous_response_id.is_none() {
            return Err(invalid_input(
                "Responses compaction requires input or previous_response_id",
            ));
        }
        Ok(())
    }
}

/// Lossless provider-native compaction resource.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiResponsesCompaction {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub output: Vec<Value>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub usage: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Provider-native request accepted by `POST /responses/input_tokens`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct OpenAiResponsesInputTokenCountRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenAiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAiResponsesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<OpenAiTruncation>,
}

impl OpenAiResponsesInputTokenCountRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(conversation.into());
        self
    }

    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn with_model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    pub fn with_personality(mut self, personality: impl Into<String>) -> Self {
        self.personality = Some(personality.into());
        self
    }

    pub fn with_previous_response_id(mut self, previous_response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(previous_response_id.into());
        self
    }

    pub fn with_reasoning(mut self, reasoning: OpenAiReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_text(mut self, text: Value) -> Self {
        self.text = Some(text);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: Value) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_tool(mut self, tool: OpenAiResponsesTool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_truncation(mut self, truncation: OpenAiTruncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    fn validate(&self) -> Result<(), Error> {
        if self.conversation.is_some() && self.previous_response_id.is_some() {
            return Err(invalid_input(
                "Responses input-token count conversation and previous_response_id are mutually exclusive",
            ));
        }
        if let Some(conversation) = &self.conversation {
            validate_resource_id("conversation", conversation)?;
        }
        if let Some(previous_response_id) = &self.previous_response_id {
            validate_resource_id("previous_response_id", previous_response_id)?;
        }
        if self.input.is_none()
            && self.conversation.is_none()
            && self.previous_response_id.is_none()
        {
            return Err(invalid_input(
                "Responses input-token count requires input, conversation, or previous_response_id",
            ));
        }
        if self
            .input
            .as_ref()
            .is_some_and(|input| !matches!(input, Value::String(_) | Value::Array(_)))
        {
            return Err(invalid_input(
                "Responses input-token count input must be a string or item array",
            ));
        }
        if self
            .instructions
            .as_deref()
            .is_some_and(|instructions| instructions.trim().is_empty())
        {
            return Err(invalid_input(
                "Responses input-token count instructions cannot be empty",
            ));
        }
        if self.personality.as_deref().is_some_and(|personality| {
            personality.trim().is_empty()
                || personality != personality.trim()
                || personality.chars().count() > 64
        }) {
            return Err(invalid_input(
                "Responses input-token count personality must contain 1..=64 trimmed characters",
            ));
        }
        if self.text.as_ref().is_some_and(|text| !text.is_object()) {
            return Err(invalid_input(
                "Responses input-token count text configuration must be a JSON object",
            ));
        }
        if self
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !matches!(choice, Value::String(_) | Value::Object(_)))
        {
            return Err(invalid_input(
                "Responses input-token count tool choice must be a string or JSON object",
            ));
        }
        for tool in &self.tools {
            tool.validate().map_err(|source| {
                invalid_input("Responses input-token count contains an invalid tool")
                    .with_source(source)
            })?;
        }
        Ok(())
    }
}

/// Exact input-token count returned by OpenAI Responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiResponsesInputTokenCount {
    pub object: String,
    pub input_tokens: u64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Typed deletion acknowledgement for a stored response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiDeletedResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn request_plan(
    method: Method,
    target: RequestTarget,
    replay_safety: ReplaySafety,
) -> Result<RequestPlan, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(|source| request_build_error(OpenAiApiMode::Responses, source))?;
    RequestPlan::new(method, target)
        .with_headers(headers)
        .with_replay_safety(replay_safety)
        .map_err(|source| request_build_error(OpenAiApiMode::Responses, source))
}

fn json_request_plan<T: Serialize>(
    method: Method,
    target: RequestTarget,
    body: &T,
    replay_safety: ReplaySafety,
) -> Result<RequestPlan, Error> {
    let plan = request_plan(method, target, replay_safety)?;
    Ok(plan.with_body(
        RequestBody::json(body)
            .map_err(|source| request_build_error(OpenAiApiMode::Responses, source))?,
    ))
}

fn target(value: &str) -> Result<RequestTarget, Error> {
    RequestTarget::new(value)
        .map_err(|source| request_build_error(OpenAiApiMode::Responses, source))
}

fn target_with_query(
    path: &str,
    pairs: Vec<(&'static str, String)>,
) -> Result<RequestTarget, Error> {
    if pairs.is_empty() {
        return target(path);
    }
    let query = pairs
        .into_iter()
        .map(|(name, value)| format!("{name}={}", urlencoding::encode(&value)))
        .collect::<Vec<_>>()
        .join("&");
    target(&format!("{path}?{query}"))
}

fn validate_resource_id(field: &'static str, value: &str) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > MAX_RESOURCE_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(invalid_input(match field {
            "after" => "Responses input item cursor is invalid",
            "previous_response_id" => "Responses previous response ID is invalid",
            _ => "OpenAI response ID is invalid",
        }));
    }
    Ok(())
}

fn validate_unique_includes(includes: &[OpenAiResponseInclude]) -> Result<(), Error> {
    if includes.iter().copied().collect::<BTreeSet<_>>().len() != includes.len() {
        return Err(invalid_input("Responses include values must be unique"));
    }
    Ok(())
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use http::StatusCode;
    use serde_json::json;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::configured::{
        OpenAiCredential, OpenAiProvider, OpenAiReasoningEffort, OpenAiTruncation,
    };

    async fn resource(server: &MockServer) -> OpenAiResponsesResource {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("responses-resource-test").unwrap(),
            ))
            .build()
            .unwrap()
            .responses_resource()
    }

    #[tokio::test]
    async fn retrieve_preserves_background_lifecycle_and_repeated_include_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/responses/resp_123"))
            .and(query_param("include", "reasoning.encrypted_content"))
            .and(query_param("include", "file_search_call.results"))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "id": "resp_123",
                    "created_at": 1,
                    "model": "gpt-5.6-sol",
                    "status": "in_progress",
                    "output": [],
                    "future_field": {"kept": true}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let response = resource(&server)
            .await
            .retrieve(
                "resp_123",
                OpenAiResponsesRetrieveOptions::default()
                    .with_include(OpenAiResponseInclude::ReasoningEncryptedContent)
                    .with_include(OpenAiResponseInclude::FileSearchResults),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(response.status.as_str(), "in_progress");
        assert_eq!(response.extra["future_field"], json!({"kept": true}));
    }

    #[tokio::test]
    async fn cancel_and_compact_use_native_resource_shapes_without_replay() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/resp_123/cancel"))
            .and(body_json(json!({})))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "id": "resp_123",
                    "model": "gpt-5.6-sol",
                    "status": "cancelled",
                    "output": []
                })),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/compact"))
            .and(body_json(json!({
                "model": "gpt-5.6-sol",
                "previous_response_id": "resp_123"
            })))
            .respond_with(
                ResponseTemplate::new(StatusCode::OK.as_u16()).set_body_json(json!({
                    "id": "cmp_123",
                    "object": "response.compaction",
                    "output": [{"type": "reasoning", "encrypted_content": "opaque"}],
                    "created_at": 1,
                    "usage": {"input_tokens": 4}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let resource = resource(&server).await;
        let cancelled = resource
            .cancel("resp_123", CallOptions::default())
            .await
            .unwrap();
        assert_eq!(cancelled.status.as_str(), "cancelled");

        let compacted = resource
            .compact(
                OpenAiResponsesCompactRequest::new(ModelId::new("gpt-5.6-sol").unwrap())
                    .with_previous_response_id("resp_123"),
                CallOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(compacted.id, "cmp_123");
        assert_eq!(compacted.output[0]["encrypted_content"], "opaque");
    }

    #[tokio::test]
    async fn input_token_count_uses_the_native_endpoint_and_preserves_future_fields() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/input_tokens"))
            .and(body_json(json!({
                "model": "gpt-5.6-sol",
                "input": "hello",
                "instructions": "Be concise",
                "parallel_tool_calls": true,
                "personality": "pragmatic",
                "reasoning": {"effort": "low"},
                "tool_choice": "auto",
                "tools": [{"type": "web_search"}],
                "truncation": "disabled"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "response.input_tokens",
                "input_tokens": 17,
                "future_detail": {"cached": 3}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let counted = resource(&server)
            .await
            .count_input_tokens(
                OpenAiResponsesInputTokenCountRequest::new()
                    .with_model(ModelId::new("gpt-5.6-sol").unwrap())
                    .with_input(json!("hello"))
                    .with_instructions("Be concise")
                    .with_parallel_tool_calls(true)
                    .with_personality("pragmatic")
                    .with_reasoning(
                        OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::Low),
                    )
                    .with_tool_choice(json!("auto"))
                    .with_truncation(OpenAiTruncation::Disabled)
                    .with_tool(OpenAiResponsesTool::web_search()),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(counted.input_tokens, 17);
        assert_eq!(counted.extra["future_detail"], json!({"cached": 3}));
    }

    #[tokio::test]
    async fn input_token_count_sanitizes_provider_error_bodies() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses/input_tokens"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "input-token-secret"
                }
            })))
            .mount(&server)
            .await;

        let error = resource(&server)
            .await
            .count_input_tokens(
                OpenAiResponsesInputTokenCountRequest::new()
                    .with_model(ModelId::new("gpt-5.6-sol").unwrap())
                    .with_input(json!("hello")),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        assert!(!format!("{error:?}").contains("input-token-secret"));
        assert!(!error.to_string().contains("input-token-secret"));
    }

    #[test]
    fn resource_ids_cannot_escape_the_provider_endpoint() {
        assert!(validate_resource_id("response_id", "resp_123").is_ok());
        for invalid in ["", "../models", "resp/123", "resp%2f123", "resp?admin=true"] {
            assert!(validate_resource_id("response_id", invalid).is_err());
        }
    }
}
