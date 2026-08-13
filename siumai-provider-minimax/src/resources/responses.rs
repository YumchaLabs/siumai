use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::{CallOptions, Error, ErrorKind, LanguageRequest, ModelId, ProviderScope};
use siumai_transport::{ReplaySafety, RequestBody};

use crate::language::encode_responses_input_token_request;
use crate::options::{MinimaxReasoningEffort, MinimaxResponsesReasoning};

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};

const INPUT_TOKENS_TARGET: &str = "responses/input_tokens";

/// Typed request accepted by MiniMax's provider-native Responses input-token endpoint.
#[derive(Clone)]
pub struct MinimaxResponsesInputTokenRequest {
    model: ModelId,
    input: MinimaxResponsesInput,
    reasoning: Option<MinimaxResponsesReasoning>,
}

/// Provider-owned input representations supported by MiniMax Responses token counting.
#[derive(Clone)]
#[non_exhaustive]
pub enum MinimaxResponsesInput {
    /// The provider's native string shorthand.
    Text(String),
    /// Role-safe message and tool input projected through Siumai's Responses codec.
    LanguageRequest(Box<LanguageRequest>),
}

impl MinimaxResponsesInputTokenRequest {
    pub fn from_text(model: impl Into<String>, text: impl Into<String>) -> Result<Self, Error> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax Responses input-token text must not be empty",
            ));
        }
        Self::new(model, MinimaxResponsesInput::Text(text))
    }

    pub fn from_language_request(
        model: impl Into<String>,
        request: LanguageRequest,
    ) -> Result<Self, Error> {
        Self::new(
            model,
            MinimaxResponsesInput::LanguageRequest(Box::new(request)),
        )
    }

    fn new(model: impl Into<String>, input: MinimaxResponsesInput) -> Result<Self, Error> {
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax Responses input-token model identifier is invalid",
            )
            .with_source(source)
        })?;
        Ok(Self {
            model,
            input,
            reasoning: None,
        })
    }

    pub const fn with_reasoning_effort(mut self, effort: MinimaxReasoningEffort) -> Self {
        self.reasoning = Some(MinimaxResponsesReasoning::new(effort));
        self
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn input(&self) -> &MinimaxResponsesInput {
        &self.input
    }

    fn body(&self, scope: &ProviderScope) -> Result<RequestBody, Error> {
        let body = match &self.input {
            MinimaxResponsesInput::Text(text) => {
                let mut body = Map::from_iter([
                    ("model".to_string(), Value::String(self.model.to_string())),
                    ("input".to_string(), Value::String(text.clone())),
                ]);
                if let Some(reasoning) = self.reasoning {
                    body.insert(
                        "reasoning".to_string(),
                        serde_json::to_value(reasoning).map_err(|source| {
                            Error::new(
                                ErrorKind::InvalidInput,
                                "MiniMax input-token reasoning options are invalid",
                            )
                            .with_source(source)
                        })?,
                    );
                }
                Value::Object(body)
            }
            MinimaxResponsesInput::LanguageRequest(request) => {
                encode_responses_input_token_request(scope, &self.model, request, self.reasoning)?
            }
        };
        RequestBody::json(&body).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax Responses input-token request is invalid",
            )
            .with_source(source)
        })
    }
}

impl fmt::Debug for MinimaxResponsesInputTokenRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("MinimaxResponsesInputTokenRequest");
        debug.field("model", &self.model);
        match &self.input {
            MinimaxResponsesInput::Text(text) => {
                debug
                    .field("input_kind", &"text")
                    .field("text_bytes", &text.len());
            }
            MinimaxResponsesInput::LanguageRequest(request) => {
                debug
                    .field("input_kind", &"language-request")
                    .field("message_count", &request.messages.len())
                    .field("tool_count", &request.tools.len());
            }
        }
        debug
            .field("has_reasoning", &self.reasoning.is_some())
            .finish()
    }
}

/// Exact input-token count returned by MiniMax Responses.
#[derive(Clone, PartialEq)]
pub struct MinimaxResponsesInputTokenCount {
    input_tokens: u64,
    extra: BTreeMap<String, Value>,
}

impl MinimaxResponsesInputTokenCount {
    pub const fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxResponsesInputTokenCount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxResponsesInputTokenCount")
            .field("input_tokens", &self.input_tokens)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Provider-native MiniMax Responses resource operations.
#[derive(Clone)]
pub struct MinimaxResponses {
    runtime: Arc<NativeRuntime>,
    scope: Arc<ProviderScope>,
}

impl MinimaxResponses {
    pub(crate) fn new(runtime: Arc<NativeRuntime>, scope: Arc<ProviderScope>) -> Self {
        Self { runtime, scope }
    }

    pub async fn count_input_tokens(
        &self,
        request: MinimaxResponsesInputTokenRequest,
    ) -> Result<MinimaxResponsesInputTokenCount, Error> {
        self.count_input_tokens_with_options(request, CallOptions::default())
            .await
    }

    pub async fn count_input_tokens_with_options(
        &self,
        request: MinimaxResponsesInputTokenRequest,
        options: CallOptions,
    ) -> Result<MinimaxResponsesInputTokenCount, Error> {
        let body = request.body(&self.scope)?;
        let response: InputTokenResponseEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(INPUT_TOKENS_TARGET)?,
            body,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        if response.object != "response.input_tokens" {
            return Err(Error::new(
                ErrorKind::Protocol,
                "MiniMax returned an unexpected Responses input-token object",
            ));
        }
        Ok(MinimaxResponsesInputTokenCount {
            input_tokens: response.input_tokens,
            extra: response.extra,
        })
    }
}

impl fmt::Debug for MinimaxResponses {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxResponses")
            .field("scope", &self.scope)
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct InputTokenResponseEnvelope {
    object: String,
    input_tokens: u64,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for InputTokenResponseEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = false;

    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}
