//! DeepSeek language profiles and bounded compatible codec policy.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use chrono::NaiveDate;
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, ContentPart, Error, ErrorKind, InvalidId,
    LanguageRequest, LanguageResponse, LanguageStreamDecoder, LanguageStreamEvent, ModelCatalog,
    ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId,
    ProfileError, ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile,
    ProviderScope, ReplayDomain, ResponseDiagnostics, StreamTerminal, SupportScope,
    TypedProviderOptions, Usage, UsageValue, VerificationDate, VerificationEvidence,
    VerifiedFidelity, VerifiedSupportClaim, Warning,
};
use siumai_openai_compatible::extension::{
    ChatCodecPolicy, CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall,
    ResponsesCodecPolicy,
};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, ChatCompletionsStreamDecoder,
    DialectError, PROTOCOL_ID as CHAT_PROTOCOL_ID, WireFieldName,
    decode_response as decode_chat_response,
};
use siumai_protocol_openai::responses::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL,
};
use siumai_transport::{EndpointConfig, RequestHeaders, ResponseHeaders};
use thiserror::Error as ThisError;

use crate::models::{DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO, is_known_chat, is_known_responses};
use crate::options::{DeepSeekChatOptions, DeepSeekResponsesOptions};

pub(crate) const PROVIDER_ID: &str = "deepseek";
pub(crate) const PLATFORM_ID: &str = "deepseek-api";
pub(crate) const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";
pub(crate) const CHAT_SOURCE: &str = "https://api-docs.deepseek.com/api/create-chat-completion";
pub(crate) const RESPONSES_SOURCE: &str = "https://api-docs.deepseek.com/guides/responses_api";
pub(crate) const VERIFIED_ON: &str = "2026-08-05";

pub(crate) fn profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, DeepSeekProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let reasoning = WireFieldName::new("reasoning_content")?;
    let cache_read = WireFieldName::new("prompt_cache_hit_tokens")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning)
        .with_cache_read_tokens_field(cache_read);

    let profile = if verified_endpoint {
        verified_profile(provider, endpoint, dialect)?.with_replay_domain(replay_domain)?
    } else {
        OpenAiCompatibleProfile::custom_chat_and_responses(
            provider,
            endpoint,
            replay_domain,
            dialect,
        )?
    };

    Ok(profile
        .with_chat_codec_policy(Arc::new(DeepSeekChatCodecPolicy))
        .with_responses_codec_policy(Arc::new(DeepSeekResponsesCodecPolicy)))
}

fn verified_profile(
    provider: ProviderId,
    endpoint: EndpointConfig,
    dialect: ChatCompletionsDialect,
) -> Result<OpenAiCompatibleProfile, DeepSeekProfileError> {
    let platform = PlatformId::new(PLATFORM_ID)?;
    let chat_scope = SupportScope::new(
        provider.clone(),
        platform.clone(),
        ModelFamily::Language,
        ProtocolId::new(CHAT_PROTOCOL_ID)?,
        ApiModeId::new(CHAT_API_MODE_ID)?,
    );
    let responses_scope = SupportScope::new(
        provider,
        platform,
        ModelFamily::Language,
        ProtocolId::new(OPENAI_RESPONSES_PROTOCOL)?,
        ApiModeId::new(RESPONSES_API_MODE_ID)?,
    );
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(|_| DeepSeekProfileError::InvalidVerificationDate)?,
    );
    let chat_evidence = VerificationEvidence::new(
        OfficialSource::new(CHAT_SOURCE)?,
        verified_at,
        ProtocolContractId::new("deepseek-v4-openai-chat-2026-08")?,
    );
    let responses_evidence = VerificationEvidence::new(
        OfficialSource::new(RESPONSES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("deepseek-v4-openai-responses-2026-08")?,
    );
    let catalog = ModelCatalog::new([
        model_profile(DEEPSEEK_V4_FLASH, &chat_scope, &chat_evidence)?,
        model_profile(DEEPSEEK_V4_PRO, &chat_scope, &chat_evidence)?,
        model_profile(DEEPSEEK_V4_FLASH, &responses_scope, &responses_evidence)?,
    ])?;
    let provider_profile = ProviderProfile::verified(
        ProfileId::new(PROVIDER_ID)?,
        vec![
            VerifiedSupportClaim::new(
                chat_scope,
                VerifiedFidelity::Compatible,
                ApiStability::Stable,
                chat_evidence,
            ),
            VerifiedSupportClaim::new(
                responses_scope,
                VerifiedFidelity::Compatible,
                ApiStability::Stable,
                responses_evidence,
            ),
        ],
        catalog,
    )?;
    Ok(OpenAiCompatibleProfile::verified_chat_and_responses(
        provider_profile,
        endpoint,
        dialect,
    )?)
}

fn model_profile(
    model: &str,
    scope: &SupportScope,
    evidence: &VerificationEvidence,
) -> Result<ModelProfile, DeepSeekProfileError> {
    Ok(ModelProfile::new(
        ModelId::new(model)?,
        scope.clone(),
        [ModelOperation::Generate, ModelOperation::Stream],
        ModelLifecycle::Active,
        evidence.clone(),
    )?)
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum DeepSeekProfileError {
    #[error("invalid DeepSeek identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid DeepSeek support profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid DeepSeek model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid DeepSeek Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid DeepSeek compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("DeepSeek verification date is invalid")]
    InvalidVerificationDate,
}

#[derive(Debug, Default)]
struct DeepSeekChatCodecPolicy;

impl ChatCodecPolicy for DeepSeekChatCodecPolicy {
    fn name(&self) -> &'static str {
        "deepseek-v4-chat-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        mut dialect: ChatCompletionsDialect,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        reject_legacy_options(&extra)?;
        reject_media(&request)?;
        let options = parse_chat_options(&extra)?;
        if let Some(strict) = options.strict_tools {
            if strict {
                if !is_known_chat(model.as_str()) {
                    return Err(invalid(
                        "DeepSeek strict tools are not verified for this model identifier",
                    ));
                }
                if request.tools.is_empty() {
                    return Err(invalid(
                        "DeepSeek strict tool mode requires at least one function tool",
                    ));
                }
                validate_strict_tool_schemas(&request)?;
            }
            dialect = dialect.with_function_tool_strict(strict);
        }
        extra.remove("strict_tools");

        let mut warnings = Vec::new();
        if request.structured_output.is_some() {
            warnings.push(Warning::provider(
                "structured_output_fallback",
                "DeepSeek Chat uses json_object plus an injected schema instruction",
            ));
        }
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: RequestHeaders::new(),
            prompt_cache_breakpoints: Vec::new(),
            warnings,
        })
    }

    fn encode_request(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        prepared: &PreparedChatCall,
        stream: bool,
    ) -> Result<Value, Error> {
        let mut body = prepared.encode(scope, model, stream)?;
        let Some(output) = prepared.request.structured_output.as_ref() else {
            return Ok(body);
        };
        {
            let body = body.as_object_mut().ok_or_else(|| {
                Error::new(
                    ErrorKind::Internal,
                    "DeepSeek Chat encoder returned a non-object request",
                )
            })?;
            body.insert(
                "response_format".to_string(),
                serde_json::json!({"type": "json_object"}),
            );
            let schema = serde_json::to_string(&output.schema).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "DeepSeek structured-output schema could not be serialized",
                )
                .with_source(source)
            })?;
            let messages = body
                .get_mut("messages")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Internal,
                        "DeepSeek Chat encoder omitted the messages array",
                    )
                })?;
            messages.insert(
                0,
                serde_json::json!({
                    "role": "system",
                    "content": format!(
                        "Return JSON that conforms to the following schema: {schema}"
                    )
                }),
            );
        }
        Ok(body)
    }

    fn decode_response(
        &self,
        scope: &ProviderScope,
        model: &ModelId,
        _headers: &ResponseHeaders,
        body: &[u8],
        dialect: &ChatCompletionsDialect,
    ) -> Result<LanguageResponse, Error> {
        let response = decode_chat_response(scope, model, body, dialect)?;
        augment_response(&response, &DeepSeekUsageDetails::from_bytes(body))
    }

    fn stream_decoder(
        &self,
        scope: ProviderScope,
        model: ModelId,
        dialect: ChatCompletionsDialect,
    ) -> CompatibleStreamDecoder {
        Box::new(DeepSeekChatStreamDecoder {
            inner: ChatCompletionsStreamDecoder::new(scope, model, dialect),
            usage: DeepSeekUsageDetails::default(),
        })
    }
}

#[derive(Debug, Default)]
struct DeepSeekResponsesCodecPolicy;

impl ResponsesCodecPolicy for DeepSeekResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "deepseek-v4-responses-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        reject_media(&request)?;
        reject_unsupported_responses_fields(&extra)?;
        let options = parse_responses_options(&extra)?;
        if is_known_chat(model.as_str()) && !is_known_responses(model.as_str()) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "DeepSeek V4 Pro is not currently supported by the Responses API",
            ));
        }

        extra.remove("reasoning_effort");
        extra.remove("native_tools");
        if let Some(effort) = options.reasoning_effort {
            let effort = serde_json::to_value(effort).map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "DeepSeek reasoning effort could not be serialized",
                )
                .with_source(source)
            })?;
            extra.insert(
                "reasoning".to_string(),
                serde_json::json!({"effort": effort}),
            );
        }

        let native_tools = options
            .native_tools
            .into_iter()
            .map(|tool| tool.as_wire_value())
            .collect();

        Ok(PreparedResponsesCall {
            request,
            extra,
            headers: RequestHeaders::new(),
            prompt_cache_breakpoints: Vec::new(),
            native_tools,
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }
}

fn parse_chat_options(extra: &BTreeMap<String, Value>) -> Result<DeepSeekChatOptions, Error> {
    let options: DeepSeekChatOptions = serde_json::from_value(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "DeepSeek Chat options have an invalid wire shape",
        )
        .with_source(source)
    })?;
    options.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "DeepSeek Chat options are invalid").with_source(source)
    })?;
    Ok(options)
}

fn parse_responses_options(
    extra: &BTreeMap<String, Value>,
) -> Result<DeepSeekResponsesOptions, Error> {
    let options: DeepSeekResponsesOptions = serde_json::from_value(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "DeepSeek Responses options have an invalid wire shape",
        )
        .with_source(source)
    })?;
    options.validate().map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "DeepSeek Responses options are invalid",
        )
        .with_source(source)
    })?;
    Ok(options)
}

fn reject_unsupported_responses_fields(extra: &BTreeMap<String, Value>) -> Result<(), Error> {
    for field in [
        "previous_response_id",
        "conversation",
        "store",
        "background",
        "metadata",
        "include",
        "prompt",
        "truncation",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "context_management",
        "stream_options",
        "max_tool_calls",
    ] {
        if extra.contains_key(field) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "DeepSeek Responses received an unsupported option",
            )
            .with_source(siumai_core::ProviderOptionError::Rejected {
                path: field.to_string(),
                reason: "the current DeepSeek Responses API does not accept this field".to_string(),
            }));
        }
    }
    Ok(())
}

fn reject_legacy_options(extra: &BTreeMap<String, Value>) -> Result<(), Error> {
    for field in [
        "enableReasoning",
        "enable_reasoning",
        "reasoningBudget",
        "reasoning_budget",
    ] {
        if extra.contains_key(field) {
            return Err(invalid(
                "legacy DeepSeek reasoning options were removed; use thinking and reasoning_effort",
            ));
        }
    }
    Ok(())
}

fn reject_media(request: &LanguageRequest) -> Result<(), Error> {
    if request.messages.iter().any(|message| {
        message
            .content()
            .iter()
            .any(|part| matches!(part.content(), ContentPart::Media(_)))
    }) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "DeepSeek V4 language endpoints do not declare media input support",
        ));
    }
    Ok(())
}

fn validate_strict_tool_schemas(request: &LanguageRequest) -> Result<(), Error> {
    for tool in &request.tools {
        let schema = tool.input_schema().as_object().ok_or_else(|| {
            invalid("DeepSeek strict tools require a top-level object JSON Schema")
        })?;
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid("DeepSeek strict tools require schema type=object"));
        }
        if schema.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
            return Err(invalid(
                "DeepSeek strict tools require additionalProperties=false",
            ));
        }
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("DeepSeek strict tools require an object properties map"))?;
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if properties
            .keys()
            .any(|name| !required.contains(name.as_str()))
        {
            return Err(invalid(
                "DeepSeek strict tools require every declared property to be required",
            ));
        }
    }
    Ok(())
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Debug, Clone, Default)]
struct DeepSeekUsageDetails {
    prompt_cache_hit_tokens: Option<u64>,
    prompt_cache_miss_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
}

impl DeepSeekUsageDetails {
    fn from_bytes(bytes: &[u8]) -> Self {
        serde_json::from_slice::<Value>(bytes)
            .ok()
            .map(|value| Self::from_value(&value))
            .unwrap_or_default()
    }

    fn from_frame(frame: &str) -> Self {
        serde_json::from_str::<Value>(frame)
            .ok()
            .map(|value| Self::from_value(&value))
            .unwrap_or_default()
    }

    fn from_value(value: &Value) -> Self {
        let Some(usage) = value.get("usage") else {
            return Self::default();
        };
        Self {
            prompt_cache_hit_tokens: usage.get("prompt_cache_hit_tokens").and_then(Value::as_u64),
            prompt_cache_miss_tokens: usage
                .get("prompt_cache_miss_tokens")
                .and_then(Value::as_u64),
            reasoning_tokens: usage
                .get("reasoning_tokens")
                .and_then(Value::as_u64)
                .or_else(|| {
                    usage
                        .get("completion_tokens_details")
                        .and_then(|details| details.get("reasoning_tokens"))
                        .and_then(Value::as_u64)
                }),
        }
    }

    fn merge(&mut self, newer: Self) {
        if newer.prompt_cache_hit_tokens.is_some() {
            self.prompt_cache_hit_tokens = newer.prompt_cache_hit_tokens;
        }
        if newer.prompt_cache_miss_tokens.is_some() {
            self.prompt_cache_miss_tokens = newer.prompt_cache_miss_tokens;
        }
        if newer.reasoning_tokens.is_some() {
            self.reasoning_tokens = newer.reasoning_tokens;
        }
    }

    fn metadata(&self) -> Option<Value> {
        let mut values = Map::new();
        if let Some(value) = self.prompt_cache_hit_tokens {
            values.insert("prompt_cache_hit_tokens".to_string(), Value::from(value));
        }
        if let Some(value) = self.prompt_cache_miss_tokens {
            values.insert("prompt_cache_miss_tokens".to_string(), Value::from(value));
        }
        (!values.is_empty()).then_some(Value::Object(values))
    }
}

fn augment_usage(usage: &Usage, details: &DeepSeekUsageDetails) -> Usage {
    let mut usage = usage.clone();
    if let Some(value) = details.prompt_cache_hit_tokens {
        usage.cache_read_tokens = UsageValue::Known(value);
    }
    if let Some(value) = details.reasoning_tokens {
        usage.reasoning_tokens = UsageValue::Known(value);
    }
    if let Some(metadata) = details.metadata() {
        usage.provider.insert(PROVIDER_ID.to_string(), metadata);
    }
    usage
}

fn augment_response(
    response: &LanguageResponse,
    details: &DeepSeekUsageDetails,
) -> Result<LanguageResponse, Error> {
    let mut metadata = response.provider_metadata().clone();
    if let Some(provider) = details.metadata() {
        metadata.insert(PROVIDER_ID.to_string(), provider);
    }
    let mut rebuilt = LanguageResponse::new(
        response.status().clone(),
        response.content().to_vec(),
        response.finish_reason().clone(),
        augment_usage(response.usage(), details),
    )
    .map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "DeepSeek usage metadata produced an inconsistent response",
        )
        .with_source(source)
    })?
    .with_warnings(response.warnings().to_vec())
    .with_provider_metadata(metadata);
    if let Some(id) = response.id() {
        rebuilt = rebuilt.with_id(id);
    }
    if let Some(model) = response.model() {
        rebuilt = rebuilt.with_model(model.clone());
    }
    Ok(rebuilt)
}

struct DeepSeekChatStreamDecoder {
    inner: ChatCompletionsStreamDecoder,
    usage: DeepSeekUsageDetails,
}

impl LanguageStreamDecoder for DeepSeekChatStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.inner.set_response_diagnostics(diagnostics);
    }

    fn decode(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.usage.merge(DeepSeekUsageDetails::from_frame(frame));
        let mut events = self.inner.decode(frame)?;
        augment_events(&mut events, &self.usage)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut events = self.inner.finish()?;
        augment_events(&mut events, &self.usage)?;
        Ok(events)
    }

    fn terminal_seen(&self) -> bool {
        self.inner.terminal_seen()
    }
}

fn augment_events(
    events: &mut [LanguageStreamEvent],
    details: &DeepSeekUsageDetails,
) -> Result<(), Error> {
    for event in events {
        match event {
            LanguageStreamEvent::Usage(usage) => {
                *usage = augment_usage(usage, details);
            }
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => {
                **response = augment_response(response.as_ref(), details)?;
            }
            LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                response: Some(response),
                ..
            })
            | LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                response: Some(response),
                ..
            }) => {
                **response = augment_response(response.as_ref(), details)?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{
        Message, MessageRole, StructuredOutputSpec, ToolCall, ToolOutcome, ToolResult, ToolSpec,
    };

    fn model(value: &str) -> ModelId {
        ModelId::new(value).expect("test model id")
    }

    fn chat_scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new(PROVIDER_ID).expect("provider"))
            .with_protocol(ProtocolId::new(CHAT_PROTOCOL_ID).expect("protocol"))
            .with_api_mode(ApiModeId::new(CHAT_API_MODE_ID).expect("api mode"))
            .with_replay_domain(ReplayDomain::custom(
                siumai_core::ReplayDomainId::new("test-endpoint").expect("replay domain"),
            ))
    }

    #[test]
    fn profile_replays_reasoning_for_every_assistant_tool_turn() {
        let policy = DeepSeekChatCodecPolicy;
        let tool = ToolSpec::new(
            "lookup",
            None,
            serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
                "additionalProperties": false
            }),
        )
        .expect("tool");
        let request = LanguageRequest {
            messages: vec![
                Message::text(MessageRole::User, "first"),
                Message::new(
                    MessageRole::Assistant,
                    [
                        ContentPart::Reasoning {
                            text: "reason one".to_string(),
                        },
                        ContentPart::ToolCall(
                            ToolCall::local("call-1", "lookup", serde_json::json!({"q": "one"}))
                                .unwrap(),
                        ),
                    ],
                ),
                Message::new(
                    MessageRole::Tool,
                    [ContentPart::ToolResult(ToolResult {
                        call_id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        outcome: ToolOutcome::Success {
                            value: serde_json::json!({"answer": 1}),
                        },
                    })],
                ),
                Message::text(MessageRole::User, "second"),
                Message::new(
                    MessageRole::Assistant,
                    [
                        ContentPart::Reasoning {
                            text: "reason two".to_string(),
                        },
                        ContentPart::ToolCall(
                            ToolCall::local("call-2", "lookup", serde_json::json!({"q": "two"}))
                                .unwrap(),
                        ),
                    ],
                ),
            ],
            generation: Default::default(),
            tools: vec![tool],
            tool_choice: None,
            structured_output: None,
        };
        let dialect = ChatCompletionsDialect::generic()
            .with_reasoning_input_field(WireFieldName::new("reasoning_content").expect("field"))
            .with_reasoning_output_field(WireFieldName::new("reasoning_content").expect("field"));
        let prepared = policy
            .prepare(&model(DEEPSEEK_V4_FLASH), request, dialect, BTreeMap::new())
            .expect("prepare");
        let body = policy
            .encode_request(&chat_scope(), &model(DEEPSEEK_V4_FLASH), &prepared, false)
            .expect("encode");

        assert_eq!(body["messages"][1]["reasoning_content"], "reason one");
        assert_eq!(body["messages"][4]["reasoning_content"], "reason two");
    }

    #[test]
    fn structured_output_uses_json_object_and_schema_instruction() {
        let policy = DeepSeekChatCodecPolicy;
        let request = LanguageRequest {
            messages: vec![Message::text(MessageRole::User, "answer")],
            generation: Default::default(),
            tools: Vec::new(),
            tool_choice: None,
            structured_output: Some(StructuredOutputSpec {
                name: "answer".to_string(),
                description: None,
                schema: serde_json::json!({"type": "object"}),
                strict: true,
            }),
        };
        let prepared = policy
            .prepare(
                &model(DEEPSEEK_V4_FLASH),
                request,
                ChatCompletionsDialect::generic(),
                BTreeMap::new(),
            )
            .expect("prepare");
        let body = policy
            .encode_request(&chat_scope(), &model(DEEPSEEK_V4_FLASH), &prepared, false)
            .expect("encode");

        assert_eq!(
            body["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .is_some_and(|value| value.contains("{\"type\":\"object\"}"))
        );
        assert_eq!(prepared.warnings.len(), 1);
    }

    #[test]
    fn strict_tools_validate_the_documented_top_level_schema_contract() {
        let policy = DeepSeekChatCodecPolicy;
        let invalid_tool = ToolSpec::new(
            "lookup",
            None,
            serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }),
        )
        .expect("tool");
        let request = LanguageRequest {
            messages: vec![Message::text(MessageRole::User, "lookup")],
            generation: Default::default(),
            tools: vec![invalid_tool],
            tool_choice: None,
            structured_output: None,
        };
        let extra = serde_json::from_value::<BTreeMap<String, Value>>(serde_json::json!({
            "strict_tools": true
        }))
        .expect("options");

        assert!(
            policy
                .prepare(
                    &model(DEEPSEEK_V4_FLASH),
                    request,
                    ChatCompletionsDialect::generic(),
                    extra,
                )
                .is_err()
        );
    }

    #[test]
    fn cache_miss_metadata_and_known_zero_are_preserved() {
        let response = decode_chat_response(
            &ProviderScope::new(ProviderId::new(PROVIDER_ID).expect("provider"))
                .with_protocol(ProtocolId::new(CHAT_PROTOCOL_ID).expect("protocol"))
                .with_api_mode(ApiModeId::new(CHAT_API_MODE_ID).expect("api mode"))
                .with_replay_domain(ReplayDomain::custom(
                    siumai_core::ReplayDomainId::new("test-endpoint").expect("replay domain"),
                )),
            &model(DEEPSEEK_V4_FLASH),
            br#"{"id":"chat-1","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":0,"reasoning_tokens":0}}"#,
            &ChatCompletionsDialect::generic()
                .with_cache_read_tokens_field(WireFieldName::new("prompt_cache_hit_tokens").expect("field")),
        )
        .expect("decode");
        let augmented = augment_response(
            &response,
            &DeepSeekUsageDetails {
                prompt_cache_hit_tokens: Some(0),
                prompt_cache_miss_tokens: Some(0),
                reasoning_tokens: Some(0),
            },
        )
        .expect("augment");

        assert_eq!(augmented.usage().cache_read_tokens, UsageValue::Known(0));
        assert_eq!(augmented.usage().reasoning_tokens, UsageValue::Known(0));
        assert_eq!(
            augmented.provider_metadata()[PROVIDER_ID]["prompt_cache_miss_tokens"],
            0
        );
    }
}
