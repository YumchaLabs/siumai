//! Groq Chat Completions and Responses profiles with provider-owned codec policy.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, Utc};
use http::header::{HeaderName, HeaderValue};
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, Error, ErrorKind, InvalidId, LanguageRequest,
    LanguageResponse, LanguageStreamDecoder, LanguageStreamEvent, ModelCatalog, ModelFamily,
    ModelId, ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId,
    ProfileError, ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile,
    ProviderScope, PublicDiagnosticText, ReplayDomain, ResponseDiagnostics, StreamTerminal,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
    Warning, WarningKind,
};
use siumai_openai_compatible::extension::v1::{
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
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL, ResponsesStreamDecoder,
    ResponsesWireDialect, decode_response as decode_responses_response,
};
use siumai_transport::{EndpointConfig, RequestHeaders, ResponseHeaders};
use thiserror::Error as ThisError;

use crate::models;
use crate::options::{GroqLanguageOptions, GroqReasoningFormat, GroqResponsesOptions};

pub const PROVIDER_ID: &str = "groq";
pub const PLATFORM_ID: &str = "groq-cloud";
pub const DEFAULT_BASE_URL: &str = "https://api.groq.com/openai/v1";
pub const CHAT_SOURCE: &str = "https://console.groq.com/docs/openai";
pub const RESPONSES_SOURCE: &str = "https://console.groq.com/docs/responses-api";
pub const MODEL_CATALOG_SOURCE: &str = "https://console.groq.com/docs/models";
pub const DEPRECATIONS_SOURCE: &str = "https://console.groq.com/docs/deprecations";
pub const VERIFIED_ON: &str = "2026-08-05";

const BROWSER_SEARCH_MARKER: &str = "__siumai_groq_browser_search";
const STRUCTURED_OUTPUTS_MARKER: &str = "__siumai_groq_structured_outputs";
const STRICT_JSON_SCHEMA_MARKER: &str = "__siumai_groq_strict_json_schema";

pub(crate) fn profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, GroqProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let dialect = groq_dialect()?;
    let profile = if verified_endpoint {
        verified_profile(provider, endpoint, replay_domain, dialect)?
    } else {
        OpenAiCompatibleProfile::custom_chat_and_responses(
            provider,
            endpoint,
            replay_domain,
            dialect,
        )?
    };
    Ok(profile
        .with_chat_codec_policy(Arc::new(GroqChatCodecPolicy))
        .with_responses_codec_policy(Arc::new(GroqResponsesCodecPolicy))
        .with_responses_wire_dialect(ResponsesWireDialect::compatible()))
}

fn verified_profile(
    provider: ProviderId,
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    dialect: ChatCompletionsDialect,
) -> Result<OpenAiCompatibleProfile, GroqProfileError> {
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
            .map_err(|_| GroqProfileError::InvalidVerificationDate)?,
    );
    let chat_evidence = VerificationEvidence::new(
        OfficialSource::new(CHAT_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-openai-chat-2026-08")?,
    );
    let responses_evidence = VerificationEvidence::new(
        OfficialSource::new(RESPONSES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-openai-responses-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        OfficialSource::new(MODEL_CATALOG_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-model-catalog-2026-08")?,
    );
    let deprecation_evidence = VerificationEvidence::new(
        OfficialSource::new(DEPRECATIONS_SOURCE)?,
        verified_at,
        ProtocolContractId::new("groq-model-deprecations-2026-08")?,
    );
    let catalog = model_catalog(
        &chat_scope,
        &responses_scope,
        &model_evidence,
        &deprecation_evidence,
    )?;
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
    Ok(
        OpenAiCompatibleProfile::verified_chat_and_responses(provider_profile, endpoint, dialect)?
            .with_replay_domain(replay_domain)?,
    )
}

fn model_profile(
    model: &str,
    scope: &SupportScope,
    evidence: &VerificationEvidence,
    lifecycle: ModelLifecycle,
) -> Result<ModelProfile, GroqProfileError> {
    Ok(ModelProfile::new(
        ModelId::new(model)?,
        scope.clone(),
        [ModelOperation::Generate, ModelOperation::Stream],
        lifecycle,
        evidence.clone(),
    )?)
}

fn model_catalog(
    chat_scope: &SupportScope,
    responses_scope: &SupportScope,
    model_evidence: &VerificationEvidence,
    deprecation_evidence: &VerificationEvidence,
) -> Result<ModelCatalog, GroqProfileError> {
    let replacement = |model: &str| ModelId::new(model).map(Some);
    let mut entries = Vec::new();
    for scope in [chat_scope, responses_scope] {
        entries.extend([
            model_profile(
                models::language::LLAMA_3_1_8B_INSTANT,
                scope,
                deprecation_evidence,
                ModelLifecycle::Deprecated {
                    replacement: replacement(models::language::GPT_OSS_20B)?,
                },
            )?,
            model_profile(
                models::language::LLAMA_3_3_70B_VERSATILE,
                scope,
                deprecation_evidence,
                ModelLifecycle::Deprecated {
                    replacement: replacement(models::language::GPT_OSS_120B)?,
                },
            )?,
            model_profile(
                models::language::GPT_OSS_20B,
                scope,
                model_evidence,
                ModelLifecycle::Active,
            )?,
            model_profile(
                models::language::GPT_OSS_120B,
                scope,
                model_evidence,
                ModelLifecycle::Active,
            )?,
            model_profile(
                models::language::COMPOUND,
                scope,
                model_evidence,
                ModelLifecycle::Active,
            )?,
            model_profile(
                models::language::COMPOUND_MINI,
                scope,
                model_evidence,
                ModelLifecycle::Active,
            )?,
            model_profile(
                models::language::QWEN3_32B,
                scope,
                deprecation_evidence,
                ModelLifecycle::Retired {
                    replacement: replacement(models::language::GPT_OSS_120B)?,
                },
            )?,
        ]);
    }
    Ok(ModelCatalog::new(entries)?)
}

fn groq_dialect() -> Result<ChatCompletionsDialect, DialectError> {
    let reasoning = WireFieldName::new("reasoning")?;
    Ok(ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning)
        .with_stream_usage(false))
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum GroqProfileError {
    #[error("invalid Groq identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Groq support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Groq model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid Groq Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid Groq compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("Groq verification date is invalid")]
    InvalidVerificationDate,
}

#[derive(Debug, Default)]
struct GroqChatCodecPolicy;

impl ChatCodecPolicy for GroqChatCodecPolicy {
    fn name(&self) -> &'static str {
        "groq-chat-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        _dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        let options = parse_chat_options(&extra)?;
        if models::is_compound(model.as_str()) && !request.tools.is_empty() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Compound systems do not support caller-defined function tools",
            ));
        }
        if models::is_compound(model.as_str())
            && request.structured_output.is_some()
            && options.structured_outputs != Some(false)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Compound systems require JSON-object mode instead of JSON Schema",
            ));
        }
        let mut warnings = Vec::new();
        let browser_search = options.browser_search == Some(true);
        let browser_search_supported = models::supports_browser_search(model.as_str());
        let known_unsupported = browser_search
            && models::is_known_language(model.as_str())
            && !browser_search_supported;
        if known_unsupported {
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "Groq browser search was omitted because the selected known model does not support it",
            ));
        }
        if browser_search && !known_unsupported && request.structured_output.is_some() {
            return Err(invalid(
                "Groq browser search cannot be combined with structured output",
            ));
        }
        if options.reasoning_format == Some(GroqReasoningFormat::Raw)
            && (browser_search || !request.tools.is_empty() || request.structured_output.is_some())
        {
            return Err(invalid(
                "Groq raw reasoning cannot be combined with tools or structured output",
            ));
        }
        if options.structured_outputs == Some(false) && request.structured_output.is_some() {
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "Groq JSON Schema structured output was lowered to JSON-object mode",
            ));
        }

        let mut wire = language_wire_options(&options)?;
        if browser_search && !known_unsupported {
            wire.insert(BROWSER_SEARCH_MARKER.to_string(), Value::Bool(true));
        }
        if let Some(enabled) = options.structured_outputs {
            wire.insert(STRUCTURED_OUTPUTS_MARKER.to_string(), Value::Bool(enabled));
        }
        if let Some(enabled) = options.strict_json_schema {
            wire.insert(STRICT_JSON_SCHEMA_MARKER.to_string(), Value::Bool(enabled));
        }
        Ok(PreparedChatCall {
            request,
            dialect: groq_dialect().map_err(dialect_error)?,
            extra: wire,
            headers: RequestHeaders::new(),
            prompt_cache_resolver: None,
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
        let object = body.as_object_mut().ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "Groq Chat encoder did not produce an object",
            )
        })?;
        let browser_search = take_marker(object, BROWSER_SEARCH_MARKER);
        let structured_outputs = take_marker(object, STRUCTURED_OUTPUTS_MARKER);
        let strict_json_schema = take_marker(object, STRICT_JSON_SCHEMA_MARKER);

        if browser_search == Some(true) {
            let tools = object
                .entry("tools".to_string())
                .or_insert_with(|| Value::Array(Vec::new()))
                .as_array_mut()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Internal,
                        "Groq Chat encoder produced a non-array tools field",
                    )
                })?;
            if !tools
                .iter()
                .any(|tool| tool.get("type").and_then(Value::as_str) == Some("browser_search"))
            {
                tools.push(serde_json::json!({"type": "browser_search"}));
            }
        }
        if structured_outputs == Some(false)
            && object
                .get("response_format")
                .and_then(|format| format.get("type"))
                .and_then(Value::as_str)
                == Some("json_schema")
        {
            object.insert(
                "response_format".to_string(),
                serde_json::json!({"type": "json_object"}),
            );
        }
        if let Some(strict) = strict_json_schema
            && let Some(schema) = object
                .get_mut("response_format")
                .and_then(Value::as_object_mut)
                .filter(|format| format.get("type").and_then(Value::as_str) == Some("json_schema"))
                .and_then(|format| format.get_mut("json_schema"))
                .and_then(Value::as_object_mut)
        {
            schema.insert("strict".to_string(), Value::Bool(strict));
        }
        Ok(body)
    }

    fn decode_response(
        &self,
        scope: &siumai_core::ProviderScope,
        model: &ModelId,
        headers: &ResponseHeaders,
        body: &[u8],
        dialect: &ChatCompletionsDialect,
    ) -> Result<LanguageResponse, Error> {
        decode_groq_response(scope, model, Some(headers), body, dialect)
    }

    fn stream_decoder(
        &self,
        scope: siumai_core::ProviderScope,
        model: ModelId,
        dialect: ChatCompletionsDialect,
    ) -> CompatibleStreamDecoder {
        Box::new(GroqStreamDecoder {
            inner: ChatCompletionsStreamDecoder::new(scope, model, dialect),
            metadata: Map::new(),
        })
    }
}

#[derive(Debug, Default)]
struct GroqResponsesCodecPolicy;

impl ResponsesCodecPolicy for GroqResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "groq-responses-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        reject_unsupported_responses_fields(&extra)?;
        let options = parse_responses_options(&extra)?;
        if models::is_compound(model.as_str()) && !request.tools.is_empty() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Compound systems do not support caller-defined function tools",
            ));
        }
        if models::is_compound(model.as_str()) && request.structured_output.is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Groq Compound Responses does not support JSON Schema structured output",
            ));
        }
        let browser_search = options.browser_search == Some(true);
        let code_execution = options.code_execution == Some(true);
        let known_model = models::is_known_language(model.as_str());
        let browser_search_supported = models::supports_responses_browser_search(model.as_str());
        let code_execution_supported = models::supports_responses_code_execution(model.as_str());
        let mut warnings = Vec::new();
        if browser_search && known_model && !browser_search_supported {
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "Groq browser search was omitted because the selected known Responses model does not support it",
            ));
        }
        if code_execution && known_model && !code_execution_supported {
            warnings.push(Warning::new(
                WarningKind::UnsupportedOption,
                "Groq code execution was omitted because the selected known Responses model does not support it",
            ));
        }
        if browser_search
            && (!known_model || browser_search_supported)
            && request.structured_output.is_some()
        {
            return Err(invalid(
                "Groq browser search cannot be combined with structured output",
            ));
        }

        extra.remove("background");
        extra.remove("browser_search");
        extra.remove("code_execution");
        extra.remove("inference_metrics");
        extra.remove("remote_mcp_tools");
        extra.remove("reasoning_effort");
        if let Some(effort) = options.reasoning_effort {
            let effort = serde_json::to_value(effort).map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "Groq reasoning effort could not be serialized",
                )
                .with_source(source)
            })?;
            extra.insert(
                "reasoning".to_string(),
                serde_json::json!({"effort": effort}),
            );
        }

        let mut native_tools = Vec::new();
        if browser_search && (!known_model || browser_search_supported) {
            native_tools.push(serde_json::json!({"type": "browser_search"}));
        }
        if code_execution && (!known_model || code_execution_supported) {
            native_tools.push(serde_json::json!({
                "type": "code_interpreter",
                "container": {"type": "auto"}
            }));
        }
        for tool in &options.remote_mcp_tools {
            native_tools.push(tool.as_value().map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "Groq remote-MCP tool could not be serialized",
                )
                .with_source(source)
            })?);
        }
        let headers = if options.inference_metrics == Some(true) {
            RequestHeaders::new()
                .try_insert(
                    HeaderName::from_static("groq-beta"),
                    HeaderValue::from_static("inference-metrics"),
                )
                .map_err(|source| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        "Groq inference-metrics header is invalid",
                    )
                    .with_source(source)
                })?
        } else {
            RequestHeaders::new()
        };
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers,
            native_tools,
            function_tools: BTreeMap::new(),
            warnings,
        })
    }

    fn decode_response(
        &self,
        scope: &siumai_core::ProviderScope,
        model: &ModelId,
        headers: &ResponseHeaders,
        body: &[u8],
    ) -> Result<LanguageResponse, Error> {
        decode_groq_responses_response(scope, model, Some(headers), body)
    }

    fn stream_decoder(
        &self,
        scope: siumai_core::ProviderScope,
        model: ModelId,
        wire_dialect: ResponsesWireDialect,
    ) -> CompatibleStreamDecoder {
        Box::new(GroqResponsesStreamDecoder {
            inner: ResponsesStreamDecoder::new(scope, model).with_wire_dialect(wire_dialect),
            metadata: Map::new(),
        })
    }
}

fn decode_groq_responses_response(
    scope: &siumai_core::ProviderScope,
    model: &ModelId,
    headers: Option<&ResponseHeaders>,
    body: &[u8],
) -> Result<LanguageResponse, Error> {
    let value = serde_json::from_slice::<Value>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Groq returned malformed Responses JSON",
        )
        .with_source(source)
    })?;
    let mut metadata = extract_responses_metadata(&value);
    if let Some(request_id) = checked_request_id(headers.and_then(response_request_id))? {
        metadata.insert("requestId".to_string(), Value::String(request_id));
    }
    let decoded = decode_responses_response(body, scope, model)?;
    let (_, response) = decoded.into_parts();
    Ok(with_groq_responses_metadata(response, &metadata))
}

struct GroqResponsesStreamDecoder {
    inner: ResponsesStreamDecoder,
    metadata: Map<String, Value>,
}

impl LanguageStreamDecoder for GroqResponsesStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.inner.set_response_diagnostics(diagnostics);
    }

    fn decode(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if let Ok(value) = serde_json::from_str::<Value>(frame) {
            merge_metadata(&mut self.metadata, extract_responses_metadata(&value));
        }
        let mut events = self.inner.decode(frame)?;
        annotate_responses_events(&mut events, &self.metadata);
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut events = self.inner.finish()?;
        annotate_responses_events(&mut events, &self.metadata);
        Ok(events)
    }

    fn terminal_seen(&self) -> bool {
        self.inner.terminal_seen()
    }
}

fn decode_groq_response(
    scope: &siumai_core::ProviderScope,
    model: &ModelId,
    headers: Option<&ResponseHeaders>,
    body: &[u8],
    dialect: &ChatCompletionsDialect,
) -> Result<LanguageResponse, Error> {
    let mut value = serde_json::from_slice::<Value>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Groq returned malformed Chat Completions JSON",
        )
        .with_source(source)
    })?;
    let mut metadata = extract_metadata(&value);
    if let Some(request_id) = checked_request_id(headers.and_then(response_request_id))? {
        metadata.insert("requestId".to_string(), Value::String(request_id));
    }
    normalize_usage(&mut value);
    let normalized = serde_json::to_vec(&value).map_err(|source| {
        Error::new(ErrorKind::Internal, "Groq response normalization failed").with_source(source)
    })?;
    let response = decode_chat_response(scope, model, &normalized, dialect)?;
    Ok(with_groq_metadata(response, &metadata))
}

struct GroqStreamDecoder {
    inner: ChatCompletionsStreamDecoder,
    metadata: Map<String, Value>,
}

impl LanguageStreamDecoder for GroqStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.inner.set_response_diagnostics(diagnostics);
    }

    fn decode(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if frame.trim() == "[DONE]" {
            let mut events = self.inner.decode(frame)?;
            annotate_events(&mut events, &self.metadata);
            return Ok(events);
        }
        let Ok(mut value) = serde_json::from_str::<Value>(frame) else {
            return self.inner.decode(frame);
        };
        merge_metadata(&mut self.metadata, extract_metadata(&value));
        normalize_usage(&mut value);
        let normalized = serde_json::to_string(&value).map_err(|source| {
            Error::new(ErrorKind::Internal, "Groq stream normalization failed").with_source(source)
        })?;
        let mut events = self.inner.decode(&normalized)?;
        annotate_events(&mut events, &self.metadata);
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut events = self.inner.finish()?;
        annotate_events(&mut events, &self.metadata);
        Ok(events)
    }

    fn terminal_seen(&self) -> bool {
        self.inner.terminal_seen()
    }
}

fn parse_chat_options(extra: &BTreeMap<String, Value>) -> Result<GroqLanguageOptions, Error> {
    let options = serde_json::from_value::<GroqLanguageOptions>(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Groq language options have an invalid shape",
        )
        .with_source(source)
    })?;
    siumai_core::TypedProviderOptions::validate(&options).map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "Groq language options are invalid").with_source(source)
    })?;
    Ok(options)
}

fn parse_responses_options(extra: &BTreeMap<String, Value>) -> Result<GroqResponsesOptions, Error> {
    let options = serde_json::from_value::<GroqResponsesOptions>(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Groq Responses options have an invalid shape",
        )
        .with_source(source)
    })?;
    siumai_core::TypedProviderOptions::validate(&options).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Groq Responses options are invalid",
        )
        .with_source(source)
    })?;
    Ok(options)
}

fn reject_unsupported_responses_fields(extra: &BTreeMap<String, Value>) -> Result<(), Error> {
    for (field, message) in [
        (
            "previous_response_id",
            "Groq Responses does not support previous_response_id",
        ),
        ("store", "Groq Responses does not support stored responses"),
        (
            "truncation",
            "Groq Responses does not support truncation control",
        ),
        (
            "include",
            "Groq Responses does not support include expansion",
        ),
        (
            "safety_identifier",
            "Groq Responses does not support safety_identifier",
        ),
        (
            "prompt_cache_key",
            "Groq Responses prompt caching is automatic and does not accept prompt_cache_key",
        ),
        ("prompt", "Groq Responses does not support prompt templates"),
    ] {
        if extra.contains_key(field) {
            return Err(Error::new(ErrorKind::Unsupported, message));
        }
    }
    Ok(())
}

fn language_wire_options(options: &GroqLanguageOptions) -> Result<BTreeMap<String, Value>, Error> {
    let mut value = serde_json::to_value(options).map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "Groq language options could not be serialized",
        )
        .with_source(source)
    })?;
    let object = value.as_object_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::Internal,
            "Groq language options did not serialize to an object",
        )
    })?;
    for control in ["browser_search", "structured_outputs", "strict_json_schema"] {
        object.remove(control);
    }
    Ok(std::mem::take(object).into_iter().collect())
}

fn take_marker(object: &mut Map<String, Value>, name: &str) -> Option<bool> {
    object.remove(name).and_then(|value| value.as_bool())
}

fn normalize_usage(value: &mut Value) {
    let x_groq_usage = value
        .get("x_groq")
        .and_then(|value| value.get("usage"))
        .cloned();
    if let Some(object) = value.as_object_mut()
        && object.get("usage").is_none_or(Value::is_null)
        && let Some(usage) = x_groq_usage
    {
        object.insert("usage".to_string(), usage);
    }
}

fn extract_metadata(value: &Value) -> Map<String, Value> {
    let mut metadata = Map::new();
    copy_string(value, "id", &mut metadata, "id");
    copy_string(value, "model", &mut metadata, "modelId");
    copy_string(value, "service_tier", &mut metadata, "serviceTier");
    copy_string(
        value,
        "system_fingerprint",
        &mut metadata,
        "systemFingerprint",
    );
    if let Some(created) = value.get("created").and_then(Value::as_i64)
        && let Some(timestamp) = DateTime::<Utc>::from_timestamp(created, 0)
    {
        metadata.insert(
            "createdAt".to_string(),
            Value::String(timestamp.to_rfc3339()),
        );
    }
    if let Some(x_groq) = bounded_x_groq_metadata(value.get("x_groq")) {
        metadata.insert("xGroq".to_string(), x_groq);
    }
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|values| values.first());
    if choice
        .and_then(|choice| choice.get("logprobs"))
        .is_some_and(|value| !value.is_null())
    {
        metadata.insert("hasLogprobs".to_string(), Value::Bool(true));
    }
    let message = choice.and_then(|choice| choice.get("message"));
    if message
        .and_then(|message| message.get("executed_tools"))
        .is_some_and(|value| !value.is_null())
    {
        metadata.insert("hasExecutedTools".to_string(), Value::Bool(true));
    }
    if message
        .and_then(|message| message.get("citations"))
        .or_else(|| value.get("citations"))
        .is_some_and(|value| !value.is_null())
    {
        metadata.insert("hasCitations".to_string(), Value::Bool(true));
    }
    metadata
}

fn extract_responses_metadata(value: &Value) -> Map<String, Value> {
    let response = value
        .get("response")
        .filter(|value| value.is_object())
        .unwrap_or(value);
    let mut metadata = Map::new();
    copy_string(response, "id", &mut metadata, "id");
    copy_string(response, "model", &mut metadata, "modelId");
    copy_string(response, "status", &mut metadata, "status");
    copy_string(response, "service_tier", &mut metadata, "serviceTier");
    if let Some(created) = response.get("created_at").and_then(Value::as_i64)
        && let Some(timestamp) = DateTime::<Utc>::from_timestamp(created, 0)
    {
        metadata.insert(
            "createdAt".to_string(),
            Value::String(timestamp.to_rfc3339()),
        );
    }
    if let Some(value) = response.get("metadata") {
        metadata.insert("hasMetadata".to_string(), Value::Bool(!value.is_null()));
        if let Some(object) = value.as_object() {
            metadata.insert(
                "metadataFieldCount".to_string(),
                Value::from(object.len().min(64) as u64),
            );
        }
    }
    if response
        .get("reasoning")
        .is_some_and(|value| !value.is_null())
    {
        metadata.insert("hasReasoning".to_string(), Value::Bool(true));
    }
    if response
        .get("incomplete_details")
        .is_some_and(|value| !value.is_null())
    {
        metadata.insert("hasIncompleteDetails".to_string(), Value::Bool(true));
    }
    if response.get("error").is_some_and(|value| !value.is_null()) {
        metadata.insert("hasError".to_string(), Value::Bool(true));
    }
    for (source, target) in [
        ("parallel_tool_calls", "parallelToolCalls"),
        ("background", "background"),
    ] {
        if let Some(value) = response.get(source).and_then(Value::as_bool) {
            metadata.insert(target.to_string(), Value::Bool(value));
        }
    }
    if let Some(value) = response.get("max_tool_calls").and_then(Value::as_u64) {
        metadata.insert("maxToolCalls".to_string(), Value::from(value.min(4096)));
    }
    if let Some(x_groq) = bounded_x_groq_metadata(response.get("x_groq")) {
        metadata.insert("xGroq".to_string(), x_groq);
    }
    metadata
}

fn bounded_x_groq_metadata(value: Option<&Value>) -> Option<Value> {
    let usage = value
        .and_then(Value::as_object)
        .and_then(|object| object.get("usage"))
        .and_then(Value::as_object)?;
    let mut bounded = Map::new();
    for (name, value) in usage.iter().take(32) {
        if name.len() <= 64 && (value.is_u64() || value.is_i64() || value.is_f64()) {
            bounded.insert(name.clone(), value.clone());
        }
    }
    (!bounded.is_empty()).then(|| {
        Value::Object(Map::from_iter([(
            "usage".to_string(),
            Value::Object(bounded),
        )]))
    })
}

fn copy_string(
    source: &Value,
    source_name: &str,
    target: &mut Map<String, Value>,
    target_name: &str,
) {
    if let Some(value) = source
        .get(source_name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        target.insert(target_name.to_string(), Value::String(value.to_string()));
    }
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .expose()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    })
}

fn checked_request_id(value: Option<String>) -> Result<Option<String>, Error> {
    value
        .map(|value| {
            PublicDiagnosticText::new(value.clone())
                .map(|_| value)
                .map_err(|source| {
                    Error::protocol_violation("Groq response contains an invalid request ID")
                        .with_source(source)
                })
        })
        .transpose()
}

fn merge_metadata(target: &mut Map<String, Value>, incoming: Map<String, Value>) {
    target.extend(incoming);
}

fn with_groq_metadata(
    response: LanguageResponse,
    metadata: &Map<String, Value>,
) -> LanguageResponse {
    if metadata.is_empty() {
        return response;
    }
    let mut provider = response.provider_metadata().clone();
    if let Some(Value::Object(openai)) = provider.remove("openai") {
        let mut merged = metadata.clone();
        for (name, value) in openai {
            merged.entry(name).or_insert(value);
        }
        provider.insert("groq".to_string(), Value::Object(merged));
    } else {
        provider.insert("groq".to_string(), Value::Object(metadata.clone()));
    }
    response.with_provider_metadata(provider)
}

fn with_groq_responses_metadata(
    response: LanguageResponse,
    metadata: &Map<String, Value>,
) -> LanguageResponse {
    let mut provider = response.provider_metadata().clone();
    let mut merged = metadata.clone();
    if let Some(native) = provider.remove(OPENAI_RESPONSES_PROTOCOL) {
        merged.insert("responses".to_string(), native);
    }
    if let Some(Value::Object(existing)) = provider.remove("groq") {
        for (name, value) in existing {
            merged.entry(name).or_insert(value);
        }
    }
    provider.insert("groq".to_string(), Value::Object(merged));
    response.with_provider_metadata(provider)
}

fn annotate_events(events: &mut [LanguageStreamEvent], metadata: &Map<String, Value>) {
    if metadata.is_empty() {
        return;
    }
    for event in events {
        let LanguageStreamEvent::Terminal(terminal) = event else {
            continue;
        };
        match terminal {
            StreamTerminal::Completed { response } => annotate_response(response, metadata),
            StreamTerminal::Failed { response, .. }
            | StreamTerminal::Cancelled { response, .. } => {
                if let Some(response) = response {
                    annotate_response(response, metadata);
                }
            }
            _ => {}
        }
    }
}

fn annotate_response(response: &mut Box<LanguageResponse>, metadata: &Map<String, Value>) {
    **response = with_groq_metadata((**response).clone(), metadata);
}

fn annotate_responses_events(events: &mut [LanguageStreamEvent], metadata: &Map<String, Value>) {
    for event in events {
        let LanguageStreamEvent::Terminal(terminal) = event else {
            continue;
        };
        match terminal {
            StreamTerminal::Completed { response } => {
                annotate_responses_response(response, metadata);
            }
            StreamTerminal::Failed { response, .. }
            | StreamTerminal::Cancelled { response, .. } => {
                if let Some(response) = response {
                    annotate_responses_response(response, metadata);
                }
            }
            _ => {}
        }
    }
}

fn annotate_responses_response(
    response: &mut Box<LanguageResponse>,
    metadata: &Map<String, Value>,
) {
    **response = with_groq_responses_metadata((**response).clone(), metadata);
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn dialect_error(source: DialectError) -> Error {
    Error::new(ErrorKind::Internal, "Groq Chat dialect is invalid").with_source(source)
}

#[cfg(test)]
mod tests {
    use siumai_core::{
        ContentPart, LanguageStreamEvent, Message, MessageRole, ProviderOptions, ReplayDomainId,
    };

    use super::*;
    use crate::GroqLanguageResponseExt;

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    fn chat_scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new(PROVIDER_ID).expect("provider"))
            .with_protocol(ProtocolId::new(CHAT_PROTOCOL_ID).expect("protocol"))
            .with_api_mode(ApiModeId::new(CHAT_API_MODE_ID).expect("api mode"))
            .with_replay_domain(test_replay_domain())
    }

    fn test_replay_domain() -> ReplayDomain {
        ReplayDomain::official(ReplayDomainId::new("groq-public-api").expect("replay domain"))
    }

    #[test]
    fn verified_catalog_distinguishes_known_and_future_models() {
        let endpoint = siumai_transport::EndpointConfig::official(
            DEFAULT_BASE_URL,
            siumai_transport::OfficialOrigin::new("https://api.groq.com").unwrap(),
        )
        .unwrap();
        let profile = profile(endpoint, test_replay_domain(), true).unwrap();
        let claims = profile.provider_profile().verified_claims().unwrap();
        assert_eq!(claims.len(), 2);
        let support_scope = claims[0].scope().clone();
        let responses_scope = claims
            .iter()
            .find(|claim| claim.scope().api_mode().as_str() == RESPONSES_API_MODE_ID)
            .unwrap()
            .scope()
            .clone();
        let catalog = profile.provider_profile().catalog().unwrap();

        assert!(
            catalog
                .get(
                    &support_scope,
                    &ModelId::new(models::language::GPT_OSS_20B).unwrap()
                )
                .is_some()
        );
        assert!(
            catalog
                .get(&support_scope, &ModelId::new("future-groq-model").unwrap())
                .is_none()
        );
        assert!(
            catalog
                .get(
                    &responses_scope,
                    &ModelId::new(models::language::GPT_OSS_120B).unwrap()
                )
                .is_some()
        );
        assert!(matches!(
            catalog
                .get(
                    &support_scope,
                    &ModelId::new(models::language::LLAMA_3_3_70B_VERSATILE).unwrap()
                )
                .unwrap()
                .lifecycle(),
            ModelLifecycle::Deprecated { .. }
        ));
        assert!(matches!(
            catalog
                .get(
                    &support_scope,
                    &ModelId::new(models::language::QWEN3_32B).unwrap()
                )
                .unwrap()
                .lifecycle(),
            ModelLifecycle::Retired { .. }
        ));
    }

    #[test]
    fn responses_options_build_hosted_tools_reasoning_and_metrics_header() {
        let options = GroqResponsesOptions::new()
            .with_background(false)
            .with_service_tier(crate::GroqResponsesServiceTier::Flex)
            .with_reasoning_effort(crate::GroqReasoningEffort::Low)
            .with_browser_search(true)
            .with_code_execution(true)
            .with_remote_mcp_tool(
                crate::GroqRemoteMcpTool::new("docs", "https://mcp.example.com/sse")
                    .with_header("Authorization", "Bearer secret")
                    .with_require_approval(crate::GroqMcpApproval::Always),
            )
            .with_inference_metrics(true)
            .with_metadata([("trace", "test")]);
        let erased = ProviderOptions::typed(&options).unwrap();
        let extra = erased
            .value()
            .clone()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let prepared = GroqResponsesCodecPolicy
            .prepare(
                &ModelId::new(models::language::GPT_OSS_20B).unwrap(),
                request(),
                extra,
            )
            .unwrap();

        assert!(!prepared.extra.contains_key("background"));
        assert_eq!(prepared.extra["service_tier"], "flex");
        assert_eq!(prepared.extra["reasoning"]["effort"], "low");
        assert_eq!(prepared.extra["metadata"]["trace"], "test");
        assert_eq!(
            prepared.native_tools,
            vec![
                serde_json::json!({"type":"browser_search"}),
                serde_json::json!({
                    "type":"code_interpreter",
                    "container":{"type":"auto"}
                }),
                serde_json::json!({
                    "type":"mcp",
                    "server_label":"docs",
                    "server_url":"https://mcp.example.com/sse",
                    "headers":{"Authorization":"Bearer secret"},
                    "require_approval":"always"
                }),
            ]
        );
        assert_eq!(
            prepared
                .headers
                .get(&HeaderName::from_static("groq-beta"))
                .unwrap()
                .to_str()
                .unwrap(),
            "inference-metrics"
        );
    }

    #[test]
    fn responses_reject_unsupported_stateful_fields_before_encoding() {
        let error = GroqResponsesCodecPolicy
            .prepare(
                &ModelId::new("future-groq-model").unwrap(),
                request(),
                BTreeMap::from([(
                    "previous_response_id".to_string(),
                    Value::String("resp-1".to_string()),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);

        let error = GroqResponsesCodecPolicy
            .prepare(
                &ModelId::new("future-groq-model").unwrap(),
                request(),
                BTreeMap::from([("background".to_string(), Value::Bool(true))]),
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn responses_decode_preserves_usage_and_rebrands_metadata() {
        let scope = siumai_core::ProviderScope::new(ProviderId::new(PROVIDER_ID).unwrap())
            .with_platform(PlatformId::new(PLATFORM_ID).unwrap())
            .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
            .with_api_mode(ApiModeId::new(RESPONSES_API_MODE_ID).unwrap())
            .with_replay_domain(test_replay_domain());
        let body = serde_json::json!({
            "id": "resp-groq-1",
            "object": "response",
            "created_at": 1_786_080_000,
            "model": models::language::GPT_OSS_20B,
            "status": "completed",
            "output": [{
                "id":"mcp-approval-1",
                "type":"mcp_approval_request",
                "server_label":"docs",
                "name":"search",
                "arguments":"{\"query\":\"siumai\"}"
            }],
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 7
            },
            "error": null,
            "incomplete_details": null,
            "reasoning": {"effort":"low"},
            "service_tier": "flex",
            "background": false,
            "metadata": {"trace":"test"}
        });
        let response = decode_groq_responses_response(
            &scope,
            &ModelId::new(models::language::GPT_OSS_20B).unwrap(),
            None,
            &serde_json::to_vec(&body).unwrap(),
        )
        .unwrap();

        assert_eq!(response.usage().cache_read_tokens.value(), Some(2));
        assert_eq!(response.usage().reasoning_tokens.value(), Some(1));
        assert_eq!(response.provider_metadata()["groq"]["serviceTier"], "flex");
        assert!(
            response
                .provider_metadata()
                .get(OPENAI_RESPONSES_PROTOCOL)
                .is_none()
        );
        assert_eq!(response.provider_metadata()["groq"]["hasMetadata"], true);
        assert_eq!(
            response.provider_metadata()["groq"]["metadataFieldCount"],
            1
        );
        assert!(!format!("{response:?}").contains("test"));
        let mcp = response.groq_mcp_outputs();
        assert_eq!(mcp.len(), 1);
        assert_eq!(mcp[0].kind(), crate::GroqMcpOutputKind::ApprovalRequest);
        assert_eq!(mcp[0].item_id(), Some("mcp-approval-1"));
        assert_eq!(mcp[0].data()["server_label"], "docs");
    }

    #[test]
    fn browser_search_and_structured_controls_are_codec_owned() {
        let options = GroqLanguageOptions::new()
            .with_browser_search(true)
            .with_structured_outputs(false);
        let erased = ProviderOptions::typed(&options).unwrap();
        let extra = erased
            .value()
            .clone()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let prepared = GroqChatCodecPolicy
            .prepare(
                &ModelId::new(models::language::GPT_OSS_20B).unwrap(),
                request(),
                ChatCompletionsDialect::generic(),
                extra,
            )
            .unwrap();
        let body = GroqChatCodecPolicy
            .encode_request(
                &chat_scope(),
                &ModelId::new(models::language::GPT_OSS_20B).unwrap(),
                &prepared,
                false,
            )
            .unwrap();

        assert_eq!(
            body["tools"],
            serde_json::json!([{"type":"browser_search"}])
        );
        assert!(body.get(BROWSER_SEARCH_MARKER).is_none());
    }

    #[test]
    fn x_groq_usage_and_reasoning_are_preserved() {
        let scope = siumai_core::ProviderScope::new(ProviderId::new(PROVIDER_ID).unwrap())
            .with_protocol(ProtocolId::new(CHAT_PROTOCOL_ID).unwrap())
            .with_api_mode(ApiModeId::new(CHAT_API_MODE_ID).unwrap())
            .with_replay_domain(test_replay_domain());
        let body = serde_json::json!({
            "id": "chatcmpl-groq-1",
            "created": 1_741_392_000,
            "model": "future-groq-model",
            "choices": [{
                "index": 0,
                "message": {"role":"assistant", "content":"answer", "reasoning":"work"},
                "finish_reason": "stop",
                "logprobs": {"content": []}
            }],
            "x_groq": {"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}
        });
        let response = decode_groq_response(
            &scope,
            &ModelId::new("future-groq-model").unwrap(),
            None,
            &serde_json::to_vec(&body).unwrap(),
            &groq_dialect().unwrap(),
        )
        .unwrap();

        assert_eq!(response.usage().input_tokens.value(), Some(3));
        assert!(
            response
                .content()
                .iter()
                .any(|part| matches!(part, ContentPart::Reasoning { text } if text == "work"))
        );
        assert_eq!(
            response.provider_metadata()["groq"]["xGroq"]["usage"]["total_tokens"],
            5
        );
    }

    #[test]
    fn streamed_x_groq_usage_reaches_usage_and_terminal_metadata() {
        let scope = siumai_core::ProviderScope::new(ProviderId::new(PROVIDER_ID).unwrap())
            .with_protocol(ProtocolId::new(CHAT_PROTOCOL_ID).unwrap())
            .with_api_mode(ApiModeId::new(CHAT_API_MODE_ID).unwrap())
            .with_replay_domain(test_replay_domain());
        let mut decoder = GroqStreamDecoder {
            inner: ChatCompletionsStreamDecoder::new(
                scope,
                ModelId::new("future-groq-model").unwrap(),
                groq_dialect().unwrap(),
            ),
            metadata: Map::new(),
        };
        let events = decoder
            .decode(
                r#"{"id":"chat-1","model":"future-groq-model","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}}"#,
            )
            .unwrap();
        assert!(events.iter().any(|event| matches!(event, LanguageStreamEvent::Usage(usage) if usage.total_tokens.value() == Some(3))));
        let terminal = decoder.decode("[DONE]").unwrap();
        let response = terminal
            .iter()
            .find_map(|event| match event {
                LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => {
                    Some(response.as_ref())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(
            response.provider_metadata()["groq"]["xGroq"]["usage"]["total_tokens"],
            3
        );
    }
}
