//! Alibaba Qwen language profiles backed by the shared OpenAI-compatible runtime.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use http::header::{HeaderName, HeaderValue};
use serde_json::Value;
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleProfile, MessagesCallOptions,
    MessagesRequestPolicy, MessagesRequestRequirements,
};
use siumai_core::{
    ApiModeId, ApiStability, Error, ErrorKind, InvalidId, LanguageRequest, ModelCatalog,
    ModelFamily, ModelId, OfficialSource, PlatformId, ProfileError, ProfileId, ProtocolContractId,
    ProtocolId, ProviderId, ProviderProfile, ProviderScope, ReplayDomain, SupportScope,
    TypedProviderOptions, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim,
};
use siumai_openai_compatible::extension::v1::{
    ChatCodecPolicy, PreparedChatCall, PreparedResponsesCall, ResponsesCodecPolicy,
};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID as MESSAGES_API_MODE_ID, CacheControlWireStyle, MessagesEncodingRules,
    PROTOCOL_ID as MESSAGES_PROTOCOL_ID,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, ChatPromptCacheBlock, DialectError,
    MaxOutputTokensField, PROTOCOL_ID as CHAT_PROTOCOL_ID, WireFieldName,
};
use siumai_protocol_openai::responses::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL,
};
use siumai_transport::{EndpointConfig, RequestHeaders};
use thiserror::Error as ThisError;

use crate::annotations::AlibabaAnnotationResolver;
use crate::options::{
    ALIBABA_SESSION_CACHE_HEADER, AlibabaChatOptions, AlibabaResponsesOptions, AlibabaResponsesTool,
};

pub const PROVIDER_ID: &str = "alibaba";
pub const PLATFORM_ID: &str = "alibaba-model-studio";
pub const LEGACY_SINGAPORE_LANGUAGE_BASE_URL: &str =
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
pub const LEGACY_SINGAPORE_MESSAGES_BASE_URL: &str =
    "https://dashscope-intl.aliyuncs.com/apps/anthropic";
pub const CHAT_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions";
pub const RESPONSES_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-responses";
pub const VERIFIED_ON: &str = "2026-08-05";
pub const MESSAGES_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages";
pub const MESSAGES_VERIFIED_ON: &str = "2026-08-08";

pub(crate) const MESSAGES_API_VERSION: &str = "2023-06-01";

pub(crate) fn messages_profile(
    endpoint: EndpointConfig,
    verified_endpoint: bool,
    replay_domain: ReplayDomain,
) -> Result<AnthropicCompatibleProfile, AlibabaProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let scope = SupportScope::new(
        provider.clone(),
        platform.clone(),
        ModelFamily::Language,
        ProtocolId::new(MESSAGES_PROTOCOL_ID)?,
        ApiModeId::new(MESSAGES_API_MODE_ID)?,
    );
    let profile = if verified_endpoint {
        let verified_at = VerificationDate::new(
            NaiveDate::parse_from_str(MESSAGES_VERIFIED_ON, "%Y-%m-%d")
                .map_err(|_| AlibabaProfileError::InvalidVerificationDate)?,
        );
        let evidence = VerificationEvidence::new(
            OfficialSource::new(MESSAGES_SOURCE)?,
            verified_at,
            ProtocolContractId::new("alibaba-anthropic-messages-2026-08")?,
        );
        let provider_profile = ProviderProfile::verified(
            ProfileId::new("alibaba-messages")?,
            vec![VerifiedSupportClaim::new(
                scope,
                VerifiedFidelity::Compatible,
                ApiStability::Stable,
                evidence,
            )],
            ModelCatalog::default(),
        )?;
        AnthropicCompatibleProfile::verified(provider_profile, endpoint, MESSAGES_API_VERSION)?
            .with_replay_domain(replay_domain)?
    } else {
        AnthropicCompatibleProfile::custom(
            ProfileId::new("alibaba-messages")?,
            provider,
            platform,
            endpoint,
            replay_domain,
            MESSAGES_API_VERSION,
        )?
    };
    Ok(profile
        .with_messages_target("v1/messages")?
        .with_annotation_resolver(Arc::new(AlibabaAnnotationResolver))
        .with_encoding_rules(
            MessagesEncodingRules::compatible_baseline()
                .with_cache_control(CacheControlWireStyle::FiveMinutesImplicit)
                .with_video_input(true),
        )
        .with_request_policy(Arc::new(AlibabaMessagesPolicy)))
}

#[derive(Debug, Clone, Copy, Default)]
struct AlibabaMessagesPolicy;

impl MessagesRequestPolicy for AlibabaMessagesPolicy {
    fn prepare(
        &self,
        _model: &ModelId,
        _request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        if options.metadata().is_some()
            || options.output_effort().is_some()
            || options.task_budget().is_some()
            || options.fallbacks().is_some()
            || options.top_k().is_some()
            || options.service_tier().is_some()
            || options.cache_control().is_some()
            || options.speed().is_some()
            || options.inference_geo().is_some()
            || options.container().is_some()
            || options.context_management().is_some()
            || options.mcp_servers().is_some()
            || !options.extra().is_empty()
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Alibaba Messages accepts only verified Alibaba request controls",
            ));
        }
        Ok(MessagesRequestRequirements::new())
    }
}

pub(crate) fn profile(
    endpoint: EndpointConfig,
    verified_endpoint: bool,
    replay_domain: ReplayDomain,
) -> Result<OpenAiCompatibleProfile, AlibabaProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let reasoning = WireFieldName::new("reasoning_content")?;
    let cache_read = WireFieldName::new("cache_read_input_tokens")?;
    let cache_write = WireFieldName::new("cache_creation_input_tokens")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning)
        .with_cache_read_tokens_field(cache_read)
        .with_cache_write_tokens_field(cache_write)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens);

    let profile = if verified_endpoint {
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
            NaiveDate::from_ymd_opt(2026, 8, 5)
                .ok_or(AlibabaProfileError::InvalidVerificationDate)?,
        );
        let chat_evidence = VerificationEvidence::new(
            OfficialSource::new(CHAT_SOURCE)?,
            verified_at,
            ProtocolContractId::new("alibaba-openai-chat-2026-08")?,
        );
        let responses_evidence = VerificationEvidence::new(
            OfficialSource::new(RESPONSES_SOURCE)?,
            verified_at,
            ProtocolContractId::new("alibaba-openai-responses-2026-08")?,
        );
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
            ModelCatalog::default(),
        )?;
        OpenAiCompatibleProfile::verified_chat_and_responses(provider_profile, endpoint, dialect)?
            .with_replay_domain(replay_domain)?
    } else {
        OpenAiCompatibleProfile::custom_chat_and_responses(
            provider,
            endpoint,
            replay_domain,
            dialect,
        )?
    };

    Ok(profile
        .with_chat_codec_policy(Arc::new(AlibabaChatCodecPolicy { verified_endpoint }))
        .with_responses_codec_policy(Arc::new(AlibabaResponsesCodecPolicy { verified_endpoint })))
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum AlibabaProfileError {
    #[error("invalid Alibaba identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid Alibaba support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid Alibaba Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid Alibaba compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid Alibaba Anthropic Messages profile: {0}")]
    MessagesCompatible(#[from] AnthropicCompatibleConfigError),
    #[error("Alibaba verification date is invalid")]
    InvalidVerificationDate,
}

#[derive(Debug)]
struct AlibabaChatCodecPolicy {
    verified_endpoint: bool,
}

impl ChatCodecPolicy for AlibabaChatCodecPolicy {
    fn name(&self) -> &'static str {
        "alibaba-qwen-chat-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        let options = parse_chat_options(&extra)?;
        if options.enable_search == Some(true)
            && self.verified_endpoint
            && is_known_model(model)
            && !supports_web_search(model)
        {
            return Err(invalid(
                "Alibaba web search is not declared for this Qwen model",
            ));
        }
        let prompt_cache_breakpoints = options
            .prompt_cache_breakpoints
            .iter()
            .map(|breakpoint| {
                ChatPromptCacheBlock::new(breakpoint.message_index, breakpoint.content_index)
            })
            .collect();
        extra.remove("prompt_cache_breakpoints");
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: siumai_transport::RequestHeaders::new(),
            prompt_cache_breakpoints,
            warnings: Vec::new(),
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
        let messages = body
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Internal,
                    "Alibaba Chat encoder omitted the messages array",
                )
            })?;
        for breakpoint in &prepared.prompt_cache_breakpoints {
            let content = messages
                .get_mut(breakpoint.message_index)
                .and_then(|message| message.get_mut("content"))
                .and_then(Value::as_array_mut)
                .and_then(|content| content.get_mut(breakpoint.content_index))
                .and_then(Value::as_object_mut)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Internal,
                        "Alibaba prompt-cache breakpoint lost its encoded content block",
                    )
                })?;
            if content.remove("prompt_cache_breakpoint").is_none() {
                return Err(Error::new(
                    ErrorKind::Internal,
                    "Alibaba prompt-cache breakpoint marker was not encoded",
                ));
            }
            content.insert(
                "cache_control".to_string(),
                serde_json::json!({"type": "ephemeral"}),
            );
        }
        Ok(body)
    }
}

#[derive(Debug)]
struct AlibabaResponsesCodecPolicy {
    verified_endpoint: bool,
}

impl ResponsesCodecPolicy for AlibabaResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "alibaba-qwen-responses-2026-08"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        let options = parse_responses_options(&extra)?;
        prepare_reasoning_options(&options, &mut extra)?;
        if extra.contains_key("background") {
            return Err(invalid(
                "Alibaba Responses does not support background execution",
            ));
        }
        for field in ["enable_source", "enable_citation", "citation_format"] {
            if extra.contains_key(field) {
                return Err(invalid(
                    "Alibaba citation controls belong to the native search API",
                ));
            }
        }
        let native_tools = options
            .native_tools
            .iter()
            .map(AlibabaResponsesTool::as_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Alibaba Responses tool options are invalid",
                )
                .with_source(source)
            })?;
        let has_web_extractor = native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("web_extractor"));
        let has_code_interpreter = native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("code_interpreter"));
        if native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("web_search"))
            && self.verified_endpoint
            && is_known_model(model)
            && !supports_web_search(model)
        {
            return Err(invalid(
                "Alibaba Responses web_search is not declared for this Qwen model",
            ));
        }
        if has_web_extractor
            && !native_tools
                .iter()
                .any(|tool| tool.get("type").and_then(Value::as_str) == Some("web_search"))
        {
            return Err(invalid(
                "Alibaba web_extractor requires a web_search tool in the same request",
            ));
        }
        if self.verified_endpoint
            && is_qwen3_max_model(model)
            && (has_web_extractor || has_code_interpreter)
            && reasoning_is_disabled(&extra)
        {
            return Err(invalid(
                "Alibaba Qwen3-Max web_extractor/code_interpreter requires reasoning to be enabled",
            ));
        }
        extra.remove("native_tools");
        extra.remove("reasoning_effort");
        extra.remove("session_cache");
        let mut headers = RequestHeaders::new();
        if options.session_cache == Some(true) {
            headers = headers
                .try_insert(
                    HeaderName::from_static(ALIBABA_SESSION_CACHE_HEADER),
                    HeaderValue::from_static("enable"),
                )
                .map_err(|source| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        "Alibaba session-cache header is invalid",
                    )
                    .with_source(source)
                })?;
        }
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers,
            prompt_cache_breakpoints: Vec::new(),
            native_tools,
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }
}

fn parse_chat_options(extra: &BTreeMap<String, Value>) -> Result<AlibabaChatOptions, Error> {
    let options: AlibabaChatOptions = serde_json::from_value(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Alibaba Chat options have an invalid wire shape",
        )
        .with_source(source)
    })?;
    options.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "Alibaba Chat options are invalid").with_source(source)
    })?;
    Ok(options)
}

fn parse_responses_options(
    extra: &BTreeMap<String, Value>,
) -> Result<AlibabaResponsesOptions, Error> {
    let options: AlibabaResponsesOptions = serde_json::from_value(Value::Object(
        extra.clone().into_iter().collect(),
    ))
    .map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Alibaba Responses options have an invalid wire shape",
        )
        .with_source(source)
    })?;
    options.validate().map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "Alibaba Responses options are invalid",
        )
        .with_source(source)
    })?;
    if options.previous_response_id.is_some() && options.conversation.is_some() {
        return Err(invalid(
            "Alibaba Responses previous_response_id and conversation are mutually exclusive",
        ));
    }
    Ok(options)
}

fn prepare_reasoning_options(
    options: &AlibabaResponsesOptions,
    extra: &mut BTreeMap<String, Value>,
) -> Result<(), Error> {
    if let Some(effort) = options.reasoning_effort {
        if extra.contains_key("reasoning") {
            return Err(invalid(
                "Alibaba reasoning_effort cannot be combined with a raw reasoning object",
            ));
        }
        extra.insert(
            "reasoning".to_string(),
            serde_json::json!({"effort": effort.as_str()}),
        );
    } else if let Some(reasoning) = extra.get("reasoning") {
        if !reasoning.is_object() {
            return Err(invalid("Alibaba reasoning must be an object"));
        }
        if let Some(effort) = reasoning.get("effort") {
            match effort.as_str() {
                Some("none" | "minimal" | "medium" | "high") => {}
                Some(_) | None => {
                    return Err(invalid("Alibaba reasoning.effort is invalid"));
                }
            }
        }
    }
    Ok(())
}

fn is_qwen3_max_model(model: &ModelId) -> bool {
    matches!(
        model.as_str(),
        "qwen3-max" | "qwen3-max-2026-01-23" | "qwen3-max-preview"
    )
}

fn reasoning_is_disabled(extra: &BTreeMap<String, Value>) -> bool {
    extra
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str)
        == Some("none")
        || extra.get("enable_thinking").and_then(Value::as_bool) == Some(false)
}

fn is_known_model(model: &ModelId) -> bool {
    matches!(
        model.as_str(),
        "qwen3.7-max"
            | "qwen3.7-max-2026-05-20"
            | "qwen3.7-max-2026-06-08"
            | "qwen3.7-plus"
            | "qwen3.7-plus-2026-05-26"
            | "qwen3.6-plus"
            | "qwen3.6-flash"
            | "qwen3.5-plus"
            | "qwen3.5-flash"
            | "qwen3-max"
            | "qwen3-max-2026-01-23"
            | "qwen3-max-preview"
            | "qwen-plus"
            | "qwen-flash"
            | "qwen3-coder-plus"
            | "qwen3-coder-flash"
    )
}

fn supports_web_search(model: &ModelId) -> bool {
    matches!(
        model.as_str(),
        "qwen3.7-max"
            | "qwen3.7-max-2026-05-20"
            | "qwen3.7-max-2026-06-08"
            | "qwen3.6-plus"
            | "qwen3.6-flash"
            | "qwen3.5-plus"
            | "qwen3.5-flash"
            | "qwen3-max"
            | "qwen3-max-2026-01-23"
    )
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
