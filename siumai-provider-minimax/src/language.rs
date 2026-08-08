use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use serde_json::Value;
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleProfile, MessagesCallOptions,
    MessagesRequestPolicy, MessagesRequestProjection, MessagesRequestProjectionContext,
    MessagesRequestRequirements, ProjectedMessagesRequest,
};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, ContentPart, Error, ErrorKind, InvalidId,
    LanguageRequest, MessageRole, ModelCatalog, ModelFamily, ModelId, ModelLifecycle,
    ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain, SupportScope,
    ToolChoice, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_openai_compatible::extension::{
    ChatCodecPolicy, PreparedChatCall, PreparedResponsesCall, ResponsesCodecPolicy,
};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID as MESSAGES_API_MODE_ID, CacheControlWireStyle, MessagesEncodingRules,
    MessagesServiceTier, PROTOCOL_ID as MESSAGES_PROTOCOL_ID, TemperatureEncodingRule,
    ThinkingConfig,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, DialectError, MaxOutputTokensField,
    PROTOCOL_ID as CHAT_PROTOCOL_ID, WireFieldName,
};
use siumai_protocol_openai::responses::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL, RequestEncodingOptions,
    ResponsesMediaDialect, encode_request_with_options as encode_responses_request,
};
use siumai_transport::{EndpointConfig, RequestHeaders};
use thiserror::Error as ThisError;

use crate::MinimaxAnnotationResolver;
use crate::models::{ALL_LANGUAGE, is_known_m2, is_m3};

pub(crate) const PROVIDER_ID: &str = "minimax";
pub(crate) const PLATFORM_ID: &str = "minimax-api";
pub(crate) const MESSAGES_PROFILE_ID: &str = "minimax.messages";
pub(crate) const OPENAI_PROFILE_ID: &str = "minimax.openai-compatible";
pub(crate) const MESSAGES_BASE_URL: &str = "https://api.minimax.io/anthropic/v1/";
pub(crate) const OPENAI_BASE_URL: &str = "https://api.minimax.io/v1/";
pub(crate) const OFFICIAL_ORIGIN: &str = "https://api.minimax.io";

const MESSAGES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/text-chat-anthropic";
const CHAT_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/text-chat-openai";
const RESPONSES_SOURCE: &str = "https://platform.minimax.io/docs/api-reference/responses-create";
const VERIFIED_ON: &str = "2026-08-06";
const COMPATIBLE_API_VERSION: &str = "2023-06-01";
const M3_MAX_OUTPUT_TOKENS: u64 = 524_288;
const M2_MAX_OUTPUT_TOKENS: u64 = 204_800;

pub(crate) fn messages_profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<AnthropicCompatibleProfile, MinimaxLanguageProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let platform = PlatformId::new(PLATFORM_ID)?;
    let profile = if verified_endpoint {
        verified_messages_profile(provider, platform, endpoint)?
            .with_replay_domain(replay_domain)?
    } else {
        AnthropicCompatibleProfile::custom(
            ProfileId::new(MESSAGES_PROFILE_ID)?,
            provider,
            platform,
            endpoint,
            replay_domain,
            COMPATIBLE_API_VERSION,
        )?
    };
    let rules = MessagesEncodingRules::compatible_baseline()
        .with_temperature(TemperatureEncodingRule::new(2.0)?)
        .with_cache_control(CacheControlWireStyle::FiveMinutesImplicit)
        .with_video_input(true);
    Ok(profile
        .with_annotation_resolver(Arc::new(MinimaxAnnotationResolver))
        .with_encoding_rules(rules)
        .with_request_policy(Arc::new(MinimaxMessagesPolicy))
        .with_request_projection(Arc::new(MinimaxMessagesProjection)))
}

pub(crate) fn openai_profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, MinimaxLanguageProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let reasoning_content = WireFieldName::new("reasoning_content")?;
    let reasoning_details = WireFieldName::new("reasoning_details")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_video_input(true)
        .with_reasoning_input_field(reasoning_content.clone())
        .with_reasoning_output_field(reasoning_content)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens)
        .with_stream_usage(true);
    let profile = if verified_endpoint {
        verified_openai_profile(provider, endpoint, dialect)?.with_replay_domain(replay_domain)?
    } else {
        OpenAiCompatibleProfile::custom_chat_and_responses(
            provider,
            endpoint,
            replay_domain,
            dialect,
        )?
    };
    Ok(profile
        .with_chat_codec_policy(Arc::new(MinimaxChatPolicy { reasoning_details }))
        .with_responses_codec_policy(Arc::new(MinimaxResponsesPolicy)))
}

fn verified_messages_profile(
    provider: ProviderId,
    platform: PlatformId,
    endpoint: EndpointConfig,
) -> Result<AnthropicCompatibleProfile, MinimaxLanguageProfileError> {
    let scope = support_scope(
        provider,
        platform,
        MESSAGES_PROTOCOL_ID,
        MESSAGES_API_MODE_ID,
    )?;
    let evidence = evidence(MESSAGES_SOURCE, "minimax-messages-2026-08")?;
    let catalog = ModelCatalog::new(
        ALL_LANGUAGE
            .iter()
            .map(|model| model_profile(model, &scope, &evidence))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let provider_profile = ProviderProfile::verified(
        ProfileId::new(MESSAGES_PROFILE_ID)?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
            ApiStability::Stable,
            evidence,
        )],
        catalog,
    )?;
    Ok(AnthropicCompatibleProfile::verified(
        provider_profile,
        endpoint,
        COMPATIBLE_API_VERSION,
    )?)
}

fn verified_openai_profile(
    provider: ProviderId,
    endpoint: EndpointConfig,
    dialect: ChatCompletionsDialect,
) -> Result<OpenAiCompatibleProfile, MinimaxLanguageProfileError> {
    let platform = PlatformId::new(PLATFORM_ID)?;
    let chat_scope = support_scope(
        provider.clone(),
        platform.clone(),
        CHAT_PROTOCOL_ID,
        CHAT_API_MODE_ID,
    )?;
    let responses_scope = support_scope(
        provider,
        platform,
        OPENAI_RESPONSES_PROTOCOL,
        RESPONSES_API_MODE_ID,
    )?;
    let chat_evidence = evidence(CHAT_SOURCE, "minimax-openai-chat-2026-08")?;
    let responses_evidence = evidence(RESPONSES_SOURCE, "minimax-responses-2026-08")?;
    let mut profiles = Vec::with_capacity(ALL_LANGUAGE.len() * 2);
    for model in ALL_LANGUAGE {
        profiles.push(model_profile(model, &chat_scope, &chat_evidence)?);
        profiles.push(model_profile(model, &responses_scope, &responses_evidence)?);
    }
    let provider_profile = ProviderProfile::verified(
        ProfileId::new(OPENAI_PROFILE_ID)?,
        vec![
            VerifiedSupportClaim::new(
                chat_scope,
                VerifiedFidelity::Compatible,
                ApiStability::Experimental,
                chat_evidence,
            ),
            VerifiedSupportClaim::new(
                responses_scope,
                VerifiedFidelity::Compatible,
                ApiStability::Experimental,
                responses_evidence,
            ),
        ],
        ModelCatalog::new(profiles)?,
    )?;
    Ok(OpenAiCompatibleProfile::verified_chat_and_responses(
        provider_profile,
        endpoint,
        dialect,
    )?)
}

fn support_scope(
    provider: ProviderId,
    platform: PlatformId,
    protocol: &str,
    api_mode: &str,
) -> Result<SupportScope, InvalidId> {
    Ok(SupportScope::new(
        provider,
        platform,
        ModelFamily::Language,
        ProtocolId::new(protocol)?,
        ApiModeId::new(api_mode)?,
    ))
}

fn evidence(
    source: &str,
    contract: &str,
) -> Result<VerificationEvidence, MinimaxLanguageProfileError> {
    let verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
            .map_err(MinimaxLanguageProfileError::VerificationDate)?,
    );
    Ok(VerificationEvidence::new(
        OfficialSource::new(source)?,
        verified_at,
        ProtocolContractId::new(contract)?,
    ))
}

fn model_profile(
    model: &str,
    scope: &SupportScope,
    evidence: &VerificationEvidence,
) -> Result<ModelProfile, MinimaxLanguageProfileError> {
    Ok(ModelProfile::new(
        ModelId::new(model)?,
        scope.clone(),
        [ModelOperation::Generate, ModelOperation::Stream],
        ModelLifecycle::Active,
        evidence.clone(),
    )?)
}

#[derive(Debug, Clone, Copy)]
struct MinimaxMessagesProjection;

impl MessagesRequestProjection for MinimaxMessagesProjection {
    fn project(
        &self,
        context: &MessagesRequestProjectionContext<'_>,
        body: Value,
    ) -> Result<ProjectedMessagesRequest, Error> {
        ProjectedMessagesRequest::new(
            context.default_target().clone(),
            body,
            RequestHeaders::new(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct MinimaxMessagesPolicy;

impl MessagesRequestPolicy for MinimaxMessagesPolicy {
    fn prepare(
        &self,
        model: &ModelId,
        request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        validate_common_request(model, request)?;
        if options.top_k().is_some() {
            return Err(unsupported("MiniMax Messages does not support top_k"));
        }
        if options.output_effort().is_some() {
            return Err(unsupported(
                "MiniMax Messages does not support Anthropic output effort",
            ));
        }
        if options.fallbacks().is_some() {
            return Err(unsupported(
                "MiniMax Messages does not support server-side fallback chains",
            ));
        }
        if options.metadata().is_some() {
            return Err(unsupported("MiniMax Messages does not support metadata"));
        }
        if !options.extra().is_empty() {
            return Err(unsupported(
                "MiniMax Messages accepts only typed MiniMax request options",
            ));
        }
        match options.service_tier() {
            None | Some(MessagesServiceTier::Standard | MessagesServiceTier::Priority) => {}
            Some(_) => {
                return Err(unsupported(
                    "MiniMax supports only standard or priority service tier",
                ));
            }
        }
        validate_thinking(model, options.thinking())?;
        if !request.generation.stop_sequences.is_empty() {
            return Err(unsupported(
                "MiniMax Messages does not support stop_sequences",
            ));
        }
        if request.structured_output.is_some() {
            return Err(unsupported(
                "MiniMax Messages does not support structured output",
            ));
        }
        validate_tool_choice(request.tool_choice.as_ref(), true)?;
        Ok(MessagesRequestRequirements::new())
    }
}

#[derive(Debug, Clone)]
struct MinimaxChatPolicy {
    reasoning_details: WireFieldName,
}

impl ChatCodecPolicy for MinimaxChatPolicy {
    fn name(&self) -> &'static str {
        "minimax-chat-completions"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        validate_common_request(model, &request)?;
        if request
            .generation
            .temperature
            .is_some_and(|value| value > 2.0)
        {
            return Err(invalid("MiniMax Chat temperature must be between 0 and 2"));
        }
        if request.generation.seed.is_some() || !request.generation.stop_sequences.is_empty() {
            return Err(unsupported(
                "MiniMax Chat does not support seed or stop-sequence controls",
            ));
        }
        if request.structured_output.is_some() {
            return Err(unsupported(
                "MiniMax Chat does not support structured output",
            ));
        }
        validate_tool_choice(request.tool_choice.as_ref(), false)?;
        validate_openai_extra(model, &extra, false)?;
        extra.insert("reasoning_split".to_string(), Value::Bool(true));
        let dialect = dialect
            .with_replayable_reasoning_details_field(self.reasoning_details.clone())
            .map_err(|source| {
                Error::new(
                    ErrorKind::Configuration,
                    "MiniMax Chat reasoning replay dialect is invalid",
                )
                .with_source(source)
            })?;
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: RequestHeaders::new(),
            prompt_cache_breakpoints: Vec::new(),
            warnings: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MinimaxResponsesPolicy;

impl ResponsesCodecPolicy for MinimaxResponsesPolicy {
    fn name(&self) -> &'static str {
        "minimax-responses"
    }

    fn prepare(
        &self,
        model: &ModelId,
        request: LanguageRequest,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        validate_common_request(model, &request)?;
        if request
            .generation
            .temperature
            .is_some_and(|value| value > 1.0)
        {
            return Err(invalid(
                "MiniMax Responses temperature must be between 0 and 1",
            ));
        }
        if request.structured_output.is_some() {
            return Err(unsupported(
                "MiniMax Responses supports text output only, not JSON Schema",
            ));
        }
        validate_tool_choice(request.tool_choice.as_ref(), true)?;
        validate_openai_extra(model, &extra, true)?;
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers: RequestHeaders::new(),
            prompt_cache_breakpoints: Vec::new(),
            native_tools: Vec::new(),
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }

    fn encode_request(
        &self,
        scope: &siumai_core::ProviderScope,
        model: &ModelId,
        prepared: &PreparedResponsesCall,
        stream: bool,
    ) -> Result<Value, Error> {
        let media = ResponsesMediaDialect::native()
            .with_video_input(true)
            .with_file_input(false);
        let options = RequestEncodingOptions::new(stream)
            .with_extra(prepared.extra.clone())
            .with_media_dialect(media);
        encode_responses_request(scope, model, &prepared.request, &options)
    }
}

fn validate_common_request(model: &ModelId, request: &LanguageRequest) -> Result<(), Error> {
    let has_developer = request
        .messages
        .iter()
        .any(|message| message.role() == MessageRole::Developer);
    if has_developer {
        return Err(unsupported(
            "MiniMax language APIs do not support developer messages",
        ));
    }
    let mut has_media = false;
    for media in request.messages.iter().flat_map(|message| {
        message
            .content()
            .iter()
            .filter_map(|part| match part.content() {
                ContentPart::Media(media) => Some(media),
                _ => None,
            })
    }) {
        has_media = true;
        if !is_m3(model.as_str()) {
            return Err(unsupported(
                "only the verified MiniMax-M3 profile accepts media input",
            ));
        }
        if !media.media_type.starts_with("image/") && !media.media_type.starts_with("video/") {
            return Err(unsupported(
                "MiniMax-M3 accepts image or video media input only",
            ));
        }
    }
    if has_media && !is_m3(model.as_str()) {
        return Err(unsupported(
            "this MiniMax model does not support media input",
        ));
    }
    validate_output_limit(model, request.generation.max_output_tokens)
}

fn validate_output_limit(model: &ModelId, requested: Option<u64>) -> Result<(), Error> {
    let maximum = if is_m3(model.as_str()) {
        Some(M3_MAX_OUTPUT_TOKENS)
    } else if is_known_m2(model.as_str()) {
        Some(M2_MAX_OUTPUT_TOKENS)
    } else {
        None
    };
    if let Some(maximum) = maximum
        && requested.is_some_and(|requested| requested > maximum)
    {
        return Err(invalid(
            "max_output_tokens exceeds this MiniMax model limit",
        ));
    }
    Ok(())
}

fn validate_thinking(model: &ModelId, thinking: Option<ThinkingConfig>) -> Result<(), Error> {
    match thinking {
        Some(ThinkingConfig::Enabled { .. }) => Err(unsupported(
            "MiniMax hosted Messages supports adaptive or disabled thinking, not enabled",
        )),
        Some(ThinkingConfig::Disabled) if is_known_m2(model.as_str()) => Err(unsupported(
            "MiniMax M2 models cannot honor disabled thinking",
        )),
        Some(_) if !is_m3(model.as_str()) && !is_known_m2(model.as_str()) => Err(unsupported(
            "thinking controls are not inferred for unknown MiniMax model IDs",
        )),
        _ => Ok(()),
    }
}

fn validate_tool_choice(choice: Option<&ToolChoice>, auto_and_none: bool) -> Result<(), Error> {
    match choice {
        None => Ok(()),
        Some(ToolChoice::Auto | ToolChoice::None) if auto_and_none => Ok(()),
        Some(_) => Err(unsupported(if auto_and_none {
            "MiniMax supports only auto or none tool choice in this API mode"
        } else {
            "MiniMax Chat Completions does not support explicit tool_choice"
        })),
    }
}

fn validate_openai_extra(
    model: &ModelId,
    extra: &BTreeMap<String, Value>,
    responses: bool,
) -> Result<(), Error> {
    for (name, value) in extra {
        match name.as_str() {
            "thinking" if !responses => {
                let thinking =
                    serde_json::from_value::<crate::options::MinimaxThinking>(value.clone())
                        .map_err(|source| {
                            invalid_source("invalid MiniMax thinking option", source)
                        })?;
                validate_minimax_thinking(model, thinking)?;
            }
            "reasoning" if responses => {
                let reasoning =
                    serde_json::from_value::<crate::options::MinimaxResponsesReasoning>(
                        value.clone(),
                    )
                    .map_err(|source| invalid_source("invalid MiniMax reasoning option", source))?;
                if !is_m3(model.as_str()) && !is_known_m2(model.as_str()) {
                    return Err(unsupported(
                        "reasoning controls are not inferred for unknown MiniMax model IDs",
                    ));
                }
                if is_known_m2(model.as_str())
                    && reasoning.effort() == crate::options::MinimaxReasoningEffort::None
                {
                    return Err(unsupported(
                        "MiniMax M2 models cannot honor disabled reasoning",
                    ));
                }
            }
            "service_tier" => validate_service_tier(value)?,
            "prompt_cache_key" if responses => validate_string(value, "prompt_cache_key")?,
            "metadata" if responses => validate_string_map(value, "metadata")?,
            _ => {
                return Err(unsupported(
                    "MiniMax accepts only typed, mode-specific provider options",
                ));
            }
        }
    }
    Ok(())
}

fn validate_minimax_thinking(
    model: &ModelId,
    thinking: crate::options::MinimaxThinking,
) -> Result<(), Error> {
    if !is_m3(model.as_str()) && !is_known_m2(model.as_str()) {
        return Err(unsupported(
            "thinking controls are not inferred for unknown MiniMax model IDs",
        ));
    }
    if is_known_m2(model.as_str()) && thinking == crate::options::MinimaxThinking::Disabled {
        return Err(unsupported(
            "MiniMax M2 models cannot honor disabled thinking",
        ));
    }
    Ok(())
}

fn validate_service_tier(value: &Value) -> Result<(), Error> {
    if matches!(value.as_str(), Some("standard" | "priority")) {
        Ok(())
    } else {
        Err(invalid("MiniMax service_tier must be standard or priority"))
    }
}

fn validate_string(value: &Value, field: &'static str) -> Result<(), Error> {
    if value.as_str().is_some_and(|value| !value.is_empty()) {
        Ok(())
    } else {
        Err(invalid(match field {
            "prompt_cache_key" => "MiniMax prompt_cache_key must be a non-empty string",
            _ => "MiniMax option must be a non-empty string",
        }))
    }
}

fn validate_string_map(value: &Value, field: &'static str) -> Result<(), Error> {
    let Some(object) = value.as_object() else {
        return Err(invalid(match field {
            "metadata" => "MiniMax metadata must be a string-to-string object",
            _ => "MiniMax option must be an object",
        }));
    };
    if object.values().all(Value::is_string) {
        Ok(())
    } else {
        Err(invalid("MiniMax metadata values must all be strings"))
    }
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn unsupported(message: &'static str) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn invalid_source(message: &'static str, source: serde_json::Error) -> Error {
    Error::new(ErrorKind::InvalidInput, message).with_source(source)
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum MinimaxLanguageProfileError {
    #[error("invalid MiniMax identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid MiniMax support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid MiniMax model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid MiniMax Messages profile: {0}")]
    Messages(#[from] AnthropicCompatibleConfigError),
    #[error("invalid MiniMax OpenAI-compatible profile: {0}")]
    OpenAi(#[from] OpenAiCompatibleConfigError),
    #[error("invalid MiniMax Chat Completions dialect: {0}")]
    ChatDialect(#[from] DialectError),
    #[error("MiniMax OpenAI-compatible profile omitted Chat Completions mode")]
    MissingChatMode,
    #[error("invalid MiniMax verification date: {0}")]
    VerificationDate(#[source] chrono::ParseError),
    #[error("invalid MiniMax Messages encoding rule: {0}")]
    MessagesRule(#[from] siumai_protocol_anthropic::messages::MessagesEncodingRuleError),
}
