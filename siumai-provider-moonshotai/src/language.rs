//! Moonshot AI's verified current Kimi Chat Completions profile.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use serde_json::Value;
use siumai_core::{
    ApiModeId, ApiStability, ContentPart, Error, ErrorKind, LanguageRequest, MediaData,
    MessageRole, ModelCatalog, ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile,
    OfficialSource, PlatformId, ProfileId, ProtocolContractId, ProtocolId, ProviderId,
    ProviderProfile, ReplayDomain, SupportScope, ToolChoice, ToolSpec, TypedProviderOptions,
    VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim, Warning,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID, ChatCompletionsDialect, MaxOutputTokensField, PROTOCOL_ID, WireFieldName,
};
use siumai_transport::EndpointConfig;

use siumai_openai_compatible::extension::v1::{ChatCodecPolicy, PreparedChatCall};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};

use crate::annotations::KimiAssistantPartial;
use crate::options::{KimiLanguageOptions, KimiThinking, KimiThinkingMode};

pub const PROVIDER_ID: &str = "moonshotai";
pub const PLATFORM_ID: &str = "kimi-public-api";
pub const DEFAULT_BASE_URL: &str = "https://api.moonshot.ai/v1";
pub const KIMI_K3: &str = "kimi-k3";
pub const KIMI_K2_7_CODE: &str = "kimi-k2.7-code";
pub const KIMI_K2_7_CODE_HIGHSPEED: &str = "kimi-k2.7-code-highspeed";
pub const KIMI_K2_6: &str = "kimi-k2.6";
pub const KIMI_K2_5: &str = "kimi-k2.5";
pub const CHAT: &str = KIMI_K3;
pub const VERIFIED_ON: &str = "2026-08-08";
pub const API_OVERVIEW_SOURCE: &str = "https://platform.kimi.ai/docs/api/overview";
pub const OFFICIAL_SOURCE: &str = "https://platform.kimi.ai/docs/api/chat";
pub const MODEL_SOURCE: &str = "https://platform.kimi.ai/docs/models";

const RETIRED_MODELS: &[&str] = &[
    "kimi-k2-0905-preview",
    "kimi-k2-0711-preview",
    "kimi-k2-turbo-preview",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
    "kimi-latest",
    "kimi-thinking-preview",
];

const MOONSHOT_V1_TEXT_MODELS: &[&str] = &["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"];

const MOONSHOT_V1_VISION_MODELS: &[&str] = &[
    "moonshot-v1-8k-vision-preview",
    "moonshot-v1-32k-vision-preview",
    "moonshot-v1-128k-vision-preview",
];

pub(crate) fn profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, OpenAiCompatibleConfigError> {
    let reasoning = WireFieldName::new("reasoning_content")
        .expect("Kimi reasoning field is a valid static wire name");
    let cached_tokens = WireFieldName::new("cached_tokens")
        .expect("Kimi cache usage field is a valid static wire name");
    let dialect = ChatCompletionsDialect::generic()
        .with_cache_read_tokens_field(cached_tokens)
        .with_stream_choice_usage(true)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens);
    let profile = if verified_endpoint {
        OpenAiCompatibleProfile::verified_chat(verified_profile()?, endpoint, dialect)?
            .with_replay_domain(replay_domain)?
    } else {
        OpenAiCompatibleProfile::custom_chat(
            ProviderId::new(PROVIDER_ID)?,
            endpoint,
            replay_domain,
            dialect,
        )?
    };

    Ok(profile.with_chat_codec_policy(Arc::new(KimiChatCodecPolicy { reasoning })))
}

fn verified_profile() -> Result<ProviderProfile, OpenAiCompatibleConfigError> {
    let scope = SupportScope::new(
        ProviderId::new(PROVIDER_ID)?,
        PlatformId::new(PLATFORM_ID)?,
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID)?,
        ApiModeId::new(API_MODE_ID)?,
    );
    let claim_evidence = VerificationEvidence::new(
        OfficialSource::new(OFFICIAL_SOURCE).expect("Kimi source URL is static and validated"),
        VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .expect("Kimi verification date is static and validated"),
        ),
        ProtocolContractId::new("moonshotai-kimi-openai-chat-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        OfficialSource::new(MODEL_SOURCE).expect("Kimi model source URL is static and validated"),
        VerificationDate::new(
            NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")
                .expect("Kimi verification date is static and validated"),
        ),
        ProtocolContractId::new("moonshotai-kimi-model-lifecycle-2026-08")?,
    );
    let replacement = ModelId::new(KIMI_K3)?;
    let model = |id: &str, lifecycle: ModelLifecycle| {
        ModelProfile::new(
            ModelId::new(id).expect("Kimi model constants are valid"),
            scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle,
            model_evidence.clone(),
        )
        .expect("Kimi model declarations have operations")
    };
    let mut models = vec![
        model(KIMI_K3, ModelLifecycle::Active),
        model(KIMI_K2_7_CODE, ModelLifecycle::Active),
        model(KIMI_K2_7_CODE_HIGHSPEED, ModelLifecycle::Active),
        model(KIMI_K2_6, ModelLifecycle::Active),
        model(
            KIMI_K2_5,
            ModelLifecycle::Deprecated {
                replacement: Some(replacement.clone()),
            },
        ),
    ];
    models.extend(
        MOONSHOT_V1_TEXT_MODELS
            .iter()
            .chain(MOONSHOT_V1_VISION_MODELS)
            .map(|id| {
                model(
                    id,
                    ModelLifecycle::Deprecated {
                        replacement: Some(replacement.clone()),
                    },
                )
            }),
    );
    models.extend(RETIRED_MODELS.iter().map(|id| {
        model(
            id,
            ModelLifecycle::Retired {
                replacement: Some(replacement.clone()),
            },
        )
    }));
    Ok(ProviderProfile::verified(
        ProfileId::new("moonshotai-kimi")?,
        vec![VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Compatible,
            ApiStability::Stable,
            claim_evidence,
        )],
        ModelCatalog::new(models)
            .expect("Kimi static catalog has no duplicate or invalid replacement rows"),
    )
    .expect("Kimi static profile and catalog scopes match"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KimiModelPolicy {
    K3,
    K2_7,
    K2_6,
    K2_5,
    LegacyText,
    LegacyVision,
}

fn model_policy(model: &ModelId) -> Option<KimiModelPolicy> {
    match model.as_str() {
        KIMI_K3 => Some(KimiModelPolicy::K3),
        KIMI_K2_7_CODE | KIMI_K2_7_CODE_HIGHSPEED => Some(KimiModelPolicy::K2_7),
        KIMI_K2_6 => Some(KimiModelPolicy::K2_6),
        KIMI_K2_5 => Some(KimiModelPolicy::K2_5),
        value if MOONSHOT_V1_TEXT_MODELS.contains(&value) => Some(KimiModelPolicy::LegacyText),
        value if MOONSHOT_V1_VISION_MODELS.contains(&value) => Some(KimiModelPolicy::LegacyVision),
        _ => None,
    }
}

#[derive(Debug)]
struct KimiChatCodecPolicy {
    reasoning: WireFieldName,
}

impl ChatCodecPolicy for KimiChatCodecPolicy {
    fn name(&self) -> &'static str {
        "kimi-current-model-policy"
    }

    fn prepare(
        &self,
        model: &ModelId,
        mut request: LanguageRequest,
        mut dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        validate_api_limits(&request)?;
        validate_media_sources(&request)?;
        kimi_partial_index(&request)?;

        let mut warnings = Vec::new();
        let Some(policy) = model_policy(model) else {
            return Ok(PreparedChatCall {
                request,
                dialect,
                extra,
                headers: siumai_transport::RequestHeaders::new(),
                prompt_cache_breakpoints: Vec::new(),
                warnings,
            });
        };

        dialect = match policy {
            KimiModelPolicy::K3 | KimiModelPolicy::K2_7 | KimiModelPolicy::K2_6 => dialect
                .with_video_input(true)
                .with_reasoning_input_field(self.reasoning.clone())
                .with_reasoning_output_field(self.reasoning.clone())
                .with_function_tool_strict(true),
            KimiModelPolicy::K2_5 => dialect
                .with_reasoning_input_field(self.reasoning.clone())
                .with_reasoning_output_field(self.reasoning.clone())
                .with_function_tool_strict(true),
            KimiModelPolicy::LegacyText | KimiModelPolicy::LegacyVision => dialect,
        };

        let options = parse_options(&extra)?;
        match policy {
            KimiModelPolicy::K3 => validate_k3(&request, &options, &extra)?,
            KimiModelPolicy::K2_7 => validate_k2_7(&request, &options, &extra)?,
            KimiModelPolicy::K2_6 => validate_k2_6(&request, &options, &extra)?,
            KimiModelPolicy::K2_5 => validate_k2_5(&request, &options, &extra)?,
            KimiModelPolicy::LegacyText => validate_legacy(&request, &options, false)?,
            KimiModelPolicy::LegacyVision => validate_legacy(&request, &options, true)?,
        }
        normalize_schema_annotations(&mut request, &mut warnings)?;

        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: siumai_transport::RequestHeaders::new(),
            prompt_cache_breakpoints: Vec::new(),
            warnings,
        })
    }

    fn encode_request(
        &self,
        scope: &siumai_core::ProviderScope,
        model: &ModelId,
        prepared: &PreparedChatCall,
        stream: bool,
    ) -> Result<Value, Error> {
        let mut body = prepared.encode(scope, model, stream)?;
        if kimi_partial_index(&prepared.request)?.is_some() {
            mark_kimi_partial(&mut body)?;
        }
        Ok(body)
    }
}

fn kimi_partial_index(request: &LanguageRequest) -> Result<Option<usize>, Error> {
    let mut partial_index = None;
    for (index, message) in request.messages.iter().enumerate() {
        let annotation = message
            .annotations()
            .decode::<KimiAssistantPartial>()
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "Kimi Partial Mode annotation is invalid",
                )
                .with_source(source)
            })?;
        if annotation.is_none() {
            continue;
        }
        if partial_index.replace(index).is_some() {
            return Err(invalid(
                "Kimi Partial Mode accepts exactly one annotated assistant message",
            ));
        }
        if message.role() != MessageRole::Assistant {
            return Err(invalid(
                "Kimi Partial Mode must annotate an assistant message",
            ));
        }
        if message.content().len() != 1
            || !matches!(message.content()[0].content(), ContentPart::Text { text } if !text.is_empty())
        {
            return Err(invalid(
                "Kimi Partial Mode requires one non-empty assistant text prefix",
            ));
        }
    }
    if partial_index.is_some_and(|index| index + 1 != request.messages.len()) {
        return Err(invalid("Kimi Partial Mode must annotate the final message"));
    }
    Ok(partial_index)
}

fn mark_kimi_partial(body: &mut Value) -> Result<(), Error> {
    let message = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
        .and_then(Value::as_object_mut)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "Kimi Chat encoding omitted the final Partial Mode message",
            )
        })?;
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err(Error::new(
            ErrorKind::Internal,
            "Kimi Chat encoding changed the Partial Mode message role",
        ));
    }
    message.insert("partial".to_string(), Value::Bool(true));
    Ok(())
}

fn parse_options(extra: &BTreeMap<String, Value>) -> Result<KimiLanguageOptions, Error> {
    let object = extra
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let options: KimiLanguageOptions =
        serde_json::from_value(Value::Object(object)).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "Kimi provider options have an invalid wire shape",
            )
            .with_source(source)
        })?;
    options.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "Kimi provider options are invalid").with_source(source)
    })?;
    Ok(options)
}

fn validate_api_limits(request: &LanguageRequest) -> Result<(), Error> {
    if request.generation.seed.is_some() {
        return Err(invalid("Kimi Chat Completions does not support seed"));
    }
    if request.generation.max_output_tokens == Some(0) {
        return Err(invalid("Kimi max output tokens must be greater than zero"));
    }
    if request.generation.stop_sequences.len() > 5 {
        return Err(invalid("Kimi accepts at most five stop sequences"));
    }
    if request
        .generation
        .stop_sequences
        .iter()
        .any(|sequence| sequence.len() > 32)
    {
        return Err(invalid("Kimi stop sequences must not exceed 32 bytes"));
    }
    if request.tools.len() > 128 {
        return Err(invalid("Kimi accepts at most 128 function tools"));
    }
    if request
        .tools
        .iter()
        .any(|tool| !is_valid_kimi_tool_name(tool.name()))
    {
        return Err(invalid(
            "Kimi function-tool names must match ^[a-zA-Z_][a-zA-Z0-9-_]{2,63}$",
        ));
    }
    if request
        .tools
        .iter()
        .any(|tool| !tool.input_schema().is_object())
    {
        return Err(invalid(
            "Kimi function-tool schemas must have an object root",
        ));
    }
    if request
        .structured_output
        .as_ref()
        .is_some_and(|output| !output.schema.is_object())
    {
        return Err(invalid(
            "Kimi structured-output schemas must have an object root",
        ));
    }
    Ok(())
}

fn is_valid_kimi_tool_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    (3..=64).contains(&bytes.len())
        && (bytes[0].is_ascii_alphabetic() || bytes[0] == b'_')
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn validate_media_sources(request: &LanguageRequest) -> Result<(), Error> {
    for media in request.messages.iter().flat_map(|message| {
        message
            .content()
            .iter()
            .filter_map(|part| match part.content() {
                ContentPart::Media(media) => Some(media),
                _ => None,
            })
    }) {
        if let MediaData::Url(url) = &media.data
            && !url.starts_with("data:")
            && !url.starts_with("ms://")
        {
            return Err(invalid(
                "Kimi media input requires inline data or an ms:// file reference",
            ));
        }
    }
    Ok(())
}

fn validate_k3(
    request: &LanguageRequest,
    options: &KimiLanguageOptions,
    extra: &BTreeMap<String, Value>,
) -> Result<(), Error> {
    if options.thinking.is_some() {
        return Err(invalid(
            "Kimi K3 always thinks; use reasoning_effort instead of thinking",
        ));
    }
    if request.generation.temperature.is_some()
        || request.generation.top_p.is_some()
        || extra.contains_key("presence_penalty")
        || extra.contains_key("frequency_penalty")
    {
        return Err(invalid(
            "Kimi K3 uses fixed sampling values; omit sampling and penalty fields",
        ));
    }
    if request
        .generation
        .max_output_tokens
        .is_some_and(|value| value > 1_048_576)
    {
        return Err(invalid(
            "Kimi K3 max_completion_tokens must not exceed 1048576",
        ));
    }
    if matches!(request.tool_choice, Some(ToolChoice::Named { .. })) {
        return Err(invalid(
            "Kimi K3 cannot force a named function while thinking is enabled",
        ));
    }
    Ok(())
}

fn validate_k2_7(
    request: &LanguageRequest,
    options: &KimiLanguageOptions,
    extra: &BTreeMap<String, Value>,
) -> Result<(), Error> {
    if options.reasoning_effort.is_some() {
        return Err(invalid("reasoning_effort is supported only by Kimi K3"));
    }
    if options
        .thinking
        .as_ref()
        .is_some_and(|thinking| matches!(thinking, KimiThinking::Disabled {}))
    {
        return Err(invalid("Kimi K2.7 thinking cannot be disabled"));
    }
    validate_fixed_number(request.generation.temperature, 1.0)?;
    validate_fixed_number(request.generation.top_p, 0.95)?;
    validate_fixed_extra(extra, "presence_penalty", 0.0)?;
    validate_fixed_extra(extra, "frequency_penalty", 0.0)?;
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Required | ToolChoice::Named { .. })
    ) {
        return Err(invalid("Kimi K2.7 tool_choice supports only auto or none"));
    }
    Ok(())
}

fn validate_k2_6(
    request: &LanguageRequest,
    options: &KimiLanguageOptions,
    extra: &BTreeMap<String, Value>,
) -> Result<(), Error> {
    if options.reasoning_effort.is_some() {
        return Err(invalid("reasoning_effort is supported only by Kimi K3"));
    }
    let thinking = options
        .thinking
        .as_ref()
        .map_or(KimiThinkingMode::Enabled, KimiThinking::mode);
    let expected_temperature = if thinking == KimiThinkingMode::Enabled {
        1.0
    } else {
        0.6
    };
    validate_fixed_number(request.generation.temperature, expected_temperature)?;
    validate_fixed_number(request.generation.top_p, 0.95)?;
    validate_fixed_extra(extra, "presence_penalty", 0.0)?;
    validate_fixed_extra(extra, "frequency_penalty", 0.0)?;
    if matches!(request.tool_choice, Some(ToolChoice::Required)) {
        return Err(invalid("Kimi K2.6 does not support required tool_choice"));
    }
    if thinking == KimiThinkingMode::Enabled
        && matches!(request.tool_choice, Some(ToolChoice::Named { .. }))
    {
        return Err(invalid("thinking Kimi K2.6 cannot force a named function"));
    }
    Ok(())
}

fn validate_k2_5(
    request: &LanguageRequest,
    options: &KimiLanguageOptions,
    _extra: &BTreeMap<String, Value>,
) -> Result<(), Error> {
    if options.reasoning_effort.is_some() {
        return Err(invalid("reasoning_effort is supported only by Kimi K3"));
    }
    if options
        .thinking
        .as_ref()
        .and_then(KimiThinking::retention)
        .is_some()
    {
        return Err(invalid("Kimi K2.5 thinking does not support keep"));
    }
    if request.messages.iter().any(|message| {
        message.content().iter().any(|part| {
            matches!(
                part.content(),
                ContentPart::Media(media) if media.media_type.starts_with("video/")
            )
        })
    }) {
        return Err(invalid(
            "Kimi K2.5 supports image input but not video input",
        ));
    }
    let thinking = options
        .thinking
        .as_ref()
        .map_or(KimiThinkingMode::Enabled, KimiThinking::mode);
    let expected_temperature = if thinking == KimiThinkingMode::Enabled {
        1.0
    } else {
        0.6
    };
    validate_fixed_number(request.generation.temperature, expected_temperature)?;
    Ok(())
}

fn validate_legacy(
    request: &LanguageRequest,
    options: &KimiLanguageOptions,
    supports_images: bool,
) -> Result<(), Error> {
    if options.reasoning_effort.is_some() || options.thinking.is_some() {
        return Err(invalid(
            "Moonshot V1 does not accept current Kimi reasoning options",
        ));
    }
    for media in request.messages.iter().flat_map(|message| {
        message
            .content()
            .iter()
            .filter_map(|part| match part.content() {
                ContentPart::Media(media) => Some(media),
                _ => None,
            })
    }) {
        if !supports_images || !media.media_type.starts_with("image/") {
            return Err(invalid(
                "the selected Moonshot V1 model does not support this media input",
            ));
        }
    }
    Ok(())
}

fn validate_fixed_number(actual: Option<f64>, expected: f64) -> Result<(), Error> {
    if actual.is_some_and(|actual| actual != expected) {
        return Err(invalid(
            "Kimi sampling value conflicts with the selected model and thinking mode",
        ));
    }
    Ok(())
}

fn validate_fixed_extra(
    extra: &BTreeMap<String, Value>,
    field: &'static str,
    expected: f64,
) -> Result<(), Error> {
    let Some(value) = extra.get(field) else {
        return Ok(());
    };
    if value.as_f64() != Some(expected) {
        return Err(invalid(
            "Kimi penalty value conflicts with the selected model policy",
        ));
    }
    Ok(())
}

fn normalize_schema_annotations(
    request: &mut LanguageRequest,
    warnings: &mut Vec<Warning>,
) -> Result<(), Error> {
    let mut removed = 0usize;
    if let Some(output) = &mut request.structured_output
        && output
            .schema
            .as_object_mut()
            .is_some_and(|schema| schema.remove("$schema").is_some())
    {
        removed += 1;
    }
    let mut normalized_tools = Vec::with_capacity(request.tools.len());
    for tool in std::mem::take(&mut request.tools) {
        let mut parts = tool.into_parts();
        if parts
            .input_schema
            .as_object_mut()
            .is_some_and(|schema| schema.remove("$schema").is_some())
        {
            removed += 1;
        }
        normalized_tools.push(ToolSpec::from_parts(parts).map_err(|source| {
            Error::new(
                ErrorKind::Internal,
                "Kimi schema normalization produced an invalid tool definition",
            )
            .with_source(source)
        })?);
    }
    request.tools = normalized_tools;
    if removed > 0 {
        warnings.push(Warning::provider(
            "schema_annotation_removed",
            "Kimi uses an MFJS-compatible schema subset; top-level $schema annotations were removed",
        ));
    }
    Ok(())
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
