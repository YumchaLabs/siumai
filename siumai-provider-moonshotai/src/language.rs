//! Moonshot AI's verified current Kimi Chat Completions profile.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use serde_json::Value;
use siumai_core::{
    ApiModeId, ApiStability, ContentPart, Error, ErrorKind, LanguageRequest, MediaData,
    MessageRole, ModelCatalog, ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile,
    OfficialSource, PlatformId, ProfileId, ProtocolContractId, ProtocolId, ProviderId,
    ProviderProfile, ReplayDomain, SupportScope, TypedProviderOptions, VerificationDate,
    VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID, ChatCompletionsDialect, MaxOutputTokensField, PROTOCOL_ID, WireFieldName,
};
use siumai_transport::EndpointConfig;

use siumai_openai_compatible::extension::v1::{ChatCodecPolicy, PreparedChatCall};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};

use crate::annotations::KimiAssistantPartial;
use crate::options::KimiLanguageOptions;

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

#[derive(Debug)]
struct KimiChatCodecPolicy {
    reasoning: WireFieldName,
}

impl ChatCodecPolicy for KimiChatCodecPolicy {
    fn name(&self) -> &'static str {
        "kimi-chat-codec"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        validate_api_limits(&request)?;
        validate_media_sources(&request)?;
        kimi_partial_index(&request)?;
        validate_schema_annotations(&request)?;
        parse_options(&extra)?;
        let dialect = dialect
            .with_video_input(true)
            .with_reasoning_input_field(self.reasoning.clone())
            .with_reasoning_output_field(self.reasoning.clone())
            .with_function_tool_strict(true);

        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: siumai_transport::RequestHeaders::new(),
            prompt_cache_resolver: None,
            warnings: Vec::new(),
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

fn validate_schema_annotations(request: &LanguageRequest) -> Result<(), Error> {
    let has_schema_annotation = request
        .structured_output
        .as_ref()
        .and_then(|output| output.schema.as_object())
        .is_some_and(|schema| schema.contains_key("$schema"))
        || request.tools.iter().any(|tool| {
            tool.input_schema()
                .as_object()
                .is_some_and(|schema| schema.contains_key("$schema"))
        });
    if has_schema_annotation {
        return Err(invalid(
            "Kimi's schema subset does not accept top-level $schema annotations",
        ));
    }
    Ok(())
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
