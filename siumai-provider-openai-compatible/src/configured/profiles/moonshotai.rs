//! Moonshot AI's verified current Kimi Chat Completions profile.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use serde_json::Value;
use siumai_core::{
    ApiModeId, ApiStability, ContentPart, Error, ErrorKind, LanguageRequest, MediaData,
    ModelCatalog, ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile,
    OfficialSource, PlatformId, ProfileId, ProtocolContractId, ProtocolId, ProviderId,
    ProviderProfile, SupportScope, ToolChoice, ToolSpec, TypedProviderOptions, VerificationDate,
    VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim, Warning,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID, ChatCompletionsDialect, MaxOutputTokensField, PROTOCOL_ID, WireFieldName,
};
use siumai_transport::{EndpointConfig, OfficialOrigin};

use crate::configured::codec_policy::{ChatCodecPolicy, PreparedChatCall};
use crate::provider_options::{KimiLanguageOptions, KimiThinking, KimiThinkingMode};
use crate::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};

pub const KIMI_K3: &str = "kimi-k3";
pub const KIMI_K2_7_CODE: &str = "kimi-k2.7-code";
pub const KIMI_K2_7_CODE_HIGHSPEED: &str = "kimi-k2.7-code-highspeed";
pub const KIMI_K2_6: &str = "kimi-k2.6";
pub const KIMI_K2_5: &str = "kimi-k2.5";
pub const CHAT: &str = KIMI_K3;
pub const VERIFIED_ON: &str = "2026-08-05";
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

pub fn profile() -> Result<OpenAiCompatibleProfile, OpenAiCompatibleConfigError> {
    let scope = SupportScope::new(
        ProviderId::new("moonshotai")?,
        PlatformId::new("kimi-public-api")?,
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
    let provider_profile = ProviderProfile::verified(
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
    .expect("Kimi static profile and catalog scopes match");
    let endpoint = EndpointConfig::official(
        "https://api.moonshot.ai/v1",
        OfficialOrigin::new("https://api.moonshot.ai")
            .expect("Kimi official origin is static and validated"),
    )?;
    let reasoning = WireFieldName::new("reasoning_content")
        .expect("Kimi reasoning field is static and validated");
    let cached_tokens = WireFieldName::new("cached_tokens")
        .expect("Kimi cache usage field is static and validated");
    let dialect = ChatCompletionsDialect::generic()
        .with_cache_read_tokens_field(cached_tokens)
        .with_stream_choice_usage(true)
        .with_max_output_tokens_field(MaxOutputTokensField::MaxCompletionTokens);
    Ok(
        OpenAiCompatibleProfile::verified(provider_profile, endpoint, dialect)?
            .with_codec_policy(Arc::new(KimiChatCodecPolicy { reasoning })),
    )
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

        let mut warnings = Vec::new();
        let Some(policy) = model_policy(model) else {
            return Ok(PreparedChatCall {
                request,
                dialect,
                extra,
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
            warnings,
        })
    }
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
        message.content.iter().filter_map(|part| match part {
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
        message.content.iter().any(|part| {
            matches!(
                part,
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
        message.content.iter().filter_map(|part| match part {
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
        let (name, description, mut schema) = tool.into_parts();
        if schema
            .as_object_mut()
            .is_some_and(|schema| schema.remove("$schema").is_some())
        {
            removed += 1;
        }
        normalized_tools.push(ToolSpec::new(name, description, schema).map_err(|source| {
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

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use serde_json::json;
    use siumai_core::{
        CallOptions, ErrorKind, GenerationConfig, LanguageModel, LanguageStreamEvent, Message,
        MessageRole, ModelPolicy, ModelPolicyContext, ProviderOptions, StreamTerminal,
        StructuredOutputSpec, SupportState, UsageValue,
    };
    use siumai_protocol_openai::chat_completions::encode_request;
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::configured::policy::OpenAiCompatibleModelPolicy;
    use crate::{OpenAiCompatibleCredential, OpenAiCompatibleProvider};

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    fn provider_for(base_url: String) -> OpenAiCompatibleProvider {
        let profile = profile()
            .unwrap()
            .with_test_endpoint(EndpointConfig::local_explicit(base_url).unwrap());
        OpenAiCompatibleProvider::builder(profile, OpenAiCompatibleCredential::unauthenticated())
            .build()
            .unwrap()
    }

    fn normalize(
        model: &str,
        request: LanguageRequest,
        options: KimiLanguageOptions,
    ) -> Result<PreparedChatCall, Error> {
        let profile = profile().unwrap();
        let extra = ProviderOptions::typed(&options)
            .unwrap()
            .value()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        profile.codec_policy().prepare(
            &ModelId::new(model).unwrap(),
            request,
            profile.dialect().clone(),
            extra,
        )
    }

    #[test]
    fn profile_has_exact_evidence_current_lifecycle_and_open_unknown_ids() {
        let profile = profile().unwrap();
        let claims = profile.provider_profile().verified_claims().unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence().source().as_str(), OFFICIAL_SOURCE);
        assert!(
            profile
                .provider_profile()
                .catalog()
                .unwrap()
                .iter()
                .all(|model| model.evidence().source().as_str() == MODEL_SOURCE)
        );
        assert_eq!(profile.scope().provider_id().as_str(), "moonshotai");
        assert_eq!(
            profile.scope().platform().unwrap().as_str(),
            "kimi-public-api"
        );
        assert_eq!(profile.scope().api_mode().unwrap().as_str(), API_MODE_ID);

        let policy = OpenAiCompatibleModelPolicy::new(
            profile.profile_arc(),
            profile.support_scope().clone(),
        );
        for current in [KIMI_K3, KIMI_K2_7_CODE, KIMI_K2_7_CODE_HIGHSPEED, KIMI_K2_6] {
            let decision = policy.evaluate(&ModelPolicyContext {
                scope: profile.scope().clone(),
                model: ModelId::new(current).unwrap(),
                family: ModelFamily::Language,
                operation: ModelOperation::Generate,
            });
            assert_eq!(decision.state(), &SupportState::Supported);
        }
        let retired = policy.evaluate(&ModelPolicyContext {
            scope: profile.scope().clone(),
            model: ModelId::new("kimi-k2-thinking").unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
        });
        assert!(matches!(retired.state(), SupportState::Unsupported { .. }));
        let unknown = policy.evaluate(&ModelPolicyContext {
            scope: profile.scope().clone(),
            model: ModelId::new("kimi-k4-future").unwrap(),
            family: ModelFamily::Language,
            operation: ModelOperation::Generate,
        });
        assert_eq!(unknown.state(), &SupportState::Unknown);
    }

    #[tokio::test]
    async fn configured_provider_sends_typed_k3_options_and_decodes_cache_usage() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"kimi-k3\""#.to_string()),
                mockito::Matcher::Regex(r#"\"reasoning_effort\":\"high\""#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":false"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"chat-1","model":"kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"why"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cached_tokens":7}}"#,
            )
            .create_async()
            .await;
        let provider = provider_for(format!("{}/v1", server.url()));
        let model = provider.language(KIMI_K3).unwrap();
        let options = CallOptions::default().with_provider_options(
            ProviderOptions::typed(
                &KimiLanguageOptions::new()
                    .with_reasoning_effort(crate::provider_options::KimiReasoningEffort::High),
            )
            .unwrap(),
        );

        let response = model.generate(request(), options).await.unwrap();

        assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(7));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_provider_decodes_choice_usage_into_one_stream_terminal() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "data: {\"id\":\"chat-1\",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\",\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"cached_tokens\":7}}]}\n\n",
                "data: [DONE]\n\n"
            ))
            .create_async()
            .await;
        let provider = provider_for(format!("{}/v1", server.url()));
        let model = provider.language(KIMI_K3).unwrap();

        let events = model
            .stream(request(), CallOptions::default())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(
            events
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        let response = events.iter().find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => {
                Some(response.as_ref())
            }
            _ => None,
        });
        assert_eq!(
            response.unwrap().usage().cache_read_tokens,
            UsageValue::Known(7)
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_provider_preserves_kimi_error_diagnostics() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "kimi-request-42")
            .with_header("retry-after", "2")
            .with_body(
                r#"{"error":{"message":"invalid thinking configuration","type":"invalid_request_error","param":"thinking.keep","code":"invalid_parameter"}}"#,
            )
            .create_async()
            .await;
        let provider = provider_for(format!("{}/v1", server.url()));
        let model = provider.language(KIMI_K3).unwrap();

        let error = model
            .generate(request(), CallOptions::default())
            .await
            .unwrap_err();
        let diagnostics = error.diagnostics().unwrap();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(diagnostics.status(), Some(400));
        assert_eq!(diagnostics.provider_code(), Some("invalid_parameter"));
        assert_eq!(diagnostics.provider_type(), Some("invalid_request_error"));
        assert_eq!(diagnostics.provider_param(), Some("thinking.keep"));
        assert_eq!(diagnostics.request_id(), Some("kimi-request-42"));
        assert_eq!(
            diagnostics.retry_after(),
            Some(std::time::Duration::from_secs(2))
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_provider_rejects_k2_5_video_before_transport() {
        let provider = provider_for("http://127.0.0.1:9/v1".to_string());
        let model = provider.language(KIMI_K2_5).unwrap();
        let mut video_request = request();
        video_request.messages[0]
            .content
            .push(ContentPart::Media(siumai_core::MediaPart {
                media_type: "video/mp4".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3].into()),
                name: None,
            }));

        let error = model
            .generate(video_request, CallOptions::default())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn k3_uses_reasoning_effort_max_completion_tokens_and_strict_tools() {
        let mut request = request().with_generation(GenerationConfig {
            max_output_tokens: Some(131_072),
            ..GenerationConfig::default()
        });
        request.tools.push(
            ToolSpec::new(
                "lookup",
                Some("Look up a value".to_string()),
                json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
            )
            .unwrap(),
        );
        request.structured_output = Some(StructuredOutputSpec {
            name: "answer".to_string(),
            description: None,
            schema: json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
            strict: true,
        });
        let prepared = normalize(
            KIMI_K3,
            request,
            KimiLanguageOptions::new()
                .with_reasoning_effort(crate::provider_options::KimiReasoningEffort::High),
        )
        .unwrap();
        let body = encode_request(
            &ModelId::new(KIMI_K3).unwrap(),
            &prepared.request,
            false,
            &prepared.dialect,
            &prepared.extra,
        )
        .unwrap();

        assert_eq!(body["max_completion_tokens"], 131_072);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["tools"][0]["function"]["strict"], true);
        assert!(
            body["tools"][0]["function"]["parameters"]
                .get("$schema")
                .is_none()
        );
        assert!(
            body["response_format"]["json_schema"]["schema"]
                .get("$schema")
                .is_none()
        );
        assert_eq!(prepared.warnings.len(), 1);
    }

    #[test]
    fn model_specific_thinking_sampling_and_tool_rules_are_enforced() {
        assert!(
            normalize(
                KIMI_K3,
                request().with_generation(GenerationConfig {
                    temperature: Some(1.0),
                    ..GenerationConfig::default()
                }),
                KimiLanguageOptions::new(),
            )
            .is_err()
        );
        assert!(
            normalize(
                KIMI_K2_7_CODE,
                request(),
                KimiLanguageOptions::new()
                    .with_thinking(crate::provider_options::KimiThinking::disabled()),
            )
            .is_err()
        );

        let mut required = request();
        required
            .tools
            .push(ToolSpec::new("lookup", None, json!({"type": "object"})).unwrap());
        required.tool_choice = Some(ToolChoice::Required);
        assert!(normalize(KIMI_K2_6, required.clone(), KimiLanguageOptions::new(),).is_err());
        assert!(
            normalize(
                KIMI_K2_6,
                required.with_generation(GenerationConfig {
                    temperature: Some(0.6),
                    top_p: Some(0.95),
                    ..GenerationConfig::default()
                }),
                KimiLanguageOptions::new()
                    .with_thinking(crate::provider_options::KimiThinking::disabled()),
            )
            .is_err()
        );
        assert!(
            normalize(
                KIMI_K2_6,
                request().with_generation(GenerationConfig {
                    temperature: Some(0.6),
                    top_p: Some(0.95),
                    ..GenerationConfig::default()
                }),
                KimiLanguageOptions::new()
                    .with_thinking(crate::provider_options::KimiThinking::disabled()),
            )
            .is_ok()
        );
    }

    #[test]
    fn deprecated_v1_and_k2_5_models_keep_their_narrow_wire_policies() {
        let mut image_request = request();
        image_request.messages[0]
            .content
            .push(ContentPart::Media(siumai_core::MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3].into()),
                name: None,
            }));
        assert!(
            normalize(
                "moonshot-v1-8k",
                image_request.clone(),
                KimiLanguageOptions::new()
            )
            .is_err()
        );
        assert!(
            normalize(
                "moonshot-v1-8k-vision-preview",
                image_request,
                KimiLanguageOptions::new(),
            )
            .is_ok()
        );

        let mut video_request = request();
        video_request.messages[0]
            .content
            .push(ContentPart::Media(siumai_core::MediaPart {
                media_type: "video/mp4".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3].into()),
                name: None,
            }));
        assert!(normalize(KIMI_K2_5, video_request, KimiLanguageOptions::new()).is_err());

        assert!(
            normalize(
                KIMI_K2_5,
                request(),
                KimiLanguageOptions::new().with_thinking(
                    crate::provider_options::KimiThinking::enabled_with_preserved_history(),
                ),
            )
            .is_err()
        );

        assert!(
            normalize(
                KIMI_K2_5,
                request().with_generation(GenerationConfig {
                    temperature: Some(0.6),
                    top_p: Some(0.5),
                    ..GenerationConfig::default()
                }),
                KimiLanguageOptions::new()
                    .with_thinking(crate::provider_options::KimiThinking::disabled()),
            )
            .is_ok()
        );

        let prepared = normalize(KIMI_K2_5, request(), KimiLanguageOptions::new()).unwrap();
        assert_eq!(
            prepared.dialect.reasoning_output_field(),
            Some("reasoning_content")
        );
    }

    #[test]
    fn unsupported_seed_and_invalid_tool_names_are_rejected_before_encoding() {
        assert!(
            normalize(
                KIMI_K3,
                request().with_generation(GenerationConfig {
                    seed: Some(7),
                    ..GenerationConfig::default()
                }),
                KimiLanguageOptions::new(),
            )
            .is_err()
        );

        let mut invalid_tool = request();
        invalid_tool
            .tools
            .push(ToolSpec::new("1bad", None, json!({"type": "object"})).unwrap());
        assert!(normalize(KIMI_K3, invalid_tool, KimiLanguageOptions::new()).is_err());
    }

    #[test]
    fn unknown_models_receive_no_model_specific_rewrite_or_default() {
        let profile = profile().unwrap();
        let extra = BTreeMap::from([(
            "future_reasoning_mode".to_string(),
            Value::String("new".to_string()),
        )]);
        let prepared = profile
            .codec_policy()
            .prepare(
                &ModelId::new("kimi-k4-future").unwrap(),
                request(),
                profile.dialect().clone(),
                extra.clone(),
            )
            .unwrap();

        assert_eq!(prepared.extra, extra);
        assert!(prepared.warnings.is_empty());
        assert!(prepared.request.generation.temperature.is_none());
        assert!(!prepared.dialect.supports_video_input());
        assert_eq!(prepared.dialect.reasoning_input_field(), None);
        assert_eq!(prepared.dialect.reasoning_output_field(), None);

        let mut tool_request = request();
        tool_request
            .tools
            .push(ToolSpec::new("lookup", None, json!({"type": "object"})).unwrap());
        let prepared = profile
            .codec_policy()
            .prepare(
                &ModelId::new("kimi-k4-future").unwrap(),
                tool_request,
                profile.dialect().clone(),
                BTreeMap::new(),
            )
            .unwrap();
        let body = encode_request(
            &ModelId::new("kimi-k4-future").unwrap(),
            &prepared.request,
            false,
            &prepared.dialect,
            &prepared.extra,
        )
        .unwrap();
        assert!(body["tools"][0]["function"].get("strict").is_none());
    }
}
