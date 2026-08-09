//! Verified Volcengine ARK Chat Completions and Responses profile.
//!
//! `volcengine` is the provider identity, ARK is the technical platform, and Doubao is one model
//! brand hosted by that platform. Keeping those concepts separate preserves identity when callers
//! use non-Doubao deployment identifiers or future model IDs.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use http::header::{HeaderName, HeaderValue};
use serde_json::Value;
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, Error, ErrorKind, LanguageRequest, ModelCatalog,
    ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId,
    ProfileError, ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile,
    ReplayDomain, SupportScope, TypedProviderOptions, VerificationDate, VerificationEvidence,
    VerifiedFidelity, VerifiedSupportClaim, Warning,
};
use siumai_openai_compatible::extension::v1::{
    ChatCodecPolicy, PreparedChatCall, PreparedResponsesCall, ResponsesCodecPolicy,
};
use siumai_openai_compatible::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, DialectError,
    PROTOCOL_ID as CHAT_PROTOCOL_ID, WireFieldName,
};
use siumai_protocol_openai::responses::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL,
};
use siumai_transport::{EndpointConfig, RequestHeaders};
use thiserror::Error as ThisError;

use crate::models::{
    DOUBAO_SEED_2_0_CODE_PREVIEW_260215, DOUBAO_SEED_2_0_LITE_260428, DOUBAO_SEED_2_0_MINI_260428,
    DOUBAO_SEED_2_0_PRO_260215, DOUBAO_SEED_2_1_PRO_260628, DOUBAO_SEED_2_1_TURBO_260628,
    DOUBAO_SEED_EVOLVING,
};
use crate::options::{
    ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ARK_BETA_MCP_HEADER,
    ArkCachingType, ArkChatOptions, ArkResponsesOptions, ArkResponsesTool,
};

pub const PROVIDER_ID: &str = "volcengine";
pub const PLATFORM_ID: &str = "ark-cn-beijing";
pub const DEFAULT_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/v3";
pub const CHAT_SOURCE: &str = "https://www.volcengine.com/docs/82379/1330626";
pub const RESPONSES_SOURCE: &str = "https://www.volcengine.com/docs/82379/1585128";
pub const MODEL_SOURCE: &str = "https://www.volcengine.com/docs/82379/1330310";
pub const VERIFIED_ON: &str = "2026-08-08";

pub(crate) fn profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, VolcengineProfileError> {
    let reasoning = WireFieldName::new("reasoning_content")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning);
    let profile = if verified_endpoint {
        OpenAiCompatibleProfile::verified_chat_and_responses(
            verified_profile()?,
            endpoint,
            dialect,
        )?
        .with_replay_domain(replay_domain)?
    } else {
        OpenAiCompatibleProfile::custom_chat_and_responses(
            ProviderId::new(PROVIDER_ID)?,
            endpoint,
            replay_domain,
            dialect,
        )?
    };

    Ok(profile
        .with_chat_codec_policy(Arc::new(ArkChatCodecPolicy))
        .with_responses_codec_policy(Arc::new(ArkResponsesCodecPolicy)))
}

fn verified_profile() -> Result<ProviderProfile, VolcengineProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
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
    let verified_at = VerificationDate::new(NaiveDate::parse_from_str(VERIFIED_ON, "%Y-%m-%d")?);
    let chat_evidence = VerificationEvidence::new(
        OfficialSource::new(CHAT_SOURCE)?,
        verified_at,
        ProtocolContractId::new("ark-openai-chat-2026-08")?,
    );
    let responses_evidence = VerificationEvidence::new(
        OfficialSource::new(RESPONSES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("ark-openai-responses-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        OfficialSource::new(MODEL_SOURCE)?,
        verified_at,
        ProtocolContractId::new("ark-model-catalog-2026-08")?,
    );
    let catalog = ModelCatalog::new(current_models(
        &chat_scope,
        &responses_scope,
        &model_evidence,
    )?)?;

    Ok(ProviderProfile::verified(
        ProfileId::new("volcengine-ark")?,
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
    )?)
}

fn current_models(
    chat_scope: &SupportScope,
    responses_scope: &SupportScope,
    evidence: &VerificationEvidence,
) -> Result<Vec<ModelProfile>, VolcengineProfileError> {
    let entries = [
        (DOUBAO_SEED_2_1_PRO_260628, ModelLifecycle::Active),
        (DOUBAO_SEED_2_1_TURBO_260628, ModelLifecycle::Active),
        (DOUBAO_SEED_2_0_PRO_260215, ModelLifecycle::Active),
        (DOUBAO_SEED_2_0_LITE_260428, ModelLifecycle::Active),
        (DOUBAO_SEED_2_0_MINI_260428, ModelLifecycle::Active),
        (DOUBAO_SEED_2_0_CODE_PREVIEW_260215, ModelLifecycle::Active),
        (DOUBAO_SEED_EVOLVING, ModelLifecycle::RollingAlias),
    ];
    let mut models = Vec::with_capacity(entries.len() * 2);
    for (model, lifecycle) in entries {
        let model = ModelId::new(model)?;
        models.push(ModelProfile::new(
            model.clone(),
            chat_scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle.clone(),
            evidence.clone(),
        )?);
        models.push(ModelProfile::new(
            model,
            responses_scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle,
            evidence.clone(),
        )?);
    }
    Ok(models)
}

#[derive(Debug, Default)]
struct ArkChatCodecPolicy;

impl ChatCodecPolicy for ArkChatCodecPolicy {
    fn name(&self) -> &'static str {
        "ark-chat-2026-08"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        let options: ArkChatOptions = parse_options(&extra, "ARK Chat options are invalid")?;
        options.validate().map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "ARK Chat options are invalid").with_source(source)
        })?;
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers: RequestHeaders::new(),
            prompt_cache_resolver: None,
            warnings: Vec::new(),
        })
    }
}

#[derive(Debug, Default)]
struct ArkResponsesCodecPolicy;

impl ResponsesCodecPolicy for ArkResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "ark-responses-2026-08"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        let options: ArkResponsesOptions =
            parse_options(&extra, "ARK Responses options are invalid")?;
        options.validate().map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "ARK Responses options are invalid")
                .with_source(source)
        })?;
        let caching_enabled = options
            .caching
            .as_ref()
            .is_some_and(|caching| caching.r#type == ArkCachingType::Enabled);
        if caching_enabled && extra.contains_key("instructions") {
            return Err(invalid(
                "ARK Responses caching cannot be combined with instructions",
            ));
        }
        if caching_enabled && !options.native_tools.is_empty() {
            return Err(invalid(
                "ARK Responses caching cannot be combined with provider built-in tools",
            ));
        }
        let native_tools = options
            .native_tools
            .iter()
            .map(ArkResponsesTool::as_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "ARK Responses tool options are invalid",
                )
                .with_source(source)
            })?;
        let mut headers = RequestHeaders::new();
        let uses_image_process = native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("image_process"));
        let uses_knowledge_search = native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("knowledge_search"));
        let uses_mcp = native_tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("mcp"));
        if uses_image_process {
            headers = beta_header(headers, ARK_BETA_IMAGE_PROCESS_HEADER)?;
        }
        if uses_knowledge_search {
            headers = beta_header(headers, ARK_BETA_KNOWLEDGE_SEARCH_HEADER)?;
        }
        if uses_mcp {
            headers = beta_header(headers, ARK_BETA_MCP_HEADER)?;
        }
        extra.remove("native_tools");
        let mut warnings = Vec::new();
        if uses_image_process || uses_knowledge_search || uses_mcp {
            warnings.push(Warning::provider(
                "experimental_provider_tool",
                "ARK beta provider tool behavior may change",
            ));
        }
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers,
            native_tools,
            function_tools: BTreeMap::new(),
            warnings,
        })
    }
}

fn parse_options<T: serde::de::DeserializeOwned>(
    extra: &BTreeMap<String, Value>,
    message: &'static str,
) -> Result<T, Error> {
    serde_json::from_value(Value::Object(extra.clone().into_iter().collect()))
        .map_err(|source| Error::new(ErrorKind::InvalidInput, message).with_source(source))
}

fn beta_header(headers: RequestHeaders, name: &'static str) -> Result<RequestHeaders, Error> {
    headers
        .try_insert(
            HeaderName::from_static(name),
            HeaderValue::from_static("true"),
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "ARK beta request header is invalid",
            )
            .with_source(source)
        })
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum VolcengineProfileError {
    #[error("invalid Volcengine compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid Volcengine model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid Volcengine evidence profile: {0}")]
    Evidence(#[from] ProfileError),
    #[error("invalid ARK Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid Volcengine identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid Volcengine verification date: {0}")]
    VerificationDate(#[from] chrono::ParseError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::ArkCaching;
    use siumai_core::{Message, MessageRole, ProviderOptions};

    #[test]
    fn responses_cache_rejects_instructions_before_transport() {
        let options = ArkResponsesOptions::new().with_caching(ArkCaching::enabled());
        let mut extra: BTreeMap<_, _> = ProviderOptions::typed(&options)
            .expect("options")
            .value()
            .clone()
            .into_iter()
            .collect();
        extra.insert(
            "instructions".to_string(),
            Value::String("system".to_string()),
        );

        let error = ArkResponsesCodecPolicy
            .prepare(
                &ModelId::new(DOUBAO_SEED_2_1_PRO_260628).expect("model"),
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                extra,
            )
            .expect_err("cache and instructions must be rejected");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
