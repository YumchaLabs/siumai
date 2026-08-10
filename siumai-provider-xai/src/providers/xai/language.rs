//! xAI language profiles and provider-owned codec policy.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use http::header::{HeaderName, HeaderValue};
use serde_json::{Map, Value};
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, Error, ErrorKind, InvalidId, LanguageCallError,
    LanguageRequest, LanguageResponse, ModelCatalog, ModelFamily, ModelId, ModelLifecycle,
    ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileError, ProfileId,
    ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain, SupportScope,
    TypedProviderOptions, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim,
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
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL, ResponsesWireDialect,
    decode_response as decode_responses_response,
};
use siumai_transport::{EndpointConfig, RequestHeaders, ResponseHeaders};
use thiserror::Error as ThisError;

use crate::provider_options::{XaiChatOptions, XaiResponsesOptions};

use super::models;

pub const PROVIDER_ID: &str = "xai";
pub const PLATFORM_ID: &str = "xai-public-api";
pub const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
pub const OFFICIAL_ORIGIN: &str = "https://api.x.ai";
pub const CHAT_SOURCE: &str = "https://docs.x.ai/developers/rest-api-reference/inference/chat";
pub const RESPONSES_SOURCE: &str =
    "https://docs.x.ai/developers/rest-api-reference/inference/responses";
pub const VERIFIED_ON: &str = "2026-08-06";
const XAI_CONVERSATION_ID_HEADER: HeaderName = HeaderName::from_static("x-grok-conv-id");

pub(crate) fn profile(
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    verified_endpoint: bool,
) -> Result<OpenAiCompatibleProfile, XaiProfileError> {
    let provider = ProviderId::new(PROVIDER_ID)?;
    let reasoning = WireFieldName::new("reasoning_content")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning);

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
        .with_chat_codec_policy(Arc::new(XaiChatCodecPolicy))
        .with_responses_codec_policy(Arc::new(XaiResponsesCodecPolicy))
        .with_responses_wire_dialect(ResponsesWireDialect::compatible()))
}

fn verified_profile(
    provider: ProviderId,
    endpoint: EndpointConfig,
    replay_domain: ReplayDomain,
    dialect: ChatCompletionsDialect,
) -> Result<OpenAiCompatibleProfile, XaiProfileError> {
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
            .map_err(|_| XaiProfileError::InvalidVerificationDate)?,
    );
    let models_verified_at = VerificationDate::new(
        NaiveDate::parse_from_str(models::VERIFIED_ON, "%Y-%m-%d")
            .map_err(|_| XaiProfileError::InvalidVerificationDate)?,
    );
    let chat_evidence = VerificationEvidence::new(
        OfficialSource::new(CHAT_SOURCE)?,
        verified_at,
        ProtocolContractId::new("xai-chat-completions-2026-08")?,
    );
    let responses_evidence = VerificationEvidence::new(
        OfficialSource::new(RESPONSES_SOURCE)?,
        verified_at,
        ProtocolContractId::new("xai-responses-2026-08")?,
    );
    let chat_model_evidence = VerificationEvidence::new(
        OfficialSource::new(models::OFFICIAL_SOURCE)?,
        models_verified_at,
        ProtocolContractId::new("xai-chat-model-catalog-2026-08")?,
    );
    let responses_model_evidence = VerificationEvidence::new(
        OfficialSource::new(models::OFFICIAL_SOURCE)?,
        models_verified_at,
        ProtocolContractId::new("xai-responses-model-catalog-2026-08")?,
    );
    let catalog = model_catalog(
        &chat_scope,
        &responses_scope,
        &chat_model_evidence,
        &responses_model_evidence,
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

fn model_catalog(
    chat_scope: &SupportScope,
    responses_scope: &SupportScope,
    chat_evidence: &VerificationEvidence,
    responses_evidence: &VerificationEvidence,
) -> Result<ModelCatalog, XaiProfileError> {
    let mut entries = Vec::new();
    add_model_profiles(
        &mut entries,
        models::catalog::CHAT_EXACT_IDS,
        chat_scope,
        ModelLifecycle::Active,
        chat_evidence,
    )?;
    add_model_profiles(
        &mut entries,
        models::catalog::CHAT_ROLLING_ALIASES,
        chat_scope,
        ModelLifecycle::RollingAlias,
        chat_evidence,
    )?;
    add_model_profiles(
        &mut entries,
        models::catalog::RESPONSES_EXACT_IDS,
        responses_scope,
        ModelLifecycle::Active,
        responses_evidence,
    )?;
    add_model_profiles(
        &mut entries,
        models::catalog::RESPONSES_ROLLING_ALIASES,
        responses_scope,
        ModelLifecycle::RollingAlias,
        responses_evidence,
    )?;
    Ok(ModelCatalog::new(entries)?)
}

fn add_model_profiles(
    entries: &mut Vec<ModelProfile>,
    models: &[&str],
    scope: &SupportScope,
    lifecycle: ModelLifecycle,
    evidence: &VerificationEvidence,
) -> Result<(), XaiProfileError> {
    for model in models {
        entries.push(ModelProfile::new(
            ModelId::new(*model)?,
            scope.clone(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle.clone(),
            evidence.clone(),
        )?);
    }
    Ok(())
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum XaiProfileError {
    #[error("invalid xAI identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid xAI support evidence: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid xAI model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid xAI Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid xAI compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("xAI verification date is invalid")]
    InvalidVerificationDate,
}

#[derive(Debug, Default)]
struct XaiChatCodecPolicy;

impl ChatCodecPolicy for XaiChatCodecPolicy {
    fn name(&self) -> &'static str {
        "xai-chat-completions-2026-08"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        dialect: ChatCompletionsDialect,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedChatCall, Error> {
        normalize_alias(&mut extra, "reasoningEffort", "reasoning_effort");
        normalize_alias(&mut extra, "topLogprobs", "top_logprobs");
        normalize_alias(&mut extra, "parallelToolCalls", "parallel_tool_calls");
        normalize_alias(&mut extra, "searchParameters", "search_parameters");
        normalize_alias(&mut extra, "promptCacheKey", "prompt_cache_key");
        if let Some(search) = extra.get_mut("search_parameters") {
            normalize_search_parameters(search)?;
        }
        let options = parse_options::<XaiChatOptions>(&extra, "Chat Completions")?;
        options.validate().map_err(option_error)?;
        let mut headers = RequestHeaders::new();
        if let Some(prompt_cache_key) = options.prompt_cache_key.as_deref() {
            let value = HeaderValue::from_str(prompt_cache_key).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "xAI prompt cache key is not a valid HTTP header value",
                )
                .with_source(source)
            })?;
            headers = headers
                .try_insert(XAI_CONVERSATION_ID_HEADER, value)
                .map_err(|source| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        "xAI prompt cache header could not be constructed",
                    )
                    .with_source(source)
                })?;
        }
        extra.remove("prompt_cache_key");
        if options.top_logprobs.is_some() {
            extra.insert("logprobs".to_string(), Value::Bool(true));
        }
        Ok(PreparedChatCall {
            request,
            dialect,
            extra,
            headers,
            prompt_cache_resolver: None,
            warnings: Vec::new(),
        })
    }
}

#[derive(Debug, Default)]
struct XaiResponsesCodecPolicy;

impl ResponsesCodecPolicy for XaiResponsesCodecPolicy {
    fn name(&self) -> &'static str {
        "xai-responses-2026-08"
    }

    fn prepare(
        &self,
        _model: &ModelId,
        request: LanguageRequest,
        mut extra: BTreeMap<String, Value>,
    ) -> Result<PreparedResponsesCall, Error> {
        normalize_alias(&mut extra, "reasoningEffort", "reasoning_effort");
        normalize_alias(&mut extra, "reasoningSummary", "reasoning_summary");
        normalize_alias(&mut extra, "topLogprobs", "top_logprobs");
        normalize_alias(&mut extra, "parallelToolCalls", "parallel_tool_calls");
        normalize_alias(&mut extra, "previousResponseId", "previous_response_id");
        normalize_alias(&mut extra, "promptCacheKey", "prompt_cache_key");
        let options = parse_options::<XaiResponsesOptions>(&extra, "Responses")?;
        options.validate().map_err(option_error)?;

        let mut reasoning = Map::new();
        if let Some(effort) = &options.reasoning_effort {
            reasoning.insert(
                "effort".to_string(),
                Value::String(effort.as_str().to_string()),
            );
        }
        if let Some(summary) = &options.reasoning_summary {
            reasoning.insert(
                "summary".to_string(),
                Value::String(summary.as_str().to_string()),
            );
        }
        if !reasoning.is_empty() {
            if extra.contains_key("reasoning") {
                return Err(invalid(
                    "xAI typed reasoning options cannot be combined with a raw reasoning object",
                ));
            }
            extra.insert("reasoning".to_string(), Value::Object(reasoning));
        }
        if options.top_logprobs.is_some() {
            extra.insert("logprobs".to_string(), Value::Bool(true));
        }
        for field in ["reasoning_effort", "reasoning_summary", "native_tools"] {
            extra.remove(field);
        }
        let native_tools = options
            .native_tools
            .into_iter()
            .map(|tool| {
                tool.as_value().map_err(|source| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        "xAI hosted tool could not be serialized",
                    )
                    .with_source(source)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers: RequestHeaders::new(),
            native_tools,
            function_tools: BTreeMap::new(),
            warnings: Vec::new(),
        })
    }

    fn decode_response(
        &self,
        scope: &siumai_core::ProviderScope,
        model: &ModelId,
        _headers: &ResponseHeaders,
        body: &[u8],
    ) -> Result<LanguageResponse, LanguageCallError> {
        let decoded = decode_responses_response(body, scope, model)?;
        decoded.into_result()
    }
}

fn parse_options<T>(extra: &BTreeMap<String, Value>, mode: &str) -> Result<T, Error>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(Value::Object(extra.clone().into_iter().collect())).map_err(|source| {
        let message = match mode {
            "Chat Completions" => "xAI Chat Completions options have an invalid shape",
            "Responses" => "xAI Responses options have an invalid shape",
            _ => "xAI options have an invalid shape",
        };
        Error::new(ErrorKind::InvalidInput, message).with_source(source)
    })
}

fn normalize_alias(extra: &mut BTreeMap<String, Value>, alias: &str, wire: &str) {
    if let Some(value) = extra.remove(alias) {
        extra.entry(wire.to_string()).or_insert(value);
    }
}

fn normalize_search_parameters(value: &mut Value) -> Result<(), Error> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("xAI search_parameters must be an object"))?;
    for (alias, wire) in [
        ("returnCitations", "return_citations"),
        ("maxSearchResults", "max_search_results"),
        ("fromDate", "from_date"),
        ("toDate", "to_date"),
    ] {
        normalize_map_alias(object, alias, wire);
    }
    if let Some(sources) = object.get_mut("sources").and_then(Value::as_array_mut) {
        for source in sources {
            let source = source
                .as_object_mut()
                .ok_or_else(|| invalid("xAI search source must be an object"))?;
            for (alias, wire) in [
                ("allowedWebsites", "allowed_websites"),
                ("excludedWebsites", "excluded_websites"),
                ("safeSearch", "safe_search"),
                ("excludedXHandles", "excluded_x_handles"),
                ("includedXHandles", "included_x_handles"),
                ("postFavoriteCount", "post_favorite_count"),
                ("postViewCount", "post_view_count"),
            ] {
                normalize_map_alias(source, alias, wire);
            }
        }
    }
    Ok(())
}

fn normalize_map_alias(object: &mut Map<String, Value>, alias: &str, wire: &str) {
    if let Some(value) = object.remove(alias) {
        object.entry(wire.to_string()).or_insert(value);
    }
}

fn option_error(source: siumai_core::ProviderOptionError) -> Error {
    Error::new(ErrorKind::InvalidInput, "xAI provider options are invalid").with_source(source)
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        XaiReasoningSummary, XaiResponsesReasoningEffort, XaiSearchParameters,
    };
    use crate::tools::XaiResponsesTool;

    fn model() -> ModelId {
        ModelId::new("future-grok-model").expect("valid future model id")
    }

    #[test]
    fn responses_options_become_native_reasoning_and_hosted_tools() {
        let options = XaiResponsesOptions::new()
            .with_reasoning_effort(XaiResponsesReasoningEffort::High)
            .with_reasoning_summary(XaiReasoningSummary::Detailed)
            .with_native_tool(XaiResponsesTool::web_search());
        let extra = serde_json::to_value(options)
            .expect("serialize options")
            .as_object()
            .expect("options object")
            .clone()
            .into_iter()
            .collect();
        let prepared = XaiResponsesCodecPolicy
            .prepare(&model(), LanguageRequest::new(Vec::new()), extra)
            .expect("prepare Responses request");
        assert_eq!(
            prepared.extra["reasoning"],
            serde_json::json!({"effort": "high", "summary": "detailed"})
        );
        assert_eq!(
            prepared.native_tools,
            vec![serde_json::json!({"type": "web_search"})]
        );
    }

    #[test]
    fn chat_options_normalize_search_fields_without_model_allowlisting() {
        let options = XaiChatOptions::new()
            .with_search(XaiSearchParameters::default())
            .with_prompt_cache_key("cache-123");
        let extra = serde_json::to_value(options)
            .expect("serialize options")
            .as_object()
            .expect("options object")
            .clone()
            .into_iter()
            .collect();
        let prepared = XaiChatCodecPolicy
            .prepare(
                &model(),
                LanguageRequest::new(Vec::new()),
                ChatCompletionsDialect::generic(),
                extra,
            )
            .expect("prepare Chat request");
        assert_eq!(prepared.extra["search_parameters"], serde_json::json!({}));
        assert!(!prepared.extra.contains_key("prompt_cache_key"));
        assert_eq!(
            prepared
                .headers
                .get(&XAI_CONVERSATION_ID_HEADER)
                .expect("prompt-cache header")
                .to_str()
                .expect("ASCII prompt-cache key"),
            "cache-123"
        );
    }
}
