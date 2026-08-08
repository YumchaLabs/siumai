//! Verified Volcengine ARK Chat + Responses profile.
//!
//! `volcengine` is the provider identity and `doubao` is the model brand. Keeping that distinction
//! lets the same ARK runtime host non-Doubao model catalogs without corrupting provider identity.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::NaiveDate;
use http::header::{HeaderName, HeaderValue};
use serde_json::Value;
use siumai_core::{
    ApiModeId, ApiStability, CatalogError, Error, ErrorKind, LanguageRequest, ModelCatalog,
    ModelFamily, ModelId, ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId,
    ProfileError, ProfileId, ProtocolContractId, ProtocolId, ProviderId, ProviderProfile,
    SupportScope, TypedProviderOptions, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim, Warning,
};
use siumai_protocol_openai::chat_completions::{
    API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, DialectError,
    PROTOCOL_ID as CHAT_PROTOCOL_ID, WireFieldName,
};
use siumai_protocol_openai::responses::{
    API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL,
};
use siumai_transport::{EndpointConfig, EndpointError, OfficialOrigin, RequestHeaders};
use thiserror::Error as ThisError;

use crate::configured::codec_policy::{
    ChatCodecPolicy, PreparedChatCall, PreparedResponsesCall, ResponsesCodecPolicy,
};
use crate::configured::{OpenAiCompatibleConfigError, OpenAiCompatibleProfile};
use crate::provider_options::ark::{
    ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ArkCachingType,
    ArkChatOptions, ArkResponsesOptions, ArkResponsesTool,
};

pub const PROVIDER_ID: &str = "volcengine";
pub const PLATFORM_ID: &str = "ark-cn-beijing";
pub const DOUBAO_SEED_2_1_PRO_260628: &str = "doubao-seed-2-1-pro-260628";
pub const OFFICIAL_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/v3";
pub const CHAT_SOURCE: &str = "https://www.volcengine.com/docs/82379/1330626";
pub const RESPONSES_SOURCE: &str = "https://www.volcengine.com/docs/82379/1585128";
pub const MODEL_SOURCE: &str = "https://www.volcengine.com/docs/82379/1330310";
pub const VERIFIED_ON: &str = "2026-08-05";

#[derive(Debug, ThisError)]
pub enum ArkProfileError {
    #[error("invalid ARK endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid ARK compatible profile: {0}")]
    Compatible(#[from] OpenAiCompatibleConfigError),
    #[error("invalid ARK model catalog: {0}")]
    Catalog(#[from] CatalogError),
    #[error("invalid ARK evidence profile: {0}")]
    Evidence(#[from] ProfileError),
    #[error("invalid ARK Chat dialect: {0}")]
    Dialect(#[from] DialectError),
    #[error("invalid ARK identifier: {0}")]
    Identifier(#[from] siumai_core::InvalidId),
    #[error("invalid ARK verification date: {0}")]
    VerificationDate(#[from] chrono::ParseError),
}

pub fn profile() -> Result<OpenAiCompatibleProfile, ArkProfileError> {
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
    let chat_source = OfficialSource::new(CHAT_SOURCE)?;
    let responses_source = OfficialSource::new(RESPONSES_SOURCE)?;
    let model_source = OfficialSource::new(MODEL_SOURCE)?;
    let chat_evidence = VerificationEvidence::new(
        chat_source,
        verified_at,
        ProtocolContractId::new("ark-openai-chat-2026-08")?,
    );
    let responses_evidence = VerificationEvidence::new(
        responses_source,
        verified_at,
        ProtocolContractId::new("ark-openai-responses-2026-08")?,
    );
    let model_evidence = VerificationEvidence::new(
        model_source,
        verified_at,
        ProtocolContractId::new("ark-model-catalog-2026-08")?,
    );
    let catalog = ModelCatalog::new(current_models(
        &chat_scope,
        &responses_scope,
        &model_evidence,
    )?)?;
    let provider_profile = ProviderProfile::verified(
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
    )?;
    let endpoint = EndpointConfig::official(
        OFFICIAL_BASE_URL,
        OfficialOrigin::new("https://ark.cn-beijing.volces.com")?,
    )?;
    let reasoning = WireFieldName::new("reasoning_content")?;
    let dialect = ChatCompletionsDialect::generic()
        .with_reasoning_input_field(reasoning.clone())
        .with_reasoning_output_field(reasoning);
    Ok(
        OpenAiCompatibleProfile::verified_chat_and_responses(provider_profile, endpoint, dialect)?
            .with_chat_codec_policy(Arc::new(ArkChatCodecPolicy))
            .with_responses_codec_policy(Arc::new(ArkResponsesCodecPolicy)),
    )
}

fn current_models(
    chat_scope: &SupportScope,
    responses_scope: &SupportScope,
    evidence: &VerificationEvidence,
) -> Result<Vec<ModelProfile>, ArkProfileError> {
    let entries = [
        (DOUBAO_SEED_2_1_PRO_260628, ModelLifecycle::Active),
        ("doubao-seed-2-1-turbo-260628", ModelLifecycle::Active),
        ("doubao-seed-2-0-pro-260215", ModelLifecycle::Active),
        ("doubao-seed-2-0-lite-260428", ModelLifecycle::Active),
        ("doubao-seed-2-0-mini-260428", ModelLifecycle::Active),
        (
            "doubao-seed-2-0-code-preview-260215",
            ModelLifecycle::Active,
        ),
        ("doubao-seed-evolving", ModelLifecycle::RollingAlias),
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
            prompt_cache_breakpoints: Vec::new(),
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
        if uses_image_process {
            headers = beta_header(headers, ARK_BETA_IMAGE_PROCESS_HEADER)?;
        }
        if uses_knowledge_search {
            headers = beta_header(headers, ARK_BETA_KNOWLEDGE_SEARCH_HEADER)?;
        }
        extra.remove("native_tools");
        let mut warnings = Vec::new();
        if uses_image_process || uses_knowledge_search {
            warnings.push(Warning::provider(
                "experimental_provider_tool",
                "ARK beta provider tool behavior may change",
            ));
        }
        Ok(PreparedResponsesCall {
            request,
            extra,
            headers,
            prompt_cache_breakpoints: Vec::new(),
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

#[cfg(test)]
mod tests {
    const TEST_MODEL: &str = super::DOUBAO_SEED_2_1_PRO_260628;

    use super::*;
    use crate::configured::{OpenAiCompatibleCredential, OpenAiCompatibleProvider};
    use crate::provider_options::{
        ArkCaching, ArkChatOptions, ArkResponsesOptions, ArkResponsesTool, ArkThinking,
    };
    use futures_util::StreamExt;
    use siumai_core::{
        CallOptions, ContentPart, LanguageModel, LanguageStreamEvent, Message, MessageRole,
        ProviderOptionError, ProviderOptions, StreamTerminal,
    };

    #[test]
    fn profile_uses_platform_identity_and_current_catalog() {
        let profile = profile().unwrap();
        assert_eq!(
            profile.recommended_mode(),
            crate::OpenAiCompatibleApiMode::Responses
        );
        let responses = profile
            .scope(crate::OpenAiCompatibleApiMode::Responses)
            .unwrap();
        assert_eq!(responses.provider_id().as_str(), PROVIDER_ID);
        assert_eq!(responses.platform().unwrap().as_str(), PLATFORM_ID);
        let scope = profile
            .support_scope(crate::OpenAiCompatibleApiMode::Responses)
            .unwrap();
        let evolving = profile
            .provider_profile()
            .catalog()
            .unwrap()
            .get(scope, &ModelId::new("doubao-seed-evolving").unwrap())
            .unwrap();
        assert!(matches!(evolving.lifecycle(), ModelLifecycle::RollingAlias));

        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        assert!(provider.responses("doubao-seed-future").is_ok());
        assert!(provider.chat_completions("doubao-seed-future").is_ok());
    }

    #[test]
    fn caching_rejects_instructions_and_provider_tools() {
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
        let error = ArkResponsesCodecPolicy
            .prepare(
                &ModelId::new(TEST_MODEL).unwrap(),
                request.clone(),
                BTreeMap::from([
                    ("caching".to_string(), serde_json::json!({"type":"enabled"})),
                    (
                        "instructions".to_string(),
                        Value::String("system".to_string()),
                    ),
                ]),
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);

        let options = ArkResponsesOptions::new()
            .with_caching(ArkCaching::enabled())
            .with_native_tool(ArkResponsesTool::web_search());
        let extra = ProviderOptions::typed(&options)
            .unwrap()
            .value()
            .clone()
            .into_iter()
            .collect();
        let error = ArkResponsesCodecPolicy
            .prepare(&ModelId::new(TEST_MODEL).unwrap(), request, extra)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn beta_tools_add_headers_and_experimental_warning() {
        let options = ArkResponsesOptions::new()
            .with_native_tool(ArkResponsesTool::ImageProcess)
            .with_native_tool(ArkResponsesTool::knowledge_search("kb-1"));
        let extra = ProviderOptions::typed(&options)
            .unwrap()
            .value()
            .clone()
            .into_iter()
            .collect();
        let prepared = ArkResponsesCodecPolicy
            .prepare(
                &ModelId::new(TEST_MODEL).unwrap(),
                LanguageRequest::new(vec![Message::text(MessageRole::User, "inspect")]),
                extra,
            )
            .unwrap();
        assert!(
            prepared
                .headers
                .get(&HeaderName::from_static(ARK_BETA_IMAGE_PROCESS_HEADER))
                .is_some()
        );
        assert!(
            prepared
                .headers
                .get(&HeaderName::from_static(ARK_BETA_KNOWLEDGE_SEARCH_HEADER))
                .is_some()
        );
        assert_eq!(prepared.warnings.len(), 1);
    }

    #[test]
    fn responses_options_match_the_official_ark_sdk_wire_shape() {
        let options = ArkResponsesOptions::new()
            .with_caching(ArkCaching::enabled().with_prefix(true))
            .with_expire_at(1_785_815_200)
            .with_max_tool_calls(4);
        let typed = ProviderOptions::typed(&options).unwrap();
        assert_eq!(
            typed.value()["caching"],
            serde_json::json!({"type":"enabled","prefix":true})
        );
        assert_eq!(
            typed.value()["expire_at"],
            serde_json::json!(1_785_815_200_i64)
        );
        assert_eq!(typed.value()["max_tool_calls"], serde_json::json!(4));

        let web_search = ArkResponsesTool::web_search().as_value().unwrap();
        assert!(web_search.get("max_tool_calls").is_none());
    }

    #[test]
    fn typed_options_are_bound_to_their_ark_api_mode() {
        let provider = OpenAiCompatibleProvider::builder(
            profile().unwrap(),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();

        let responses = CallOptions::default().with_provider_options(
            ProviderOptions::typed(&ArkResponsesOptions::new().with_store(true)).unwrap(),
        );
        assert!(matches!(
            provider
                .runtime
                .merge_options(crate::OpenAiCompatibleApiMode::ChatCompletions, &responses),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));

        let chat = CallOptions::default().with_provider_options(
            ProviderOptions::typed(&ArkChatOptions::new().with_thinking(ArkThinking::enabled()))
                .unwrap(),
        );
        assert!(matches!(
            provider
                .runtime
                .merge_options(crate::OpenAiCompatibleApiMode::Responses, &chat),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn configured_chat_sends_ark_thinking_and_decodes_reasoning() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(format!(r#"\"model\":\"{TEST_MODEL}\""#)),
                mockito::Matcher::Regex(r#"\"thinking\":\{\"type\":\"enabled\"\}"#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":false"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"id":"chat-ark","model":"{TEST_MODEL}","choices":[{{"index":0,"message":{{"role":"assistant","content":"answer","reasoning_content":"thought"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}}}"#
            ))
            .expect(1)
            .create_async()
            .await;
        let provider = OpenAiCompatibleProvider::builder(
            profile().unwrap().with_test_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap(),
            ),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let options = ArkChatOptions::new().with_thinking(ArkThinking::enabled());
        let response = provider
            .chat_completions(TEST_MODEL)
            .unwrap()
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default()
                    .with_provider_options(ProviderOptions::typed(&options).unwrap()),
            )
            .await
            .unwrap();
        assert!(
            response
                .content()
                .iter()
                .any(|part| matches!(part, ContentPart::Reasoning { text } if text == "thought"))
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_responses_stream_preserves_ark_doc_citation() {
        let mut server = mockito::Server::new_async().await;
        let response_id = "resp-ark-stream";
        let completed_item = serde_json::json!({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "answer",
                "annotations": [{
                    "type": "doc_citation",
                    "title": "Doc",
                    "url": "https://example.com/doc"
                }]
            }]
        });
        let frames = [
            serde_json::json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": response_id,
                    "created_at": 1_785_811_200_i64,
                    "model": TEST_MODEL,
                    "status": "in_progress",
                    "output": [],
                    "usage": null,
                    "error": null,
                    "incomplete_details": null,
                    "reasoning": null
                }
            }),
            serde_json::json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": completed_item.clone()
            }),
            serde_json::json!({
                "type": "response.completed",
                "sequence_number": 3,
                "response": {
                    "id": response_id,
                    "created_at": 1_785_811_200_i64,
                    "model": TEST_MODEL,
                    "status": "completed",
                    "output": [completed_item],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2
                    },
                    "error": null,
                    "incomplete_details": null,
                    "reasoning": null
                }
            }),
        ];
        let stream_body = frames
            .into_iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>();
        let mock = server
            .mock("POST", "/v1/responses")
            .match_header(ARK_BETA_KNOWLEDGE_SEARCH_HEADER, "true")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"type\":\"knowledge_search\""#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(stream_body)
            .expect(1)
            .create_async()
            .await;
        let provider = OpenAiCompatibleProvider::builder(
            profile().unwrap().with_test_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap(),
            ),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let options =
            ArkResponsesOptions::new().with_native_tool(ArkResponsesTool::knowledge_search("kb-1"));
        let events = provider
            .responses(TEST_MODEL)
            .unwrap()
            .stream(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "search")]),
                CallOptions::default()
                    .with_provider_options(ProviderOptions::typed(&options).unwrap()),
            )
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
        let response = response.unwrap_or_else(|| panic!("unexpected stream events: {events:#?}"));
        assert!(response.content().iter().any(|part| {
            matches!(
                part,
                ContentPart::Citation(citation)
                    if citation.title.as_deref() == Some("Doc")
                        && citation.provider[OPENAI_RESPONSES_PROTOCOL]["type"]
                            == serde_json::json!("doc_citation")
            )
        }));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_ark_preserves_http_error_diagnostics() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/responses")
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_header("x-request-id", "ark-request-42")
            .with_body(
                r#"{"error":{"message":"invalid tool configuration","type":"invalid_request_error","param":"tools.0","code":"invalid_parameter"}}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let provider = OpenAiCompatibleProvider::builder(
            profile().unwrap().with_test_endpoint(
                EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap(),
            ),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let error = provider
            .responses(TEST_MODEL)
            .unwrap()
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        let diagnostics = error.diagnostics().unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(diagnostics.provider_code(), Some("invalid_parameter"));
        assert_eq!(diagnostics.provider_param(), Some("tools.0"));
        assert_eq!(diagnostics.request_id(), Some("ark-request-42"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_responses_sends_beta_header_and_decodes_annotations() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/responses")
            .match_header(ARK_BETA_KNOWLEDGE_SEARCH_HEADER, "true")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(format!(r#"\"model\":\"{TEST_MODEL}\""#)),
                mockito::Matcher::Regex(r#"\"type\":\"knowledge_search\""#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"id":"resp-ark","object":"response","created_at":1785811200,"model":"{TEST_MODEL}","status":"completed","output":[{{"id":"msg-1","type":"message","role":"assistant","status":"completed","content":[{{"type":"output_text","text":"answer","annotations":[{{"type":"doc_citation","title":"Doc","url":"https://example.com/doc"}}]}}]}}],"usage":{{"input_tokens":1,"input_tokens_details":{{"cached_tokens":0}},"output_tokens":1,"output_tokens_details":{{"reasoning_tokens":0}},"total_tokens":2}},"error":null,"incomplete_details":null,"reasoning":null}}"#
            ))
            .expect(1)
            .create_async()
            .await;
        let profile = profile().unwrap().with_test_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap(),
        );
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let options =
            ArkResponsesOptions::new().with_native_tool(ArkResponsesTool::knowledge_search("kb-1"));
        let response = provider
            .responses(TEST_MODEL)
            .unwrap()
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "search")]),
                CallOptions::default()
                    .with_provider_options(ProviderOptions::typed(&options).unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(response.id(), Some("resp-ark"));
        assert_eq!(response.warnings().len(), 1);
        let citation = response
            .content()
            .iter()
            .find_map(|part| match part {
                ContentPart::Citation(citation) => Some(citation),
                _ => None,
            })
            .expect("ARK doc_citation must survive canonical projection");
        assert_eq!(citation.title.as_deref(), Some("Doc"));
        assert_eq!(citation.url.as_deref(), Some("https://example.com/doc"));
        assert_eq!(
            citation.provider[OPENAI_RESPONSES_PROTOCOL]["type"],
            serde_json::json!("doc_citation")
        );
        mock.assert_async().await;
    }
}
