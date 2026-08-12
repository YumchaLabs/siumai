use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use siumai_core::{
    InvalidId, LanguageModel, LanguageModelProvider, Model, ModelId, ModelLookupError,
    ProfileError, Provider, ProviderInstanceId, ProviderOptionError, ProviderOptionSelection,
    ProviderOptions, ProviderRegistration, ProviderScope,
};
use siumai_protocol_openai::chat_completions::is_protected_option_field as is_chat_protected_field;
use siumai_protocol_openai::responses::is_protected_option_field as is_responses_protected_field;
use siumai_transport::{
    AuthApplier, EndpointError, ProviderTransport, ReplaySafety, RetryPolicy, TransportConfigError,
    TransportLimits,
};
use thiserror::Error;

use super::credentials::{CredentialSourceError, OpenAiCompatibleCredential};
use super::mode::OpenAiCompatibleApiMode;
use super::model::OpenAiCompatibleLanguageModel;
use super::profile::OpenAiCompatibleProfile;

/// A synchronously configured OpenAI-compatible provider.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
}

impl OpenAiCompatibleProvider {
    pub fn builder(
        profile: OpenAiCompatibleProfile,
        credential: OpenAiCompatibleCredential,
    ) -> OpenAiCompatibleProviderBuilder {
        OpenAiCompatibleProviderBuilder::new(profile, credential)
    }

    /// Build a branded compatible provider around an already configured auth applier.
    #[doc(hidden)]
    pub fn builder_with_auth(
        profile: OpenAiCompatibleProfile,
        auth: Arc<dyn AuthApplier>,
    ) -> OpenAiCompatibleProviderBuilder {
        OpenAiCompatibleProviderBuilder::with_auth(profile, auth)
    }

    /// Create a lightweight model in the profile's recommended mode.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        self.language_for(self.recommended_mode(), model)
    }

    /// Create a lightweight Responses model handle.
    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        self.language_for(OpenAiCompatibleApiMode::Responses, model)
    }

    /// Create a lightweight Chat Completions model handle.
    pub fn chat_completions(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        self.language_for(OpenAiCompatibleApiMode::ChatCompletions, model)
    }

    pub fn language_for(
        &self,
        mode: OpenAiCompatibleApiMode,
        model: impl Into<String>,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        let model = parse_model_id(model)?;
        self.create_language_model(mode, model)
    }

    /// Construct the recommended language family handle.
    pub fn language_model(
        &self,
        model: ModelId,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        self.create_language_model(self.recommended_mode(), model)
    }

    pub fn registration(&self) -> ProviderRegistration {
        self.registration_with_scope(
            self.recommended_mode(),
            self.runtime.profile.recommended_scope().clone(),
        )
    }

    pub fn responses_registration(&self) -> Option<ProviderRegistration> {
        self.registration_for(OpenAiCompatibleApiMode::Responses)
    }

    pub fn chat_completions_registration(&self) -> Option<ProviderRegistration> {
        self.registration_for(OpenAiCompatibleApiMode::ChatCompletions)
    }

    pub fn registration_for(&self, mode: OpenAiCompatibleApiMode) -> Option<ProviderRegistration> {
        let scope = self.runtime.scope_arc(mode)?;
        Some(self.registration_with_scope(mode, scope))
    }

    fn registration_with_scope(
        &self,
        mode: OpenAiCompatibleApiMode,
        scope: Arc<ProviderScope>,
    ) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_language(
            scope,
            Arc::new(move |model| {
                Ok(Arc::new(provider.create_language_model(mode, model)?)
                    as Arc<dyn LanguageModel>)
            }),
        )
    }

    pub fn profile(&self) -> &OpenAiCompatibleProfile {
        &self.runtime.profile
    }

    pub fn recommended_mode(&self) -> OpenAiCompatibleApiMode {
        self.runtime.profile.recommended_mode()
    }

    fn create_language_model(
        &self,
        mode: OpenAiCompatibleApiMode,
        model: ModelId,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        let mode_profile = self
            .runtime
            .profile
            .language_mode(mode)
            .ok_or_else(|| unavailable_mode(self.runtime.provider_id(), mode))?;
        Ok(OpenAiCompatibleLanguageModel::new(
            self.runtime.clone(),
            mode_profile,
            model,
        ))
    }
}

impl Provider for OpenAiCompatibleProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.provider_id()
    }
}

impl LanguageModelProvider for OpenAiCompatibleProvider {
    type Model = OpenAiCompatibleLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.create_language_model(self.recommended_mode(), model)
    }
}

impl fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProvider")
            .field("provider_id", self.provider_id())
            .field("recommended_mode", &self.recommended_mode())
            .field("profile", &self.runtime.profile)
            .finish()
    }
}

pub struct OpenAiCompatibleProviderBuilder {
    profile: OpenAiCompatibleProfile,
    auth: CompatibleAuth,
    instance_id: Option<ProviderInstanceId>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    chat_defaults: BTreeMap<String, Value>,
    responses_defaults: BTreeMap<String, Value>,
}

impl OpenAiCompatibleProviderBuilder {
    fn new(profile: OpenAiCompatibleProfile, credential: OpenAiCompatibleCredential) -> Self {
        Self {
            profile,
            auth: CompatibleAuth::Credential(credential),
            instance_id: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            chat_defaults: BTreeMap::new(),
            responses_defaults: BTreeMap::new(),
        }
    }

    fn with_auth(profile: OpenAiCompatibleProfile, auth: Arc<dyn AuthApplier>) -> Self {
        Self {
            profile,
            auth: CompatibleAuth::Applied(auth),
            instance_id: None,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            chat_defaults: BTreeMap::new(),
            responses_defaults: BTreeMap::new(),
        }
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Reuse the owning branded provider's configured-instance capability.
    #[doc(hidden)]
    pub fn with_provider_instance(mut self, instance_id: ProviderInstanceId) -> Self {
        self.instance_id = Some(instance_id);
        self
    }

    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = Some(timeout);
        self
    }

    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = Some(timeout);
        self
    }

    pub fn with_default_option(
        mut self,
        mode: OpenAiCompatibleApiMode,
        name: impl Into<String>,
        value: Value,
    ) -> Self {
        match mode {
            OpenAiCompatibleApiMode::Responses => {
                self.responses_defaults.insert(name.into(), value);
            }
            OpenAiCompatibleApiMode::ChatCompletions => {
                self.chat_defaults.insert(name.into(), value);
            }
        }
        self
    }

    /// Validate static settings and construct one shared provider runtime.
    pub fn build(self) -> Result<OpenAiCompatibleProvider, OpenAiCompatibleConfigError> {
        self.profile
            .recommended_scope()
            .replay_domain()
            .ok_or(OpenAiCompatibleConfigError::MissingReplayDomain)?;
        let auth = match self.auth {
            CompatibleAuth::Credential(credential) => {
                credential.validate_static()?;
                credential.into_auth()
            }
            CompatibleAuth::Applied(auth) => auth,
        };
        validate_default_options(
            &self.profile,
            OpenAiCompatibleApiMode::ChatCompletions,
            &self.chat_defaults,
        )?;
        validate_default_options(
            &self.profile,
            OpenAiCompatibleApiMode::Responses,
            &self.responses_defaults,
        )?;

        let mut transport = ProviderTransport::builder(self.profile.endpoint().clone())
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            transport = transport.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            transport = transport.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            transport = transport.with_read_timeout(timeout);
        }
        let transport = transport.build()?;
        let instance_id = self.instance_id.unwrap_or_default();
        Ok(OpenAiCompatibleProvider {
            runtime: Arc::new(ProviderRuntime {
                instance_id,
                profile: self.profile,
                transport,
                chat_options: CompatibleOptionMerger::new(
                    OpenAiCompatibleApiMode::ChatCompletions,
                    self.chat_defaults,
                ),
                responses_options: CompatibleOptionMerger::new(
                    OpenAiCompatibleApiMode::Responses,
                    self.responses_defaults,
                ),
                replay_safety: ReplaySafety::Never,
            }),
        })
    }
}

enum CompatibleAuth {
    Credential(OpenAiCompatibleCredential),
    Applied(Arc<dyn AuthApplier>),
}

pub(crate) struct ProviderRuntime {
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) profile: OpenAiCompatibleProfile,
    pub(crate) transport: ProviderTransport,
    chat_options: CompatibleOptionMerger,
    responses_options: CompatibleOptionMerger,
    pub(crate) replay_safety: ReplaySafety,
}

impl ProviderRuntime {
    pub(crate) fn provider_id(&self) -> &siumai_core::ProviderId {
        self.profile.provider_profile().provider_id()
    }

    pub(crate) fn scope(&self, mode: OpenAiCompatibleApiMode) -> Option<&ProviderScope> {
        self.profile.scope(mode)
    }

    pub(crate) fn scope_arc(&self, mode: OpenAiCompatibleApiMode) -> Option<Arc<ProviderScope>> {
        self.profile.scope_arc(mode)
    }

    pub(crate) fn options_for<M: Model + ?Sized>(
        &self,
        model: &M,
        mode: OpenAiCompatibleApiMode,
        options: &siumai_core::CallOptions,
    ) -> Result<CompatibleCallOptions, ProviderOptionError> {
        self.scope(mode)
            .ok_or_else(|| ProviderOptionError::Rejected {
                path: "api_mode".to_string(),
                reason: "the configured profile does not expose this language mode".to_string(),
            })?;
        let selection = options.provider_options_for(model)?;
        let merger = match mode {
            OpenAiCompatibleApiMode::Responses => &self.responses_options,
            OpenAiCompatibleApiMode::ChatCompletions => &self.chat_options,
        };
        merger.merge_selected(&selection)
    }
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("profile_id", self.profile.provider_profile().id())
            .field("recommended_mode", &self.profile.recommended_mode())
            .field(
                "chat_scope",
                &self.profile.scope(OpenAiCompatibleApiMode::ChatCompletions),
            )
            .field(
                "responses_scope",
                &self.profile.scope(OpenAiCompatibleApiMode::Responses),
            )
            .field("transport", &"shared")
            .field(
                "chat_default_option_fields",
                &self.chat_options.defaults.keys().collect::<Vec<_>>(),
            )
            .field(
                "responses_default_option_fields",
                &self.responses_options.defaults.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

struct CompatibleOptionMerger {
    mode: OpenAiCompatibleApiMode,
    defaults: BTreeMap<String, Value>,
}

pub(crate) struct CompatibleCallOptions {
    pub(crate) typed: BTreeMap<String, Value>,
    pub(crate) raw: Option<Map<String, Value>>,
}

impl CompatibleOptionMerger {
    fn new(mode: OpenAiCompatibleApiMode, defaults: BTreeMap<String, Value>) -> Self {
        Self { mode, defaults }
    }

    fn merge_selected(
        &self,
        selection: &ProviderOptionSelection<'_>,
    ) -> Result<CompatibleCallOptions, ProviderOptionError> {
        let mut typed = self.defaults.clone();
        for options in selection.typed() {
            self.validate_options(options)?;
            overlay_options(&mut typed, options);
        }
        let raw = selection
            .raw_override()
            .map(|options| options.value().clone());
        Ok(CompatibleCallOptions { typed, raw })
    }

    fn validate_options(&self, options: &ProviderOptions) -> Result<(), ProviderOptionError> {
        validate_option_fields(self.mode, options.value()).map_err(|path| {
            ProviderOptionError::Rejected {
                path,
                reason: format!(
                    "field is owned by the canonical {} request",
                    self.mode.as_str()
                ),
            }
        })
    }
}

fn overlay_options(merged: &mut BTreeMap<String, Value>, options: &ProviderOptions) {
    merged.extend(
        options
            .value()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
}

fn validate_default_options(
    profile: &OpenAiCompatibleProfile,
    mode: OpenAiCompatibleApiMode,
    options: &BTreeMap<String, Value>,
) -> Result<(), OpenAiCompatibleConfigError> {
    if !options.is_empty() && !profile.supports_mode(mode) {
        return Err(OpenAiCompatibleConfigError::DefaultsForUnavailableMode(
            mode,
        ));
    }
    validate_option_fields(mode, options).map_err(OpenAiCompatibleConfigError::InvalidDefaultOption)
}

fn parse_model_id(model: impl Into<String>) -> Result<ModelId, ModelLookupError> {
    ModelId::new(model.into()).map_err(ModelLookupError::from)
}

fn unavailable_mode(
    provider: &siumai_core::ProviderId,
    mode: OpenAiCompatibleApiMode,
) -> ModelLookupError {
    ModelLookupError::Construction {
        source: siumai_core::Error::new(
            siumai_core::ErrorKind::Unsupported,
            match mode {
                OpenAiCompatibleApiMode::Responses => {
                    "configured provider does not expose the Responses API mode"
                }
                OpenAiCompatibleApiMode::ChatCompletions => {
                    "configured provider does not expose Chat Completions"
                }
            },
        )
        .with_context(siumai_core::ErrorContext {
            provider: Some(provider.clone()),
            ..siumai_core::ErrorContext::default()
        }),
    }
}

fn validate_option_fields(
    mode: OpenAiCompatibleApiMode,
    options: &impl OptionFields,
) -> Result<(), String> {
    options
        .field_names()
        .find(|name| match mode {
            OpenAiCompatibleApiMode::Responses => is_responses_protected_field(name),
            OpenAiCompatibleApiMode::ChatCompletions => is_chat_protected_field(name),
        })
        .map_or(Ok(()), |name| Err(name.to_string()))
}

trait OptionFields {
    fn field_names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
}

impl OptionFields for BTreeMap<String, Value> {
    fn field_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.keys().map(String::as_str))
    }
}

impl OptionFields for serde_json::Map<String, Value> {
    fn field_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.keys().map(String::as_str))
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpenAiCompatibleConfigError {
    #[error("invalid provider identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid provider profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("invalid provider endpoint: {0}")]
    Endpoint(EndpointError),
    #[error("invalid provider transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid static credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("verified profile requires evidence-backed support claims")]
    ExpectedVerifiedProfile,
    #[error("OpenAI-compatible profile requires at least one language mode claim")]
    MissingLanguageModeClaim,
    #[error("OpenAI-compatible profile contains a duplicate language mode claim")]
    DuplicateLanguageModeClaim,
    #[error("profile claims do not match the configured Chat/Responses mode slots")]
    LanguageModeClaimMismatch,
    #[error(
        "Chat and Responses claims sharing one endpoint must use the same provider and platform"
    )]
    SharedEndpointScopeMismatch,
    #[error("verified profile endpoint must use an exact official-origin policy")]
    VerifiedEndpointMustBeOfficial,
    #[error("compatible profile requires an explicit non-secret replay domain")]
    MissingReplayDomain,
    #[error("replay audience does not match compatible profile ownership")]
    ReplayAudienceMismatch,
    #[error("support scope is not an OpenAI-family Chat or Responses language mode")]
    IncompatibleSupportScope,
    #[error("default options were configured for unavailable mode {0:?}")]
    DefaultsForUnavailableMode(OpenAiCompatibleApiMode),
    #[error("default option `{0}` attempts to override a canonical request field")]
    InvalidDefaultOption(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use futures_util::StreamExt;
    use http::header::{HeaderName, HeaderValue};
    use serde_json::json;
    use siumai_core::{
        ApiModeId, ApiStability, CallOptions, ContentPart, Error, ErrorKind, LanguageModel,
        LanguageRequest, Message, MessageRole, Model, ModelCatalog, ModelFamily, ModelId,
        ModelLifecycle, ModelOperation, ModelProfile, OfficialSource, PlatformId, ProfileId,
        ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomain, ReplayDomainId,
        SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity,
        VerifiedSupportClaim,
    };
    use siumai_protocol_openai::chat_completions::{
        API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID as CHAT_PROTOCOL_ID,
    };
    use siumai_protocol_openai::responses::{
        API_MODE_ID as RESPONSES_API_MODE_ID, OPENAI_RESPONSES_PROTOCOL, ResponsesWireDialect,
    };
    use siumai_transport::{EndpointConfig, OfficialOrigin, RequestHeaders};

    use crate::configured::codec_policy::{
        ChatCodecPolicy, PreparedChatCall, PreparedResponsesCall, ResponsesCodecPolicy,
    };

    #[derive(Debug)]
    struct ProtocolHeaderPolicy;

    impl ChatCodecPolicy for ProtocolHeaderPolicy {
        fn name(&self) -> &'static str {
            "protocol-header-test"
        }

        fn prepare(
            &self,
            _model: &ModelId,
            request: LanguageRequest,
            dialect: ChatCompletionsDialect,
            extra: BTreeMap<String, Value>,
        ) -> Result<PreparedChatCall, Error> {
            let headers = RequestHeaders::new()
                .try_insert(
                    HeaderName::from_static("x-provider-beta"),
                    HeaderValue::from_static("enabled"),
                )
                .map_err(|source| {
                    Error::new(ErrorKind::InvalidInput, "test protocol header is invalid")
                        .with_source(source)
                })?;
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

    #[derive(Debug)]
    struct CompatibleResponsesPolicy;

    impl ResponsesCodecPolicy for CompatibleResponsesPolicy {
        fn name(&self) -> &'static str {
            "compatible-responses-test"
        }

        fn prepare(
            &self,
            _model: &ModelId,
            request: LanguageRequest,
            extra: BTreeMap<String, Value>,
        ) -> Result<PreparedResponsesCall, Error> {
            Ok(PreparedResponsesCall {
                request,
                extra,
                headers: RequestHeaders::new(),
                native_tools: Vec::new(),
                function_tools: BTreeMap::new(),
                warnings: Vec::new(),
            })
        }
    }

    fn dual_mode_profile(base_url: &str) -> OpenAiCompatibleProfile {
        dual_mode_profile_with_platforms("public-api", "public-api")
            .unwrap()
            .with_test_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
    }

    fn dual_mode_profile_with_platforms(
        chat_platform: &str,
        responses_platform: &str,
    ) -> Result<OpenAiCompatibleProfile, OpenAiCompatibleConfigError> {
        let provider = ProviderId::new("dual-test").unwrap();
        let chat_scope = SupportScope::new(
            provider.clone(),
            PlatformId::new(chat_platform).unwrap(),
            ModelFamily::Language,
            ProtocolId::new(CHAT_PROTOCOL_ID).unwrap(),
            ApiModeId::new(CHAT_API_MODE_ID).unwrap(),
        );
        let responses_scope = SupportScope::new(
            provider,
            PlatformId::new(responses_platform).unwrap(),
            ModelFamily::Language,
            ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap(),
            ApiModeId::new(RESPONSES_API_MODE_ID).unwrap(),
        );
        let evidence = VerificationEvidence::new(
            OfficialSource::new("https://api.dual.example/docs").unwrap(),
            VerificationDate::new(NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()),
            ProtocolContractId::new("dual-openai-family-2026-08").unwrap(),
        );
        let model = ModelId::new("dual-model").unwrap();
        let catalog = ModelCatalog::new([
            ModelProfile::new(
                model.clone(),
                responses_scope.clone(),
                [ModelOperation::Generate, ModelOperation::Stream],
                ModelLifecycle::Retired { replacement: None },
                evidence.clone(),
            )
            .unwrap(),
            ModelProfile::new(
                model,
                chat_scope.clone(),
                [ModelOperation::Generate, ModelOperation::Stream],
                ModelLifecycle::Retired { replacement: None },
                evidence.clone(),
            )
            .unwrap(),
        ])
        .unwrap();
        let profile = ProviderProfile::verified(
            ProfileId::new("dual-test").unwrap(),
            vec![
                VerifiedSupportClaim::new(
                    responses_scope,
                    VerifiedFidelity::Compatible,
                    ApiStability::Stable,
                    evidence.clone(),
                ),
                VerifiedSupportClaim::new(
                    chat_scope,
                    VerifiedFidelity::Compatible,
                    ApiStability::Stable,
                    evidence,
                ),
            ],
            catalog,
        )
        .unwrap();
        let official_endpoint = EndpointConfig::official(
            "https://api.dual.example/v1",
            OfficialOrigin::new("https://api.dual.example").unwrap(),
        )
        .unwrap();

        OpenAiCompatibleProfile::verified_chat_and_responses(
            profile,
            official_endpoint,
            ChatCompletionsDialect::generic(),
        )
    }

    #[test]
    fn dual_mode_profile_rejects_mixed_platforms_on_one_endpoint() {
        assert!(matches!(
            dual_mode_profile_with_platforms("chat-platform", "responses-platform"),
            Err(OpenAiCompatibleConfigError::SharedEndpointScopeMismatch)
        ));
    }

    #[test]
    fn profile_kind_constrains_replay_audience() {
        let verified = dual_mode_profile_with_platforms("public-api", "public-api").unwrap();
        assert!(matches!(
            verified.with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("caller-relay").unwrap(),
            )),
            Err(OpenAiCompatibleConfigError::ReplayAudienceMismatch)
        ));

        let generic = OpenAiCompatibleProfile::public_custom(
            ProviderId::new("custom-test").unwrap(),
            "https://relay.example/v1",
            ReplayDomainId::new("caller-relay").unwrap(),
            OpenAiCompatibleApiMode::Responses,
        )
        .unwrap();
        assert!(matches!(
            generic.with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("official").unwrap(),
            )),
            Err(OpenAiCompatibleConfigError::ReplayAudienceMismatch)
        ));
    }

    #[test]
    fn custom_endpoint_preserves_explicit_shared_address_policy() {
        let endpoint =
            EndpointConfig::shared_address_space_explicit("http://100.64.0.42:8080").unwrap();
        let profile = OpenAiCompatibleProfile::custom_endpoint(
            ProviderId::new("caller-relay").unwrap(),
            endpoint,
            ReplayDomain::custom(ReplayDomainId::new("caller-relay").unwrap()),
            OpenAiCompatibleApiMode::Responses,
        )
        .unwrap()
        .with_responses_wire_dialect(ResponsesWireDialect::compatible());

        let scope = profile.scope(OpenAiCompatibleApiMode::Responses).unwrap();
        assert_eq!(scope.platform().unwrap().as_str(), "local");
        assert!(!scope.replay_domain().unwrap().audience().is_official());
        assert!(matches!(
            profile.endpoint().policy(),
            siumai_transport::EndpointPolicy::LocalExplicit(
                siumai_transport::LocalNetworkGrant::SharedAddressSpace
            )
        ));
    }

    #[test]
    fn static_validation_rejects_credentials_and_protected_defaults_synchronously() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        assert!(
            OpenAiCompatibleProvider::builder(
                profile.clone(),
                OpenAiCompatibleCredential::api_key("")
            )
            .build()
            .is_err()
        );
        assert!(
            OpenAiCompatibleProvider::builder(
                profile,
                OpenAiCompatibleCredential::unauthenticated()
            )
            .with_default_option(
                OpenAiCompatibleApiMode::ChatCompletions,
                "model",
                Value::String("rewritten".to_string()),
            )
            .build()
            .is_err()
        );
    }

    #[tokio::test]
    async fn chat_codec_can_add_non_credential_protocol_headers() {
        let mut server = mockito::Server::new_async().await;
        let chat_mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("x-provider-beta", "enabled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"chat-header","model":"dual-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let profile = dual_mode_profile(&format!("{}/v1", server.url()))
            .with_chat_codec_policy(Arc::new(ProtocolHeaderPolicy));
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();

        let response = provider
            .chat_completions("dual-model")
            .unwrap()
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(response.id(), Some("chat-header"));
        chat_mock.assert_async().await;
    }

    #[test]
    fn thousands_of_models_share_one_runtime_and_registration_uses_same_scope() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let runtime = Arc::as_ptr(&provider.runtime);
        for index in 0..10_000 {
            let model = provider.language(format!("future:{index}")).unwrap();
            assert_eq!(Arc::as_ptr(&model.runtime), runtime);
        }

        let direct = provider.language("future:model").unwrap();
        let erased = provider
            .registration()
            .language_model(ModelId::new("future:model").unwrap())
            .unwrap();
        assert_eq!(direct.descriptor(), erased.descriptor());
        assert_eq!(
            direct.descriptor().instance_id(),
            erased.descriptor().instance_id()
        );
    }

    #[test]
    fn cloned_profile_builds_receive_distinct_instance_identities() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let first = OpenAiCompatibleProvider::builder(
            profile.clone(),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let second = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();

        let first_model = first.language("future:model").unwrap();
        let second_model = second.language("future:model").unwrap();

        assert_ne!(
            first_model.descriptor().instance_id(),
            second_model.descriptor().instance_id()
        );
    }

    #[test]
    fn exact_raw_options_reach_only_the_selected_compatible_instance() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let first = OpenAiCompatibleProvider::builder(
            profile.clone(),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let second = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let first_model = first.language("future:model").unwrap();
        let second_model = second.language("future:model").unwrap();
        let options = CallOptions::default()
            .with_raw_provider_options_for(
                &first_model,
                json!({"future_compatible_field": "enabled"}),
            )
            .unwrap();

        let selected = first
            .runtime
            .options_for(
                &first_model,
                OpenAiCompatibleApiMode::ChatCompletions,
                &options,
            )
            .unwrap();
        assert_eq!(
            selected.raw.as_ref().unwrap()["future_compatible_field"],
            "enabled"
        );
        assert!(!selected.typed.contains_key("future_compatible_field"));
        assert!(matches!(
            second.runtime.options_for(
                &second_model,
                OpenAiCompatibleApiMode::ChatCompletions,
                &options,
            ),
            Err(ProviderOptionError::ExactTargetMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn identity_codec_applies_raw_only_to_the_final_request_body() {
        let mut server = mockito::Server::new_async().await;
        let chat_mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::Regex(
                r#"\"future_compatible_field\":\"enabled\""#.to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"chat-raw","model":"future:model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            format!("{}/v1", server.url()),
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let model = provider.language("future:model").unwrap();
        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"future_compatible_field": "enabled"}))
            .unwrap();

        let response = model
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                options,
            )
            .await
            .unwrap();

        assert_eq!(response.id(), Some("chat-raw"));
        chat_mock.assert_async().await;
    }

    #[tokio::test]
    async fn unreviewed_branded_codec_rejects_raw_before_transport() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:9/v1",
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap()
        .with_chat_codec_policy(Arc::new(ProtocolHeaderPolicy));
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let model = provider.language("future:model").unwrap();
        let options = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"future_field": true}))
            .unwrap();

        let error = model
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                options,
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "InvalidInput: provider options are invalid for Chat Completions"
        );
        assert!(
            error
                .sensitive_source()
                .unwrap()
                .expose()
                .to_string()
                .contains("raw provider options")
        );
    }

    #[tokio::test]
    async fn dual_mode_profile_routes_exact_registrations_through_one_runtime() {
        let mut server = mockito::Server::new_async().await;
        let responses_mock = server
            .mock("POST", "/v1/responses")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"dual-model\""#.to_string()),
                mockito::Matcher::Regex(r#"\"input\":"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"resp-dual","object":"response","created_at":1785811200,"model":"dual-model","status":"completed","output":[{"id":"msg-dual","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"responses-ok","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"error":null,"incomplete_details":null,"reasoning":null}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let chat_mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"dual-model\""#.to_string()),
                mockito::Matcher::Regex(r#"\"messages\":"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"chat-dual","model":"dual-model","choices":[{"index":0,"message":{"role":"assistant","content":"chat-ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let provider = OpenAiCompatibleProvider::builder(
            dual_mode_profile(&format!("{}/v1", server.url())),
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();

        assert_eq!(
            provider.recommended_mode(),
            OpenAiCompatibleApiMode::Responses
        );
        let recommended = provider.registration();
        let responses_registration = provider.responses_registration().unwrap();
        let chat_registration = provider.chat_completions_registration().unwrap();
        assert_eq!(
            recommended.scope(ModelFamily::Language),
            responses_registration.scope(ModelFamily::Language)
        );
        assert_eq!(
            responses_registration
                .api_mode(ModelFamily::Language)
                .unwrap()
                .as_str(),
            RESPONSES_API_MODE_ID
        );
        assert_eq!(
            responses_registration
                .protocol(ModelFamily::Language)
                .unwrap()
                .as_str(),
            OPENAI_RESPONSES_PROTOCOL
        );
        assert_eq!(
            chat_registration
                .api_mode(ModelFamily::Language)
                .unwrap()
                .as_str(),
            CHAT_API_MODE_ID
        );
        assert_eq!(
            chat_registration
                .protocol(ModelFamily::Language)
                .unwrap()
                .as_str(),
            CHAT_PROTOCOL_ID
        );
        assert_ne!(
            responses_registration.scope(ModelFamily::Language),
            chat_registration.scope(ModelFamily::Language)
        );
        let retired_profiles = provider
            .profile()
            .provider_profile()
            .catalog()
            .unwrap()
            .iter()
            .filter(|entry| entry.model().as_str() == "dual-model")
            .collect::<Vec<_>>();
        assert_eq!(retired_profiles.len(), 2);
        assert!(
            retired_profiles
                .iter()
                .all(|entry| matches!(entry.lifecycle(), ModelLifecycle::Retired { .. }))
        );

        let responses = provider.responses("dual-model").unwrap();
        let chat = provider.chat_completions("dual-model").unwrap();
        assert!(Arc::ptr_eq(&responses.runtime, &chat.runtime));
        assert_eq!(
            responses.descriptor().scope(),
            responses_registration
                .scope(ModelFamily::Language)
                .expect("responses language scope")
        );
        assert_eq!(
            chat.descriptor().scope(),
            chat_registration
                .scope(ModelFamily::Language)
                .expect("chat language scope")
        );
        assert_eq!(
            provider.language("dual-model").unwrap().api_mode(),
            OpenAiCompatibleApiMode::Responses
        );

        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
        let responses_result = responses
            .generate(request.clone(), CallOptions::default())
            .await
            .unwrap();
        let chat_result = chat
            .generate(request, CallOptions::default())
            .await
            .unwrap();
        assert_eq!(responses_result.id(), Some("resp-dual"));
        assert_eq!(chat_result.id(), Some("chat-dual"));

        let future = ModelId::new("future:model").unwrap();
        for registration in [&responses_registration, &chat_registration] {
            assert!(registration.language_model(future.clone()).is_ok());
        }
        responses_mock.assert_async().await;
        chat_mock.assert_async().await;
    }

    #[tokio::test]
    async fn responses_stream_has_one_terminal_and_incomplete_eof_fails() {
        let mut server = mockito::Server::new_async().await;
        let completed = server
            .mock("POST", "/v1/responses")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"dual-model\""#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp-stream\",\"created_at\":1785811200,\"model\":\"dual-model\",\"status\":\"in_progress\",\"output\":[],\"usage\":null,\"error\":null,\"incomplete_details\":null,\"reasoning\":null}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"msg-compatible\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}\n\n",
                "data: {\"type\":\"response.output_item.done\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"msg-compatible\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"compatible\",\"annotations\":[]}]}}\n\n",
                "data: {\"type\":\"response.completed\",\"sequence_number\":3,\"response\":{\"id\":\"resp-stream\",\"created_at\":1785811200,\"model\":\"dual-model\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"compatible\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":0,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":0},\"error\":null,\"incomplete_details\":null,\"reasoning\":null}}\n\n",
            ))
            .expect(1)
            .create_async()
            .await;
        let incomplete = server
            .mock("POST", "/v1/responses")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"future:eof\""#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(
                "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp-eof\",\"created_at\":1785811200,\"model\":\"future:eof\",\"status\":\"in_progress\",\"output\":[],\"usage\":null,\"error\":null,\"incomplete_details\":null,\"reasoning\":null}}\n\n",
            )
            .expect(1)
            .create_async()
            .await;
        let profile = dual_mode_profile(&format!("{}/v1", server.url()))
            .with_responses_codec_policy(Arc::new(CompatibleResponsesPolicy))
            .with_responses_wire_dialect(ResponsesWireDialect::compatible());
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);

        let completed_events = provider
            .responses("dual-model")
            .unwrap()
            .stream(request.clone(), CallOptions::default())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            completed_events
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        assert!(matches!(
            completed_events.last(),
            Some(siumai_core::LanguageStreamEvent::Terminal(
                siumai_core::StreamTerminal::Completed { response }
            )) if response.id() == Some("resp-stream")
                && response.content().iter().any(
                    |part| matches!(part, ContentPart::Text { text } if text == "compatible")
                )
        ));

        let incomplete_events = provider
            .responses("future:eof")
            .unwrap()
            .stream(request, CallOptions::default())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            incomplete_events
                .iter()
                .filter(|event| event.terminal().is_some())
                .count(),
            1
        );
        assert!(matches!(
            incomplete_events.last(),
            Some(siumai_core::LanguageStreamEvent::Terminal(
                siumai_core::StreamTerminal::Failed { error, .. }
            )) if error.kind() == siumai_core::ErrorKind::UnexpectedEof
        ));
        completed.assert_async().await;
        incomplete.assert_async().await;
    }

    #[tokio::test]
    async fn direct_and_erased_models_use_the_same_authenticated_wire_contract() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_header("accept", "application/json")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(r#"\"model\":\"future:model\""#.to_string()),
                mockito::Matcher::Regex(r#"\"content\":\"hello\""#.to_string()),
                mockito::Matcher::Regex(r#"\"stream\":false"#.to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"chat-1","model":"future:model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            )
            .expect(2)
            .create_async()
            .await;
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            format!("{}/v1", server.url()),
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::api_key("test-key"),
        )
        .build()
        .unwrap();
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);

        let direct = provider.language("future:model").unwrap();
        let direct_response = direct
            .generate(request.clone(), CallOptions::default())
            .await
            .unwrap();
        let erased = provider
            .registration()
            .language_model(ModelId::new("future:model").unwrap())
            .unwrap();
        let erased_response = erased
            .generate(request, CallOptions::default())
            .await
            .unwrap();

        assert_eq!(direct_response, erased_response);
        assert!(direct_response.warnings().is_empty());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn stream_has_one_completed_terminal_and_preserves_usage_only_chunk() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::Regex(r#"\"stream\":true"#.to_string()))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "data: {\"id\":\"chat-1\",\"model\":\"future:model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":1,\"total_tokens\":1}}\n\n",
                "data: [DONE]\n\n"
            ))
            .expect(1)
            .create_async()
            .await;
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            format!("{}/v1", server.url()),
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let model = provider.language("future:model").unwrap();
        let events = model
            .stream(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default(),
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
        assert!(events.iter().any(|event| matches!(
            event,
            siumai_core::LanguageStreamEvent::Usage(usage)
                if usage.usage().input_tokens == siumai_core::UsageValue::Known(0)
        )));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn provider_error_keeps_remote_canaries_off_default_diagnostics() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(400)
            .with_header("x-request-id", "safe-request-id")
            .with_header("x-private-canary", "canary-header-secret")
            .with_body("canary-body-secret")
            .create_async()
            .await;
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            format!("{}/v1", server.url()),
            ReplayDomainId::new("local-test").unwrap(),
            OpenAiCompatibleApiMode::ChatCompletions,
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let error = provider
            .language("future:model")
            .unwrap()
            .generate(
                LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
                CallOptions::default(),
            )
            .await
            .unwrap_err();
        let debug = format!("{error:?}");
        assert!(!debug.contains("canary-header-secret"));
        assert!(!debug.contains("canary-body-secret"));
        assert_eq!(
            error.diagnostics().and_then(|value| value.status()),
            Some(400)
        );
        assert_eq!(
            error.diagnostics().and_then(|value| value.request_id()),
            Some("safe-request-id")
        );
    }
}
