use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use siumai_core::{
    InvalidId, LanguageModel, LanguageModelProvider, ModelId, ModelLookupError, Provider,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, ProviderRegistration, ProviderScope,
};
use siumai_protocol_openai::chat_completions::is_protected_option_field;
use siumai_transport::{
    EndpointError, ProviderTransport, ReplaySafety, RetryPolicy, TransportConfigError,
    TransportLimits,
};
use thiserror::Error;

use super::credentials::{CredentialSourceError, OpenAiCompatibleCredential};
use super::model::OpenAiCompatibleLanguageModel;
use super::policy::{OpenAiCompatibleModelPolicy, RetiredModelBehavior};
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

    pub fn language_model(
        &self,
        model: impl Into<String>,
    ) -> Result<OpenAiCompatibleLanguageModel, ModelLookupError> {
        let model = ModelId::new(model.into())
            .map_err(|error| ModelLookupError::InvalidReference(error.to_string()))?;
        Ok(self.create_language_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_scope(self.runtime.scope.clone(), self.runtime.policy.clone())
            .with_language(Arc::new(move |model| {
                Ok(Arc::new(provider.create_language_model(model)) as Arc<dyn LanguageModel>)
            }))
    }

    pub fn profile(&self) -> &OpenAiCompatibleProfile {
        &self.runtime.profile
    }

    fn create_language_model(&self, model: ModelId) -> OpenAiCompatibleLanguageModel {
        OpenAiCompatibleLanguageModel::new(self.runtime.clone(), model)
    }
}

impl Provider for OpenAiCompatibleProvider {
    fn scope(&self) -> &ProviderScope {
        &self.runtime.scope
    }
}

impl LanguageModelProvider for OpenAiCompatibleProvider {
    type Model = OpenAiCompatibleLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_language_model(model))
    }
}

impl fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProvider")
            .field("scope", &self.runtime.scope)
            .field("profile", &self.runtime.profile)
            .finish()
    }
}

pub struct OpenAiCompatibleProviderBuilder {
    profile: OpenAiCompatibleProfile,
    credential: OpenAiCompatibleCredential,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    default_options: BTreeMap<String, Value>,
    retired_models: RetiredModelBehavior,
}

impl OpenAiCompatibleProviderBuilder {
    fn new(profile: OpenAiCompatibleProfile, credential: OpenAiCompatibleCredential) -> Self {
        Self {
            profile,
            credential,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            default_options: BTreeMap::new(),
            retired_models: RetiredModelBehavior::default(),
        }
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
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

    pub fn with_default_option(mut self, name: impl Into<String>, value: Value) -> Self {
        self.default_options.insert(name.into(), value);
        self
    }

    pub fn with_retired_model_behavior(mut self, behavior: RetiredModelBehavior) -> Self {
        self.retired_models = behavior;
        self
    }

    /// Validate static settings and construct one shared provider runtime.
    pub fn build(self) -> Result<OpenAiCompatibleProvider, OpenAiCompatibleConfigError> {
        self.credential.validate_static()?;
        validate_option_fields(&self.default_options)
            .map_err(OpenAiCompatibleConfigError::InvalidDefaultOption)?;

        let mut transport = ProviderTransport::builder(self.profile.endpoint().clone())
            .with_auth(self.credential.into_auth())
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
        let policy = Arc::new(OpenAiCompatibleModelPolicy::new(
            self.profile.profile_arc(),
            self.profile.support_scope().clone(),
            self.retired_models,
        ));
        Ok(OpenAiCompatibleProvider {
            runtime: Arc::new(ProviderRuntime {
                scope: self.profile.scope().clone(),
                profile: self.profile,
                transport,
                policy,
                option_merger: CompatibleOptionMerger {
                    defaults: self.default_options,
                },
                replay_safety: ReplaySafety::Never,
            }),
        })
    }
}

pub(crate) struct ProviderRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) profile: OpenAiCompatibleProfile,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<OpenAiCompatibleModelPolicy>,
    option_merger: CompatibleOptionMerger,
    pub(crate) replay_safety: ReplaySafety,
}

impl ProviderRuntime {
    pub(crate) fn merge_options(
        &self,
        options: &siumai_core::CallOptions,
    ) -> Result<BTreeMap<String, Value>, ProviderOptionError> {
        let layers = options
            .apply_provider_options(self.scope.provider_id(), ProviderOptionLayers::default())?;
        layers.merge_for(self.scope.provider_id(), &self.option_merger)
    }
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("scope", &self.scope)
            .field("profile_id", self.profile.provider_profile().id())
            .field("transport", &"shared")
            .field(
                "default_option_fields",
                &self.option_merger.defaults.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

struct CompatibleOptionMerger {
    defaults: BTreeMap<String, Value>,
}

impl ProviderOptionMerger for CompatibleOptionMerger {
    type Output = BTreeMap<String, Value>;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        validate_option_fields(options.value()).map_err(|path| ProviderOptionError::Rejected {
            path,
            reason: "field is owned by the canonical Chat Completions request".to_string(),
        })
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            for (name, value) in options.value() {
                merged.insert(name.clone(), value.clone());
            }
        }
        Ok(merged)
    }
}

fn validate_option_fields(options: &impl OptionFields) -> Result<(), String> {
    options
        .field_names()
        .find(|name| is_protected_option_field(name))
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
    #[error("invalid provider endpoint: {0}")]
    Endpoint(EndpointError),
    #[error("invalid provider transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("invalid static credential: {0}")]
    Credential(#[from] CredentialSourceError),
    #[error("verified profile requires evidence-backed support claims")]
    ExpectedVerifiedProfile,
    #[error("OpenAI-compatible profile requires exactly one language support claim")]
    ExpectedSingleLanguageClaim,
    #[error("verified profile endpoint must use an exact official-origin policy")]
    VerifiedEndpointMustBeOfficial,
    #[error("support scope is not OpenAI Chat Completions language")]
    IncompatibleSupportScope,
    #[error("default option `{0}` attempts to override a canonical request field")]
    InvalidDefaultOption(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use siumai_core::{
        CallOptions, LanguageModel, LanguageRequest, Message, MessageRole, Model, ProviderId,
    };

    #[test]
    fn static_validation_rejects_credentials_and_protected_defaults_synchronously() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
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
            .with_default_option("model", Value::String("rewritten".to_string()))
            .build()
            .is_err()
        );
    }

    #[test]
    fn thousands_of_models_share_one_runtime_and_registration_uses_same_scope() {
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            "http://127.0.0.1:11434/v1",
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
            let model = provider.language_model(format!("future:{index}")).unwrap();
            assert_eq!(Arc::as_ptr(&model.runtime), runtime);
        }

        let direct = provider.language_model("future:model").unwrap();
        let erased = provider
            .registration()
            .language_model(ModelId::new("future:model").unwrap())
            .unwrap();
        assert_eq!(direct.descriptor(), erased.descriptor());
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
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::api_key("test-key"),
        )
        .build()
        .unwrap();
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);

        let direct = provider.language_model("future:model").unwrap();
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
        assert!(matches!(
            direct_response.warnings[0].kind(),
            siumai_core::WarningKind::UnknownModel
        ));
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
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let model = provider.language_model("future:model").unwrap();
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
                if usage.input_tokens == siumai_core::UsageValue::Known(0)
        )));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn provider_error_keeps_remote_canaries_off_default_diagnostics() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(400)
            .with_header("x-request-id", "canary-header-secret")
            .with_body("canary-body-secret")
            .create_async()
            .await;
        let profile = OpenAiCompatibleProfile::local_explicit(
            ProviderId::new("local-test").unwrap(),
            format!("{}/v1", server.url()),
        )
        .unwrap();
        let provider = OpenAiCompatibleProvider::builder(
            profile,
            OpenAiCompatibleCredential::unauthenticated(),
        )
        .build()
        .unwrap();
        let error = provider
            .language_model("future:model")
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
    }
}
