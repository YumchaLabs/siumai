use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleLanguageModel, AnthropicCompatibleProfile,
    AnthropicCompatibleProvider,
};
use siumai_core::{
    ApiStability, CallOptions, Error, InvalidId, LanguageCallError, LanguageModel,
    LanguageModelProvider, LanguageRequest, LanguageResponse, LanguageStream, Model,
    ModelDescriptor, ModelId, ModelLookupError, NativeSupportScope, NativeSurfaceId,
    NativeSurfaceKind, NativeVerificationEvidence, OfficialSource, ProfileError, Provider,
    ProviderInstanceId, ProviderOptionError, ProviderRegistration, ProviderSupportManifest,
    ReplayDomain, ReplayDomainId, SupportManifestError, TypedProviderOptions, VerificationDate,
    VerifiedFidelity, VerifiedNativeSupportClaim,
};
use siumai_transport::{
    AuthApplier, EndpointConfig, EndpointError, OfficialOrigin, ProviderTransport, RetryPolicy,
    TransportConfigError, TransportLimits,
};
use thiserror::Error as ThisError;

use crate::annotations::{
    AnthropicAnnotationResolver, AnthropicContentOptions, AnthropicMessageCache,
    AnthropicToolOptions,
};
use crate::auth::{AnthropicCredential, AnthropicCredentialError};
use crate::options::AnthropicMessagesOptions;
use crate::profile::{
    API_VERSION, AnthropicProfileError, DEFAULT_BASE_URL, OFFICIAL_ORIGIN, PROVIDER_ID, profile,
};
use crate::resources::{
    AnthropicFiles, AnthropicMessageBatches, AnthropicSkills, AnthropicTokens, NativeRuntime,
};

const SUPPORT_VERIFIED_ON: &str = "2026-08-11";
const FILES_SOURCE: &str = "https://platform.claude.com/docs/en/build-with-claude/files";
const MESSAGE_BATCHES_SOURCE: &str =
    "https://platform.claude.com/docs/en/build-with-claude/batch-processing";
const TOKEN_COUNTING_SOURCE: &str = "https://platform.claude.com/docs/en/api/messages-count-tokens";
const SKILLS_SOURCE: &str = "https://platform.claude.com/docs/en/build-with-claude/skills-guide";

/// Long-lived configured Anthropic provider.
#[derive(Clone)]
pub struct AnthropicProvider {
    language: AnthropicCompatibleProvider,
    native: Arc<NativeRuntime>,
    support_manifest: Arc<ProviderSupportManifest>,
}

impl AnthropicProvider {
    pub fn builder(credential: AnthropicCredential) -> AnthropicProviderBuilder {
        AnthropicProviderBuilder::new(credential)
    }

    /// Configure an already implemented rotating credential source or request signer.
    pub fn builder_with_auth(auth: Arc<dyn AuthApplier>) -> AnthropicProviderBuilder {
        AnthropicProviderBuilder::with_auth(auth)
    }

    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<AnthropicLanguageModel, ModelLookupError> {
        self.language
            .language(model)
            .map(|inner| AnthropicLanguageModel { inner })
    }

    pub fn registration(&self) -> ProviderRegistration {
        self.language.registration()
    }

    pub fn profile(&self) -> &AnthropicCompatibleProfile {
        self.language.profile()
    }

    /// Inspect the exact model and provider-native scopes configured on this provider.
    pub fn support_manifest(&self) -> &ProviderSupportManifest {
        self.support_manifest.as_ref()
    }

    pub fn files(&self) -> AnthropicFiles {
        AnthropicFiles::new(self.native.clone())
    }

    pub fn message_batches(&self) -> AnthropicMessageBatches {
        AnthropicMessageBatches::new(self.native.clone())
    }

    pub fn tokens(&self) -> AnthropicTokens {
        AnthropicTokens::new(self.native.clone())
    }

    pub fn skills(&self) -> AnthropicSkills {
        AnthropicSkills::new(self.native.clone())
    }
}

impl Provider for AnthropicProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.support_manifest.provider_id()
    }
}

impl LanguageModelProvider for AnthropicProvider {
    type Model = AnthropicLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.language
            .language_model(model)
            .map(|inner| AnthropicLanguageModel { inner })
    }
}

impl fmt::Debug for AnthropicProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicProvider")
            .field("provider_id", &PROVIDER_ID)
            .field("language_runtime", &"shared")
            .field("resource_runtime", &"shared")
            .finish()
    }
}

enum ConfiguredAuth {
    Credential(AnthropicCredential),
    Applied(Arc<dyn AuthApplier>),
}

/// Builder for one immutable, network-free Anthropic runtime.
pub struct AnthropicProviderBuilder {
    auth: ConfiguredAuth,
    endpoint: Result<EndpointConfig, EndpointError>,
    custom_endpoint: bool,
    replay_domain: Option<ReplayDomain>,
    caller_scope: Option<ReplayDomainId>,
    defaults: AnthropicMessagesOptions,
    beta_features: Vec<String>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
}

impl AnthropicProviderBuilder {
    fn new(credential: AnthropicCredential) -> Self {
        Self::from_auth(ConfiguredAuth::Credential(credential))
    }

    fn with_auth(auth: Arc<dyn AuthApplier>) -> Self {
        Self::from_auth(ConfiguredAuth::Applied(auth))
    }

    fn from_auth(auth: ConfiguredAuth) -> Self {
        Self {
            auth,
            endpoint: official_endpoint(),
            custom_endpoint: false,
            replay_domain: None,
            caller_scope: None,
            defaults: AnthropicMessagesOptions::default(),
            beta_features: Vec::new(),
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
        }
    }

    /// Replace the provider-owned endpoint with a caller-controlled endpoint.
    ///
    /// The endpoint's transport policy does not grant Anthropic support claims or an
    /// official replay audience. Callers must also select a custom replay domain.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self.custom_endpoint = true;
        self
    }

    /// Select an explicit public HTTPS endpoint. Named model fidelity is not
    /// claimed for custom endpoints even when they use Anthropic-compatible wire shapes.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.endpoint = EndpointConfig::public_custom(base_url);
        self.custom_endpoint = true;
        self
    }

    /// Bind provider-native history to a non-secret endpoint and caller scope.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    /// Bind replay-sensitive Anthropic resources to one caller-owned account or workspace scope.
    ///
    /// The provider keeps ownership of the official or custom replay audience selected by the
    /// endpoint configuration. This method changes only the caller scope and is the preferred path
    /// for Files-in-Messages and hosted-tool continuation on the official endpoint.
    pub fn with_caller_scope(mut self, caller_scope: ReplayDomainId) -> Self {
        self.caller_scope = Some(caller_scope);
        self
    }

    pub fn with_default_options(mut self, defaults: AnthropicMessagesOptions) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn with_beta_feature(mut self, feature: impl Into<String>) -> Self {
        self.beta_features.push(feature.into());
        self
    }

    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Self {
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

    /// Validate static configuration and build shared runtimes without network I/O.
    pub fn build(self) -> Result<AnthropicProvider, AnthropicConfigError> {
        self.defaults.validate()?;
        let endpoint = self.endpoint?;
        let provider_verified_endpoint = !self.custom_endpoint;
        let replay_domain = match (self.replay_domain, provider_verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => ReplayDomain::official(ReplayDomainId::new("anthropic-public-api")?),
            (None, false) => return Err(AnthropicConfigError::CustomEndpointRequiresReplayDomain),
        };
        let replay_domain = match self.caller_scope {
            Some(caller_scope) => replay_domain.with_caller_scope(caller_scope),
            None => replay_domain,
        };
        if replay_domain.audience().is_official() != provider_verified_endpoint {
            return Err(AnthropicConfigError::ReplayAudienceMismatch);
        }
        if provider_verified_endpoint
            && matches!(
                &self.auth,
                ConfiguredAuth::Credential(credential) if credential.is_unauthenticated()
            )
        {
            return Err(AnthropicConfigError::OfficialEndpointRequiresAuthentication);
        }
        let auth = match self.auth {
            ConfiguredAuth::Credential(credential) => credential.into_auth()?,
            ConfiguredAuth::Applied(auth) => auth,
        };
        let resolver = Arc::new(AnthropicAnnotationResolver);
        let profile = profile(
            endpoint.clone(),
            provider_verified_endpoint,
            replay_domain,
            resolver.clone(),
            &self.beta_features,
        )?;
        let native_scope = Arc::new(profile.scope().clone());
        let support_manifest = Arc::new(ProviderSupportManifest::new(
            siumai_core::ProviderId::new(PROVIDER_ID)?,
            [profile.provider_profile().clone()],
            if provider_verified_endpoint {
                native_support_claims()?
            } else {
                Vec::new()
            },
        )?);

        let instance_id = ProviderInstanceId::new();
        let mut language_builder =
            AnthropicCompatibleProvider::builder_with_auth(profile, auth.clone())
                .with_provider_instance(instance_id)
                .with_default_options(self.defaults.to_engine())
                .with_limits(self.limits.clone())
                .with_retry_policy(self.retry_policy);
        let mut resource_builder = ProviderTransport::builder(endpoint)
            .with_auth(auth)
            .with_limits(self.limits)
            .with_retry_policy(self.retry_policy);
        if let Some(timeout) = self.connect_timeout {
            language_builder = language_builder.with_connect_timeout(timeout);
            resource_builder = resource_builder.with_connect_timeout(timeout);
        }
        if let Some(timeout) = self.call_timeout {
            language_builder = language_builder.with_call_timeout(timeout);
            resource_builder = resource_builder.with_call_timeout(timeout);
        }
        if let Some(timeout) = self.read_timeout {
            language_builder = language_builder.with_read_timeout(timeout);
            resource_builder = resource_builder.with_read_timeout(timeout);
        }
        let language = language_builder.build()?;
        let native = Arc::new(NativeRuntime {
            scope: native_scope,
            transport: resource_builder.build()?,
            api_version: Arc::from(API_VERSION),
            beta_features: self.beta_features.into(),
            annotation_resolver: resolver,
        });
        Ok(AnthropicProvider {
            language,
            native,
            support_manifest,
        })
    }
}

/// Lightweight Anthropic language-model handle sharing one configured runtime.
#[derive(Clone)]
pub struct AnthropicLanguageModel {
    inner: AnthropicCompatibleLanguageModel,
}

impl AnthropicLanguageModel {
    /// Validate and prewarm prompt-cache content.
    ///
    /// This is the ordinary Messages operation with `max_tokens: 0`; it shares
    /// the same request policy, codec, transport, response decoder, and cache
    /// breakpoint rules as [`LanguageModel::generate`].
    pub async fn prewarm_cache(
        &self,
        request: LanguageRequest,
        options: AnthropicMessagesOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.prewarm_cache_with_call_options(request, options, CallOptions::default())
            .await
    }

    pub async fn prewarm_cache_with_call_options(
        &self,
        mut request: LanguageRequest,
        options: AnthropicMessagesOptions,
        call_options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        if call_options.has_provider_options() {
            return Err(Error::new(
                siumai_core::ErrorKind::InvalidInput,
                "Anthropic cache prewarming accepts one explicit provider-option layer",
            )
            .into());
        }
        if options.automatic_cache_ttl().is_none() && !has_explicit_cache_marker(&request)? {
            return Err(Error::new(
                siumai_core::ErrorKind::InvalidInput,
                "Anthropic cache prewarming requires automatic caching or an explicit cache annotation",
            )
            .into());
        }
        request.generation.max_output_tokens = Some(0);
        let call_options = call_options
            .with_provider_options_for(&self.inner, &options)
            .map_err(|source| {
                Error::new(
                    siumai_core::ErrorKind::InvalidInput,
                    "Anthropic cache prewarming options are invalid",
                )
                .with_source(source)
            })?;
        self.inner.generate(request, call_options).await
    }
}

impl Model for AnthropicLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
}

#[async_trait]
impl LanguageModel for AnthropicLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.inner.generate(request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.inner.stream(request, options).await
    }
}

impl fmt::Debug for AnthropicLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicLanguageModel")
            .field("descriptor", self.descriptor())
            .field("runtime", &"shared")
            .finish()
    }
}

fn has_explicit_cache_marker(request: &LanguageRequest) -> Result<bool, Error> {
    for message in &request.messages {
        if message
            .annotations()
            .decode::<AnthropicMessageCache>()
            .map_err(annotation_error)?
            .is_some()
        {
            return Ok(true);
        }
        for part in message.content() {
            if part
                .annotations()
                .decode::<AnthropicContentOptions>()
                .map_err(annotation_error)?
                .and_then(|options| options.cache_ttl())
                .is_some()
            {
                return Ok(true);
            }
        }
    }
    for tool in &request.tools {
        if tool
            .annotations()
            .decode::<AnthropicToolOptions>()
            .map_err(annotation_error)?
            .and_then(|options| options.cache_ttl())
            .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn annotation_error(source: siumai_core::ProviderAnnotationError) -> Error {
    Error::new(
        siumai_core::ErrorKind::InvalidInput,
        "invalid Anthropic cache annotation",
    )
    .with_source(source)
}

fn official_endpoint() -> Result<EndpointConfig, EndpointError> {
    EndpointConfig::official(DEFAULT_BASE_URL, OfficialOrigin::new(OFFICIAL_ORIGIN)?)
}

fn native_support_claims() -> Result<Vec<VerifiedNativeSupportClaim>, AnthropicConfigError> {
    let provider = siumai_core::ProviderId::new(PROVIDER_ID)?;
    let platform = siumai_core::PlatformId::new("anthropic-api")?;
    let verified_at =
        VerificationDate::new(NaiveDate::parse_from_str(SUPPORT_VERIFIED_ON, "%Y-%m-%d")?);
    [
        (
            "files",
            NativeSurfaceKind::Resource,
            ApiStability::Experimental,
            FILES_SOURCE,
        ),
        (
            "message-batches",
            NativeSurfaceKind::Job,
            ApiStability::Stable,
            MESSAGE_BATCHES_SOURCE,
        ),
        (
            "token-counting",
            NativeSurfaceKind::Resource,
            ApiStability::Stable,
            TOKEN_COUNTING_SOURCE,
        ),
        (
            "skills",
            NativeSurfaceKind::Resource,
            ApiStability::Experimental,
            SKILLS_SOURCE,
        ),
    ]
    .into_iter()
    .map(|(surface, kind, stability, source)| {
        Ok(VerifiedNativeSupportClaim::new(
            NativeSupportScope::surface(
                provider.clone(),
                platform.clone(),
                kind,
                NativeSurfaceId::new(surface)?,
            ),
            VerifiedFidelity::Native,
            stability,
            NativeVerificationEvidence::new(OfficialSource::new(source)?, verified_at),
        ))
    })
    .collect()
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum AnthropicConfigError {
    #[error("invalid Anthropic endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Anthropic credential: {0}")]
    Credential(#[from] AnthropicCredentialError),
    #[error("the official Anthropic endpoint requires an authenticated credential source")]
    OfficialEndpointRequiresAuthentication,
    #[error("a custom Anthropic endpoint requires an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("replay audience does not match the configured Anthropic endpoint identity")]
    ReplayAudienceMismatch,
    #[error("invalid Anthropic profile: {0}")]
    Profile(#[from] AnthropicProfileError),
    #[error("invalid Anthropic support identity: {0}")]
    SupportIdentity(#[from] InvalidId),
    #[error("invalid Anthropic support evidence: {0}")]
    SupportEvidence(#[from] ProfileError),
    #[error("invalid Anthropic support verification date: {0}")]
    SupportDate(#[from] chrono::ParseError),
    #[error("invalid Anthropic support manifest: {0}")]
    SupportManifest(#[from] SupportManifestError),
    #[error("invalid Anthropic Messages defaults: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("invalid Anthropic language runtime: {0}")]
    Compatible(#[from] AnthropicCompatibleConfigError),
    #[error("invalid Anthropic resource transport: {0}")]
    Transport(#[from] TransportConfigError),
}
