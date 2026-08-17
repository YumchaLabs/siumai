use std::fmt;
use std::sync::Arc;

use siumai_core::{
    CallOptions, InvalidId, LanguageModel, LanguageModelProvider, Model, ModelId, ModelLookupError,
    Provider, ProviderInstanceId, ProviderOptionError, ProviderRegistration, ProviderScope,
};
use siumai_transport::{
    AuthApplier, EndpointError, ProviderHttpTransportSettings, ProviderTransport, ReplaySafety,
    TransportConfigError,
};
use thiserror::Error;

use crate::auth::{AnthropicCompatibleCredential, CredentialError};
use crate::model::AnthropicCompatibleLanguageModel;
use crate::options::{MessagesCallOptions, MessagesOptionMerger};
use crate::profile::AnthropicCompatibleProfile;

/// Long-lived, synchronously configured Anthropic Messages-compatible runtime.
#[derive(Clone)]
pub struct AnthropicCompatibleProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
}

impl AnthropicCompatibleProvider {
    pub fn builder(
        profile: AnthropicCompatibleProfile,
        credential: AnthropicCompatibleCredential,
    ) -> AnthropicCompatibleProviderBuilder {
        AnthropicCompatibleProviderBuilder::new(profile, credential)
    }

    /// Configure an already implemented authentication source or request signer.
    pub fn builder_with_auth(
        profile: AnthropicCompatibleProfile,
        auth: Arc<dyn AuthApplier>,
    ) -> AnthropicCompatibleProviderBuilder {
        AnthropicCompatibleProviderBuilder::with_auth(profile, auth)
    }

    /// Build a provider-author integration around an already configured transport.
    ///
    /// The transport must use the exact endpoint owned by `profile`. Its
    /// authentication, settings, retry policy, and admission state remain
    /// immutable and are shared by every model created from the provider.
    #[doc(hidden)]
    pub fn builder_with_transport(
        profile: AnthropicCompatibleProfile,
        transport: ProviderTransport,
    ) -> AnthropicCompatibleProviderBuilder {
        AnthropicCompatibleProviderBuilder::with_transport(profile, transport)
    }

    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<AnthropicCompatibleLanguageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        self.create_language_model(model)
    }

    pub fn language_model(
        &self,
        model: ModelId,
    ) -> Result<AnthropicCompatibleLanguageModel, ModelLookupError> {
        self.create_language_model(model)
    }

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_language(
            self.runtime.scope.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(provider.create_language_model(model)?) as Arc<dyn LanguageModel>)
            }),
        )
    }

    pub fn profile(&self) -> &AnthropicCompatibleProfile {
        &self.runtime.profile
    }

    fn create_language_model(
        &self,
        model: ModelId,
    ) -> Result<AnthropicCompatibleLanguageModel, ModelLookupError> {
        Ok(AnthropicCompatibleLanguageModel::new(
            self.runtime.clone(),
            model,
        ))
    }
}

impl Provider for AnthropicCompatibleProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.scope.provider_id()
    }
}

impl LanguageModelProvider for AnthropicCompatibleProvider {
    type Model = AnthropicCompatibleLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        self.create_language_model(model)
    }
}

impl fmt::Debug for AnthropicCompatibleProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicCompatibleProvider")
            .field("scope", &self.runtime.scope)
            .field("profile", &self.runtime.profile)
            .field("runtime", &"shared")
            .finish()
    }
}

/// Builder for one network-free configured runtime.
pub struct AnthropicCompatibleProviderBuilder {
    profile: AnthropicCompatibleProfile,
    transport_source: ConfiguredTransportSource,
    instance_id: Option<ProviderInstanceId>,
    defaults: MessagesCallOptions,
    replay_safety: ReplaySafety,
    http_transport_settings: Option<ProviderHttpTransportSettings>,
}

impl AnthropicCompatibleProviderBuilder {
    fn new(profile: AnthropicCompatibleProfile, credential: AnthropicCompatibleCredential) -> Self {
        Self {
            profile,
            transport_source: ConfiguredTransportSource::Credential(credential),
            instance_id: None,
            defaults: MessagesCallOptions::default(),
            replay_safety: ReplaySafety::Never,
            http_transport_settings: Some(ProviderHttpTransportSettings::default()),
        }
    }

    fn with_auth(profile: AnthropicCompatibleProfile, auth: Arc<dyn AuthApplier>) -> Self {
        Self {
            profile,
            transport_source: ConfiguredTransportSource::Applied(auth),
            instance_id: None,
            defaults: MessagesCallOptions::default(),
            replay_safety: ReplaySafety::Never,
            http_transport_settings: Some(ProviderHttpTransportSettings::default()),
        }
    }

    fn with_transport(profile: AnthropicCompatibleProfile, transport: ProviderTransport) -> Self {
        Self {
            profile,
            transport_source: ConfiguredTransportSource::Prebuilt(transport),
            instance_id: None,
            defaults: MessagesCallOptions::default(),
            replay_safety: ReplaySafety::Never,
            http_transport_settings: None,
        }
    }

    pub fn with_default_options(mut self, options: MessagesCallOptions) -> Self {
        self.defaults = options;
        self
    }

    /// Reuse the owning branded provider's configured-instance capability.
    #[doc(hidden)]
    pub fn with_provider_instance(mut self, instance_id: ProviderInstanceId) -> Self {
        self.instance_id = Some(instance_id);
        self
    }

    /// Override the conservative default only when the profile proves replay safety.
    pub fn with_replay_safety(mut self, replay_safety: ReplaySafety) -> Self {
        self.replay_safety = replay_safety;
        self
    }

    /// Apply the complete provider stateless-HTTP infrastructure settings.
    pub fn with_http_transport_settings(mut self, settings: ProviderHttpTransportSettings) -> Self {
        self.http_transport_settings = Some(settings);
        self
    }

    /// Validate static configuration and build one shared runtime without network I/O.
    pub fn build(self) -> Result<AnthropicCompatibleProvider, AnthropicCompatibleConfigError> {
        self.profile
            .scope()
            .replay_domain()
            .ok_or(AnthropicCompatibleConfigError::MissingReplayDomain)?;
        let option_merger = MessagesOptionMerger::new(self.defaults)?;
        let transport = match self.transport_source {
            ConfiguredTransportSource::Credential(credential) => {
                credential.validate()?;
                build_transport(
                    &self.profile,
                    credential.into_auth(),
                    required_settings(self.http_transport_settings)?,
                )?
            }
            ConfiguredTransportSource::Applied(auth) => build_transport(
                &self.profile,
                auth,
                required_settings(self.http_transport_settings)?,
            )?,
            ConfiguredTransportSource::Prebuilt(transport) => {
                if self.http_transport_settings.is_some() {
                    return Err(AnthropicCompatibleConfigError::PrebuiltTransportSettingsConflict);
                }
                validate_transport_endpoint(self.profile.endpoint(), &transport)?;
                transport
            }
        };
        let scope = self.profile.scope_arc();
        let instance_id = self.instance_id.unwrap_or_default();
        Ok(AnthropicCompatibleProvider {
            runtime: Arc::new(ProviderRuntime {
                profile: self.profile,
                scope,
                instance_id,
                transport,
                option_merger,
                replay_safety: self.replay_safety,
            }),
        })
    }
}

enum ConfiguredTransportSource {
    Credential(AnthropicCompatibleCredential),
    Applied(Arc<dyn AuthApplier>),
    Prebuilt(ProviderTransport),
}

fn required_settings(
    settings: Option<ProviderHttpTransportSettings>,
) -> Result<ProviderHttpTransportSettings, AnthropicCompatibleConfigError> {
    settings.ok_or(AnthropicCompatibleConfigError::MissingTransportSettings)
}

fn build_transport(
    profile: &AnthropicCompatibleProfile,
    auth: Arc<dyn AuthApplier>,
    settings: ProviderHttpTransportSettings,
) -> Result<ProviderTransport, TransportConfigError> {
    ProviderTransport::builder(profile.endpoint().clone())
        .with_auth(auth)
        .with_http_transport_settings(settings)
        .build()
}

fn validate_transport_endpoint(
    expected: &siumai_transport::EndpointConfig,
    transport: &ProviderTransport,
) -> Result<(), AnthropicCompatibleConfigError> {
    let actual = transport.endpoint();
    if expected.expose_base_url() != actual.expose_base_url()
        || expected.policy() != actual.policy()
    {
        return Err(AnthropicCompatibleConfigError::PrebuiltTransportEndpointMismatch);
    }
    Ok(())
}

pub(crate) struct ProviderRuntime {
    pub(crate) profile: AnthropicCompatibleProfile,
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) instance_id: ProviderInstanceId,
    pub(crate) transport: ProviderTransport,
    option_merger: MessagesOptionMerger,
    pub(crate) replay_safety: ReplaySafety,
}

impl ProviderRuntime {
    pub(crate) fn merge_options_for<M: Model + ?Sized>(
        &self,
        model: &M,
        options: &CallOptions,
    ) -> Result<MessagesCallOptions, ProviderOptionError> {
        let selection = options.provider_options_for(model)?;
        self.option_merger.merge_selected(&selection)
    }
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("scope", &self.scope)
            .field("profile_id", self.profile.provider_profile().id())
            .field("transport", &"shared")
            .field("replay_safety", &self.replay_safety)
            .finish()
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnthropicCompatibleConfigError {
    #[error("invalid compatible identity: {0}")]
    Identity(#[from] InvalidId),
    #[error("invalid compatible endpoint: {0}")]
    Endpoint(EndpointError),
    #[error("invalid Messages request target: {0}")]
    RequestTarget(#[source] siumai_transport::RequestBuildError),
    #[error("invalid provider transport settings: {0}")]
    Transport(#[from] TransportConfigError),
    #[error("configured transport settings are missing")]
    MissingTransportSettings,
    #[error("a prebuilt transport cannot be combined with another settings snapshot")]
    PrebuiltTransportSettingsConflict,
    #[error("prebuilt transport endpoint does not match the compatible profile")]
    PrebuiltTransportEndpointMismatch,
    #[error("invalid static credential: {0}")]
    Credential(#[from] CredentialError),
    #[error("invalid default Messages options: {0}")]
    DefaultOptions(#[from] ProviderOptionError),
    #[error("verified profile requires evidence-backed support claims")]
    ExpectedVerifiedProfile,
    #[error("verified profile requires exactly one Anthropic Messages language claim")]
    MissingMessagesClaim,
    #[error("verified profile contains more than one Anthropic Messages language claim")]
    DuplicateMessagesClaim,
    #[error("verified profile endpoint must use an exact official-origin policy")]
    VerifiedEndpointMustBeOfficial,
    #[error("compatible profile requires an explicit non-secret replay domain")]
    MissingReplayDomain,
    #[error("replay audience does not match compatible profile ownership")]
    ReplayAudienceMismatch,
    #[error("support scope is not the Anthropic Messages language mode")]
    IncompatibleSupportScope,
    #[error("provider profile identity does not match its Messages support scope")]
    ProfileProviderMismatch,
    #[error("API version must be a bounded ASCII identifier")]
    InvalidApiVersion,
    #[error("beta feature must be a bounded ASCII identifier")]
    InvalidBetaFeature,
    #[error("too many beta features; maximum is {maximum}")]
    TooManyBetaFeatures { maximum: usize },
    #[error("combined beta header exceeds {maximum} bytes")]
    BetaHeaderTooLarge { maximum: usize },
}
