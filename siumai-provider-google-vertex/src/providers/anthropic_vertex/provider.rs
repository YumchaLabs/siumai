use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use siumai_anthropic_compatible::{
    AnthropicCompatibleConfigError, AnthropicCompatibleLanguageModel, AnthropicCompatibleProfile,
    AnthropicCompatibleProvider,
};
use siumai_core::{
    CallOptions, Error, LanguageCallError, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelId, ModelLookupError, Provider,
    ProviderInstanceId, ProviderOptionError, ProviderRegistration, ReplayDomain,
    TypedProviderOptions,
};
use siumai_transport::{AuthApplier, EndpointConfig, ProviderHttpTransportSettings};
use thiserror::Error as ThisError;

use super::annotations::GoogleVertexAnthropicAnnotationResolver;
use super::auth::{GoogleVertexCredential, GoogleVertexCredentialError};
use super::endpoint::{GoogleVertexAnthropicEndpointError, official_endpoint};
use super::options::GoogleVertexAnthropicMessagesOptions;
use super::profile::{GoogleVertexAnthropicProfileError, PROVIDER_ID, profile};
use super::projection::validate_model_segment;

/// Long-lived configured Anthropic-on-Vertex provider.
#[derive(Clone)]
pub struct GoogleVertexAnthropicProvider {
    language: AnthropicCompatibleProvider,
}

impl GoogleVertexAnthropicProvider {
    /// Configure the official Google endpoint with a Google access-token source.
    pub fn builder(
        project: impl Into<String>,
        location: impl Into<String>,
        credential: GoogleVertexCredential,
    ) -> GoogleVertexAnthropicProviderBuilder {
        GoogleVertexAnthropicProviderBuilder::new(project, location, credential)
    }

    /// Configure the official endpoint with an already implemented auth source.
    ///
    /// This is also the only builder path that may select a custom endpoint,
    /// preventing a Google access token from being forwarded to another audience.
    pub fn builder_with_auth(
        project: impl Into<String>,
        location: impl Into<String>,
        auth: Arc<dyn AuthApplier>,
    ) -> GoogleVertexAnthropicProviderBuilder {
        GoogleVertexAnthropicProviderBuilder::with_auth(project, location, auth)
    }

    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<GoogleVertexAnthropicLanguageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        self.language_model(model)
    }

    pub fn registration(&self) -> ProviderRegistration {
        self.language.registration()
    }

    pub fn profile(&self) -> &AnthropicCompatibleProfile {
        self.language.profile()
    }
}

impl Provider for GoogleVertexAnthropicProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.language.provider_id()
    }
}

impl LanguageModelProvider for GoogleVertexAnthropicProvider {
    type Model = GoogleVertexAnthropicLanguageModel;

    fn language_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        validate_model_segment(model.as_str())
            .map_err(|source| ModelLookupError::InvalidModelReference { source })?;
        self.language
            .language_model(model)
            .map(|inner| GoogleVertexAnthropicLanguageModel { inner })
    }
}

impl fmt::Debug for GoogleVertexAnthropicProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GoogleVertexAnthropicProvider")
            .field("provider_id", &PROVIDER_ID)
            .field("runtime", &"shared")
            .finish()
    }
}

enum ConfiguredAuth {
    Credential(GoogleVertexCredential),
    Applied(Arc<dyn AuthApplier>),
}

enum EndpointSelection {
    Official(EndpointConfig),
    Custom(EndpointConfig),
}

/// Stable official replay audience for Anthropic models served by Vertex AI.
pub const GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE: &str = "google-vertex-anthropic";

/// Builder for one immutable, network-free Anthropic-on-Vertex runtime.
pub struct GoogleVertexAnthropicProviderBuilder {
    auth: ConfiguredAuth,
    endpoint: Result<EndpointSelection, GoogleVertexAnthropicEndpointError>,
    replay_domain: Option<ReplayDomain>,
    defaults: GoogleVertexAnthropicMessagesOptions,
    http_transport_settings: ProviderHttpTransportSettings,
}

impl GoogleVertexAnthropicProviderBuilder {
    fn new(
        project: impl Into<String>,
        location: impl Into<String>,
        credential: GoogleVertexCredential,
    ) -> Self {
        Self::from_auth(project, location, ConfiguredAuth::Credential(credential))
    }

    fn with_auth(
        project: impl Into<String>,
        location: impl Into<String>,
        auth: Arc<dyn AuthApplier>,
    ) -> Self {
        Self::from_auth(project, location, ConfiguredAuth::Applied(auth))
    }

    fn from_auth(
        project: impl Into<String>,
        location: impl Into<String>,
        auth: ConfiguredAuth,
    ) -> Self {
        let project = project.into();
        let location = location.into();
        Self {
            auth,
            endpoint: official_endpoint(&project, &location).map(EndpointSelection::Official),
            replay_domain: None,
            defaults: GoogleVertexAnthropicMessagesOptions::default(),
            http_transport_settings: ProviderHttpTransportSettings::default(),
        }
    }

    /// Select an explicit endpoint and drop all named compatibility claims.
    ///
    /// Custom endpoints require `builder_with_auth`; the provider will not send
    /// a Google credential to a caller-selected audience implicitly.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(EndpointSelection::Custom(endpoint));
        self
    }

    /// Bind provider-native history to a non-secret endpoint and caller scope.
    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.replay_domain = Some(replay_domain);
        self
    }

    pub fn with_default_options(mut self, options: GoogleVertexAnthropicMessagesOptions) -> Self {
        self.defaults = options;
        self
    }

    /// Apply the complete provider stateless-HTTP infrastructure settings.
    pub fn with_http_transport_settings(mut self, settings: ProviderHttpTransportSettings) -> Self {
        self.http_transport_settings = settings;
        self
    }

    /// Validate configuration and build the shared runtime without network I/O.
    pub fn build(self) -> Result<GoogleVertexAnthropicProvider, GoogleVertexAnthropicConfigError> {
        self.defaults.validate()?;
        let (endpoint, verified_endpoint) = match self.endpoint? {
            EndpointSelection::Official(endpoint) => (endpoint, true),
            EndpointSelection::Custom(endpoint) => {
                if matches!(&self.auth, ConfiguredAuth::Credential(_)) {
                    return Err(
                        GoogleVertexAnthropicConfigError::CustomEndpointRequiresExplicitAuth,
                    );
                }
                (endpoint, false)
            }
        };
        let replay_domain = match (self.replay_domain, verified_endpoint) {
            (Some(replay_domain), _) => replay_domain,
            (None, true) => {
                return Err(GoogleVertexAnthropicConfigError::OfficialProjectRequiresReplayDomain);
            }
            (None, false) => {
                return Err(GoogleVertexAnthropicConfigError::CustomEndpointRequiresReplayDomain);
            }
        };
        if replay_domain.audience().is_official() != verified_endpoint {
            return Err(GoogleVertexAnthropicConfigError::ReplayAudienceMismatch);
        }
        if verified_endpoint && replay_domain.caller_scope().is_none() {
            return Err(GoogleVertexAnthropicConfigError::OfficialProjectRequiresCallerScope);
        }
        let auth = match self.auth {
            ConfiguredAuth::Credential(credential) => credential.into_auth()?,
            ConfiguredAuth::Applied(auth) => auth,
        };
        let resolver = Arc::new(GoogleVertexAnthropicAnnotationResolver);
        let profile = profile(endpoint, verified_endpoint, replay_domain, resolver)?;
        let builder = AnthropicCompatibleProvider::builder_with_auth(profile, auth)
            .with_provider_instance(ProviderInstanceId::new())
            .with_default_options(self.defaults.to_engine())
            .with_http_transport_settings(self.http_transport_settings);
        Ok(GoogleVertexAnthropicProvider {
            language: builder.build()?,
        })
    }
}

/// Lightweight Claude-on-Vertex language-model handle.
#[derive(Clone)]
pub struct GoogleVertexAnthropicLanguageModel {
    inner: AnthropicCompatibleLanguageModel,
}

impl Model for GoogleVertexAnthropicLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
}

#[async_trait]
impl LanguageModel for GoogleVertexAnthropicLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        let options = options.resolve_deadline().map_err(Error::from)?;
        self.inner.generate(request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let options = options.resolve_deadline().map_err(Error::from)?;
        self.inner.stream(request, options).await
    }
}

impl fmt::Debug for GoogleVertexAnthropicLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GoogleVertexAnthropicLanguageModel")
            .field("descriptor", self.descriptor())
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum GoogleVertexAnthropicConfigError {
    #[error("invalid Anthropic-on-Vertex endpoint: {0}")]
    Endpoint(#[from] GoogleVertexAnthropicEndpointError),
    #[error("custom Vertex endpoints require an explicit AuthApplier")]
    CustomEndpointRequiresExplicitAuth,
    #[error("custom Vertex endpoints require an explicit non-secret replay domain")]
    CustomEndpointRequiresReplayDomain,
    #[error("official Vertex projects require an explicit non-secret replay domain")]
    OfficialProjectRequiresReplayDomain,
    #[error("official Vertex replay domains require a non-secret caller scope")]
    OfficialProjectRequiresCallerScope,
    #[error("replay audience does not match the configured Vertex endpoint policy")]
    ReplayAudienceMismatch,
    #[error("invalid Google Vertex credential: {0}")]
    Credential(#[from] GoogleVertexCredentialError),
    #[error("invalid Google Vertex Anthropic profile: {0}")]
    Profile(#[from] GoogleVertexAnthropicProfileError),
    #[error("invalid Google Vertex Anthropic defaults: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("invalid Anthropic-compatible runtime: {0}")]
    Compatible(#[from] AnthropicCompatibleConfigError),
}
