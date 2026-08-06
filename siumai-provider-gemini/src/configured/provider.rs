use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use http::header::{HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{
    Error as CoreError, ErrorKind, ImageModel, ImageModelProvider, ModelId, ModelLookupError,
    ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision, Provider,
    ProviderOptionError, ProviderOptions, ProviderRegistration, ProviderScope, UnsupportedReason,
};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, EndpointConfig, EndpointError,
    EndpointPolicy, OfficialOrigin, ProviderTransport, RetryPolicy, TransportConfigError,
    TransportLimits,
};
use thiserror::Error;

use super::model::GoogleImageModel;
use super::models::is_current;
use super::options::GoogleImageOptions;
use super::profile::{
    API_MODE_ID, GoogleImageProfile, GoogleImageProfileError, PROTOCOL_ID, PROVIDER_ID,
};

const OFFICIAL_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const OFFICIAL_ORIGIN: &str = "https://generativelanguage.googleapis.com";
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

/// Authentication used by one configured Google runtime.
#[derive(Clone)]
pub enum GoogleCredential {
    ApiKey(SecretString),
    Unauthenticated,
}

impl GoogleCredential {
    pub fn api_key(value: impl Into<String>) -> Self {
        Self::ApiKey(SecretString::from(value.into()))
    }

    /// Explicitly disable authentication for a trusted local test double or gateway.
    pub fn unauthenticated() -> Self {
        Self::Unauthenticated
    }

    fn validate(&self) -> Result<(), GoogleImageConfigError> {
        let Self::ApiKey(value) = self else {
            return Ok(());
        };
        let value = value.expose_secret();
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > MAX_CREDENTIAL_BYTES
            || HeaderValue::from_str(value).is_err()
        {
            return Err(GoogleImageConfigError::InvalidCredential);
        }
        Ok(())
    }

    fn into_auth(self) -> Arc<dyn AuthApplier> {
        match self {
            Self::ApiKey(value) => Arc::new(GoogleApiKeyAuth { value }),
            Self::Unauthenticated => Arc::new(siumai_transport::NoAuth),
        }
    }
}

impl fmt::Debug for GoogleCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GoogleCredential")
            .field(&match self {
                Self::ApiKey(_) => "ApiKey([REDACTED])",
                Self::Unauthenticated => "Unauthenticated",
            })
            .finish()
    }
}

struct GoogleApiKeyAuth {
    value: SecretString,
}

#[async_trait]
impl AuthApplier for GoogleApiKeyAuth {
    async fn apply(
        &self,
        _context: AuthContext<'_>,
        _refresh: AuthRefresh,
    ) -> Result<CredentialPatch, CoreError> {
        let value = HeaderValue::from_str(self.value.expose_secret()).map_err(|source| {
            CoreError::new(
                ErrorKind::Authentication,
                "Google API key could not be encoded as a request header",
            )
            .with_source(source)
        })?;
        CredentialPatch::new()
            .try_insert(HeaderName::from_static("x-goog-api-key"), value)
            .map_err(|source| {
                CoreError::new(
                    ErrorKind::Configuration,
                    "Google authentication conflicts with the transport contract",
                )
                .with_source(source)
            })
    }
}

/// A synchronously configured Google image provider using Gemini Interactions.
#[derive(Clone)]
pub struct GoogleImageProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
    profile: GoogleImageProfile,
}

impl GoogleImageProvider {
    pub fn builder(credential: GoogleCredential) -> GoogleImageProviderBuilder {
        GoogleImageProviderBuilder::new(credential)
    }

    pub fn image(&self, model: impl Into<String>) -> Result<GoogleImageModel, ModelLookupError> {
        let model = ModelId::new(model.into())?;
        Ok(self.create_image_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_image(
            self.runtime.scope.clone(),
            self.runtime.policy.clone(),
            Arc::new(move |model| {
                Ok(Arc::new(provider.create_image_model(model)) as Arc<dyn ImageModel>)
            }),
        )
    }

    pub fn profile(&self) -> &GoogleImageProfile {
        &self.profile
    }

    fn create_image_model(&self, model: ModelId) -> GoogleImageModel {
        GoogleImageModel::new(self.runtime.clone(), model)
    }
}

impl Provider for GoogleImageProvider {
    fn provider_id(&self) -> &siumai_core::ProviderId {
        self.runtime.scope.provider_id()
    }
}

impl ImageModelProvider for GoogleImageProvider {
    type Model = GoogleImageModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl fmt::Debug for GoogleImageProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GoogleImageProvider")
            .field("scope", &self.runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

/// Builder for one shared Google Interactions image runtime.
pub struct GoogleImageProviderBuilder {
    credential: GoogleCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    default_options: GoogleImageOptions,
}

impl GoogleImageProviderBuilder {
    fn new(credential: GoogleCredential) -> Self {
        let endpoint = OfficialOrigin::new(OFFICIAL_ORIGIN)
            .and_then(|origin| EndpointConfig::official(OFFICIAL_BASE_URL, origin));
        Self {
            credential,
            endpoint,
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: None,
            call_timeout: None,
            read_timeout: None,
            default_options: GoogleImageOptions::default(),
        }
    }

    /// Replace the default official endpoint with an explicitly policy-bound endpoint.
    pub fn with_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.endpoint = Ok(endpoint);
        self
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

    pub fn with_default_options(mut self, options: GoogleImageOptions) -> Self {
        self.default_options = options;
        self
    }

    /// Validate static settings and create exactly one shared transport runtime.
    pub fn build(self) -> Result<GoogleImageProvider, GoogleImageConfigError> {
        self.credential.validate()?;
        ProviderOptions::typed(&self.default_options)?;
        let endpoint = self.endpoint?;
        let verified_endpoint = matches!(endpoint.policy(), EndpointPolicy::Official(_));
        let profile = if verified_endpoint {
            GoogleImageProfile::current()?
        } else {
            GoogleImageProfile::custom()?
        };
        let mut transport = ProviderTransport::builder(endpoint)
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
        Ok(GoogleImageProvider {
            runtime: Arc::new(ProviderRuntime {
                scope: profile.scope(),
                transport: transport.build()?,
                policy: Arc::new(GoogleImagePolicy { verified_endpoint }),
                default_options: self.default_options,
            }),
            profile,
        })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GoogleImageConfigError {
    #[error("Google API key must be non-empty and at most 16 KiB")]
    InvalidCredential,
    #[error("invalid Google image profile: {0}")]
    Profile(#[from] GoogleImageProfileError),
    #[error("invalid Google image options: {0}")]
    Options(#[from] ProviderOptionError),
    #[error("invalid Google endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Google transport settings: {0}")]
    Transport(#[from] TransportConfigError),
}

pub(crate) struct ProviderRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<GoogleImagePolicy>,
    pub(crate) default_options: GoogleImageOptions,
}

impl fmt::Debug for ProviderRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("default_options", &self.default_options)
            .finish()
    }
}

pub(crate) struct GoogleImagePolicy {
    verified_endpoint: bool,
}

impl ModelPolicy for GoogleImagePolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope().provider_id().as_str() == PROVIDER_ID
            && context.scope().protocol().map(|value| value.as_str()) == Some(PROTOCOL_ID)
            && context.scope().api_mode().map(|value| value.as_str()) == Some(API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if context.operation() != ModelOperation::GenerateImage {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if self.verified_endpoint && is_current(context.model().as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}
