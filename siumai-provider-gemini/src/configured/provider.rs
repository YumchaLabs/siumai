use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use http::header::{HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use siumai_core::{
    ApiModeId, Error as CoreError, ErrorKind, ImageModel, ImageModelProvider, InvalidId,
    ModelFamily, ModelId, ModelLookupError, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, PlatformId, ProtocolId, Provider, ProviderId, ProviderRegistration,
    ProviderScope, UnsupportedReason,
};
use siumai_transport::{
    AuthApplier, AuthContext, AuthRefresh, CredentialPatch, EndpointConfig, EndpointError,
    OfficialOrigin, ProviderTransport, RetryPolicy, TransportConfigError, TransportLimits,
};
use thiserror::Error;

use super::model::GoogleImagenModel;
use super::options::GoogleImagenOptions;

const OFFICIAL_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const OFFICIAL_ORIGIN: &str = "https://generativelanguage.googleapis.com";
const GOOGLE_PROVIDER_ID: &str = "google";
const GEMINI_API_PLATFORM_ID: &str = "gemini-api";
const IMAGEN_PROTOCOL_ID: &str = "google-imagen";
const IMAGEN_PREDICT_API_MODE_ID: &str = "imagen-predict";
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

    fn validate(&self) -> Result<(), GoogleImagenConfigError> {
        let Self::ApiKey(value) = self else {
            return Ok(());
        };
        let value = value.expose_secret();
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > MAX_CREDENTIAL_BYTES
            || HeaderValue::from_str(value).is_err()
        {
            return Err(GoogleImagenConfigError::InvalidCredential);
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

/// A synchronously configured Google Imagen provider in explicit `:predict` mode.
#[derive(Clone)]
pub struct GoogleImagenProvider {
    pub(crate) runtime: Arc<ProviderRuntime>,
}

impl GoogleImagenProvider {
    pub fn builder(credential: GoogleCredential) -> GoogleImagenProviderBuilder {
        GoogleImagenProviderBuilder::new(credential)
    }

    pub fn imagen(&self, model: impl Into<String>) -> Result<GoogleImagenModel, ModelLookupError> {
        let model = ModelId::new(model.into())
            .map_err(|error| ModelLookupError::InvalidReference(error.to_string()))?;
        Ok(self.create_image_model(model))
    }

    pub fn registration(&self) -> ProviderRegistration {
        let provider = self.clone();
        ProviderRegistration::from_scope(self.runtime.scope.clone(), self.runtime.policy.clone())
            .with_image(Arc::new(move |model| {
                Ok(Arc::new(provider.create_image_model(model)) as Arc<dyn ImageModel>)
            }))
    }

    fn create_image_model(&self, model: ModelId) -> GoogleImagenModel {
        GoogleImagenModel::new(self.runtime.clone(), model)
    }
}

impl Provider for GoogleImagenProvider {
    fn scope(&self) -> &ProviderScope {
        &self.runtime.scope
    }
}

impl ImageModelProvider for GoogleImagenProvider {
    type Model = GoogleImagenModel;

    fn image_model(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
        Ok(self.create_image_model(model))
    }
}

impl fmt::Debug for GoogleImagenProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GoogleImagenProvider")
            .field("scope", &self.runtime.scope)
            .field("transport", &"shared")
            .finish()
    }
}

/// Builder for one shared Google Imagen runtime.
pub struct GoogleImagenProviderBuilder {
    credential: GoogleCredential,
    endpoint: Result<EndpointConfig, EndpointError>,
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Option<Duration>,
    call_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    default_options: GoogleImagenOptions,
}

impl GoogleImagenProviderBuilder {
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
            default_options: GoogleImagenOptions::default(),
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

    pub fn with_default_options(mut self, options: GoogleImagenOptions) -> Self {
        self.default_options = options;
        self
    }

    /// Validate static settings and create exactly one shared transport runtime.
    pub fn build(self) -> Result<GoogleImagenProvider, GoogleImagenConfigError> {
        self.credential.validate()?;
        let endpoint = self.endpoint?;
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
        let scope = Arc::new(
            ProviderScope::new(ProviderId::new(GOOGLE_PROVIDER_ID)?)
                .with_platform(PlatformId::new(GEMINI_API_PLATFORM_ID)?)
                .with_protocol(ProtocolId::new(IMAGEN_PROTOCOL_ID)?)
                .with_api_mode(ApiModeId::new(IMAGEN_PREDICT_API_MODE_ID)?),
        );
        Ok(GoogleImagenProvider {
            runtime: Arc::new(ProviderRuntime {
                scope,
                transport: transport.build()?,
                policy: Arc::new(GoogleImagenPolicy),
                default_options: self.default_options,
            }),
        })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GoogleImagenConfigError {
    #[error("Google API key must be non-empty and at most 16 KiB")]
    InvalidCredential,
    #[error("invalid Google provider identifier: {0}")]
    Identifier(#[from] InvalidId),
    #[error("invalid Google endpoint: {0}")]
    Endpoint(#[from] EndpointError),
    #[error("invalid Google transport settings: {0}")]
    Transport(#[from] TransportConfigError),
}

pub(crate) struct ProviderRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) policy: Arc<GoogleImagenPolicy>,
    pub(crate) default_options: GoogleImagenOptions,
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

pub(crate) struct GoogleImagenPolicy;

impl ModelPolicy for GoogleImagenPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let matches_scope = context.scope.provider_id().as_str() == GOOGLE_PROVIDER_ID
            && context.scope.platform().map(PlatformId::as_str) == Some(GEMINI_API_PLATFORM_ID)
            && context.scope.protocol().map(ProtocolId::as_str) == Some(IMAGEN_PROTOCOL_ID)
            && context.scope.api_mode().map(ApiModeId::as_str) == Some(IMAGEN_PREDICT_API_MODE_ID);
        if !matches_scope {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if context.family != ModelFamily::Image {
            return ModelPolicyDecision::unsupported(UnsupportedReason::FamilyNotImplemented);
        }
        if context.operation != ModelOperation::GenerateImage {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        if is_known_imagen_model(context.model.as_str()) {
            ModelPolicyDecision::supported()
        } else {
            ModelPolicyDecision::unknown_model()
        }
    }
}

fn is_known_imagen_model(model: &str) -> bool {
    matches!(
        model,
        "imagen-4.0-generate-001"
            | "imagen-4.0-ultra-generate-001"
            | "imagen-4.0-fast-generate-001"
    )
}
