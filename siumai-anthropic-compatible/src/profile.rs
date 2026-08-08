use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use siumai_core::{
    ApiModeId, ApiStability, Error, GenericSupportClaim, LanguageRequest, ModelFamily, ModelId,
    PlatformId, ProfileId, ProtocolId, ProviderId, ProviderProfile, ProviderScope, ReplayDomain,
    ReplayDomainId, SupportScope,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, MESSAGES_TARGET, MessagesAnnotationResolver, MessagesEncodingRules,
    NoMessagesAnnotations, PROTOCOL_ID,
};
use siumai_transport::{EndpointConfig, EndpointPolicy, RequestTarget};

use crate::options::MessagesCallOptions;
use crate::projection::{MessagesRequestProjection, NativeMessagesRequestProjection};
use crate::provider::AnthropicCompatibleConfigError;

const MAX_API_VERSION_BYTES: usize = 128;
const MAX_BETA_FEATURES: usize = 64;
const MAX_BETA_HEADER_BYTES: usize = 8 * 1024;

/// Additional execution requirements discovered while preparing one Messages request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MessagesRequestRequirements {
    beta_features: BTreeSet<String>,
}

impl MessagesRequestRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    /// Require one bounded `anthropic-beta` contract for this call.
    pub fn with_beta_feature(mut self, feature: impl Into<String>) -> Result<Self, Error> {
        let feature = feature.into();
        validate_beta_feature(&feature).map_err(config_error)?;
        self.beta_features.insert(feature);
        validate_beta_features(&self.beta_features).map_err(config_error)?;
        Ok(self)
    }

    pub fn beta_features(&self) -> impl Iterator<Item = &str> {
        self.beta_features.iter().map(String::as_str)
    }
}

/// Provider- or dialect-owned preparation for one merged Messages request.
///
/// The compatible engine owns option merging and execution, but it cannot infer branded
/// model rules. A concrete provider can inject this small hook to apply documented defaults,
/// reject model-specific combinations, and add required call-scoped beta contracts before
/// protocol encoding. Unknown model IDs should normally be left unchanged.
pub trait MessagesRequestPolicy: Send + Sync {
    fn prepare(
        &self,
        model: &ModelId,
        request: &LanguageRequest,
        options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error>;
}

/// No-op request policy used by generic compatible profiles.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMessagesRequestPolicy;

impl MessagesRequestPolicy for NoMessagesRequestPolicy {
    fn prepare(
        &self,
        _model: &ModelId,
        _request: &LanguageRequest,
        _options: &mut MessagesCallOptions,
    ) -> Result<MessagesRequestRequirements, Error> {
        Ok(MessagesRequestRequirements::new())
    }
}

/// Immutable endpoint, identity, evidence, and protocol policy for one compatible runtime.
#[derive(Clone)]
pub struct AnthropicCompatibleProfile {
    provider_profile: Arc<ProviderProfile>,
    endpoint: EndpointConfig,
    support_scope: SupportScope,
    scope: Arc<ProviderScope>,
    api_version: Arc<str>,
    messages_target: RequestTarget,
    beta_features: Arc<[String]>,
    annotation_resolver: Arc<dyn MessagesAnnotationResolver>,
    encoding_rules: MessagesEncodingRules,
    request_policy: Arc<dyn MessagesRequestPolicy>,
    request_projection: Arc<dyn MessagesRequestProjection>,
}

impl AnthropicCompatibleProfile {
    /// Construct an evidence-backed compatible profile from one Messages support claim.
    pub fn verified(
        provider_profile: ProviderProfile,
        endpoint: EndpointConfig,
        api_version: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        if !matches!(endpoint.policy(), EndpointPolicy::Official(_)) {
            return Err(AnthropicCompatibleConfigError::VerifiedEndpointMustBeOfficial);
        }
        let claims = provider_profile
            .verified_claims()
            .ok_or(AnthropicCompatibleConfigError::ExpectedVerifiedProfile)?;
        let mut matches = claims
            .iter()
            .filter(|claim| is_messages_scope(claim.scope()));
        let support_scope = matches
            .next()
            .ok_or(AnthropicCompatibleConfigError::MissingMessagesClaim)?
            .scope()
            .clone();
        if matches.next().is_some() {
            return Err(AnthropicCompatibleConfigError::DuplicateMessagesClaim);
        }
        Self::from_parts(provider_profile, endpoint, support_scope, api_version)?
            .with_replay_domain(ReplayDomain::official(ReplayDomainId::new("official")?))
    }

    /// Construct the explicit generic custom-compatible escape hatch.
    ///
    /// Generic profiles make no named model or native-fidelity claim. Unknown model IDs
    /// remain callable with an advisory, and callers must select an explicit endpoint policy.
    pub fn custom(
        profile_id: ProfileId,
        provider: ProviderId,
        platform: PlatformId,
        endpoint: EndpointConfig,
        replay_domain: ReplayDomain,
        api_version: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        let support_scope = messages_scope(provider, platform)?;
        let profile = ProviderProfile::generic(
            profile_id,
            GenericSupportClaim::new(support_scope.clone(), ApiStability::Experimental),
        );
        Self::from_parts(profile, endpoint, support_scope, api_version)?
            .with_replay_domain(replay_domain)
    }

    pub fn public_custom(
        profile_id: ProfileId,
        provider: ProviderId,
        platform: PlatformId,
        base_url: impl AsRef<str>,
        replay_domain: ReplayDomainId,
        api_version: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        let endpoint = EndpointConfig::public_custom(base_url)
            .map_err(AnthropicCompatibleConfigError::Endpoint)?;
        Self::custom(
            profile_id,
            provider,
            platform,
            endpoint,
            ReplayDomain::custom(replay_domain),
            api_version,
        )
    }

    pub fn local_explicit(
        profile_id: ProfileId,
        provider: ProviderId,
        platform: PlatformId,
        base_url: impl AsRef<str>,
        replay_domain: ReplayDomainId,
        api_version: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        let endpoint = EndpointConfig::local_explicit(base_url)
            .map_err(AnthropicCompatibleConfigError::Endpoint)?;
        Self::custom(
            profile_id,
            provider,
            platform,
            endpoint,
            ReplayDomain::custom(replay_domain),
            api_version,
        )
    }

    fn from_parts(
        provider_profile: ProviderProfile,
        endpoint: EndpointConfig,
        support_scope: SupportScope,
        api_version: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        if provider_profile.provider_id() != support_scope.provider() {
            return Err(AnthropicCompatibleConfigError::ProfileProviderMismatch);
        }
        if !is_messages_scope(&support_scope) {
            return Err(AnthropicCompatibleConfigError::IncompatibleSupportScope);
        }
        let api_version = validate_api_version(api_version.into())?;
        let scope = Arc::new(
            ProviderScope::new(support_scope.provider().clone())
                .with_platform(support_scope.platform().clone())
                .with_protocol(support_scope.protocol().clone())
                .with_api_mode(support_scope.api_mode().clone()),
        );
        let messages_target = RequestTarget::new(MESSAGES_TARGET)
            .map_err(AnthropicCompatibleConfigError::RequestTarget)?;
        Ok(Self {
            provider_profile: Arc::new(provider_profile),
            endpoint,
            support_scope,
            scope,
            api_version: Arc::from(api_version),
            messages_target,
            beta_features: Vec::<String>::new().into(),
            annotation_resolver: Arc::new(NoMessagesAnnotations),
            encoding_rules: MessagesEncodingRules::compatible_baseline(),
            request_policy: Arc::new(NoMessagesRequestPolicy),
            request_projection: Arc::new(NativeMessagesRequestProjection),
        })
    }

    /// Override the relative Messages operation target without changing the endpoint audience.
    pub fn with_messages_target(
        mut self,
        target: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        self.messages_target = RequestTarget::new(target.into())
            .map_err(AnthropicCompatibleConfigError::RequestTarget)?;
        Ok(self)
    }

    /// Bind provider-native history to a non-secret configured replay domain.
    pub fn with_replay_domain(
        mut self,
        replay_domain: ReplayDomain,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        let profile_is_verified = self.provider_profile.verified_claims().is_some();
        if replay_domain.audience().is_official() != profile_is_verified {
            return Err(AnthropicCompatibleConfigError::ReplayAudienceMismatch);
        }
        self.scope = Arc::new(
            self.scope
                .as_ref()
                .clone()
                .with_replay_domain(replay_domain),
        );
        Ok(self)
    }

    /// Add one profile-owned beta contract name for the `anthropic-beta` header.
    pub fn with_beta_feature(
        mut self,
        feature: impl Into<String>,
    ) -> Result<Self, AnthropicCompatibleConfigError> {
        let feature = feature.into();
        validate_beta_feature(&feature)?;
        let mut features = self.beta_features.iter().cloned().collect::<BTreeSet<_>>();
        features.insert(feature);
        if features.len() > MAX_BETA_FEATURES {
            return Err(AnthropicCompatibleConfigError::TooManyBetaFeatures {
                maximum: MAX_BETA_FEATURES,
            });
        }
        validate_beta_features(&features)?;
        self.beta_features = features.into_iter().collect::<Vec<_>>().into();
        Ok(self)
    }

    /// Inject the provider-owned durable-annotation projection for this profile.
    pub fn with_annotation_resolver(
        mut self,
        resolver: Arc<dyn MessagesAnnotationResolver>,
    ) -> Self {
        self.annotation_resolver = resolver;
        self
    }

    /// Select bounded request-encoding rules for this compatible dialect.
    pub const fn with_encoding_rules(mut self, rules: MessagesEncodingRules) -> Self {
        self.encoding_rules = rules;
        self
    }

    /// Inject provider-owned model defaults and request compatibility rules.
    pub fn with_request_policy(mut self, policy: Arc<dyn MessagesRequestPolicy>) -> Self {
        self.request_policy = policy;
        self
    }

    /// Inject the bounded wire projection used after canonical Messages encoding.
    pub fn with_request_projection(
        mut self,
        projection: Arc<dyn MessagesRequestProjection>,
    ) -> Self {
        self.request_projection = projection;
        self
    }

    pub fn provider_profile(&self) -> &ProviderProfile {
        &self.provider_profile
    }

    pub fn support_scope(&self) -> &SupportScope {
        &self.support_scope
    }

    pub fn scope(&self) -> &ProviderScope {
        self.scope.as_ref()
    }

    pub(crate) fn scope_arc(&self) -> Arc<ProviderScope> {
        self.scope.clone()
    }

    pub fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    pub fn messages_target(&self) -> &RequestTarget {
        &self.messages_target
    }

    pub fn beta_features(&self) -> &[String] {
        &self.beta_features
    }

    pub const fn encoding_rules(&self) -> &MessagesEncodingRules {
        &self.encoding_rules
    }

    pub(crate) fn beta_header(
        &self,
        requirements: &MessagesRequestRequirements,
    ) -> Result<Option<String>, Error> {
        let mut features = self.beta_features.iter().cloned().collect::<BTreeSet<_>>();
        features.extend(requirements.beta_features.iter().cloned());
        validate_beta_features(&features).map_err(config_error)?;
        if features.is_empty() {
            Ok(None)
        } else {
            Ok(Some(features.into_iter().collect::<Vec<_>>().join(",")))
        }
    }

    pub(crate) fn provider_profile_arc(&self) -> Arc<ProviderProfile> {
        self.provider_profile.clone()
    }

    pub(crate) fn annotation_resolver(&self) -> &Arc<dyn MessagesAnnotationResolver> {
        &self.annotation_resolver
    }

    pub(crate) fn request_policy(&self) -> &Arc<dyn MessagesRequestPolicy> {
        &self.request_policy
    }

    pub(crate) fn request_projection(&self) -> &Arc<dyn MessagesRequestProjection> {
        &self.request_projection
    }
}

impl fmt::Debug for AnthropicCompatibleProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicCompatibleProfile")
            .field("profile_id", self.provider_profile.id())
            .field("scope", &self.scope)
            .field("api_version", &self.api_version)
            .field("messages_target", &self.messages_target)
            .field("beta_features", &self.beta_features)
            .field("encoding_rules", &self.encoding_rules)
            .field("endpoint", &"configured")
            .field("annotation_resolver", &"configured")
            .field("request_policy", &"configured")
            .field("request_projection", &"configured")
            .finish()
    }
}

fn messages_scope(
    provider: ProviderId,
    platform: PlatformId,
) -> Result<SupportScope, AnthropicCompatibleConfigError> {
    Ok(SupportScope::new(
        provider,
        platform,
        ModelFamily::Language,
        ProtocolId::new(PROTOCOL_ID)?,
        ApiModeId::new(API_MODE_ID)?,
    ))
}

fn is_messages_scope(scope: &SupportScope) -> bool {
    scope.family() == ModelFamily::Language
        && scope.protocol().as_str() == PROTOCOL_ID
        && scope.api_mode().as_str() == API_MODE_ID
}

fn validate_api_version(value: String) -> Result<String, AnthropicCompatibleConfigError> {
    if value.is_empty()
        || value.len() > MAX_API_VERSION_BYTES
        || !value.is_ascii()
        || value
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_')))
    {
        return Err(AnthropicCompatibleConfigError::InvalidApiVersion);
    }
    Ok(value)
}

fn validate_beta_feature(feature: &str) -> Result<(), AnthropicCompatibleConfigError> {
    if feature.is_empty()
        || feature.len() > 256
        || !feature.is_ascii()
        || feature
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_')))
    {
        Err(AnthropicCompatibleConfigError::InvalidBetaFeature)
    } else {
        Ok(())
    }
}

fn validate_beta_features(
    features: &BTreeSet<String>,
) -> Result<(), AnthropicCompatibleConfigError> {
    if features.len() > MAX_BETA_FEATURES {
        return Err(AnthropicCompatibleConfigError::TooManyBetaFeatures {
            maximum: MAX_BETA_FEATURES,
        });
    }
    let combined_bytes =
        features.iter().map(String::len).sum::<usize>() + features.len().saturating_sub(1);
    if combined_bytes > MAX_BETA_HEADER_BYTES {
        return Err(AnthropicCompatibleConfigError::BetaHeaderTooLarge {
            maximum: MAX_BETA_HEADER_BYTES,
        });
    }
    Ok(())
}

fn config_error(error: AnthropicCompatibleConfigError) -> Error {
    Error::new(
        siumai_core::ErrorKind::Configuration,
        "Messages request policy produced an invalid beta contract",
    )
    .with_source(error)
}
