//! Structured provider support claims and advisory model catalogs.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDate;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;
use url::Url;

use crate::model::ModelFamily;
use crate::provider::{
    ApiModeId, ModelId, ModelOperation, NativeSurfaceId, PlatformId, ProfileId, ProtocolContractId,
    ProtocolId, ProviderId,
};

/// Fidelity of one provider surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SupportFidelity {
    Native,
    VerifiedCompatible,
    GenericCompatible,
}

/// Fidelity values that require verification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VerifiedFidelity {
    Native,
    Compatible,
}

impl From<VerifiedFidelity> for SupportFidelity {
    fn from(value: VerifiedFidelity) -> Self {
        match value {
            VerifiedFidelity::Native => Self::Native,
            VerifiedFidelity::Compatible => Self::VerifiedCompatible,
        }
    }
}

/// Stability of a public provider API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApiStability {
    Stable,
    Experimental,
}

/// Exact identity covered by one support claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SupportScope {
    provider: ProviderId,
    platform: PlatformId,
    family: ModelFamily,
    protocol: ProtocolId,
    api_mode: ApiModeId,
}

impl SupportScope {
    pub fn new(
        provider: ProviderId,
        platform: PlatformId,
        family: ModelFamily,
        protocol: ProtocolId,
        api_mode: ApiModeId,
    ) -> Self {
        Self {
            provider,
            platform,
            family,
            protocol,
            api_mode,
        }
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn platform(&self) -> &PlatformId {
        &self.platform
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn protocol(&self) -> &ProtocolId {
        &self.protocol
    }

    pub fn api_mode(&self) -> &ApiModeId {
        &self.api_mode
    }
}

/// An official HTTPS documentation or specification source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct OfficialSource(String);

impl OfficialSource {
    pub fn new(value: impl Into<String>) -> Result<Self, ProfileError> {
        let value = value.into();
        let parsed = Url::parse(&value).map_err(|_| ProfileError::InvalidOfficialSource)?;
        if parsed.scheme() != "https"
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err(ProfileError::InvalidOfficialSource);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for OfficialSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// A source-verification date without an implied runtime freshness policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VerificationDate(NaiveDate);

impl VerificationDate {
    pub const fn new(value: NaiveDate) -> Self {
        Self(value)
    }

    pub const fn value(self) -> NaiveDate {
        self.0
    }
}

/// Evidence required for every named support claim and catalog row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    source: OfficialSource,
    verified_at: VerificationDate,
    contract: ProtocolContractId,
}

impl VerificationEvidence {
    pub fn new(
        source: OfficialSource,
        verified_at: VerificationDate,
        contract: ProtocolContractId,
    ) -> Self {
        Self {
            source,
            verified_at,
            contract,
        }
    }

    pub fn source(&self) -> &OfficialSource {
        &self.source
    }

    pub fn verified_at(&self) -> VerificationDate {
        self.verified_at
    }

    pub fn contract(&self) -> &ProtocolContractId {
        &self.contract
    }
}

/// A named support claim that cannot exist without evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedSupportClaim {
    scope: SupportScope,
    fidelity: VerifiedFidelity,
    stability: ApiStability,
    evidence: VerificationEvidence,
}

impl VerifiedSupportClaim {
    pub fn new(
        scope: SupportScope,
        fidelity: VerifiedFidelity,
        stability: ApiStability,
        evidence: VerificationEvidence,
    ) -> Self {
        Self {
            scope,
            fidelity,
            stability,
            evidence,
        }
    }

    pub fn scope(&self) -> &SupportScope {
        &self.scope
    }

    pub fn fidelity(&self) -> VerifiedFidelity {
        self.fidelity
    }

    pub fn stability(&self) -> ApiStability {
        self.stability
    }

    pub fn evidence(&self) -> &VerificationEvidence {
        &self.evidence
    }
}

/// An unverified compatible surface that must not imply named model support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenericSupportClaim {
    scope: SupportScope,
    stability: ApiStability,
}

impl GenericSupportClaim {
    pub fn new(scope: SupportScope, stability: ApiStability) -> Self {
        Self { scope, stability }
    }

    pub fn scope(&self) -> &SupportScope {
        &self.scope
    }

    pub fn stability(&self) -> ApiStability {
        self.stability
    }
}

/// Kind of provider-native surface outside the portable model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NativeSurfaceKind {
    Resource,
    Session,
    Job,
}

/// Technical contract that identifies a provider-native surface.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NativeSurfaceBinding {
    Protocol {
        protocol: ProtocolId,
        api_mode: ApiModeId,
    },
    Surface(NativeSurfaceId),
}

impl NativeSurfaceBinding {
    pub fn protocol(&self) -> Option<&ProtocolId> {
        match self {
            Self::Protocol { protocol, .. } => Some(protocol),
            Self::Surface(_) => None,
        }
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        match self {
            Self::Protocol { api_mode, .. } => Some(api_mode),
            Self::Surface(_) => None,
        }
    }

    pub fn surface_id(&self) -> Option<&NativeSurfaceId> {
        match self {
            Self::Protocol { .. } => None,
            Self::Surface(surface) => Some(surface),
        }
    }
}

/// Exact identity covered by one provider-native support claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeSupportScope {
    provider: ProviderId,
    platform: PlatformId,
    kind: NativeSurfaceKind,
    binding: NativeSurfaceBinding,
}

impl NativeSupportScope {
    pub fn protocol(
        provider: ProviderId,
        platform: PlatformId,
        kind: NativeSurfaceKind,
        protocol: ProtocolId,
        api_mode: ApiModeId,
    ) -> Self {
        Self {
            provider,
            platform,
            kind,
            binding: NativeSurfaceBinding::Protocol { protocol, api_mode },
        }
    }

    pub fn surface(
        provider: ProviderId,
        platform: PlatformId,
        kind: NativeSurfaceKind,
        surface: NativeSurfaceId,
    ) -> Self {
        Self {
            provider,
            platform,
            kind,
            binding: NativeSurfaceBinding::Surface(surface),
        }
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn platform(&self) -> &PlatformId {
        &self.platform
    }

    pub fn kind(&self) -> NativeSurfaceKind {
        self.kind
    }

    pub fn binding(&self) -> &NativeSurfaceBinding {
        &self.binding
    }
}

/// Official evidence for a provider-native surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeVerificationEvidence {
    source: OfficialSource,
    verified_at: VerificationDate,
}

impl NativeVerificationEvidence {
    pub fn new(source: OfficialSource, verified_at: VerificationDate) -> Self {
        Self {
            source,
            verified_at,
        }
    }

    pub fn source(&self) -> &OfficialSource {
        &self.source
    }

    pub fn verified_at(&self) -> VerificationDate {
        self.verified_at
    }
}

/// An evidence-backed provider-native resource, session, or job declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedNativeSupportClaim {
    scope: NativeSupportScope,
    fidelity: VerifiedFidelity,
    stability: ApiStability,
    evidence: NativeVerificationEvidence,
}

impl VerifiedNativeSupportClaim {
    pub fn new(
        scope: NativeSupportScope,
        fidelity: VerifiedFidelity,
        stability: ApiStability,
        evidence: NativeVerificationEvidence,
    ) -> Self {
        Self {
            scope,
            fidelity,
            stability,
            evidence,
        }
    }

    pub fn scope(&self) -> &NativeSupportScope {
        &self.scope
    }

    pub fn fidelity(&self) -> VerifiedFidelity {
        self.fidelity
    }

    pub fn stability(&self) -> ApiStability {
        self.stability
    }

    pub fn evidence(&self) -> &NativeVerificationEvidence {
        &self.evidence
    }
}

/// Lifecycle advice for an exact model ID. It never rewrites requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelLifecycle {
    Active,
    Deprecated { replacement: Option<ModelId> },
    Retired { replacement: Option<ModelId> },
    RollingAlias,
}

impl ModelLifecycle {
    fn replacement(&self) -> Option<&ModelId> {
        match self {
            Self::Deprecated { replacement } | Self::Retired { replacement } => {
                replacement.as_ref()
            }
            Self::Active | Self::RollingAlias => None,
        }
    }
}

/// One evidence-backed exact model declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    model: ModelId,
    scope: SupportScope,
    operations: BTreeSet<ModelOperation>,
    lifecycle: ModelLifecycle,
    evidence: VerificationEvidence,
}

impl ModelProfile {
    pub fn new(
        model: ModelId,
        scope: SupportScope,
        operations: impl IntoIterator<Item = ModelOperation>,
        lifecycle: ModelLifecycle,
        evidence: VerificationEvidence,
    ) -> Result<Self, CatalogError> {
        let operations = operations.into_iter().collect::<BTreeSet<_>>();
        if operations.is_empty() {
            return Err(CatalogError::EmptyOperations { model });
        }
        Ok(Self {
            model,
            scope,
            operations,
            lifecycle,
            evidence,
        })
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn scope(&self) -> &SupportScope {
        &self.scope
    }

    pub fn operations(&self) -> &BTreeSet<ModelOperation> {
        &self.operations
    }

    pub fn lifecycle(&self) -> &ModelLifecycle {
        &self.lifecycle
    }

    pub fn evidence(&self) -> &VerificationEvidence {
        &self.evidence
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CatalogKey {
    model: ModelId,
    scope: SupportScope,
}

/// Immutable, exact-match advisory model catalog.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCatalog {
    entries: BTreeMap<CatalogKey, ModelProfile>,
}

impl ModelCatalog {
    pub fn new(entries: impl IntoIterator<Item = ModelProfile>) -> Result<Self, CatalogError> {
        let mut catalog = Self::default();
        for entry in entries {
            let key = CatalogKey {
                model: entry.model.clone(),
                scope: entry.scope.clone(),
            };
            if catalog.entries.insert(key.clone(), entry).is_some() {
                return Err(CatalogError::DuplicateModelScope { model: key.model });
            }
        }
        catalog.validate_replacements()?;
        Ok(catalog)
    }

    pub fn get(&self, scope: &SupportScope, model: &ModelId) -> Option<&ModelProfile> {
        self.entries.get(&CatalogKey {
            model: model.clone(),
            scope: scope.clone(),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &ModelProfile> {
        self.entries.values()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn validate_replacements(&self) -> Result<(), CatalogError> {
        for (key, entry) in &self.entries {
            let Some(replacement) = entry.lifecycle.replacement() else {
                continue;
            };
            if replacement == &key.model {
                return Err(CatalogError::ReplacementCycle {
                    model: key.model.clone(),
                });
            }
            let replacement_key = CatalogKey {
                model: replacement.clone(),
                scope: key.scope.clone(),
            };
            if !self.entries.contains_key(&replacement_key) {
                return Err(CatalogError::MissingReplacement {
                    model: key.model.clone(),
                    replacement: replacement.clone(),
                });
            }
        }

        for start in self.entries.keys() {
            let mut seen = BTreeSet::new();
            let mut current = start.clone();
            loop {
                if !seen.insert(current.clone()) {
                    return Err(CatalogError::ReplacementCycle {
                        model: start.model.clone(),
                    });
                }
                let Some(next) = self
                    .entries
                    .get(&current)
                    .and_then(|entry| entry.lifecycle.replacement())
                else {
                    break;
                };
                current = CatalogKey {
                    model: next.clone(),
                    scope: current.scope.clone(),
                };
            }
        }
        Ok(())
    }
}

/// A provider-owned profile with mutually exclusive verified and generic forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderProfile {
    id: ProfileId,
    provider: ProviderId,
    kind: ProviderProfileKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProviderProfileKind {
    Verified {
        claims: Box<[VerifiedSupportClaim]>,
        catalog: ModelCatalog,
    },
    Generic {
        claims: Box<[GenericSupportClaim]>,
    },
}

impl ProviderProfile {
    pub fn verified(
        id: ProfileId,
        claims: Vec<VerifiedSupportClaim>,
        catalog: ModelCatalog,
    ) -> Result<Self, ProfileError> {
        let Some(first) = claims.first() else {
            return Err(ProfileError::EmptyVerifiedClaims);
        };
        if claims
            .iter()
            .any(|claim| claim.scope.provider != first.scope.provider)
        {
            return Err(ProfileError::MixedProviders);
        }
        if catalog
            .iter()
            .any(|model| !claims.iter().any(|claim| claim.scope == *model.scope()))
        {
            return Err(ProfileError::CatalogScopeNotClaimed);
        }
        Ok(Self {
            id,
            provider: first.scope.provider.clone(),
            kind: ProviderProfileKind::Verified {
                claims: claims.into_boxed_slice(),
                catalog,
            },
        })
    }

    pub fn generic(id: ProfileId, claim: GenericSupportClaim) -> Self {
        Self {
            id,
            provider: claim.scope.provider.clone(),
            kind: ProviderProfileKind::Generic {
                claims: vec![claim].into_boxed_slice(),
            },
        }
    }

    pub fn generic_many(
        id: ProfileId,
        claims: Vec<GenericSupportClaim>,
    ) -> Result<Self, ProfileError> {
        let Some(first) = claims.first() else {
            return Err(ProfileError::EmptyGenericClaims);
        };
        if claims
            .iter()
            .any(|claim| claim.scope.provider != first.scope.provider)
        {
            return Err(ProfileError::MixedProviders);
        }
        Ok(Self {
            id,
            provider: first.scope.provider.clone(),
            kind: ProviderProfileKind::Generic {
                claims: claims.into_boxed_slice(),
            },
        })
    }

    pub fn id(&self) -> &ProfileId {
        &self.id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider
    }

    pub fn verified_claims(&self) -> Option<&[VerifiedSupportClaim]> {
        match &self.kind {
            ProviderProfileKind::Verified { claims, .. } => Some(claims),
            ProviderProfileKind::Generic { .. } => None,
        }
    }

    pub fn generic_claim(&self) -> Option<&GenericSupportClaim> {
        match &self.kind {
            ProviderProfileKind::Verified { .. } => None,
            ProviderProfileKind::Generic { claims } => claims.first(),
        }
    }

    pub fn generic_claims(&self) -> Option<&[GenericSupportClaim]> {
        match &self.kind {
            ProviderProfileKind::Verified { .. } => None,
            ProviderProfileKind::Generic { claims } => Some(claims),
        }
    }

    pub fn catalog(&self) -> Option<&ModelCatalog> {
        match &self.kind {
            ProviderProfileKind::Verified { catalog, .. } => Some(catalog),
            ProviderProfileKind::Generic { .. } => None,
        }
    }

    fn support_scopes(&self) -> impl Iterator<Item = &SupportScope> {
        self.verified_claims()
            .into_iter()
            .flatten()
            .map(VerifiedSupportClaim::scope)
            .chain(
                self.generic_claims()
                    .into_iter()
                    .flatten()
                    .map(GenericSupportClaim::scope),
            )
    }
}

/// Provider-wide support metadata assembled from portable profiles and native surfaces.
///
/// This is an evidence/introspection surface, not an executable capability allowlist. The provider
/// identity is supplied explicitly so custom or native-only configurations can publish an empty
/// manifest without borrowing identity from an arbitrary execution scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSupportManifest {
    provider: ProviderId,
    profiles: Box<[ProviderProfile]>,
    native_claims: Box<[VerifiedNativeSupportClaim]>,
}

impl ProviderSupportManifest {
    /// Build a manifest for one canonical provider.
    ///
    /// `profiles` and `native_claims` may both be empty when the configured provider intentionally
    /// makes no named support assertion. Callers must not infer official support from an empty
    /// manifest.
    pub fn new(
        provider: ProviderId,
        profiles: impl IntoIterator<Item = ProviderProfile>,
        native_claims: impl IntoIterator<Item = VerifiedNativeSupportClaim>,
    ) -> Result<Self, SupportManifestError> {
        let profiles = profiles.into_iter().collect::<Vec<_>>();
        let native_claims = native_claims.into_iter().collect::<Vec<_>>();

        if let Some(profile) = profiles
            .iter()
            .find(|profile| profile.provider_id() != &provider)
        {
            return Err(SupportManifestError::MixedProviders {
                expected: provider,
                found: profile.provider_id().clone(),
            });
        }

        let mut portable_scopes = BTreeSet::<SupportScope>::new();
        for scope in profiles.iter().flat_map(ProviderProfile::support_scopes) {
            if !portable_scopes.insert(scope.clone()) {
                return Err(SupportManifestError::DuplicatePortableScope {
                    scope: scope.clone(),
                });
            }
        }

        if let Some(claim) = native_claims
            .iter()
            .find(|claim| claim.scope.provider() != &provider)
        {
            return Err(SupportManifestError::MixedProviders {
                expected: provider,
                found: claim.scope.provider().clone(),
            });
        }
        let mut native_scopes = BTreeSet::<NativeSupportScope>::new();
        for claim in &native_claims {
            if !native_scopes.insert(claim.scope().clone()) {
                return Err(SupportManifestError::DuplicateNativeScope {
                    scope: claim.scope().clone(),
                });
            }
        }

        Ok(Self {
            provider,
            profiles: profiles.into_boxed_slice(),
            native_claims: native_claims.into_boxed_slice(),
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider
    }

    pub fn profiles(&self) -> &[ProviderProfile] {
        &self.profiles
    }

    pub fn native_claims(&self) -> &[VerifiedNativeSupportClaim] {
        &self.native_claims
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CatalogError {
    #[error("model `{model}` has no declared operations")]
    EmptyOperations { model: ModelId },
    #[error("model `{model}` is declared more than once for the same support scope")]
    DuplicateModelScope { model: ModelId },
    #[error("model `{model}` names missing replacement `{replacement}`")]
    MissingReplacement {
        model: ModelId,
        replacement: ModelId,
    },
    #[error("model replacement chain containing `{model}` has a cycle")]
    ReplacementCycle { model: ModelId },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProfileError {
    #[error("official source must be an HTTPS URL without credentials")]
    InvalidOfficialSource,
    #[error("verified provider profile requires at least one claim")]
    EmptyVerifiedClaims,
    #[error("generic provider profile requires at least one claim")]
    EmptyGenericClaims,
    #[error("provider profile cannot mix canonical providers")]
    MixedProviders,
    #[error("model catalog contains a scope absent from the verified claims")]
    CatalogScopeNotClaimed,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SupportManifestError {
    #[error("provider support manifest cannot mix `{expected}` and `{found}`")]
    MixedProviders {
        expected: ProviderId,
        found: ProviderId,
    },
    #[error("provider support manifest contains duplicate portable scope {scope:?}")]
    DuplicatePortableScope { scope: SupportScope },
    #[error("provider support manifest contains duplicate provider-native scope {scope:?}")]
    DuplicateNativeScope { scope: NativeSupportScope },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> SupportScope {
        SupportScope::new(
            ProviderId::new("deepseek").unwrap(),
            PlatformId::new("public-api").unwrap(),
            ModelFamily::Language,
            ProtocolId::new("openai").unwrap(),
            ApiModeId::new("chat-completions").unwrap(),
        )
    }

    fn evidence() -> VerificationEvidence {
        VerificationEvidence::new(
            OfficialSource::new("https://api-docs.deepseek.com/").unwrap(),
            VerificationDate::new(NaiveDate::from_ymd_opt(2026, 8, 4).unwrap()),
            ProtocolContractId::new("openai-chat-v1").unwrap(),
        )
    }

    fn model(id: &str, lifecycle: ModelLifecycle) -> ModelProfile {
        ModelProfile::new(
            ModelId::new(id).unwrap(),
            scope(),
            [ModelOperation::Generate, ModelOperation::Stream],
            lifecycle,
            evidence(),
        )
        .unwrap()
    }

    fn verified_profile(provider: &str, family: ModelFamily, api_mode: &str) -> ProviderProfile {
        let scope = SupportScope::new(
            ProviderId::new(provider).unwrap(),
            PlatformId::new("public-api").unwrap(),
            family,
            ProtocolId::new("openai").unwrap(),
            ApiModeId::new(api_mode).unwrap(),
        );
        let claim = VerifiedSupportClaim::new(
            scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            VerificationEvidence::new(
                OfficialSource::new("https://platform.openai.com/docs").unwrap(),
                VerificationDate::new(NaiveDate::from_ymd_opt(2026, 8, 6).unwrap()),
                ProtocolContractId::new(format!("{api_mode}-v1")).unwrap(),
            ),
        );
        ProviderProfile::verified(
            ProfileId::new(format!("{provider}-{api_mode}")).unwrap(),
            vec![claim],
            ModelCatalog::default(),
        )
        .unwrap()
    }

    fn native_evidence() -> NativeVerificationEvidence {
        NativeVerificationEvidence::new(
            OfficialSource::new("https://platform.openai.com/docs").unwrap(),
            VerificationDate::new(NaiveDate::from_ymd_opt(2026, 8, 6).unwrap()),
        )
    }

    fn native_claim(scope: NativeSupportScope) -> VerifiedNativeSupportClaim {
        VerifiedNativeSupportClaim::new(
            scope,
            VerifiedFidelity::Native,
            ApiStability::Stable,
            native_evidence(),
        )
    }

    #[test]
    fn generic_profile_cannot_carry_verified_claims_or_a_named_catalog() {
        let profile = ProviderProfile::generic(
            ProfileId::new("custom").unwrap(),
            GenericSupportClaim::new(scope(), ApiStability::Experimental),
        );
        assert!(profile.generic_claim().is_some());
        assert!(profile.catalog().is_none());
    }

    #[test]
    fn verified_profile_requires_claims_and_matching_catalog_scopes() {
        let catalog = ModelCatalog::new([model("deepseek-chat", ModelLifecycle::Active)]).unwrap();
        assert_eq!(
            ProviderProfile::verified(ProfileId::new("deepseek").unwrap(), vec![], catalog),
            Err(ProfileError::EmptyVerifiedClaims)
        );
    }

    #[test]
    fn catalog_rejects_missing_replacements_and_cycles() {
        let missing = ModelCatalog::new([model(
            "old",
            ModelLifecycle::Deprecated {
                replacement: Some(ModelId::new("new").unwrap()),
            },
        )]);
        assert!(matches!(
            missing,
            Err(CatalogError::MissingReplacement { .. })
        ));

        let cycle = ModelCatalog::new([
            model(
                "one",
                ModelLifecycle::Deprecated {
                    replacement: Some(ModelId::new("two").unwrap()),
                },
            ),
            model(
                "two",
                ModelLifecycle::Deprecated {
                    replacement: Some(ModelId::new("one").unwrap()),
                },
            ),
        ]);
        assert!(matches!(cycle, Err(CatalogError::ReplacementCycle { .. })));
    }

    #[test]
    fn official_sources_reject_credentials_and_insecure_urls() {
        assert!(OfficialSource::new("http://example.com/docs").is_err());
        assert!(OfficialSource::new("https://secret@example.com/docs").is_err());
        assert!(serde_json::from_str::<OfficialSource>("\"http://example.com/docs\"").is_err());
    }

    #[test]
    fn manifest_combines_portable_profiles_and_native_surfaces() {
        let files_scope = NativeSupportScope::protocol(
            ProviderId::new("openai").unwrap(),
            PlatformId::new("public-api").unwrap(),
            NativeSurfaceKind::Resource,
            ProtocolId::new("openai").unwrap(),
            ApiModeId::new("files").unwrap(),
        );
        let batch_scope = NativeSupportScope::surface(
            ProviderId::new("openai").unwrap(),
            PlatformId::new("public-api").unwrap(),
            NativeSurfaceKind::Job,
            NativeSurfaceId::new("batches").unwrap(),
        );

        let manifest = ProviderSupportManifest::new(
            ProviderId::new("openai").unwrap(),
            [
                verified_profile("openai", ModelFamily::Language, "responses"),
                verified_profile("openai", ModelFamily::Image, "images"),
            ],
            [
                native_claim(files_scope.clone()),
                native_claim(batch_scope.clone()),
            ],
        )
        .unwrap();

        assert_eq!(manifest.provider_id().as_str(), "openai");
        assert_eq!(manifest.profiles().len(), 2);
        assert_eq!(manifest.native_claims().len(), 2);
        assert_eq!(files_scope.binding().api_mode().unwrap().as_str(), "files");
        assert_eq!(
            batch_scope.binding().surface_id().unwrap().as_str(),
            "batches"
        );
        assert_eq!(
            manifest.native_claims()[0].evidence().source().as_str(),
            "https://platform.openai.com/docs"
        );
    }

    #[test]
    fn manifest_can_describe_a_native_only_provider_surface() {
        let claim = native_claim(NativeSupportScope::surface(
            ProviderId::new("native-only").unwrap(),
            PlatformId::new("public-api").unwrap(),
            NativeSurfaceKind::Resource,
            NativeSurfaceId::new("files").unwrap(),
        ));

        let manifest =
            ProviderSupportManifest::new(ProviderId::new("native-only").unwrap(), [], [claim])
                .unwrap();

        assert_eq!(manifest.provider_id().as_str(), "native-only");
        assert!(manifest.profiles().is_empty());
        assert_eq!(manifest.native_claims().len(), 1);
    }

    #[test]
    fn manifest_can_be_empty_for_an_unverified_custom_provider() {
        let manifest =
            ProviderSupportManifest::new(ProviderId::new("custom-provider").unwrap(), [], [])
                .unwrap();

        assert_eq!(manifest.provider_id().as_str(), "custom-provider");
        assert!(manifest.profiles().is_empty());
        assert!(manifest.native_claims().is_empty());
    }

    #[test]
    fn manifest_rejects_mixed_providers() {
        let native = native_claim(NativeSupportScope::surface(
            ProviderId::new("anthropic").unwrap(),
            PlatformId::new("public-api").unwrap(),
            NativeSurfaceKind::Session,
            NativeSurfaceId::new("message-batches").unwrap(),
        ));

        assert!(matches!(
            ProviderSupportManifest::new(
                ProviderId::new("openai").unwrap(),
                [verified_profile(
                    "openai",
                    ModelFamily::Language,
                    "responses"
                )],
                [native],
            ),
            Err(SupportManifestError::MixedProviders { expected, found })
                if expected.as_str() == "openai" && found.as_str() == "anthropic"
        ));
    }

    #[test]
    fn manifest_rejects_duplicate_native_scopes() {
        let native = native_claim(NativeSupportScope::surface(
            ProviderId::new("openai").unwrap(),
            PlatformId::new("public-api").unwrap(),
            NativeSurfaceKind::Job,
            NativeSurfaceId::new("batches").unwrap(),
        ));

        assert!(matches!(
            ProviderSupportManifest::new(
                ProviderId::new("openai").unwrap(),
                [verified_profile(
                    "openai",
                    ModelFamily::Language,
                    "responses"
                )],
                [native.clone(), native],
            ),
            Err(SupportManifestError::DuplicateNativeScope { scope })
                if scope.binding().surface_id().is_some_and(|id| id.as_str() == "batches")
        ));
    }

    #[test]
    fn generic_model_profile_can_coexist_with_separately_scoped_native_evidence() {
        let profile = ProviderProfile::generic(
            ProfileId::new("custom-endpoint").unwrap(),
            GenericSupportClaim::new(
                SupportScope::new(
                    ProviderId::new("custom-endpoint").unwrap(),
                    PlatformId::new("custom-endpoint").unwrap(),
                    ModelFamily::Language,
                    ProtocolId::new("openai-compatible").unwrap(),
                    ApiModeId::new("chat-completions").unwrap(),
                ),
                ApiStability::Experimental,
            ),
        );
        let native = native_claim(NativeSupportScope::surface(
            ProviderId::new("custom-endpoint").unwrap(),
            PlatformId::new("custom-endpoint").unwrap(),
            NativeSurfaceKind::Resource,
            NativeSurfaceId::new("files").unwrap(),
        ));

        let manifest = ProviderSupportManifest::new(
            ProviderId::new("custom-endpoint").unwrap(),
            [profile],
            [native],
        )
        .unwrap();

        assert!(manifest.profiles()[0].generic_claims().is_some());
        assert_eq!(manifest.native_claims().len(), 1);
    }
}
