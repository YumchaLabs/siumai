//! Structured provider support claims and advisory model catalogs.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

use crate::model::ModelFamily;
use crate::provider::{
    ApiModeId, ModelId, ModelOperation, PlatformId, ProfileId, ProtocolContractId, ProtocolId,
    ProviderId,
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

/// Optional region or deployment restriction on a support claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AvailabilityScope {
    Global,
    Region(PlatformId),
    Deployment(ProfileId),
}

/// Exact identity covered by one support claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SupportScope {
    provider: ProviderId,
    platform: PlatformId,
    family: ModelFamily,
    protocol: ProtocolId,
    api_mode: ApiModeId,
    availability: Option<AvailabilityScope>,
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
            availability: None,
        }
    }

    pub fn with_availability(mut self, availability: AvailabilityScope) -> Self {
        self.availability = Some(availability);
        self
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

    pub fn availability(&self) -> Option<&AvailabilityScope> {
        self.availability.as_ref()
    }
}

/// An official HTTPS documentation or specification source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    kind: ProviderProfileKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProviderProfileKind {
    Verified {
        claims: Box<[VerifiedSupportClaim]>,
        catalog: ModelCatalog,
    },
    Generic {
        claim: GenericSupportClaim,
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
            kind: ProviderProfileKind::Verified {
                claims: claims.into_boxed_slice(),
                catalog,
            },
        })
    }

    pub fn generic(id: ProfileId, claim: GenericSupportClaim) -> Self {
        Self {
            id,
            kind: ProviderProfileKind::Generic { claim },
        }
    }

    pub fn id(&self) -> &ProfileId {
        &self.id
    }

    pub fn provider_id(&self) -> &ProviderId {
        match &self.kind {
            ProviderProfileKind::Verified { claims, .. } => claims[0].scope.provider(),
            ProviderProfileKind::Generic { claim } => claim.scope.provider(),
        }
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
            ProviderProfileKind::Generic { claim } => Some(claim),
        }
    }

    pub fn catalog(&self) -> Option<&ModelCatalog> {
        match &self.kind {
            ProviderProfileKind::Verified { catalog, .. } => Some(catalog),
            ProviderProfileKind::Generic { .. } => None,
        }
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
    #[error("verified provider profile cannot mix canonical providers")]
    MixedProviders,
    #[error("model catalog contains a scope absent from the verified claims")]
    CatalogScopeNotClaimed,
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
    }
}
