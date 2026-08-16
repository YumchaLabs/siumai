//! Runtime-owned provider-deferred continuation state.
//!
//! Provider-deferred observations are not durable merely because they were
//! seen on an established stream. A model call stages them here and only an
//! authoritative completed terminal may commit the staged batch.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{OpaqueProviderItem, ProviderScope};
use thiserror::Error;

use crate::ModelTarget;
use crate::snapshot::ProviderStateSnapshot;

const MAX_CORRELATION_ID_BYTES: usize = 1_024;
const PROVIDER_STATE_ENCODING: &str = "application/json";

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ProviderDeferredKey {
    scope: ProviderScope,
    correlation_id: String,
}

impl ProviderDeferredKey {
    fn new(scope: ProviderScope, correlation_id: String) -> Self {
        Self {
            scope,
            correlation_id,
        }
    }

    fn from_observation(observation: &ProviderDeferredObservation) -> Self {
        Self::new(
            observation.item.provenance().scope().clone(),
            observation.correlation_id.clone(),
        )
    }

    pub(crate) fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn namespace(&self) -> Result<String, ProviderDeferredError> {
        let protocol = self
            .scope
            .protocol()
            .ok_or(ProviderDeferredError::MissingProtocol)?;
        Ok(format!(
            "provider-deferred:{}:{protocol}",
            self.scope.provider_id()
        ))
    }
}

/// One provider-deferred observation retained in a durable run report.
///
/// Serialized snapshots are sensitive replay state and retain the exact
/// correlation identity and provider payload. Default diagnostics expose only
/// structure. A resolved observation remains in the ledger as monotonic proof
/// that the same key cannot be reopened by a later snapshot.
#[derive(Clone, PartialEq, Serialize)]
pub struct ProviderDeferredObservation {
    correlation_id: String,
    item: OpaqueProviderItem,
    resolved: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderDeferredObservationWire {
    correlation_id: String,
    item: OpaqueProviderItem,
    resolved: bool,
}

impl<'de> Deserialize<'de> for ProviderDeferredObservation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderDeferredObservationWire::deserialize(deserializer)?;
        validate_correlation_id(&wire.correlation_id).map_err(serde::de::Error::custom)?;
        Ok(Self {
            correlation_id: wire.correlation_id,
            item: wire.item,
            resolved: wire.resolved,
        })
    }
}

impl ProviderDeferredObservation {
    fn new(correlation_id: String, item: OpaqueProviderItem, resolved: bool) -> Self {
        Self {
            correlation_id,
            item,
            resolved,
        }
    }

    pub fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    pub fn item(&self) -> &OpaqueProviderItem {
        &self.item
    }

    pub const fn is_resolved(&self) -> bool {
        self.resolved
    }

    fn key(&self) -> ProviderDeferredKey {
        ProviderDeferredKey::from_observation(self)
    }
}

impl fmt::Debug for ProviderDeferredObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderDeferredObservation")
            .field("correlation_id", &"<redacted>")
            .field("item", &self.item)
            .field("resolved", &self.resolved)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ProviderDeferredError {
    #[error("provider-deferred correlation identifier is invalid")]
    InvalidCorrelationId,
    #[error("provider-deferred observation scope does not match the active model target")]
    ScopeMismatch,
    #[error("provider-deferred replay scope requires a protocol identity")]
    MissingProtocol,
    #[error("a resolved provider-deferred key cannot be observed again")]
    ResolvedKeyReopened,
    #[error("provider-deferred state could not be encoded for a durable checkpoint")]
    Encoding,
    #[error("provider-deferred durable keys must be unique")]
    DuplicateKey,
}

/// Monotonic provider-owned continuation ledger.
#[derive(Clone, Default, PartialEq, Serialize)]
pub(crate) struct ProviderDeferredLedger {
    observations: Vec<ProviderDeferredObservation>,
}

impl fmt::Debug for ProviderDeferredLedger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderDeferredLedger")
            .field("observations", &self.observations.len())
            .field(
                "pending",
                &self
                    .observations
                    .iter()
                    .filter(|observation| !observation.resolved)
                    .count(),
            )
            .finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderDeferredLedgerWire {
    observations: Vec<ProviderDeferredObservation>,
}

impl<'de> Deserialize<'de> for ProviderDeferredLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderDeferredLedgerWire::deserialize(deserializer)?;
        let ledger = Self {
            observations: wire.observations,
        };
        ledger.validate().map_err(serde::de::Error::custom)?;
        Ok(ledger)
    }
}

impl ProviderDeferredLedger {
    pub(crate) fn observations(&self) -> &[ProviderDeferredObservation] {
        &self.observations
    }

    pub(crate) fn begin_step(&self, step: u32, target: &ModelTarget) -> ProviderDeferredStep {
        ProviderDeferredStep {
            step,
            target_scope: target.scope().clone(),
            operations: Vec::new(),
            error: None,
        }
    }

    pub(crate) fn has_unresolved(&self) -> bool {
        self.observations
            .iter()
            .any(|observation| !observation.resolved)
    }

    pub(crate) fn pending_projection(
        &self,
        scope: &ProviderScope,
    ) -> Result<Vec<ProviderStateSnapshot>, ProviderDeferredError> {
        self.observations
            .iter()
            .filter(|observation| {
                !observation.resolved && observation.item.provenance().scope() == scope
            })
            .map(provider_state_from_observation)
            .collect()
    }

    pub(crate) fn validate_pending_projection(
        &self,
        scope: &ProviderScope,
        actual: &[ProviderStateSnapshot],
    ) -> Result<(), ProviderDeferredError> {
        if self.observations.iter().any(|observation| {
            !observation.resolved && observation.item.provenance().scope() != scope
        }) {
            return Err(ProviderDeferredError::ScopeMismatch);
        }
        let expected = self.pending_projection(scope)?;
        if expected == actual {
            Ok(())
        } else {
            Err(ProviderDeferredError::Encoding)
        }
    }

    pub(crate) fn validate_successor(&self, successor: &Self) -> bool {
        successor.observations.len() >= self.observations.len()
            && self
                .observations
                .iter()
                .zip(&successor.observations)
                .all(|(previous, next)| {
                    if previous.key() != next.key() {
                        return false;
                    }
                    if previous.resolved {
                        previous == next
                    } else {
                        true
                    }
                })
    }

    fn validate(&self) -> Result<(), ProviderDeferredError> {
        let mut keys = BTreeSet::new();
        for observation in &self.observations {
            validate_correlation_id(&observation.correlation_id)?;
            if !keys.insert(observation.key()) {
                return Err(ProviderDeferredError::DuplicateKey);
            }
        }
        Ok(())
    }

    fn apply_observation(
        &mut self,
        correlation_id: String,
        item: OpaqueProviderItem,
    ) -> Result<(), ProviderDeferredError> {
        let key =
            ProviderDeferredKey::new(item.provenance().scope().clone(), correlation_id.clone());
        if let Some(existing) = self
            .observations
            .iter_mut()
            .find(|observation| observation.key() == key)
        {
            if existing.resolved {
                return Err(ProviderDeferredError::ResolvedKeyReopened);
            }
            existing.item = item;
        } else {
            self.observations.push(ProviderDeferredObservation::new(
                correlation_id,
                item,
                false,
            ));
        }
        Ok(())
    }

    fn resolve(&mut self, key: &ProviderDeferredKey) {
        if let Some(observation) = self
            .observations
            .iter_mut()
            .find(|observation| observation.key() == *key)
        {
            observation.resolved = true;
        }
    }
}

enum ProviderDeferredOperation {
    Observe {
        correlation_id: String,
        item: OpaqueProviderItem,
    },
    Resolve(ProviderDeferredKey),
}

/// Call-local provider-deferred staging area.
pub(crate) struct ProviderDeferredStep {
    step: u32,
    target_scope: ProviderScope,
    operations: Vec<ProviderDeferredOperation>,
    error: Option<ProviderDeferredError>,
}

impl ProviderDeferredStep {
    pub(crate) fn observe(&mut self, correlation_id: &str, item: &OpaqueProviderItem) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = validate_correlation_id(correlation_id) {
            self.error = Some(error);
            return;
        }
        if item.provenance().scope() != &self.target_scope {
            self.error = Some(ProviderDeferredError::ScopeMismatch);
            return;
        }

        self.operations.push(ProviderDeferredOperation::Observe {
            correlation_id: correlation_id.to_string(),
            item: item.clone(),
        });
    }

    pub(crate) fn resolve_provider_result(&mut self, correlation_id: &str) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = validate_correlation_id(correlation_id) {
            self.error = Some(error);
            return;
        }
        self.operations.push(ProviderDeferredOperation::Resolve(
            ProviderDeferredKey::new(self.target_scope.clone(), correlation_id.to_string()),
        ));
    }

    pub(crate) fn finish_completed(
        self,
        base: &ProviderDeferredLedger,
    ) -> Result<ProviderDeferredCommit, ProviderDeferredError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let mut ledger = base.clone();
        for operation in self.operations {
            match operation {
                ProviderDeferredOperation::Observe {
                    correlation_id,
                    item,
                } => ledger.apply_observation(correlation_id, item)?,
                ProviderDeferredOperation::Resolve(key) => ledger.resolve(&key),
            }
        }
        ledger.validate()?;
        let pending = ledger.pending_projection(&self.target_scope)?;
        Ok(ProviderDeferredCommit {
            ledger,
            step: self.step,
            target_scope: self.target_scope,
            pending,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ProviderDeferredCommit {
    ledger: ProviderDeferredLedger,
    step: u32,
    target_scope: ProviderScope,
    pending: Vec<ProviderStateSnapshot>,
}

impl ProviderDeferredCommit {
    pub(crate) fn into_parts(
        self,
    ) -> (
        ProviderDeferredLedger,
        u32,
        ProviderScope,
        Vec<ProviderStateSnapshot>,
    ) {
        (self.ledger, self.step, self.target_scope, self.pending)
    }

    #[cfg(test)]
    pub(crate) fn into_ledger(self) -> ProviderDeferredLedger {
        self.ledger
    }

    #[cfg(test)]
    pub(crate) fn pending(&self) -> &[ProviderStateSnapshot] {
        &self.pending
    }
}

fn provider_state_from_observation(
    observation: &ProviderDeferredObservation,
) -> Result<ProviderStateSnapshot, ProviderDeferredError> {
    let key = observation.key();
    let payload =
        serde_json::to_vec(&observation.item).map_err(|_| ProviderDeferredError::Encoding)?;
    Ok(ProviderStateSnapshot::new(
        key.namespace()?,
        observation.item.provenance().scope().clone(),
        key.correlation_id().to_string(),
        PROVIDER_STATE_ENCODING.to_string(),
        payload,
    ))
}

fn validate_correlation_id(value: &str) -> Result<(), ProviderDeferredError> {
    if value.is_empty()
        || value.len() > MAX_CORRELATION_ID_BYTES
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(ProviderDeferredError::InvalidCorrelationId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        ApiModeId, ModelDescriptor, ModelFamily, ModelId, PlatformId, ProtocolId, ProviderId,
        ProviderProvenance, ReplayDomain, ReplayDomainId,
    };

    use super::*;

    fn target(scope_suffix: &str) -> ModelTarget {
        let descriptor = ModelDescriptor::new(
            ProviderId::new("provider").unwrap(),
            ModelId::new("model").unwrap(),
            ModelFamily::Language,
        )
        .with_platform(PlatformId::new(format!("platform-{scope_suffix}")).unwrap())
        .with_protocol(ProtocolId::new("protocol").unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new(format!("domain-{scope_suffix}")).unwrap(),
        ));
        ModelTarget::new(descriptor.provider().clone(), descriptor.model().clone())
            .with_platform(descriptor.scope().platform().unwrap().clone())
            .with_protocol(descriptor.scope().protocol().unwrap().clone())
            .with_api_mode(descriptor.scope().api_mode().unwrap().clone())
            .with_replay_domain(descriptor.scope().replay_domain().unwrap().clone())
    }

    fn item(target: &ModelTarget, status: &str) -> OpaqueProviderItem {
        OpaqueProviderItem::new(
            ProviderProvenance::from_scope(target.scope(), target.model().clone()).unwrap(),
            "provider.deferred",
            json!({ "status": status }),
        )
        .unwrap()
    }

    #[test]
    fn ordered_upsert_and_resolution_are_monotonic() {
        let target = target("a");
        let ledger = ProviderDeferredLedger::default();
        let mut step = ledger.begin_step(0, &target);
        step.observe("a", &item(&target, "queued"));
        step.observe("b", &item(&target, "queued"));
        step.observe("a", &item(&target, "in_progress"));
        step.resolve_provider_result("a");
        let commit = step.finish_completed(&ledger).unwrap();
        let observations = commit.ledger.observations();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].correlation_id(), "a");
        assert_eq!(observations[0].item().data()["status"], "in_progress");
        assert!(observations[0].is_resolved());
        assert_eq!(observations[1].correlation_id(), "b");
        assert!(!observations[1].is_resolved());
        assert_eq!(commit.pending().len(), 1);
    }

    #[test]
    fn resolving_an_unknown_key_does_not_resolve_a_later_observation() {
        let target = target("a");
        let ledger = ProviderDeferredLedger::default();
        let mut step = ledger.begin_step(0, &target);
        step.resolve_provider_result("state");
        step.observe("state", &item(&target, "queued"));

        // A resolution only has authority over state present when it arrives;
        // it cannot resolve a later observation for the same key.
        let commit = step.finish_completed(&ledger).unwrap();
        let observation = &commit.ledger.observations()[0];
        assert_eq!(observation.correlation_id(), "state");
        assert!(!observation.is_resolved());
        assert_eq!(commit.pending().len(), 1);
    }

    #[test]
    fn same_step_observation_after_resolution_fails_closed() {
        let target = target("a");
        let ledger = ProviderDeferredLedger::default();
        let mut step = ledger.begin_step(0, &target);
        step.observe("state", &item(&target, "queued"));
        step.resolve_provider_result("state");
        step.observe("state", &item(&target, "reopened"));

        assert_eq!(
            step.finish_completed(&ledger).unwrap_err(),
            ProviderDeferredError::ResolvedKeyReopened
        );
    }

    #[test]
    fn exact_scope_keeps_equal_correlations_distinct() {
        let first = target("a");
        let second = target("b");
        let mut ledger = ProviderDeferredLedger::default();
        ledger
            .apply_observation("shared".to_string(), item(&first, "queued"))
            .unwrap();
        ledger
            .apply_observation("shared".to_string(), item(&second, "queued"))
            .unwrap();
        assert_eq!(ledger.observations().len(), 2);
    }

    #[test]
    fn resolved_keys_cannot_reopen() {
        let target = target("a");
        let mut first = ProviderDeferredLedger::default().begin_step(0, &target);
        first.observe("state", &item(&target, "queued"));
        first.resolve_provider_result("state");
        let base = ProviderDeferredLedger::default();
        let committed = first.finish_completed(&base).unwrap().into_ledger();

        let mut next = committed.begin_step(1, &target);
        next.observe("state", &item(&target, "reopened"));
        assert_eq!(
            next.finish_completed(&committed).unwrap_err(),
            ProviderDeferredError::ResolvedKeyReopened
        );
    }
}
