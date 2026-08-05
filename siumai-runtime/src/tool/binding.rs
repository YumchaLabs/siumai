use std::collections::{BTreeMap, btree_map};
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use siumai_core::{ExecutionOwner, ToolBindingIdentity, ToolCall, ToolSpec};
use thiserror::Error;

use super::execution::{
    ApprovalPolicy, RecoveryPolicy, ToolArgumentError, ToolArgumentValidator, ToolConcurrency,
    ToolEffect, ToolExecutionError, ToolExecutionFuture, ToolExecutionRequest, ToolExecutor,
    ToolIdempotencyKey, ToolIdempotencyKeyError, ToolIdempotencyKeyProvider,
};

const BINDING_FINGERPRINT_VERSION: &[u8] = b"siumai.tool-binding.v2";
const CATALOG_FINGERPRINT_VERSION: &[u8] = b"siumai.tool-catalog.v2";
const ARGUMENT_FINGERPRINT_VERSION: &[u8] = b"siumai.tool-arguments.v1";

struct ToolBindingInner {
    spec: ToolSpec,
    revision: Arc<str>,
    identity: ToolBindingIdentity,
    validator: Arc<dyn ToolArgumentValidator>,
    executor: Arc<dyn ToolExecutor>,
    effect: ToolEffect,
    concurrency: ToolConcurrency,
    approval_policy: ApprovalPolicy,
    recovery_policy: RecoveryPolicy,
    idempotency_key_provider: Option<Arc<dyn ToolIdempotencyKeyProvider>>,
}

/// Immutable, clone-cheap association between a model-visible spec and trusted host code.
#[derive(Clone)]
pub struct ToolBinding {
    inner: Arc<ToolBindingInner>,
}

/// Invalid host-controlled binding identity configuration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolBindingConfigError {
    #[error("tool binding revision must be 1..=128 bytes and contain no control characters")]
    InvalidRevision,
}

impl fmt::Debug for ToolBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolBinding")
            .field("identity", &self.inner.identity)
            .field("revision", &self.inner.revision)
            .field("effect", &self.inner.effect)
            .field("concurrency", &self.inner.concurrency)
            .field("approval_policy", &self.inner.approval_policy)
            .field("recovery_policy", &self.inner.recovery_policy)
            .field(
                "has_stable_idempotency_key",
                &self.inner.idempotency_key_provider.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl ToolBinding {
    /// Bind explicit validator and executor implementations with safe defaults.
    ///
    /// The host must change `revision` whenever executable code or trusted
    /// semantics change. Function addresses are deliberately not identities.
    pub fn new(
        spec: ToolSpec,
        revision: impl Into<String>,
        validator: Arc<dyn ToolArgumentValidator>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Result<Self, ToolBindingConfigError> {
        let revision = validate_revision(revision.into())?;
        Ok(Self::from_parts(
            spec,
            revision,
            validator,
            executor,
            ToolEffect::default(),
            ToolConcurrency::default(),
            ApprovalPolicy::default(),
            RecoveryPolicy::default(),
            None,
        ))
    }

    /// Create a binding from validation and async execution closures.
    ///
    /// The host-controlled revision follows the same identity contract as
    /// [`ToolBinding::new`].
    ///
    /// The executor receives an owned clone of the frozen request so the
    /// returned future does not need to borrow the closure argument.
    pub fn from_fn<V, F, Fut>(
        spec: ToolSpec,
        revision: impl Into<String>,
        validate: V,
        execute: F,
    ) -> Result<Self, ToolBindingConfigError>
    where
        V: Fn(&Value) -> Result<(), ToolArgumentError> + Send + Sync + 'static,
        F: Fn(ToolExecutionRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<siumai_core::ToolOutcome, ToolExecutionError>> + Send + 'static,
    {
        Self::new(
            spec,
            revision,
            Arc::new(FnArgumentValidator(validate)),
            Arc::new(FnToolExecutor(execute)),
        )
    }

    pub fn with_effect(self, effect: ToolEffect) -> Self {
        self.rebuild(
            effect,
            self.inner.concurrency,
            self.inner.approval_policy,
            self.inner.recovery_policy,
            self.inner.idempotency_key_provider.clone(),
        )
    }

    pub fn with_concurrency(self, concurrency: ToolConcurrency) -> Self {
        self.rebuild(
            self.inner.effect,
            concurrency,
            self.inner.approval_policy,
            self.inner.recovery_policy,
            self.inner.idempotency_key_provider.clone(),
        )
    }

    pub fn with_approval_policy(self, approval_policy: ApprovalPolicy) -> Self {
        self.rebuild(
            self.inner.effect,
            self.inner.concurrency,
            approval_policy,
            self.inner.recovery_policy,
            self.inner.idempotency_key_provider.clone(),
        )
    }

    pub fn with_recovery_policy(self, recovery_policy: RecoveryPolicy) -> Self {
        self.rebuild(
            self.inner.effect,
            self.inner.concurrency,
            self.inner.approval_policy,
            recovery_policy,
            self.inner.idempotency_key_provider.clone(),
        )
    }

    /// Install a binding-owned derivation seam for one stable logical-call key.
    ///
    /// The key is derived exactly once when a request is frozen and is reused
    /// for recovery attempts. Change the binding revision whenever derivation
    /// semantics change.
    pub fn with_stable_idempotency_key_provider(
        self,
        provider: Arc<dyn ToolIdempotencyKeyProvider>,
    ) -> Self {
        self.rebuild(
            self.inner.effect,
            self.inner.concurrency,
            self.inner.approval_policy,
            self.inner.recovery_policy,
            Some(provider),
        )
    }

    /// Closure-based form of [`ToolBinding::with_stable_idempotency_key_provider`].
    pub fn with_stable_idempotency_key<F>(self, derive: F) -> Self
    where
        F: Fn(&ToolCall) -> Result<ToolIdempotencyKey, ToolIdempotencyKeyError>
            + Send
            + Sync
            + 'static,
    {
        self.with_stable_idempotency_key_provider(Arc::new(FnIdempotencyKeyProvider(derive)))
    }

    pub fn spec(&self) -> &ToolSpec {
        &self.inner.spec
    }

    pub fn name(&self) -> &str {
        self.inner.spec.name()
    }

    pub fn identity(&self) -> &ToolBindingIdentity {
        &self.inner.identity
    }

    /// Host-controlled implementation revision included in the binding identity.
    pub fn revision(&self) -> &str {
        &self.inner.revision
    }

    pub fn effect(&self) -> ToolEffect {
        self.inner.effect
    }

    pub fn concurrency(&self) -> ToolConcurrency {
        self.inner.concurrency
    }

    pub fn approval_policy(&self) -> ApprovalPolicy {
        self.inner.approval_policy
    }

    pub fn recovery_policy(&self) -> RecoveryPolicy {
        self.inner.recovery_policy
    }

    pub fn has_stable_idempotency_key(&self) -> bool {
        self.inner.idempotency_key_provider.is_some()
    }

    /// Return whether two handles point to the exact same frozen binding.
    pub fn same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn validate(&self, arguments: &Value) -> Result<(), ToolArgumentError> {
        self.inner.validator.validate(arguments)
    }

    pub(crate) fn execute<'a>(
        &'a self,
        request: &'a ToolExecutionRequest,
    ) -> ToolExecutionFuture<'a> {
        self.inner.executor.execute(request)
    }

    pub(crate) fn stable_idempotency_key(
        &self,
        call: &ToolCall,
    ) -> Result<Option<ToolIdempotencyKey>, ToolIdempotencyKeyError> {
        self.inner
            .idempotency_key_provider
            .as_ref()
            .map(|provider| provider.stable_key(call))
            .transpose()
    }

    fn rebuild(
        &self,
        effect: ToolEffect,
        concurrency: ToolConcurrency,
        approval_policy: ApprovalPolicy,
        recovery_policy: RecoveryPolicy,
        idempotency_key_provider: Option<Arc<dyn ToolIdempotencyKeyProvider>>,
    ) -> Self {
        Self::from_parts(
            self.inner.spec.clone(),
            Arc::clone(&self.inner.revision),
            Arc::clone(&self.inner.validator),
            Arc::clone(&self.inner.executor),
            effect,
            concurrency,
            approval_policy,
            recovery_policy,
            idempotency_key_provider,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        spec: ToolSpec,
        revision: Arc<str>,
        validator: Arc<dyn ToolArgumentValidator>,
        executor: Arc<dyn ToolExecutor>,
        effect: ToolEffect,
        concurrency: ToolConcurrency,
        approval_policy: ApprovalPolicy,
        recovery_policy: RecoveryPolicy,
        idempotency_key_provider: Option<Arc<dyn ToolIdempotencyKeyProvider>>,
    ) -> Self {
        let identity = binding_identity(
            &spec,
            &revision,
            effect,
            concurrency,
            approval_policy,
            recovery_policy,
            idempotency_key_provider.is_some(),
        );
        Self {
            inner: Arc::new(ToolBindingInner {
                spec,
                revision,
                identity,
                validator,
                executor,
                effect,
                concurrency,
                approval_policy,
                recovery_policy,
                idempotency_key_provider,
            }),
        }
    }
}

struct FnArgumentValidator<V>(V);

impl<V> ToolArgumentValidator for FnArgumentValidator<V>
where
    V: Fn(&Value) -> Result<(), ToolArgumentError> + Send + Sync + 'static,
{
    fn validate(&self, arguments: &Value) -> Result<(), ToolArgumentError> {
        (self.0)(arguments)
    }
}

struct FnIdempotencyKeyProvider<F>(F);

impl<F> ToolIdempotencyKeyProvider for FnIdempotencyKeyProvider<F>
where
    F: Fn(&ToolCall) -> Result<ToolIdempotencyKey, ToolIdempotencyKeyError> + Send + Sync + 'static,
{
    fn stable_key(&self, call: &ToolCall) -> Result<ToolIdempotencyKey, ToolIdempotencyKeyError> {
        (self.0)(call)
    }
}

struct FnToolExecutor<F>(F);

impl<F, Fut> ToolExecutor for FnToolExecutor<F>
where
    F: Fn(ToolExecutionRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<siumai_core::ToolOutcome, ToolExecutionError>> + Send + 'static,
{
    fn execute<'a>(&'a self, request: &'a ToolExecutionRequest) -> ToolExecutionFuture<'a> {
        Box::pin((self.0)(request.clone()))
    }
}

/// Stable, versioned digest of a complete sorted tool catalog.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CatalogFingerprint(Arc<str>);

impl CatalogFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CatalogFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug)]
struct ToolSetInner {
    bindings: BTreeMap<String, ToolBinding>,
    specs: Arc<[ToolSpec]>,
    fingerprint: CatalogFingerprint,
}

/// Immutable, clone-cheap catalog of trusted local tool bindings.
#[derive(Debug, Clone)]
pub struct ToolSet {
    inner: Arc<ToolSetInner>,
}

impl Default for ToolSet {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ToolSet {
    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    pub fn from_bindings(
        bindings: impl IntoIterator<Item = ToolBinding>,
    ) -> Result<Self, ToolSetBuildError> {
        let mut builder = Self::builder();
        for binding in bindings {
            builder.insert(binding)?;
        }
        Ok(builder.build())
    }

    pub fn len(&self) -> usize {
        self.inner.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.bindings.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ToolBinding> {
        self.inner.bindings.get(name)
    }

    pub fn bindings(&self) -> impl ExactSizeIterator<Item = &ToolBinding> {
        self.inner.bindings.values()
    }

    /// Model-visible specs sorted by tool name.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.inner.specs
    }

    pub fn fingerprint(&self) -> &CatalogFingerprint {
        &self.inner.fingerprint
    }

    /// Resolve only a local call and freeze the exact selected binding.
    ///
    /// Provider-owned calls are rejected before name lookup, preventing a
    /// provider tool with a colliding name from reaching trusted local code.
    pub fn resolve(&self, call: ToolCall) -> Result<ToolExecutionRequest, ToolExecutionError> {
        self.resolve_binding(call, None)
    }

    /// Restore a local call only when the current catalog contains the exact
    /// binding identity captured by a snapshot or approval context.
    pub fn resolve_frozen(
        &self,
        call: ToolCall,
        expected: &ToolBindingIdentity,
    ) -> Result<ToolExecutionRequest, ToolExecutionError> {
        self.resolve_binding(call, Some(expected))
    }

    fn resolve_binding(
        &self,
        call: ToolCall,
        expected: Option<&ToolBindingIdentity>,
    ) -> Result<ToolExecutionRequest, ToolExecutionError> {
        match &call.owner {
            ExecutionOwner::Local => {}
            ExecutionOwner::Provider { provider } => {
                return Err(ToolExecutionError::ProviderOwnedCall {
                    call_id: call.id.clone(),
                    tool: call.name.clone(),
                    provider: provider.clone(),
                });
            }
            _ => {
                return Err(ToolExecutionError::UnsupportedExecutionOwner {
                    call_id: call.id.clone(),
                    tool: call.name.clone(),
                });
            }
        }

        let binding = self
            .inner
            .bindings
            .get(&call.name)
            .cloned()
            .ok_or_else(|| ToolExecutionError::UnknownLocalTool {
                call_id: call.id.clone(),
                tool: call.name.clone(),
            })?;

        if let Some(expected) = expected
            && binding.identity() != expected
        {
            return Err(ToolExecutionError::BindingIdentityMismatch {
                call_id: call.id.clone(),
                tool: call.name.clone(),
                expected: Box::new(expected.clone()),
                actual: Box::new(binding.identity().clone()),
            });
        }

        ToolExecutionRequest::local(binding, call)
    }
}

/// Mutable construction phase for an immutable [`ToolSet`].
#[derive(Debug, Default)]
pub struct ToolSetBuilder {
    bindings: BTreeMap<String, ToolBinding>,
}

impl ToolSetBuilder {
    pub fn insert(&mut self, binding: ToolBinding) -> Result<&mut Self, ToolSetBuildError> {
        let name = binding.name().to_string();
        match self.bindings.entry(name) {
            btree_map::Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(self)
            }
            btree_map::Entry::Occupied(entry) => Err(ToolSetBuildError::DuplicateName {
                name: entry.key().clone(),
            }),
        }
    }

    pub fn with_binding(mut self, binding: ToolBinding) -> Result<Self, ToolSetBuildError> {
        self.insert(binding)?;
        Ok(self)
    }

    pub fn build(self) -> ToolSet {
        let specs: Arc<[ToolSpec]> = self
            .bindings
            .values()
            .map(|binding| binding.spec().clone())
            .collect::<Vec<_>>()
            .into();
        let fingerprint = catalog_fingerprint(self.bindings.values());
        ToolSet {
            inner: Arc::new(ToolSetInner {
                bindings: self.bindings,
                specs,
                fingerprint,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ToolSetBuildError {
    #[error("local tool binding `{name}` is already registered")]
    DuplicateName { name: String },
}

fn binding_identity(
    spec: &ToolSpec,
    revision: &str,
    effect: ToolEffect,
    concurrency: ToolConcurrency,
    approval_policy: ApprovalPolicy,
    recovery_policy: RecoveryPolicy,
    has_stable_idempotency_key: bool,
) -> ToolBindingIdentity {
    let mut digest = Sha256::new();
    digest.update(BINDING_FINGERPRINT_VERSION);
    update_string(&mut digest, spec.name());
    update_string(&mut digest, revision);
    match spec.description() {
        Some(description) => {
            digest.update([1]);
            update_string(&mut digest, description);
        }
        None => digest.update([0]),
    }
    CanonicalJson(spec.input_schema()).update_digest(&mut digest);
    update_effect(&mut digest, effect);
    update_concurrency(&mut digest, concurrency);
    update_approval_policy(&mut digest, approval_policy);
    update_recovery_policy(&mut digest, recovery_policy);
    digest.update([u8::from(has_stable_idempotency_key)]);

    let fingerprint = digest.finalize();
    ToolBindingIdentity {
        name: spec.name().to_string(),
        fingerprint: sha256_string(&fingerprint),
    }
}

fn catalog_fingerprint<'a>(bindings: impl Iterator<Item = &'a ToolBinding>) -> CatalogFingerprint {
    let mut digest = Sha256::new();
    digest.update(CATALOG_FINGERPRINT_VERSION);
    for binding in bindings {
        update_string(&mut digest, &binding.identity().name);
        update_string(&mut digest, &binding.identity().fingerprint);
    }
    let fingerprint = digest.finalize();
    CatalogFingerprint(sha256_string(&fingerprint).into())
}

/// Return the runtime-defined digest used to bind untrusted tool arguments to
/// approvals and snapshots.
pub fn canonical_arguments_digest(arguments: &Value) -> String {
    let mut digest = Sha256::new();
    digest.update(ARGUMENT_FINGERPRINT_VERSION);
    CanonicalJson(arguments).update_digest(&mut digest);
    sha256_string(&digest.finalize())
}

fn validate_revision(revision: String) -> Result<Arc<str>, ToolBindingConfigError> {
    if revision.is_empty() || revision.len() > 128 || revision.chars().any(char::is_control) {
        return Err(ToolBindingConfigError::InvalidRevision);
    }
    Ok(revision.into())
}

/// Small deterministic structural encoder used only for fingerprints.
///
/// Object keys are sorted and arrays retain their original order. This is not
/// a general-purpose or RFC canonical JSON implementation.
struct CanonicalJson<'a>(&'a Value);

impl CanonicalJson<'_> {
    fn update_digest(&self, digest: &mut Sha256) {
        match self.0 {
            Value::Null => digest.update([0]),
            Value::Bool(value) => digest.update([1, u8::from(*value)]),
            Value::Number(value) => {
                digest.update([2]);
                update_string(digest, &value.to_string());
            }
            Value::String(value) => {
                digest.update([3]);
                update_string(digest, value);
            }
            Value::Array(values) => {
                digest.update([4]);
                update_len(digest, values.len());
                for value in values {
                    Self(value).update_digest(digest);
                }
            }
            Value::Object(values) => {
                digest.update([5]);
                update_len(digest, values.len());
                let mut entries = values.iter().collect::<Vec<_>>();
                entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
                for (key, value) in entries {
                    update_string(digest, key);
                    Self(value).update_digest(digest);
                }
            }
        }
    }
}

fn update_len(digest: &mut Sha256, len: usize) {
    digest.update((len as u64).to_be_bytes());
}

fn update_string(digest: &mut Sha256, value: &str) {
    update_len(digest, value.len());
    digest.update(value.as_bytes());
}

fn update_effect(digest: &mut Sha256, effect: ToolEffect) {
    digest.update([match effect {
        ToolEffect::ReadOnly => 0,
        ToolEffect::SideEffecting => 1,
    }]);
}

fn update_concurrency(digest: &mut Sha256, concurrency: ToolConcurrency) {
    match concurrency {
        ToolConcurrency::Sequential => digest.update([0]),
        ToolConcurrency::SafeParallel { max_in_flight } => {
            digest.update([1]);
            update_len(digest, max_in_flight.get());
        }
    }
}

fn update_approval_policy(digest: &mut Sha256, policy: ApprovalPolicy) {
    digest.update([match policy {
        ApprovalPolicy::NotRequired => 0,
        ApprovalPolicy::Required => 1,
    }]);
}

fn update_recovery_policy(digest: &mut Sha256, policy: RecoveryPolicy) {
    digest.update([match policy {
        RecoveryPolicy::NeverReplay => 0,
        RecoveryPolicy::RetryWhenKnownNotApplied => 1,
        RecoveryPolicy::ReplayWithStableIdempotencyKey => 2,
    }]);
}

fn sha256_string(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity("sha256:".len() + bytes.len() * 2);
    output.push_str("sha256:");
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
