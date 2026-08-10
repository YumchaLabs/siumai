//! Request-scoped controls and provider-owned option serialization.

use std::fmt;
use std::io::{self, Write};
use std::time::Instant;

use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use crate::model::{Model, ModelFamily};
use crate::provider::{ApiModeId, ProviderId, ProviderInstanceId, ProviderScope, RouteId};

/// Cloneable request cancellation shared by model, transport, and runtime layers.
#[derive(Clone, Default)]
pub struct Cancellation {
    token: CancellationToken,
}

impl fmt::Debug for Cancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Cancellation")
            .field("is_cancelled", &self.is_cancelled())
            .finish()
    }
}

impl Cancellation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.token.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    pub fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
        }
    }

    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

/// Caller intent; transport still requires operation-specific replay proof.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryIntent {
    /// Apply the configured provider policy only when replay safety is proven.
    #[default]
    ProviderPolicy,
    /// Do not retry this logical call.
    Never,
}

/// The source of one provider-option layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOptionOrigin {
    ProviderDefault,
    RouteDefault,
    ModelDefault,
    RuntimeStep,
    Call,
    RawOverride,
}

/// Provider option validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderOptionError {
    #[error("provider options namespace `{0}` is invalid")]
    InvalidNamespace(String),
    #[error("provider options for `{namespace}` must serialize to an object")]
    ExpectedObject { namespace: String },
    #[error("provider option `{path}` is protected and cannot be set per call")]
    ProtectedField { path: String },
    #[error("provider options namespace `{actual}` does not match `{expected}`")]
    NamespaceMismatch { expected: String, actual: String },
    #[error("provider options API mode `{0}` is invalid")]
    InvalidApiMode(String),
    #[error(
        "typed provider options target {actual_family:?}/{actual_api_mode:?} does not match {expected_family:?}/{expected_api_mode:?}"
    )]
    TargetMismatch {
        expected_family: ModelFamily,
        expected_api_mode: Option<String>,
        actual_family: ModelFamily,
        actual_api_mode: Option<String>,
    },
    #[error("raw provider options may only occupy the explicit raw-override layer")]
    RawLayerMismatch,
    #[error("typed provider options are required for this precedence layer")]
    TypedLayerRequired,
    #[error("provider option layer {origin:?} was configured more than once")]
    DuplicateLayer { origin: ProviderOptionOrigin },
    #[error("provider options exceed the {maximum}-entry call limit")]
    TooManyEntries { maximum: usize },
    #[error("provider options exceed the {maximum}-target call limit")]
    TooManyTargets { maximum: usize },
    #[error("provider options exceed the {maximum}-byte aggregate call limit")]
    AggregateTooLarge { maximum: usize },
    #[error("instance-sensitive provider options for `{namespace}` require an exact model binding")]
    InstanceBindingRequired { namespace: String },
    #[error("provider options for `{provider}` do not match the selected exact target")]
    ExactTargetMismatch { provider: String },
    #[error("raw provider options for one exact target may be configured only once")]
    DuplicateRawTarget,
    #[error("raw provider options contain invalid JSON: {0}")]
    InvalidJson(String),
    #[error("provider options exceed the {maximum}-byte limit")]
    TooLarge { maximum: usize },
    #[error("provider options exceed the maximum JSON nesting depth of {maximum}")]
    TooDeep { maximum: usize },
    #[error("provider options exceed the {maximum}-field limit")]
    TooManyFields { maximum: usize },
    #[error("provider rejected option `{path}`: {reason}")]
    Rejected { path: String, reason: String },
    #[error("failed to serialize provider options: {0}")]
    Serialization(String),
}

/// Implemented by typed provider option structs in provider crates.
///
/// Implementations are ergonomic codecs, not a trust boundary. After erasure,
/// every layer is still checked by [`ProviderOptionMerger::validate_layer`].
pub trait TypedProviderOptions: Serialize {
    const NAMESPACE: &'static str;
    const MODEL_FAMILY: ModelFamily;
    /// Exact API mode for mode-specific options; `None` denotes a family-wide contract.
    const API_MODE: Option<&'static str> = None;

    /// Validate provider-specific relationships before serialization.
    fn validate(&self) -> Result<(), ProviderOptionError> {
        Ok(())
    }

    /// Return whether this value contains credentials, replay state, or other
    /// body data that must be bound to one configured provider instance.
    ///
    /// The secure default is sensitive. A provider option author must opt into
    /// reusable unbound values explicitly after reviewing its fields.
    fn binding_requirement(&self) -> ProviderOptionBindingRequirement {
        ProviderOptionBindingRequirement::ConfiguredInstance
    }
}

/// Whether a typed option value may be reused across configured provider instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProviderOptionBindingRequirement {
    /// The value contains no credentials or replay-sensitive provider body state.
    Reusable,
    /// The value must remain bound to one configured provider instance.
    ConfiguredInstance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProviderOptionKind {
    Typed {
        family: ModelFamily,
        api_mode: Option<ApiModeId>,
        binding_requirement: ProviderOptionBindingRequirement,
    },
    Raw,
}

/// One opaque but validated provider-option layer.
#[derive(Clone, PartialEq)]
pub struct ProviderOptions {
    namespace: ProviderId,
    value: Map<String, Value>,
    kind: ProviderOptionKind,
    retained_bytes: usize,
}

impl fmt::Debug for ProviderOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptions")
            .field("namespace", &self.namespace)
            .field("kind", &self.kind)
            .field("field_count", &self.value.len())
            .field("retained_bytes", &self.retained_bytes)
            .finish()
    }
}

impl ProviderOptions {
    /// Serialize a provider-owned typed option struct.
    pub fn typed<T: TypedProviderOptions>(value: &T) -> Result<Self, ProviderOptionError> {
        value.validate()?;
        let binding_requirement = value.binding_requirement();
        let namespace = ProviderId::new(T::NAMESPACE)
            .map_err(|_| ProviderOptionError::InvalidNamespace(T::NAMESPACE.to_string()))?;
        let api_mode = T::API_MODE.map(ApiModeId::new).transpose().map_err(|_| {
            ProviderOptionError::InvalidApiMode(T::API_MODE.unwrap_or_default().to_string())
        })?;
        let value = serde_json::to_value(value)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        Self::from_value(
            namespace,
            value,
            ProviderOptionKind::Typed {
                family: T::MODEL_FAMILY,
                api_mode,
                binding_requirement,
            },
        )
    }

    /// Build the explicit checked raw escape hatch.
    pub fn checked_raw(namespace: ProviderId, value: Value) -> Result<Self, ProviderOptionError> {
        Self::from_value(namespace, value, ProviderOptionKind::Raw)
    }

    /// Build checked raw provider options from bounded JSON bytes.
    ///
    /// The encoded input limit is enforced before JSON materialization.
    pub fn checked_raw_json(
        namespace: ProviderId,
        encoded: &[u8],
    ) -> Result<Self, ProviderOptionError> {
        if encoded.len() > MAX_PROVIDER_OPTION_BYTES {
            return Err(ProviderOptionError::TooLarge {
                maximum: MAX_PROVIDER_OPTION_BYTES,
            });
        }
        let value = serde_json::from_slice(encoded)
            .map_err(|error| ProviderOptionError::InvalidJson(error.to_string()))?;
        Self::from_value(namespace, value, ProviderOptionKind::Raw)
    }

    fn from_value(
        namespace: ProviderId,
        value: Value,
        kind: ProviderOptionKind,
    ) -> Result<Self, ProviderOptionError> {
        let Value::Object(value) = value else {
            return Err(ProviderOptionError::ExpectedObject {
                namespace: namespace.to_string(),
            });
        };
        let retained_bytes = validate_option_shape(&value)?;
        reject_protected_fields(&value, "")?;
        Ok(Self {
            namespace,
            value,
            kind,
            retained_bytes,
        })
    }

    pub fn namespace(&self) -> &ProviderId {
        &self.namespace
    }

    pub fn value(&self) -> &Map<String, Value> {
        &self.value
    }

    pub fn is_raw(&self) -> bool {
        matches!(self.kind, ProviderOptionKind::Raw)
    }

    pub fn model_family(&self) -> Option<ModelFamily> {
        match self.kind {
            ProviderOptionKind::Typed { family, .. } => Some(family),
            ProviderOptionKind::Raw => None,
        }
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        match &self.kind {
            ProviderOptionKind::Typed { api_mode, .. } => api_mode.as_ref(),
            ProviderOptionKind::Raw => None,
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub fn is_instance_sensitive(&self) -> bool {
        matches!(
            self.kind,
            ProviderOptionKind::Typed {
                binding_requirement: ProviderOptionBindingRequirement::ConfiguredInstance,
                ..
            } | ProviderOptionKind::Raw
        )
    }

    fn validate_target(
        &self,
        context: ProviderOptionContext<'_>,
    ) -> Result<(), ProviderOptionError> {
        let ProviderOptionKind::Typed {
            family: actual_family,
            api_mode: actual_api_mode,
            ..
        } = &self.kind
        else {
            return Ok(());
        };
        let actual_family = *actual_family;
        let family_matches = actual_family == context.family;
        let mode_matches = actual_api_mode
            .as_ref()
            .is_none_or(|actual| Some(actual) == context.api_mode);
        if family_matches && mode_matches {
            return Ok(());
        }
        Err(ProviderOptionError::TargetMismatch {
            expected_family: context.family,
            expected_api_mode: context.api_mode.map(ApiModeId::to_string),
            actual_family,
            actual_api_mode: actual_api_mode.as_ref().map(ApiModeId::to_string),
        })
    }
}

/// Whether one exact-target option entry is mandatory for the selected model
/// or intentionally retained as a fallback for another route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
enum ProviderOptionApplicability {
    Required,
    OptionalFallback,
}

#[derive(Clone, PartialEq, Eq)]
enum ProviderOptionBinding {
    Unbound,
    Model {
        route: Option<RouteId>,
        scope: ProviderScope,
        instance_id: ProviderInstanceId,
    },
}

/// Exact provider, family, API-mode, route, and configured-instance target for
/// one provider-option entry.
#[derive(Clone, PartialEq, Eq)]
pub struct ProviderOptionTarget {
    provider: ProviderId,
    family: ModelFamily,
    api_mode: Option<ApiModeId>,
    binding: ProviderOptionBinding,
}

impl fmt::Debug for ProviderOptionTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptionTarget")
            .field("provider", &self.provider)
            .field("family", &self.family)
            .field("api_mode", &self.api_mode)
            .field(
                "binding",
                &match self.binding {
                    ProviderOptionBinding::Unbound => "required-selected-model",
                    ProviderOptionBinding::Model { .. } => "exact-configured-instance",
                },
            )
            .finish()
    }
}

impl ProviderOptionTarget {
    fn typed<T: TypedProviderOptions>() -> Result<Self, ProviderOptionError> {
        let provider = ProviderId::new(T::NAMESPACE)
            .map_err(|_| ProviderOptionError::InvalidNamespace(T::NAMESPACE.to_string()))?;
        let api_mode = T::API_MODE.map(ApiModeId::new).transpose().map_err(|_| {
            ProviderOptionError::InvalidApiMode(T::API_MODE.unwrap_or_default().to_string())
        })?;
        Ok(Self {
            provider,
            family: T::MODEL_FAMILY,
            api_mode,
            binding: ProviderOptionBinding::Unbound,
        })
    }

    /// Bind a target to one concrete direct or Registry-selected model.
    pub fn for_model<M: Model + ?Sized>(model: &M) -> Self {
        let descriptor = model.descriptor();
        let scope = descriptor.scope().clone();
        Self {
            provider: model.provider_id().clone(),
            family: model.family(),
            api_mode: scope.api_mode().cloned(),
            binding: ProviderOptionBinding::Model {
                route: model.route_id().cloned(),
                instance_id: descriptor.instance_id().clone(),
                scope,
            },
        }
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub const fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.api_mode.as_ref()
    }

    pub const fn is_instance_bound(&self) -> bool {
        matches!(self.binding, ProviderOptionBinding::Model { .. })
    }

    fn matches_model<M: Model + ?Sized>(&self, model: &M) -> bool {
        if self.provider != *model.provider_id()
            || self.family != model.family()
            || self.api_mode.as_ref() != model.descriptor().scope().api_mode()
        {
            return false;
        }
        match &self.binding {
            ProviderOptionBinding::Unbound => true,
            ProviderOptionBinding::Model {
                route,
                scope,
                instance_id,
            } => {
                route.as_ref() == model.route_id()
                    && scope == model.descriptor().scope()
                    && instance_id == model.descriptor().instance_id()
            }
        }
    }

    fn mismatch_error<M: Model + ?Sized>(&self, model: &M) -> ProviderOptionError {
        if self.provider != *model.provider_id() {
            return ProviderOptionError::NamespaceMismatch {
                expected: model.provider_id().to_string(),
                actual: self.provider.to_string(),
            };
        }
        if self.family != model.family()
            || self.api_mode.as_ref() != model.descriptor().scope().api_mode()
        {
            return ProviderOptionError::TargetMismatch {
                expected_family: model.family(),
                expected_api_mode: model
                    .descriptor()
                    .scope()
                    .api_mode()
                    .map(ApiModeId::to_string),
                actual_family: self.family,
                actual_api_mode: self.api_mode.as_ref().map(ApiModeId::to_string),
            };
        }
        ProviderOptionError::ExactTargetMismatch {
            provider: self.provider.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct ExactProviderOptionEntry {
    applicability: ProviderOptionApplicability,
    target: ProviderOptionTarget,
    options: ProviderOptions,
}

/// Borrowed exact-target view consumed by one configured model.
pub struct ProviderOptionSelection<'a> {
    typed: Vec<&'a ProviderOptions>,
    raw_override: Option<&'a ProviderOptions>,
    unconsumed: Vec<&'a ProviderOptionTarget>,
}

impl fmt::Debug for ProviderOptionSelection<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptionSelection")
            .field("typed_count", &self.typed.len())
            .field("has_raw_override", &self.raw_override.is_some())
            .field("unconsumed_count", &self.unconsumed.len())
            .finish()
    }
}

impl<'a> ProviderOptionSelection<'a> {
    pub fn typed(&self) -> impl ExactSizeIterator<Item = &'a ProviderOptions> + '_ {
        self.typed.iter().copied()
    }

    pub const fn raw_override(&self) -> Option<&'a ProviderOptions> {
        self.raw_override
    }

    pub fn unconsumed_targets(
        &self,
    ) -> impl ExactSizeIterator<Item = &'a ProviderOptionTarget> + '_ {
        self.unconsumed.iter().copied()
    }

    pub const fn unconsumed_count(&self) -> usize {
        self.unconsumed.len()
    }
}

fn validate_typed_target<T: TypedProviderOptions>(
    target: &ProviderOptionTarget,
) -> Result<(), ProviderOptionError> {
    let declared = ProviderOptionTarget::typed::<T>()?;
    if declared.provider == target.provider
        && declared.family == target.family
        && declared.api_mode == target.api_mode
    {
        return Ok(());
    }
    if declared.provider != target.provider {
        return Err(ProviderOptionError::NamespaceMismatch {
            expected: target.provider.to_string(),
            actual: declared.provider.to_string(),
        });
    }
    Err(ProviderOptionError::TargetMismatch {
        expected_family: target.family,
        expected_api_mode: target.api_mode.as_ref().map(ApiModeId::to_string),
        actual_family: declared.family,
        actual_api_mode: declared.api_mode.as_ref().map(ApiModeId::to_string),
    })
}

/// Exact model call context used to validate erased typed provider options.
#[derive(Debug, Clone, Copy)]
pub struct ProviderOptionContext<'a> {
    provider: &'a ProviderId,
    family: ModelFamily,
    api_mode: Option<&'a ApiModeId>,
}

impl<'a> ProviderOptionContext<'a> {
    pub const fn new(
        provider: &'a ProviderId,
        family: ModelFamily,
        api_mode: Option<&'a ApiModeId>,
    ) -> Self {
        Self {
            provider,
            family,
            api_mode,
        }
    }

    pub const fn provider(self) -> &'a ProviderId {
        self.provider
    }

    pub const fn family(self) -> ModelFamily {
        self.family
    }

    pub const fn api_mode(self) -> Option<&'a ApiModeId> {
        self.api_mode
    }
}

/// Explicit precedence stack. The provider owns the merge algorithm.
#[derive(Debug, Clone, Default)]
pub struct ProviderOptionLayers {
    provider_default: Option<ProviderOptions>,
    route_default: Option<ProviderOptions>,
    model_default: Option<ProviderOptions>,
    runtime_step: Option<ProviderOptions>,
    call: Option<ProviderOptions>,
    raw_override: Option<ProviderOptions>,
}

impl ProviderOptionLayers {
    pub fn with_provider_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(
            &self.provider_default,
            ProviderOptionOrigin::ProviderDefault,
        )?;
        self.provider_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_route_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.route_default, ProviderOptionOrigin::RouteDefault)?;
        self.route_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_model_default(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.model_default, ProviderOptionOrigin::ModelDefault)?;
        self.model_default = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_runtime_step(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.runtime_step, ProviderOptionOrigin::RuntimeStep)?;
        self.runtime_step = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_call(mut self, options: ProviderOptions) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.call, ProviderOptionOrigin::Call)?;
        self.call = Some(require_typed(options)?);
        Ok(self)
    }

    pub fn with_raw_override(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, ProviderOptionError> {
        ensure_empty(&self.raw_override, ProviderOptionOrigin::RawOverride)?;
        if !options.is_raw() {
            return Err(ProviderOptionError::RawLayerMismatch);
        }
        self.raw_override = Some(options);
        Ok(self)
    }

    pub fn in_precedence_order(
        &self,
    ) -> impl Iterator<Item = (ProviderOptionOrigin, &ProviderOptions)> {
        [
            (
                ProviderOptionOrigin::ProviderDefault,
                self.provider_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::RouteDefault,
                self.route_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::ModelDefault,
                self.model_default.as_ref(),
            ),
            (
                ProviderOptionOrigin::RuntimeStep,
                self.runtime_step.as_ref(),
            ),
            (ProviderOptionOrigin::Call, self.call.as_ref()),
            (
                ProviderOptionOrigin::RawOverride,
                self.raw_override.as_ref(),
            ),
        ]
        .into_iter()
        .filter_map(|(origin, options)| options.map(|options| (origin, options)))
    }

    /// Validate namespace ownership and invoke the provider-owned raw schema
    /// and merge policy. Raw fields never reach transport or authentication
    /// configuration through this contract.
    pub fn merge_for<M: ProviderOptionMerger>(
        &self,
        context: ProviderOptionContext<'_>,
        merger: &M,
    ) -> Result<M::Output, ProviderOptionError> {
        for (origin, options) in self.in_precedence_order() {
            if options.namespace() != context.provider {
                return Err(ProviderOptionError::NamespaceMismatch {
                    expected: context.provider.to_string(),
                    actual: options.namespace().to_string(),
                });
            }
            options.validate_target(context)?;
            merger.validate_layer(origin, options)?;
        }
        merger.merge(self)
    }
}

fn require_typed(options: ProviderOptions) -> Result<ProviderOptions, ProviderOptionError> {
    if options.is_raw() {
        Err(ProviderOptionError::TypedLayerRequired)
    } else {
        Ok(options)
    }
}

fn ensure_empty(
    slot: &Option<ProviderOptions>,
    origin: ProviderOptionOrigin,
) -> Result<(), ProviderOptionError> {
    if slot.is_some() {
        Err(ProviderOptionError::DuplicateLayer { origin })
    } else {
        Ok(())
    }
}

/// Provider-owned merge and rejection policy.
pub trait ProviderOptionMerger: Send + Sync {
    type Output;

    /// Validate every erased typed or raw layer against the selected provider's
    /// request-body schema. A public Rust trait implementation is not treated as
    /// proof that typed fields are provider-owned.
    fn validate_layer(
        &self,
        origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError>;

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError>;
}

#[derive(Debug, Clone)]
struct ProviderOptionEntry {
    origin: ProviderOptionOrigin,
    options: ProviderOptions,
}

/// Controls shared by all six stable model families.
#[derive(Clone, Default)]
pub struct CallOptions {
    deadline: Option<Instant>,
    cancellation: Cancellation,
    retry: RetryIntent,
    provider_options: Vec<ProviderOptionEntry>,
    exact_provider_options: Vec<ExactProviderOptionEntry>,
}

impl fmt::Debug for CallOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallOptions")
            .field("deadline", &self.deadline)
            .field("cancellation", &self.cancellation)
            .field("retry", &self.retry)
            .field(
                "legacy_provider_option_namespaces",
                &self
                    .provider_options
                    .iter()
                    .map(|entry| entry.options.namespace().as_str())
                    .collect::<Vec<_>>(),
            )
            .field(
                "exact_provider_option_targets",
                &self
                    .exact_provider_options
                    .iter()
                    .map(|entry| &entry.target)
                    .collect::<Vec<_>>(),
            )
            .field(
                "exact_provider_option_bytes",
                &self
                    .exact_provider_options
                    .iter()
                    .map(|entry| entry.options.retained_bytes())
                    .sum::<usize>(),
            )
            .finish()
    }
}

impl CallOptions {
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn cancellation(&self) -> &Cancellation {
        &self.cancellation
    }

    pub fn retry(&self) -> RetryIntent {
        self.retry
    }

    pub fn provider_options(&self) -> impl Iterator<Item = &ProviderOptions> {
        self.provider_options
            .iter()
            .map(|entry| &entry.options)
            .chain(
                self.exact_provider_options
                    .iter()
                    .map(|entry| &entry.options),
            )
    }

    pub fn has_provider_options(&self) -> bool {
        !self.provider_options.is_empty() || !self.exact_provider_options.is_empty()
    }

    /// Select the exact-target entries that belong to one configured model.
    pub fn provider_options_for<M: Model + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ProviderOptionSelection<'_>, ProviderOptionError> {
        let mut typed = Vec::new();
        let mut raw_override = None;
        let mut unconsumed = Vec::new();

        for entry in &self.exact_provider_options {
            if !entry.target.matches_model(model) {
                if entry.applicability == ProviderOptionApplicability::Required {
                    return Err(entry.target.mismatch_error(model));
                }
                unconsumed.push(&entry.target);
                continue;
            }
            if entry.options.is_raw() {
                if raw_override.replace(&entry.options).is_some() {
                    return Err(ProviderOptionError::DuplicateRawTarget);
                }
            } else {
                typed.push(&entry.options);
            }
        }

        Ok(ProviderOptionSelection {
            typed,
            raw_override,
            unconsumed,
        })
    }

    /// Add reusable typed options for the model selected by this call.
    ///
    /// Instance-sensitive values must use
    /// [`Self::with_typed_provider_options_for`] instead.
    pub fn with_typed_provider_options<T: TypedProviderOptions>(
        mut self,
        value: &T,
    ) -> Result<Self, ProviderOptionError> {
        let options = ProviderOptions::typed(value)?;
        if options.is_instance_sensitive() {
            return Err(ProviderOptionError::InstanceBindingRequired {
                namespace: options.namespace().to_string(),
            });
        }
        let target = ProviderOptionTarget::typed::<T>()?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add typed options bound to the concrete model receiving the call.
    pub fn with_typed_provider_options_for<M, T>(
        mut self,
        model: &M,
        value: &T,
    ) -> Result<Self, ProviderOptionError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let options = ProviderOptions::typed(value)?;
        let target = ProviderOptionTarget::for_model(model);
        validate_typed_target::<T>(&target)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add an optional typed fallback for one exact configured model.
    pub fn with_optional_typed_provider_options_for<M, T>(
        mut self,
        model: &M,
        value: &T,
    ) -> Result<Self, ProviderOptionError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let options = ProviderOptions::typed(value)?;
        let target = ProviderOptionTarget::for_model(model);
        validate_typed_target::<T>(&target)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::OptionalFallback,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add raw provider-body options required for one exact configured model.
    pub fn with_raw_provider_options_for<M: Model + ?Sized>(
        mut self,
        model: &M,
        value: Value,
    ) -> Result<Self, ProviderOptionError> {
        let target = ProviderOptionTarget::for_model(model);
        let options = ProviderOptions::checked_raw(target.provider().clone(), value)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add an optional raw fallback for one exact configured model.
    pub fn with_optional_raw_provider_options_for<M: Model + ?Sized>(
        mut self,
        model: &M,
        value: Value,
    ) -> Result<Self, ProviderOptionError> {
        let target = ProviderOptionTarget::for_model(model);
        let options = ProviderOptions::checked_raw(target.provider().clone(), value)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::OptionalFallback,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add bounded raw JSON required for one exact configured model.
    pub fn with_raw_provider_json_for<M: Model + ?Sized>(
        mut self,
        model: &M,
        encoded: &[u8],
    ) -> Result<Self, ProviderOptionError> {
        let target = ProviderOptionTarget::for_model(model);
        let options = ProviderOptions::checked_raw_json(target.provider().clone(), encoded)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    fn push_exact_provider_option(
        &mut self,
        entry: ExactProviderOptionEntry,
    ) -> Result<(), ProviderOptionError> {
        if self.provider_options.len() + self.exact_provider_options.len()
            >= MAX_PROVIDER_OPTION_ENTRIES
        {
            return Err(ProviderOptionError::TooManyEntries {
                maximum: MAX_PROVIDER_OPTION_ENTRIES,
            });
        }

        let target_count = self
            .exact_provider_options
            .iter()
            .map(|entry| &entry.target)
            .chain(std::iter::once(&entry.target))
            .fold(
                Vec::<&ProviderOptionTarget>::new(),
                |mut targets, target| {
                    if !targets.contains(&target) {
                        targets.push(target);
                    }
                    targets
                },
            )
            .len();
        if target_count > MAX_PROVIDER_OPTION_TARGETS {
            return Err(ProviderOptionError::TooManyTargets {
                maximum: MAX_PROVIDER_OPTION_TARGETS,
            });
        }

        let retained_bytes = self
            .provider_options
            .iter()
            .map(|entry| entry.options.retained_bytes())
            .chain(
                self.exact_provider_options
                    .iter()
                    .map(|entry| entry.options.retained_bytes()),
            )
            .try_fold(entry.options.retained_bytes(), usize::checked_add)
            .ok_or(ProviderOptionError::AggregateTooLarge {
                maximum: MAX_PROVIDER_OPTION_TOTAL_BYTES,
            })?;
        if retained_bytes > MAX_PROVIDER_OPTION_TOTAL_BYTES {
            return Err(ProviderOptionError::AggregateTooLarge {
                maximum: MAX_PROVIDER_OPTION_TOTAL_BYTES,
            });
        }

        if entry.options.is_raw()
            && self
                .exact_provider_options
                .iter()
                .any(|existing| existing.options.is_raw() && existing.target == entry.target)
        {
            return Err(ProviderOptionError::DuplicateRawTarget);
        }

        self.exact_provider_options.push(entry);
        Ok(())
    }

    /// Add call-scoped typed and raw options to an existing precedence stack.
    /// Foreign namespaces and duplicate call/raw layers are rejected.
    pub fn apply_provider_options(
        &self,
        expected: &ProviderId,
        mut layers: ProviderOptionLayers,
    ) -> Result<ProviderOptionLayers, ProviderOptionError> {
        for entry in &self.provider_options {
            let options = &entry.options;
            if options.namespace() != expected {
                return Err(ProviderOptionError::NamespaceMismatch {
                    expected: expected.to_string(),
                    actual: options.namespace().to_string(),
                });
            }
            layers = match entry.origin {
                ProviderOptionOrigin::ProviderDefault => {
                    layers.with_provider_default(options.clone())?
                }
                ProviderOptionOrigin::RouteDefault => layers.with_route_default(options.clone())?,
                ProviderOptionOrigin::ModelDefault => layers.with_model_default(options.clone())?,
                ProviderOptionOrigin::RuntimeStep => layers.with_runtime_step(options.clone())?,
                ProviderOptionOrigin::Call => layers.with_call(options.clone())?,
                ProviderOptionOrigin::RawOverride => layers.with_raw_override(options.clone())?,
            };
        }
        Ok(layers)
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_cancellation(mut self, cancellation: Cancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn without_retry(mut self) -> Self {
        self.retry = RetryIntent::Never;
        self
    }

    pub fn with_provider_options(mut self, options: ProviderOptions) -> Self {
        let origin = if options.is_raw() {
            ProviderOptionOrigin::RawOverride
        } else {
            ProviderOptionOrigin::Call
        };
        self.provider_options
            .push(ProviderOptionEntry { origin, options });
        self
    }

    /// Attach typed defaults selected by a configured Registry route.
    pub fn with_route_default_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::RouteDefault,
            options,
        });
        self
    }

    /// Attach typed defaults selected for one concrete model target.
    pub fn with_model_default_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::ModelDefault,
            options,
        });
        self
    }

    /// Attach typed options selected for one runtime model step.
    pub fn with_runtime_step_provider_options(mut self, options: ProviderOptions) -> Self {
        self.provider_options.push(ProviderOptionEntry {
            origin: ProviderOptionOrigin::RuntimeStep,
            options,
        });
        self
    }
}

const PROTECTED_FIELDS: &[&str] = &[
    "apikey",
    "apitoken",
    "accesstoken",
    "authorization",
    "auth",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "credentials",
    "credential",
    "secretkey",
    "baseurl",
    "baseuri",
    "endpoint",
    "audience",
    "proxy",
    "proxyurl",
    "tls",
    "host",
    "redirect",
    "redirectpolicy",
    "headers",
    "defaultheaders",
];

const MAX_PROVIDER_OPTION_BYTES: usize = 64 * 1024;
const MAX_PROVIDER_OPTION_DEPTH: usize = 32;
const MAX_PROVIDER_OPTION_FIELDS: usize = 1024;
pub const MAX_PROVIDER_OPTION_ENTRIES: usize = 64;
pub const MAX_PROVIDER_OPTION_TARGETS: usize = 32;
pub const MAX_PROVIDER_OPTION_TOTAL_BYTES: usize = 512 * 1024;

pub(crate) fn validate_option_shape(
    object: &Map<String, Value>,
) -> Result<usize, ProviderOptionError> {
    validate_option_structure(object)?;
    let mut writer = LimitedJsonWriter::new(MAX_PROVIDER_OPTION_BYTES);
    if let Err(error) = serde_json::to_writer(&mut writer, object) {
        if writer.exceeded {
            return Err(ProviderOptionError::TooLarge {
                maximum: MAX_PROVIDER_OPTION_BYTES,
            });
        }
        return Err(ProviderOptionError::Serialization(error.to_string()));
    }
    Ok(writer.written)
}

struct LimitedJsonWriter {
    maximum: usize,
    written: usize,
    exceeded: bool,
}

impl LimitedJsonWriter {
    const fn new(maximum: usize) -> Self {
        Self {
            maximum,
            written: 0,
            exceeded: false,
        }
    }
}

impl Write for LimitedJsonWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(total) = self.written.checked_add(bytes.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("provider options exceed their byte limit"));
        };
        if total > self.maximum {
            self.exceeded = true;
            return Err(io::Error::other("provider options exceed their byte limit"));
        }
        self.written = total;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn validate_option_structure(
    object: &Map<String, Value>,
) -> Result<(), ProviderOptionError> {
    let mut fields = 0;
    let mut minimum_bytes = 0;
    validate_option_object(object, 0, &mut fields, &mut minimum_bytes)
}

fn validate_option_object(
    object: &Map<String, Value>,
    depth: usize,
    fields: &mut usize,
    minimum_bytes: &mut usize,
) -> Result<(), ProviderOptionError> {
    ensure_option_depth(depth)?;
    add_option_minimum_bytes(minimum_bytes, 2)?;
    *fields = fields.saturating_add(object.len());
    if *fields > MAX_PROVIDER_OPTION_FIELDS {
        return Err(ProviderOptionError::TooManyFields {
            maximum: MAX_PROVIDER_OPTION_FIELDS,
        });
    }
    for (index, (key, value)) in object.iter().enumerate() {
        if index > 0 {
            add_option_minimum_bytes(minimum_bytes, 1)?;
        }
        add_option_minimum_bytes(minimum_bytes, key.len().saturating_add(3))?;
        validate_option_value(value, depth + 1, fields, minimum_bytes)?;
    }
    Ok(())
}

fn validate_option_value(
    value: &Value,
    depth: usize,
    fields: &mut usize,
    minimum_bytes: &mut usize,
) -> Result<(), ProviderOptionError> {
    ensure_option_depth(depth)?;
    match value {
        Value::Object(object) => {
            validate_option_object(object, depth, fields, minimum_bytes)?;
        }
        Value::Array(items) => {
            add_option_minimum_bytes(minimum_bytes, 2)?;
            for (index, value) in items.iter().enumerate() {
                if index > 0 {
                    add_option_minimum_bytes(minimum_bytes, 1)?;
                }
                validate_option_value(value, depth + 1, fields, minimum_bytes)?;
            }
        }
        Value::Null => add_option_minimum_bytes(minimum_bytes, 4)?,
        Value::Bool(value) => {
            add_option_minimum_bytes(minimum_bytes, if *value { 4 } else { 5 })?;
        }
        Value::Number(value) => add_option_minimum_bytes(minimum_bytes, value.to_string().len())?,
        Value::String(value) => {
            add_option_minimum_bytes(minimum_bytes, value.len().saturating_add(2))?;
        }
    }
    Ok(())
}

fn ensure_option_depth(depth: usize) -> Result<(), ProviderOptionError> {
    if depth > MAX_PROVIDER_OPTION_DEPTH {
        return Err(ProviderOptionError::TooDeep {
            maximum: MAX_PROVIDER_OPTION_DEPTH,
        });
    }
    Ok(())
}

fn add_option_minimum_bytes(
    total: &mut usize,
    additional: usize,
) -> Result<(), ProviderOptionError> {
    *total = total
        .checked_add(additional)
        .ok_or(ProviderOptionError::TooLarge {
            maximum: MAX_PROVIDER_OPTION_BYTES,
        })?;
    if *total > MAX_PROVIDER_OPTION_BYTES {
        return Err(ProviderOptionError::TooLarge {
            maximum: MAX_PROVIDER_OPTION_BYTES,
        });
    }
    Ok(())
}

pub(crate) fn reject_protected_fields(
    object: &Map<String, Value>,
    parent: &str,
) -> Result<(), ProviderOptionError> {
    for (key, value) in object {
        let path = if parent.is_empty() {
            key.clone()
        } else {
            format!("{parent}.{key}")
        };
        let normalized = key
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if PROTECTED_FIELDS.contains(&normalized.as_str()) {
            return Err(ProviderOptionError::ProtectedField { path });
        }
        reject_protected_value(value, &path)?;
    }
    Ok(())
}

fn reject_protected_value(value: &Value, path: &str) -> Result<(), ProviderOptionError> {
    match value {
        Value::Object(object) => reject_protected_fields(object, path),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                reject_protected_value(item, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use serde_json::json;

    use crate::model::{Model, ModelDescriptor};
    use crate::provider::{ModelId, ProviderScope, RouteId};

    use super::*;

    struct FakeModel {
        descriptor: ModelDescriptor,
        route: Option<RouteId>,
    }

    impl Model for FakeModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn route_id(&self) -> Option<&RouteId> {
            self.route.as_ref()
        }
    }

    fn fake_model(provider: &str, mode: &str, route: Option<&str>) -> FakeModel {
        let scope = ProviderScope::new(ProviderId::new(provider).unwrap())
            .with_api_mode(ApiModeId::new(mode).unwrap());
        FakeModel {
            descriptor: ModelDescriptor::from_scope(
                scope,
                ModelId::new("future-model").unwrap(),
                ModelFamily::Language,
                ProviderInstanceId::new(),
            ),
            route: route.map(|value| RouteId::new(value).unwrap()),
        }
    }

    #[derive(Serialize)]
    struct OpenAiOptions {
        reasoning_effort: &'static str,
    }

    impl TypedProviderOptions for OpenAiOptions {
        const NAMESPACE: &'static str = "openai";
        const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
        const API_MODE: Option<&'static str> = Some("responses");

        fn binding_requirement(&self) -> ProviderOptionBindingRequirement {
            ProviderOptionBindingRequirement::Reusable
        }
    }

    #[derive(Serialize)]
    struct SensitiveOpenAiOptions {
        mcp_authorization: &'static str,
    }

    impl TypedProviderOptions for SensitiveOpenAiOptions {
        const NAMESPACE: &'static str = "openai";
        const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
        const API_MODE: Option<&'static str> = Some("responses");
    }

    #[test]
    fn typed_options_keep_namespace_and_hide_values_from_debug() {
        let options = ProviderOptions::typed(&OpenAiOptions {
            reasoning_effort: "high",
        })
        .unwrap();

        assert_eq!(options.namespace().as_str(), "openai");
        assert_eq!(options.value()["reasoning_effort"], "high");
        assert!(!format!("{options:?}").contains("high"));
    }

    #[test]
    fn raw_options_reject_protected_fields_recursively() {
        let error = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"transport": {"api_key": "secret"}}),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ProviderOptionError::ProtectedField { ref path }
                if path == "transport.api_key"
        ));
    }

    #[test]
    fn raw_options_reject_protected_fields_inside_arrays() {
        let error = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"items": [{"headers": {"x-secret": "secret"}}]}),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ProviderOptionError::ProtectedField { ref path }
                if path == "items[0].headers"
        ));
    }

    #[test]
    fn foreign_namespace_is_rejected_instead_of_ignored() {
        struct Merger;
        impl ProviderOptionMerger for Merger {
            type Output = ();

            fn validate_layer(
                &self,
                _origin: ProviderOptionOrigin,
                _options: &ProviderOptions,
            ) -> Result<(), ProviderOptionError> {
                Ok(())
            }

            fn merge(
                &self,
                _layers: &ProviderOptionLayers,
            ) -> Result<Self::Output, ProviderOptionError> {
                Ok(())
            }
        }

        let options = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").unwrap(),
            json!({"thinking": {"type": "adaptive"}}),
        )
        .unwrap();
        let layers = ProviderOptionLayers::default()
            .with_raw_override(options)
            .unwrap();

        let provider = ProviderId::new("openai").unwrap();
        let api_mode = ApiModeId::new("responses").unwrap();
        let error = layers
            .merge_for(
                ProviderOptionContext::new(&provider, ModelFamily::Language, Some(&api_mode)),
                &Merger,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ProviderOptionError::NamespaceMismatch { .. }
        ));
    }

    #[test]
    fn layers_expose_the_fixed_precedence_without_generic_merging() {
        fn layer() -> ProviderOptions {
            ProviderOptions::typed(&OpenAiOptions {
                reasoning_effort: "high",
            })
            .unwrap()
        }

        let layers = ProviderOptionLayers::default()
            .with_provider_default(layer())
            .unwrap()
            .with_route_default(layer())
            .unwrap()
            .with_model_default(layer())
            .unwrap()
            .with_runtime_step(layer())
            .unwrap()
            .with_call(layer())
            .unwrap()
            .with_raw_override(
                ProviderOptions::checked_raw(
                    ProviderId::new("openai").unwrap(),
                    json!({"reasoning_effort": "xhigh"}),
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(
            layers
                .in_precedence_order()
                .map(|(origin, _)| origin)
                .collect::<Vec<_>>(),
            vec![
                ProviderOptionOrigin::ProviderDefault,
                ProviderOptionOrigin::RouteDefault,
                ProviderOptionOrigin::ModelDefault,
                ProviderOptionOrigin::RuntimeStep,
                ProviderOptionOrigin::Call,
                ProviderOptionOrigin::RawOverride,
            ]
        );
    }

    #[test]
    fn raw_options_cannot_occupy_a_typed_precedence_layer() {
        let raw = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({"reasoning_effort": "high"}),
        )
        .unwrap();
        assert!(matches!(
            ProviderOptionLayers::default().with_call(raw),
            Err(ProviderOptionError::TypedLayerRequired)
        ));
    }

    #[test]
    fn call_options_reject_foreign_namespaces_and_duplicate_layers() {
        let openai = ProviderOptions::typed(&OpenAiOptions {
            reasoning_effort: "high",
        })
        .unwrap();
        let anthropic = ProviderOptions::checked_raw(
            ProviderId::new("anthropic").unwrap(),
            json!({"thinking": {"type": "adaptive"}}),
        )
        .unwrap();
        let foreign = CallOptions::default()
            .with_provider_options(openai.clone())
            .with_provider_options(anthropic)
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default(),
            )
            .unwrap_err();
        assert!(matches!(
            foreign,
            ProviderOptionError::NamespaceMismatch { .. }
        ));

        let duplicate = CallOptions::default()
            .with_provider_options(openai.clone())
            .with_provider_options(openai)
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default(),
            )
            .unwrap_err();
        assert!(matches!(
            duplicate,
            ProviderOptionError::DuplicateLayer {
                origin: ProviderOptionOrigin::Call
            }
        ));
    }

    #[test]
    fn runtime_option_origins_join_the_provider_owned_precedence_stack() {
        fn layer(value: &'static str) -> ProviderOptions {
            #[derive(Serialize)]
            struct Layer {
                value: &'static str,
            }

            impl TypedProviderOptions for Layer {
                const NAMESPACE: &'static str = "openai";
                const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
                const API_MODE: Option<&'static str> = Some("responses");
            }

            ProviderOptions::typed(&Layer { value }).unwrap()
        }

        let layers = CallOptions::default()
            .with_route_default_provider_options(layer("route"))
            .with_model_default_provider_options(layer("model"))
            .with_runtime_step_provider_options(layer("step"))
            .with_provider_options(layer("call"))
            .apply_provider_options(
                &ProviderId::new("openai").unwrap(),
                ProviderOptionLayers::default()
                    .with_provider_default(layer("provider"))
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            layers
                .in_precedence_order()
                .map(|(origin, options)| (origin, options.value()["value"].as_str().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                (ProviderOptionOrigin::ProviderDefault, "provider"),
                (ProviderOptionOrigin::RouteDefault, "route"),
                (ProviderOptionOrigin::ModelDefault, "model"),
                (ProviderOptionOrigin::RuntimeStep, "step"),
                (ProviderOptionOrigin::Call, "call"),
            ]
        );
    }

    #[test]
    fn provider_schema_validates_erased_typed_layers_too() {
        #[derive(Serialize)]
        struct ForgedOpenAiOptions {
            unrecognized: bool,
        }

        impl TypedProviderOptions for ForgedOpenAiOptions {
            const NAMESPACE: &'static str = "openai";
            const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
            const API_MODE: Option<&'static str> = Some("responses");
        }

        struct StrictMerger;
        impl ProviderOptionMerger for StrictMerger {
            type Output = ();

            fn validate_layer(
                &self,
                _origin: ProviderOptionOrigin,
                options: &ProviderOptions,
            ) -> Result<(), ProviderOptionError> {
                if options.value().contains_key("unrecognized") {
                    return Err(ProviderOptionError::Rejected {
                        path: "unrecognized".to_string(),
                        reason: "unknown request-body field".to_string(),
                    });
                }
                Ok(())
            }

            fn merge(
                &self,
                _layers: &ProviderOptionLayers,
            ) -> Result<Self::Output, ProviderOptionError> {
                Ok(())
            }
        }

        let forged = ProviderOptions::typed(&ForgedOpenAiOptions { unrecognized: true }).unwrap();
        let layers = ProviderOptionLayers::default().with_call(forged).unwrap();
        let provider = ProviderId::new("openai").unwrap();
        let api_mode = ApiModeId::new("responses").unwrap();
        assert!(matches!(
            layers.merge_for(
                ProviderOptionContext::new(&provider, ModelFamily::Language, Some(&api_mode)),
                &StrictMerger
            ),
            Err(ProviderOptionError::Rejected { .. })
        ));
    }

    #[test]
    fn typed_options_require_their_declared_family_and_api_mode() {
        struct Merger;
        impl ProviderOptionMerger for Merger {
            type Output = ();

            fn validate_layer(
                &self,
                _origin: ProviderOptionOrigin,
                _options: &ProviderOptions,
            ) -> Result<(), ProviderOptionError> {
                Ok(())
            }

            fn merge(
                &self,
                _layers: &ProviderOptionLayers,
            ) -> Result<Self::Output, ProviderOptionError> {
                Ok(())
            }
        }

        let provider = ProviderId::new("openai").unwrap();
        let chat = ApiModeId::new("chat-completions").unwrap();
        let typed = ProviderOptionLayers::default()
            .with_call(
                ProviderOptions::typed(&OpenAiOptions {
                    reasoning_effort: "high",
                })
                .unwrap(),
            )
            .unwrap();
        assert!(matches!(
            typed.merge_for(
                ProviderOptionContext::new(&provider, ModelFamily::Language, Some(&chat)),
                &Merger
            ),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));

        let raw = ProviderOptionLayers::default()
            .with_raw_override(
                ProviderOptions::checked_raw(provider.clone(), json!({"reasoning_effort":"high"}))
                    .unwrap(),
            )
            .unwrap();
        assert!(
            raw.merge_for(
                ProviderOptionContext::new(&provider, ModelFamily::Language, Some(&chat)),
                &Merger
            )
            .is_ok()
        );
    }

    #[test]
    fn raw_options_reject_common_transport_and_auth_aliases() {
        for key in [
            "base_uri",
            "proxy_url",
            "client_secret",
            "access-token",
            "default_headers",
            "redirect_policy",
        ] {
            let error = ProviderOptions::checked_raw(
                ProviderId::new("openai").unwrap(),
                Value::Object(Map::from_iter([(key.to_string(), json!("secret"))])),
            )
            .unwrap_err();
            assert!(matches!(error, ProviderOptionError::ProtectedField { .. }));
        }
    }

    #[test]
    fn typed_option_entry_infers_exact_target_and_rejects_another_model() {
        let options = OpenAiOptions {
            reasoning_effort: "high",
        };
        let call = CallOptions::default()
            .with_typed_provider_options(&options)
            .unwrap();
        let selected = call
            .provider_options_for(&fake_model("openai", "responses", None))
            .unwrap();
        assert_eq!(selected.typed().count(), 1);

        let error = call
            .provider_options_for(&fake_model("openai", "chat-completions", None))
            .unwrap_err();
        assert!(matches!(error, ProviderOptionError::TargetMismatch { .. }));
    }

    #[test]
    fn optional_bound_options_do_not_cross_same_label_instances() {
        let first = fake_model("openai", "responses", Some("primary"));
        let second = fake_model("openai", "responses", Some("primary"));
        let raw = json!({"future_provider_field": "sentinel"});
        let call = CallOptions::default()
            .with_optional_raw_provider_options_for(&first, raw)
            .unwrap();

        assert!(
            call.provider_options_for(&first)
                .unwrap()
                .raw_override()
                .is_some()
        );
        let other = call.provider_options_for(&second).unwrap();
        assert_eq!(other.raw_override(), None);
        assert_eq!(other.unconsumed_count(), 1);
        assert!(!format!("{call:?}").contains("sentinel"));
    }

    #[test]
    fn instance_sensitive_typed_options_require_and_preserve_exact_binding() {
        let options = SensitiveOpenAiOptions {
            mcp_authorization: "sentinel-secret",
        };
        let error = CallOptions::default()
            .with_typed_provider_options(&options)
            .unwrap_err();
        assert!(matches!(
            error,
            ProviderOptionError::InstanceBindingRequired { .. }
        ));

        let first = fake_model("openai", "responses", Some("shared-label"));
        let second = fake_model("openai", "responses", Some("shared-label"));
        let call = CallOptions::default()
            .with_typed_provider_options_for(&first, &options)
            .unwrap();
        assert_eq!(
            call.provider_options_for(&first).unwrap().typed().count(),
            1
        );
        assert!(matches!(
            call.provider_options_for(&second),
            Err(ProviderOptionError::ExactTargetMismatch { .. })
        ));
        assert!(!format!("{call:?}").contains("sentinel-secret"));
    }

    #[test]
    fn bounded_option_entries_reject_entry_target_and_byte_overflow() {
        let model = fake_model("openai", "responses", None);
        let mut call = CallOptions::default();
        for _ in 0..64 {
            call = call
                .with_optional_typed_provider_options_for(
                    &model,
                    &OpenAiOptions {
                        reasoning_effort: "high",
                    },
                )
                .unwrap();
        }
        let error = call
            .with_optional_typed_provider_options_for(
                &model,
                &OpenAiOptions {
                    reasoning_effort: "high",
                },
            )
            .unwrap_err();
        assert!(matches!(error, ProviderOptionError::TooManyEntries { .. }));

        let too_large = vec![b' '; MAX_PROVIDER_OPTION_BYTES + 1];
        let error = CallOptions::default()
            .with_raw_provider_json_for(&model, &too_large)
            .unwrap_err();
        assert!(matches!(error, ProviderOptionError::TooLarge { .. }));

        let huge_value = json!({"items": vec![Value::Null; MAX_PROVIDER_OPTION_BYTES]});
        let error = CallOptions::default()
            .with_raw_provider_options_for(&model, huge_value)
            .unwrap_err();
        assert!(matches!(error, ProviderOptionError::TooLarge { .. }));

        let mut targets = CallOptions::default();
        for index in 0..MAX_PROVIDER_OPTION_TARGETS {
            let target = fake_model("openai", "responses", Some(&format!("route-{index}")));
            targets = targets
                .with_optional_typed_provider_options_for(
                    &target,
                    &OpenAiOptions {
                        reasoning_effort: "high",
                    },
                )
                .unwrap();
        }
        let overflow = fake_model("openai", "responses", Some("route-overflow"));
        let error = targets
            .with_optional_typed_provider_options_for(
                &overflow,
                &OpenAiOptions {
                    reasoning_effort: "high",
                },
            )
            .unwrap_err();
        assert!(matches!(error, ProviderOptionError::TooManyTargets { .. }));

        let mut aggregate = CallOptions::default();
        let payload = "x".repeat(MAX_PROVIDER_OPTION_BYTES - 1024);
        for index in 0..8 {
            let target = fake_model("openai", "responses", Some(&format!("raw-{index}")));
            aggregate = aggregate
                .with_optional_raw_provider_options_for(
                    &target,
                    json!({"future_blob": payload.clone()}),
                )
                .unwrap();
        }
        let overflow = fake_model("openai", "responses", Some("raw-overflow"));
        let error = aggregate
            .with_optional_raw_provider_options_for(&overflow, json!({"future_blob": payload}))
            .unwrap_err();
        assert!(matches!(
            error,
            ProviderOptionError::AggregateTooLarge { .. }
        ));
    }
}
