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
/// Implementations are ergonomic codecs, not a trust boundary. The selected
/// provider still owns request-body schema and relationship validation after
/// core has checked the exact target and generic resource bounds.
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

/// One opaque but bounded provider-option patch.
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

    fn matches_model<M: Model + ?Sized>(
        &self,
        model: &M,
        selected_route: Option<&RouteId>,
    ) -> bool {
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
                route.as_ref() == selected_route
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
}

impl fmt::Debug for ProviderOptionSelection<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptionSelection")
            .field("typed_count", &self.typed.len())
            .field("has_raw_override", &self.raw_override.is_some())
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
}

fn validate_options_target(
    target: &ProviderOptionTarget,
    options: &ProviderOptions,
) -> Result<(), ProviderOptionError> {
    if options.namespace() != target.provider() {
        return Err(ProviderOptionError::NamespaceMismatch {
            expected: target.provider().to_string(),
            actual: options.namespace().to_string(),
        });
    }

    if let Some(actual_family) = options.model_family()
        && (actual_family != target.family() || options.api_mode() != target.api_mode())
    {
        return Err(ProviderOptionError::TargetMismatch {
            expected_family: target.family(),
            expected_api_mode: target.api_mode().map(ApiModeId::to_string),
            actual_family,
            actual_api_mode: options.api_mode().map(ApiModeId::to_string),
        });
    }

    if options.is_instance_sensitive() && !target.is_instance_bound() {
        return Err(ProviderOptionError::InstanceBindingRequired {
            namespace: options.namespace().to_string(),
        });
    }
    Ok(())
}

/// Core-owned validated provider-option patch used by runtime assembly.
///
/// Ordinary callers should use the typed [`CallOptions`] builders. This
/// carrier exists so higher-level orchestration can retain one validated
/// target/options pair without duplicating route, scope, family, API-mode, or
/// configured-instance identity.
#[doc(hidden)]
#[derive(Clone)]
pub struct ProviderOptionPatch {
    target: ProviderOptionTarget,
    options: ProviderOptions,
}

impl fmt::Debug for ProviderOptionPatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderOptionPatch")
            .field("target", &self.target)
            .field("options", &self.options)
            .finish()
    }
}

impl ProviderOptionPatch {
    /// Create a validated typed patch bound to one configured model.
    pub fn typed_for_model<M, T>(model: &M, value: &T) -> Result<Self, ProviderOptionError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let options = ProviderOptions::typed(value)?;
        let target = ProviderOptionTarget::for_model(model);
        validate_options_target(&target, &options)?;
        Ok(Self { target, options })
    }

    pub fn target(&self) -> &ProviderOptionTarget {
        &self.target
    }

    pub fn matches_model<M: Model + ?Sized>(&self, model: &M) -> bool {
        self.target.matches_model(model, model.route_id())
    }

    pub fn same_target(&self, other: &Self) -> bool {
        self.target == other.target
    }

    fn into_parts(self) -> (ProviderOptionTarget, ProviderOptions) {
        (self.target, self.options)
    }
}

/// Controls shared by all six stable model families.
#[derive(Clone, Default)]
pub struct CallOptions {
    deadline: Option<Instant>,
    cancellation: Cancellation,
    retry: RetryIntent,
    selected_route_context: Option<RouteId>,
    exact_provider_options: Vec<ExactProviderOptionEntry>,
}

impl fmt::Debug for CallOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CallOptions")
            .field("deadline", &self.deadline)
            .field("cancellation", &self.cancellation)
            .field("retry", &self.retry)
            .field("selected_route_context", &self.selected_route_context)
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

    pub fn has_provider_options(&self) -> bool {
        !self.exact_provider_options.is_empty()
    }

    /// Select the exact-target entries that belong to one configured model.
    pub fn provider_options_for<M: Model + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ProviderOptionSelection<'_>, ProviderOptionError> {
        let selected_route = model.route_id().or(self.selected_route_context.as_ref());
        let mut typed = Vec::new();
        let mut raw_override = None;

        for entry in &self.exact_provider_options {
            if !entry.target.matches_model(model, selected_route) {
                if entry.applicability == ProviderOptionApplicability::Required {
                    return Err(entry.target.mismatch_error(model));
                }
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
        })
    }

    /// Add reusable typed options for the model selected by this call.
    ///
    /// Values use a fail-closed configured-instance default. A value that has
    /// not explicitly opted into [`ProviderOptionBindingRequirement::Reusable`]
    /// must use [`Self::with_provider_options_for`] instead.
    pub fn with_provider_options<T: TypedProviderOptions>(
        mut self,
        value: &T,
    ) -> Result<Self, ProviderOptionError> {
        let options = ProviderOptions::typed(value)?;
        let target = ProviderOptionTarget::typed::<T>()?;
        validate_options_target(&target, &options)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add typed options bound to the concrete model receiving the call.
    pub fn with_provider_options_for<M, T>(
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
        validate_options_target(&target, &options)?;
        self.push_exact_provider_option(ExactProviderOptionEntry {
            applicability: ProviderOptionApplicability::Required,
            target,
            options,
        })?;
        Ok(self)
    }

    /// Add an optional typed fallback for one exact configured model.
    pub fn with_optional_provider_options_for<M, T>(
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
        validate_options_target(&target, &options)?;
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

    /// Prepend runtime-owned exact-target patches before the caller entries.
    ///
    /// This is an assembly seam for runtime defaults and is intentionally
    /// hidden from ordinary callers. The input order is preserved, every entry
    /// is required for its exact target, and all normal bounds are rechecked
    /// against the existing call entries.
    #[doc(hidden)]
    pub fn prepend_provider_options<I>(mut self, patches: I) -> Result<Self, ProviderOptionError>
    where
        I: IntoIterator<Item = ProviderOptionPatch>,
    {
        let existing = std::mem::take(&mut self.exact_provider_options);
        for patch in patches {
            let (target, options) = patch.into_parts();
            self.push_exact_provider_option(ExactProviderOptionEntry {
                applicability: ProviderOptionApplicability::Required,
                target,
                options,
            })?;
        }
        for entry in existing {
            self.push_exact_provider_option(entry)?;
        }
        Ok(self)
    }

    fn push_exact_provider_option(
        &mut self,
        entry: ExactProviderOptionEntry,
    ) -> Result<(), ProviderOptionError> {
        if self.exact_provider_options.len() >= MAX_PROVIDER_OPTION_ENTRIES {
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
            .exact_provider_options
            .iter()
            .map(|entry| entry.options.retained_bytes())
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

    /// Preserve the Registry route selected by an outer model wrapper while
    /// the configured provider consumes options through its route-less inner
    /// model handle.
    ///
    /// This is an assembly seam for Registry and equivalent routing layers,
    /// not a provider option or a business routing decision.
    #[doc(hidden)]
    pub fn with_selected_route_context(mut self, route: RouteId) -> Self {
        self.selected_route_context = Some(route);
        self
    }
}

// Provider annotations retain their existing node-local validation through
// this crate-private helper. Provider call options deliberately do not use it;
// selected provider codecs own canonical and protected request-body paths.
const ANNOTATION_PROTECTED_FIELDS: &[&str] = &[
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
        if ANNOTATION_PROTECTED_FIELDS.contains(&normalized.as_str()) {
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
    use serde_json::{Map, Value, json};

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

    #[derive(Serialize)]
    struct Layer {
        value: &'static str,
    }

    impl TypedProviderOptions for Layer {
        const NAMESPACE: &'static str = "openai";
        const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
        const API_MODE: Option<&'static str> = Some("responses");

        fn binding_requirement(&self) -> ProviderOptionBindingRequirement {
            ProviderOptionBindingRequirement::Reusable
        }
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
    fn nested_provider_body_headers_are_accepted_by_core_bounds() {
        let options = ProviderOptions::checked_raw(
            ProviderId::new("openai").unwrap(),
            json!({
                "provider_payload": {
                    "headers": {
                        "x-provider-feature": "future"
                    }
                }
            }),
        )
        .unwrap();

        assert_eq!(
            options.value()["provider_payload"]["headers"]["x-provider-feature"],
            "future"
        );
    }

    #[test]
    fn raw_shape_bounds_remain_strict_without_a_global_field_denylist() {
        let too_large = vec![b' '; MAX_PROVIDER_OPTION_BYTES + 1];
        assert!(matches!(
            ProviderOptions::checked_raw_json(ProviderId::new("openai").unwrap(), &too_large),
            Err(ProviderOptionError::TooLarge { .. })
        ));

        let mut too_many_fields = Map::new();
        for index in 0..=MAX_PROVIDER_OPTION_FIELDS {
            too_many_fields.insert(format!("field-{index}"), Value::Null);
        }
        assert!(matches!(
            ProviderOptions::checked_raw(
                ProviderId::new("openai").unwrap(),
                Value::Object(too_many_fields),
            ),
            Err(ProviderOptionError::TooManyFields { .. })
        ));

        let mut too_deep = Value::Bool(true);
        for _ in 0..=MAX_PROVIDER_OPTION_DEPTH {
            too_deep = json!({"nested": too_deep});
        }
        assert!(matches!(
            ProviderOptions::checked_raw(ProviderId::new("openai").unwrap(), too_deep),
            Err(ProviderOptionError::TooDeep { .. })
        ));
    }

    #[test]
    fn ordinary_typed_options_infer_a_required_target() {
        let call = CallOptions::default()
            .with_provider_options(&OpenAiOptions {
                reasoning_effort: "high",
            })
            .unwrap();

        assert_eq!(
            call.provider_options_for(&fake_model("openai", "responses", None))
                .unwrap()
                .typed()
                .count(),
            1
        );
        assert!(matches!(
            call.provider_options_for(&fake_model("openai", "chat-completions", None)),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));
    }

    #[test]
    fn exact_typed_insertion_validates_namespace_family_and_api_mode() {
        #[derive(Serialize)]
        struct ForeignOptions {
            enabled: bool,
        }

        impl TypedProviderOptions for ForeignOptions {
            const NAMESPACE: &'static str = "anthropic";
            const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
            const API_MODE: Option<&'static str> = Some("responses");
        }

        #[derive(Serialize)]
        struct EmbeddingOptions {
            enabled: bool,
        }

        impl TypedProviderOptions for EmbeddingOptions {
            const NAMESPACE: &'static str = "openai";
            const MODEL_FAMILY: ModelFamily = ModelFamily::Embedding;
            const API_MODE: Option<&'static str> = Some("responses");
        }

        let responses = fake_model("openai", "responses", None);
        assert!(matches!(
            CallOptions::default()
                .with_provider_options_for(&responses, &ForeignOptions { enabled: true }),
            Err(ProviderOptionError::NamespaceMismatch { .. })
        ));
        assert!(matches!(
            CallOptions::default()
                .with_provider_options_for(&responses, &EmbeddingOptions { enabled: true }),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));

        let chat = fake_model("openai", "chat-completions", None);
        assert!(matches!(
            CallOptions::default().with_provider_options_for(
                &chat,
                &OpenAiOptions {
                    reasoning_effort: "high",
                },
            ),
            Err(ProviderOptionError::TargetMismatch { .. })
        ));
    }

    #[test]
    fn required_and_optional_targets_have_typed_mismatch_behavior() {
        let first = fake_model("openai", "responses", Some("primary"));
        let second = fake_model("openai", "responses", Some("primary"));

        let required = CallOptions::default()
            .with_raw_provider_options_for(&first, json!({"future": true}))
            .unwrap();
        assert!(matches!(
            required.provider_options_for(&second),
            Err(ProviderOptionError::ExactTargetMismatch { .. })
        ));

        let optional = CallOptions::default()
            .with_optional_raw_provider_options_for(&first, json!({"future": true}))
            .unwrap();
        let selection = optional.provider_options_for(&second).unwrap();
        assert_eq!(selection.raw_override(), None);
    }

    #[test]
    fn sensitive_typed_options_fail_closed_and_isolate_same_label_instances() {
        let sensitive = SensitiveOpenAiOptions {
            mcp_authorization: "sentinel-secret",
        };
        assert!(matches!(
            CallOptions::default().with_provider_options(&sensitive),
            Err(ProviderOptionError::InstanceBindingRequired { .. })
        ));

        let first = fake_model("openai", "responses", Some("shared-label"));
        let second = fake_model("openai", "responses", Some("shared-label"));
        let call = CallOptions::default()
            .with_provider_options_for(&first, &sensitive)
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
    fn ordered_typed_patches_precede_the_final_raw_override() {
        let model = fake_model("openai", "responses", None);
        let call = CallOptions::default()
            .with_provider_options(&Layer { value: "call" })
            .unwrap()
            .with_provider_options_for(&model, &Layer { value: "model" })
            .unwrap()
            .with_raw_provider_options_for(&model, json!({"future": "raw"}))
            .unwrap();

        let selection = call.provider_options_for(&model).unwrap();
        let values = selection
            .typed()
            .map(|options| options.value()["value"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values, vec!["call", "model"]);
        assert_eq!(selection.raw_override().unwrap().value()["future"], "raw");
    }

    #[test]
    fn prepend_runtime_patches_preserve_source_order_before_call_entries() {
        let model = fake_model("openai", "responses", None);
        let call = CallOptions::default()
            .with_provider_options(&Layer { value: "call" })
            .unwrap()
            .prepend_provider_options(vec![
                ProviderOptionPatch::typed_for_model(&model, &Layer { value: "route" }).unwrap(),
                ProviderOptionPatch::typed_for_model(&model, &Layer { value: "step" }).unwrap(),
            ])
            .unwrap();

        let values = call
            .provider_options_for(&model)
            .unwrap()
            .typed()
            .map(|options| options.value()["value"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values, vec!["route", "step", "call"]);
    }

    #[test]
    fn selected_route_context_preserves_binding_through_a_route_less_delegate() {
        let routed = fake_model("openai", "responses", Some("primary"));
        let inner = FakeModel {
            descriptor: routed.descriptor.clone(),
            route: None,
        };
        let call = CallOptions::default()
            .with_raw_provider_options_for(&routed, json!({"future": true}))
            .unwrap();

        assert!(matches!(
            call.provider_options_for(&inner),
            Err(ProviderOptionError::ExactTargetMismatch { .. })
        ));
        let delegated = call.with_selected_route_context(routed.route.clone().unwrap());
        assert!(
            delegated
                .provider_options_for(&inner)
                .unwrap()
                .raw_override()
                .is_some()
        );
    }

    #[test]
    fn duplicate_raw_target_and_bounds_are_enforced() {
        let model = fake_model("openai", "responses", None);
        let mut entries = CallOptions::default();
        for _ in 0..MAX_PROVIDER_OPTION_ENTRIES {
            entries = entries
                .with_optional_provider_options_for(
                    &model,
                    &OpenAiOptions {
                        reasoning_effort: "high",
                    },
                )
                .unwrap();
        }
        assert!(matches!(
            entries.with_optional_provider_options_for(
                &model,
                &OpenAiOptions {
                    reasoning_effort: "high",
                },
            ),
            Err(ProviderOptionError::TooManyEntries { .. })
        ));

        let duplicate = CallOptions::default()
            .with_raw_provider_options_for(&model, json!({"future": true}))
            .unwrap()
            .with_raw_provider_options_for(&model, json!({"future": false}));
        assert!(matches!(
            duplicate,
            Err(ProviderOptionError::DuplicateRawTarget)
        ));

        let mut targets = CallOptions::default();
        for index in 0..MAX_PROVIDER_OPTION_TARGETS {
            let target = fake_model("openai", "responses", Some(&format!("route-{index}")));
            targets = targets
                .with_optional_provider_options_for(
                    &target,
                    &OpenAiOptions {
                        reasoning_effort: "high",
                    },
                )
                .unwrap();
        }
        let overflow = fake_model("openai", "responses", Some("route-overflow"));
        assert!(matches!(
            targets.with_optional_provider_options_for(
                &overflow,
                &OpenAiOptions {
                    reasoning_effort: "high",
                },
            ),
            Err(ProviderOptionError::TooManyTargets { .. })
        ));

        let payload = "x".repeat(MAX_PROVIDER_OPTION_BYTES - 1024);
        let mut aggregate = CallOptions::default();
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
        assert!(matches!(
            aggregate.with_optional_raw_provider_options_for(
                &overflow,
                json!({"future_blob": payload}),
            ),
            Err(ProviderOptionError::AggregateTooLarge { .. })
        ));
    }
}
