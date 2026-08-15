use std::sync::Arc;

use serde::{Deserialize, Serialize};
use siumai_core::{
    ApiModeId, CallOptions, LanguageCallError, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, Model, ModelId, PlatformId, ProtocolId, ProviderId, ProviderOptionError,
    ProviderOptionPatch, ProviderScope, ReplayDomain, RouteId, TypedProviderOptions,
};
use thiserror::Error;

use crate::RunBudget;

/// Complete model target used for model-default selection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelTarget {
    route: Option<RouteId>,
    #[serde(flatten)]
    scope: ProviderScope,
    model: ModelId,
}

impl ModelTarget {
    /// Create a provider/model target without Registry or protocol metadata.
    pub fn new(provider: ProviderId, model: ModelId) -> Self {
        Self {
            route: None,
            scope: ProviderScope::new(provider),
            model,
        }
    }

    pub fn from_model<M>(model: &M) -> Self
    where
        M: Model + ?Sized,
    {
        let descriptor = model.descriptor();
        let scope = descriptor.scope();
        Self {
            route: model.route_id().cloned(),
            scope: scope.clone(),
            model: descriptor.model().clone(),
        }
    }

    /// Select defaults independent of a Registry route.
    pub fn without_route(mut self) -> Self {
        self.route = None;
        self
    }

    pub fn with_route(mut self, route: RouteId) -> Self {
        self.route = Some(route);
        self
    }

    pub fn with_platform(mut self, platform: PlatformId) -> Self {
        self.scope = self.scope.with_platform(platform);
        self
    }

    pub fn with_protocol(mut self, protocol: ProtocolId) -> Self {
        self.scope = self.scope.with_protocol(protocol);
        self
    }

    pub fn with_api_mode(mut self, api_mode: ApiModeId) -> Self {
        self.scope = self.scope.with_api_mode(api_mode);
        self
    }

    pub fn with_replay_domain(mut self, replay_domain: ReplayDomain) -> Self {
        self.scope = self.scope.with_replay_domain(replay_domain);
        self
    }

    pub fn route(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }

    pub fn provider(&self) -> &ProviderId {
        self.scope.provider_id()
    }

    pub fn platform(&self) -> Option<&PlatformId> {
        self.scope.platform()
    }

    pub fn protocol(&self) -> Option<&ProtocolId> {
        self.scope.protocol()
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.scope.api_mode()
    }

    pub fn replay_domain(&self) -> Option<&ReplayDomain> {
        self.scope.replay_domain()
    }

    pub fn scope(&self) -> &ProviderScope {
        &self.scope
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RuntimeConfigError {
    #[error("route `{route}` already has provider-option defaults")]
    DuplicateRouteDefaults { route: RouteId },
    #[error("model target already has provider-option defaults")]
    DuplicateModelDefaults { target: Box<ModelTarget> },
    #[error("runtime step already has provider options for this exact model target")]
    DuplicateStepOptions { target: Box<ModelTarget> },
    #[error("route defaults require a Registry-selected model target")]
    MissingRoute { target: Box<ModelTarget> },
    #[error("invalid runtime provider options: {0}")]
    ProviderOptions(#[from] ProviderOptionError),
}

/// Options owned by one model step inside the high-level runtime.
#[derive(Debug, Clone, Default)]
pub struct StepOptions {
    patches: Vec<ProviderOptionPatch>,
}

impl StepOptions {
    /// Add typed options for one exact configured model that may own this step.
    pub fn with_provider_options<M, T>(
        mut self,
        model: &M,
        value: &T,
    ) -> Result<Self, RuntimeConfigError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let patch = ProviderOptionPatch::typed_for_model(model, value)?;
        if self
            .patches
            .iter()
            .any(|existing| existing.same_target(&patch))
        {
            return Err(RuntimeConfigError::DuplicateStepOptions {
                target: Box::new(ModelTarget::from_model(model)),
            });
        }
        self.patches.push(patch);
        Ok(self)
    }
}

#[derive(Debug, Clone)]
struct RouteDefaults {
    route: RouteId,
    patch: ProviderOptionPatch,
}

#[derive(Debug, Clone)]
struct ModelDefaults {
    target: ModelTarget,
    patch: ProviderOptionPatch,
}

#[derive(Debug, Clone, Default)]
struct RuntimeDefaults {
    route: Vec<RouteDefaults>,
    model: Vec<ModelDefaults>,
    budget: RunBudget,
}

/// Runtime-private source ordering. Providers consume only the resulting
/// exact-target patch order and never observe these host-level source labels.
struct OrderedProviderPatches<'a> {
    patches: Vec<&'a ProviderOptionPatch>,
}

impl<'a> OrderedProviderPatches<'a> {
    fn for_call<M>(defaults: &'a RuntimeDefaults, model: &M, step: &'a StepOptions) -> Self
    where
        M: Model + ?Sized,
    {
        let mut patches = Vec::new();

        if let Some(route) = model.route_id() {
            patches.extend(
                defaults
                    .route
                    .iter()
                    .filter(|defaults| &defaults.route == route)
                    .map(|defaults| &defaults.patch),
            );
        }

        let exact_target = ModelTarget::from_model(model);
        let route_independent_target = exact_target.clone().without_route();
        patches.extend(
            defaults
                .model
                .iter()
                .filter(|defaults| {
                    defaults.target == exact_target || defaults.target == route_independent_target
                })
                .map(|defaults| &defaults.patch),
        );
        patches.extend(step.patches.iter());
        patches.retain(|patch| patch.matches_model(model));

        Self { patches }
    }
}

/// Immutable, clone-cheap high-level runtime configuration.
#[derive(Debug, Clone, Default)]
pub struct Runtime {
    defaults: Arc<RuntimeDefaults>,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub fn run_budget(&self) -> &RunBudget {
        &self.defaults.budget
    }

    pub async fn generate<M>(
        &self,
        model: &M,
        request: LanguageRequest,
        step: StepOptions,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError>
    where
        M: LanguageModel + ?Sized,
    {
        crate::single_step::SingleStep::new(self, model, &step)
            .generate(request, options)
            .await
    }

    pub async fn stream<M>(
        &self,
        model: &M,
        request: LanguageRequest,
        step: StepOptions,
        options: CallOptions,
    ) -> Result<LanguageStream, siumai_core::Error>
    where
        M: LanguageModel + ?Sized,
    {
        crate::single_step::SingleStep::new(self, model, &step)
            .stream(request, options)
            .await
    }

    pub(crate) fn prepare_options<M>(
        &self,
        model: &M,
        step: &StepOptions,
        options: CallOptions,
    ) -> Result<CallOptions, siumai_core::Error>
    where
        M: Model + ?Sized,
    {
        let patches = OrderedProviderPatches::for_call(&self.defaults, model, step)
            .patches
            .into_iter()
            .cloned();
        options
            .prepend_provider_options(patches)
            .map_err(runtime_provider_options_error)
    }
}

/// Mutable assembly for one immutable [`Runtime`] configuration.
#[derive(Debug, Clone, Default)]
pub struct RuntimeBuilder {
    defaults: RuntimeDefaults,
}

impl RuntimeBuilder {
    pub fn with_run_budget(mut self, budget: RunBudget) -> Self {
        self.defaults.budget = budget;
        self
    }

    /// Add route-scoped typed defaults bound to one exact configured model.
    pub fn with_route_defaults<M, T>(
        mut self,
        model: &M,
        value: &T,
    ) -> Result<Self, RuntimeConfigError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let target = ModelTarget::from_model(model);
        let route = model
            .route_id()
            .cloned()
            .ok_or_else(|| RuntimeConfigError::MissingRoute {
                target: Box::new(target),
            })?;
        let patch = ProviderOptionPatch::typed_for_model(model, value)?;
        if self
            .defaults
            .route
            .iter()
            .any(|existing| existing.route == route && existing.patch.same_target(&patch))
        {
            return Err(RuntimeConfigError::DuplicateRouteDefaults { route });
        }
        self.defaults.route.push(RouteDefaults { route, patch });
        Ok(self)
    }

    /// Add model-scoped typed defaults bound to one exact configured model.
    pub fn with_model_defaults<M, T>(
        mut self,
        model: &M,
        value: &T,
    ) -> Result<Self, RuntimeConfigError>
    where
        M: Model + ?Sized,
        T: TypedProviderOptions,
    {
        let target = ModelTarget::from_model(model);
        let patch = ProviderOptionPatch::typed_for_model(model, value)?;
        if self
            .defaults
            .model
            .iter()
            .any(|existing| existing.target == target && existing.patch.same_target(&patch))
        {
            return Err(RuntimeConfigError::DuplicateModelDefaults {
                target: Box::new(target),
            });
        }
        self.defaults.model.push(ModelDefaults { target, patch });
        Ok(self)
    }

    pub fn build(self) -> Runtime {
        Runtime {
            defaults: Arc::new(self.defaults),
        }
    }
}

fn runtime_provider_options_error(source: ProviderOptionError) -> siumai_core::Error {
    siumai_core::Error::new(
        siumai_core::ErrorKind::Configuration,
        "invalid runtime provider options",
    )
    .with_source(source)
}
