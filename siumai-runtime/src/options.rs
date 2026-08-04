use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use siumai_core::{
    ApiModeId, CallOptions, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
    Model, ModelId, PlatformId, ProtocolId, ProviderId, ProviderOptions, RouteId,
};
use thiserror::Error;

use crate::RunBudget;
use crate::call::validate_request;

/// Complete model target used for model-default selection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelTarget {
    route: Option<RouteId>,
    provider: ProviderId,
    platform: Option<PlatformId>,
    protocol: Option<ProtocolId>,
    api_mode: Option<ApiModeId>,
    model: ModelId,
}

impl ModelTarget {
    /// Create a provider/model target without Registry or protocol metadata.
    pub fn new(provider: ProviderId, model: ModelId) -> Self {
        Self {
            route: None,
            provider,
            platform: None,
            protocol: None,
            api_mode: None,
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
            provider: scope.provider_id().clone(),
            platform: scope.platform().cloned(),
            protocol: scope.protocol().cloned(),
            api_mode: scope.api_mode().cloned(),
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
        self.platform = Some(platform);
        self
    }

    pub fn with_protocol(mut self, protocol: ProtocolId) -> Self {
        self.protocol = Some(protocol);
        self
    }

    pub fn with_api_mode(mut self, api_mode: ApiModeId) -> Self {
        self.api_mode = Some(api_mode);
        self
    }

    pub fn route(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn platform(&self) -> Option<&PlatformId> {
        self.platform.as_ref()
    }

    pub fn protocol(&self) -> Option<&ProtocolId> {
        self.protocol.as_ref()
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.api_mode.as_ref()
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
    #[error("runtime defaults and step options must use typed provider options")]
    RawProviderOptions,
    #[error("runtime step already has provider options for `{provider}`")]
    DuplicateStepOptions { provider: ProviderId },
    #[error("model defaults target `{expected}` but options use namespace `{actual}`")]
    ModelDefaultsNamespace {
        expected: ProviderId,
        actual: ProviderId,
    },
}

/// Options owned by one model step inside the high-level runtime.
#[derive(Debug, Clone, Default)]
pub struct StepOptions {
    provider_options: BTreeMap<ProviderId, ProviderOptions>,
}

impl StepOptions {
    pub fn with_provider_options(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        let provider = options.namespace().clone();
        if self.provider_options.contains_key(&provider) {
            return Err(RuntimeConfigError::DuplicateStepOptions { provider });
        }
        self.provider_options.insert(provider, options);
        Ok(self)
    }

    pub fn provider_options_for(&self, provider: &ProviderId) -> Option<&ProviderOptions> {
        self.provider_options.get(provider)
    }
}

#[derive(Debug, Clone, Default)]
struct RuntimeDefaults {
    route: BTreeMap<RouteId, ProviderOptions>,
    model: BTreeMap<ModelTarget, ProviderOptions>,
    budget: RunBudget,
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
    ) -> Result<LanguageResponse, siumai_core::Error>
    where
        M: LanguageModel + ?Sized,
    {
        validate_request(&request)?;
        model
            .generate(request, self.prepare_options(model, &step, options))
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
        validate_request(&request)?;
        model
            .stream(request, self.prepare_options(model, &step, options))
            .await
    }

    fn prepare_options<M>(
        &self,
        model: &M,
        step: &StepOptions,
        mut options: CallOptions,
    ) -> CallOptions
    where
        M: Model + ?Sized,
    {
        if let Some(route) = model.route_id()
            && let Some(defaults) = self.defaults.route.get(route)
        {
            options = options.with_route_default_provider_options(defaults.clone());
        }

        let exact_target = ModelTarget::from_model(model);
        let route_independent_target = exact_target.clone().without_route();
        if let Some(defaults) = self
            .defaults
            .model
            .get(&exact_target)
            .or_else(|| self.defaults.model.get(&route_independent_target))
        {
            options = options.with_model_default_provider_options(defaults.clone());
        }

        if let Some(step_options) = step.provider_options_for(model.provider_id()) {
            options = options.with_runtime_step_provider_options(step_options.clone());
        }
        options
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

    pub fn with_route_defaults(
        mut self,
        route: RouteId,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        if self.defaults.route.contains_key(&route) {
            return Err(RuntimeConfigError::DuplicateRouteDefaults { route });
        }
        self.defaults.route.insert(route, options);
        Ok(self)
    }

    pub fn with_model_defaults(
        mut self,
        target: ModelTarget,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        if target.provider() != options.namespace() {
            return Err(RuntimeConfigError::ModelDefaultsNamespace {
                expected: target.provider().clone(),
                actual: options.namespace().clone(),
            });
        }
        if self.defaults.model.contains_key(&target) {
            return Err(RuntimeConfigError::DuplicateModelDefaults {
                target: Box::new(target),
            });
        }
        self.defaults.model.insert(target, options);
        Ok(self)
    }

    pub fn build(self) -> Runtime {
        Runtime {
            defaults: Arc::new(self.defaults),
        }
    }
}

fn ensure_typed(options: &ProviderOptions) -> Result<(), RuntimeConfigError> {
    if options.is_raw() {
        Err(RuntimeConfigError::RawProviderOptions)
    } else {
        Ok(())
    }
}
