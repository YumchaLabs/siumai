use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use siumai_core::{
    ApiModeId, CallOptions, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
    Model, ModelId, PlatformId, ProtocolId, ProviderId, ProviderOptions, RouteId,
};
use thiserror::Error;

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

    pub fn route(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
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
    #[error("one runtime step can contain only one typed provider-option layer")]
    DuplicateStepOptions,
}

/// Options owned by one model step inside the high-level runtime.
#[derive(Debug, Clone, Default)]
pub struct StepOptions {
    provider_options: Option<ProviderOptions>,
}

impl StepOptions {
    pub fn with_provider_options(
        mut self,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        if self.provider_options.is_some() {
            return Err(RuntimeConfigError::DuplicateStepOptions);
        }
        self.provider_options = Some(options);
        Ok(self)
    }

    pub fn provider_options(&self) -> Option<&ProviderOptions> {
        self.provider_options.as_ref()
    }
}

#[derive(Debug, Clone, Default)]
struct RuntimeDefaults {
    route: BTreeMap<RouteId, ProviderOptions>,
    model: BTreeMap<ModelTarget, ProviderOptions>,
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

        if let Some(step_options) = step.provider_options() {
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
    pub fn with_route_defaults(
        mut self,
        route: RouteId,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        if self.defaults.route.insert(route.clone(), options).is_some() {
            return Err(RuntimeConfigError::DuplicateRouteDefaults { route });
        }
        Ok(self)
    }

    pub fn with_model_defaults(
        mut self,
        target: ModelTarget,
        options: ProviderOptions,
    ) -> Result<Self, RuntimeConfigError> {
        ensure_typed(&options)?;
        if self
            .defaults
            .model
            .insert(target.clone(), options)
            .is_some()
        {
            return Err(RuntimeConfigError::DuplicateModelDefaults {
                target: Box::new(target),
            });
        }
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
