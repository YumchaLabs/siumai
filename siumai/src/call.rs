use siumai_core::{
    CallOptions, CallOptionsError, Error, Model, ProviderOptionError, ProviderOptionPatch,
    TypedProviderOptions,
};

pub(crate) struct CallState<'a, M: ?Sized, R> {
    model: &'a M,
    request: R,
    base_options: CallOptions,
    provider_patches: Vec<ProviderOptionPatch>,
}

impl<'a, M, R> CallState<'a, M, R>
where
    M: Model + ?Sized,
{
    pub(crate) fn new(model: &'a M, request: R) -> Self {
        Self {
            model,
            request,
            base_options: CallOptions::default(),
            provider_patches: Vec::new(),
        }
    }

    pub(crate) fn request(&self) -> &R {
        &self.request
    }

    pub(crate) fn base_options(&self) -> &CallOptions {
        &self.base_options
    }

    pub(crate) fn with_options(
        mut self,
        options: CallOptions,
    ) -> Result<Self, ProviderOptionError> {
        let mut candidate = options.clone();
        candidate.append_provider_options(self.provider_patches.iter().cloned())?;
        candidate.provider_options_for(self.model)?;
        self.base_options = options;
        Ok(self)
    }

    pub(crate) fn with_provider_options<T>(mut self, value: &T) -> Result<Self, ProviderOptionError>
    where
        T: TypedProviderOptions,
    {
        let patch = ProviderOptionPatch::typed_for_model(self.model, value)?;
        let mut candidate = self.base_options.clone();
        candidate.append_provider_options(
            self.provider_patches
                .iter()
                .cloned()
                .chain(std::iter::once(patch.clone())),
        )?;
        candidate.provider_options_for(self.model)?;
        self.provider_patches.push(patch);
        Ok(self)
    }

    pub(crate) fn resolve_deadline(&self) -> Result<CallOptions, CallOptionsError> {
        self.base_options.clone().resolve_deadline()
    }

    pub(crate) fn validate_provider_options(
        &self,
        options: &mut CallOptions,
    ) -> Result<(), ProviderOptionError> {
        options.append_provider_options(self.provider_patches.iter().cloned())?;
        options.provider_options_for(self.model).map(|_| ())
    }

    pub(crate) fn contextualize_error(&self, error: Error) -> Error {
        match self.model.route_id() {
            Some(route) => error.with_route(route.clone()),
            None => error,
        }
    }

    pub(crate) fn into_target(self) -> (&'a M, R) {
        (self.model, self.request)
    }
}
