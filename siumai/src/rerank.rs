//! Provider-neutral reranking facade.

use crate::call::CallState;
use crate::{CallOptions, Error, ErrorKind, RerankModel, RerankRequest, RerankResponse};
use siumai_core::ProviderOptionError;

/// A single-use rerank call bound to one live model handle.
pub struct RerankCall<'a, M: RerankModel + ?Sized> {
    state: CallState<'a, M, RerankRequest>,
}

impl<'a, M> RerankCall<'a, M>
where
    M: RerankModel + ?Sized,
{
    fn new(model: &'a M, request: RerankRequest) -> Self {
        Self {
            state: CallState::new(model, request),
        }
    }

    /// Return the complete portable request owned by this call.
    pub fn request(&self) -> &RerankRequest {
        self.state.request()
    }

    /// Return the replaceable caller-provided option baseline.
    ///
    /// Builder-owned provider patches are intentionally not exposed through
    /// this borrowed view; effective assembly is fallible and remains private.
    pub fn base_options(&self) -> &CallOptions {
        self.state.base_options()
    }

    /// Replace the call-option baseline while retaining builder-owned patches.
    ///
    /// The complete baseline-plus-patch candidate is validated synchronously
    /// before the replacement is accepted.
    pub fn with_options(self, options: CallOptions) -> Result<Self, ProviderOptionError> {
        Ok(Self {
            state: self.state.with_options(options)?,
        })
    }

    /// Append one typed provider-option patch for this exact live model.
    ///
    /// The patch is validated together with the current baseline and all
    /// earlier builder patches before it is accepted.
    pub fn with_provider_options<T>(self, value: &T) -> Result<Self, ProviderOptionError>
    where
        T: siumai_core::TypedProviderOptions,
    {
        Ok(Self {
            state: self.state.with_provider_options(value)?,
        })
    }

    fn preflight(&self) -> Result<CallOptions, Error> {
        let mut options = self
            .state
            .resolve_deadline()
            .map_err(Error::from)
            .map_err(|error| self.state.contextualize_error(error))?;
        self.state
            .model()
            .limits()
            .validate(self.state.request())
            .map_err(|error| self.state.contextualize_error(error))?;
        self.state
            .validate_provider_options(&mut options)
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "provider options do not match the selected rerank model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Execute one rerank request and return the complete response.
    pub async fn rerank(self) -> Result<RerankResponse, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.rerank(request, options).await
    }
}

/// Bind one complete rerank request to the selected live model.
pub fn call<M>(model: &M, request: RerankRequest) -> RerankCall<'_, M>
where
    M: RerankModel + ?Sized,
{
    RerankCall::new(model, request)
}

/// Execute one rerank request with default call options.
pub async fn rerank<M>(model: &M, request: RerankRequest) -> Result<RerankResponse, Error>
where
    M: RerankModel + ?Sized,
{
    call(model, request).rerank().await
}
