//! Provider-neutral reranking facade.

use crate::call::define_call_wrapper;
use crate::{CallOptions, Error, ErrorKind, RerankModel, RerankRequest, RerankResponse};
define_call_wrapper!(
    /// A single-use rerank call bound to one live model handle.
    RerankCall,
    RerankModel,
    RerankRequest
);

impl<'a, M> RerankCall<'a, M>
where
    M: RerankModel + ?Sized,
{
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
