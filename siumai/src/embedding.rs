//! Provider-neutral embedding facade.

use crate::call::define_call_wrapper;
use crate::{CallOptions, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error, ErrorKind};
define_call_wrapper!(
    /// A single-use embedding call bound to one live model handle.
    EmbeddingCall,
    EmbeddingModel,
    EmbeddingRequest
);

impl<'a, M> EmbeddingCall<'a, M>
where
    M: EmbeddingModel + ?Sized,
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
                    "provider options do not match the selected embedding model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Execute one embedding request and return the complete response.
    pub async fn embed(self) -> Result<EmbeddingResponse, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.embed(request, options).await
    }
}

/// Bind one complete embedding request to the selected live model.
pub fn call<M>(model: &M, request: EmbeddingRequest) -> EmbeddingCall<'_, M>
where
    M: EmbeddingModel + ?Sized,
{
    EmbeddingCall::new(model, request)
}

/// Execute one embedding request with default call options.
pub async fn embed<M>(model: &M, request: EmbeddingRequest) -> Result<EmbeddingResponse, Error>
where
    M: EmbeddingModel + ?Sized,
{
    call(model, request).embed().await
}
