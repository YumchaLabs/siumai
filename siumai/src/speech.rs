//! Provider-neutral speech synthesis facade.

use crate::call::define_call_wrapper;
use crate::{CallOptions, Error, ErrorKind, SpeechModel, SpeechRequest, SpeechResponse};
define_call_wrapper!(
    /// A single-use speech call bound to one live model handle.
    SpeechCall,
    SpeechModel,
    SpeechRequest
);

impl<'a, M> SpeechCall<'a, M>
where
    M: SpeechModel + ?Sized,
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
                    "provider options do not match the selected speech model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Synthesize one complete buffered speech response.
    pub async fn synthesize(self) -> Result<SpeechResponse, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.synthesize(request, options).await
    }
}

/// Bind one complete speech request to the selected live model.
pub fn call<M>(model: &M, request: SpeechRequest) -> SpeechCall<'_, M>
where
    M: SpeechModel + ?Sized,
{
    SpeechCall::new(model, request)
}

/// Synthesize one complete buffered response with default call options.
pub async fn synthesize<M>(model: &M, request: SpeechRequest) -> Result<SpeechResponse, Error>
where
    M: SpeechModel + ?Sized,
{
    call(model, request).synthesize().await
}
