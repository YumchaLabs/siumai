//! Provider-neutral transcription facade.

use crate::call::define_call_wrapper;
use crate::{
    CallOptions, Error, ErrorKind, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
};
define_call_wrapper!(
    /// A single-use transcription call bound to one live model handle.
    TranscriptionCall,
    TranscriptionModel,
    TranscriptionRequest
);

impl<'a, M> TranscriptionCall<'a, M>
where
    M: TranscriptionModel + ?Sized,
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
                    "provider options do not match the selected transcription model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Transcribe one complete audio request.
    pub async fn transcribe(self) -> Result<TranscriptionResponse, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.transcribe(request, options).await
    }
}

/// Bind one complete transcription request to the selected live model.
pub fn call<M>(model: &M, request: TranscriptionRequest) -> TranscriptionCall<'_, M>
where
    M: TranscriptionModel + ?Sized,
{
    TranscriptionCall::new(model, request)
}

/// Transcribe one complete audio request with default call options.
pub async fn transcribe<M>(
    model: &M,
    request: TranscriptionRequest,
) -> Result<TranscriptionResponse, Error>
where
    M: TranscriptionModel + ?Sized,
{
    call(model, request).transcribe().await
}
