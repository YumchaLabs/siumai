//! Provider-neutral transcription facade.

use crate::call::CallState;
use crate::{
    CallOptions, Error, ErrorKind, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
};
use siumai_core::ProviderOptionError;

/// A single-use transcription call bound to one live model handle.
pub struct TranscriptionCall<'a, M: TranscriptionModel + ?Sized> {
    state: CallState<'a, M, TranscriptionRequest>,
}

impl<'a, M> TranscriptionCall<'a, M>
where
    M: TranscriptionModel + ?Sized,
{
    fn new(model: &'a M, request: TranscriptionRequest) -> Self {
        Self {
            state: CallState::new(model, request),
        }
    }

    /// Return the complete portable request owned by this call.
    pub fn request(&self) -> &TranscriptionRequest {
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
