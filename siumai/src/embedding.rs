//! Provider-neutral embedding facade.

use crate::call::CallState;
use crate::{CallOptions, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error, ErrorKind};
use siumai_core::ProviderOptionError;

/// A single-use embedding call bound to one live model handle.
pub struct EmbeddingCall<'a, M: EmbeddingModel + ?Sized> {
    state: CallState<'a, M, EmbeddingRequest>,
}

impl<'a, M> EmbeddingCall<'a, M>
where
    M: EmbeddingModel + ?Sized,
{
    fn new(model: &'a M, request: EmbeddingRequest) -> Self {
        Self {
            state: CallState::new(model, request),
        }
    }

    /// Return the complete portable request owned by this call.
    pub fn request(&self) -> &EmbeddingRequest {
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
