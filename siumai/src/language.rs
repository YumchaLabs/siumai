//! Provider-neutral language generation and streaming facade.

use crate::call::CallState;
use crate::{
    CallOptions, Error, ErrorKind, LanguageCallError, LanguageInput, LanguageModel,
    LanguageRequest, LanguageResponse, LanguageStream,
};
use siumai_core::ProviderOptionError;

/// A single-use language call bound to one live model handle.
pub struct LanguageCall<'a, M: LanguageModel + ?Sized> {
    state: CallState<'a, M, LanguageRequest>,
}

impl<'a, M> LanguageCall<'a, M>
where
    M: LanguageModel + ?Sized,
{
    fn new(model: &'a M, request: LanguageRequest) -> Self {
        Self {
            state: CallState::new(model, request),
        }
    }

    /// Return the complete portable request owned by this call.
    pub fn request(&self) -> &LanguageRequest {
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
            .request()
            .validate()
            .map_err(|source| {
                Error::new(ErrorKind::InvalidInput, "language request is invalid")
                    .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        self.state
            .validate_provider_options(&mut options)
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "provider options do not match the selected language model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Generate one complete language response.
    pub async fn generate(self) -> Result<LanguageResponse, LanguageCallError> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.generate(request, options).await
    }

    /// Establish a complete language stream with the existing lifecycle.
    pub async fn stream(self) -> Result<LanguageStream, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.stream(request, options).await
    }
}

/// Bind one normalized language input to the selected live model.
///
/// Strings, messages, message lists, and complete requests all normalize
/// through [`LanguageInput`] without resolving another provider or route.
pub fn call<M, I>(model: &M, input: I) -> LanguageCall<'_, M>
where
    M: LanguageModel + ?Sized,
    I: Into<LanguageInput>,
{
    LanguageCall::new(model, input.into().into_request())
}

/// Generate one complete response with default call options.
pub async fn generate<M, I>(model: &M, input: I) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
    I: Into<LanguageInput>,
{
    call(model, input).generate().await
}

/// Establish a language stream with default call options.
pub async fn stream<M, I>(model: &M, input: I) -> Result<LanguageStream, Error>
where
    M: LanguageModel + ?Sized,
    I: Into<LanguageInput>,
{
    call(model, input).stream().await
}
