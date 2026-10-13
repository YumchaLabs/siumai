//! Provider-neutral language generation and streaming facade.

use crate::call::define_call_wrapper;
use crate::{
    CallOptions, Error, ErrorKind, LanguageCallError, LanguageInput, LanguageModel,
    LanguageRequest, LanguageResponse, LanguageStream,
};
define_call_wrapper!(
    /// A single-use language call bound to one live model handle.
    LanguageCall,
    LanguageModel,
    LanguageRequest
);

impl<'a, M> LanguageCall<'a, M>
where
    M: LanguageModel + ?Sized,
{
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
