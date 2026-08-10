use siumai_core::{
    CallOptions, Error, ErrorKind, LanguageCallError, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream,
};

use crate::{Runtime, StepOptions};

/// Perform exactly one non-streaming model call.
///
/// Tool calls in the response remain data. This function never looks up or
/// executes a local tool binding.
pub async fn generate<M>(
    model: &M,
    request: LanguageRequest,
    options: CallOptions,
) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
{
    Runtime::default()
        .generate(model, request, StepOptions::default(), options)
        .await
}

/// Establish exactly one provider language stream.
///
/// Tool calls remain stream events. This function never starts a tool loop.
pub async fn stream<M>(
    model: &M,
    request: LanguageRequest,
    options: CallOptions,
) -> Result<LanguageStream, Error>
where
    M: LanguageModel + ?Sized,
{
    Runtime::default()
        .stream(model, request, StepOptions::default(), options)
        .await
}

pub(crate) fn validate_request(request: &LanguageRequest) -> Result<(), Error> {
    request.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "invalid language request").with_source(source)
    })
}
