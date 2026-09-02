use siumai::{
    CallOptions, LanguageCallError, LanguageModel, LanguageRequest, LanguageResponse,
    ProviderOptionError,
};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
enum ApplicationCallError {
    Setup(ProviderOptionError),
    Call(LanguageCallError),
}

impl fmt::Display for ApplicationCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup(_) => formatter.write_str("invalid provider options"),
            Self::Call(_) => formatter.write_str("language call failed"),
        }
    }
}

impl StdError for ApplicationCallError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Setup(error) => Some(error),
            Self::Call(error) => Some(error),
        }
    }
}

impl From<ProviderOptionError> for ApplicationCallError {
    fn from(error: ProviderOptionError) -> Self {
        Self::Setup(error)
    }
}

impl From<LanguageCallError> for ApplicationCallError {
    fn from(error: LanguageCallError) -> Self {
        Self::Call(error)
    }
}

#[test]
fn migrated_language_call_can_preserve_both_error_phases() {
    async fn migrated<M>(
        model: &M,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, ApplicationCallError>
    where
        M: LanguageModel + ?Sized,
    {
        Ok(siumai::language::call(model, request)
            .with_options(options)?
            .generate()
            .await?)
    }

    let _ = migrated::<dyn LanguageModel>;
}
