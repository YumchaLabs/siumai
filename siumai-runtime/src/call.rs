use siumai_core::{Error, ErrorKind, LanguageRequest};

pub(crate) fn validate_request(request: &LanguageRequest) -> Result<(), Error> {
    request.validate().map_err(|source| {
        Error::new(ErrorKind::InvalidInput, "invalid language request").with_source(source)
    })
}
