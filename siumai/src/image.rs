//! Provider-neutral image generation facade.

use crate::call::define_call_wrapper;
use crate::{CallOptions, Error, ErrorKind, ImageModel, ImageRequest, ImageResponse};
define_call_wrapper!(
    /// A single-use image call bound to one live model handle.
    ImageCall,
    ImageModel,
    ImageRequest
);

impl<'a, M> ImageCall<'a, M>
where
    M: ImageModel + ?Sized,
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
                    "provider options do not match the selected image model",
                )
                .with_source(source)
            })
            .map_err(|error| self.state.contextualize_error(error))?;
        Ok(options)
    }

    /// Generate one set of final image artifacts.
    pub async fn generate(self) -> Result<ImageResponse, Error> {
        let options = self.preflight()?;
        let (model, request) = self.state.into_target();
        model.generate_image(request, options).await
    }
}

/// Bind one complete image request to the selected live model.
pub fn call<M>(model: &M, request: ImageRequest) -> ImageCall<'_, M>
where
    M: ImageModel + ?Sized,
{
    ImageCall::new(model, request)
}

/// Generate one set of final image artifacts with default call options.
pub async fn generate<M>(model: &M, request: ImageRequest) -> Result<ImageResponse, Error>
where
    M: ImageModel + ?Sized,
{
    call(model, request).generate().await
}
