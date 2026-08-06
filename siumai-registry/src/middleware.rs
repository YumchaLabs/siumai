use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ImageLimits, ImageModel, ImageRequest, ImageResponse, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelLookupError, RerankLimits,
    RerankModel, RerankRequest, RerankResponse, RouteId, SpeechLimits, SpeechModel, SpeechRequest,
    SpeechResponse, TranscriptionLimits, TranscriptionModel, TranscriptionRequest,
    TranscriptionResponse,
};

use crate::RegistryResolveError;

/// Route identity available while one model handle is resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryModelContext {
    requested_route: RouteId,
    route: RouteId,
}

impl RegistryModelContext {
    pub(crate) fn new(requested_route: RouteId, route: RouteId) -> Self {
        Self {
            requested_route,
            route,
        }
    }

    /// Route present in the caller's `route:model` reference.
    pub fn requested_route(&self) -> &RouteId {
        &self.requested_route
    }

    /// Canonical configured route after alias resolution.
    pub fn route(&self) -> &RouteId {
        &self.route
    }
}

impl fmt::Display for RegistryModelContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.requested_route == self.route {
            write!(formatter, "route `{}`", self.route)
        } else {
            write!(
                formatter,
                "requested route `{}` (canonical route `{}`)",
                self.requested_route, self.route
            )
        }
    }
}

/// Optional model-handle middleware applied exactly once per resolution.
///
/// Middleware may wrap behavior but must preserve the complete model
/// descriptor. Registry validates that invariant after every layer.
pub trait RegistryMiddleware: std::fmt::Debug + Send + Sync + 'static {
    fn wrap_language(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn LanguageModel>,
    ) -> Arc<dyn LanguageModel> {
        model
    }

    fn wrap_embedding(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn EmbeddingModel>,
    ) -> Arc<dyn EmbeddingModel> {
        model
    }

    fn wrap_rerank(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn RerankModel>,
    ) -> Arc<dyn RerankModel> {
        model
    }

    fn wrap_image(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn ImageModel>,
    ) -> Arc<dyn ImageModel> {
        model
    }

    fn wrap_speech(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn SpeechModel>,
    ) -> Arc<dyn SpeechModel> {
        model
    }

    fn wrap_transcription(
        &self,
        _context: &RegistryModelContext,
        model: Arc<dyn TranscriptionModel>,
    ) -> Arc<dyn TranscriptionModel> {
        model
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MiddlewareStack {
    layers: Vec<Arc<dyn RegistryMiddleware>>,
}

impl MiddlewareStack {
    pub(crate) fn push(&mut self, middleware: Arc<dyn RegistryMiddleware>) {
        self.layers.push(middleware);
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &Arc<dyn RegistryMiddleware>> {
        self.layers.iter()
    }

    fn apply<T: Model + ?Sized>(
        &self,
        mut model: Arc<T>,
        mut wrap: impl FnMut(&dyn RegistryMiddleware, Arc<T>) -> Arc<T>,
    ) -> Result<Arc<T>, ModelLookupError> {
        let expected = model.descriptor().clone();
        for middleware in &self.layers {
            model = wrap(middleware.as_ref(), model);
            if model.descriptor() != &expected {
                return Err(ModelLookupError::IdentityMismatch {
                    expected: Box::new(expected),
                    actual: Box::new(model.descriptor().clone()),
                });
            }
        }
        Ok(model)
    }

    pub(crate) fn language(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn LanguageModel>,
    ) -> Result<Arc<dyn LanguageModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_language(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteLanguageModel::new(
            model,
            context.route.clone(),
        )))
    }

    pub(crate) fn embedding(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn EmbeddingModel>,
    ) -> Result<Arc<dyn EmbeddingModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_embedding(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteEmbeddingModel::new(
            model,
            context.route.clone(),
        )))
    }

    pub(crate) fn rerank(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn RerankModel>,
    ) -> Result<Arc<dyn RerankModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_rerank(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteRerankModel::new(
            model,
            context.route.clone(),
        )))
    }

    pub(crate) fn image(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn ImageModel>,
    ) -> Result<Arc<dyn ImageModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_image(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteImageModel::new(model, context.route.clone())))
    }

    pub(crate) fn speech(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn SpeechModel>,
    ) -> Result<Arc<dyn SpeechModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_speech(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteSpeechModel::new(
            model,
            context.route.clone(),
        )))
    }

    pub(crate) fn transcription(
        &self,
        context: &RegistryModelContext,
        model: Arc<dyn TranscriptionModel>,
    ) -> Result<Arc<dyn TranscriptionModel>, RegistryResolveError> {
        let model = self
            .apply(model, |middleware, model| {
                middleware.wrap_transcription(context, model)
            })
            .map_err(|source| RegistryResolveError::Model {
                context: context.clone(),
                source: Box::new(source),
            })?;
        Ok(Arc::new(RouteTranscriptionModel::new(
            model,
            context.route.clone(),
        )))
    }
}

pub(crate) fn contextualize_lookup(error: ModelLookupError, route: &RouteId) -> ModelLookupError {
    match error {
        ModelLookupError::Construction { source } => ModelLookupError::Construction {
            source: source.with_route(route.clone()),
        },
        error => error,
    }
}

struct RouteLanguageModel {
    inner: Arc<dyn LanguageModel>,
    route: RouteId,
}

impl RouteLanguageModel {
    fn new(inner: Arc<dyn LanguageModel>, route: RouteId) -> Self {
        Self { inner, route }
    }
}

impl Model for RouteLanguageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }

    fn route_id(&self) -> Option<&RouteId> {
        Some(&self.route)
    }
}

#[async_trait]
impl LanguageModel for RouteLanguageModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        self.inner
            .generate(request, options)
            .await
            .map_err(|error| error.with_route(self.route.clone()))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.inner
            .stream(request, options)
            .await
            .map(|stream| stream.with_route_context(self.route.clone()))
            .map_err(|error| error.with_route(self.route.clone()))
    }
}

macro_rules! route_model_wrapper {
    ($wrapper:ident, $trait:ident, $limits:ident, $request:ident, $response:ident, $method:ident) => {
        struct $wrapper {
            inner: Arc<dyn $trait>,
            route: RouteId,
        }

        impl $wrapper {
            fn new(inner: Arc<dyn $trait>, route: RouteId) -> Self {
                Self { inner, route }
            }
        }

        impl Model for $wrapper {
            fn descriptor(&self) -> &ModelDescriptor {
                self.inner.descriptor()
            }

            fn route_id(&self) -> Option<&RouteId> {
                Some(&self.route)
            }
        }

        #[async_trait]
        impl $trait for $wrapper {
            fn limits(&self) -> $limits {
                self.inner.limits()
            }

            async fn $method(
                &self,
                request: $request,
                options: CallOptions,
            ) -> Result<$response, Error> {
                self.inner
                    .$method(request, options)
                    .await
                    .map_err(|error| error.with_route(self.route.clone()))
            }
        }
    };
}

route_model_wrapper!(
    RouteEmbeddingModel,
    EmbeddingModel,
    EmbeddingLimits,
    EmbeddingRequest,
    EmbeddingResponse,
    embed
);
route_model_wrapper!(
    RouteRerankModel,
    RerankModel,
    RerankLimits,
    RerankRequest,
    RerankResponse,
    rerank
);
route_model_wrapper!(
    RouteImageModel,
    ImageModel,
    ImageLimits,
    ImageRequest,
    ImageResponse,
    generate_image
);
route_model_wrapper!(
    RouteSpeechModel,
    SpeechModel,
    SpeechLimits,
    SpeechRequest,
    SpeechResponse,
    synthesize
);
route_model_wrapper!(
    RouteTranscriptionModel,
    TranscriptionModel,
    TranscriptionLimits,
    TranscriptionRequest,
    TranscriptionResponse,
    transcribe
);
