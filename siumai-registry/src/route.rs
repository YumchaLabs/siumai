use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ImageLimits, ImageModel, ImageRequest, ImageResponse, LanguageCallError, LanguageModel,
    LanguageRequest, LanguageResponse, LanguageStream, Model, ModelDescriptor, ModelLookupError,
    RerankLimits, RerankModel, RerankRequest, RerankResponse, RouteId, SpeechLimits, SpeechModel,
    SpeechRequest, SpeechResponse, TranscriptionLimits, TranscriptionModel, TranscriptionRequest,
    TranscriptionResponse,
};

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

pub(crate) fn contextualize_lookup(error: ModelLookupError, route: &RouteId) -> ModelLookupError {
    match error {
        ModelLookupError::Construction { source } => ModelLookupError::Construction {
            source: source.with_route(route.clone()),
        },
        error => error,
    }
}

fn resolve_options(options: CallOptions, route: &RouteId) -> Result<CallOptions, Error> {
    options
        .resolve_deadline()
        .map_err(Error::from)
        .map_err(|error| error.with_route(route.clone()))
}

pub(crate) fn with_language_route(
    context: &RegistryModelContext,
    model: Arc<dyn LanguageModel>,
) -> Arc<dyn LanguageModel> {
    Arc::new(RouteLanguageModel::new(model, context.route.clone()))
}

pub(crate) fn with_embedding_route(
    context: &RegistryModelContext,
    model: Arc<dyn EmbeddingModel>,
) -> Arc<dyn EmbeddingModel> {
    Arc::new(RouteEmbeddingModel::new(model, context.route.clone()))
}

pub(crate) fn with_rerank_route(
    context: &RegistryModelContext,
    model: Arc<dyn RerankModel>,
) -> Arc<dyn RerankModel> {
    Arc::new(RouteRerankModel::new(model, context.route.clone()))
}

pub(crate) fn with_image_route(
    context: &RegistryModelContext,
    model: Arc<dyn ImageModel>,
) -> Arc<dyn ImageModel> {
    Arc::new(RouteImageModel::new(model, context.route.clone()))
}

pub(crate) fn with_speech_route(
    context: &RegistryModelContext,
    model: Arc<dyn SpeechModel>,
) -> Arc<dyn SpeechModel> {
    Arc::new(RouteSpeechModel::new(model, context.route.clone()))
}

pub(crate) fn with_transcription_route(
    context: &RegistryModelContext,
    model: Arc<dyn TranscriptionModel>,
) -> Arc<dyn TranscriptionModel> {
    Arc::new(RouteTranscriptionModel::new(model, context.route.clone()))
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
    ) -> Result<LanguageResponse, LanguageCallError> {
        let options = resolve_options(options, &self.route)?;
        self.inner
            .generate(
                request,
                options.with_selected_route_context(self.route.clone()),
            )
            .await
            .map_err(|error| error.with_route(self.route.clone()))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let options = resolve_options(options, &self.route)?;
        self.inner
            .stream(
                request,
                options.with_selected_route_context(self.route.clone()),
            )
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
                let options = resolve_options(options, &self.route)?;
                self.inner
                    .$method(
                        request,
                        options.with_selected_route_context(self.route.clone()),
                    )
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
