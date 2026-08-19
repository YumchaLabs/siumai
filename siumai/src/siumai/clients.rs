use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use siumai_core::{
    CallOptions, EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error,
    ImageLimits, ImageModel, ImageRequest, ImageResponse, LanguageCallError, LanguageInput,
    LanguageModel, LanguageRequest, LanguageResponse, LanguageStream, Model, ModelDescriptor,
    Provider, RerankLimits, RerankModel, RerankRequest, RerankResponse, RouteId, SpeechLimits,
    SpeechModel, SpeechRequest, SpeechResponse, TranscriptionLimits, TranscriptionModel,
    TranscriptionRequest, TranscriptionResponse,
};

struct ModelBinding<P, M> {
    provider: Arc<P>,
    model: Arc<M>,
}

impl<P, M> ModelBinding<P, M> {
    fn new(provider: Arc<P>, model: M) -> Self {
        Self {
            provider,
            model: Arc::new(model),
        }
    }

    fn provider(&self) -> &P {
        self.provider.as_ref()
    }

    fn model(&self) -> &M {
        self.model.as_ref()
    }
}

impl<P, M> Clone for ModelBinding<P, M> {
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            model: self.model.clone(),
        }
    }
}

macro_rules! define_family_client {
    ($(#[$meta:meta])* $name:ident, $model_trait:ident) => {
        $(#[$meta])*
        pub struct $name<P, M> {
            binding: ModelBinding<P, M>,
        }

        impl<P, M> $name<P, M> {
            pub(super) fn new(provider: Arc<P>, model: M) -> Self {
                Self {
                    binding: ModelBinding::new(provider, model),
                }
            }

            /// Return the configured provider shared with the hub that created this client.
            pub fn provider(&self) -> &P {
                self.binding.provider()
            }

            /// Return the exact provider-owned model bound to this client.
            pub fn model(&self) -> &M {
                self.binding.model()
            }
        }

        impl<P, M> Clone for $name<P, M> {
            fn clone(&self) -> Self {
                Self {
                    binding: self.binding.clone(),
                }
            }
        }

        impl<P, M> fmt::Debug for $name<P, M>
        where
            P: Provider,
            M: Model,
        {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("provider_id", self.provider().provider_id())
                    .field("descriptor", self.model().descriptor())
                    .field("route_id", &self.model().route_id())
                    .finish()
            }
        }

        impl<P, M> Model for $name<P, M>
        where
            P: Provider,
            M: $model_trait,
        {
            fn descriptor(&self) -> &ModelDescriptor {
                Model::descriptor(self.model())
            }

            fn route_id(&self) -> Option<&RouteId> {
                Model::route_id(self.model())
            }
        }
    };
}

define_family_client!(
    /// A language model bound to one concrete configured provider.
    ///
    /// Clients can only be obtained from a compatible [`crate::Siumai`] hub;
    /// arbitrary provider/model pairing is intentionally unavailable.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use siumai::LanguageClient;
    ///
    /// struct UnrelatedProvider;
    /// struct UnrelatedModel;
    ///
    /// let _ = LanguageClient::new(Arc::new(UnrelatedProvider), UnrelatedModel);
    /// ```
    LanguageClient,
    LanguageModel
);
define_family_client!(
    /// An embedding model bound to one concrete configured provider.
    EmbeddingClient,
    EmbeddingModel
);
define_family_client!(
    /// A rerank model bound to one concrete configured provider.
    RerankClient,
    RerankModel
);
define_family_client!(
    /// An image model bound to one concrete configured provider.
    ImageClient,
    ImageModel
);
define_family_client!(
    /// A speech model bound to one concrete configured provider.
    SpeechClient,
    SpeechModel
);
define_family_client!(
    /// A transcription model bound to one concrete configured provider.
    TranscriptionClient,
    TranscriptionModel
);

impl<P, M> LanguageClient<P, M>
where
    P: Provider,
    M: LanguageModel,
{
    /// Bind one portable language input to this exact model target.
    pub fn call<I>(&self, input: I) -> crate::language::LanguageCall<'_, Self>
    where
        I: Into<LanguageInput>,
    {
        crate::language::call(self, input)
    }

    /// Generate one complete language response with default call options.
    pub async fn generate<I>(&self, input: I) -> Result<LanguageResponse, LanguageCallError>
    where
        I: Into<LanguageInput>,
    {
        crate::language::generate(self.model(), input).await
    }

    /// Establish a language stream with the existing lifecycle contract.
    pub async fn stream<I>(&self, input: I) -> Result<LanguageStream, Error>
    where
        I: Into<LanguageInput>,
    {
        crate::language::stream(self.model(), input).await
    }
}

impl<P, M> EmbeddingClient<P, M>
where
    P: Provider,
    M: EmbeddingModel,
{
    /// Bind one complete embedding request to this exact model target.
    pub fn call(&self, request: EmbeddingRequest) -> crate::embedding::EmbeddingCall<'_, Self> {
        crate::embedding::call(self, request)
    }

    /// Execute one embedding request with default call options.
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        crate::embedding::embed(self.model(), request).await
    }
}

impl<P, M> RerankClient<P, M>
where
    P: Provider,
    M: RerankModel,
{
    /// Bind one complete rerank request to this exact model target.
    pub fn call(&self, request: RerankRequest) -> crate::rerank::RerankCall<'_, Self> {
        crate::rerank::call(self, request)
    }

    /// Execute one rerank request with default call options.
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
        crate::rerank::rerank(self.model(), request).await
    }
}

impl<P, M> ImageClient<P, M>
where
    P: Provider,
    M: ImageModel,
{
    /// Bind one complete image request to this exact model target.
    pub fn call(&self, request: ImageRequest) -> crate::image::ImageCall<'_, Self> {
        crate::image::call(self, request)
    }

    /// Generate one complete set of image artifacts with default call options.
    pub async fn generate(&self, request: ImageRequest) -> Result<ImageResponse, Error> {
        crate::image::generate(self.model(), request).await
    }
}

impl<P, M> SpeechClient<P, M>
where
    P: Provider,
    M: SpeechModel,
{
    /// Bind one complete speech request to this exact model target.
    pub fn call(&self, request: SpeechRequest) -> crate::speech::SpeechCall<'_, Self> {
        crate::speech::call(self, request)
    }

    /// Synthesize one complete buffered speech response with default call options.
    pub async fn synthesize(&self, request: SpeechRequest) -> Result<SpeechResponse, Error> {
        crate::speech::synthesize(self.model(), request).await
    }
}

impl<P, M> TranscriptionClient<P, M>
where
    P: Provider,
    M: TranscriptionModel,
{
    /// Bind one complete transcription request to this exact model target.
    pub fn call(
        &self,
        request: TranscriptionRequest,
    ) -> crate::transcription::TranscriptionCall<'_, Self> {
        crate::transcription::call(self, request)
    }

    /// Transcribe one complete audio request with default call options.
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, Error> {
        crate::transcription::transcribe(self.model(), request).await
    }
}

#[async_trait]
impl<P, M> LanguageModel for LanguageClient<P, M>
where
    P: Provider,
    M: LanguageModel,
{
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        LanguageModel::generate(self.model(), request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        LanguageModel::stream(self.model(), request, options).await
    }
}

#[async_trait]
impl<P, M> EmbeddingModel for EmbeddingClient<P, M>
where
    P: Provider,
    M: EmbeddingModel,
{
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingModel::limits(self.model())
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        EmbeddingModel::embed(self.model(), request, options).await
    }
}

#[async_trait]
impl<P, M> RerankModel for RerankClient<P, M>
where
    P: Provider,
    M: RerankModel,
{
    fn limits(&self) -> RerankLimits {
        RerankModel::limits(self.model())
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        RerankModel::rerank(self.model(), request, options).await
    }
}

#[async_trait]
impl<P, M> ImageModel for ImageClient<P, M>
where
    P: Provider,
    M: ImageModel,
{
    fn limits(&self) -> ImageLimits {
        ImageModel::limits(self.model())
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        ImageModel::generate_image(self.model(), request, options).await
    }
}

#[async_trait]
impl<P, M> SpeechModel for SpeechClient<P, M>
where
    P: Provider,
    M: SpeechModel,
{
    fn limits(&self) -> SpeechLimits {
        SpeechModel::limits(self.model())
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        SpeechModel::synthesize(self.model(), request, options).await
    }
}

#[async_trait]
impl<P, M> TranscriptionModel for TranscriptionClient<P, M>
where
    P: Provider,
    M: TranscriptionModel,
{
    fn limits(&self) -> TranscriptionLimits {
        TranscriptionModel::limits(self.model())
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        TranscriptionModel::transcribe(self.model(), request, options).await
    }
}
