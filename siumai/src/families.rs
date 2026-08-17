//! Direct helpers for the six stable model families.

use crate::{CallOptions, Error};

fn resolve_options(options: CallOptions) -> Result<CallOptions, Error> {
    options.resolve_deadline().map_err(Error::from)
}

pub mod language {
    use crate::{
        CallOptions, Error, LanguageCallError, LanguageModel, LanguageRequest, LanguageResponse,
        LanguageStream,
    };

    pub async fn generate<M>(
        model: &M,
        request: LanguageRequest,
    ) -> Result<LanguageResponse, LanguageCallError>
    where
        M: LanguageModel + ?Sized,
    {
        generate_with_options(model, request, CallOptions::default()).await
    }

    pub async fn generate_with_options<M>(
        model: &M,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError>
    where
        M: LanguageModel + ?Sized,
    {
        model
            .generate(request, super::resolve_options(options)?)
            .await
    }

    pub async fn stream<M>(model: &M, request: LanguageRequest) -> Result<LanguageStream, Error>
    where
        M: LanguageModel + ?Sized,
    {
        stream_with_options(model, request, CallOptions::default()).await
    }

    pub async fn stream_with_options<M>(
        model: &M,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error>
    where
        M: LanguageModel + ?Sized,
    {
        model
            .stream(request, super::resolve_options(options)?)
            .await
    }
}

pub mod embedding {
    use crate::{CallOptions, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, Error};

    pub async fn embed<M>(model: &M, request: EmbeddingRequest) -> Result<EmbeddingResponse, Error>
    where
        M: EmbeddingModel + ?Sized,
    {
        embed_with_options(model, request, CallOptions::default()).await
    }

    pub async fn embed_with_options<M>(
        model: &M,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error>
    where
        M: EmbeddingModel + ?Sized,
    {
        model.embed(request, super::resolve_options(options)?).await
    }
}

pub mod rerank {
    use crate::{CallOptions, Error, RerankModel, RerankRequest, RerankResponse};

    pub async fn rerank<M>(model: &M, request: RerankRequest) -> Result<RerankResponse, Error>
    where
        M: RerankModel + ?Sized,
    {
        rerank_with_options(model, request, CallOptions::default()).await
    }

    pub async fn rerank_with_options<M>(
        model: &M,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error>
    where
        M: RerankModel + ?Sized,
    {
        model
            .rerank(request, super::resolve_options(options)?)
            .await
    }
}

pub mod image {
    use crate::{CallOptions, Error, ImageModel, ImageRequest, ImageResponse};

    pub async fn generate<M>(model: &M, request: ImageRequest) -> Result<ImageResponse, Error>
    where
        M: ImageModel + ?Sized,
    {
        generate_with_options(model, request, CallOptions::default()).await
    }

    pub async fn generate_with_options<M>(
        model: &M,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error>
    where
        M: ImageModel + ?Sized,
    {
        model
            .generate_image(request, super::resolve_options(options)?)
            .await
    }
}

pub mod speech {
    use crate::{CallOptions, Error, SpeechModel, SpeechRequest, SpeechResponse};

    pub async fn synthesize<M>(model: &M, request: SpeechRequest) -> Result<SpeechResponse, Error>
    where
        M: SpeechModel + ?Sized,
    {
        synthesize_with_options(model, request, CallOptions::default()).await
    }

    pub async fn synthesize_with_options<M>(
        model: &M,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error>
    where
        M: SpeechModel + ?Sized,
    {
        model
            .synthesize(request, super::resolve_options(options)?)
            .await
    }
}

pub mod transcription {
    use crate::{
        CallOptions, Error, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
    };

    pub async fn transcribe<M>(
        model: &M,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, Error>
    where
        M: TranscriptionModel + ?Sized,
    {
        transcribe_with_options(model, request, CallOptions::default()).await
    }

    pub async fn transcribe_with_options<M>(
        model: &M,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error>
    where
        M: TranscriptionModel + ?Sized,
    {
        model
            .transcribe(request, super::resolve_options(options)?)
            .await
    }
}
