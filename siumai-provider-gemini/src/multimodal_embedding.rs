use std::fmt;
use std::sync::Arc;

use http::Method;
use http::header::{ACCEPT, HeaderValue};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionError, ProviderScope,
};
use siumai_protocol_gemini::multimodal_embedding::{
    GeminiMultimodalEmbeddingRequest, GeminiMultimodalEmbeddingResponse,
    decode_multimodal_embedding_response, encode_multimodal_embedding_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use crate::http::response_error;
use crate::provider::ProviderRuntime;

/// Provider API mode used by Gemini v1beta multimodal embedding.
pub const GEMINI_MULTIMODAL_EMBEDDING_API_MODE_ID: &str = "embed-content-v1beta-multimodal";

/// Lightweight provider-native Gemini multimodal embedding handle.
#[derive(Clone)]
pub struct GeminiMultimodalEmbeddingModel {
    runtime: Arc<ProviderRuntime>,
    descriptor: ModelDescriptor,
}

impl GeminiMultimodalEmbeddingModel {
    pub(crate) fn new(
        runtime: Arc<ProviderRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
    ) -> Self {
        let descriptor = ModelDescriptor::from_scope(
            scope,
            model,
            ModelFamily::Embedding,
            runtime.instance_id.clone(),
        );
        Self {
            runtime,
            descriptor,
        }
    }

    /// Embed one ordered mixture of text, image, audio, video, or PDF parts.
    pub async fn embed(
        &self,
        request: GeminiMultimodalEmbeddingRequest,
        options: CallOptions,
    ) -> Result<GeminiMultimodalEmbeddingResponse, Error> {
        request
            .validate()
            .map_err(|error| self.contextualize(error))?;
        let selection = options
            .provider_options_for(self)
            .map_err(option_error)
            .map_err(|error| self.contextualize(error))?;
        if selection.typed().len() != 0 || selection.raw_override().is_some() {
            return Err(
                self.contextualize(option_error(ProviderOptionError::Rejected {
                    path: "$".to_string(),
                    reason: "Gemini multimodal embedding does not accept provider-option patches"
                        .to_string(),
                })),
            );
        }
        let encoded = encode_multimodal_embedding_request(&request, self.model_id())
            .map_err(|error| self.contextualize(error))?;
        let (target, body) = encoded.into_parts();
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_build_error)
            .map_err(|error| self.contextualize(error))?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(target)
                .map_err(request_build_error)
                .map_err(|error| self.contextualize(error))?,
        )
        .with_headers(headers)
        .with_body(
            RequestBody::json(&body)
                .map_err(request_build_error)
                .map_err(|error| self.contextualize(error))?,
        )
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(request_build_error)
        .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the v1beta multimodal embedding request",
            )));
        }
        decode_multimodal_embedding_response(response.body())
            .map_err(|error| self.contextualize(error))
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::Embed),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for GeminiMultimodalEmbeddingModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiMultimodalEmbeddingModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for GeminiMultimodalEmbeddingModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for Gemini multimodal embedding",
    )
    .with_source(source)
}

fn request_build_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini multimodal embedding request violates the transport contract",
    )
    .with_source(source)
}
