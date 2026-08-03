//! The six stable, object-safe model family contracts.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorKind};
use crate::language::{LanguageRequest, LanguageResponse, MediaData, Warning};
use crate::options::CallOptions;
use crate::provider::{ModelId, ProviderId};
use crate::stream::LanguageStream;
use crate::usage::Usage;

/// Stable callable model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    Language,
    Embedding,
    Rerank,
    Image,
    Speech,
    Transcription,
}

/// Immutable identity captured by a lightweight model handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    provider: ProviderId,
    model: ModelId,
    family: ModelFamily,
    platform: Option<String>,
    protocol: Option<String>,
    api_mode: Option<String>,
}

impl ModelDescriptor {
    pub fn new(provider: ProviderId, model: ModelId, family: ModelFamily) -> Self {
        Self {
            provider,
            model,
            family,
            platform: None,
            protocol: None,
            api_mode: None,
        }
    }

    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = Some(platform.into());
        self
    }

    pub fn with_protocol(mut self, protocol: impl Into<String>) -> Self {
        self.protocol = Some(protocol.into());
        self
    }

    pub fn with_api_mode(mut self, api_mode: impl Into<String>) -> Self {
        self.api_mode = Some(api_mode.into());
        self
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn platform(&self) -> Option<&str> {
        self.platform.as_deref()
    }

    pub fn protocol(&self) -> Option<&str> {
        self.protocol.as_deref()
    }

    pub fn api_mode(&self) -> Option<&str> {
        self.api_mode.as_deref()
    }
}

/// Shared metadata contract for every stable family.
pub trait Model: Send + Sync {
    fn descriptor(&self) -> &ModelDescriptor;

    fn provider_id(&self) -> &ProviderId {
        self.descriptor().provider()
    }

    fn model_id(&self) -> &ModelId {
        self.descriptor().model()
    }

    fn family(&self) -> ModelFamily {
        self.descriptor().family()
    }
}

/// A canonical language generation and streaming model.
#[async_trait]
pub trait LanguageModel: Model {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error>;

    /// Return an established stream. Validation, encoding, and handshake
    /// failures are returned as this method's outer error.
    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingRequest {
    inputs: Vec<EmbeddingInput>,
    dimensions: Option<usize>,
}

impl EmbeddingRequest {
    pub fn new(inputs: Vec<EmbeddingInput>) -> Result<Self, Error> {
        if inputs.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "embedding request requires at least one input",
            ));
        }
        Ok(Self {
            inputs,
            dimensions: None,
        })
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Result<Self, Error> {
        if dimensions == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "embedding dimensions must be greater than zero",
            ));
        }
        self.dimensions = Some(dimensions);
        Ok(self)
    }

    pub fn inputs(&self) -> &[EmbeddingInput] {
        &self.inputs
    }

    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    pub fn into_parts(self) -> (Vec<EmbeddingInput>, Option<usize>) {
        (self.inputs, self.dimensions)
    }
}

#[derive(Deserialize)]
struct EmbeddingRequestWire {
    inputs: Vec<EmbeddingInput>,
    dimensions: Option<usize>,
}

impl<'de> Deserialize<'de> for EmbeddingRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EmbeddingRequestWire::deserialize(deserializer)?;
        let request = Self::new(wire.inputs).map_err(serde::de::Error::custom)?;
        match wire.dimensions {
            Some(dimensions) => request
                .with_dimensions(dimensions)
                .map_err(serde::de::Error::custom),
            None => Ok(request),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[async_trait]
pub trait EmbeddingModel: Model {
    /// Execute one provider request containing one or more inputs.
    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankCandidate {
    pub id: Option<String>,
    pub text: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub candidates: Vec<RerankCandidate>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f64,
    pub candidate: Option<RerankCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[async_trait]
pub trait RerankModel: Model {
    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRequest {
    pub prompt: String,
    pub count: u32,
    pub size: Option<(u32, u32)>,
    pub format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub media_type: String,
    pub data: MediaData,
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    pub images: Vec<ImageArtifact>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[async_trait]
pub trait ImageModel: Model {
    /// Generate final image artifacts without hidden polling or batching.
    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechRequest {
    pub text: String,
    pub voice: Option<String>,
    pub format: Option<String>,
    pub speed: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechResponse {
    pub media_type: String,
    pub audio: Vec<u8>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[async_trait]
pub trait SpeechModel: Model {
    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    pub audio: Vec<u8>,
    pub media_type: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<TranscriptSegment>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[async_trait]
pub trait TranscriptionModel: Model {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error>;
}
