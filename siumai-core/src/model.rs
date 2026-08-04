//! The six stable, object-safe model family contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorKind, ResourceKind};
use crate::language::{LanguageRequest, LanguageResponse, MediaData, Warning};
use crate::options::CallOptions;
use crate::provider::{ModelId, ProviderId, ProviderScope};
use crate::stream::LanguageStream;
use crate::usage::Usage;

/// Stable callable model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    scope: Arc<ProviderScope>,
    model: ModelId,
    family: ModelFamily,
}

impl ModelDescriptor {
    pub fn new(provider: ProviderId, model: ModelId, family: ModelFamily) -> Self {
        Self::from_scope(Arc::new(ProviderScope::new(provider)), model, family)
    }

    pub fn from_scope(scope: Arc<ProviderScope>, model: ModelId, family: ModelFamily) -> Self {
        Self {
            scope,
            model,
            family,
        }
    }

    pub fn with_platform(mut self, platform: crate::provider::PlatformId) -> Self {
        self.scope = Arc::new(self.scope.as_ref().clone().with_platform(platform));
        self
    }

    pub fn with_protocol(mut self, protocol: crate::provider::ProtocolId) -> Self {
        self.scope = Arc::new(self.scope.as_ref().clone().with_protocol(protocol));
        self
    }

    pub fn with_api_mode(mut self, api_mode: crate::provider::ApiModeId) -> Self {
        self.scope = Arc::new(self.scope.as_ref().clone().with_api_mode(api_mode));
        self
    }

    pub fn provider(&self) -> &ProviderId {
        self.scope.provider_id()
    }

    pub fn scope(&self) -> &Arc<ProviderScope> {
        &self.scope
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn platform(&self) -> Option<&str> {
        self.scope
            .platform()
            .map(crate::provider::PlatformId::as_str)
    }

    pub fn protocol(&self) -> Option<&str> {
        self.scope
            .protocol()
            .map(crate::provider::ProtocolId::as_str)
    }

    pub fn api_mode(&self) -> Option<&str> {
        self.scope
            .api_mode()
            .map(crate::provider::ApiModeId::as_str)
    }
}

/// Shared metadata contract for every stable family.
pub trait Model: Send + Sync {
    fn descriptor(&self) -> &ModelDescriptor;

    /// Canonical Registry route that selected this model, when applicable.
    ///
    /// Direct provider models return `None`. Registry wrappers override this
    /// without changing provider/model identity in [`ModelDescriptor`].
    fn route_id(&self) -> Option<&crate::provider::RouteId> {
        None
    }

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

impl<T> Model for Arc<T>
where
    T: Model + ?Sized,
{
    fn descriptor(&self) -> &ModelDescriptor {
        self.as_ref().descriptor()
    }

    fn route_id(&self) -> Option<&crate::provider::RouteId> {
        self.as_ref().route_id()
    }
}

/// A canonical language generation and streaming model.
#[async_trait]
pub trait LanguageModel: Model {
    /// Generate one terminal response resource.
    ///
    /// A provider-returned failed or cancelled resource remains an `Ok`
    /// [`LanguageResponse`] with the corresponding status so its identity,
    /// content, usage, and native items are not discarded. Validation,
    /// encoding, authentication, transport, and protocol failures that produce
    /// no response resource remain outer [`Error`] values.
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

#[async_trait]
impl<T> LanguageModel for Arc<T>
where
    T: LanguageModel + ?Sized,
{
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        self.as_ref().generate(request, options).await
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.as_ref().stream(request, options).await
    }
}

/// Lightweight identifiers returned with a provider operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub response_id: Option<String>,
    pub request_id: Option<String>,
    pub model: Option<ModelId>,
}

/// Published limits for one embedding model and API mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingLimits {
    pub max_inputs: Option<usize>,
    pub max_input_tokens: Option<u64>,
}

impl EmbeddingLimits {
    pub fn validate(&self, request: &EmbeddingRequest) -> Result<(), Error> {
        if let Some(maximum) = self.max_inputs
            && request.inputs.len() > maximum
        {
            return Err(Error::limit_exceeded(
                ResourceKind::EmbeddingInputs,
                as_u64(request.inputs.len()),
                as_u64(maximum),
            ));
        }
        Ok(())
    }
}

/// One provider request containing one or more text inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingRequest {
    inputs: Vec<String>,
    dimensions: Option<NonZeroU32>,
}

impl EmbeddingRequest {
    pub fn new<I, S>(inputs: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect::<Vec<_>>();
        if inputs.is_empty() {
            return Err(invalid_input(
                "embedding request requires at least one input",
            ));
        }
        if inputs.iter().any(|input| input.trim().is_empty()) {
            return Err(invalid_input("embedding inputs must not be empty"));
        }
        Ok(Self {
            inputs,
            dimensions: None,
        })
    }

    pub fn single(input: impl Into<String>) -> Result<Self, Error> {
        Self::new([input])
    }

    pub fn with_dimensions(mut self, dimensions: u32) -> Result<Self, Error> {
        self.dimensions = Some(
            NonZeroU32::new(dimensions)
                .ok_or_else(|| invalid_input("embedding dimensions must be greater than zero"))?,
        );
        Ok(self)
    }

    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    pub fn dimensions(&self) -> Option<NonZeroU32> {
        self.dimensions
    }

    pub fn into_parts(self) -> (Vec<String>, Option<NonZeroU32>) {
        (self.inputs, self.dimensions)
    }
}

#[derive(Deserialize)]
struct EmbeddingRequestWire {
    inputs: Vec<String>,
    dimensions: Option<u32>,
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
    #[serde(default)]
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl EmbeddingResponse {
    pub fn validate(&self, request: &EmbeddingRequest) -> Result<(), Error> {
        if self.embeddings.len() != request.inputs.len() {
            return Err(Error::protocol_violation(
                "embedding response count does not match request input count",
            ));
        }
        let requested_dimensions = request.dimensions.map(NonZeroU32::get);
        for embedding in &self.embeddings {
            if embedding.is_empty()
                || embedding.iter().any(|value| !value.is_finite())
                || requested_dimensions
                    .is_some_and(|dimensions| embedding.len() != dimensions as usize)
            {
                return Err(Error::protocol_violation(
                    "embedding response contains an invalid vector",
                ));
            }
        }
        Ok(())
    }
}

#[async_trait]
pub trait EmbeddingModel: Model {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits::default()
    }

    /// Execute exactly one provider request containing all request inputs.
    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error>;
}

#[async_trait]
impl<T> EmbeddingModel for Arc<T>
where
    T: EmbeddingModel + ?Sized,
{
    fn limits(&self) -> EmbeddingLimits {
        self.as_ref().limits()
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.as_ref().embed(request, options).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RerankCandidate {
    id: Option<String>,
    text: String,
}

impl RerankCandidate {
    pub fn new(text: impl Into<String>) -> Result<Self, Error> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(invalid_input("rerank candidate text must not be empty"));
        }
        Ok(Self { id: None, text })
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Result<Self, Error> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(invalid_input("rerank candidate ID must not be empty"));
        }
        self.id = Some(id);
        Ok(self)
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Deserialize)]
struct RerankCandidateWire {
    id: Option<String>,
    text: String,
}

impl<'de> Deserialize<'de> for RerankCandidate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RerankCandidateWire::deserialize(deserializer)?;
        let candidate = Self::new(wire.text).map_err(serde::de::Error::custom)?;
        match wire.id {
            Some(id) => candidate.with_id(id).map_err(serde::de::Error::custom),
            None => Ok(candidate),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RerankRequest {
    query: String,
    candidates: Vec<RerankCandidate>,
    top_n: Option<NonZeroUsize>,
}

impl RerankRequest {
    pub fn new(query: impl Into<String>, candidates: Vec<RerankCandidate>) -> Result<Self, Error> {
        let query = query.into();
        if query.trim().is_empty() {
            return Err(invalid_input("rerank query must not be empty"));
        }
        if candidates.is_empty() {
            return Err(invalid_input(
                "rerank request requires at least one candidate",
            ));
        }
        let mut ids = BTreeSet::new();
        for candidate in &candidates {
            if let Some(id) = candidate.id()
                && !ids.insert(id)
            {
                return Err(invalid_input("rerank candidate IDs must be unique"));
            }
        }
        Ok(Self {
            query,
            candidates,
            top_n: None,
        })
    }

    pub fn with_top_n(mut self, top_n: usize) -> Result<Self, Error> {
        let top_n = NonZeroUsize::new(top_n)
            .ok_or_else(|| invalid_input("rerank top_n must be greater than zero"))?;
        if top_n.get() > self.candidates.len() {
            return Err(invalid_input(
                "rerank top_n must not exceed the candidate count",
            ));
        }
        self.top_n = Some(top_n);
        Ok(self)
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn candidates(&self) -> &[RerankCandidate] {
        &self.candidates
    }

    pub fn top_n(&self) -> Option<usize> {
        self.top_n.map(NonZeroUsize::get)
    }

    pub fn into_parts(self) -> (String, Vec<RerankCandidate>, Option<usize>) {
        (
            self.query,
            self.candidates,
            self.top_n.map(NonZeroUsize::get),
        )
    }
}

#[derive(Deserialize)]
struct RerankRequestWire {
    query: String,
    candidates: Vec<RerankCandidate>,
    top_n: Option<usize>,
}

impl<'de> Deserialize<'de> for RerankRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RerankRequestWire::deserialize(deserializer)?;
        let request = Self::new(wire.query, wire.candidates).map_err(serde::de::Error::custom)?;
        match wire.top_n {
            Some(top_n) => request.with_top_n(top_n).map_err(serde::de::Error::custom),
            None => Ok(request),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankLimits {
    pub max_candidates: Option<usize>,
}

impl RerankLimits {
    pub fn validate(&self, request: &RerankRequest) -> Result<(), Error> {
        if let Some(maximum) = self.max_candidates
            && request.candidates.len() > maximum
        {
            return Err(Error::limit_exceeded(
                ResourceKind::RerankCandidates,
                as_u64(request.candidates.len()),
                as_u64(maximum),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    index: usize,
    score: f64,
    candidate_id: Option<String>,
}

impl RerankResult {
    pub fn new(index: usize, score: f64, candidate_id: Option<String>) -> Result<Self, Error> {
        if !score.is_finite() {
            return Err(Error::protocol_violation(
                "rerank response score must be finite",
            ));
        }
        if candidate_id
            .as_ref()
            .is_some_and(|candidate_id| candidate_id.trim().is_empty())
        {
            return Err(Error::protocol_violation(
                "rerank response candidate ID must not be empty",
            ));
        }
        Ok(Self {
            index,
            score,
            candidate_id,
        })
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn candidate_id(&self) -> Option<&str> {
        self.candidate_id.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    #[serde(default)]
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl RerankResponse {
    pub fn validate(&self, request: &RerankRequest) -> Result<(), Error> {
        let expected = request.top_n().unwrap_or(request.candidates.len());
        if self.results.len() < expected {
            return Err(Error::partial_result(
                ResourceKind::RerankCandidates,
                as_u64(expected),
                as_u64(self.results.len()),
            ));
        }
        if self.results.len() > expected {
            return Err(Error::protocol_violation(
                "rerank response contains more results than requested",
            ));
        }
        let mut indices = BTreeSet::new();
        for result in &self.results {
            let Some(candidate) = request.candidates.get(result.index) else {
                return Err(Error::protocol_violation(
                    "rerank response candidate index is out of range",
                ));
            };
            if !indices.insert(result.index) {
                return Err(Error::protocol_violation(
                    "rerank response contains a duplicate candidate index",
                ));
            }
            if !result.score.is_finite() || result.candidate_id() != candidate.id() {
                return Err(Error::protocol_violation(
                    "rerank response candidate identity or score is invalid",
                ));
            }
        }
        Ok(())
    }
}

#[async_trait]
pub trait RerankModel: Model {
    fn limits(&self) -> RerankLimits {
        RerankLimits::default()
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error>;
}

#[async_trait]
impl<T> RerankModel for Arc<T>
where
    T: RerankModel + ?Sized,
{
    fn limits(&self) -> RerankLimits {
        self.as_ref().limits()
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        self.as_ref().rerank(request, options).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSize {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl ImageSize {
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        let width = NonZeroU32::new(width)
            .ok_or_else(|| invalid_input("image width must be greater than zero"))?;
        let height = NonZeroU32::new(height)
            .ok_or_else(|| invalid_input("image height must be greater than zero"))?;
        Ok(Self { width, height })
    }

    pub fn width(self) -> u32 {
        self.width.get()
    }

    pub fn height(self) -> u32 {
        self.height.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageRequest {
    prompt: String,
    count: NonZeroU32,
    size: Option<ImageSize>,
    format: Option<String>,
}

impl ImageRequest {
    pub fn new(prompt: impl Into<String>) -> Result<Self, Error> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(invalid_input("image prompt must not be empty"));
        }
        Ok(Self {
            prompt,
            count: NonZeroU32::MIN,
            size: None,
            format: None,
        })
    }

    pub fn with_count(mut self, count: u32) -> Result<Self, Error> {
        self.count = NonZeroU32::new(count)
            .ok_or_else(|| invalid_input("image count must be greater than zero"))?;
        Ok(self)
    }

    pub fn with_size(mut self, width: u32, height: u32) -> Result<Self, Error> {
        self.size = Some(ImageSize::new(width, height)?);
        Ok(self)
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Result<Self, Error> {
        let format = format.into();
        if format.trim().is_empty() {
            return Err(invalid_input("image format must not be empty"));
        }
        self.format = Some(format);
        Ok(self)
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn count(&self) -> u32 {
        self.count.get()
    }

    pub fn size(&self) -> Option<ImageSize> {
        self.size
    }

    pub fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }
}

#[derive(Deserialize)]
struct ImageRequestWire {
    prompt: String,
    count: u32,
    size: Option<ImageSize>,
    format: Option<String>,
}

impl<'de> Deserialize<'de> for ImageRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ImageRequestWire::deserialize(deserializer)?;
        let mut request = Self::new(wire.prompt)
            .and_then(|request| request.with_count(wire.count))
            .map_err(serde::de::Error::custom)?;
        request.size = wire.size;
        if let Some(format) = wire.format {
            request = request
                .with_format(format)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageLimits {
    pub max_outputs_per_call: Option<u32>,
}

impl ImageLimits {
    pub fn validate(&self, request: &ImageRequest) -> Result<(), Error> {
        if let Some(maximum) = self.max_outputs_per_call
            && request.count() > maximum
        {
            return Err(Error::limit_exceeded(
                ResourceKind::ImageOutputs,
                u64::from(request.count()),
                u64::from(maximum),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub media_type: String,
    pub data: MediaData,
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    pub images: Vec<ImageArtifact>,
    #[serde(default)]
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl ImageResponse {
    pub fn validate(&self, request: &ImageRequest) -> Result<(), Error> {
        let expected = usize::try_from(request.count()).unwrap_or(usize::MAX);
        if self.images.len() < expected {
            return Err(Error::partial_result(
                ResourceKind::ImageOutputs,
                as_u64(expected),
                as_u64(self.images.len()),
            ));
        }
        if self.images.len() > expected {
            return Err(Error::protocol_violation(
                "image response contains more outputs than requested",
            ));
        }
        for image in &self.images {
            if !valid_media_type(&image.media_type)
                || matches!(&image.data, MediaData::Bytes(bytes) if bytes.is_empty())
                || matches!(&image.data, MediaData::Url(url) if url.trim().is_empty())
            {
                return Err(Error::protocol_violation(
                    "image response contains an invalid artifact",
                ));
            }
        }
        Ok(())
    }
}

#[async_trait]
pub trait ImageModel: Model {
    fn limits(&self) -> ImageLimits {
        ImageLimits::default()
    }

    /// Generate final artifacts in exactly one provider operation.
    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error>;
}

#[async_trait]
impl<T> ImageModel for Arc<T>
where
    T: ImageModel + ?Sized,
{
    fn limits(&self) -> ImageLimits {
        self.as_ref().limits()
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.as_ref().generate_image(request, options).await
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeechRequest {
    text: String,
    voice: Option<String>,
    format: Option<String>,
    language: Option<String>,
    speed: Option<f32>,
}

impl SpeechRequest {
    pub fn new(text: impl Into<String>) -> Result<Self, Error> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(invalid_input("speech text must not be empty"));
        }
        Ok(Self {
            text,
            voice: None,
            format: None,
            language: None,
            speed: None,
        })
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Result<Self, Error> {
        self.voice = Some(non_empty_option(voice, "speech voice must not be empty")?);
        Ok(self)
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Result<Self, Error> {
        self.format = Some(non_empty_option(format, "speech format must not be empty")?);
        Ok(self)
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self, Error> {
        self.language = Some(non_empty_option(
            language,
            "speech language must not be empty",
        )?);
        Ok(self)
    }

    pub fn with_speed(mut self, speed: f32) -> Result<Self, Error> {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(invalid_input(
                "speech speed must be finite and greater than zero",
            ));
        }
        self.speed = Some(speed);
        Ok(self)
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn voice(&self) -> Option<&str> {
        self.voice.as_deref()
    }

    pub fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn speed(&self) -> Option<f32> {
        self.speed
    }
}

#[derive(Deserialize)]
struct SpeechRequestWire {
    text: String,
    voice: Option<String>,
    format: Option<String>,
    language: Option<String>,
    speed: Option<f32>,
}

impl<'de> Deserialize<'de> for SpeechRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpeechRequestWire::deserialize(deserializer)?;
        let mut request = Self::new(wire.text).map_err(serde::de::Error::custom)?;
        if let Some(voice) = wire.voice {
            request = request
                .with_voice(voice)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(format) = wire.format {
            request = request
                .with_format(format)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(language) = wire.language {
            request = request
                .with_language(language)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(speed) = wire.speed {
            request = request
                .with_speed(speed)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechLimits {
    pub max_text_bytes: Option<usize>,
    pub max_text_chars: Option<usize>,
}

impl SpeechLimits {
    pub fn validate(&self, request: &SpeechRequest) -> Result<(), Error> {
        if let Some(maximum) = self.max_text_bytes
            && request.text.len() > maximum
        {
            return Err(Error::limit_exceeded(
                ResourceKind::SpeechTextBytes,
                as_u64(request.text.len()),
                as_u64(maximum),
            ));
        }
        if let Some(maximum) = self.max_text_chars {
            let actual = request.text.chars().count();
            if actual > maximum {
                return Err(Error::limit_exceeded(
                    ResourceKind::SpeechTextCharacters,
                    as_u64(actual),
                    as_u64(maximum),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechResponse {
    pub media_type: String,
    pub audio: Bytes,
    pub duration_seconds: Option<f64>,
    pub sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl SpeechResponse {
    pub fn validate(&self) -> Result<(), Error> {
        if !valid_media_type(&self.media_type) || self.audio.is_empty() {
            return Err(Error::protocol_violation(
                "speech response contains invalid or empty audio",
            ));
        }
        if self
            .duration_seconds
            .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
            || self.sample_rate_hz == Some(0)
        {
            return Err(Error::protocol_violation(
                "speech response contains invalid audio metadata",
            ));
        }
        Ok(())
    }
}

#[async_trait]
pub trait SpeechModel: Model {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits::default()
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error>;
}

#[async_trait]
impl<T> SpeechModel for Arc<T>
where
    T: SpeechModel + ?Sized,
{
    fn limits(&self) -> SpeechLimits {
        self.as_ref().limits()
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        self.as_ref().synthesize(request, options).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TranscriptionRequest {
    audio: Bytes,
    media_type: String,
    language: Option<String>,
    prompt: Option<String>,
}

impl TranscriptionRequest {
    pub fn new(audio: impl Into<Bytes>, media_type: impl Into<String>) -> Result<Self, Error> {
        let audio = audio.into();
        let media_type = media_type.into();
        if audio.is_empty() {
            return Err(invalid_input("transcription audio must not be empty"));
        }
        if !valid_media_type(&media_type) {
            return Err(invalid_input(
                "transcription media type must be a valid MIME type",
            ));
        }
        Ok(Self {
            audio,
            media_type,
            language: None,
            prompt: None,
        })
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self, Error> {
        self.language = Some(non_empty_option(
            language,
            "transcription language must not be empty",
        )?);
        Ok(self)
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Result<Self, Error> {
        self.prompt = Some(non_empty_option(
            prompt,
            "transcription prompt must not be empty",
        )?);
        Ok(self)
    }

    pub fn audio(&self) -> &Bytes {
        &self.audio
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn prompt(&self) -> Option<&str> {
        self.prompt.as_deref()
    }
}

#[derive(Deserialize)]
struct TranscriptionRequestWire {
    audio: Bytes,
    media_type: String,
    language: Option<String>,
    prompt: Option<String>,
}

impl<'de> Deserialize<'de> for TranscriptionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TranscriptionRequestWire::deserialize(deserializer)?;
        let mut request =
            Self::new(wire.audio, wire.media_type).map_err(serde::de::Error::custom)?;
        if let Some(language) = wire.language {
            request = request
                .with_language(language)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(prompt) = wire.prompt {
            request = request
                .with_prompt(prompt)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionLimits {
    pub max_audio_bytes: Option<usize>,
    pub max_duration_seconds: Option<f64>,
}

impl TranscriptionLimits {
    pub fn validate(&self, request: &TranscriptionRequest) -> Result<(), Error> {
        if let Some(maximum) = self.max_audio_bytes
            && request.audio.len() > maximum
        {
            return Err(Error::limit_exceeded(
                ResourceKind::TranscriptionAudioBytes,
                as_u64(request.audio.len()),
                as_u64(maximum),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
    pub confidence: Option<f64>,
}

impl TranscriptSegment {
    pub fn validate(&self) -> Result<(), Error> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || self.start_seconds < 0.0
            || self.end_seconds < self.start_seconds
            || self
                .confidence
                .is_some_and(|confidence| !valid_confidence(confidence))
        {
            return Err(Error::protocol_violation(
                "transcription segment contains invalid timing or confidence",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub confidence: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub segments: Vec<TranscriptSegment>,
    #[serde(default)]
    pub metadata: ResponseMetadata,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

impl TranscriptionResponse {
    pub fn validate(&self) -> Result<(), Error> {
        if self
            .confidence
            .is_some_and(|confidence| !valid_confidence(confidence))
            || self
                .duration_seconds
                .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
        {
            return Err(Error::protocol_violation(
                "transcription response contains invalid metadata",
            ));
        }
        self.segments
            .iter()
            .try_for_each(TranscriptSegment::validate)
    }
}

#[async_trait]
pub trait TranscriptionModel: Model {
    fn limits(&self) -> TranscriptionLimits {
        TranscriptionLimits::default()
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error>;
}

#[async_trait]
impl<T> TranscriptionModel for Arc<T>
where
    T: TranscriptionModel + ?Sized,
{
    fn limits(&self) -> TranscriptionLimits {
        self.as_ref().limits()
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        self.as_ref().transcribe(request, options).await
    }
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn non_empty_option(value: impl Into<String>, message: &'static str) -> Result<String, Error> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(invalid_input(message))
    } else {
        Ok(value)
    }
}

fn valid_media_type(media_type: &str) -> bool {
    let media_type = media_type.trim();
    if media_type.is_empty() || media_type.chars().any(char::is_control) {
        return false;
    }
    let essence = media_type.split(';').next().unwrap_or_default().trim();
    essence
        .split_once('/')
        .is_some_and(|(kind, subtype)| !kind.is_empty() && !subtype.is_empty())
}

fn valid_confidence(confidence: f64) -> bool {
    confidence.is_finite() && (0.0..=1.0).contains(&confidence)
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorDetail;

    #[test]
    fn embedding_requests_cannot_encode_empty_or_token_like_inputs() {
        assert!(EmbeddingRequest::new(Vec::<String>::new()).is_err());
        assert!(EmbeddingRequest::single("  ").is_err());
        assert!(
            serde_json::from_str::<EmbeddingRequest>(r#"{"inputs":[],"dimensions":null}"#).is_err()
        );

        let request = EmbeddingRequest::new(["first", "second"])
            .unwrap()
            .with_dimensions(2)
            .unwrap();
        assert_eq!(request.inputs(), ["first", "second"]);
        assert_eq!(request.dimensions().map(NonZeroU32::get), Some(2));
    }

    #[test]
    fn model_limits_return_matchable_details_before_network_work() {
        let request = EmbeddingRequest::new(["one", "two"]).unwrap();
        let error = EmbeddingLimits {
            max_inputs: Some(1),
            max_input_tokens: None,
        }
        .validate(&request)
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        assert_eq!(
            error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::EmbeddingInputs,
                actual: 2,
                maximum: 1,
            })
        );
    }

    #[test]
    fn rerank_results_preserve_request_identity_and_reject_partial_data() {
        let request = RerankRequest::new(
            "query",
            vec![
                RerankCandidate::new("a").unwrap().with_id("a-id").unwrap(),
                RerankCandidate::new("b").unwrap().with_id("b-id").unwrap(),
            ],
        )
        .unwrap();
        let response = RerankResponse {
            results: vec![RerankResult::new(0, 0.9, Some("a-id".to_string())).unwrap()],
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };

        let error = response.validate(&request).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PartialResult);
        assert!(matches!(
            error.detail(),
            Some(ErrorDetail::PartialResult {
                resource: ResourceKind::RerankCandidates,
                expected: 2,
                actual: 1,
            })
        ));
    }

    #[test]
    fn final_media_requests_reject_invalid_intrinsic_state() {
        assert!(ImageRequest::new("").is_err());
        assert!(ImageRequest::new("image").unwrap().with_count(0).is_err());
        assert!(
            SpeechRequest::new("speech")
                .unwrap()
                .with_speed(f32::NAN)
                .is_err()
        );
        assert!(TranscriptionRequest::new(Vec::<u8>::new(), "audio/wav").is_err());
        assert!(TranscriptionRequest::new(vec![1_u8], "not-a-media-type").is_err());
    }

    #[test]
    fn speech_limits_distinguish_utf8_bytes_from_provider_characters() {
        let request = SpeechRequest::new("\u{4f60}\u{597d}").unwrap();

        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(2),
        }
        .validate(&request)
        .unwrap();
        let error = SpeechLimits {
            max_text_bytes: Some(2),
            max_text_chars: None,
        }
        .validate(&request)
        .unwrap_err();

        assert_eq!(
            error.detail(),
            Some(&ErrorDetail::LimitExceeded {
                resource: ResourceKind::SpeechTextBytes,
                actual: 6,
                maximum: 2,
            })
        );
    }

    #[test]
    fn owned_binary_results_are_clone_cheap_and_task_safe() {
        let original = Bytes::from_static(b"audio");
        let cloned = original.clone();
        let returned = std::thread::spawn(move || cloned).join().unwrap();

        assert_eq!(returned, original);
    }
}
