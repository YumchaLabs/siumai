#[cfg(feature = "alibaba")]
mod alibaba;
#[cfg(feature = "anthropic")]
mod anthropic;
mod clients;
#[cfg(feature = "cohere")]
mod cohere;
#[cfg(feature = "deepgram")]
mod deepgram;
#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "elevenlabs")]
mod elevenlabs;
#[cfg(feature = "google")]
mod gemini;
#[cfg(feature = "groq")]
mod groq;
#[cfg(feature = "minimax")]
mod minimax;
#[cfg(feature = "moonshotai")]
mod moonshot;
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai-compatible")]
mod openai_compatible;
#[cfg(feature = "google-vertex-anthropic")]
mod vertex_anthropic;
#[cfg(feature = "volcengine")]
mod volcengine;
#[cfg(feature = "xai")]
mod xai;

use std::fmt;
use std::sync::Arc;

use siumai_core::{
    EmbeddingModelProvider, ImageModelProvider, LanguageModelProvider, ModelId, ModelLookupError,
    Provider, RerankModelProvider, SpeechModelProvider, TranscriptionModelProvider,
};

#[cfg(feature = "alibaba")]
pub use alibaba::{AlibabaConfigurationStage, AlibabaCredentialStage, AlibabaProviderStage};
#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicCredentialStage, AnthropicProviderStage};
pub use clients::{
    EmbeddingClient, ImageClient, LanguageClient, RerankClient, SpeechClient, TranscriptionClient,
};
#[cfg(feature = "cohere")]
pub use cohere::{CohereApiKeyStage, CohereProviderStage};
#[cfg(feature = "deepgram")]
pub use deepgram::{DeepgramCredentialStage, DeepgramProviderStage};
#[cfg(feature = "deepseek")]
pub use deepseek::{DeepSeekCredentialStage, DeepSeekProviderStage};
#[cfg(feature = "elevenlabs")]
pub use elevenlabs::{ElevenLabsCredentialStage, ElevenLabsProfileStage, ElevenLabsProviderStage};
#[cfg(feature = "google")]
pub use gemini::{GeminiCredentialStage, GeminiProviderStage};
#[cfg(feature = "groq")]
pub use groq::{GroqCredentialStage, GroqProviderStage};
#[cfg(feature = "minimax")]
pub use minimax::{MinimaxCredentialStage, MinimaxProviderStage};
#[cfg(feature = "moonshotai")]
pub use moonshot::{MoonshotCredentialStage, MoonshotProviderStage};
#[cfg(feature = "openai")]
pub use openai::{OpenAiCredentialStage, OpenAiProviderStage};
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::{
    OpenAiCompatibleCredentialStage, OpenAiCompatibleProfileStage, OpenAiCompatibleProviderStage,
};
#[cfg(feature = "google-vertex-anthropic")]
pub use vertex_anthropic::{
    VertexAnthropicCredentialStage, VertexAnthropicLocationStage, VertexAnthropicProjectStage,
    VertexAnthropicProviderStage,
};
#[cfg(feature = "volcengine")]
pub use volcengine::{VolcengineCredentialStage, VolcengineProviderStage};
#[cfg(feature = "xai")]
pub use xai::{XaiCredentialStage, XaiProviderStage};

/// A typed hub around one concrete configured provider.
///
/// The hub retains no default models, capability matrix, Registry, or runtime
/// routing state. Family selectors exist only when `P` implements the
/// corresponding provider trait.
///
/// ```compile_fail
/// use siumai::{Provider, ProviderId, Siumai};
///
/// struct ProviderWithoutEmbedding(ProviderId);
///
/// impl Provider for ProviderWithoutEmbedding {
///     fn provider_id(&self) -> &ProviderId {
///         &self.0
///     }
/// }
///
/// let provider = ProviderWithoutEmbedding(
///     ProviderId::new("language-only").expect("the static provider ID is valid"),
/// );
/// let hub = Siumai::from_provider(provider);
/// let _ = hub.embedding("unsupported-model");
/// ```
pub struct Siumai<P = ()> {
    provider: Arc<P>,
}

impl Siumai<()> {
    /// Start the zero-state typed provider construction journey.
    pub const fn builder() -> SiumaiBuilder {
        SiumaiBuilder::new()
    }

    /// Wrap an already configured provider without rebuilding it.
    pub fn from_provider<P>(provider: P) -> Siumai<P>
    where
        P: Provider,
    {
        Siumai {
            provider: Arc::new(provider),
        }
    }
}

impl<P> Siumai<P> {
    /// Return the concrete configured provider retained by this hub.
    pub fn provider(&self) -> &P {
        self.provider.as_ref()
    }
}

impl<P> Clone for Siumai<P> {
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
        }
    }
}

impl<P> fmt::Debug for Siumai<P>
where
    P: Provider,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Siumai")
            .field("provider_id", self.provider().provider_id())
            .finish_non_exhaustive()
    }
}

impl<P> Siumai<P>
where
    P: LanguageModelProvider,
{
    /// Bind a language model produced by this configured provider.
    pub fn language(
        &self,
        model: impl Into<String>,
    ) -> Result<LanguageClient<P, P::Model>, ModelLookupError> {
        let model = self
            .provider
            .language_model(parse_model_id(model.into())?)?;
        Ok(LanguageClient::new(self.provider.clone(), model))
    }
}

impl<P> Siumai<P>
where
    P: EmbeddingModelProvider,
{
    /// Bind an embedding model produced by this configured provider.
    pub fn embedding(
        &self,
        model: impl Into<String>,
    ) -> Result<EmbeddingClient<P, P::Model>, ModelLookupError> {
        let model = self
            .provider
            .embedding_model(parse_model_id(model.into())?)?;
        Ok(EmbeddingClient::new(self.provider.clone(), model))
    }
}

impl<P> Siumai<P>
where
    P: RerankModelProvider,
{
    /// Bind a rerank model produced by this configured provider.
    pub fn rerank(
        &self,
        model: impl Into<String>,
    ) -> Result<RerankClient<P, P::Model>, ModelLookupError> {
        let model = self.provider.rerank_model(parse_model_id(model.into())?)?;
        Ok(RerankClient::new(self.provider.clone(), model))
    }
}

impl<P> Siumai<P>
where
    P: ImageModelProvider,
{
    /// Bind an image model produced by this configured provider.
    pub fn image(
        &self,
        model: impl Into<String>,
    ) -> Result<ImageClient<P, P::Model>, ModelLookupError> {
        let model = self.provider.image_model(parse_model_id(model.into())?)?;
        Ok(ImageClient::new(self.provider.clone(), model))
    }
}

impl<P> Siumai<P>
where
    P: SpeechModelProvider,
{
    /// Bind a speech model produced by this configured provider.
    pub fn speech(
        &self,
        model: impl Into<String>,
    ) -> Result<SpeechClient<P, P::Model>, ModelLookupError> {
        let model = self.provider.speech_model(parse_model_id(model.into())?)?;
        Ok(SpeechClient::new(self.provider.clone(), model))
    }
}

impl<P> Siumai<P>
where
    P: TranscriptionModelProvider,
{
    /// Bind a transcription model produced by this configured provider.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Result<TranscriptionClient<P, P::Model>, ModelLookupError> {
        let model = self
            .provider
            .transcription_model(parse_model_id(model.into())?)?;
        Ok(TranscriptionClient::new(self.provider.clone(), model))
    }
}

fn parse_model_id(model: String) -> Result<ModelId, ModelLookupError> {
    ModelId::new(model).map_err(ModelLookupError::from)
}

/// The zero-state entry for feature-gated provider construction adapters.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct SiumaiBuilder {
    _private: (),
}

impl SiumaiBuilder {
    const fn new() -> Self {
        Self { _private: () }
    }
}

impl fmt::Debug for SiumaiBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SiumaiBuilder")
    }
}
