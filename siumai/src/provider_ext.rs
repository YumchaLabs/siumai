#[cfg(feature = "openai")]
pub mod openai;

/// OpenAI-compatible vendors (DeepSeek/OpenRouter/Moonshot/etc.) via the OpenAI-like protocol family.
#[cfg(feature = "openai")]
pub mod openai_compatible;

#[cfg(feature = "openai")]
pub mod openrouter;

#[cfg(feature = "openai")]
pub mod mistral;

#[cfg(feature = "openai")]
pub mod perplexity;

#[cfg(feature = "openai")]
pub mod fireworks;

#[cfg(feature = "openai")]
pub mod moonshotai;

#[cfg(feature = "deepinfra")]
pub mod deepinfra;

#[cfg(feature = "google-vertex")]
pub mod vertex_maas;

#[cfg(feature = "google-vertex")]
pub mod google_vertex_xai;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "cohere")]
pub mod cohere;

#[cfg(feature = "togetherai")]
pub mod togetherai;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "google")]
pub mod gemini;

/// Vercel alignment: the AI SDK uses `@ai-sdk/google` for Gemini.
///
/// This is a stable alias for the Gemini provider extension surface.
#[cfg(feature = "google")]
pub mod google;

#[cfg(feature = "google-vertex")]
pub mod google_vertex;

#[cfg(feature = "minimaxi")]
pub mod minimaxi;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "google-vertex")]
pub mod anthropic_vertex;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "groq")]
pub mod groq;
