//! Feature-gated provider namespaces.

#[cfg(feature = "bedrock")]
pub use siumai_provider_amazon_bedrock as bedrock;
#[cfg(feature = "anthropic")]
pub use siumai_provider_anthropic as anthropic;
#[cfg(feature = "azure")]
pub use siumai_provider_azure as azure;
#[cfg(feature = "cohere")]
pub use siumai_provider_cohere as cohere;
#[cfg(feature = "deepgram")]
pub use siumai_provider_deepgram as deepgram;
#[cfg(feature = "deepseek")]
pub use siumai_provider_deepseek as deepseek;
#[cfg(feature = "elevenlabs")]
pub use siumai_provider_elevenlabs as elevenlabs;
#[cfg(feature = "gateway")]
pub use siumai_provider_gateway as gateway;
#[cfg(feature = "google")]
pub use siumai_provider_gemini as google;
#[cfg(feature = "google-vertex")]
pub use siumai_provider_google_vertex as google_vertex;
#[cfg(feature = "groq")]
pub use siumai_provider_groq as groq;
#[cfg(feature = "minimaxi")]
pub use siumai_provider_minimaxi as minimaxi;
#[cfg(feature = "ollama")]
pub use siumai_provider_ollama as ollama;
#[cfg(feature = "openai")]
pub use siumai_provider_openai as openai;
#[cfg(feature = "openai-compatible")]
pub use siumai_provider_openai_compatible as openai_compatible;
#[cfg(feature = "togetherai")]
pub use siumai_provider_togetherai as togetherai;
#[cfg(feature = "xai")]
pub use siumai_provider_xai as xai;
