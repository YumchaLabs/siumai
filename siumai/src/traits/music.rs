//! Music generation capability trait

use crate::error::LlmError;
use crate::types::music::{MusicGenerationRequest, MusicGenerationResponse, MusicStyle};
use async_trait::async_trait;

/// Capability for music generation
///
/// This trait provides a unified interface for AI music generation across different providers.
/// Providers may support different features (lyrics, style transfer, continuation, etc.),
/// so optional methods have default implementations that return `UnsupportedOperation` errors.
///
/// # Supported Providers
///
/// - **MiniMaxi**: Music 2.0 model with lyrics support
/// - **Suno** (future): Full song generation with vocals
/// - **Udio** (future): Music generation with style control
/// - **Stable Audio** (future): Audio and music generation
///
/// # Example
///
/// ```ignore
/// use siumai::prelude::*;
/// use siumai::types::music::MusicGenerationRequest;
///
/// let request = MusicGenerationRequest::new("music-2.0", "Indie folk, melancholic")
///     .with_lyrics("[verse]\nLyrics here");
///
/// let response = client.generate_music(request).await?;
/// std::fs::write("music.mp3", &response.audio_data)?;
/// ```
#[async_trait]
pub trait MusicGenerationCapability: Send + Sync {
    /// Generate music from text description and optional lyrics
    ///
    /// This is the primary method for music generation. All providers must implement this.
    ///
    /// # Arguments
    ///
    /// * `request` - Music generation request with prompt, optional lyrics, and audio settings
    ///
    /// # Returns
    ///
    /// Music generation response containing audio data and metadata
    ///
    /// # Errors
    ///
    /// Returns error if the request fails or the response cannot be parsed
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError>;

    /// Continue or extend an existing music piece
    ///
    /// Some providers support extending music from a seed audio or continuing from a timestamp.
    /// Default implementation returns `UnsupportedOperation`.
    ///
    /// # Arguments
    ///
    /// * `request` - Music generation request with continuation parameters
    ///
    /// # Returns
    ///
    /// Extended music response
    async fn continue_music(
        &self,
        _request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Music continuation not supported by this provider".to_string(),
        ))
    }

    /// Get list of supported music generation models
    ///
    /// # Returns
    ///
    /// Vector of model names that can be used for music generation
    fn get_supported_music_models(&self) -> Vec<String>;

    /// Get list of supported music styles/genres
    ///
    /// Some providers have predefined styles or genres. Default implementation returns empty.
    ///
    /// # Returns
    ///
    /// Vector of supported music styles
    fn get_supported_styles(&self) -> Vec<MusicStyle> {
        vec![]
    }

    /// Get supported audio formats for music output
    ///
    /// # Returns
    ///
    /// Vector of supported audio formats (e.g., "mp3", "wav", "flac")
    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec!["mp3".to_string(), "wav".to_string()]
    }

    /// Check if provider supports lyrics-based generation
    ///
    /// # Returns
    ///
    /// `true` if provider supports lyrics input
    fn supports_lyrics(&self) -> bool {
        false
    }

    /// Check if provider supports music continuation/extension
    ///
    /// # Returns
    ///
    /// `true` if provider supports continuing existing music
    fn supports_continuation(&self) -> bool {
        false
    }

    /// Check if provider supports instrumental-only generation
    ///
    /// # Returns
    ///
    /// `true` if provider can generate music without vocals
    fn supports_instrumental(&self) -> bool {
        true
    }

    /// Convenience method: Generate music from a simple text prompt
    ///
    /// This is a simplified interface for quick music generation without lyrics.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Text description of the music style and mood
    /// * `duration` - Optional duration in seconds
    ///
    /// # Returns
    ///
    /// Generated music audio data
    async fn generate_from_prompt(
        &self,
        prompt: String,
        duration: Option<u32>,
    ) -> Result<Vec<u8>, LlmError> {
        let models = self.get_supported_music_models();
        let model = models
            .first()
            .ok_or_else(|| LlmError::provider_error("music", "No supported models available"))?;

        let mut request = MusicGenerationRequest::new(model.clone(), prompt);
        if let Some(dur) = duration {
            request.duration = Some(dur);
        }

        let response = self.generate_music(request).await?;
        Ok(response.audio_data)
    }
}
