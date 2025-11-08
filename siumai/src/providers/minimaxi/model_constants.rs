//! MiniMaxi Model Constants
//!
//! This module provides convenient constants for MiniMaxi models, making it easy
//! for developers to reference specific models without hardcoding strings.
//!
//! # Model Families
//!
//! - **Text**: M2 model for chat completion and function calling
//! - **Audio**: Speech 2.6 HD/Turbo for text-to-speech
//! - **Video**: Hailuo 2.3 & 2.3 Fast for video generation
//! - **Music**: Music 2.0 for music generation
//! - **Images**: Image generation models

// ============================================================================
// Text Models
// ============================================================================

/// Text generation model family constants
pub mod text {
    /// MiniMax M2 - Advanced text generation model
    ///
    /// Specialized for efficient coding and Agent workflows.
    /// Max tokens: 204,800 (input + output)
    ///
    /// Supports:
    /// - Chat completion
    /// - Function calling
    /// - Streaming
    /// - OpenAI and Anthropic API compatibility
    pub const MINIMAX_M2: &str = "MiniMax-M2";

    /// MiniMax M2 Stable - Stable version for production use
    ///
    /// Stable version of M2 model optimized for production environments.
    /// Max tokens: 204,800 (input + output)
    ///
    /// Supports:
    /// - Chat completion
    /// - Function calling
    /// - Streaming
    /// - OpenAI and Anthropic API compatibility
    pub const MINIMAX_M2_STABLE: &str = "MiniMax-M2-Stable";

    /// All text models
    pub const ALL: &[&str] = &[MINIMAX_M2, MINIMAX_M2_STABLE];
}

// ============================================================================
// Audio Models (Text-to-Speech)
// ============================================================================

/// Audio model family constants (Text-to-Speech)
pub mod audio {
    /// Speech 2.6 HD - High quality text-to-speech
    ///
    /// High-definition audio quality for professional use cases.
    pub const SPEECH_2_6_HD: &str = "speech-2.6-hd";

    /// Speech 2.6 Turbo - Fast text-to-speech
    ///
    /// Optimized for speed while maintaining good quality.
    pub const SPEECH_2_6_TURBO: &str = "speech-2.6-turbo";

    /// All audio models
    pub const ALL: &[&str] = &[SPEECH_2_6_HD, SPEECH_2_6_TURBO];
}

// ============================================================================
// Voice IDs (for Text-to-Speech)
// ============================================================================

/// Voice ID constants for text-to-speech
pub mod voice {
    /// Male voice - Qingse (清澈)
    pub const MALE_QN_QINGSE: &str = "male-qn-qingse";

    /// Female voice - Shaonv (少女)
    pub const FEMALE_SHAONV: &str = "female-shaonv";

    /// All available voices
    pub const ALL: &[&str] = &[MALE_QN_QINGSE, FEMALE_SHAONV];
}

// ============================================================================
// Video Models
// ============================================================================

/// Video generation model family constants
pub mod video {
    /// Hailuo 2.3 - High quality video generation
    ///
    /// Premium quality video generation with advanced features.
    pub const HAILUO_2_3: &str = "hailuo-2.3";

    /// Hailuo 2.3 Fast - Fast video generation
    ///
    /// Optimized for faster generation with good quality.
    pub const HAILUO_2_3_FAST: &str = "hailuo-2.3-fast";

    /// All video models
    pub const ALL: &[&str] = &[HAILUO_2_3, HAILUO_2_3_FAST];
}

// ============================================================================
// Music Models
// ============================================================================

/// Music generation model family constants
pub mod music {
    /// Music 2.0 - Music generation model
    ///
    /// Advanced music generation with lyrics and instrumental support.
    pub const MUSIC_2_0: &str = "music-2.0";

    /// All music models
    pub const ALL: &[&str] = &[MUSIC_2_0];
}

// ============================================================================
// Image Models
// ============================================================================

/// Image generation model family constants
pub mod images {
    /// Image-01 - Standard image generation model
    pub const IMAGE_01: &str = "image-01";

    /// Image-01-Live - Real-time image generation model
    pub const IMAGE_01_LIVE: &str = "image-01-live";

    /// All image models
    pub const ALL: &[&str] = &[IMAGE_01, IMAGE_01_LIVE];
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get all text models
pub fn all_text_models() -> Vec<&'static str> {
    text::ALL.to_vec()
}

/// Get all audio models
pub fn all_audio_models() -> Vec<&'static str> {
    audio::ALL.to_vec()
}

/// Get all video models
pub fn all_video_models() -> Vec<&'static str> {
    video::ALL.to_vec()
}

/// Get all music models
pub fn all_music_models() -> Vec<&'static str> {
    music::ALL.to_vec()
}

/// Get all image models
pub fn all_image_models() -> Vec<&'static str> {
    images::ALL.to_vec()
}

/// Get all MiniMaxi models across all capabilities
pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(text::ALL);
    models.extend_from_slice(audio::ALL);
    models.extend_from_slice(video::ALL);
    models.extend_from_slice(music::ALL);
    models.extend_from_slice(images::ALL);
    models
}

// ============================================================================
// Popular Model Recommendations
// ============================================================================

/// Popular model recommendations for different use cases
pub mod popular {
    use super::*;

    /// Best model for chat and coding
    pub const CHAT: &str = text::MINIMAX_M2;

    /// Best model for high-quality speech synthesis
    pub const TTS_HD: &str = audio::SPEECH_2_6_HD;

    /// Best model for fast speech synthesis
    pub const TTS_FAST: &str = audio::SPEECH_2_6_TURBO;

    /// Best model for high-quality video generation
    pub const VIDEO_HD: &str = video::HAILUO_2_3;

    /// Best model for fast video generation
    pub const VIDEO_FAST: &str = video::HAILUO_2_3_FAST;

    /// Best model for music generation
    pub const MUSIC: &str = music::MUSIC_2_0;

    /// Best model for image generation
    pub const IMAGE: &str = images::IMAGE_01;

    /// Best model for real-time image generation
    pub const IMAGE_LIVE: &str = images::IMAGE_01_LIVE;
}

// ============================================================================
// Model Capabilities
// ============================================================================

/// Model capabilities by category
pub mod capabilities {
    /// Models that support function calling
    pub const FUNCTION_CALLING_MODELS: &[&str] =
        &[super::text::MINIMAX_M2, super::text::MINIMAX_M2_STABLE];

    /// Models that support streaming
    pub const STREAMING_MODELS: &[&str] =
        &[super::text::MINIMAX_M2, super::text::MINIMAX_M2_STABLE];

    /// Models that support OpenAI API compatibility
    pub const OPENAI_COMPATIBLE_MODELS: &[&str] =
        &[super::text::MINIMAX_M2, super::text::MINIMAX_M2_STABLE];

    /// Models that support Anthropic API compatibility
    pub const ANTHROPIC_COMPATIBLE_MODELS: &[&str] =
        &[super::text::MINIMAX_M2, super::text::MINIMAX_M2_STABLE];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_text_models() {
        let models = all_text_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&text::MINIMAX_M2));
        assert!(models.contains(&text::MINIMAX_M2_STABLE));
    }

    #[test]
    fn test_all_audio_models() {
        let models = all_audio_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&audio::SPEECH_2_6_HD));
        assert!(models.contains(&audio::SPEECH_2_6_TURBO));
    }

    #[test]
    fn test_all_video_models() {
        let models = all_video_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&video::HAILUO_2_3));
        assert!(models.contains(&video::HAILUO_2_3_FAST));
    }

    #[test]
    fn test_all_music_models() {
        let models = all_music_models();
        assert_eq!(models.len(), 1);
        assert!(models.contains(&music::MUSIC_2_0));
    }

    #[test]
    fn test_all_image_models() {
        let models = all_image_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&images::IMAGE_01));
        assert!(models.contains(&images::IMAGE_01_LIVE));
    }

    #[test]
    fn test_all_models() {
        let models = all_models();
        assert_eq!(models.len(), 9); // 2 text + 2 audio + 2 video + 1 music + 2 image
    }

    #[test]
    fn test_popular_recommendations() {
        assert_eq!(popular::CHAT, text::MINIMAX_M2);
        assert_eq!(popular::TTS_HD, audio::SPEECH_2_6_HD);
        assert_eq!(popular::VIDEO_HD, video::HAILUO_2_3);
        assert_eq!(popular::MUSIC, music::MUSIC_2_0);
    }
}
