//! xAI Model Constants
//!
//! This module provides convenient constants for xAI Grok models, making it easy
//! for developers to reference specific models without hardcoding strings.
//!
//! # Model Families
//!
//! - **Grok 4**: Latest flagship reasoning model with vision support
//! - **Grok 3**: General purpose chat models with various sizes
//! - **Grok 2**: Previous generation models including image generation
//! - **Legacy**: Older models for compatibility

/// Grok 4 model family constants (latest flagship)
pub mod grok_4 {
    /// Grok 4 - Latest flagship reasoning model with vision support
    pub const GROK_4: &str = "grok-4";
    /// Grok 4 (2024-07-09) - Specific version of Grok 4
    pub const GROK_4_0709: &str = "grok-4-0709";
    /// Grok 4 Latest - Alias for latest Grok 4
    pub const GROK_4_LATEST: &str = "grok-4-latest";

    /// All Grok 4 models
    pub const ALL: &[&str] = &[GROK_4, GROK_4_0709, GROK_4_LATEST];
}

/// Grok 3 model family constants
pub mod grok_3 {
    /// Grok 3 - General purpose chat model
    pub const GROK_3: &str = "grok-3";
    /// Grok 3 Latest - Alias for latest Grok 3
    pub const GROK_3_LATEST: &str = "grok-3-latest";

    /// Grok 3 Mini - Lightweight version of Grok 3
    pub const GROK_3_MINI: &str = "grok-3-mini";

    /// Grok 3 Fast - Fast version for quick responses
    pub const GROK_3_FAST: &str = "grok-3-fast";
    /// Grok 3 Fast Latest - Alias for latest fast version
    pub const GROK_3_FAST_LATEST: &str = "grok-3-fast-latest";
    /// Grok 3 Fast Beta - Beta version of fast model
    pub const GROK_3_FAST_BETA: &str = "grok-3-fast-beta";

    /// All Grok 3 models
    pub const ALL: &[&str] = &[
        GROK_3,
        GROK_3_LATEST,
        GROK_3_MINI,
        GROK_3_FAST,
        GROK_3_FAST_LATEST,
        GROK_3_FAST_BETA,
    ];
}

/// Grok 2 model family constants
pub mod grok_2 {
    /// Grok 2 - Previous generation model
    pub const GROK_2: &str = "grok-2";
    /// Grok 2 Latest - Alias for latest Grok 2
    pub const GROK_2_LATEST: &str = "grok-2-latest";
    /// Grok 2 (2024-12-12) - Specific version
    pub const GROK_2_1212: &str = "grok-2-1212";

    /// All Grok 2 models
    pub const ALL: &[&str] = &[GROK_2, GROK_2_LATEST, GROK_2_1212];
}

/// Image generation model constants
pub mod images {
    /// Grok 2 Image - Image generation model
    pub const GROK_2_IMAGE: &str = "grok-2-image";
    /// Grok 2 Image (2024-12-12) - Specific image generation version
    pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";

    /// All image generation models
    pub const ALL: &[&str] = &[GROK_2_IMAGE, GROK_2_IMAGE_1212];
}

/// Legacy model constants
pub mod legacy {
    /// Legacy Grok Beta - Early access model (deprecated)
    pub const GROK_BETA: &str = "grok-beta";

    /// All legacy models
    pub const ALL: &[&str] = &[GROK_BETA];
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most capable model for general use
    pub const FLAGSHIP: &str = grok_4::GROK_4;
    /// Best balance of capability and speed
    pub const BALANCED: &str = grok_3::GROK_3;
    /// Fastest model for quick responses
    pub const FAST: &str = grok_3::GROK_3_FAST;
    /// Lightweight model for simple tasks
    pub const LIGHTWEIGHT: &str = grok_3::GROK_3_MINI;
    /// Best for reasoning tasks
    pub const REASONING: &str = grok_4::GROK_4;
    /// Latest and most advanced
    pub const LATEST: &str = grok_4::GROK_4_LATEST;
    /// Best for image generation
    pub const IMAGE_GENERATION: &str = images::GROK_2_IMAGE;
}

pub use grok_2::GROK_2;
pub use grok_3::GROK_3;
pub use grok_3::GROK_3_FAST;
pub use grok_3::GROK_3_MINI;
/// Simplified access to popular models (top-level constants)
pub use grok_4::GROK_4;
pub use images::GROK_2_IMAGE;

/// Get all available models
pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(grok_4::ALL);
    models.extend_from_slice(grok_3::ALL);
    models.extend_from_slice(grok_2::ALL);
    models.extend_from_slice(images::ALL);
    models.extend_from_slice(legacy::ALL);
    models
}

/// Get models by capability
pub mod by_capability {
    use super::*;

    /// Models that support reasoning
    pub const REASONING: &[&str] = &[grok_4::GROK_4, grok_4::GROK_4_0709, grok_4::GROK_4_LATEST];

    /// Models that support vision/image input
    pub const VISION: &[&str] = &[grok_4::GROK_4, grok_4::GROK_4_0709, grok_4::GROK_4_LATEST];

    /// Models that support image generation
    pub const IMAGE_GENERATION: &[&str] = images::ALL;

    /// Models optimized for speed
    pub const FAST: &[&str] = &[
        grok_3::GROK_3_FAST,
        grok_3::GROK_3_FAST_LATEST,
        grok_3::GROK_3_FAST_BETA,
        grok_3::GROK_3_MINI,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_model_constants() {
        // Test that constants are not empty
        assert!(!grok_4::GROK_4.is_empty());
        assert!(!grok_3::GROK_3.is_empty());
        assert!(!grok_2::GROK_2.is_empty());
        assert!(!images::GROK_2_IMAGE.is_empty());
    }

    #[test]
    fn test_all_models() {
        let models = all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&grok_4::GROK_4));
        assert!(models.contains(&grok_3::GROK_3));
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_popular_recommendations() {
        assert!(!popular::FLAGSHIP.is_empty());
        assert!(!popular::BALANCED.is_empty());
        assert!(!popular::REASONING.is_empty());
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_capability_groups() {
        assert!(!by_capability::REASONING.is_empty());
        assert!(!by_capability::VISION.is_empty());
        assert!(!by_capability::IMAGE_GENERATION.is_empty());
        assert!(!by_capability::FAST.is_empty());
    }
}
