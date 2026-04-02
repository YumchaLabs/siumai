//! xAI Model Constants
//!
//! This module provides convenient constants for xAI Grok models, making it easy
//! for developers to reference specific models without hardcoding strings.
//!
//! # Model Families
//!
//! - **Grok 4**: Latest flagship reasoning model with vision support
//! - **Grok 3**: General purpose chat models with various sizes
//! - **Grok 2**: Previous generation chat models
//! - **Imagine**: Native image and video generation models
//! - **Legacy**: Older models for compatibility

/// Grok 4 model family constants (latest flagship)
pub mod grok_4 {
    /// Grok 4.1 Fast Reasoning - Fast reasoning-tuned Grok 4.1 variant
    pub const GROK_4_1_FAST_REASONING: &str = "grok-4-1-fast-reasoning";
    /// Grok 4.1 Fast Non-Reasoning - Fast non-reasoning Grok 4.1 variant
    pub const GROK_4_1_FAST_NON_REASONING: &str = "grok-4-1-fast-non-reasoning";
    /// Grok 4 Fast Non-Reasoning - Fast non-reasoning Grok 4 variant
    pub const GROK_4_FAST_NON_REASONING: &str = "grok-4-fast-non-reasoning";
    /// Grok 4 Fast Reasoning - Fast reasoning Grok 4 variant
    pub const GROK_4_FAST_REASONING: &str = "grok-4-fast-reasoning";
    /// Grok 4.20 (2025-03-09) Non-Reasoning - Snapshot model
    pub const GROK_4_20_0309_NON_REASONING: &str = "grok-4.20-0309-non-reasoning";
    /// Grok 4.20 (2025-03-09) Reasoning - Snapshot model
    pub const GROK_4_20_0309_REASONING: &str = "grok-4.20-0309-reasoning";
    /// Grok 4.20 Multi-Agent (2025-03-09) - Multi-agent snapshot model
    pub const GROK_4_20_MULTI_AGENT_0309: &str = "grok-4.20-multi-agent-0309";
    /// Grok 4 - Latest flagship reasoning model with vision support
    pub const GROK_4: &str = "grok-4";
    /// Grok 4 (2024-07-09) - Specific version of Grok 4
    pub const GROK_4_0709: &str = "grok-4-0709";
    /// Grok 4 Latest - Alias for latest Grok 4
    pub const GROK_4_LATEST: &str = "grok-4-latest";

    /// All Grok 4 models
    pub const ALL: &[&str] = &[
        GROK_4_1_FAST_REASONING,
        GROK_4_1_FAST_NON_REASONING,
        GROK_4_FAST_NON_REASONING,
        GROK_4_FAST_REASONING,
        GROK_4_20_0309_NON_REASONING,
        GROK_4_20_0309_REASONING,
        GROK_4_20_MULTI_AGENT_0309,
        GROK_4,
        GROK_4_0709,
        GROK_4_LATEST,
    ];
}

/// Grok 3 model family constants
pub mod grok_3 {
    /// Grok 3 - General purpose chat model
    pub const GROK_3: &str = "grok-3";
    /// Grok 3 Latest - Alias for latest Grok 3
    pub const GROK_3_LATEST: &str = "grok-3-latest";

    /// Grok 3 Mini - Lightweight version of Grok 3
    pub const GROK_3_MINI: &str = "grok-3-mini";
    /// Grok 3 Mini Latest - Alias for latest Grok 3 Mini
    pub const GROK_3_MINI_LATEST: &str = "grok-3-mini-latest";

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
        GROK_3_MINI_LATEST,
        GROK_3_FAST,
        GROK_3_FAST_LATEST,
        GROK_3_FAST_BETA,
    ];
}

/// Code-specialized Grok model constants
pub mod code {
    /// Grok Code Fast 1 - Fast Grok coding model
    pub const GROK_CODE_FAST_1: &str = "grok-code-fast-1";

    /// All code-specialized models
    pub const ALL: &[&str] = &[GROK_CODE_FAST_1];
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

/// Native image generation model constants
pub mod images {
    /// Grok Imagine Image - standard image generation model
    pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
    /// Grok Imagine Image Pro - higher-end image generation model
    pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

    /// All native image generation models
    pub const ALL: &[&str] = &[GROK_IMAGINE_IMAGE, GROK_IMAGINE_IMAGE_PRO];
}

/// Native video generation model constants
pub mod video {
    /// Grok Imagine Video - native video generation model
    pub const GROK_IMAGINE_VIDEO: &str = "grok-imagine-video";

    /// All native video generation models
    pub const ALL: &[&str] = &[GROK_IMAGINE_VIDEO];
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
    pub const FAST: &str = grok_4::GROK_4_1_FAST_NON_REASONING;
    /// Lightweight model for simple tasks
    pub const LIGHTWEIGHT: &str = grok_3::GROK_3_MINI_LATEST;
    /// Best for reasoning tasks
    pub const REASONING: &str = grok_4::GROK_4_1_FAST_REASONING;
    /// Latest and most advanced
    pub const LATEST: &str = grok_4::GROK_4_LATEST;
    /// Best for coding-heavy workflows
    pub const CODING: &str = code::GROK_CODE_FAST_1;
    /// Best for image generation
    pub const IMAGE_GENERATION: &str = images::GROK_IMAGINE_IMAGE;
    /// Best for video generation
    pub const VIDEO_GENERATION: &str = video::GROK_IMAGINE_VIDEO;
}

pub use code::GROK_CODE_FAST_1;
pub use grok_2::GROK_2;
pub use grok_3::GROK_3;
pub use grok_3::GROK_3_FAST;
pub use grok_3::GROK_3_MINI;
/// Simplified access to popular models (top-level constants)
pub use grok_4::GROK_4;
pub use images::GROK_IMAGINE_IMAGE;

/// Get all available models
pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(grok_4::ALL);
    models.extend_from_slice(grok_3::ALL);
    models.extend_from_slice(code::ALL);
    models.extend_from_slice(grok_2::ALL);
    models.extend_from_slice(images::ALL);
    models.extend_from_slice(video::ALL);
    models.extend_from_slice(legacy::ALL);
    models
}

/// Get models by capability
pub mod by_capability {
    use super::*;

    /// Models that support reasoning
    pub const REASONING: &[&str] = &[
        grok_4::GROK_4_1_FAST_REASONING,
        grok_4::GROK_4_FAST_REASONING,
        grok_4::GROK_4_20_0309_REASONING,
        grok_4::GROK_4,
        grok_4::GROK_4_0709,
        grok_4::GROK_4_LATEST,
    ];

    /// Models that support vision/image input
    pub const VISION: &[&str] = &[
        grok_4::GROK_4_1_FAST_REASONING,
        grok_4::GROK_4_1_FAST_NON_REASONING,
        grok_4::GROK_4_FAST_NON_REASONING,
        grok_4::GROK_4_FAST_REASONING,
        grok_4::GROK_4_20_0309_NON_REASONING,
        grok_4::GROK_4_20_0309_REASONING,
        grok_4::GROK_4_20_MULTI_AGENT_0309,
        grok_4::GROK_4,
        grok_4::GROK_4_0709,
        grok_4::GROK_4_LATEST,
    ];

    /// Models that support image generation
    pub const IMAGE_GENERATION: &[&str] = images::ALL;

    /// Models that support video generation
    pub const VIDEO_GENERATION: &[&str] = video::ALL;

    /// Models optimized for speed
    pub const FAST: &[&str] = &[
        grok_4::GROK_4_1_FAST_NON_REASONING,
        grok_4::GROK_4_1_FAST_REASONING,
        grok_4::GROK_4_FAST_NON_REASONING,
        grok_4::GROK_4_FAST_REASONING,
        code::GROK_CODE_FAST_1,
        grok_3::GROK_3_FAST,
        grok_3::GROK_3_FAST_LATEST,
        grok_3::GROK_3_FAST_BETA,
        grok_3::GROK_3_MINI,
        grok_3::GROK_3_MINI_LATEST,
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
        assert!(!images::GROK_IMAGINE_IMAGE.is_empty());
        assert!(!video::GROK_IMAGINE_VIDEO.is_empty());
    }

    #[test]
    fn test_all_models() {
        let models = all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&grok_4::GROK_4));
        assert!(models.contains(&grok_3::GROK_3));
        assert!(models.contains(&grok_4::GROK_4_1_FAST_REASONING));
        assert!(models.contains(&grok_3::GROK_3_MINI_LATEST));
        assert!(models.contains(&code::GROK_CODE_FAST_1));
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
        assert!(!by_capability::VIDEO_GENERATION.is_empty());
        assert!(!by_capability::FAST.is_empty());
    }
}
