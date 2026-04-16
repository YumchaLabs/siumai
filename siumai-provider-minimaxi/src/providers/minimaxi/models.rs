//! Curated MiniMaxi model constants for the public provider surface.
//!
//! This module sits above the broader `model_constants` compatibility layer and exposes the
//! stable model-family surface that the facade and provider catalog should share.

/// MiniMaxi chat/language-model constants.
pub mod chat {
    pub const MINIMAX_M2: &str = super::super::model_constants::text::MINIMAX_M2;
    pub const MINIMAX_M2_STABLE: &str = super::super::model_constants::text::MINIMAX_M2_STABLE;
}

/// MiniMaxi speech/TTS model constants.
pub mod speech {
    pub const SPEECH_2_6_HD: &str = super::super::model_constants::audio::SPEECH_2_6_HD;
    pub const SPEECH_2_6_TURBO: &str = super::super::model_constants::audio::SPEECH_2_6_TURBO;
}

/// MiniMaxi video model constants.
pub mod video {
    pub const HAILUO_2_3: &str = super::super::model_constants::video::HAILUO_2_3;
    pub const HAILUO_2_3_FAST: &str = super::super::model_constants::video::HAILUO_2_3_FAST;
}

/// MiniMaxi music model constants.
pub mod music {
    pub const MUSIC_2_0: &str = super::super::model_constants::music::MUSIC_2_0;
}

/// MiniMaxi image model constants.
pub mod image {
    pub const IMAGE_01: &str = super::super::model_constants::images::IMAGE_01;
    pub const IMAGE_01_LIVE: &str = super::super::model_constants::images::IMAGE_01_LIVE;
}

pub const CHAT: &str = chat::MINIMAX_M2;
pub const SPEECH: &str = speech::SPEECH_2_6_HD;
pub const VIDEO: &str = video::HAILUO_2_3;
pub const MUSIC: &str = music::MUSIC_2_0;
pub const IMAGE: &str = image::IMAGE_01;

pub const ALL_CHAT: &[&str] = &[chat::MINIMAX_M2, chat::MINIMAX_M2_STABLE];
pub const ALL_SPEECH: &[&str] = &[speech::SPEECH_2_6_HD, speech::SPEECH_2_6_TURBO];
pub const ALL_VIDEO: &[&str] = &[video::HAILUO_2_3, video::HAILUO_2_3_FAST];
pub const ALL_MUSIC: &[&str] = &[music::MUSIC_2_0];
pub const ALL_IMAGE: &[&str] = &[image::IMAGE_01, image::IMAGE_01_LIVE];

pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::with_capacity(
        ALL_CHAT.len() + ALL_SPEECH.len() + ALL_VIDEO.len() + ALL_MUSIC.len() + ALL_IMAGE.len(),
    );
    models.extend_from_slice(ALL_CHAT);
    models.extend_from_slice(ALL_SPEECH);
    models.extend_from_slice(ALL_VIDEO);
    models.extend_from_slice(ALL_MUSIC);
    models.extend_from_slice(ALL_IMAGE);
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_defaults_match_primary_families() {
        assert_eq!(CHAT, chat::MINIMAX_M2);
        assert_eq!(SPEECH, speech::SPEECH_2_6_HD);
        assert_eq!(VIDEO, video::HAILUO_2_3);
        assert_eq!(MUSIC, music::MUSIC_2_0);
        assert_eq!(IMAGE, image::IMAGE_01);
    }

    #[test]
    fn curated_lists_cover_primary_defaults() {
        assert!(ALL_CHAT.contains(&CHAT));
        assert!(ALL_SPEECH.contains(&SPEECH));
        assert!(ALL_VIDEO.contains(&VIDEO));
        assert!(ALL_MUSIC.contains(&MUSIC));
        assert!(ALL_IMAGE.contains(&IMAGE));
        assert!(all_models().contains(&IMAGE));
    }
}
