//! Curated MiniMax model constants for the public provider surface.
//!
//! Chat profiles are intentionally endpoint-specific. Unknown ids remain caller-selectable, but
//! they do not receive capabilities inferred from their spelling.

/// Primary official source for the curated model catalog.
pub const MODEL_CATALOG_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/api-overview";
/// Date on which the current-model policies were verified.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-05";

/// MiniMax chat/language-model constants available through the Anthropic-compatible API.
pub mod chat {
    pub const MINIMAX_M3: &str = "MiniMax-M3";
    pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
    pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
    pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
    pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";
    pub const MINIMAX_M2_1: &str = "MiniMax-M2.1";
    pub const MINIMAX_M2_1_HIGHSPEED: &str = "MiniMax-M2.1-highspeed";
    pub const MINIMAX_M2: &str = "MiniMax-M2";

    /// Legacy id retained for source compatibility but not advertised as a current model.
    pub const MINIMAX_M2_STABLE: &str = "MiniMax-M2-Stable";
}

/// Native MiniMax chat endpoints represented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimaxiChatEndpoint {
    /// Native `/v1/text/chatcompletion_v2` API.
    NativeText,
    /// Anthropic-compatible Messages API.
    AnthropicMessages,
    /// OpenAI-compatible Chat Completions API.
    OpenAiChatCompletions,
}

/// Verified endpoint profile for one MiniMax chat model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinimaxiChatModelProfile {
    /// Provider model id.
    pub id: &'static str,
    /// Chat endpoints documented for this model.
    pub endpoints: &'static [MinimaxiChatEndpoint],
    /// Maximum combined context length documented for the model.
    pub context_window: u32,
    /// Whether provider-side thinking output is supported.
    pub supports_thinking: bool,
    /// Whether image input is documented for this model and route family.
    pub supports_vision: bool,
    /// Whether video input is documented for this model and route family.
    pub supports_video_input: bool,
}

impl MinimaxiChatModelProfile {
    /// Return whether this model is documented for the given endpoint.
    pub fn supports_endpoint(self, endpoint: MinimaxiChatEndpoint) -> bool {
        self.endpoints.contains(&endpoint)
    }
}

const NATIVE_TEXT: &[MinimaxiChatEndpoint] = &[MinimaxiChatEndpoint::NativeText];
const M2_TEXT_ENDPOINTS: &[MinimaxiChatEndpoint] = &[
    MinimaxiChatEndpoint::NativeText,
    MinimaxiChatEndpoint::AnthropicMessages,
    MinimaxiChatEndpoint::OpenAiChatCompletions,
];

/// Current MiniMax chat-model profiles.
pub const CHAT_PROFILES: &[MinimaxiChatModelProfile] = &[
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M3,
        endpoints: NATIVE_TEXT,
        context_window: 1_000_000,
        supports_thinking: true,
        supports_vision: true,
        supports_video_input: true,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_7,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_7_HIGHSPEED,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_5,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_5_HIGHSPEED,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_1,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2_1_HIGHSPEED,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxiChatModelProfile {
        id: chat::MINIMAX_M2,
        endpoints: M2_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
];

/// Return the verified endpoint profile for a current chat model id.
pub fn chat_profile(model: &str) -> Option<&'static MinimaxiChatModelProfile> {
    CHAT_PROFILES.iter().find(|candidate| candidate.id == model)
}

/// Return whether a chat model is known to support an endpoint.
pub fn supports_chat_endpoint(model: &str, endpoint: MinimaxiChatEndpoint) -> bool {
    chat_profile(model).is_some_and(|candidate| candidate.supports_endpoint(endpoint))
}

/// Return the current chat models that are documented for an endpoint.
pub fn chat_models_for_endpoint(endpoint: MinimaxiChatEndpoint) -> Vec<&'static str> {
    CHAT_PROFILES
        .iter()
        .filter(|profile| profile.supports_endpoint(endpoint))
        .map(|profile| profile.id)
        .collect()
}

/// MiniMaxi speech/TTS model constants.
pub mod speech {
    pub const SPEECH_2_8_HD: &str = "speech-2.8-hd";
    pub const SPEECH_2_8_TURBO: &str = "speech-2.8-turbo";
    pub const SPEECH_2_6_HD: &str = "speech-2.6-hd";
    pub const SPEECH_2_6_TURBO: &str = "speech-2.6-turbo";
}

/// MiniMaxi video model constants.
pub mod video {
    pub const HAILUO_2_3: &str = "MiniMax-Hailuo-2.3";
    pub const HAILUO_2_3_FAST: &str = "MiniMax-Hailuo-2.3-Fast";
}

/// MiniMaxi music model constants.
pub mod music {
    pub const MUSIC_2_6: &str = "music-2.6";
    pub const MUSIC_COVER: &str = "music-cover";
    pub const MUSIC_2_0: &str = "music-2.0";
}

/// MiniMaxi image model constants.
pub mod image {
    pub const IMAGE_01: &str = "image-01";
    pub const IMAGE_01_LIVE: &str = "image-01-live";
}

/// Current flagship model on the native text route.
pub const FLAGSHIP: &str = chat::MINIMAX_M3;
/// Current default for this crate's Anthropic-compatible chat runtime.
pub const CHAT: &str = chat::MINIMAX_M2_7;
pub const SPEECH: &str = speech::SPEECH_2_8_HD;
pub const VIDEO: &str = video::HAILUO_2_3;
pub const MUSIC: &str = music::MUSIC_2_6;
pub const IMAGE: &str = image::IMAGE_01;

pub const ALL_CHAT: &[&str] = &[
    chat::MINIMAX_M3,
    chat::MINIMAX_M2_7,
    chat::MINIMAX_M2_7_HIGHSPEED,
    chat::MINIMAX_M2_5,
    chat::MINIMAX_M2_5_HIGHSPEED,
    chat::MINIMAX_M2_1,
    chat::MINIMAX_M2_1_HIGHSPEED,
    chat::MINIMAX_M2,
];
pub const ALL_SPEECH: &[&str] = &[
    speech::SPEECH_2_8_HD,
    speech::SPEECH_2_8_TURBO,
    speech::SPEECH_2_6_HD,
    speech::SPEECH_2_6_TURBO,
];
pub const ALL_VIDEO: &[&str] = &[video::HAILUO_2_3, video::HAILUO_2_3_FAST];
pub const ALL_MUSIC: &[&str] = &[music::MUSIC_2_6, music::MUSIC_COVER, music::MUSIC_2_0];
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
    fn current_chat_catalog_separates_flagship_from_route_default() {
        assert_eq!(FLAGSHIP, chat::MINIMAX_M3);
        assert_eq!(CHAT, chat::MINIMAX_M2_7);
        assert_eq!(ALL_CHAT.len(), 8);
        assert!(ALL_CHAT.contains(&chat::MINIMAX_M2));
        assert!(!ALL_CHAT.contains(&chat::MINIMAX_M2_STABLE));
    }

    #[test]
    fn current_chat_profiles_are_endpoint_specific() {
        let m3 = chat_profile(chat::MINIMAX_M3).expect("M3 profile");
        assert!(m3.supports_endpoint(MinimaxiChatEndpoint::NativeText));
        assert!(!m3.supports_endpoint(MinimaxiChatEndpoint::AnthropicMessages));
        assert_eq!(m3.context_window, 1_000_000);
        assert!(m3.supports_vision);
        assert!(m3.supports_video_input);

        for model in &ALL_CHAT[1..] {
            let profile = chat_profile(model).expect("current MiniMax chat profile");
            assert!(profile.supports_endpoint(MinimaxiChatEndpoint::AnthropicMessages));
            assert!(profile.supports_endpoint(MinimaxiChatEndpoint::OpenAiChatCompletions));
            assert_eq!(profile.context_window, 204_800);
            assert!(!profile.supports_vision);
        }

        assert_eq!(
            chat_models_for_endpoint(MinimaxiChatEndpoint::AnthropicMessages),
            ALL_CHAT[1..]
        );
    }

    #[test]
    fn unknown_chat_models_do_not_receive_invented_endpoint_support() {
        assert!(chat_profile("MiniMax-M4-preview").is_none());
        assert!(!supports_chat_endpoint(
            "MiniMax-M4-preview",
            MinimaxiChatEndpoint::AnthropicMessages
        ));
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
