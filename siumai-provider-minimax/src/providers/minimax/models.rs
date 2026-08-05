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
}

/// Native MiniMax chat endpoints represented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxChatEndpoint {
    /// Native `/v1/text/chatcompletion_v2` API.
    NativeText,
    /// Anthropic-compatible Messages API.
    AnthropicMessages,
    /// OpenAI-compatible Chat Completions API.
    OpenAiChatCompletions,
}

/// Thinking behavior documented for a model on a specific chat endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimaxThinkingPolicy {
    /// Thinking is always enabled; a disable request is accepted remotely but ignored.
    AlwaysOn,
    /// Adaptive thinking can be enabled or disabled, with the documented route default.
    OptionalAdaptive {
        /// Whether omitting `thinking` enables it by default. `None` means undocumented.
        default_enabled: Option<bool>,
    },
}

/// Verified endpoint profile for one MiniMax chat model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinimaxChatModelProfile {
    /// Provider model id.
    pub id: &'static str,
    /// Chat endpoints documented for this model.
    pub endpoints: &'static [MinimaxChatEndpoint],
    /// Maximum combined context length documented for the model.
    pub context_window: u32,
    /// Whether provider-side thinking output is supported.
    pub supports_thinking: bool,
    /// Whether image input is documented for this model and route family.
    pub supports_vision: bool,
    /// Whether video input is documented for this model and route family.
    pub supports_video_input: bool,
}

impl MinimaxChatModelProfile {
    /// Return whether this model is documented for the given endpoint.
    pub fn supports_endpoint(self, endpoint: MinimaxChatEndpoint) -> bool {
        self.endpoints.contains(&endpoint)
    }
}

const CURRENT_TEXT_ENDPOINTS: &[MinimaxChatEndpoint] = &[
    MinimaxChatEndpoint::NativeText,
    MinimaxChatEndpoint::AnthropicMessages,
    MinimaxChatEndpoint::OpenAiChatCompletions,
];

/// Current MiniMax chat-model profiles.
pub const CHAT_PROFILES: &[MinimaxChatModelProfile] = &[
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M3,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 1_000_000,
        supports_thinking: true,
        supports_vision: true,
        supports_video_input: true,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_7,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_7_HIGHSPEED,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_5,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_5_HIGHSPEED,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_1,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2_1_HIGHSPEED,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
    MinimaxChatModelProfile {
        id: chat::MINIMAX_M2,
        endpoints: CURRENT_TEXT_ENDPOINTS,
        context_window: 204_800,
        supports_thinking: true,
        supports_vision: false,
        supports_video_input: false,
    },
];

/// Return the verified endpoint profile for a current chat model id.
pub fn chat_profile(model: &str) -> Option<&'static MinimaxChatModelProfile> {
    CHAT_PROFILES.iter().find(|candidate| candidate.id == model)
}

/// Return whether a chat model is known to support an endpoint.
pub fn supports_chat_endpoint(model: &str, endpoint: MinimaxChatEndpoint) -> bool {
    chat_profile(model).is_some_and(|candidate| candidate.supports_endpoint(endpoint))
}

/// Return the documented thinking policy for a known model and endpoint.
pub fn thinking_policy(
    model: &str,
    endpoint: MinimaxChatEndpoint,
) -> Option<MinimaxThinkingPolicy> {
    let profile = chat_profile(model)?;
    if !profile.supports_endpoint(endpoint) {
        return None;
    }

    if model == chat::MINIMAX_M3 {
        let default_enabled = match endpoint {
            MinimaxChatEndpoint::AnthropicMessages => Some(false),
            MinimaxChatEndpoint::OpenAiChatCompletions => Some(true),
            MinimaxChatEndpoint::NativeText => None,
        };
        Some(MinimaxThinkingPolicy::OptionalAdaptive { default_enabled })
    } else {
        Some(MinimaxThinkingPolicy::AlwaysOn)
    }
}

/// Return the current chat models that are documented for an endpoint.
pub fn chat_models_for_endpoint(endpoint: MinimaxChatEndpoint) -> Vec<&'static str> {
    CHAT_PROFILES
        .iter()
        .filter(|profile| profile.supports_endpoint(endpoint))
        .map(|profile| profile.id)
        .collect()
}

/// MiniMax speech/TTS model constants.
pub mod speech {
    pub const SPEECH_2_8_HD: &str = "speech-2.8-hd";
    pub const SPEECH_2_8_TURBO: &str = "speech-2.8-turbo";
    pub const SPEECH_2_6_HD: &str = "speech-2.6-hd";
    pub const SPEECH_2_6_TURBO: &str = "speech-2.6-turbo";
    pub const SPEECH_02_HD: &str = "speech-02-hd";
    pub const SPEECH_02_TURBO: &str = "speech-02-turbo";
}

/// MiniMax video model constants.
pub mod video {
    pub const HAILUO_2_3: &str = "MiniMax-Hailuo-2.3";
    pub const HAILUO_2_3_FAST: &str = "MiniMax-Hailuo-2.3-Fast";
    pub const HAILUO_02: &str = "MiniMax-Hailuo-02";
    pub const T2V_01_DIRECTOR: &str = "T2V-01-Director";
    pub const T2V_01: &str = "T2V-01";
}

/// MiniMax music model constants.
pub mod music {
    pub const MUSIC_2_6: &str = "music-2.6";
    pub const MUSIC_2_6_FREE: &str = "music-2.6-free";
    pub const MUSIC_COVER: &str = "music-cover";
    pub const MUSIC_COVER_FREE: &str = "music-cover-free";
}

/// MiniMax image model constants.
pub mod image {
    pub const IMAGE_01: &str = "image-01";
    pub const IMAGE_01_LIVE: &str = "image-01-live";
}

/// Current flagship model on the native text route.
pub const FLAGSHIP: &str = chat::MINIMAX_M3;
/// Current default for this crate's chat runtime.
pub const CHAT: &str = chat::MINIMAX_M3;
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
    speech::SPEECH_02_HD,
    speech::SPEECH_02_TURBO,
];
pub const ALL_VIDEO: &[&str] = &[
    video::HAILUO_2_3,
    video::HAILUO_2_3_FAST,
    video::HAILUO_02,
    video::T2V_01_DIRECTOR,
    video::T2V_01,
];
pub const ALL_MUSIC: &[&str] = &[
    music::MUSIC_2_6,
    music::MUSIC_2_6_FREE,
    music::MUSIC_COVER,
    music::MUSIC_COVER_FREE,
];
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
        assert_eq!(CHAT, chat::MINIMAX_M3);
        assert_eq!(ALL_CHAT.len(), 8);
        assert!(ALL_CHAT.contains(&chat::MINIMAX_M2));
    }

    #[test]
    fn current_chat_profiles_are_endpoint_specific() {
        let m3 = chat_profile(chat::MINIMAX_M3).expect("M3 profile");
        assert!(m3.supports_endpoint(MinimaxChatEndpoint::NativeText));
        assert!(m3.supports_endpoint(MinimaxChatEndpoint::AnthropicMessages));
        assert!(m3.supports_endpoint(MinimaxChatEndpoint::OpenAiChatCompletions));
        assert_eq!(m3.context_window, 1_000_000);
        assert!(m3.supports_vision);
        assert!(m3.supports_video_input);

        for model in &ALL_CHAT[1..] {
            let profile = chat_profile(model).expect("current MiniMax chat profile");
            assert!(profile.supports_endpoint(MinimaxChatEndpoint::AnthropicMessages));
            assert!(profile.supports_endpoint(MinimaxChatEndpoint::OpenAiChatCompletions));
            assert_eq!(profile.context_window, 204_800);
            assert!(!profile.supports_vision);
        }

        assert_eq!(
            chat_models_for_endpoint(MinimaxChatEndpoint::AnthropicMessages),
            ALL_CHAT
        );

        assert_eq!(
            thinking_policy(chat::MINIMAX_M3, MinimaxChatEndpoint::AnthropicMessages),
            Some(MinimaxThinkingPolicy::OptionalAdaptive {
                default_enabled: Some(false)
            })
        );
        assert_eq!(
            thinking_policy(chat::MINIMAX_M3, MinimaxChatEndpoint::OpenAiChatCompletions),
            Some(MinimaxThinkingPolicy::OptionalAdaptive {
                default_enabled: Some(true)
            })
        );
        assert_eq!(
            thinking_policy(chat::MINIMAX_M2_7, MinimaxChatEndpoint::AnthropicMessages),
            Some(MinimaxThinkingPolicy::AlwaysOn)
        );
    }

    #[test]
    fn unknown_chat_models_do_not_receive_invented_endpoint_support() {
        assert!(chat_profile("MiniMax-M4-preview").is_none());
        assert!(!supports_chat_endpoint(
            "MiniMax-M4-preview",
            MinimaxChatEndpoint::AnthropicMessages
        ));
    }

    #[test]
    fn curated_lists_cover_primary_defaults() {
        assert!(ALL_CHAT.contains(&CHAT));
        assert!(ALL_SPEECH.contains(&SPEECH));
        assert!(ALL_VIDEO.contains(&VIDEO));
        assert!(ALL_MUSIC.contains(&MUSIC));
        assert!(ALL_MUSIC.contains(&music::MUSIC_2_6_FREE));
        assert!(ALL_MUSIC.contains(&music::MUSIC_COVER_FREE));
        assert!(ALL_IMAGE.contains(&IMAGE));
        assert!(all_models().contains(&IMAGE));
    }
}
