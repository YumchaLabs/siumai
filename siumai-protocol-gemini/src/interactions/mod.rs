//! Stable Gemini Interactions protocol mapping.

mod image;
mod language;
mod speech;
mod stream;

pub use image::{
    ImageAspectRatio, ImageDelivery, ImageMimeType, ImageResponseFormat, ImageSize,
    STABLE_V1_CREATE_TARGET, decode_image_response, encode_image_request,
};
pub use language::{
    DecodedInteraction, INTERACTIONS_CONTENT_KIND, INTERACTIONS_STEP_KIND,
    InteractionLanguageConfig, InteractionStorage, InteractionThinkingLevel,
    InteractionThinkingSummaries, STABLE_V1_LANGUAGE_TARGET, decode_interaction_resource,
    decode_language_response, encode_language_request,
};
pub use speech::{
    InteractionSpeechConfig, V1BETA_SPEECH_TARGET, decode_speech_response, encode_speech_request,
};
pub use stream::InteractionsStreamDecoder;
