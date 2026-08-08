//! Stable v1 wire mapping for Google's Legacy Generate Content API mode.
//!
//! "Legacy" describes Google's current product posture. The wire endpoints in
//! this module remain published by the stable v1 discovery document.

mod language;
mod stream;

pub use language::{
    DecodedGenerateContent, GENERATE_CONTENT_PART_KIND, GenerateContentLanguageConfig,
    GenerateContentServiceTier, GenerateContentThinkingConfig, GenerateContentThinkingLevel,
    LEGACY_STABLE_V1_GENERATE_CONTENT_TARGET, LEGACY_STABLE_V1_STREAM_GENERATE_CONTENT_TARGET,
    decode_language_response, encode_language_request,
};
pub use stream::GenerateContentStreamDecoder;
