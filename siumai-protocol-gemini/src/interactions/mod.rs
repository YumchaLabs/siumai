//! Stable Gemini Interactions protocol mapping.

mod image;

pub use image::{
    ImageAspectRatio, ImageDelivery, ImageMimeType, ImageResponseFormat, ImageSize,
    STABLE_V1_CREATE_TARGET, decode_image_response, encode_image_request,
};
