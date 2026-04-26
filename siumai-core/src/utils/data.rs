//! AI SDK-style data utility helpers.

use crate::error::LlmError;
use crate::types::ImageEditInput;
use base64::{Engine, engine::general_purpose::STANDARD};

/// Calculate cosine similarity between two numeric vectors.
///
/// Returns `0.0` for empty vectors or when either vector has zero magnitude, matching AI SDK
/// `cosineSimilarity`. Length mismatches are returned as `LlmError::InvalidParameter` instead of
/// throwing.
pub fn cosine_similarity<A, B>(vector1: &[A], vector2: &[B]) -> Result<f64, LlmError>
where
    A: Copy + Into<f64>,
    B: Copy + Into<f64>,
{
    if vector1.len() != vector2.len() {
        return Err(LlmError::InvalidParameter(format!(
            "Vectors must have the same length (vector1: {}, vector2: {})",
            vector1.len(),
            vector2.len()
        )));
    }

    if vector1.is_empty() {
        return Ok(0.0);
    }

    let mut magnitude_squared1 = 0.0;
    let mut magnitude_squared2 = 0.0;
    let mut dot_product = 0.0;

    for (value1, value2) in vector1.iter().copied().zip(vector2.iter().copied()) {
        let value1 = value1.into();
        let value2 = value2.into();
        magnitude_squared1 += value1 * value1;
        magnitude_squared2 += value2 * value2;
        dot_product += value1 * value2;
    }

    if magnitude_squared1 == 0.0 || magnitude_squared2 == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot_product / (magnitude_squared1.sqrt() * magnitude_squared2.sqrt()))
    }
}

/// Decode the text payload from a base64 data URL.
///
/// This mirrors AI SDK `getTextFromDataUrl` for `text/*`-style data URLs. The media type is
/// validated for data URL shape but not restricted, preserving upstream's permissive behavior.
pub fn get_text_from_data_url(data_url: &str) -> Result<String, LlmError> {
    let (header, base64_content) = data_url
        .split_once(',')
        .ok_or_else(|| LlmError::InvalidInput("Invalid data URL format".to_string()))?;

    let _media_type = header
        .strip_prefix("data:")
        .and_then(|value| value.split(';').next())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| LlmError::InvalidInput("Invalid data URL format".to_string()))?;

    let bytes = STANDARD
        .decode(base64_content)
        .map_err(|error| LlmError::InvalidInput(format!("Error decoding data URL: {error}")))?;

    String::from_utf8(bytes)
        .map_err(|error| LlmError::InvalidInput(format!("Decoded data URL is not UTF-8: {error}")))
}

/// Convert an image-model file input to either a URL or a data URI.
///
/// This mirrors AI SDK `convertImageModelFileToDataUri` over Siumai's
/// `ImageEditInput` carrier. URL-backed inputs are returned as-is; file-backed
/// inputs require a media type and are returned as `data:<mediaType>;base64,...`.
pub fn convert_image_model_file_to_data_uri(file: &ImageEditInput) -> Result<String, LlmError> {
    match file {
        ImageEditInput::Url { url, .. } => Ok(url.clone()),
        ImageEditInput::File {
            data, media_type, ..
        } => {
            let media_type = media_type.as_deref().ok_or_else(|| {
                LlmError::InvalidInput(
                    "Image model file input requires a media type for data URI conversion"
                        .to_string(),
                )
            })?;
            Ok(format!("data:{media_type};base64,{}", data.as_base64()))
        }
    }
}

/// Deep-equal comparison for JSON data.
///
/// Serde JSON value equality is the Rust equivalent for AI SDK `isDeepEqualData` on parsed JSON
/// data: object key order is ignored, arrays remain order-sensitive, and scalar values compare by
/// value.
pub fn is_deep_equal_data(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    left == right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_matches_ai_sdk_semantics() {
        let same = cosine_similarity(&[1.0_f32, 0.0], &[1.0_f32, 0.0])
            .expect("same vectors should compare");
        assert!((same - 1.0).abs() < f64::EPSILON);

        let orthogonal = cosine_similarity(&[1.0_f64, 0.0], &[0.0_f64, 1.0])
            .expect("orthogonal vectors should compare");
        assert!((orthogonal - 0.0).abs() < f64::EPSILON);

        let zero = cosine_similarity(&[0.0_f32, 0.0], &[1.0_f32, 1.0])
            .expect("zero vector should compare");
        assert_eq!(zero, 0.0);

        let empty = cosine_similarity::<f32, f32>(&[], &[]).expect("empty vectors should compare");
        assert_eq!(empty, 0.0);

        let err = cosine_similarity(&[1.0_f32], &[1.0_f32, 2.0])
            .expect_err("length mismatch should fail");
        assert!(matches!(err, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn data_url_text_and_deep_equal_helpers_match_ai_sdk_shape() {
        assert_eq!(
            get_text_from_data_url("data:text/plain;base64,aGVsbG8=").expect("decode data URL"),
            "hello"
        );

        assert!(get_text_from_data_url("not-a-data-url").is_err());
        assert!(is_deep_equal_data(
            &serde_json::json!({ "a": [1, true], "b": null }),
            &serde_json::json!({ "b": null, "a": [1, true] })
        ));
        assert!(!is_deep_equal_data(
            &serde_json::json!({ "a": [1, 2] }),
            &serde_json::json!({ "a": [2, 1] })
        ));
    }

    #[test]
    fn converts_image_model_file_to_data_uri() {
        assert_eq!(
            convert_image_model_file_to_data_uri(&ImageEditInput::url("https://example.com/a.png"))
                .expect("url image"),
            "https://example.com/a.png"
        );

        assert_eq!(
            convert_image_model_file_to_data_uri(&ImageEditInput::base64_with_media_type(
                "aGVsbG8=",
                "image/png",
            ))
            .expect("base64 image"),
            "data:image/png;base64,aGVsbG8="
        );

        assert_eq!(
            convert_image_model_file_to_data_uri(&ImageEditInput::file_with_media_type(
                b"hello".to_vec(),
                "image/png",
            ))
            .expect("binary image"),
            "data:image/png;base64,aGVsbG8="
        );

        assert!(
            convert_image_model_file_to_data_uri(&ImageEditInput::file(b"hello".to_vec())).is_err()
        );
    }
}
