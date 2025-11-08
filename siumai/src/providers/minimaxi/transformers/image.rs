//! MiniMaxi Image Generation Adapter
//!
//! Adapts MiniMaxi's image generation response format to OpenAI standard format.

use crate::error::LlmError;
use crate::standards::openai::image::OpenAiImageAdapter;
use std::sync::Arc;

/// MiniMaxi Image Adapter
///
/// Transforms MiniMaxi's image response format:
/// ```json
/// {
///   "data": {
///     "image_urls": ["url1", "url2"]
///   }
/// }
/// ```
///
/// To OpenAI standard format:
/// ```json
/// {
///   "data": [
///     {"url": "url1"},
///     {"url": "url2"}
///   ]
/// }
/// ```
#[derive(Clone)]
pub struct MinimaxiImageAdapter;

impl OpenAiImageAdapter for MinimaxiImageAdapter {
    fn transform_response(&self, resp: &mut serde_json::Value) -> Result<(), LlmError> {
        // Check for API errors first
        if let Some(base_resp) = resp.get("base_resp")
            && let Some(status_code) = base_resp.get("status_code").and_then(|v| v.as_i64())
            && status_code != 0
        {
            let status_msg = base_resp
                .get("status_msg")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");

            let error_hint = match status_code {
                2013 => " Supported models: image-01, image-01-live",
                2054 => " Please check the voice ID or other parameters",
                _ => "",
            };

            return Err(LlmError::provider_error_with_code(
                "minimaxi",
                format!("{}.{}", status_msg, error_hint),
                status_code.to_string(),
            ));
        }

        // Check if response has MiniMaxi format: data.image_urls
        if let Some(data) = resp.get("data")
            && let Some(image_urls) = data.get("image_urls")
            && let Some(urls) = image_urls.as_array()
        {
            // Transform to OpenAI format
            let openai_data: Vec<serde_json::Value> = urls
                .iter()
                .filter_map(|url| url.as_str())
                .map(|url| serde_json::json!({"url": url}))
                .collect();

            *resp = serde_json::json!({
                "data": openai_data
            });
        }

        Ok(())
    }
}

/// Create OpenAI Image Standard with MiniMaxi adapter
pub fn create_minimaxi_image_standard() -> crate::standards::openai::image::OpenAiImageStandard {
    crate::standards::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
        MinimaxiImageAdapter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimaxi_image_adapter() {
        let adapter = MinimaxiImageAdapter;

        // MiniMaxi response format
        let mut response = serde_json::json!({
            "id": "test-id",
            "data": {
                "image_urls": [
                    "https://example.com/image1.png",
                    "https://example.com/image2.png"
                ]
            },
            "base_resp": {
                "status_code": 0,
                "status_msg": "success"
            }
        });

        // Transform
        adapter.transform_response(&mut response).unwrap();

        // Verify OpenAI format
        assert!(response.get("data").is_some());
        let data = response.get("data").unwrap().as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(
            data[0].get("url").unwrap().as_str().unwrap(),
            "https://example.com/image1.png"
        );
        assert_eq!(
            data[1].get("url").unwrap().as_str().unwrap(),
            "https://example.com/image2.png"
        );
    }
}
