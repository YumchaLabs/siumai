use super::*;

pub(super) fn guess_image_media_type_from_bytes(bytes: &[u8]) -> String {
    let guessed = crate::utils::mime::guess_mime_from_bytes(bytes);
    match guessed.as_deref() {
        Some(m) if m.starts_with("image/") => m.to_string(),
        _ => "image/jpeg".to_string(),
    }
}

pub fn build_headers(
    api_key: &str,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut builder = HttpHeaderBuilder::new()
        .with_custom_auth("x-api-key", api_key)?
        .with_json_content_type()
        .with_header("anthropic-version", "2023-06-01")?;

    if let Some(beta_features) = custom_headers.get("anthropic-beta") {
        builder = builder.with_header("anthropic-beta", beta_features)?;
    }

    let filtered_headers: std::collections::HashMap<String, String> = custom_headers
        .iter()
        .filter(|(k, _)| k.as_str() != "anthropic-beta")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    builder = builder.with_custom_headers(&filtered_headers)?;
    Ok(builder.build())
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn build_headers_includes_required_anthropic_headers() {
        let headers = build_headers("k", &std::collections::HashMap::new()).unwrap();
        assert_eq!(
            headers.get("x-api-key").and_then(|v| v.to_str().ok()),
            Some("k")
        );
        assert!(headers.contains_key("anthropic-version"));
        assert_eq!(
            headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn build_headers_preserves_anthropic_beta_header() {
        let mut custom = std::collections::HashMap::new();
        custom.insert(
            "anthropic-beta".to_string(),
            "feature-a,feature-b".to_string(),
        );
        let headers = build_headers("k", &custom).unwrap();
        assert_eq!(
            headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
            Some("feature-a,feature-b")
        );
    }

    #[test]
    fn guess_image_media_type_prefers_known_image_mime() {
        // Minimal PNG signature; infer should classify as image/png.
        let png_bytes: &[u8] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        assert_eq!(guess_image_media_type_from_bytes(png_bytes), "image/png");
    }
}
