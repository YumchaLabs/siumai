//! AI SDK-style HTTP header utility helpers.

use std::collections::BTreeMap;

use reqwest::header::HeaderMap;

/// Deterministic plain HTTP header record used by public utility helpers.
pub type HeaderRecord = BTreeMap<String, String>;

/// Normalize header entries into lower-case keys.
///
/// This mirrors AI SDK `normalizeHeaders` for Rust iterator inputs. Invalid `HeaderMap` values are
/// handled by [`normalize_header_map`], while this function is for already-valid string pairs.
pub fn normalize_headers<I, K, V>(headers: I) -> HeaderRecord
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    headers
        .into_iter()
        .map(|(key, value)| {
            (
                key.as_ref().to_ascii_lowercase(),
                value.as_ref().to_string(),
            )
        })
        .collect()
}

/// Normalize optional header entries into lower-case keys and remove `None` values.
pub fn normalize_optional_headers<I, K, V>(headers: I) -> HeaderRecord
where
    I: IntoIterator<Item = (K, Option<V>)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    headers
        .into_iter()
        .filter_map(|(key, value)| {
            value.map(|value| {
                (
                    key.as_ref().to_ascii_lowercase(),
                    value.as_ref().to_string(),
                )
            })
        })
        .collect()
}

/// Normalize a reqwest [`HeaderMap`] into lower-case UTF-8 header entries.
pub fn normalize_header_map(headers: &HeaderMap) -> HeaderRecord {
    headers
        .iter()
        .filter_map(|(key, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (key.as_str().to_ascii_lowercase(), value.to_string()))
        })
        .collect()
}

/// Combine multiple header records with later records overriding earlier ones.
pub fn combine_headers<I, M, K, V>(header_maps: I) -> HeaderRecord
where
    I: IntoIterator<Item = M>,
    M: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut combined = HeaderRecord::new();
    for headers in header_maps {
        combined.extend(normalize_headers(headers));
    }
    combined
}

/// Append suffix parts to the `user-agent` header.
///
/// Empty suffix parts are ignored. Existing header inputs are normalized before the suffix is
/// applied, matching AI SDK `withUserAgentSuffix`.
pub fn with_user_agent_suffix<I, K, V, S>(
    headers: I,
    suffix_parts: impl IntoIterator<Item = S>,
) -> HeaderRecord
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
    S: AsRef<str>,
{
    let mut normalized = normalize_headers(headers);
    let mut parts = Vec::new();

    if let Some(current) = normalized.get("user-agent")
        && !current.is_empty()
    {
        parts.push(current.clone());
    }

    parts.extend(
        suffix_parts
            .into_iter()
            .map(|part| part.as_ref().to_string())
            .filter(|part| !part.is_empty()),
    );

    normalized.insert("user-agent".to_string(), parts.join(" "));
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};

    #[test]
    fn normalize_headers_lowercases_keys() {
        let headers =
            normalize_headers([("Content-Type", "application/json"), ("X-Custom", "value")]);

        assert_eq!(
            headers.get("content-type"),
            Some(&"application/json".to_string())
        );
        assert_eq!(headers.get("x-custom"), Some(&"value".to_string()));
    }

    #[test]
    fn normalize_optional_headers_removes_missing_values() {
        let headers = normalize_optional_headers([("X-Keep", Some("yes")), ("X-Drop", None)]);

        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x-keep"), Some(&"yes".to_string()));
    }

    #[test]
    fn normalize_header_map_filters_non_utf8_values() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Ok", HeaderValue::from_static("yes"));
        headers.insert(
            HeaderName::from_static("x-binary"),
            HeaderValue::from_bytes(&[0xff]).expect("opaque header value"),
        );

        let normalized = normalize_header_map(&headers);

        assert_eq!(normalized.get("x-ok"), Some(&"yes".to_string()));
        assert!(!normalized.contains_key("x-binary"));
    }

    #[test]
    fn combine_headers_uses_later_values() {
        let first = HeaderRecord::from([
            ("x-one".to_string(), "1".to_string()),
            ("x-shared".to_string(), "old".to_string()),
        ]);
        let second = HeaderRecord::from([
            ("X-Two".to_string(), "2".to_string()),
            ("X-Shared".to_string(), "new".to_string()),
        ]);

        let combined = combine_headers([&first, &second]);

        assert_eq!(combined.get("x-one"), Some(&"1".to_string()));
        assert_eq!(combined.get("x-two"), Some(&"2".to_string()));
        assert_eq!(combined.get("x-shared"), Some(&"new".to_string()));
    }

    #[test]
    fn with_user_agent_suffix_appends_non_empty_parts() {
        let headers = with_user_agent_suffix(
            [("User-Agent", "siumai/0.1")],
            ["ai-sdk/1.2.3", "", "runtime/test"],
        );

        assert_eq!(
            headers.get("user-agent"),
            Some(&"siumai/0.1 ai-sdk/1.2.3 runtime/test".to_string())
        );
    }
}
