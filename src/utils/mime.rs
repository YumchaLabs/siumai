//! MIME type detection utilities

/// Guess MIME by inspecting bytes (magic numbers)
pub fn guess_mime_from_bytes(bytes: &[u8]) -> Option<String> {
    infer::get(bytes).map(|k| k.mime_type().to_string())
}

/// Guess MIME by file path or URL (extension-based)
pub fn guess_mime_from_path_or_url(path_or_url: &str) -> Option<String> {
    mime_guess::from_path(path_or_url)
        .first_raw()
        .map(|s| s.to_string())
}

/// Combined guess: prefer bytes, fall back to extension, otherwise octet-stream
pub fn guess_mime(bytes: Option<&[u8]>, path_or_url: Option<&str>) -> String {
    if let Some(b) = bytes
        && let Some(m) = guess_mime_from_bytes(b)
    {
        return m;
    }
    if let Some(p) = path_or_url
        && let Some(m) = guess_mime_from_path_or_url(p)
    {
        return m;
    }
    "application/octet-stream".to_string()
}
