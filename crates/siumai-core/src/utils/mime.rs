//! MIME Type Detection Utilities (core)
//!
//! Provides utilities for detecting MIME types from file bytes,
//! file paths, or URLs. It supports both magic number detection
//! (via the `infer` crate) and extension-based detection.

/// Guess MIME by inspecting bytes (magic numbers)
pub fn guess_mime_from_bytes(bytes: &[u8]) -> Option<String> {
    infer::get(bytes).map(|k| k.mime_type().to_string())
}

/// Guess MIME by file path or URL (extension-based)
/// Uses a simple extension mapping for common file types
pub fn guess_mime_from_path_or_url(path_or_url: &str) -> Option<String> {
    // Extract extension from path or URL
    let extension = path_or_url
        .rsplit('.')
        .next()?
        .split('?') // Handle query parameters in URLs
        .next()?
        .to_lowercase();

    // Common MIME type mappings for file uploads
    let mime = match extension.as_str() {
        // Images
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",

        // Audio
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",

        // Video
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "avi" => "video/x-msvideo",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",

        // Documents
        "pdf" => "application/pdf",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ppt" => "application/vnd.ms-powerpoint",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",

        // Text
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" => "text/javascript",
        "json" => "application/json",
        "xml" => "application/xml",
        "csv" => "text/csv",

        // Archives
        "zip" => "application/zip",
        "tar" => "application/x-tar",
        "gz" => "application/gzip",
        "7z" => "application/x-7z-compressed",
        "rar" => "application/vnd.rar",

        _ => return None,
    };

    Some(mime.to_string())
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
