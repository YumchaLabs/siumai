//! MIME Type Detection Utilities
//!
//! This module provides utilities for detecting MIME types from file bytes,
//! file paths, or URLs. It supports both magic number detection (via the `infer` crate)
//! and extension-based detection.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_mime_from_path() {
        // Test common image formats
        assert_eq!(
            guess_mime_from_path_or_url("image.jpg"),
            Some("image/jpeg".to_string())
        );
        assert_eq!(
            guess_mime_from_path_or_url("photo.png"),
            Some("image/png".to_string())
        );
        assert_eq!(
            guess_mime_from_path_or_url("icon.webp"),
            Some("image/webp".to_string())
        );

        // Test with URL and query parameters
        assert_eq!(
            guess_mime_from_path_or_url("https://example.com/file.pdf?v=1"),
            Some("application/pdf".to_string())
        );

        // Test audio/video
        assert_eq!(
            guess_mime_from_path_or_url("song.mp3"),
            Some("audio/mpeg".to_string())
        );
        assert_eq!(
            guess_mime_from_path_or_url("video.mp4"),
            Some("video/mp4".to_string())
        );

        // Test unknown extension
        assert_eq!(guess_mime_from_path_or_url("file.unknown"), None);

        // Test case insensitivity
        assert_eq!(
            guess_mime_from_path_or_url("IMAGE.JPG"),
            Some("image/jpeg".to_string())
        );
    }

    #[test]
    fn test_guess_mime_combined() {
        // Test with bytes (should prefer bytes over extension)
        let png_bytes = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let mime = guess_mime(Some(png_bytes), Some("file.jpg"));
        assert_eq!(mime, "image/png"); // Should detect PNG from bytes, not JPG from extension

        // Test with only path
        let mime = guess_mime(None, Some("document.pdf"));
        assert_eq!(mime, "application/pdf");

        // Test with neither
        let mime = guess_mime(None, None);
        assert_eq!(mime, "application/octet-stream");

        // Test with unknown extension
        let mime = guess_mime(None, Some("file.xyz"));
        assert_eq!(mime, "application/octet-stream");
    }
}
