//! AI SDK-style download helpers.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE};
use serde_json::json;

use crate::types::{DownloadError, JSONValue};

/// Default maximum download size: 2 GiB.
pub const DEFAULT_MAX_DOWNLOAD_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Options for [`create_download`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DownloadOptions {
    /// Maximum allowed download size in bytes.
    ///
    /// Defaults to [`DEFAULT_MAX_DOWNLOAD_SIZE`].
    pub max_bytes: Option<u64>,
}

/// Downloaded asset payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadedFile {
    /// Downloaded bytes.
    pub data: Vec<u8>,

    /// Response or data URL media type when available.
    pub media_type: Option<String>,
}

/// Configured AI SDK-style download helper.
#[derive(Debug, Clone)]
pub struct Download {
    client: reqwest::Client,
    max_bytes: u64,
}

impl Download {
    /// Create a download helper with the default HTTP client.
    pub fn new(options: DownloadOptions) -> Self {
        Self::with_client(reqwest::Client::new(), options)
    }

    /// Create a download helper with a caller-provided HTTP client.
    pub fn with_client(client: reqwest::Client, options: DownloadOptions) -> Self {
        Self {
            client,
            max_bytes: options.max_bytes.unwrap_or(DEFAULT_MAX_DOWNLOAD_SIZE),
        }
    }

    /// Download a URL, enforcing AI SDK-style URL validation and response size limits.
    pub async fn download(&self, url: impl AsRef<str>) -> Result<DownloadedFile, DownloadError> {
        download_with_client(&self.client, url.as_ref(), self.max_bytes).await
    }
}

/// Create a configured download helper.
pub fn create_download(options: DownloadOptions) -> Download {
    Download::new(options)
}

/// Download a URL using default options and the default HTTP client.
pub async fn download_url(url: impl AsRef<str>) -> Result<DownloadedFile, DownloadError> {
    create_download(DownloadOptions::default())
        .download(url.as_ref())
        .await
}

/// Validate that a URL is safe to download from.
///
/// This mirrors AI SDK `validateDownloadUrl`: `data:` is allowed as inline content, `http` and
/// `https` are allowed when the hostname is not localhost/private/link-local, and other schemes
/// are rejected.
pub fn validate_download_url(url: &str) -> Result<(), DownloadError> {
    let parsed =
        reqwest::Url::parse(url).map_err(|_| download_error(url, format!("Invalid URL: {url}")))?;

    if parsed.scheme() == "data" {
        return Ok(());
    }

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(download_error(
            url,
            format!(
                "URL scheme must be http, https, or data, got {}:",
                parsed.scheme()
            ),
        ));
    }

    let hostname = parsed
        .host_str()
        .ok_or_else(|| download_error(url, "URL must have a hostname"))?;
    let hostname = hostname.trim_start_matches('[').trim_end_matches(']');
    let hostname_lower = hostname.to_ascii_lowercase();

    if hostname_lower == "localhost"
        || hostname_lower.ends_with(".local")
        || hostname_lower.ends_with(".localhost")
    {
        return Err(download_error(
            url,
            format!("URL with hostname {hostname} is not allowed"),
        ));
    }

    if let Ok(ip) = hostname.parse::<IpAddr>() {
        match ip {
            IpAddr::V4(ipv4) if is_private_ipv4(ipv4) => {
                return Err(download_error(
                    url,
                    format!("URL with IP address {hostname} is not allowed"),
                ));
            }
            IpAddr::V6(ipv6) if is_private_ipv6(ipv6) => {
                return Err(download_error(
                    url,
                    format!("URL with IPv6 address {hostname} is not allowed"),
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

async fn download_with_client(
    client: &reqwest::Client,
    url: &str,
    max_bytes: u64,
) -> Result<DownloadedFile, DownloadError> {
    validate_download_url(url)?;

    if url.starts_with("data:") {
        return download_data_url(url, max_bytes);
    }

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| DownloadError::from_cause(url, json!(error.to_string())))?;

    let final_url = response.url().to_string();
    if final_url != url {
        validate_download_url(&final_url)?;
    }

    if !response.status().is_success() {
        let status = response.status();
        return Err(DownloadError::from_status(
            url,
            status.as_u16(),
            status.canonical_reason().unwrap_or_default(),
        ));
    }

    read_response_with_size_limit(response, url, max_bytes).await
}

/// Read a response body with a byte limit.
pub async fn read_response_with_size_limit(
    response: reqwest::Response,
    url: &str,
    max_bytes: u64,
) -> Result<DownloadedFile, DownloadError> {
    if let Some(content_length) = response.headers().get(CONTENT_LENGTH)
        && let Ok(content_length) = content_length.to_str()
        && let Ok(content_length) = content_length.parse::<u64>()
        && content_length > max_bytes
    {
        return Err(download_error(
            url,
            format!(
                "Download of {url} exceeded maximum size of {max_bytes} bytes (Content-Length: {content_length})."
            ),
        ));
    }

    let media_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    let mut stream = response.bytes_stream();
    let mut data = Vec::new();
    let mut total_bytes = 0_u64;

    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|error| DownloadError::from_cause(url, json!(error.to_string())))?;
        total_bytes = total_bytes.saturating_add(chunk.len() as u64);
        if total_bytes > max_bytes {
            return Err(download_error(
                url,
                format!("Download of {url} exceeded maximum size of {max_bytes} bytes."),
            ));
        }
        data.extend_from_slice(&chunk);
    }

    Ok(DownloadedFile { data, media_type })
}

fn download_data_url(url: &str, max_bytes: u64) -> Result<DownloadedFile, DownloadError> {
    let (header, payload) = url
        .split_once(',')
        .ok_or_else(|| download_error(url, "Invalid data URL format"))?;
    let header = header
        .strip_prefix("data:")
        .ok_or_else(|| download_error(url, "Invalid data URL format"))?;

    let mut media_type = None;
    let mut is_base64 = false;
    for (index, part) in header.split(';').enumerate() {
        if index == 0 && !part.is_empty() {
            media_type = Some(part.to_string());
        } else if part.eq_ignore_ascii_case("base64") {
            is_base64 = true;
        }
    }

    let data = if is_base64 {
        STANDARD
            .decode(payload)
            .map_err(|error| DownloadError::from_cause(url, json!(error.to_string())))?
    } else {
        percent_decode_bytes(payload).map_err(|message| download_error(url, message))?
    };

    if data.len() as u64 > max_bytes {
        return Err(download_error(
            url,
            format!("Download of {url} exceeded maximum size of {max_bytes} bytes."),
        ));
    }

    Ok(DownloadedFile { data, media_type })
}

fn percent_decode_bytes(input: &str) -> Result<Vec<u8>, String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes
                .get(index + 1)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| "Invalid percent-encoded data URL payload".to_string())?;
            let low = bytes
                .get(index + 2)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| "Invalid percent-encoded data URL payload".to_string())?;
            out.push((high << 4) | low);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }

    Ok(out)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    a == 0
        || a == 10
        || a == 127
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
}

fn is_private_ipv6(ip: Ipv6Addr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() {
        return true;
    }

    if let Some(mapped) = ip.to_ipv4_mapped() {
        return is_private_ipv4(mapped);
    }

    let segments = ip.segments();
    (segments[0] & 0xfe00) == 0xfc00 || (segments[0] & 0xffc0) == 0xfe80
}

fn download_error(url: &str, message: impl Into<String>) -> DownloadError {
    DownloadError {
        message: message.into(),
        url: url.to_string(),
        status_code: None,
        status_text: None,
        cause: None::<JSONValue>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_download_url_matches_ai_sdk_safety_rules() {
        assert!(validate_download_url("https://example.com/image.png").is_ok());
        assert!(validate_download_url("http://example.com/image.png").is_ok());
        assert!(validate_download_url("https://203.0.113.1/file").is_ok());
        assert!(validate_download_url("https://example.com:8080/file").is_ok());
        assert!(validate_download_url("data:text/plain;base64,aGVsbG8=").is_ok());

        assert!(validate_download_url("file:///etc/passwd").is_err());
        assert!(validate_download_url("ftp://example.com/file").is_err());
        assert!(validate_download_url("javascript:alert(1)").is_err());
        assert!(validate_download_url("not-a-url").is_err());
        assert!(validate_download_url("http://localhost/file").is_err());
        assert!(validate_download_url("http://myhost.local/file").is_err());
        assert!(validate_download_url("http://app.localhost/file").is_err());
        assert!(validate_download_url("http://127.0.0.1/file").is_err());
        assert!(validate_download_url("http://10.0.0.1/file").is_err());
        assert!(validate_download_url("http://172.16.0.1/file").is_err());
        assert!(validate_download_url("http://172.31.255.255/file").is_err());
        assert!(validate_download_url("http://172.15.0.1/file").is_ok());
        assert!(validate_download_url("http://172.32.0.1/file").is_ok());
        assert!(validate_download_url("http://192.168.1.1/file").is_err());
        assert!(validate_download_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_download_url("http://0.0.0.0/file").is_err());
        assert!(validate_download_url("http://[::1]/file").is_err());
        assert!(validate_download_url("http://[::]/file").is_err());
        assert!(validate_download_url("http://[fc00::1]/file").is_err());
        assert!(validate_download_url("http://[fd12::1]/file").is_err());
        assert!(validate_download_url("http://[fe80::1]/file").is_err());
        assert!(validate_download_url("http://[::ffff:127.0.0.1]/file").is_err());
        assert!(validate_download_url("http://[::ffff:10.0.0.1]/file").is_err());
        assert!(validate_download_url("http://[::ffff:169.254.169.254]/file").is_err());
        assert!(validate_download_url("http://[::ffff:203.0.113.1]/file").is_ok());
    }

    #[tokio::test]
    async fn download_data_url_decodes_bytes_and_media_type() {
        let result = create_download(DownloadOptions {
            max_bytes: Some(16),
        })
        .download("data:text/plain;base64,aGVsbG8=")
        .await
        .expect("download data URL");

        assert_eq!(result.data, b"hello");
        assert_eq!(result.media_type.as_deref(), Some("text/plain"));
    }

    #[tokio::test]
    async fn download_data_url_enforces_size_limit() {
        let err = create_download(DownloadOptions { max_bytes: Some(4) })
            .download("data:text/plain;base64,aGVsbG8=")
            .await
            .expect_err("size limit should fail");

        assert!(err.message.contains("exceeded maximum size"));
    }

    #[tokio::test]
    async fn download_data_url_decodes_percent_payloads() {
        let result = download_url("data:text/plain,hello%20world")
            .await
            .expect("download percent data URL");

        assert_eq!(result.data, b"hello world");
        assert_eq!(result.media_type.as_deref(), Some("text/plain"));
    }
}
