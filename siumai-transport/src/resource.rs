//! Structurally unauthenticated resource downloads with per-hop validation.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::read::DecoderReader;
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use http::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, LOCATION};
use reqwest::Url;
use siumai_core::{
    Cancellation, Error, ErrorKind, ResponseDiagnostics, SafeResponseHeaders, SensitiveResponse,
};
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::endpoint::{EndpointConfig, Resolver, SystemResolver};
use crate::transport::{
    ResponseHeaders, bounded_body_prefix, build_guarded_client, response_limit_error,
    run_controlled, transport_source_error,
};
use crate::{EndpointError, EndpointPolicy, TransportConfigError, TransportLimits};

const MAX_NETWORK_RESOURCE_URL_BYTES: usize = 16 * 1024;
const MAX_DATA_RESOURCE_URL_BYTES: usize = 256 * 1024 * 1024;
const DATA_URL_METADATA_BYTES: usize = 1024;
const DECODE_CHECK_BYTES: usize = 64 * 1024;
const ERROR_BODY_CAPTURE_BYTES: usize = 64 * 1024;

/// Invalid resource URL. Raw URLs are intentionally absent from diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ResourceUrlError {
    #[error("resource URL is invalid")]
    Invalid,
    #[error("resource URL scheme is not allowed")]
    SchemeNotAllowed,
    #[error("resource URL exceeds the hard input limit")]
    TooLong,
    #[error("resource URL failed endpoint security validation")]
    Endpoint,
}

/// Validated public/local network resource or bounded-at-read inline data URL.
#[derive(Clone)]
pub struct ResourceUrl {
    url: Url,
    policy: EndpointPolicy,
}

impl ResourceUrl {
    pub fn public(value: impl AsRef<str>) -> Result<Self, ResourceUrlError> {
        Self::new(value.as_ref(), EndpointPolicy::PublicCustom)
    }

    pub fn local_explicit(value: impl AsRef<str>) -> Result<Self, ResourceUrlError> {
        Self::new(value.as_ref(), EndpointPolicy::LocalExplicit)
    }

    fn new(value: &str, policy: EndpointPolicy) -> Result<Self, ResourceUrlError> {
        let is_data = value
            .get(.."data:".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"));
        let maximum = if is_data {
            MAX_DATA_RESOURCE_URL_BYTES
        } else {
            MAX_NETWORK_RESOURCE_URL_BYTES
        };
        if value.len() > maximum {
            return Err(ResourceUrlError::TooLong);
        }
        if is_data && !value.is_ascii() {
            return Err(ResourceUrlError::Invalid);
        }
        let url = Url::parse(value).map_err(|_| ResourceUrlError::Invalid)?;
        if url.scheme() == "data" {
            if url.fragment().is_some() {
                return Err(ResourceUrlError::Invalid);
            }
            return Ok(Self { url, policy });
        }
        if !matches!(url.scheme(), "http" | "https")
            || matches!(
                policy,
                EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom
            ) && url.scheme() != "https"
        {
            return Err(ResourceUrlError::SchemeNotAllowed);
        }
        EndpointConfig::new(value, policy.clone()).map_err(|_| ResourceUrlError::Endpoint)?;
        Ok(Self { url, policy })
    }

    pub fn scheme(&self) -> &str {
        self.url.scheme()
    }

    /// Explicit access for callers that must persist provenance. Avoid logging
    /// the result because network URLs may have signed query parameters.
    pub fn expose(&self) -> &Url {
        &self.url
    }
}

impl fmt::Debug for ResourceUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceUrl")
            .field("scheme", &self.url.scheme())
            .field("policy", &self.policy)
            .field("url", &"[REDACTED]")
            .finish()
    }
}

/// Cancellation and deadline for one resource fetch.
#[derive(Clone, Default)]
pub struct ResourceDownloadOptions {
    deadline: Option<Instant>,
    cancellation: Cancellation,
}

impl ResourceDownloadOptions {
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_cancellation(mut self, cancellation: Cancellation) -> Self {
        self.cancellation = cancellation;
        self
    }
}

impl fmt::Debug for ResourceDownloadOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceDownloadOptions")
            .field("deadline", &self.deadline)
            .field("cancellation", &self.cancellation)
            .finish()
    }
}

/// Downloaded bytes and media evidence. No URL or payload is printed by default.
pub struct DownloadedResource {
    data: Bytes,
    declared_media_type: Option<String>,
    detected_media_type: Option<String>,
    final_url: ResourceUrl,
}

impl DownloadedResource {
    pub fn data(&self) -> &Bytes {
        &self.data
    }

    pub fn declared_media_type(&self) -> Option<&str> {
        self.declared_media_type.as_deref()
    }

    pub fn detected_media_type(&self) -> Option<&str> {
        self.detected_media_type.as_deref()
    }

    pub fn final_url(&self) -> &ResourceUrl {
        &self.final_url
    }
}

impl fmt::Debug for DownloadedResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DownloadedResource")
            .field("data_bytes", &self.data.len())
            .field("data", &"[REDACTED]")
            .field(
                "has_declared_media_type",
                &self.declared_media_type.is_some(),
            )
            .field("detected_media_type", &self.detected_media_type)
            .field("final_url", &self.final_url)
            .finish()
    }
}

/// Builder for a resource downloader. There is intentionally no auth or
/// external-client injection point.
pub struct ResourceDownloaderBuilder {
    resolver: Arc<dyn Resolver>,
    limits: TransportLimits,
    connect_timeout: Duration,
    read_timeout: Duration,
}

impl ResourceDownloaderBuilder {
    pub fn new() -> Self {
        Self {
            resolver: Arc::new(SystemResolver),
            limits: TransportLimits::default(),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_resolver(mut self, resolver: Arc<dyn Resolver>) -> Self {
        self.resolver = resolver;
        self
    }

    pub fn with_limits(mut self, limits: TransportLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<ResourceDownloader, TransportConfigError> {
        self.limits.validate()?;
        for (name, timeout) in [
            ("connect_timeout", self.connect_timeout),
            ("read_timeout", self.read_timeout),
        ] {
            if timeout.is_zero() {
                return Err(TransportConfigError::ZeroTimeout { name });
            }
        }
        let admission_capacity = self
            .limits
            .max_in_flight_requests
            .checked_add(self.limits.max_queued_requests)
            .ok_or(TransportConfigError::CapacityOverflow)?;
        Ok(ResourceDownloader {
            resolver: self.resolver,
            admission: Arc::new(Semaphore::new(admission_capacity)),
            in_flight: Arc::new(Semaphore::new(self.limits.max_in_flight_requests)),
            limits: self.limits,
            connect_timeout: self.connect_timeout,
            read_timeout: self.read_timeout,
        })
    }
}

impl Default for ResourceDownloaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Separate, structurally unauthenticated network client for provider-returned resources.
#[derive(Clone)]
pub struct ResourceDownloader {
    resolver: Arc<dyn Resolver>,
    admission: Arc<Semaphore>,
    in_flight: Arc<Semaphore>,
    limits: TransportLimits,
    connect_timeout: Duration,
    read_timeout: Duration,
}

impl ResourceDownloader {
    pub fn builder() -> ResourceDownloaderBuilder {
        ResourceDownloaderBuilder::new()
    }

    pub async fn download(
        &self,
        resource: ResourceUrl,
        options: ResourceDownloadOptions,
    ) -> Result<DownloadedResource, Error> {
        let cancellation = options.cancellation.child();
        let permits = self.acquire(&cancellation, options.deadline).await?;
        if resource.url.scheme() == "data" {
            return self
                .decode_data_url(resource, permits, cancellation, options.deadline)
                .await;
        }
        let _permits = permits;
        let mut current = resource;
        let mut visited = HashSet::new();
        for redirect_count in 0..=self.limits.max_redirects {
            if !visited.insert(current.url.as_str().to_owned()) {
                return Err(Error::new(
                    ErrorKind::Transport,
                    "resource redirect loop detected",
                ));
            }
            let endpoint = EndpointConfig::new(current.url.as_str(), current.policy.clone())
                .map_err(endpoint_download_error)?;
            let client = build_guarded_client(
                &endpoint,
                self.resolver.clone(),
                self.connect_timeout,
                self.read_timeout,
                &self.limits,
            )
            .map_err(|error| {
                Error::new(
                    ErrorKind::Configuration,
                    "resource HTTP client construction failed",
                )
                .with_source(error)
            })?;
            let response = run_controlled(
                client.get(current.url.clone()).send(),
                &cancellation,
                options.deadline,
            )
            .await?
            .map_err(transport_source_error)?;
            if let Some(remote) = response.remote_addr()
                && endpoint.validate_remote(remote).is_err()
            {
                return Err(Error::new(
                    ErrorKind::Transport,
                    "resource peer failed endpoint validation",
                ));
            }
            ResponseHeaders::validate(response.headers(), &self.limits).map_err(|detail| {
                response_limit_error(response.status(), response.headers(), Vec::new(), detail)
            })?;

            if response.status().is_redirection() {
                if redirect_count == self.limits.max_redirects {
                    return Err(Error::new(
                        ErrorKind::Transport,
                        "resource redirect limit exceeded",
                    ));
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::Protocol,
                            "resource redirect has no valid location",
                        )
                    })?;
                let redirected = current.url.join(location).map_err(|_| {
                    Error::new(ErrorKind::Protocol, "resource redirect location is invalid")
                })?;
                current = ResourceUrl::new(redirected.as_str(), current.policy.clone()).map_err(
                    |error| {
                        Error::new(
                            ErrorKind::Transport,
                            "resource redirect target failed security validation",
                        )
                        .with_source(error)
                    },
                )?;
                continue;
            }
            if !response.status().is_success() {
                let error =
                    resource_status_error(response, &cancellation, options.deadline).await?;
                return Err(error);
            }
            return self
                .read_success(response, current, &cancellation, options.deadline)
                .await;
        }
        Err(Error::new(
            ErrorKind::Internal,
            "resource redirect state became inconsistent",
        ))
    }

    async fn read_success(
        &self,
        response: reqwest::Response,
        final_url: ResourceUrl,
        cancellation: &Cancellation,
        deadline: Option<Instant>,
    ) -> Result<DownloadedResource, Error> {
        if let Some(length) = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            && length > self.limits.max_response_bytes as u64
        {
            return Err(response_limit_error(
                response.status(),
                response.headers(),
                Vec::new(),
                "resource Content-Length exceeds the configured limit",
            ));
        }
        let declared_media_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let status = response.status();
        let headers = response.headers().clone();
        let mut body = BytesMut::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = run_controlled(stream.next(), cancellation, deadline).await? {
            let chunk = chunk.map_err(transport_source_error)?;
            if body.len().saturating_add(chunk.len()) > self.limits.max_response_bytes {
                return Err(response_limit_error(
                    status,
                    &headers,
                    bounded_body_prefix(&body, Some(&chunk)),
                    "resource body exceeds the configured limit",
                ));
            }
            body.extend_from_slice(&chunk);
        }
        let data = body.freeze();
        let detected_media_type = infer::get(&data).map(|kind| kind.mime_type().to_owned());
        Ok(DownloadedResource {
            data,
            declared_media_type,
            detected_media_type,
            final_url,
        })
    }

    async fn decode_data_url(
        &self,
        resource: ResourceUrl,
        permits: ResourcePermits,
        cancellation: Cancellation,
        deadline: Option<Instant>,
    ) -> Result<DownloadedResource, Error> {
        let maximum = self.limits.max_response_bytes;
        let decode_cancellation = cancellation.clone();
        let task = tokio::task::spawn_blocking(move || {
            let _permits = permits;
            decode_data_url_blocking(resource, maximum, &decode_cancellation, deadline)
        });
        let joined = run_controlled(task, &cancellation, deadline).await?;
        joined.map_err(|error| {
            Error::new(ErrorKind::Internal, "data URL decoder task failed").with_source(error)
        })?
    }

    async fn acquire(
        &self,
        cancellation: &Cancellation,
        deadline: Option<Instant>,
    ) -> Result<ResourcePermits, Error> {
        let admission = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorKind::Transport, "resource download queue is full"))?;
        let in_flight = run_controlled(
            self.in_flight.clone().acquire_owned(),
            cancellation,
            deadline,
        )
        .await?
        .map_err(|_| Error::new(ErrorKind::Internal, "resource admission control is closed"))?;
        Ok(ResourcePermits {
            _admission: admission,
            _in_flight: in_flight,
        })
    }
}

impl fmt::Debug for ResourceDownloader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceDownloader")
            .field("limits", &self.limits)
            .field("connect_timeout", &self.connect_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("authentication", &"disabled")
            .field("proxy", &"disabled")
            .finish()
    }
}

struct ResourcePermits {
    _admission: OwnedSemaphorePermit,
    _in_flight: OwnedSemaphorePermit,
}

async fn resource_status_error(
    response: reqwest::Response,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<Error, Error> {
    let status = response.status();
    let headers = response.headers().clone();
    let mut body = BytesMut::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = run_controlled(stream.next(), cancellation, deadline).await? {
        let chunk = chunk.map_err(transport_source_error)?;
        if body.len().saturating_add(chunk.len()) > ERROR_BODY_CAPTURE_BYTES {
            let remaining = ERROR_BODY_CAPTURE_BYTES.saturating_sub(body.len());
            body.extend_from_slice(&chunk[..remaining.min(chunk.len())]);
            break;
        }
        body.extend_from_slice(&chunk);
    }
    let mut safe_headers = SafeResponseHeaders::default();
    for (name, value) in &headers {
        if let Ok(value) = value.to_str() {
            let _ = safe_headers.try_insert(name.as_str(), value.to_owned());
        }
    }
    let diagnostics = ResponseDiagnostics::default()
        .with_status(status.as_u16())
        .with_headers(safe_headers);
    Ok(
        Error::new(ErrorKind::Provider, "resource server returned an error")
            .with_diagnostics(diagnostics)
            .with_sensitive_response(SensitiveResponse::new(raw_headers(&headers), body.to_vec())),
    )
}

fn endpoint_download_error(error: EndpointError) -> Error {
    Error::new(
        ErrorKind::Transport,
        "resource endpoint failed security validation",
    )
    .with_source(error)
}

fn decode_data_url_blocking(
    resource: ResourceUrl,
    maximum: usize,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<DownloadedResource, Error> {
    ensure_decode_active(cancellation, deadline)?;
    let value = resource.url.as_str();
    let (metadata, payload) = value
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "data URL is invalid"))?;
    if metadata.len() > DATA_URL_METADATA_BYTES {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "data URL metadata exceeds the configured limit",
        ));
    }
    let mut parts = metadata.split(';');
    let declared_media_type = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let is_base64 = parts.any(|value| value.eq_ignore_ascii_case("base64"));
    let data = if is_base64 {
        decode_base64_bounded(payload, maximum, cancellation, deadline)?
    } else {
        percent_decode_bounded(payload, maximum, cancellation, deadline)?
    };
    ensure_decode_active(cancellation, deadline)?;
    let data = Bytes::from(data);
    let detected_media_type = infer::get(&data).map(|kind| kind.mime_type().to_owned());
    Ok(DownloadedResource {
        data,
        declared_media_type,
        detected_media_type,
        final_url: resource,
    })
}

fn decode_base64_bounded(
    payload: &str,
    maximum: usize,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<Vec<u8>, Error> {
    let maximum_encoded = maximum
        .saturating_add(2)
        .saturating_div(3)
        .saturating_mul(4);
    if payload.len() > maximum_encoded {
        return Err(data_url_limit_error());
    }

    let estimated = payload
        .len()
        .saturating_div(4)
        .saturating_mul(3)
        .saturating_add(3)
        .min(maximum);
    let mut output = Vec::with_capacity(estimated);
    let mut decoder = DecoderReader::new(
        payload.as_bytes(),
        &base64::engine::general_purpose::STANDARD,
    );
    let mut buffer = [0_u8; DECODE_CHECK_BYTES];
    loop {
        ensure_decode_active(cancellation, deadline)?;
        let remaining_with_overflow_probe = maximum
            .saturating_sub(output.len())
            .saturating_add(1)
            .min(buffer.len());
        let count = decoder
            .read(&mut buffer[..remaining_with_overflow_probe])
            .map_err(|error| {
                Error::new(ErrorKind::InvalidInput, "data URL base64 is invalid").with_source(error)
            })?;
        if count == 0 {
            break;
        }
        if output.len().saturating_add(count) > maximum {
            return Err(data_url_limit_error());
        }
        output.extend_from_slice(&buffer[..count]);
    }
    Ok(output)
}

fn data_url_limit_error() -> Error {
    Error::new(
        ErrorKind::ResponseLimit,
        "data URL exceeds the configured resource limit",
    )
}

fn percent_decode_bounded(
    value: &str,
    maximum: usize,
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<Vec<u8>, Error> {
    if value.len() > maximum.saturating_mul(3) {
        return Err(data_url_limit_error());
    }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len().min(maximum));
    let mut index = 0;
    let mut bytes_since_control_check = 0;
    while index < bytes.len() {
        if bytes_since_control_check >= DECODE_CHECK_BYTES {
            ensure_decode_active(cancellation, deadline)?;
            bytes_since_control_check = 0;
        }
        if output.len() >= maximum {
            return Err(data_url_limit_error());
        }
        if bytes[index] == b'%' {
            let high = bytes
                .get(index + 1)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "data URL escape is invalid"))?;
            let low = bytes
                .get(index + 2)
                .copied()
                .and_then(hex_value)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "data URL escape is invalid"))?;
            output.push((high << 4) | low);
            index += 3;
            bytes_since_control_check += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
            bytes_since_control_check += 1;
        }
    }
    Ok(output)
}

fn ensure_decode_active(
    cancellation: &Cancellation,
    deadline: Option<Instant>,
) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        return Err(Error::cancelled("resource download was cancelled"));
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(Error::new(
            ErrorKind::Timeout,
            "resource download deadline elapsed",
        ));
    }
    Ok(())
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn raw_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_url_debug_redacts_signed_query_and_data() {
        for resource in [
            ResourceUrl::public("https://example.com/file?sig=canary-secret").unwrap(),
            ResourceUrl::public("data:text/plain,canary-secret").unwrap(),
        ] {
            assert!(!format!("{resource:?}").contains("canary-secret"));
        }
    }

    #[test]
    fn network_urls_have_a_small_hard_limit_without_restricting_data_urls() {
        let oversized_network = format!(
            "https://example.com/{}",
            "a".repeat(MAX_NETWORK_RESOURCE_URL_BYTES)
        );
        assert!(matches!(
            ResourceUrl::public(oversized_network),
            Err(ResourceUrlError::TooLong)
        ));

        let inline_payload = "a".repeat(MAX_NETWORK_RESOURCE_URL_BYTES);
        assert!(ResourceUrl::public(format!("data:text/plain,{inline_payload}")).is_ok());
    }

    #[tokio::test]
    async fn data_urls_are_bounded_before_or_during_decode() {
        let downloader = ResourceDownloader::builder()
            .with_limits(TransportLimits {
                max_response_bytes: 4,
                ..TransportLimits::default()
            })
            .build()
            .unwrap();
        let valid = downloader
            .download(
                ResourceUrl::public("data:text/plain,okay").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(valid.data(), &Bytes::from_static(b"okay"));
        let error = downloader
            .download(
                ResourceUrl::public("data:text/plain;base64,Y2FuYXJ5").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResponseLimit);

        let percent_error = downloader
            .download(
                ResourceUrl::public("data:text/plain,%41%41%41%41%41").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(percent_error.kind(), ErrorKind::ResponseLimit);
    }

    #[tokio::test]
    async fn data_url_debug_and_errors_do_not_expose_payload_canaries() {
        let downloader = ResourceDownloader::builder().build().unwrap();
        let downloaded = downloader
            .download(
                ResourceUrl::public("data:canary-media-type,canary-secret").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap();
        let downloaded_debug = format!("{downloaded:?}");
        assert!(!downloaded_debug.contains("canary-secret"));
        assert!(!downloaded_debug.contains("canary-media-type"));

        let error = downloader
            .download(
                ResourceUrl::public("data:text/plain;base64,canary-secret").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(!format!("{error:?}").contains("canary-secret"));
        assert!(!error.to_string().contains("canary-secret"));
    }

    #[test]
    fn blocking_data_decode_observes_cancellation_and_deadline() {
        let cancelled = Cancellation::new();
        cancelled.cancel();
        let cancellation_error = decode_data_url_blocking(
            ResourceUrl::public("data:text/plain,canary-secret").unwrap(),
            64,
            &cancelled,
            None,
        )
        .unwrap_err();
        assert_eq!(cancellation_error.kind(), ErrorKind::Cancelled);

        let deadline_error = decode_data_url_blocking(
            ResourceUrl::public("data:text/plain,canary-secret").unwrap(),
            64,
            &Cancellation::new(),
            Some(Instant::now()),
        )
        .unwrap_err();
        assert_eq!(deadline_error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn data_urls_share_resource_queue_and_in_flight_limits() {
        let downloader = ResourceDownloader::builder()
            .with_limits(TransportLimits {
                max_connections: 1,
                max_in_flight_requests: 1,
                max_queued_requests: 1,
                ..TransportLimits::default()
            })
            .build()
            .unwrap();
        let in_flight_blocker = downloader.in_flight.clone().acquire_owned().await.unwrap();

        let first_cancellation = Cancellation::new();
        let first_task = tokio::spawn({
            let downloader = downloader.clone();
            let cancellation = first_cancellation.clone();
            async move {
                downloader
                    .download(
                        ResourceUrl::public("data:text/plain,first").unwrap(),
                        ResourceDownloadOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        wait_for_available_permits(&downloader.admission, 1).await;

        let second_cancellation = Cancellation::new();
        let second_task = tokio::spawn({
            let downloader = downloader.clone();
            let cancellation = second_cancellation.clone();
            async move {
                downloader
                    .download(
                        ResourceUrl::public("data:text/plain,second").unwrap(),
                        ResourceDownloadOptions::default().with_cancellation(cancellation),
                    )
                    .await
            }
        });
        wait_for_available_permits(&downloader.admission, 0).await;

        let queue_error = downloader
            .download(
                ResourceUrl::public("data:text/plain,third").unwrap(),
                ResourceDownloadOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(queue_error.kind(), ErrorKind::Transport);

        first_cancellation.cancel();
        second_cancellation.cancel();
        assert_eq!(
            first_task.await.unwrap().unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        assert_eq!(
            second_task.await.unwrap().unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        drop(in_flight_blocker);
        assert_eq!(
            downloader.admission.available_permits(),
            downloader.limits.max_in_flight_requests + downloader.limits.max_queued_requests
        );
    }

    async fn wait_for_available_permits(semaphore: &Semaphore, expected: usize) {
        for _ in 0..1_000 {
            if semaphore.available_permits() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("semaphore did not reach the expected permit count");
    }
}
