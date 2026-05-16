use siumai_spec::types::{
    ProviderReference, VideoGenerationRequest, VideoTaskStatus, VideoTaskStatusResponse,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn video_source() -> String {
    let path = crate_root().join("src/types/video.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

#[test]
fn video_generation_surface_remains_passive_data_contract() {
    let source = video_source();

    for forbidden in [
        "siumai_core::",
        "siumai_provider_",
        "siumai_protocol_",
        "tokio",
        "reqwest",
        "hyper::",
        "ureq::",
        "isahc::",
        "axum::",
        "async_trait",
        "pub async fn",
        "async fn",
        ".await",
        "spawn_blocking",
        "std::net",
        "std::process",
        "std::thread",
        "std::fs",
        "std::env",
        "CARGO_PKG_VERSION",
        "HttpClient",
        "ClientBuilder",
        "pub fn poll",
        "pub fn wait",
        "pub fn download",
        "pub fn fetch",
        "poll_task",
        "wait_for",
        "download_video",
        "fetch_video",
    ] {
        assert!(
            !source.contains(forbidden),
            "siumai-spec::types::video must stay passive data and must not contain runtime/provider execution fragment `{forbidden}`"
        );
    }
}

#[test]
fn video_request_header_helper_uses_empty_http_override_config() {
    let source = video_source();

    assert!(
        source.contains("unwrap_or_else(HttpConfig::empty)"),
        "VideoGenerationRequest::with_header should create request-level override config from HttpConfig::empty()"
    );
    assert!(
        !source.contains("HttpConfig::default"),
        "siumai-spec video request helpers must not use HttpConfig::default() as a runtime-default shortcut"
    );

    let request = VideoGenerationRequest::new_without_prompt("veo-3.1-generate-preview")
        .with_header("x-test", "1");
    let config = request.http_config.as_ref().expect("request http config");

    assert_eq!(config.headers.get("x-test").map(String::as_str), Some("1"));
    assert_eq!(config.timeout, None);
    assert_eq!(config.connect_timeout, None);
    assert_eq!(config.proxy, None);
    assert_eq!(config.user_agent, None);
    assert!(!config.stream_disable_compression);
}

#[test]
fn video_task_provider_reference_resolution_is_data_projection_only() {
    let response = VideoTaskStatusResponse {
        task_id: "task-123".to_string(),
        status: VideoTaskStatus::Success,
        file_id: Some("legacy-file".to_string()),
        video_url: None,
        provider_reference: Some(ProviderReference::from([("gemini", "files/123")])),
        duration: None,
        video_width: None,
        video_height: None,
        base_resp: None,
        metadata: HashMap::new(),
        response: None,
    };

    let effective = response
        .effective_provider_reference("fallback")
        .expect("provider reference");

    assert_eq!(effective.get("gemini"), Some("files/123"));
    assert_eq!(effective.get("fallback"), None);
}
