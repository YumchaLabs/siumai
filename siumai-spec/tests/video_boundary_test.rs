use siumai_spec::types::{
    ProviderReference, VideoGenerationRequest, VideoTaskStatus, VideoTaskStatusResponse,
};
use std::collections::HashMap;

#[test]
fn video_request_header_helper_uses_empty_http_override_config() {
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
