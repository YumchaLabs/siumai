use serde_json::{Value, json};
use siumai_core::experimental::{JobStatus, VideoJobModel};
use siumai_core::{CallOptions, ErrorKind, ProviderOptions, WarningKind};
use siumai_provider_alibaba::{
    AlibabaChatOptions, AlibabaCredential, AlibabaProvider,
    experimental::{
        AlibabaVideoDownloadPolicy, AlibabaVideoMedia, AlibabaVideoParameters,
        AlibabaVideoProviderBuilderExt, AlibabaVideoProviderExt, AlibabaVideoRequest, WAN_2_7_I2V,
    },
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CREATE_PATH: &str = "/api/v1/services/aigc/video-generation/video-synthesis";

fn provider(server: &MockServer, download_policy: AlibabaVideoDownloadPolicy) -> AlibabaProvider {
    AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_video_endpoint(
            EndpointConfig::local_explicit(format!("{}/api/v1", server.uri())).unwrap(),
        )
        .with_video_download_policy(download_policy)
        .build()
        .unwrap()
}

#[test]
fn video_models_share_the_alibaba_identity_without_entering_stable_registry_families() {
    let endpoint = EndpointConfig::local_explicit("http://127.0.0.1:9/api/v1").unwrap();
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_video_endpoint(endpoint)
        .build()
        .unwrap();

    let model = provider.video("future-wan-model").unwrap();
    assert_eq!(model.provider_id().as_str(), "alibaba");
    assert_eq!(
        model.scope().platform().map(|value| value.as_str()),
        Some("local")
    );
    assert_eq!(
        model.scope().protocol().map(|value| value.as_str()),
        Some("alibaba-native")
    );
    assert_eq!(
        model.scope().api_mode().map(|value| value.as_str()),
        Some("video-job")
    );
}

#[tokio::test]
async fn typed_video_job_create_poll_and_materialize_preserve_native_wire_semantics() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .and(header("authorization", "Bearer test-key"))
        .and(header("x-dashscope-async", "enable"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "create-request-42",
            "output": {"task_id": "video-task-42", "task_status": "PENDING"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/video-task-42"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "poll-request-42",
            "output": {
                "task_id": "video-task-42",
                "task_status": "SUCCEEDED",
                "video_url": format!("{}/assets/video-task-42.mp4?token=canary", server.uri())
            },
            "usage": {
                "duration": 5.0,
                "output_video_duration": 5.0,
                "SR": 1.0,
                "size": "1920*1080"
            }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/video-task-42.mp4"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "video/mp4")
                .set_body_bytes(b"video-bytes"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let request = AlibabaVideoRequest::image(
        "A paper boat crosses a moonlit lake",
        "https://example.com/first.png",
    )
    .with_media(AlibabaVideoMedia::last_frame(
        "https://example.com/last.png",
    ))
    .with_audio_url("https://example.com/score.mp3")
    .with_parameters(
        AlibabaVideoParameters::new()
            .with_resolution("1080P", "16:9")
            .with_duration(5)
            .with_prompt_extension(true)
            .with_watermark(false)
            .with_audio(true),
    );
    let model = provider(&server, AlibabaVideoDownloadPolicy::LoopbackExplicit)
        .video(WAN_2_7_I2V)
        .unwrap();
    let created = model
        .create_video(request, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(created.status(), &JobStatus::Queued);
    assert_eq!(created.request_id(), Some("create-request-42"));
    assert!(created.warnings().is_empty());

    let completed = model
        .poll_video(&created, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(completed.status(), &JobStatus::Completed);
    assert_eq!(completed.request_id(), Some("poll-request-42"));
    assert_eq!(completed.usage().output_video_duration, Some(5.0));
    assert_eq!(completed.usage().size.as_deref(), Some("1920*1080"));
    let debug = format!("{completed:?}");
    assert!(!debug.contains("token=canary"));

    let bytes = model
        .materialize_video(&completed, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(bytes, b"video-bytes");

    let requests = server.received_requests().await.unwrap();
    let create = requests
        .iter()
        .find(|request| request.url.path() == CREATE_PATH)
        .unwrap();
    let body: Value = serde_json::from_slice(&create.body).unwrap();
    assert_eq!(body["model"], json!(WAN_2_7_I2V));
    assert_eq!(
        body["input"]["media"],
        json!([
            {"type": "first_frame", "url": "https://example.com/first.png"},
            {"type": "last_frame", "url": "https://example.com/last.png"}
        ])
    );
    assert_eq!(
        body["input"]["audio_url"],
        json!("https://example.com/score.mp3")
    );
    assert!(body["input"].get("parameters").is_none());
    assert_eq!(body["parameters"]["resolution"], json!("1080P"));
    assert_eq!(body["parameters"]["ratio"], json!("16:9"));
    assert_eq!(body["parameters"]["audio"], json!(true));

    let download = requests
        .iter()
        .find(|request| request.url.path() == "/assets/video-task-42.mp4")
        .unwrap();
    assert!(!download.headers.contains_key("authorization"));
}

#[tokio::test]
async fn dynamic_video_snapshot_omits_signed_url_and_repolls_before_materialization() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"task_id": "persisted-task-42", "task_status": "PENDING"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/persisted-task-42"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "task_id": "persisted-task-42",
                "task_status": "SUCCEEDED",
                "video_url": format!(
                    "{}/assets/persisted-task-42.mp4?token=serialized-canary",
                    server.uri()
                )
            }
        })))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/assets/persisted-task-42.mp4"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"persisted-video"))
        .expect(1)
        .mount(&server)
        .await;

    let model = provider(&server, AlibabaVideoDownloadPolicy::LoopbackExplicit)
        .video(WAN_2_7_I2V)
        .unwrap();
    let created = VideoJobModel::create(
        &model,
        serde_json::to_value(AlibabaVideoRequest::text("persist safely")).unwrap(),
        CallOptions::default(),
    )
    .await
    .unwrap();
    let completed = VideoJobModel::poll(&model, &created, CallOptions::default())
        .await
        .unwrap();

    let snapshot = serde_json::to_string(&completed).unwrap();
    assert!(!snapshot.contains("serialized-canary"));
    assert!(!snapshot.contains("video_url"));
    assert!(!format!("{completed:?}").contains("serialized-canary"));

    let restored: siumai_core::experimental::MediaJob = serde_json::from_str(&snapshot).unwrap();
    let bytes = VideoJobModel::materialize(&model, &restored, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(bytes, b"persisted-video");
}

#[tokio::test]
async fn dynamic_video_job_snapshots_update_on_cancel_and_keep_future_models_callable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"task_id": "future-task-42", "task_status": "PENDING"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/tasks/future-task-42/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"task_id": "future-task-42", "task_status": "CANCELED"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let model = provider(&server, AlibabaVideoDownloadPolicy::LoopbackExplicit)
        .video("future-wan-3")
        .unwrap();
    let request = AlibabaVideoRequest::text("future request").with_parameters(
        AlibabaVideoParameters::new()
            .with_resolution("future-tier", "future-ratio")
            .with_duration(12),
    );
    let created = VideoJobModel::create(
        &model,
        serde_json::to_value(request).unwrap(),
        CallOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(created.status(), &JobStatus::Queued);
    let cancelled = VideoJobModel::cancel(&model, &created, CallOptions::default())
        .await
        .unwrap();
    assert_eq!(cancelled.status(), &JobStatus::Cancelled);

    let state = cancelled.state();
    assert!(matches!(
        state["warnings"][0]["kind"].as_str(),
        Some("UnknownModel")
    ));
    let requests = server.received_requests().await.unwrap();
    let create = requests
        .iter()
        .find(|request| request.url.path() == CREATE_PATH)
        .unwrap();
    let body: Value = serde_json::from_slice(&create.body).unwrap();
    assert_eq!(body["parameters"]["resolution"], json!("future-tier"));
    assert_eq!(body["parameters"]["ratio"], json!("future-ratio"));
    assert_eq!(body["parameters"]["duration"], json!(12));
}

#[tokio::test]
async fn video_validation_and_foreign_call_options_fail_before_wire() {
    let server = MockServer::start().await;
    let model = provider(&server, AlibabaVideoDownloadPolicy::LoopbackExplicit)
        .video(WAN_2_7_I2V)
        .unwrap();

    let error = model
        .create_video(AlibabaVideoRequest::new(), CallOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    let chat_options = AlibabaChatOptions::new().with_enable_search(true);
    let error = model
        .create_video(
            AlibabaVideoRequest::text("hello"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&chat_options).unwrap()),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn video_errors_expose_safe_code_and_request_id_only() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("x-private-canary", "canary-header-secret")
                .set_body_json(json!({
                    "code": "InvalidParameter",
                    "request_id": "video-error-42",
                    "message": "canary-body-secret"
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let error = provider(&server, AlibabaVideoDownloadPolicy::LoopbackExplicit)
        .video("future-wan-3")
        .unwrap()
        .create_video(AlibabaVideoRequest::text("hello"), CallOptions::default())
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        error.diagnostics().and_then(|value| value.provider_code()),
        Some("InvalidParameter")
    );
    assert_eq!(
        error.diagnostics().and_then(|value| value.request_id()),
        Some("video-error-42")
    );
    let debug = format!("{error:?}");
    let display = error.to_string();
    assert!(!debug.contains("canary-header-secret"));
    assert!(!debug.contains("canary-body-secret"));
    assert!(!display.contains("canary-header-secret"));
    assert!(!display.contains("canary-body-secret"));
}

#[test]
fn unknown_model_warning_shape_is_stable_for_typed_jobs() {
    let warning = siumai_core::Warning::new(
        WarningKind::UnknownModel,
        "model is absent from the verified Alibaba video advisory catalog",
    );
    assert_eq!(warning.kind(), &WarningKind::UnknownModel);
}
