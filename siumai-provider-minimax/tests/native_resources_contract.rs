use serde_json::{Value, json};
use siumai_core::ErrorKind;
use siumai_provider_minimax::{
    MinimaxCredential, MinimaxProvider,
    models::{image::IMAGE_01, music::MUSIC_3_0, speech::SPEECH_2_8_HD, video::MINIMAX_H3},
    resources::{
        MinimaxFileDeletePurpose, MinimaxFileId, MinimaxImageAspectRatio, MinimaxImageRequest,
        MinimaxMusicAudioFormat, MinimaxMusicAudioSetting, MinimaxMusicBitrate,
        MinimaxMusicRequest, MinimaxMusicSampleRate, MinimaxMusicStatus, MinimaxSpeechAudioFormat,
        MinimaxSpeechAudioSettings, MinimaxSpeechBitrate, MinimaxSpeechChannels,
        MinimaxSpeechOutputFormat, MinimaxSpeechSampleRate, MinimaxSpeechSynthesisRequest,
        MinimaxSpeechTaskId, MinimaxSpeechTaskStatus, MinimaxSubtitleGranularity,
        MinimaxVideoRatio, MinimaxVideoRequest, MinimaxVideoResolution, MinimaxVoiceId,
        MinimaxVoiceSettings,
    },
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> MinimaxProvider {
    MinimaxProvider::builder(MinimaxCredential::api_key("resource-contract-key"))
        .with_resource_endpoint(
            EndpointConfig::local_explicit(server.uri()).expect("local resource endpoint"),
        )
        .build()
        .expect("MiniMax provider")
}

#[tokio::test]
async fn files_delete_sends_one_explicit_request_without_hidden_retrieve() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/files/delete"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "file_id": 42,
            "request_id": "delete-request-42",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let file_id = MinimaxFileId::new(42).expect("file id");
    let result = provider(&server)
        .files()
        .delete(file_id, MinimaxFileDeletePurpose::VideoGeneration)
        .await
        .expect("delete file");

    assert_eq!(result.id(), file_id);
    assert!(result.deleted());
    assert_eq!(
        result.extra().get("request_id").and_then(Value::as_str),
        Some("delete-request-42")
    );

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(
        requests.len(),
        1,
        "delete must not perform a hidden retrieve"
    );
    let body: Value = serde_json::from_slice(&requests[0].body).expect("delete JSON body");
    assert_eq!(body, json!({"file_id": 42, "purpose": "video_generation"}));
}

#[tokio::test]
async fn video_create_preserves_h3_v2_wire_shape_and_does_not_poll() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/video_generation"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "task_id": "video-task-42",
            "request_id": "video-create-42"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let request = MinimaxVideoRequest::new(
        MINIMAX_H3,
        "A paper boat crosses a moonlit lake",
        MinimaxVideoResolution::P768,
        6,
    )
    .expect("video request")
    .with_ratio(MinimaxVideoRatio::Landscape16By9);
    let result = provider(&server)
        .video()
        .create(request)
        .await
        .expect("create video task");

    assert_eq!(result.task_id().as_str(), "video-task-42");
    assert_eq!(
        result.extra().get("request_id").and_then(Value::as_str),
        Some("video-create-42")
    );

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 1, "create must not hide task polling");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("video JSON body");
    assert_eq!(body["model"], json!(MINIMAX_H3));
    assert_eq!(body["resolution"], json!("768P"));
    assert_eq!(body["duration"], json!(6));
    assert_eq!(body["ratio"], json!("16:9"));
    assert_eq!(
        body["content"],
        json!([{
            "type": "text",
            "text": "A paper boat crosses a moonlit lake"
        }])
    );
}

#[tokio::test]
async fn image_generation_encodes_typed_options_and_decodes_urls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/image_generation"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "image-generation-42",
            "data": {
                "image_urls": ["https://cdn.example.test/image.png?token=image-canary"],
                "provider_additive": true
            },
            "metadata": {"success_count": "1", "failed_count": 0},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let request = MinimaxImageRequest::new(IMAGE_01, "A geometric red fox")
        .expect("image request")
        .with_aspect_ratio(MinimaxImageAspectRatio::Square)
        .with_seed(7)
        .with_prompt_optimizer(true);
    let generation = provider(&server)
        .images()
        .generate(request)
        .await
        .expect("generate image");

    assert_eq!(generation.id(), Some("image-generation-42"));
    assert_eq!(
        generation.urls(),
        ["https://cdn.example.test/image.png?token=image-canary"]
    );
    assert_eq!(generation.metadata().success_count(), Some(1));
    assert_eq!(generation.metadata().failed_count(), Some(0));
    assert_eq!(
        generation
            .data_extra()
            .get("provider_additive")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert!(!format!("{generation:?}").contains("image-canary"));

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).expect("image JSON body");
    assert_eq!(body["model"], json!(IMAGE_01));
    assert_eq!(body["prompt"], json!("A geometric red fox"));
    assert_eq!(body["aspect_ratio"], json!("1:1"));
    assert_eq!(body["response_format"], json!("url"));
    assert_eq!(body["seed"], json!(7));
    assert_eq!(body["n"], json!(1));
    assert_eq!(body["prompt_optimizer"], json!(true));
}

#[tokio::test]
async fn music_generation_is_one_non_streaming_json_request_and_decodes_hex_audio() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/music_generation"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"status": 2, "audio": "494433"},
            "trace_id": "music-trace-42",
            "extra_info": {
                "music_duration": 1234,
                "music_sample_rate": 44100,
                "music_channel": 2,
                "bitrate": 128000,
                "music_size": 3
            },
            "analysis_info": {"key": "C"},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let audio_setting = MinimaxMusicAudioSetting::new()
        .with_sample_rate(MinimaxMusicSampleRate::Hz44100)
        .with_bitrate(MinimaxMusicBitrate::Bps128000)
        .with_format(MinimaxMusicAudioFormat::Mp3);
    let request = MinimaxMusicRequest::from_lyrics(MUSIC_3_0, "[Verse]\nMoonlit water")
        .expect("music request")
        .with_style_prompt("warm acoustic folk")
        .expect("style prompt")
        .with_audio_setting(audio_setting);
    let generation = provider(&server)
        .music()
        .generate(request)
        .await
        .expect("generate music");

    assert_eq!(generation.status(), MinimaxMusicStatus::Completed);
    assert_eq!(
        generation.output().and_then(|output| output.as_audio()),
        Some(b"ID3".as_slice())
    );
    assert_eq!(generation.trace_id(), Some("music-trace-42"));
    assert_eq!(generation.metadata().duration(), Some(1234));
    assert_eq!(generation.metadata().sample_rate(), Some(44_100));
    assert_eq!(generation.metadata().channels(), Some(2));
    assert_eq!(generation.metadata().bitrate(), Some(128_000));
    assert_eq!(generation.metadata().size(), Some(3));
    assert_eq!(generation.analysis_info(), Some(&json!({"key": "C"})));

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 1, "buffered music must submit only once");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("music JSON body");
    assert_eq!(body["model"], json!(MUSIC_3_0));
    assert_eq!(body["prompt"], json!("warm acoustic folk"));
    assert_eq!(body["lyrics"], json!("[Verse]\nMoonlit water"));
    assert_eq!(body["stream"], json!(false));
    assert_eq!(body["output_format"], json!("hex"));
    assert_eq!(
        body["audio_setting"],
        json!({"sample_rate": 44100, "bitrate": 128000, "format": "mp3"})
    );
}

#[tokio::test]
async fn speech_synthesis_sends_buffered_wire_contract_and_decodes_audio() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/t2a_v2"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "audio": "494433",
                "status": 2,
                "subtitle_file": "https://cdn.example.test/subtitles.json?token=subtitle-canary"
            },
            "trace_id": "speech-trace-42",
            "extra_info": {
                "audio_length": 850,
                "audio_sample_rate": 44100,
                "audio_size": 3,
                "bitrate": 128000,
                "word_count": 2,
                "usage_characters": 11,
                "audio_format": "mp3",
                "audio_channel": 1
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let voice =
        MinimaxVoiceSettings::new(MinimaxVoiceId::new("English_Graceful_Lady").expect("voice id"));
    let audio = MinimaxSpeechAudioSettings::new(MinimaxSpeechAudioFormat::Mp3)
        .with_sample_rate(MinimaxSpeechSampleRate::Hz44100)
        .with_bitrate(MinimaxSpeechBitrate::Kbps128)
        .with_channels(MinimaxSpeechChannels::Mono);
    let request = MinimaxSpeechSynthesisRequest::new(SPEECH_2_8_HD, "Hello world", voice)
        .expect("speech request")
        .with_audio(audio)
        .with_output_format(MinimaxSpeechOutputFormat::Hex)
        .with_subtitles(MinimaxSubtitleGranularity::Word)
        .with_text_normalization(true);
    let response = provider(&server)
        .speech()
        .synthesize(request)
        .await
        .expect("synthesize speech");

    assert_eq!(response.output().audio(), Some(b"ID3".as_slice()));
    assert_eq!(response.trace_id(), Some("speech-trace-42"));
    assert_eq!(
        response.subtitle_url().map(|url| url.as_str()),
        Some("https://cdn.example.test/subtitles.json?token=subtitle-canary")
    );
    let info = response.info().expect("speech response info");
    assert_eq!(info.duration_millis(), Some(850));
    assert_eq!(info.sample_rate_hertz(), Some(44_100));
    assert_eq!(info.audio_size_bytes(), Some(3));
    assert_eq!(info.bitrate_bits_per_second(), Some(128_000));
    assert_eq!(info.word_count(), Some(2));
    assert_eq!(info.usage_characters(), Some(11));
    assert_eq!(info.audio_format(), Some("mp3"));
    assert_eq!(info.channel_count(), Some(1));
    assert!(!format!("{response:?}").contains("subtitle-canary"));

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(
        requests.len(),
        1,
        "buffered synthesis must submit only once"
    );
    let body: Value = serde_json::from_slice(&requests[0].body).expect("speech JSON body");
    assert_eq!(body["model"], json!(SPEECH_2_8_HD));
    assert_eq!(body["text"], json!("Hello world"));
    assert_eq!(body["stream"], json!(false));
    assert_eq!(body["output_format"], json!("hex"));
    assert_eq!(
        body["voice_setting"]["voice_id"],
        json!("English_Graceful_Lady")
    );
    assert_eq!(body["voice_setting"]["text_normalization"], json!(true));
    assert_eq!(
        body["audio_setting"],
        json!({"format": "mp3", "sample_rate": 44100, "bitrate": 128000, "channel": 1})
    );
    assert_eq!(body["subtitle_enable"], json!(true));
    assert_eq!(body["subtitle_type"], json!("word"));
}

#[tokio::test]
async fn speech_query_performs_exactly_one_task_lookup_without_polling() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/query/t2a_async_query_v2"))
        .and(query_param("task_id", "77"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "task_id": 77,
            "status": "success",
            "file_id": 88,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let task_id = MinimaxSpeechTaskId::new(77).expect("speech task id");
    let task = provider(&server)
        .speech()
        .query(task_id)
        .await
        .expect("query speech task");

    assert_eq!(task.task_id(), task_id);
    assert_eq!(task.status(), &MinimaxSpeechTaskStatus::Succeeded);
    assert_eq!(task.file_id().map(MinimaxFileId::get), Some(88));
    assert!(task.status().is_terminal());

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 1, "query must not hide a polling loop");
    assert_eq!(requests[0].url.query(), Some("task_id=77"));
}

#[tokio::test]
async fn native_resource_errors_keep_provider_payloads_off_default_surfaces() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/image_generation"))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("x-private-canary", "canary-header-secret")
                .set_body_json(json!({
                    "base_resp": {
                        "status_code": 1004,
                        "status_msg": "canary-body-secret"
                    },
                    "private_detail": "second-body-canary"
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let request = MinimaxImageRequest::new(IMAGE_01, "A safe fixture")
        .expect("image request")
        .with_aspect_ratio(MinimaxImageAspectRatio::Square);
    let error = provider(&server)
        .images()
        .generate(request)
        .await
        .expect_err("provider error");

    assert_eq!(error.kind(), ErrorKind::Authentication);
    let diagnostics = error.diagnostics().expect("response diagnostics");
    assert_eq!(diagnostics.status(), Some(400));
    assert_eq!(diagnostics.provider_type(), Some("base_resp"));
    assert_eq!(diagnostics.provider_code(), Some("1004"));
    for rendered in [
        error.to_string(),
        format!("{error:?}"),
        serde_json::to_string(&error).expect("serialized error"),
    ] {
        assert!(!rendered.contains("canary-header-secret"));
        assert!(!rendered.contains("canary-body-secret"));
        assert!(!rendered.contains("second-body-canary"));
    }

    let (_, sensitive_body) = error
        .sensitive_response()
        .expect("explicit sensitive response")
        .expose();
    let sensitive_body = std::str::from_utf8(sensitive_body).expect("UTF-8 fixture body");
    assert!(sensitive_body.contains("canary-body-secret"));
    assert!(sensitive_body.contains("second-body-canary"));

    assert_eq!(
        server
            .received_requests()
            .await
            .expect("received requests")
            .len(),
        1
    );
}
