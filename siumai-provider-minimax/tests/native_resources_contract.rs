use serde_json::{Value, json};
use siumai_core::{
    CallOptions, ErrorKind, ImageModel, ImageRequest, LanguageRequest, MediaData, Message, ModelId,
    ReplayDomain, ReplayDomainId, SpeechModel, SpeechRequest,
};
use siumai_provider_minimax::{
    MinimaxCredential, MinimaxProvider,
    models::{image::IMAGE_01, music::MUSIC_3_0, speech::SPEECH_2_8_HD, video::MINIMAX_H3},
    resources::{
        MinimaxCustomVoiceId, MinimaxCustomVoiceKind, MinimaxFileDeletePurpose, MinimaxFileId,
        MinimaxImageAspectRatio, MinimaxImageRequest, MinimaxMusicAudioFormat,
        MinimaxMusicAudioSetting, MinimaxMusicBitrate, MinimaxMusicRequest, MinimaxMusicSampleRate,
        MinimaxMusicStatus, MinimaxResponsesInputTokenRequest, MinimaxSpeechAudioFormat,
        MinimaxSpeechAudioSettings, MinimaxSpeechBitrate, MinimaxSpeechChannels,
        MinimaxSpeechOutputFormat, MinimaxSpeechSampleRate, MinimaxSpeechSynthesisRequest,
        MinimaxSpeechTaskId, MinimaxSpeechTaskStatus, MinimaxSubtitleGranularity,
        MinimaxVideoRatio, MinimaxVideoRequest, MinimaxVideoResolution, MinimaxVoiceClonePreview,
        MinimaxVoiceClonePrompt, MinimaxVoiceCloneRequest, MinimaxVoiceDesignRequest,
        MinimaxVoiceId, MinimaxVoiceListKind, MinimaxVoicePreviewModel, MinimaxVoiceSettings,
    },
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> MinimaxProvider {
    MinimaxProvider::builder(MinimaxCredential::api_key("resource-contract-key"))
        .with_openai_endpoint(
            EndpointConfig::local_explicit(server.uri()).expect("local OpenAI endpoint"),
        )
        .with_openai_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("minimax-contract-relay").expect("replay domain"),
        ))
        .with_resource_endpoint(
            EndpointConfig::local_explicit(server.uri()).expect("local resource endpoint"),
        )
        .build()
        .expect("MiniMax provider")
}

#[tokio::test]
async fn portable_image_and_speech_adapters_preserve_family_contracts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/image_generation"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "portable-image-42",
            "data": {"image_base64": ["iVBORw0KGgo="]},
            "metadata": {"success_count": 1, "failed_count": 0},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/t2a_v2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"audio": "494433"},
            "trace_id": "portable-speech-42",
            "extra_info": {
                "audio_length": 1250,
                "audio_sample_rate": 24000,
                "usage_characters": 5,
                "audio_format": "mp3"
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider(&server);
    let image_request = ImageRequest::new("A tiny copper robot").expect("image request");
    let image = provider
        .image(IMAGE_01)
        .expect("image model")
        .generate_image(image_request, CallOptions::default())
        .await
        .expect("portable image");
    assert_eq!(
        image.metadata.response_id.as_deref(),
        Some("portable-image-42")
    );
    assert!(matches!(
        &image.images[0].data,
        MediaData::Bytes(bytes) if bytes.as_ref() == b"\x89PNG\r\n\x1a\n"
    ));
    assert_eq!(image.images[0].media_type, "image/png");
    assert_eq!(image.provider["success_count"], json!(1));

    let speech_request = SpeechRequest::new("hello")
        .and_then(|request| request.with_voice("English_Graceful_Lady"))
        .and_then(|request| request.with_format("mp3"))
        .expect("speech request");
    let speech = provider
        .speech_model(ModelId::new(SPEECH_2_8_HD).expect("speech model id"))
        .expect("speech model")
        .synthesize(speech_request, CallOptions::default())
        .await
        .expect("portable speech");
    assert_eq!(speech.audio.as_ref(), b"ID3");
    assert_eq!(speech.media_type, "audio/mpeg");
    assert_eq!(speech.duration_seconds, Some(1.25));
    assert_eq!(speech.sample_rate_hz, Some(24_000));
    assert_eq!(
        speech.metadata.request_id.as_deref(),
        Some("portable-speech-42")
    );
    assert_eq!(speech.usage.provider["characters"], json!(5));
}

#[tokio::test]
async fn responses_input_tokens_supports_native_text_and_role_safe_items() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses/input_tokens"))
        .and(header("authorization", "Bearer resource-contract-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "response.input_tokens",
            "input_tokens": 23,
            "cache_detail": {"read": 7}
        })))
        .expect(2)
        .mount(&server)
        .await;

    let request = MinimaxResponsesInputTokenRequest::from_language_request(
        "MiniMax-M3",
        LanguageRequest::new(vec![Message::user("hello")]),
    )
    .expect("input-token request");
    let counted = provider(&server)
        .responses_resource()
        .count_input_tokens(request)
        .await
        .expect("input-token count");
    assert_eq!(counted.input_tokens(), 23);
    assert_eq!(counted.extra()["cache_detail"], json!({"read": 7}));
    let text_counted = provider(&server)
        .responses_resource()
        .count_input_tokens(
            MinimaxResponsesInputTokenRequest::from_text("MiniMax-M3", "hello")
                .expect("text input-token request"),
        )
        .await
        .expect("text input-token count");
    assert_eq!(text_counted.input_tokens(), 23);

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 2);
    let body: Value = serde_json::from_slice(&requests[0].body).expect("input-token JSON body");
    assert_eq!(body["model"], json!("MiniMax-M3"));
    assert!(body["input"].is_array());
    assert!(body.get("stream").is_none());
    assert!(body.get("max_output_tokens").is_none());
    let text_body: Value =
        serde_json::from_slice(&requests[1].body).expect("text input-token JSON body");
    assert_eq!(text_body, json!({"model": "MiniMax-M3", "input": "hello"}));
}

#[tokio::test]
async fn voice_lifecycle_is_typed_bounded_and_explicitly_non_polling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/voice_clone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "demo_audio": "",
            "input_sensitive": false,
            "input_sensitive_type": 0,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/voice_design"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "voice_id": "DesignVoice01",
            "trial_audio": "494433",
            "input_sensitive": false,
            "input_sensitive_type": 0,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/get_voice"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "system_voice": [{
                "voice_id": "system-voice",
                "voice_name": "System",
                "description": ["Warm", "Narration"]
            }],
            "voice_cloning": [{
                "voice_id": "Clone_Voice-01",
                "created_time": "2026-08-09T00:00:00Z"
            }],
            "voice_generation": [{
                "voice_id": "DesignVoice01",
                "description": ["Calm"],
                "created_time": "2026-08-09T00:00:00Z"
            }],
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/delete_voice"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "voice_id": "Clone_Voice-01",
            "created_time": "2026-08-09T00:00:00Z",
            "request_trace": "delete-42",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider(&server);
    let voices = provider.voices();
    let cloned = voices
        .clone_voice(
            MinimaxVoiceCloneRequest::new(
                MinimaxFileId::new(42).expect("clone file"),
                MinimaxCustomVoiceId::new("Clone_Voice-01").expect("clone voice id"),
            )
            .with_prompt(
                MinimaxVoiceClonePrompt::new(
                    MinimaxFileId::new(43).expect("prompt file"),
                    "reference transcript",
                )
                .expect("clone prompt"),
            )
            .with_preview(
                MinimaxVoiceClonePreview::new("preview text", MinimaxVoicePreviewModel::Speech28Hd)
                    .expect("clone preview"),
            ),
        )
        .await
        .expect("clone voice");
    assert_eq!(cloned.voice_id().as_str(), "Clone_Voice-01");
    assert!(cloned.demo_audio().is_none());
    assert_eq!(cloned.safety().flagged(), Some(false));

    let designed = voices
        .design_voice(
            MinimaxVoiceDesignRequest::new("Warm, calm narrator", "hello world")
                .expect("design request")
                .with_voice_id(
                    MinimaxCustomVoiceId::new("DesignVoice01").expect("design voice id"),
                ),
        )
        .await
        .expect("design voice");
    assert_eq!(designed.voice_id().as_str(), "DesignVoice01");
    assert_eq!(designed.trial_audio(), b"ID3");

    let listed = voices
        .list(MinimaxVoiceListKind::All)
        .await
        .expect("list voices");
    assert_eq!(listed.system().len(), 1);
    assert_eq!(listed.system()[0].description(), ["Warm", "Narration"]);
    assert_eq!(
        listed.voice_cloning()[0].voice_id().as_str(),
        "Clone_Voice-01"
    );
    assert_eq!(
        listed.voice_cloning()[0].created_time(),
        Some("2026-08-09T00:00:00Z")
    );
    assert_eq!(
        listed.voice_generation()[0].voice_id().as_str(),
        "DesignVoice01"
    );

    let deleted = voices
        .delete(
            MinimaxCustomVoiceId::new("Clone_Voice-01").expect("delete voice id"),
            MinimaxCustomVoiceKind::VoiceCloning,
        )
        .await
        .expect("delete voice");
    assert_eq!(deleted.voice_id().as_str(), "Clone_Voice-01");
    assert_eq!(deleted.created_time(), "2026-08-09T00:00:00Z");
    assert_eq!(deleted.extra()["request_trace"], json!("delete-42"));

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 4, "voice operations must not hide polling");
    let clone_body: Value = serde_json::from_slice(&requests[0].body).expect("clone JSON body");
    assert_eq!(clone_body["file_id"], json!(42));
    assert_eq!(clone_body["voice_id"], json!("Clone_Voice-01"));
    assert_eq!(clone_body["model"], json!("speech-2.8-hd"));
    assert_eq!(clone_body["clone_prompt"]["prompt_audio"], json!(43));
    let list_body: Value = serde_json::from_slice(&requests[2].body).expect("list JSON body");
    assert_eq!(list_body, json!({"voice_type": "all"}));
    let delete_body: Value = serde_json::from_slice(&requests[3].body).expect("delete JSON body");
    assert_eq!(
        delete_body,
        json!({"voice_id": "Clone_Voice-01", "voice_type": "voice_cloning"})
    );
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
