use siumai_core::{CallOptions, ImageModel, ImageRequest, MediaData, ModelFamily, ProviderOptions};
use siumai_core::{ReplayDomain, ReplayDomainId};
use siumai_provider_volcengine::models::{DREAMINA_SEEDANCE_2_0_260128, SEEDREAM_5_0_260128};
use siumai_provider_volcengine::{
    ArkImageOptions, ArkVideoCreateRequest, ArkVideoTaskListQuery, ArkVideoTaskStatus,
    VolcengineCredential, VolcengineProvider,
};
use siumai_transport::EndpointConfig;

fn test_provider(base_url: &str) -> VolcengineProvider {
    VolcengineProvider::builder(VolcengineCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit(base_url).expect("local endpoint"))
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("volcengine-media-test").expect("replay domain"),
        ))
        .build()
        .expect("provider")
}

#[tokio::test]
async fn portable_image_adapter_projects_one_bounded_seedream_call() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/api/v3/images/generations")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(format!(r#"\"model\":\"{SEEDREAM_5_0_260128}\""#)),
            mockito::Matcher::Regex(r#"\"prompt\":\"draw a small moon\""#.to_string()),
            mockito::Matcher::Regex(r#"\"size\":\"1024x1024\""#.to_string()),
            mockito::Matcher::Regex(r#"\"output_format\":\"png\""#.to_string()),
            mockito::Matcher::Regex(r#"\"response_format\":\"b64_json\""#.to_string()),
            mockito::Matcher::Regex(r#"\"watermark\":false"#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"created":1786118400,"data":[{"b64_json":"aW1hZ2U="}]}"#)
        .expect(1)
        .create_async()
        .await;
    let provider = test_provider(&format!("{}/api/v3", server.url()));
    let request = ImageRequest::new("draw a small moon")
        .expect("request")
        .with_size(1024, 1024)
        .expect("size")
        .with_format("image/png")
        .expect("format");
    let options = ArkImageOptions::new().with_watermark(false);

    let response = provider
        .image(SEEDREAM_5_0_260128)
        .expect("model")
        .generate_image(
            request,
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("options")),
        )
        .await
        .expect("response");

    assert_eq!(response.images.len(), 1);
    assert_eq!(response.images[0].media_type, "image/png");
    assert!(matches!(
        &response.images[0].data,
        MediaData::Bytes(bytes) if bytes.as_ref() == b"image"
    ));
    assert!(provider.registration().supports_family(ModelFamily::Image));
    mock.assert_async().await;
}

#[tokio::test]
async fn native_video_tasks_cover_create_retrieve_list_and_delete() {
    let mut server = mockito::Server::new_async().await;
    let create = server
        .mock("POST", "/api/v3/contents/generations/tasks")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(format!(r#"\"model\":\"{DREAMINA_SEEDANCE_2_0_260128}\""#)),
            mockito::Matcher::Regex(r#"\"text\":\"ocean at sunrise\""#.to_string()),
            mockito::Matcher::Regex(r#"\"return_last_frame\":true"#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"id":"cgt-media-1"}"#)
        .expect(1)
        .create_async()
        .await;
    let retrieve = server
        .mock("GET", "/api/v3/contents/generations/tasks/cgt-media-1")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(format!(
            r#"{{"id":"cgt-media-1","model":"{DREAMINA_SEEDANCE_2_0_260128}","status":"succeeded","content":{{"video_url":"https://media.example.test/private.mp4"}},"usage":{{"completion_tokens":42}}}}"#
        ))
        .expect(1)
        .create_async()
        .await;
    let list = server
        .mock("GET", "/api/v3/contents/generations/tasks")
        .match_query("task_ids=cgt-media-1&status=succeeded&page_num=1&page_size=10")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(format!(
            r#"{{"items":[{{"id":"cgt-media-1","model":"{DREAMINA_SEEDANCE_2_0_260128}","status":"succeeded"}}],"total":1,"page_num":1,"page_size":10}}"#
        ))
        .expect(1)
        .create_async()
        .await;
    let delete = server
        .mock("DELETE", "/api/v3/contents/generations/tasks/cgt-media-1")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;
    let provider = test_provider(&format!("{}/api/v3", server.url()));
    let tasks = provider.video_tasks();
    let created = tasks
        .create(
            ArkVideoCreateRequest::text(DREAMINA_SEEDANCE_2_0_260128, "ocean at sunrise")
                .expect("request")
                .with_return_last_frame(true),
        )
        .await
        .expect("create");
    let task = tasks.retrieve(&created.id).await.expect("retrieve");
    assert_eq!(task.status, ArkVideoTaskStatus::Succeeded);
    assert!(
        task.content
            .as_ref()
            .and_then(|value| value.video_url.as_ref())
            .is_some()
    );
    assert!(!format!("{task:?}").contains("private.mp4"));

    let query = ArkVideoTaskListQuery::new()
        .with_task_ids([created.id.clone()])
        .expect("ids")
        .with_status(ArkVideoTaskStatus::Succeeded)
        .expect("status")
        .with_page(1, 10)
        .expect("page");
    let listed = tasks.list(&query).await.expect("list");
    assert_eq!(listed.items.len(), 1);
    assert_eq!(listed.total, Some(1));
    let deleted = tasks.delete(&created.id).await.expect("delete");
    assert_eq!(deleted.task_id, created.id);

    create.assert_async().await;
    retrieve.assert_async().await;
    list.assert_async().await;
    delete.assert_async().await;
}
