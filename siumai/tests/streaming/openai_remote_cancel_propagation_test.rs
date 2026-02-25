use futures_util::StreamExt;
use siumai::prelude::unified::*;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_remote_cancel_propagates_through_siumai_wrapper() {
    let server = MockServer::start().await;

    // Streaming `/responses` request (SSE over HTTP).
    let sse = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0,\"output\":[]}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse, "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    // Remote cancel endpoint.
    Mock::given(method("POST"))
        .and(path("/v1/responses/resp_1/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-test")
        .build()
        .await
        .unwrap();

    let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_streaming(true)
        .with_provider_option(
            "openai",
            serde_json::json!({
                "responsesApi": {
                    "enabled": true
                }
            }),
        );

    let ChatStreamHandle { mut stream, cancel } =
        client.chat_stream_request_with_cancel(req).await.unwrap();

    // Wait until we see response metadata (so the response id is known), then cancel.
    while let Some(item) = stream.next().await {
        let ev = item.unwrap();
        if let ChatStreamEvent::Custom { event_type, .. } = ev
            && event_type == "openai:response-metadata"
        {
            break;
        }
    }

    cancel.cancel();
    drop(stream);

    // Wait for the async remote cancel task to fire.
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let requests = server.received_requests().await.expect("requests");
            if requests
                .iter()
                .any(|r| r.url.path() == "/v1/responses/resp_1/cancel")
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("expected HTTP cancel request");
}
