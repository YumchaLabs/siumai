#![cfg(feature = "openai-websocket")]

use siumai::prelude::unified::*;

#[tokio::test]
async fn siumai_builder_can_build_openai_websocket_session_without_connecting() {
    let session = Siumai::builder()
        .openai()
        .api_key("test")
        .base_url("http://127.0.0.1:1/v1")
        .model("gpt-test")
        .use_openai_websocket_session()
        .await
        .expect("session build should not require network");

    assert!(session.remote_cancel());
}

#[tokio::test]
async fn siumai_builder_can_build_openai_incremental_websocket_session_without_connecting() {
    let _inc = Siumai::builder()
        .openai()
        .api_key("test")
        .base_url("http://127.0.0.1:1/v1")
        .model("gpt-test")
        .use_openai_incremental_websocket_session()
        .await
        .expect("incremental session build should not require network");
}
