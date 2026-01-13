#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use siumai::extensions::FileManagementCapability;
use siumai::extensions::types::{FileListQuery, FileUploadRequest};
use siumai::prelude::unified::Siumai;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use wiremock::matchers::{
    body_string_contains, header, header_exists, header_regex, method, path, query_param,
};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("files")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[tokio::test]
async fn openai_files_lifecycle_upload_list_retrieve_content_delete() {
    let server = MockServer::start().await;

    let file_body: serde_json::Value = read_json(fixtures_dir().join("file.json"));
    let list_body: serde_json::Value = read_json(fixtures_dir().join("list.json"));

    // Upload (multipart)
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header_exists("content-type"))
        .and(header_regex("content-type", "multipart/form-data"))
        .and(body_string_contains("name=\"purpose\""))
        .and(body_string_contains("assistants"))
        .and(body_string_contains("name=\"file\""))
        .and(body_string_contains("hello.txt"))
        .and(body_string_contains("hello"))
        .respond_with(ResponseTemplate::new(200).set_body_json(file_body.clone()))
        .expect(1)
        .mount(&server)
        .await;

    // List with query params
    Mock::given(method("GET"))
        .and(path("/v1/files"))
        .and(query_param("purpose", "assistants"))
        .and(query_param("limit", "10"))
        .and(query_param("after", "cursor123"))
        .and(query_param("order", "desc"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(list_body))
        .expect(1)
        .mount(&server)
        .await;

    // Retrieve
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(file_body.clone()))
        .expect(1)
        .mount(&server)
        .await;

    // Content
    Mock::given(method("GET"))
        .and(path("/v1/files/file_123/content"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(b"hello".to_vec())
                .insert_header("content-type", "application/octet-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    // Delete (body ignored)
    Mock::given(method("DELETE"))
        .and(path("/v1/files/file_123"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o")
        .build()
        .await
        .expect("build ok");

    let uploaded = client
        .upload_file(FileUploadRequest {
            content: b"hello".to_vec(),
            filename: "hello.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: HashMap::new(),
            http_config: None,
        })
        .await
        .expect("upload ok");
    assert_eq!(uploaded.id, "file_123");
    assert_eq!(uploaded.bytes, 5);

    let list = client
        .list_files(Some(FileListQuery {
            purpose: Some("assistants".to_string()),
            limit: Some(10),
            after: Some("cursor123".to_string()),
            order: Some("desc".to_string()),
            http_config: None,
        }))
        .await
        .expect("list ok");
    assert_eq!(list.files.len(), 1);
    assert_eq!(list.files[0].id, "file_123");

    let retrieved = client
        .retrieve_file("file_123".to_string())
        .await
        .expect("retrieve ok");
    assert_eq!(retrieved.filename, "hello.txt");

    let content = client
        .get_file_content("file_123".to_string())
        .await
        .expect("content ok");
    assert_eq!(content, b"hello");

    let deleted = client
        .delete_file("file_123".to_string())
        .await
        .expect("delete ok");
    assert!(deleted.deleted);
    assert_eq!(deleted.id, "file_123");
}
