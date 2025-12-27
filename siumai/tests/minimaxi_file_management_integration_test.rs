#![cfg(feature = "minimaxi")]
//! Real MiniMaxi file management integration test (ignored by default).
//!
//! This test exercises the MiniMaxi File Management API against the real service:
//! upload -> list -> retrieve_content -> delete.
//!
//! Required env vars:
//! - `MINIMAXI_API_KEY`
//!
//! Optional env vars:
//! - `MINIMAXI_BASE_URL` (defaults to `https://api.minimaxi.com/anthropic`)

use siumai::prelude::*;
use siumai::traits::FileManagementCapability;
use siumai::types::{FileListQuery, FileUploadRequest};
use std::collections::HashMap;
use std::env;

#[tokio::test]
#[ignore]
async fn test_minimaxi_files_real_lifecycle() {
    let api_key = match env::var("MINIMAXI_API_KEY") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return,
    };

    let mut builder = LlmBuilder::new().minimaxi().api_key(api_key);
    if let Ok(base_url) = env::var("MINIMAXI_BASE_URL") {
        if !base_url.trim().is_empty() {
            builder = builder.base_url(base_url);
        }
    }

    let client = builder.build().await.expect("Failed to build MiniMaxi client");

    let mut created_file_id: Option<String> = None;
    let test_result: Result<(), siumai::error::LlmError> = async {
        let uploaded = client
            .upload_file(FileUploadRequest {
                content: b"hello".to_vec(),
                filename: "siumai-minimaxi-test.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                purpose: "t2a_async_input".to_string(),
                metadata: HashMap::new(),
                http_config: None,
            })
            .await?;

        created_file_id = Some(uploaded.id.clone());

        let list = client
            .list_files(Some(FileListQuery {
                purpose: Some("t2a_async_input".to_string()),
                ..Default::default()
            }))
            .await?;
        assert!(
            list.files.iter().any(|f| f.id == uploaded.id),
            "Uploaded file should appear in list_files()"
        );

        let content = client.get_file_content(uploaded.id.clone()).await?;
        assert_eq!(content, b"hello");

        let deleted = client.delete_file(uploaded.id).await?;
        assert!(deleted.deleted);

        Ok(())
    }
    .await;

    if let Some(id) = created_file_id {
        let _ = client.delete_file(id).await;
    }

    test_result.expect("MiniMaxi files real lifecycle failed");
}

