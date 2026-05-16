use siumai_spec::types::{
    CompletionRequest, EmbeddingRequest, FileListQuery, RerankRequest, SkillUploadFile,
    SkillUploadRequest,
};
use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_source(relative_path: &str) -> String {
    let path = crate_root().join(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

#[test]
fn non_ai_sdk_request_carriers_remain_passive_data_contracts() {
    for relative_path in [
        "src/types/audio.rs",
        "src/types/completion.rs",
        "src/types/embedding/request.rs",
        "src/types/files.rs",
        "src/types/image.rs",
        "src/types/rerank.rs",
        "src/types/skills.rs",
    ] {
        let source = read_source(relative_path);

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
            "std::env",
            "CARGO_PKG_VERSION",
            "HttpClient",
            "ClientBuilder",
            "pub fn poll",
            "pub fn wait",
            "pub fn download",
            "pub fn fetch",
            "poll_",
            "wait_for",
            "download_",
            "fetch_",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative_path} must stay a passive spec data carrier and must not contain runtime/provider execution fragment `{forbidden}`"
            );
        }
    }
}

#[test]
fn request_header_helpers_use_empty_http_override_config() {
    for relative_path in [
        "src/types/completion.rs",
        "src/types/embedding/request.rs",
        "src/types/files.rs",
        "src/types/rerank.rs",
        "src/types/skills.rs",
    ] {
        let source = read_source(relative_path);

        assert!(
            !source.contains("HttpConfig::default"),
            "{relative_path} must not use HttpConfig::default() as a runtime-default shortcut"
        );
        assert!(
            source.contains("unwrap_or_else(HttpConfig::empty)"),
            "{relative_path} request header helpers should create override config from HttpConfig::empty()"
        );
    }

    let completion = CompletionRequest::new("hello").with_header("x-test", "1");
    let embedding = EmbeddingRequest::single("hello").with_header("x-test", "1");
    let files = FileListQuery::default().with_header("x-test", "1");
    let rerank = RerankRequest::new("rerank-model".into(), "query".into(), vec!["doc".into()])
        .with_header("x-test", "1");
    let skills = SkillUploadRequest::new(vec![SkillUploadFile::bytes("skill.md", vec![1])])
        .with_header("x-test", "1");

    for config in [
        completion.http_config.as_ref(),
        embedding.http_config.as_ref(),
        files.http_config.as_ref(),
        rerank.http_config.as_ref(),
        skills.http_config.as_ref(),
    ] {
        let config = config.expect("request http config");

        assert_eq!(config.headers.get("x-test").map(String::as_str), Some("1"));
        assert_eq!(config.timeout, None);
        assert_eq!(config.connect_timeout, None);
        assert_eq!(config.proxy, None);
        assert_eq!(config.user_agent, None);
        assert!(!config.stream_disable_compression);
    }
}
