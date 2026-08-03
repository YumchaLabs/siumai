use siumai_spec::types::{
    CompletionRequest, EmbeddingRequest, FileListQuery, RerankRequest, SkillUploadFile,
    SkillUploadRequest,
};

#[test]
fn request_header_helpers_use_empty_http_override_config() {
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
