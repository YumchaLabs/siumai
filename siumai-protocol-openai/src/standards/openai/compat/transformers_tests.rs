use super::adapter::OpenAiStandardAdapter;
use super::openai_config::OpenAiCompatibleConfig;
use super::transformers::CompatRequestTransformer;
use crate::execution::transformers::request::RequestTransformer;
use crate::types::RerankRequest;
use std::sync::Arc;

#[test]
fn openai_compatible_rerank_payload_fields() {
    let adapter = Arc::new(OpenAiStandardAdapter {
        base_url: String::new(),
    });
    let cfg = OpenAiCompatibleConfig::new(
        "siliconflow",
        "sk-test",
        "https://api.example/v1",
        adapter.clone(),
    )
    .with_model("BAAI/bge-reranker-v2-m3");

    let tx = CompatRequestTransformer {
        config: cfg,
        adapter,
    };

    let req = RerankRequest::new(
        "BAAI/bge-reranker-v2-m3".into(),
        "what is rust?".into(),
        vec!["doc1".into(), "doc2".into()],
    )
    .with_top_n(1)
    .with_instruction("focus on programming language".to_string())
    .with_return_documents(true)
    .with_max_chunks_per_doc(3)
    .with_overlap_tokens(30);

    let json = tx.transform_rerank(&req).expect("transform rerank");
    assert_eq!(json["model"], "BAAI/bge-reranker-v2-m3");
    assert_eq!(json["query"], "what is rust?");
    assert_eq!(json["documents"].as_array().unwrap().len(), 2);
    assert_eq!(json["top_n"], 1);
    assert_eq!(json["instruction"], "focus on programming language");
    assert_eq!(json["return_documents"], true);
    assert_eq!(json["max_chunks_per_doc"], 3);
    assert_eq!(json["overlap_tokens"], 30);
}
