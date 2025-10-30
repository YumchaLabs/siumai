use siumai::execution::RequestTransformer;
use siumai::types::{ChatRequest, CommonParams};

#[test]
fn gemini_transformer_moves_common_params_into_generation_config() {
    // Prepare Gemini config with default base and bare model id
    let mut cfg = siumai::providers::gemini::types::GeminiConfig::default();
    cfg = cfg.with_model("gemini-2.5-flash".to_string());

    // Build a simple chat request with unified common_params set
    let cp = CommonParams {
        model: "gemini-2.5-flash".to_string(),
        temperature: Some(0.3),
        max_tokens: Some(999),
        max_completion_tokens: None,
        top_p: Some(0.7),
        stop_sequences: Some(vec!["END".to_string()]),
        seed: None,
    };
    let req = ChatRequest::builder()
        .messages(vec![siumai::user!("Hello")])
        .common_params(cp)
        .build();

    // Transform via provider request transformer
    let tx = siumai::providers::gemini::transformers::GeminiRequestTransformer { config: cfg };
    let body = tx.transform_chat(&req).expect("transform should succeed");

    // Assert model id remains bare (path builder adds /models/{id})
    assert_eq!(body["model"], "gemini-2.5-flash");

    // Assert generationConfig fields picked from common_params
    let gen_cfg = body
        .get("generationConfig")
        .expect("generationConfig present");
    assert_eq!(gen_cfg["temperature"], 0.3);
    assert_eq!(gen_cfg["topP"], 0.7);
    assert_eq!(gen_cfg["maxOutputTokens"], 999);
    assert_eq!(gen_cfg["stopSequences"], serde_json::json!(["END"]));
}
