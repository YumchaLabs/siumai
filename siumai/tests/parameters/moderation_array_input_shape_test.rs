//! OpenAI Moderation request mapping test for array inputs.
//!
//! References:
//! - OpenAI Moderations API: https://platform.openai.com/docs/api-reference/moderations

use siumai::transformers::request::RequestTransformer;

#[test]
fn moderation_accepts_array_input_and_maps_to_input_array() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::types::ModerationRequest;

    let req = ModerationRequest {
        input: String::new(),
        inputs: Some(vec!["你好".into(), "Hello".into()]),
        model: None,
    };

    let json = OpenAiRequestTransformer
        .transform_moderation(&req)
        .expect("transform moderation");

    assert!(json["model"].is_string());
    assert!(json["input"].is_array());
    let arr = json["input"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], "你好");
    assert_eq!(arr[1], "Hello");
}
