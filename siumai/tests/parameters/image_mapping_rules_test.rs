//! Image mapping rules tests focusing on JSON body placement and param merging.

use siumai::execution::transformers::request::RequestTransformer;

#[test]
fn openai_image_json_mapping_and_merge() {
    use siumai::providers::openai::transformers::OpenAiRequestTransformer;
    use siumai::types::ImageGenerationRequest;

    let mut req = ImageGenerationRequest::default();
    req.prompt = "Draw a cat".into();
    req.count = 2;
    req.size = Some("1024x1024".into());
    req.response_format = Some("url".into());
    req.model = Some("gpt-image-1".into());
    req.negative_prompt = Some("dog".into());
    req.extra_params.insert("background".into(), serde_json::json!("white"));

    let json = OpenAiRequestTransformer.transform_image(&req).expect("map");
    assert_eq!(json["prompt"], "Draw a cat");
    assert_eq!(json["n"], 2);
    assert_eq!(json["size"], "1024x1024");
    assert_eq!(json["response_format"], "url");
    assert_eq!(json["model"], "gpt-image-1");
    assert_eq!(json["negative_prompt"], "dog");
    assert_eq!(json["background"], "white");
}

