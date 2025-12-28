//! Image Generation Integration Tests
//!
//! Tests for image generation capabilities across different providers.

use siumai::prelude::*;
use siumai::types::ImageGenerationRequest;

#[tokio::test]
async fn test_openai_image_generation_request_conversion() {
    // No client needed here; just validate request mapping readiness

    // Test basic request
    let request = ImageGenerationRequest {
        prompt: "A beautiful sunset".to_string(),
        negative_prompt: Some("dark, gloomy".to_string()),
        size: Some("1024x1024".to_string()),
        count: 2,
        model: Some("dall-e-3".to_string()),
        quality: Some("hd".to_string()),
        style: Some("vivid".to_string()),
        seed: Some(12345),
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: std::collections::HashMap::new(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    // Test that the request can be created without errors
    assert_eq!(request.prompt, "A beautiful sunset");
    assert_eq!(request.count, 2);
    assert_eq!(request.model, Some("dall-e-3".to_string()));
}

#[tokio::test]
async fn test_siliconflow_image_generation_request_conversion() {
    // No client needed here; just validate request mapping readiness

    // Test SiliconFlow-specific request
    let request = ImageGenerationRequest {
        prompt: "A futuristic city".to_string(),
        negative_prompt: Some("old, vintage".to_string()),
        size: Some("960x1280".to_string()),
        count: 1,
        model: Some("Kwai-Kolors/Kolors".to_string()),
        quality: None,
        style: None,
        seed: Some(67890),
        steps: Some(20),
        guidance_scale: Some(7.5),
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: std::collections::HashMap::new(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    // Test that the request can be created without errors
    assert_eq!(request.prompt, "A futuristic city");
    assert_eq!(request.count, 1);
    assert_eq!(request.model, Some("Kwai-Kolors/Kolors".to_string()));
}

#[tokio::test]
async fn test_openai_client_image_generation_capability() {
    use siumai::client::LlmClient;
    use siumai::providers::openai::OpenAiClient;

    let config = siumai::providers::openai::OpenAiConfig::new("test-key")
        .with_base_url("https://api.openai.com/v1"); // Explicitly use OpenAI endpoint
    let client = OpenAiClient::new(config, reqwest::Client::new());

    // Test that the client provides image generation capability
    let image_capability = client.as_image_generation_capability();
    assert!(image_capability.is_some());

    if let Some(_capability) = image_capability {
        let extras = client.as_image_extras();
        assert!(extras.is_some());
        let extras = extras.unwrap();

        // Test supported sizes
        let sizes = extras.get_supported_sizes();
        assert!(!sizes.is_empty());
        assert!(sizes.contains(&"1024x1024".to_string()));

        // Test supported formats
        let formats = extras.get_supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&"url".to_string()));

        // Test capabilities
        assert!(extras.supports_image_editing());
        assert!(extras.supports_image_variations());
    }
}

#[tokio::test]
async fn test_siliconflow_client_image_generation_capability() {
    use siumai::client::LlmClient;
    use siumai::providers::openai::OpenAiClient;

    // Create SiliconFlow client using OpenAI client with SiliconFlow endpoint
    let config = siumai::providers::openai::OpenAiConfig::new("test-key")
        .with_base_url("https://api.siliconflow.cn/v1");
    let client = OpenAiClient::new(config, reqwest::Client::new());

    // Test that the client provides image generation capability
    let image_capability = client.as_image_generation_capability();
    assert!(image_capability.is_some());

    if let Some(_capability) = image_capability {
        let extras = client.as_image_extras();
        assert!(extras.is_some());
        let extras = extras.unwrap();

        // Test SiliconFlow-specific supported sizes
        let sizes = extras.get_supported_sizes();
        assert!(!sizes.is_empty());
        assert!(sizes.contains(&"1024x1024".to_string()));
        assert!(sizes.contains(&"960x1280".to_string()));

        // Test SiliconFlow-specific supported formats
        let formats = extras.get_supported_formats();
        assert_eq!(formats, vec!["url".to_string()]);

        // Test SiliconFlow-specific capabilities
        assert!(!extras.supports_image_editing()); // SiliconFlow doesn't support editing
        assert!(!extras.supports_image_variations()); // SiliconFlow doesn't support variations
    }
}

#[tokio::test]
async fn test_image_generation_builder_integration() {
    // Test OpenAI builder
    let openai_result = LlmBuilder::new()
        .openai()
        .api_key("test-key")
        .model("gpt-4")
        .build()
        .await;

    assert!(openai_result.is_ok());
    if let Ok(client) = openai_result {
        assert!(client.as_image_generation_capability().is_some());
    }

    // Test SiliconFlow builder
    let siliconflow_result = LlmBuilder::new()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model("deepseek-chat")
        .build()
        .await;

    assert!(siliconflow_result.is_ok());
    if let Ok(client) = siliconflow_result {
        assert!(client.as_image_generation_capability().is_some());
    }
}

#[test]
fn test_image_generation_model_constants() {
    use siumai::constants::openai_compatible::siliconflow;

    // Test SiliconFlow model constants
    assert_eq!(siliconflow::KOLORS, "Kwai-Kolors/Kolors");
    assert_eq!(
        siliconflow::FLUX_1_SCHNELL,
        "black-forest-labs/FLUX.1-schnell"
    );
    assert_eq!(
        siliconflow::STABLE_DIFFUSION_3_5_LARGE,
        "stabilityai/stable-diffusion-3.5-large"
    );

    // Test that image models are included in all models
    let all_models = siliconflow::all_models();
    assert!(all_models.contains(&siliconflow::KOLORS.to_string()));
    assert!(all_models.contains(&siliconflow::FLUX_1_SCHNELL.to_string()));
    assert!(all_models.contains(&siliconflow::STABLE_DIFFUSION_3_5_LARGE.to_string()));

    // Test image-specific models
    let image_models = siliconflow::all_image_models();
    assert!(image_models.contains(&siliconflow::KOLORS.to_string()));
    assert!(image_models.contains(&siliconflow::FLUX_1_SCHNELL.to_string()));
    assert!(image_models.contains(&siliconflow::STABLE_DIFFUSION_3_5_LARGE.to_string()));
}

#[test]
fn test_image_generation_request_validation() {
    // Test valid request
    let valid_request = ImageGenerationRequest {
        prompt: "A beautiful landscape".to_string(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some("dall-e-3".to_string()),
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: std::collections::HashMap::new(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    assert!(!valid_request.prompt.is_empty());
    assert!(valid_request.count > 0);

    // Test request with negative prompt
    let request_with_negative = ImageGenerationRequest {
        prompt: "A sunny day".to_string(),
        negative_prompt: Some("rain, clouds".to_string()),
        size: Some("512x512".to_string()),
        count: 2,
        model: Some("Kwai-Kolors/Kolors".to_string()),
        quality: None,
        style: None,
        seed: Some(42),
        steps: Some(25),
        guidance_scale: Some(8.0),
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: std::collections::HashMap::new(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    assert_eq!(
        request_with_negative.negative_prompt,
        Some("rain, clouds".to_string())
    );
    assert_eq!(request_with_negative.seed, Some(42));
}
