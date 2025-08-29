# Image Generation Examples

This directory contains examples demonstrating image generation capabilities across different AI providers.

## Overview

The Siumai library supports image generation through multiple providers:

- **OpenAI DALL-E**: High-quality image generation with DALL-E 2 and DALL-E 3 models
- **SiliconFlow**: Cost-effective image generation with multiple models including Kolors, FLUX, and Stable Diffusion

## Features

### OpenAI DALL-E
- **Models**: `dall-e-2`, `dall-e-3`, `gpt-image-1`
- **Capabilities**: Image generation, editing, variations
- **Sizes**: 256x256, 512x512, 1024x1024, 1792x1024, 1024x1792, 2048x2048
- **Formats**: URL, Base64 JSON

### SiliconFlow
- **Models**: `Kwai-Kolors/Kolors`, `black-forest-labs/FLUX.1-schnell`, `stabilityai/stable-diffusion-3.5-large`
- **Capabilities**: Image generation only
- **Sizes**: 1024x1024, 960x1280, 768x1024, 720x1440, 720x1280
- **Formats**: URL only
- **Special Features**: Negative prompts, guidance scale, inference steps control

## Quick Start

### Basic Image Generation

```rust
use siumai::prelude::*;
use siumai::traits::ImageGenerationCapability;
use siumai::types::ImageGenerationRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // OpenAI DALL-E
    let client = LlmBuilder::new()
        .openai()
        .api_key("your-openai-api-key")
        .build()
        .await?;

    let request = ImageGenerationRequest {
        prompt: "A beautiful sunset over mountains".to_string(),
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some("dall-e-3".to_string()),
        ..Default::default()
    };

    let response = client.generate_images(request).await?;
    
    for image in response.images {
        if let Some(url) = image.url {
            println!("Generated image: {}", url);
        }
    }

    Ok(())
}
```

### SiliconFlow with Advanced Parameters

```rust
use siumai::prelude::*;
use siumai::providers::openai_compatible::siliconflow;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .siliconflow()
        .api_key("your-siliconflow-api-key")
        .build()
        .await?;

    let request = ImageGenerationRequest {
        prompt: "A futuristic cityscape at night".to_string(),
        negative_prompt: Some("blurry, low quality".to_string()),
        size: Some("1024x1024".to_string()),
        count: 1,
        model: Some(siliconflow::KOLORS.to_string()),
        steps: Some(20),
        guidance_scale: Some(7.5),
        seed: Some(42),
        ..Default::default()
    };

    let response = client.generate_images(request).await?;
    println!("Generated {} images", response.images.len());

    Ok(())
}
```

## Environment Variables

Set the following environment variables to run the examples:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
```

## Running Examples

```bash
# Run the main showcase
cargo run --example image_generation_showcase

# Run with environment variables
OPENAI_API_KEY=your-key SILICONFLOW_API_KEY=your-key cargo run --example image_generation_showcase
```

## Model Constants

### OpenAI Models
```rust
use siumai::models::openai;

// Available models
openai::DALL_E_2      // "dall-e-2"
openai::DALL_E_3      // "dall-e-3"
openai::GPT_IMAGE_1   // "gpt-image-1"
```

### SiliconFlow Models
```rust
use siumai::providers::openai_compatible::siliconflow;

// Available models
siliconflow::KOLORS                      // "Kwai-Kolors/Kolors"
siliconflow::FLUX_1_SCHNELL             // "black-forest-labs/FLUX.1-schnell"
siliconflow::STABLE_DIFFUSION_3_5_LARGE // "stabilityai/stable-diffusion-3.5-large"
```

## Provider Capabilities

### Checking Capabilities

```rust
if let Some(image_cap) = client.as_image_generation_capability() {
    println!("Supported sizes: {:?}", image_cap.get_supported_sizes());
    println!("Supported formats: {:?}", image_cap.get_supported_formats());
    println!("Image editing: {}", image_cap.supports_image_editing());
    println!("Image variations: {}", image_cap.supports_image_variations());
}
```

### Provider Differences

| Feature | OpenAI DALL-E | SiliconFlow |
|---------|---------------|-------------|
| Image Generation | ✅ | ✅ |
| Image Editing | ✅ | ❌ |
| Image Variations | ✅ | ❌ |
| Negative Prompts | ❌ | ✅ |
| Guidance Scale | ❌ | ✅ |
| Inference Steps | ❌ | ✅ |
| Seed Control | ❌ | ✅ |

## Error Handling

```rust
match client.generate_images(request).await {
    Ok(response) => {
        println!("Success! Generated {} images", response.images.len());
    }
    Err(LlmError::InvalidInput(msg)) => {
        println!("Invalid input: {}", msg);
    }
    Err(LlmError::ApiError { code, message, .. }) => {
        println!("API error {}: {}", code, message);
    }
    Err(e) => {
        println!("Other error: {}", e);
    }
}
```

## Best Practices

1. **Descriptive Prompts**: Use detailed, specific prompts for better results
2. **Model Selection**: Choose the right model for your use case
3. **Size Optimization**: Select appropriate image sizes for your application
4. **Error Handling**: Always handle potential API errors gracefully
5. **Rate Limiting**: Be mindful of API rate limits
6. **Cost Management**: Monitor usage, especially with premium models

## Troubleshooting

### Common Issues

1. **Invalid Size Error**: Check supported sizes for your chosen model
2. **Too Many Images**: DALL-E 3 only supports generating 1 image at a time
3. **API Key Issues**: Ensure your API key is valid and has sufficient credits
4. **Network Errors**: Check your internet connection and API endpoint availability

### Debug Mode

Enable debug logging to see detailed request/response information:

```rust
tracing_subscriber::fmt::init();
```

## Additional Resources

- [OpenAI DALL-E API Documentation](https://platform.openai.com/docs/api-reference/images)
- [SiliconFlow API Documentation](https://docs.siliconflow.cn/)
- [Siumai Library Documentation](../../README.md)
