//! Custom Configurations Example
//!
//! This example demonstrates advanced configuration patterns for production deployments,
//! including custom parameter mapping, provider-specific optimizations, and performance tuning.

use siumai::transformers::request::RequestTransformer;
use siumai::types::ChatRequest;
use siumai::types::{CommonParams, ProviderParams, ProviderType};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom Configurations Example");
    println!("=================================\n");

    // 1. Environment-based Configuration
    demonstrate_environment_configs().await?;

    // 2. Use Case Specific Configurations
    demonstrate_use_case_configs().await?;

    // 3. Performance Optimization Configurations
    demonstrate_performance_configs().await?;

    // 4. Multi-Provider Fallback Configuration
    demonstrate_fallback_configs().await?;

    // 5. Custom Parameter Mapping
    demonstrate_custom_parameter_mapping().await?;

    Ok(())
}

/// Demonstrates environment-based configurations
async fn demonstrate_environment_configs() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 1. Environment-based Configuration");
    println!("   Different settings for development, staging, and production\n");

    // Development configuration
    println!("   🔧 Development Environment:");
    let dev_config = create_development_config();
    print_config_summary("Development", &dev_config);

    // Staging configuration
    println!("\n   🧪 Staging Environment:");
    let staging_config = create_staging_config();
    print_config_summary("Staging", &staging_config);

    // Production configuration
    println!("\n   🚀 Production Environment:");
    let prod_config = create_production_config();
    print_config_summary("Production", &prod_config);

    println!("\n");
    Ok(())
}

/// Demonstrates use case specific configurations
async fn demonstrate_use_case_configs() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 2. Use Case Specific Configurations");
    println!("   Optimized settings for different application types\n");

    let use_cases = vec![
        ("Creative Writing", create_creative_config()),
        ("Code Generation", create_code_config()),
        ("Data Analysis", create_analysis_config()),
        ("Customer Support", create_support_config()),
        ("Content Moderation", create_moderation_config()),
    ];

    for (use_case, config) in use_cases {
        println!("   📝 {}", use_case);
        print_config_summary(use_case, &config);
        println!();
    }

    Ok(())
}

/// Demonstrates performance optimization configurations
async fn demonstrate_performance_configs() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ 3. Performance Optimization Configurations");
    println!("   Settings optimized for speed, cost, or quality\n");

    // Speed-optimized configuration
    println!("   🏃 Speed-Optimized (Low Latency):");
    let speed_config = ConfigProfile {
        name: "Speed".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o-mini".to_string(),
        temperature: 0.3,
        max_tokens: 500,
        timeout: Duration::from_secs(10),
        retry_attempts: 1,
        description: "Optimized for fastest response times".to_string(),
    };
    print_config_summary("Speed", &speed_config);

    // Cost-optimized configuration
    println!("\n   💰 Cost-Optimized (Budget-Friendly):");
    let cost_config = ConfigProfile {
        name: "Cost".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o-mini".to_string(),
        temperature: 0.7,
        max_tokens: 1000,
        timeout: Duration::from_secs(30),
        retry_attempts: 3,
        description: "Optimized for cost efficiency".to_string(),
    };
    print_config_summary("Cost", &cost_config);

    // Quality-optimized configuration
    println!("\n   🎯 Quality-Optimized (Best Results):");
    let quality_config = ConfigProfile {
        name: "Quality".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o".to_string(),
        temperature: 0.1,
        max_tokens: 2000,
        timeout: Duration::from_secs(60),
        retry_attempts: 2,
        description: "Optimized for highest quality output".to_string(),
    };
    print_config_summary("Quality", &quality_config);

    println!("\n");
    Ok(())
}

/// Demonstrates multi-provider fallback configuration
async fn demonstrate_fallback_configs() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 4. Multi-Provider Fallback Configuration");
    println!("   Automatic fallback between providers for reliability\n");

    let fallback_chain = vec![
        (
            "Primary",
            ProviderType::OpenAi,
            "gpt-4o",
            "Main provider for best quality",
        ),
        (
            "Secondary",
            ProviderType::Anthropic,
            "claude-3-5-sonnet-20241022",
            "Fallback for high-quality responses",
        ),
        (
            "Tertiary",
            ProviderType::OpenAi,
            "gpt-4o-mini",
            "Fast fallback for basic responses",
        ),
        (
            "Emergency",
            ProviderType::Ollama,
            "llama3.2",
            "Local fallback when all cloud providers fail",
        ),
    ];

    println!("   🔗 Fallback Chain:");
    for (i, (level, provider, model, description)) in fallback_chain.iter().enumerate() {
        println!("   {}. {} ({:?})", i + 1, level, provider);
        println!("      Model: {}", model);
        println!("      Purpose: {}", description);
        println!();
    }

    println!("   💡 Fallback Strategy Benefits:");
    println!("     • High availability and reliability");
    println!("     • Automatic error recovery");
    println!("     • Cost optimization through provider selection");
    println!("     • Performance optimization based on load");

    println!("\n");
    Ok(())
}

/// Demonstrates custom parameter mapping
async fn demonstrate_custom_parameter_mapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎛️  5. Custom Parameter Mapping");
    println!("   Advanced parameter customization for specific needs\n");

    // Custom parameter sets for different scenarios
    let scenarios = vec![
        ("High Creativity", create_creative_params()),
        ("Precise Analysis", create_precise_params()),
        ("Balanced General", create_balanced_params()),
        ("Experimental", create_experimental_params()),
    ];

    for (scenario, params) in scenarios {
        println!("   🔬 {}", scenario);

        // Show how parameters map to different providers
        for provider in [
            ProviderType::OpenAi,
            ProviderType::Anthropic,
            ProviderType::Gemini,
        ] {
            let mapped = convert_params_via_transformers(&params.0, Some(&params.1), &provider)?;
            println!(
                "     {:?}: {} parameters",
                provider,
                mapped.as_object().unwrap().len()
            );
        }
        println!();
    }

    println!("   🔧 Custom Parameter Benefits:");
    println!("     • Fine-tuned control over AI behavior");
    println!("     • Provider-specific optimizations");
    println!("     • Consistent behavior across providers");
    println!("     • Easy parameter experimentation");

    println!("\n✨ Custom configurations complete! You now understand how to");
    println!("   create sophisticated configurations for production deployments.");

    Ok(())
}

fn convert_params_via_transformers(
    common: &CommonParams,
    provider_params: Option<&ProviderParams>,
    provider: &ProviderType,
) -> Result<serde_json::Value, siumai::LlmError> {
    let req = ChatRequest {
        messages: vec![],
        tools: None,
        common_params: common.clone(),
        provider_params: provider_params.cloned(),
        http_config: None,
        web_search: None,
        stream: false,
    };

    match provider {
        ProviderType::OpenAi => {
            let tx = siumai::providers::openai::transformers::OpenAiRequestTransformer;
            tx.transform_chat(&req)
        }
        ProviderType::Anthropic => {
            let tx =
                siumai::providers::anthropic::transformers::AnthropicRequestTransformer::default();
            tx.transform_chat(&req)
        }
        ProviderType::Gemini => {
            let cfg = siumai::providers::gemini::types::GeminiConfig {
                api_key: "test".to_string(),
                base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                model: common.model.clone(),
                generation_config: None,
                safety_settings: None,
                timeout: Some(30),
                http_config: Some(siumai::types::HttpConfig::default()),
            };
            let tx =
                siumai::providers::gemini::transformers::GeminiRequestTransformer { config: cfg };
            tx.transform_chat(&req)
        }
        other => Err(siumai::LlmError::UnsupportedOperation(format!(
            "Provider {:?} not supported in this example",
            other
        ))),
    }
}

// Configuration profile structure
#[derive(Debug, Clone)]
struct ConfigProfile {
    name: String,
    provider: ProviderType,
    model: String,
    temperature: f32,
    max_tokens: u32,
    timeout: Duration,
    retry_attempts: u32,
    description: String,
}

// Helper functions for creating different configurations
fn create_development_config() -> ConfigProfile {
    ConfigProfile {
        name: "Development".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o-mini".to_string(),
        temperature: 0.7,
        max_tokens: 1000,
        timeout: Duration::from_secs(30),
        retry_attempts: 2,
        description: "Fast iteration with reasonable costs".to_string(),
    }
}

fn create_staging_config() -> ConfigProfile {
    ConfigProfile {
        name: "Staging".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o".to_string(),
        temperature: 0.5,
        max_tokens: 1500,
        timeout: Duration::from_secs(45),
        retry_attempts: 3,
        description: "Production-like testing environment".to_string(),
    }
}

fn create_production_config() -> ConfigProfile {
    ConfigProfile {
        name: "Production".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o".to_string(),
        temperature: 0.3,
        max_tokens: 2000,
        timeout: Duration::from_secs(60),
        retry_attempts: 3,
        description: "Optimized for reliability and quality".to_string(),
    }
}

fn create_creative_config() -> ConfigProfile {
    ConfigProfile {
        name: "Creative".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o".to_string(),
        temperature: 0.9,
        max_tokens: 2000,
        timeout: Duration::from_secs(45),
        retry_attempts: 2,
        description: "High creativity for writing and brainstorming".to_string(),
    }
}

fn create_code_config() -> ConfigProfile {
    ConfigProfile {
        name: "Code".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o".to_string(),
        temperature: 0.1,
        max_tokens: 1500,
        timeout: Duration::from_secs(30),
        retry_attempts: 2,
        description: "Precise and accurate code generation".to_string(),
    }
}

fn create_analysis_config() -> ConfigProfile {
    ConfigProfile {
        name: "Analysis".to_string(),
        provider: ProviderType::Anthropic,
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: 0.2,
        max_tokens: 2000,
        timeout: Duration::from_secs(60),
        retry_attempts: 2,
        description: "Detailed analysis and reasoning".to_string(),
    }
}

fn create_support_config() -> ConfigProfile {
    ConfigProfile {
        name: "Support".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o-mini".to_string(),
        temperature: 0.4,
        max_tokens: 800,
        timeout: Duration::from_secs(20),
        retry_attempts: 3,
        description: "Fast, helpful customer support responses".to_string(),
    }
}

fn create_moderation_config() -> ConfigProfile {
    ConfigProfile {
        name: "Moderation".to_string(),
        provider: ProviderType::OpenAi,
        model: "gpt-4o-mini".to_string(),
        temperature: 0.0,
        max_tokens: 200,
        timeout: Duration::from_secs(15),
        retry_attempts: 1,
        description: "Consistent content moderation decisions".to_string(),
    }
}

// Helper functions for parameter creation
fn create_creative_params() -> (CommonParams, ProviderParams) {
    let common = CommonParams {
        model: "gpt-4o".to_string(),
        temperature: Some(0.9),
        max_tokens: Some(2000),
        top_p: Some(0.95),
        ..Default::default()
    };

    let provider = ProviderParams::new()
        .with_param("frequency_penalty", 0.1)
        .with_param("presence_penalty", 0.2);

    (common, provider)
}

fn create_precise_params() -> (CommonParams, ProviderParams) {
    let common = CommonParams {
        model: "gpt-4o".to_string(),
        temperature: Some(0.1),
        max_tokens: Some(1500),
        top_p: Some(0.1),
        seed: Some(42),
        ..Default::default()
    };

    let provider = ProviderParams::new()
        .with_param("frequency_penalty", 0.0)
        .with_param("presence_penalty", 0.0);

    (common, provider)
}

fn create_balanced_params() -> (CommonParams, ProviderParams) {
    let common = CommonParams {
        model: "gpt-4o-mini".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        ..Default::default()
    };

    let provider = ProviderParams::new();

    (common, provider)
}

fn create_experimental_params() -> (CommonParams, ProviderParams) {
    let common = CommonParams {
        model: "gpt-4o".to_string(),
        temperature: Some(1.2),
        max_tokens: Some(1000),
        top_p: Some(0.8),
        ..Default::default()
    };

    let provider = ProviderParams::new()
        .with_param("frequency_penalty", 0.5)
        .with_param("presence_penalty", 0.3)
        .with_param("experimental_feature", true);

    (common, provider)
}

fn print_config_summary(_name: &str, config: &ConfigProfile) {
    println!("     • Config Name: {}", config.name);
    println!("     • Provider: {:?}", config.provider);
    println!("     • Model: {}", config.model);
    println!("     • Temperature: {}", config.temperature);
    println!("     • Max Tokens: {}", config.max_tokens);
    println!("     • Timeout: {:?}", config.timeout);
    println!("     • Retries: {}", config.retry_attempts);
    println!("     • Purpose: {}", config.description);
}
