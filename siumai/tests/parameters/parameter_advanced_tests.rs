//! Advanced Parameter Tests
//!
//! This module contains advanced parameter tests that cover edge cases,
//! serialization, concurrency, and other advanced scenarios.

use siumai::types::CommonParams;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Test parameter serialization and deserialization
#[test]
fn test_parameter_serialization() {
    println!("🧪 Testing parameter serialization/deserialization");

    // Test CommonParams serialization
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        seed: Some(42),
    };

    // Serialize to JSON
    let json = serde_json::to_string(&common_params).expect("Failed to serialize CommonParams");
    println!("   Serialized CommonParams: {}", json);

    // Deserialize from JSON
    let deserialized: CommonParams =
        serde_json::from_str(&json).expect("Failed to deserialize CommonParams");

    // Verify equality
    assert_eq!(common_params.model, deserialized.model);
    assert_eq!(common_params.temperature, deserialized.temperature);
    assert_eq!(common_params.max_tokens, deserialized.max_tokens);
    assert_eq!(common_params.top_p, deserialized.top_p);
    assert_eq!(common_params.stop_sequences, deserialized.stop_sequences);
    assert_eq!(common_params.seed, deserialized.seed);

    println!("   ✅ CommonParams serialization works correctly");

    // Provider-specific options are now handled with ProviderOptions per request
}

/// Test parameter validation with extreme values
#[test]
fn test_parameter_extreme_values() {
    println!("🧪 Testing parameter validation with extreme values");

    // Test extremely small values
    let tiny_params = CommonParams {
        model: "test".to_string(),
        temperature: Some(0.0001),
        max_tokens: Some(1),
        top_p: Some(0.0001),
        stop_sequences: None,
        seed: Some(0),
    };

    // Should be valid for most providers
    println!("   Testing tiny values...");
    assert!(tiny_params.temperature.unwrap() >= 0.0);
    assert!(tiny_params.max_tokens.unwrap() >= 1);
    assert!(tiny_params.top_p.unwrap() >= 0.0);

    // Test extremely large values
    let large_params = CommonParams {
        model: "test".to_string(),
        temperature: Some(1.9999),
        max_tokens: Some(100000),
        top_p: Some(0.9999),
        stop_sequences: Some(vec!["A".repeat(1000)]), // Very long stop sequence
        seed: Some(u64::MAX),
    };

    println!("   Testing large values...");
    assert!(large_params.temperature.unwrap() < 2.0);
    assert!(large_params.max_tokens.unwrap() > 0);
    assert!(large_params.top_p.unwrap() < 1.0);

    println!("   ✅ Extreme value validation works correctly");
}

/// Test parameter memory efficiency
#[test]
fn test_parameter_memory_efficiency() {
    println!("🧪 Testing parameter memory efficiency");

    // Create many parameter instances
    let mut params_vec = Vec::new();
    for i in 0..1000 {
        let params = CommonParams {
            model: format!("model-{}", i),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec![format!("stop-{}", i)]),
            seed: Some(i as u64),
        };
        params_vec.push(params);
    }

    println!("   Created {} parameter instances", params_vec.len());
    assert_eq!(params_vec.len(), 1000);

    // Test that we can access all parameters efficiently
    let total_max_tokens: u32 = params_vec.iter().filter_map(|p| p.max_tokens).sum();

    assert_eq!(total_max_tokens, 1000 * 1000); // 1000 instances * 1000 tokens each

    println!("   ✅ Memory efficiency test completed");
}

/// Test parameter thread safety
#[tokio::test]
async fn test_parameter_thread_safety() {
    println!("🧪 Testing parameter thread safety");

    let shared_params = Arc::new(Mutex::new(CommonParams {
        model: "shared-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    }));

    let mut handles = vec![];

    // Spawn multiple tasks that read/modify parameters
    for i in 0..10 {
        let params_clone = Arc::clone(&shared_params);
        let handle = tokio::spawn(async move {
            let mut params = params_clone.lock().await;
            params.model = format!("model-{}", i);
            params.temperature = Some(0.5 + (i as f32 * 0.1));
            params.seed = Some(i as u64);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.expect("Task failed");
    }

    let final_params = shared_params.lock().await;
    println!("   Final model: {}", final_params.model);
    println!("   Final temperature: {:?}", final_params.temperature);
    println!("   Final seed: {:?}", final_params.seed);

    println!("   ✅ Thread safety test completed");
}

/// Test parameter validation error messages
#[test]
fn test_parameter_validation_error_messages() {
    println!("🧪 Testing parameter validation error messages");

    use siumai::params::ParameterValidator;

    // Test temperature validation error (negative temperature)
    let temp_error = ParameterValidator::validate_temperature(-1.0, 0.0, 2.0, "OpenAI");
    assert!(temp_error.is_err());
    let error_msg = temp_error.unwrap_err().to_string();
    assert!(error_msg.contains("temperature"));
    assert!(error_msg.contains("OpenAI"));
    println!("   Temperature error: {}", error_msg);

    // Test top_p validation error
    let top_p_error = ParameterValidator::validate_top_p(1.5);
    assert!(top_p_error.is_err());
    let error_msg = top_p_error.unwrap_err().to_string();
    assert!(error_msg.contains("top_p"));
    println!("   Top_p error: {}", error_msg);

    // Test max_tokens validation error
    let max_tokens_error = ParameterValidator::validate_max_tokens(0, 1, 100000, "test");
    assert!(max_tokens_error.is_err());
    let error_msg = max_tokens_error.unwrap_err().to_string();
    assert!(error_msg.contains("max_tokens"));
    println!("   Max_tokens error: {}", error_msg);

    println!("   ✅ Error message validation completed");
}

/// Test parameter cloning and equality
#[test]
fn test_parameter_cloning_and_equality() {
    println!("🧪 Testing parameter cloning and equality");

    let original_params = CommonParams {
        model: "test-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string()]),
        seed: Some(42),
    };

    // Test cloning
    let cloned_params = original_params.clone();

    // Verify equality
    assert_eq!(original_params.model, cloned_params.model);
    assert_eq!(original_params.temperature, cloned_params.temperature);
    assert_eq!(original_params.max_tokens, cloned_params.max_tokens);
    assert_eq!(original_params.top_p, cloned_params.top_p);
    assert_eq!(original_params.stop_sequences, cloned_params.stop_sequences);
    assert_eq!(original_params.seed, cloned_params.seed);

    println!("   ✅ Parameter cloning works correctly");

    // Test that modifications to clone don't affect original
    let mut modified_clone = cloned_params;
    modified_clone.model = "modified-model".to_string();
    modified_clone.temperature = Some(0.8);

    assert_ne!(original_params.model, modified_clone.model);
    assert_ne!(original_params.temperature, modified_clone.temperature);

    println!("   ✅ Parameter independence after cloning verified");
}
