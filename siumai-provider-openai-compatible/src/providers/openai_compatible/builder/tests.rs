
use super::*;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Default)]
struct NoopMiddleware;

impl LanguageModelMiddleware for NoopMiddleware {}

#[test]
fn builder_shell_keeps_provider_reasoning_mapping_split() {
    let source = include_str!("../builder.rs")
        .split("#[cfg(test)]")
        .next()
        .unwrap_or_default();
    let reasoning = include_str!("reasoning.rs");

    assert!(
        source.contains("mod reasoning;"),
        "OpenAI-compatible builder shell should keep provider reasoning mapping in a dedicated module"
    );
    assert!(
        include_str!("../builder.rs").contains("mod tests;"),
        "OpenAI-compatible builder shell should keep tests in a dedicated module"
    );

    for marker in [
        "fn provider_thinking_value",
        "pub fn with_thinking(",
        "pub fn with_thinking_budget(",
        "pub fn reasoning(",
        "pub fn reasoning_budget(",
    ] {
        assert!(
            !source.contains(marker),
            "OpenAI-compatible builder shell should not own `{marker}`"
        );
        assert!(
            reasoning.contains(marker),
            "builder reasoning module should own `{marker}`"
        );
    }
}

#[test]
fn openai_compatible_builder_into_config_converges() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .temperature(0.4)
        .max_tokens(256)
        .top_p(0.9)
        .stop(vec!["END"])
        .seed(7)
        .reasoning(true)
        .reasoning_budget(2048)
        .timeout(Duration::from_secs(15))
        .connect_timeout(Duration::from_secs(5))
        .http_stream_disable_compression(true)
        .user_agent("siumai-test/1.0")
        .proxy("http://127.0.0.1:8080")
        .custom_headers(std::collections::HashMap::from([(
            "x-one".to_string(),
            "1".to_string(),
        )]))
        .header("x-two", "2")
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .http_debug(true)
        .into_config()
        .expect("into_config ok");

    assert_eq!(config.provider_id, "deepseek");
    assert_eq!(config.model, "deepseek-chat");
    assert_eq!(config.common_params.temperature, Some(0.4));
    assert_eq!(config.common_params.max_tokens, Some(256));
    assert_eq!(config.common_params.top_p, Some(0.9));
    assert_eq!(
        config.common_params.stop_sequences,
        Some(vec!["END".to_string()])
    );
    assert_eq!(config.common_params.seed, Some(7));
    let mut params = serde_json::json!({});
    config
        .adapter
        .transform_request_params(
            &mut params,
            &config.model,
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("transform request params");
    assert_eq!(
        params["thinking"],
        serde_json::json!({
            "type": "enabled"
        })
    );
    assert!(params.get("enable_reasoning").is_none());
    assert!(params.get("reasoning_budget").is_none());
    assert_eq!(config.http_config.timeout, Some(Duration::from_secs(15)));
    assert_eq!(
        config.http_config.connect_timeout,
        Some(Duration::from_secs(5))
    );
    assert!(config.http_config.stream_disable_compression);
    assert_eq!(
        config.http_config.user_agent.as_deref(),
        Some("siumai-test/1.0")
    );
    assert_eq!(
        config.http_config.proxy.as_deref(),
        Some("http://127.0.0.1:8080")
    );
    assert_eq!(
        config.http_config.headers.get("x-one"),
        Some(&"1".to_string())
    );
    assert_eq!(
        config.http_config.headers.get("x-two"),
        Some(&"2".to_string())
    );
    assert_eq!(config.http_interceptors.len(), 2);
}

#[test]
fn openai_compatible_builder_maps_qwen_reasoning_to_alibaba_thinking_fields() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "qwen")
        .api_key("test-key")
        .model("qwen-plus")
        .reasoning(true)
        .reasoning_budget(2048)
        .into_config()
        .expect("into_config ok");

    assert_eq!(config.provider_id, "qwen");

    let mut params = serde_json::json!({});
    config
        .adapter
        .transform_request_params(
            &mut params,
            &config.model,
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("transform request params");

    assert_eq!(params["enable_thinking"], serde_json::json!(true));
    assert_eq!(params["thinking_budget"], serde_json::json!(2048));
    assert!(params.get("enable_reasoning").is_none());
    assert!(params.get("reasoning_budget").is_none());
}

#[test]
fn openai_compatible_builder_maps_xai_reasoning_to_effort() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "xai")
        .api_key("test-key")
        .model("grok-4")
        .reasoning(true)
        .reasoning_budget(2048)
        .into_config()
        .expect("into_config ok");

    assert_eq!(config.provider_id, "xai");

    let mut params = serde_json::json!({});
    config
        .adapter
        .transform_request_params(
            &mut params,
            &config.model,
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("transform request params");

    assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
    assert!(params.get("enable_reasoning").is_none());
    assert!(params.get("reasoning_budget").is_none());
    assert!(params.get("enable_thinking").is_none());
    assert!(params.get("thinking_budget").is_none());
}

#[test]
fn openai_compatible_builder_allows_external_authorization_without_api_key() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .model("deepseek-chat")
        .header("Authorization", "Bearer external-token")
        .into_config()
        .expect("authorization header should satisfy compat auth");

    assert_eq!(config.provider_id, "deepseek");
    assert!(config.api_key.is_empty());
    assert_eq!(
        config
            .http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer external-token")
    );
}

#[test]
fn openai_compatible_builder_carries_token_provider_without_api_key() {
    let token_provider = Arc::new(crate::auth::StaticTokenProvider::new("test-token"));
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .model("deepseek-chat")
        .with_token_provider(token_provider)
        .into_config()
        .expect("token provider should satisfy compat auth");

    assert_eq!(config.provider_id, "deepseek");
    assert!(config.api_key.is_empty());
    assert!(config.token_provider.is_some());
}

#[test]
fn openai_compatible_builder_generic_provider_defaults_to_optional_auth() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "local-gateway")
        .base_url("http://localhost:11434/v1")
        .model("llama3.2")
        .into_config()
        .expect("generic compat config");

    assert_eq!(config.provider_id, "local-gateway");
    assert!(config.api_key.is_empty());
    assert!(!config.auth_required);
    assert!(config.validate().is_ok());
}

#[test]
fn openai_compatible_builder_generic_provider_honors_explicit_auth_requirement() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "private-gateway")
        .base_url("https://gateway.example.com/v1")
        .model("gateway-model")
        .with_auth_required(true)
        .header("Authorization", "Bearer external-token")
        .into_config()
        .expect("generic compat config with explicit auth");

    assert_eq!(config.provider_id, "private-gateway");
    assert!(config.api_key.is_empty());
    assert!(config.auth_required);
    assert_eq!(
        config
            .http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer external-token")
    );
    assert!(config.validate().is_ok());
}

#[test]
fn openai_compatible_builder_installs_metadata_extractor() {
    let extractor: Arc<dyn ResponseMetadataExtractor> = Arc::new(|raw: &serde_json::Value| {
        raw.get("test_field").map(|value| {
            std::collections::HashMap::from([(
                "test-provider".to_string(),
                serde_json::json!({ "value": value }),
            )])
        })
    });

    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_metadata_extractor(extractor)
        .into_config()
        .expect("into_config ok");

    let metadata = config
        .adapter
        .extract_response_provider_metadata(&serde_json::json!({
            "test_field": "test-value"
        }))
        .expect("metadata");
    let provider = metadata.get("test-provider").expect("provider metadata");
    assert_eq!(
        provider.get("value"),
        Some(&serde_json::json!("test-value"))
    );
}

#[test]
fn openai_compatible_builder_records_include_usage_setting() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_include_usage(true)
        .into_config()
        .expect("into_config ok");

    assert_eq!(config.include_usage, Some(true));
}

#[test]
fn openai_compatible_builder_defaults_ai_sdk_usage_streaming_providers() {
    for provider_id in ["alibaba", "deepseek", "moonshotai", "qwen", "xai"] {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), provider_id)
            .api_key("test-key")
            .model("test-model")
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.include_usage, Some(true), "{provider_id}");
    }
}

#[test]
fn openai_compatible_builder_respects_explicit_defaulted_include_usage() {
    for provider_id in ["alibaba", "deepseek", "moonshotai", "qwen", "xai"] {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), provider_id)
            .api_key("test-key")
            .model("test-model")
            .with_include_usage(false)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.include_usage, Some(false), "{provider_id}");
    }
}

#[test]
fn openai_compatible_builder_records_query_params_setting() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_query_params([("api-version", "2025-04-01"), ("tenant", "acme")])
        .into_config()
        .expect("into_config ok");

    assert_eq!(
        config.query_params.get("api-version").map(String::as_str),
        Some("2025-04-01")
    );
    assert_eq!(
        config.query_params.get("tenant").map(String::as_str),
        Some("acme")
    );
}

#[test]
fn openai_compatible_builder_records_structured_outputs_policy() {
    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_supports_structured_outputs(false)
        .into_config()
        .expect("into_config ok");

    assert_eq!(config.supports_structured_outputs, Some(false));
}

#[test]
fn openai_compatible_builder_installs_request_body_transformer() {
    let transformer: Arc<dyn RequestBodyTransformer> = Arc::new(
        |body: &mut serde_json::Value,
         _model: &str,
         request_type: crate::providers::openai_compatible::RequestType| {
            assert!(matches!(
                request_type,
                crate::providers::openai_compatible::RequestType::Chat
            ));
            body["custom"] = serde_json::json!(true);
            Ok(())
        },
    );

    let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_request_body_transformer(transformer)
        .into_config()
        .expect("into_config ok");

    let hook = config
        .request_body_transformer
        .as_ref()
        .expect("request body transformer");
    let mut body = serde_json::json!({});
    hook.transform_request_body(
        &mut body,
        "deepseek-chat",
        crate::providers::openai_compatible::RequestType::Chat,
    )
    .expect("transform body");
    assert_eq!(body.get("custom"), Some(&serde_json::json!(true)));
}

#[test]
fn openai_compatible_builder_into_config_matches_manual_compatible_config() {
    let builder_config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .temperature(0.4)
        .max_tokens(256)
        .top_p(0.9)
        .stop(vec!["END"])
        .seed(7)
        .reasoning(true)
        .reasoning_budget(2048)
        .timeout(Duration::from_secs(15))
        .connect_timeout(Duration::from_secs(5))
        .http_stream_disable_compression(true)
        .user_agent("siumai-test/1.0")
        .proxy("http://127.0.0.1:8080")
        .custom_headers(std::collections::HashMap::from([(
            "x-one".to_string(),
            "1".to_string(),
        )]))
        .header("x-two", "2")
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .http_debug(true)
        .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
        .into_config()
        .expect("builder config");

    let provider = crate::providers::openai_compatible::get_provider_config("deepseek")
        .expect("provider config");
    let adapter = Arc::new(crate::providers::openai_compatible::ConfigurableAdapter::new(provider));
    let manual_config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
        "deepseek",
        "test-key",
        "https://api.deepseek.com",
        adapter,
    )
    .with_model("deepseek-chat")
    .with_temperature(0.4)
    .with_max_tokens(256)
    .with_top_p(0.9)
    .with_stop_sequences(vec!["END".to_string()])
    .with_seed(7)
    .with_reasoning(true)
    .with_reasoning_budget(2048)
    .with_timeout(Duration::from_secs(15))
    .with_connect_timeout(Duration::from_secs(5))
    .with_http_stream_disable_compression(true)
    .with_user_agent("siumai-test/1.0")
    .with_proxy("http://127.0.0.1:8080")
    .with_http_headers(std::collections::HashMap::from([(
        "x-one".to_string(),
        "1".to_string(),
    )]))
    .with_http_header("x-two", "2")
    .with_http_interceptor(Arc::new(
        crate::execution::http::interceptor::LoggingInterceptor,
    ))
    .with_http_interceptor(Arc::new(
        crate::execution::http::interceptor::LoggingInterceptor,
    ))
    .with_model_middlewares({
        let mut middlewares =
            crate::execution::middleware::build_auto_middlewares_vec("deepseek", "deepseek-chat");
        middlewares.push(Arc::new(NoopMiddleware));
        middlewares
    });

    assert_eq!(builder_config.provider_id, manual_config.provider_id);
    assert_eq!(builder_config.base_url, manual_config.base_url);
    assert_eq!(builder_config.model, manual_config.model);
    assert_eq!(
        builder_config.common_params.temperature,
        manual_config.common_params.temperature
    );
    assert_eq!(
        builder_config.common_params.max_tokens,
        manual_config.common_params.max_tokens
    );
    assert_eq!(
        builder_config.common_params.top_p,
        manual_config.common_params.top_p
    );
    assert_eq!(
        builder_config.common_params.stop_sequences,
        manual_config.common_params.stop_sequences
    );
    assert_eq!(
        builder_config.common_params.seed,
        manual_config.common_params.seed
    );
    let mut builder_params = serde_json::json!({});
    builder_config
        .adapter
        .transform_request_params(
            &mut builder_params,
            &builder_config.model,
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("builder transform request params");
    let mut manual_params = serde_json::json!({});
    manual_config
        .adapter
        .transform_request_params(
            &mut manual_params,
            &manual_config.model,
            crate::providers::openai_compatible::RequestType::Chat,
        )
        .expect("manual transform request params");
    assert_eq!(builder_params, manual_params);
    assert_eq!(
        builder_config.http_config.timeout,
        manual_config.http_config.timeout
    );
    assert_eq!(
        builder_config.http_config.connect_timeout,
        manual_config.http_config.connect_timeout
    );
    assert_eq!(
        builder_config.http_config.stream_disable_compression,
        manual_config.http_config.stream_disable_compression
    );
    assert_eq!(
        builder_config.http_config.user_agent,
        manual_config.http_config.user_agent
    );
    assert_eq!(
        builder_config.http_config.proxy,
        manual_config.http_config.proxy
    );
    assert_eq!(
        builder_config.http_config.headers,
        manual_config.http_config.headers
    );
    assert_eq!(
        builder_config.http_interceptors.len(),
        manual_config.http_interceptors.len()
    );
    assert_eq!(
        builder_config.model_middlewares.len(),
        manual_config.model_middlewares.len()
    );
}

#[tokio::test]
async fn openai_compatible_builder_build_preserves_http_client_override_and_retry_options() {
    let client = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
        .api_key("test-key")
        .model("deepseek-chat")
        .with_http_config(crate::types::HttpConfig {
            proxy: Some("not-a-url".to_string()),
            ..Default::default()
        })
        .with_http_client(reqwest::Client::new())
        .with_retry(RetryOptions::default())
        .build()
        .await
        .expect("build client with explicit http client");

    assert!(client.retry_options().is_some());
}

#[test]
fn openai_compatible_builder_falls_back_to_provider_config_default_model() {
    let mistral = OpenAiCompatibleBuilder::new(BuilderBase::default(), "mistral")
        .api_key("test-key")
        .into_config()
        .expect("mistral config");
    let cohere = OpenAiCompatibleBuilder::new(BuilderBase::default(), "cohere")
        .api_key("test-key")
        .into_config()
        .expect("cohere config");

    assert_eq!(mistral.model, "mistral-large-latest");
    assert_eq!(mistral.common_params.model, mistral.model);
    assert_eq!(cohere.model, "command-r-plus");
    assert_eq!(cohere.common_params.model, cohere.model);
}
