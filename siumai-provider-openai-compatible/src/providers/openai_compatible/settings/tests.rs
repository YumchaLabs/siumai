use super::*;
use crate::execution::http::transport::{
    HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use siumai_core::traits::ModelMetadata;

#[derive(Clone, Default)]
struct NoopTransport;

#[test]
fn simple_provider_settings_stay_macro_generated() {
    let source = include_str!("../settings.rs")
        .split("#[cfg(test)]")
        .next()
        .unwrap_or_default();
    assert!(
        include_str!("../settings.rs").contains("mod tests;"),
        "OpenAI-compatible provider settings shell should keep tests in a dedicated module"
    );

    for (settings, config) in [
        ("DeepSeekProviderSettings", "DeepSeekConfig"),
        ("GroqProviderSettings", "GroqConfig"),
        ("XaiProviderSettings", "XaiConfig"),
        ("TogetherAIProviderSettings", "TogetherAIConfig"),
        ("DeepInfraProviderSettings", "DeepInfraConfig"),
        ("MoonshotAIProviderSettings", "MoonshotAIConfig"),
        ("FireworksProviderSettings", "FireworksConfig"),
        ("MistralProviderSettings", "MistralConfig"),
        ("PerplexityProviderSettings", "PerplexityConfig"),
    ] {
        let marker = format!("pub struct {settings} => {config}");
        assert!(
            source.contains(&marker),
            "{settings} should use simple_compat_provider_settings!"
        );

        let manual_impl = format!("impl {settings} {{");
        assert!(
            !source.contains(&manual_impl),
            "{settings} should not duplicate the simple settings adapter impl"
        );
    }

    for custom_impl in [
        "impl OpenAICompatibleProviderSettings {",
        "impl GoogleVertexMaasProviderSettings {",
        "impl AlibabaProviderSettings {",
    ] {
        assert!(
            source.contains(custom_impl),
            "custom settings adapter should remain hand-written: {custom_impl}"
        );
    }
}

#[async_trait]
impl HttpTransport for NoopTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Ok(HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: b"{}".to_vec(),
        })
    }

    async fn execute_get(
        &self,
        _request: HttpTransportGetRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Ok(HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: b"{}".to_vec(),
        })
    }
}

#[test]
fn openai_compatible_provider_settings_into_config_preserve_supported_inputs() {
    let config = OpenAICompatibleProviderSettings::new("acme", "https://example.com/v1")
        .with_api_key("test-key")
        .with_header("x-test", "1")
        .with_query_param("api-version", "2025-04-01")
        .with_include_usage(true)
        .with_supports_structured_outputs(false)
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("acme-chat")
        .expect("settings into config");

    assert_eq!(config.provider_id, "acme");
    assert_eq!(config.base_url, "https://example.com/v1");
    assert_eq!(config.common_params.model, "acme-chat");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        config.query_params.get("api-version").map(String::as_str),
        Some("2025-04-01")
    );
    assert_eq!(config.include_usage, Some(true));
    assert_eq!(config.supports_structured_outputs, Some(false));
    assert!(!config.auth_required);
    assert!(config.http_transport.is_some());
}

#[test]
fn openai_compatible_provider_settings_allow_unauthenticated_gateway() {
    let config = OpenAICompatibleProviderSettings::new("local", "http://localhost:11434/v1")
        .into_config_for_model("llama3.2")
        .expect("settings into config");

    assert_eq!(config.provider_id, "local");
    assert!(config.api_key.is_empty());
    assert!(!config.auth_required);
    assert!(config.validate().is_ok());
}

#[test]
fn openai_compatible_provider_settings_do_not_reuse_builtin_preset_on_name_collision() {
    let config = OpenAICompatibleProviderSettings::new("groq", "https://example.com/v1")
        .into_config_for_model("custom-groq-model")
        .expect("settings into config");

    assert_eq!(config.provider_id, "openai-compatible:groq");
    assert_eq!(config.base_url, "https://example.com/v1");
    assert_eq!(config.adapter.base_url(), "https://example.com/v1");
    assert!(config.validate().is_ok());
}

#[test]
fn deepseek_provider_settings_into_config_preserve_supported_inputs() {
    let config = DeepSeekProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepseek")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("deepseek-chat")
        .expect("settings into config");

    assert_eq!(config.provider_id, "deepseek");
    assert_eq!(config.base_url, "https://example.com/deepseek");
    assert_eq!(config.common_params.model, "deepseek-chat");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn groq_provider_settings_into_config_preserve_supported_inputs() {
    let config = GroqProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/groq/openai/v1")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("openai/gpt-oss-20b")
        .expect("settings into config");

    assert_eq!(config.provider_id, "groq");
    assert_eq!(config.base_url, "https://example.com/groq/openai/v1");
    assert_eq!(config.common_params.model, "openai/gpt-oss-20b");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn xai_provider_settings_into_config_preserve_supported_inputs() {
    let config = XaiProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/xai/v1")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("grok-4")
        .expect("settings into config");

    assert_eq!(config.provider_id, "xai");
    assert_eq!(config.base_url, "https://example.com/xai/v1");
    assert_eq!(config.common_params.model, "grok-4");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn togetherai_provider_settings_into_config_preserve_supported_inputs() {
    let config = TogetherAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/together/v1")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        .expect("settings into config");

    assert_eq!(config.provider_id, "togetherai");
    assert_eq!(config.base_url, "https://example.com/together/v1");
    assert_eq!(
        config.common_params.model,
        "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn mistral_provider_settings_into_config_preserve_supported_inputs() {
    let config = MistralProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/mistral")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("mistral-large-latest")
        .expect("settings into config");

    assert_eq!(config.provider_id, "mistral");
    assert_eq!(config.base_url, "https://example.com/mistral");
    assert_eq!(config.common_params.model, "mistral-large-latest");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn perplexity_provider_settings_into_config_preserve_supported_inputs() {
    let config = PerplexityProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/perplexity")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("sonar")
        .expect("settings into config");

    assert_eq!(config.provider_id, "perplexity");
    assert_eq!(config.base_url, "https://example.com/perplexity");
    assert_eq!(config.common_params.model, "sonar");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn fireworks_provider_settings_into_config_preserve_supported_inputs() {
    let config = FireworksProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/fireworks")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .expect("settings into config");

    assert_eq!(config.provider_id, "fireworks");
    assert_eq!(config.base_url, "https://example.com/fireworks");
    assert_eq!(
        config.common_params.model,
        "accounts/fireworks/models/llama-v3p1-8b-instruct"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn moonshotai_provider_settings_into_config_preserve_supported_inputs() {
    let config = MoonshotAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/moonshot")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("kimi-k2.5")
        .expect("settings into config");

    assert_eq!(config.provider_id, "moonshotai");
    assert_eq!(config.base_url, "https://example.com/moonshot");
    assert_eq!(config.common_params.model, "kimi-k2.5");
    assert_eq!(config.include_usage, Some(true));
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn alibaba_provider_settings_into_config_preserve_supported_inputs() {
    let config = AlibabaProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/alibaba/compatible-mode/v1")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("qwen-plus")
        .expect("settings into config");

    assert_eq!(config.provider_id, "alibaba");
    assert_eq!(
        config.base_url,
        "https://example.com/alibaba/compatible-mode/v1"
    );
    assert_eq!(config.common_params.model, "qwen-plus");
    assert_eq!(config.include_usage, Some(true));
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());

    let no_usage = AlibabaProviderSettings::new()
        .with_api_key("test-key")
        .with_include_usage(false)
        .into_config_for_model("qwen-plus")
        .expect("settings into config");
    assert_eq!(no_usage.include_usage, Some(false));

    let video_model = AlibabaProviderSettings::new()
        .with_api_key("test-key")
        .with_video_base_url("https://example.com/dashscope")
        .with_header("x-video", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_video_model("wan2.6-t2v");
    assert_eq!(video_model.provider_id(), "alibaba.video");
    assert_eq!(video_model.model_id(), "wan2.6-t2v");
    assert_eq!(video_model.base_url(), "https://example.com/dashscope");
}

#[test]
fn deepinfra_provider_settings_into_config_preserve_supported_inputs() {
    let config = DeepInfraProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepinfra")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("meta-llama/Llama-3.3-70B-Instruct")
        .expect("settings into config");

    assert_eq!(config.provider_id, "deepinfra");
    assert_eq!(config.base_url, "https://example.com/deepinfra/openai");
    assert_eq!(
        config.common_params.model,
        "meta-llama/Llama-3.3-70B-Instruct"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn google_vertex_maas_provider_settings_into_config_preserve_supported_inputs() {
    let config = GoogleVertexMaasProviderSettings::new()
        .with_project("test-project")
        .with_location("us-central1")
        .with_header("Authorization", "Bearer test-token")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(NoopTransport))
        .into_config_for_model("deepseek-ai/deepseek-v3.2-maas")
        .expect("settings into config");

    assert_eq!(config.provider_id, "vertex-maas");
    assert_eq!(
        config.base_url,
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi"
    );
    assert_eq!(config.common_params.model, "deepseek-ai/deepseek-v3.2-maas");
    assert_eq!(
        config
            .http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer test-token")
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
}

#[test]
fn google_vertex_maas_provider_settings_support_token_provider_auth() {
    let config = GoogleVertexMaasProviderSettings::new()
        .with_project("test-project")
        .with_token_provider(Arc::new(crate::auth::StaticTokenProvider::new(
            "test-token",
        )))
        .into_config_for_model("deepseek-ai/deepseek-v3.2-maas")
        .expect("settings into config");

    assert_eq!(config.provider_id, "vertex-maas");
    assert_eq!(
        config.base_url,
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi"
    );
    assert!(config.api_key.is_empty());
    assert!(config.token_provider.is_some());
}

#[test]
fn google_vertex_xai_request_transformer_strips_reasoning_knobs() {
    let transformer = google_vertex_xai_request_body_transformer();
    let mut body = serde_json::json!({
        "model": "xai/grok-4.1-fast-reasoning",
        "messages": [{ "role": "user", "content": "hi" }],
        "reasoning_effort": "high",
        "reasoningEffort": "high",
        "enable_reasoning": true,
        "enableReasoning": true,
        "reasoning_budget": 8192,
        "reasoningBudget": 8192,
        "enable_thinking": true,
        "enableThinking": true,
        "thinking_budget": 8192,
        "thinkingBudget": 8192
    });

    transformer
        .transform_request_body(
            &mut body,
            "xai/grok-4.1-fast-reasoning",
            super::super::RequestType::Chat,
        )
        .expect("transform google vertex xai body");

    let obj = body.as_object().expect("object body");
    for stripped in [
        "reasoning_effort",
        "reasoningEffort",
        "enable_reasoning",
        "enableReasoning",
        "reasoning_budget",
        "reasoningBudget",
        "enable_thinking",
        "enableThinking",
        "thinking_budget",
        "thinkingBudget",
    ] {
        assert!(
            !obj.contains_key(stripped),
            "google-vertex-xai should strip {stripped}; reasoning mode is model-id owned"
        );
    }
    assert_eq!(
        body["model"],
        serde_json::json!("xai/grok-4.1-fast-reasoning")
    );
    assert_eq!(
        body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
}
