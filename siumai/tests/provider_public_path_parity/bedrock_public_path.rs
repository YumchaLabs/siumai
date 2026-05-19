use super::*;
use futures_util::StreamExt;
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::{
    ChatStreamEvent, EmbeddingExtensions, EmbeddingRequest, FinishReason, ResponseFormat, Tool,
    ToolChoice,
};
use siumai::provider_ext::bedrock::{
    AmazonBedrockProviderSettings, BedrockChatOptions, BedrockChatRequestExt,
    BedrockChatResponseExt, BedrockEmbeddingOptions, BedrockEmbeddingRequestExt,
    BedrockRerankOptions, BedrockRerankRequestExt,
};

fn bedrock_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("bedrock", "bedrock")
}

fn bedrock_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("bedrock", "bedrock")
}

fn make_bedrock_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    bedrock_transport: Arc<dyn HttpTransport>,
    runtime_base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    bedrock_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/not-bedrock")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "bedrock",
            "ctx-key",
            runtime_base_url,
            bedrock_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    bedrock_registry_builder()
        .with_provider_api_key_base_url_fetch("bedrock", "test-key", base_url, transport)
        .build()
        .expect("build registry")
}

#[test]
fn bedrock_package_settings_preserve_supported_provider_inputs() {
    let config = AmazonBedrockProviderSettings::new()
        .with_api_key("test-key")
        .with_region("us-west-2")
        .with_header("x-test", "1")
        .into_config_for_model("amazon.nova-lite-v1:0")
        .expect("settings into config");

    assert_eq!(config.region, "us-west-2");
    assert_eq!(config.common_params.model, "amazon.nova-lite-v1:0");
    assert_eq!(
        config.runtime_base_url,
        "https://bedrock-runtime.us-west-2.amazonaws.com"
    );
    assert_eq!(
        config.agent_runtime_base_url,
        "https://bedrock-agent-runtime.us-west-2.amazonaws.com"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_chat_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_bedrock_chat_options(
        BedrockChatOptions::new()
            .with_additional_model_request_fields(serde_json::json!({ "topK": 42 })),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_chat_stream_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_streaming(true)
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 8 })),
        );

    let _ = siumai_client.chat_stream_request(request.clone()).await;
    let _ = provider_client.chat_stream_request(request.clone()).await;
    let _ = config_client.chat_stream_request(request).await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
}

#[tokio::test]
async fn bedrock_registry_chat_request_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .language_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_bedrock_chat_options(
        BedrockChatOptions::new()
            .with_additional_model_request_fields(serde_json::json!({ "topK": 42 })),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
    assert_eq!(
        registry_req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 42 })
    );
}

#[tokio::test]
async fn bedrock_registry_chat_stream_request_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .language_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_streaming(true)
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 8 })),
        );

    let _ = config_client.chat_stream_request(request.clone()).await;
    let _ = registry_model.chat_stream_request(request).await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
    assert_eq!(
        registry_req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 8 })
    );
}

#[tokio::test]
async fn bedrock_registry_chat_handle_prefers_provider_specific_build_overrides() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let global_transport = CaptureTransport::default();
    let bedrock_transport = CaptureTransport::default();

    let registry = make_bedrock_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(bedrock_transport.clone()),
        runtime_base_url,
    );

    let handle = registry
        .language_model(&format!("bedrock:{model}"))
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_bedrock_chat_options(
        BedrockChatOptions::new()
            .with_additional_model_request_fields(serde_json::json!({ "topK": 24 })),
    );

    let _ = handle.chat_request(request).await;

    let req = bedrock_transport.take().expect("captured bedrock request");
    assert!(global_transport.take().is_none());
    assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    assert_eq!(
        req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
    assert_eq!(
        req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 24 })
    );
}

#[tokio::test]
async fn bedrock_registry_chat_stream_handle_prefers_provider_specific_build_overrides() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";

    let global_transport = CaptureTransport::default();
    let bedrock_transport = CaptureTransport::default();

    let registry = make_bedrock_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(bedrock_transport.clone()),
        runtime_base_url,
    );

    let handle = registry
        .language_model(&format!("bedrock:{model}"))
        .expect("build registry language model");

    let _ = handle
        .chat_stream_request(
            make_chat_request_with_model(model).with_bedrock_chat_options(
                BedrockChatOptions::new()
                    .with_additional_model_request_fields(serde_json::json!({ "topK": 24 })),
            ),
        )
        .await;

    let req = bedrock_transport
        .take_stream()
        .expect("captured bedrock stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
    assert_eq!(
        req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 24 })
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "elements": { "type": "array" }
        },
        "required": ["elements"],
        "additionalProperties": false
    });

    let response_json = serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "json-tool-id",
                            "name": "json",
                            "input": {
                                "elements": [
                                    {
                                        "location": "San Francisco",
                                        "temperature": -5,
                                        "condition": "snowy"
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        },
        "usage": {
            "inputTokens": 15,
            "outputTokens": 42,
            "totalTokens": 57
        },
        "stopReason": "tool_use",
        "additionalModelResponseFields": {
            "delta": {
                "stop_sequence": "END_OF_TURN"
            }
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request)
        .await
        .expect("config response ok");

    let siumai_meta = siumai_resp
        .bedrock_metadata()
        .expect("siumai bedrock metadata");
    let provider_meta = provider_resp
        .bedrock_metadata()
        .expect("provider bedrock metadata");
    let config_meta = config_resp
        .bedrock_metadata()
        .expect("config bedrock metadata");

    assert!(
        siumai_resp
            .content_text()
            .unwrap_or_default()
            .contains("San Francisco"),
        "unexpected siumai content: {:?}",
        siumai_resp.content_text()
    );
    assert!(
        provider_resp
            .content_text()
            .unwrap_or_default()
            .contains("San Francisco"),
        "unexpected provider content: {:?}",
        provider_resp.content_text()
    );
    assert!(
        config_resp
            .content_text()
            .unwrap_or_default()
            .contains("San Francisco"),
        "unexpected config content: {:?}",
        config_resp.content_text()
    );
    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(siumai_meta.is_json_response_from_tool, Some(true));
    assert_eq!(provider_meta.is_json_response_from_tool, Some(true));
    assert_eq!(config_meta.is_json_response_from_tool, Some(true));
    assert_eq!(
        siumai_meta.stop_sequence,
        Some(serde_json::json!("END_OF_TURN"))
    );
    assert_eq!(
        provider_meta.stop_sequence,
        Some(serde_json::json!("END_OF_TURN"))
    );
    assert_eq!(
        config_meta.stop_sequence,
        Some(serde_json::json!("END_OF_TURN"))
    );
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
}

#[tokio::test]
async fn bedrock_registry_chat_response_metadata_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "elements": { "type": "array" }
        },
        "required": ["elements"],
        "additionalProperties": false
    });

    let response_json = serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "json-tool-id",
                            "name": "json",
                            "input": {
                                "elements": [
                                    {
                                        "location": "San Francisco",
                                        "temperature": -5,
                                        "condition": "snowy"
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        },
        "usage": {
            "inputTokens": 15,
            "outputTokens": 42,
            "totalTokens": 57
        },
        "stopReason": "tool_use",
        "additionalModelResponseFields": {
            "delta": {
                "stop_sequence": "END_OF_TURN"
            }
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .language_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let config_meta = config_resp
        .bedrock_metadata()
        .expect("config bedrock metadata");
    let registry_meta = registry_resp
        .bedrock_metadata()
        .expect("registry bedrock metadata");

    assert!(
        config_resp
            .content_text()
            .unwrap_or_default()
            .contains("San Francisco"),
        "unexpected config content: {:?}",
        config_resp.content_text()
    );
    assert!(
        registry_resp
            .content_text()
            .unwrap_or_default()
            .contains("San Francisco"),
        "unexpected registry content: {:?}",
        registry_resp.content_text()
    );
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_meta.is_json_response_from_tool, Some(true));
    assert_eq!(registry_meta.is_json_response_from_tool, Some(true));
    assert_eq!(
        config_meta.stop_sequence,
        Some(serde_json::json!("END_OF_TURN"))
    );
    assert_eq!(
        registry_meta.stop_sequence,
        Some(serde_json::json!("END_OF_TURN"))
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let stream_body = concat!(
            "{\"contentBlockStart\":{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"json-tool-id\",\"name\":\"json\"}}}}\n",
            "{\"contentBlockDelta\":{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"value\\\":\\\"test\\\"}\"}}}}\n",
            "{\"contentBlockStop\":{\"contentBlockIndex\":0}}\n",
            "{\"metadata\":{\"usage\":{\"inputTokens\":15,\"outputTokens\":42,\"totalTokens\":57}}}\n",
            "{\"messageStop\":{\"stopReason\":\"tool_use\"}}\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    use futures_util::StreamExt;

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut config_end = None;

    while let Some(event) = siumai_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            siumai_end = Some(response);
            break;
        }
    }
    while let Some(event) = provider_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            provider_end = Some(response);
            break;
        }
    }
    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");

    let siumai_meta = siumai_resp
        .bedrock_metadata()
        .expect("siumai bedrock metadata");
    let provider_meta = provider_resp
        .bedrock_metadata()
        .expect("provider bedrock metadata");
    let config_meta = config_resp
        .bedrock_metadata()
        .expect("config bedrock metadata");

    assert_eq!(siumai_resp.content_text(), Some("{\"value\":\"test\"}"));
    assert_eq!(provider_resp.content_text(), Some("{\"value\":\"test\"}"));
    assert_eq!(config_resp.content_text(), Some("{\"value\":\"test\"}"));
    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(siumai_meta.is_json_response_from_tool, Some(true));
    assert_eq!(provider_meta.is_json_response_from_tool, Some(true));
    assert_eq!(config_meta.is_json_response_from_tool, Some(true));
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
}

#[tokio::test]
async fn bedrock_registry_stream_end_metadata_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let stream_body = concat!(
            "{\"contentBlockStart\":{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"json-tool-id\",\"name\":\"json\"}}}}\n",
            "{\"contentBlockDelta\":{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"value\\\":\\\"test\\\"}\"}}}}\n",
            "{\"contentBlockStop\":{\"contentBlockIndex\":0}}\n",
            "{\"metadata\":{\"usage\":{\"inputTokens\":15,\"outputTokens\":42,\"totalTokens\":57}}}\n",
            "{\"messageStop\":{\"stopReason\":\"tool_use\"}}\n"
        )
        .as_bytes()
        .to_vec();

    let config_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let registry_transport = JsonStreamSuccessTransport::new(stream_body);

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .language_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    use futures_util::StreamExt;

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let mut config_end = None;
    let mut registry_end = None;

    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    let config_meta = config_resp
        .bedrock_metadata()
        .expect("config bedrock metadata");
    let registry_meta = registry_resp
        .bedrock_metadata()
        .expect("registry bedrock metadata");

    assert_eq!(config_resp.content_text(), Some("{\"value\":\"test\"}"));
    assert_eq!(registry_resp.content_text(), Some("{\"value\":\"test\"}"));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_meta.is_json_response_from_tool, Some(true));
    assert_eq!(registry_meta.is_json_response_from_tool, Some(true));
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(57)
    );

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
}

#[tokio::test]
async fn bedrock_structured_output_reserved_json_stream_extracts_across_public_paths() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let stream_body = concat!(
            "{\"contentBlockStart\":{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"json-tool-id\",\"name\":\"json\"}}}}\n",
            "{\"contentBlockDelta\":{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"value\\\":\\\"test\\\"}\"}}}}\n",
            "{\"contentBlockStop\":{\"contentBlockIndex\":0}}\n",
            "{\"metadata\":{\"usage\":{\"inputTokens\":15,\"outputTokens\":42,\"totalTokens\":57}}}\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request)
            .await
            .expect("config stream ok"),
    )
    .await
    .expect("config structured output");

    assert_eq!(siumai_value["value"], "test");
    assert_eq!(provider_value["value"], "test");
    assert_eq!(config_value["value"], "test");

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
}

#[tokio::test]
async fn bedrock_clean_eof_stream_end_keeps_reserved_json_text_on_public_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let stream_body = concat!(
            "{\"contentBlockStart\":{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"json-tool-id\",\"name\":\"json\"}}}}\n",
            "{\"contentBlockDelta\":{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"value\\\":\\\"test\\\"}\"}}}}\n",
            "{\"contentBlockStop\":{\"contentBlockIndex\":0}}\n",
            "{\"metadata\":{\"usage\":{\"inputTokens\":15,\"outputTokens\":42,\"totalTokens\":57}}}\n"
        )
        .as_bytes()
        .to_vec();

    let transport = JsonStreamSuccessTransport::new(stream_body);
    let client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema));

    let mut stream = client
        .chat_stream_request(request)
        .await
        .expect("siumai stream ok");

    let mut stream_end = None;
    let mut saw_text_delta = false;
    let mut saw_json_tool_call = false;
    while let Some(event) = stream.next().await {
        if let Ok(event) = event {
            match event {
                ChatStreamEvent::Part {
                    part: siumai::prelude::unified::ChatStreamPart::TextDelta { delta, .. },
                } if delta == "{\"value\":\"test\"}" => {
                    saw_text_delta = true;
                }
                ChatStreamEvent::Part {
                    part: siumai::prelude::unified::ChatStreamPart::ToolCall(tool_call),
                } if tool_call.tool_name == "json" => {
                    saw_json_tool_call = true;
                }
                ChatStreamEvent::StreamEnd { response } => {
                    stream_end = Some(response);
                    break;
                }
                _ => {}
            }
        }
    }

    let response = stream_end.expect("stream end response");
    assert_eq!(response.finish_reason, Some(FinishReason::Unknown));
    assert_eq!(
        response.text().as_deref(),
        Some("{\"value\":\"test\"}"),
        "unexpected bedrock stream-end content: {:?}, saw_text_delta={}, saw_json_tool_call={}",
        response.content,
        saw_text_delta,
        saw_json_tool_call
    );
    assert!(
        saw_text_delta,
        "expected clean EOF stream to emit text delta"
    );
    assert!(
        !saw_json_tool_call,
        "reserved JSON output should not surface as a public tool call"
    );
    assert_eq!(
        response
            .bedrock_metadata()
            .and_then(|metadata| metadata.is_json_response_from_tool),
        Some(true)
    );
}

#[tokio::test]
async fn bedrock_structured_output_reserved_json_stream_fails_consistently_across_public_paths() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" }
        },
        "required": ["value"],
        "additionalProperties": false
    });

    let stream_body = concat!(
            "{\"contentBlockStart\":{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"json-tool-id\",\"name\":\"json\"}}}}\n",
            "{\"contentBlockDelta\":{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"value\\\":\"}}}}\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let provider_transport = JsonStreamSuccessTransport::new(stream_body.clone());
    let config_transport = JsonStreamSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_response_format(ResponseFormat::json_schema(schema.clone()));

    let siumai_err = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect_err("siumai interrupted stream should fail");
    let provider_err = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect_err("provider interrupted stream should fail");
    let config_err = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request)
            .await
            .expect("config stream ok"),
    )
    .await
    .expect_err("config interrupted stream should fail");

    for err in [siumai_err, provider_err, config_err] {
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_rerank_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.rerank-v1:0";

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_rerank_request_with_model(model)
        .with_top_n(1)
        .with_bedrock_rerank_options(
            BedrockRerankOptions::new()
                .with_region("us-east-1")
                .with_next_token("token-1")
                .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
        );

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
    );
}

#[tokio::test]
async fn bedrock_registry_rerank_request_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.rerank-v1:0";

    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .reranking_model("bedrock:amazon.rerank-v1:0")
        .expect("build registry rerank model");

    let request = make_rerank_request_with_model(model)
        .with_top_n(1)
        .with_bedrock_rerank_options(
            BedrockRerankOptions::new()
                .with_region("us-east-1")
                .with_next_token("token-1")
                .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
        );

    let _ = config_client.rerank(request.clone()).await;
    let _ = registry_model.rerank(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
    );
    assert_eq!(registry_req.body["nextToken"], serde_json::json!("token-1"));
    assert_eq!(
        registry_req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
            ["modelArn"],
        serde_json::json!("arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0")
    );
    assert_eq!(
        registry_req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
            ["additionalModelRequestFields"]["topK"],
        serde_json::json!(4)
    );
}

#[tokio::test]
async fn bedrock_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.rerank-v1:0";

    let global_transport = CaptureTransport::default();
    let bedrock_transport = CaptureTransport::default();

    let registry = make_bedrock_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(bedrock_transport.clone()),
        runtime_base_url,
    );

    let handle = registry
        .reranking_model(&format!("bedrock:{model}"))
        .expect("build registry rerank model");

    let _ = handle
        .rerank(
            make_rerank_request_with_model(model)
                .with_top_n(1)
                .with_bedrock_rerank_options(
                    BedrockRerankOptions::new()
                        .with_region("us-east-1")
                        .with_next_token("token-1")
                        .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
                ),
        )
        .await;

    let req = bedrock_transport
        .take()
        .expect("captured bedrock rerank request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
    );
    assert_eq!(req.body["nextToken"], serde_json::json!("token-1"));
    assert_eq!(
        req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]["modelArn"],
        serde_json::json!("arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0")
    );
    assert_eq!(
        req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]["additionalModelRequestFields"]
            ["topK"],
        serde_json::json!(4)
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_stable_request_options_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model)
        .with_tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .with_tool_choice(ToolChoice::Required)
        .with_response_format(ResponseFormat::json_schema(schema.clone()))
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 16 })),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
    assert_eq!(
        siumai_req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 16 })
    );
    assert_eq!(
        siumai_req.body["toolConfig"]["toolChoice"],
        serde_json::json!({ "any": {} })
    );
    let tools = siumai_req.body["toolConfig"]["tools"]
        .as_array()
        .expect("tools array");
    assert_eq!(tools.len(), 2);
    assert_eq!(
        tools[0]["toolSpec"]["name"],
        serde_json::json!("lookup_weather")
    );
    assert_eq!(tools[1]["toolSpec"]["name"], serde_json::json!("json"));
    assert_eq!(tools[1]["toolSpec"]["inputSchema"]["json"], schema);
}

#[tokio::test]
async fn bedrock_registry_stable_request_options_match_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "anthropic.claude-3-haiku-20240307-v1:0";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .language_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model)
        .with_tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .with_tool_choice(ToolChoice::Required)
        .with_response_format(ResponseFormat::json_schema(schema.clone()))
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 16 })),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    );
    assert_eq!(
        registry_req.body["additionalModelRequestFields"],
        serde_json::json!({ "topK": 16 })
    );
    assert_eq!(
        registry_req.body["toolConfig"]["toolChoice"],
        serde_json::json!({ "any": {} })
    );
    let tools = registry_req.body["toolConfig"]["tools"]
        .as_array()
        .expect("tools array");
    assert_eq!(tools.len(), 2);
    assert_eq!(
        tools[0]["toolSpec"]["name"],
        serde_json::json!("lookup_weather")
    );
    assert_eq!(tools[1]["toolSpec"]["name"], serde_json::json!("json"));
    assert_eq!(tools[1]["toolSpec"]["inputSchema"]["json"], schema);
}

#[cfg(feature = "bedrock")]
#[cfg(feature = "bedrock")]
#[tokio::test]
async fn bedrock_siumai_provider_config_embedding_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.titan-embed-text-v2:0";

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::single("bedrock embedding boundary")
        .with_model(model)
        .with_bedrock_embedding_options(
            BedrockEmbeddingOptions::new()
                .with_dimensions(512)
                .with_normalize(true),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert!(provider_client.as_embedding_capability().is_some());
    assert!(config_client.as_embedding_capability().is_some());
    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke"
    );
    assert_eq!(
        siumai_req.body,
        serde_json::json!({
            "inputText": "bedrock embedding boundary",
            "dimensions": 512,
            "normalize": true
        })
    );
}

#[tokio::test]
async fn bedrock_registry_embedding_request_matches_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.titan-embed-text-v2:0";
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .embedding_model("bedrock:amazon.titan-embed-text-v2:0")
        .expect("build registry embedding handle");

    let request = EmbeddingRequest::single("bedrock embedding boundary")
        .with_model(model)
        .with_bedrock_embedding_options(
            BedrockEmbeddingOptions::new()
                .with_dimensions(512)
                .with_normalize(true),
        );

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke"
    );
    assert_eq!(
        registry_req.body,
        serde_json::json!({
            "inputText": "bedrock embedding boundary",
            "dimensions": 512,
            "normalize": true
        })
    );
}

#[tokio::test]
async fn bedrock_siumai_provider_config_image_request_are_equivalent() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.nova-canvas-v1:0";
    let response_json = serde_json::json!({
        "images": ["aGVsbG8="]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::bedrock()
        .api_key("test-key")
        .base_url(runtime_base_url)
        .image_model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ImageGenerationRequest {
        prompt: "a tiny silver robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 2,
        model: Some(model.to_string()),
        quality: Some("premium".to_string()),
        style: Some("photographic".to_string()),
        seed: Some(7),
        steps: None,
        guidance_scale: Some(6.5),
        enhance_prompt: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image generation ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image generation ok");
    let config_resp = config_client
        .generate_images(request)
        .await
        .expect("config image generation ok");

    assert!(provider_client.as_image_generation_capability().is_some());
    assert!(config_client.as_image_generation_capability().is_some());
    assert_eq!(siumai_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );
    assert_eq!(config_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-canvas-v1%3A0/invoke"
    );
    assert_eq!(
        siumai_req.body,
        serde_json::json!({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": "a tiny silver robot",
                "negativeText": "blurry",
                "style": "photographic"
            },
            "imageGenerationConfig": {
                "width": 1024,
                "height": 1024,
                "seed": 7,
                "numberOfImages": 2,
                "quality": "premium",
                "cfgScale": 6.5
            }
        })
    );
}

#[tokio::test]
async fn bedrock_registry_image_request_matches_config_path() {
    let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let model = "amazon.nova-canvas-v1:0";
    let response_json = serde_json::json!({
        "images": ["aGVsbG8="]
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = siumai::provider_ext::bedrock::BedrockClient::from_config(
        siumai::provider_ext::bedrock::BedrockConfig::new()
            .with_api_key("test-key")
            .with_base_url(runtime_base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), runtime_base_url);
    let registry_model = registry
        .image_model("bedrock:amazon.nova-canvas-v1:0")
        .expect("build registry image model");

    let request = ImageGenerationRequest {
        prompt: "a tiny silver robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 2,
        model: Some(model.to_string()),
        quality: Some("premium".to_string()),
        style: Some("photographic".to_string()),
        seed: Some(7),
        steps: None,
        guidance_scale: Some(6.5),
        enhance_prompt: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let config_resp = config_client
        .generate_images(request.clone())
        .await
        .expect("config image generation ok");
    let registry_resp = registry_model
        .generate_images(request)
        .await
        .expect("registry image generation ok");

    assert_eq!(config_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        registry_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        registry_req.url,
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-canvas-v1%3A0/invoke"
    );
    assert_eq!(
        registry_req.body["taskType"],
        serde_json::json!("TEXT_IMAGE")
    );
}
