//! Non-streaming JSON gateway helpers (Axum).
//!
//! English-only comments in code as requested.

use std::fmt;
use std::sync::Arc;

use axum::{body::Body, http::header, response::Response};

#[cfg(feature = "anthropic")]
use siumai::experimental::bridge::bridge_chat_response_to_anthropic_messages_json_bytes_with_options;
#[cfg(feature = "google")]
use siumai::experimental::bridge::bridge_chat_response_to_gemini_generate_content_json_bytes_with_options;
use siumai::experimental::bridge::{
    BridgeCustomization, BridgeMode, BridgeOptions, BridgeOptionsOverride, BridgeReport,
    BridgeTarget,
};
#[cfg(feature = "openai")]
use siumai::experimental::bridge::{
    bridge_chat_response_to_openai_chat_completions_json_bytes_with_options,
    bridge_chat_response_to_openai_responses_json_bytes_with_options,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::prelude::unified::{ChatResponse, LlmError};

use crate::server::{GatewayBridgePolicy, gateway_bridge_headers, resolve_gateway_bridge_options};

/// Target JSON wire format for non-streaming response transcoding helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetJsonFormat {
    /// OpenAI Responses API JSON response.
    OpenAiResponses,
    /// OpenAI Chat Completions JSON response.
    OpenAiChatCompletions,
    /// Anthropic Messages API JSON response.
    AnthropicMessages,
    /// Gemini/Vertex GenerateContent JSON response.
    GeminiGenerateContent,
}

/// Options for transcoding a `ChatResponse` into a provider JSON response body.
#[derive(Clone, Default)]
pub struct TranscodeJsonOptions {
    /// Whether to pretty-print the JSON output.
    pub pretty: bool,
    /// Optional bridge customization applied before target JSON serialization.
    pub bridge_options: Option<BridgeOptions>,
    /// Optional partial bridge override applied on top of route/policy defaults.
    pub bridge_options_override: Option<BridgeOptionsOverride>,
    /// Optional gateway bridge policy applied by the helper.
    pub policy: Option<GatewayBridgePolicy>,
}

impl fmt::Debug for TranscodeJsonOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranscodeJsonOptions")
            .field("pretty", &self.pretty)
            .field("has_bridge_options", &self.bridge_options.is_some())
            .field(
                "has_bridge_options_override",
                &self.bridge_options_override.is_some(),
            )
            .field("has_policy", &self.policy.is_some())
            .finish()
    }
}

impl TranscodeJsonOptions {
    /// Attach bridge customization options to the JSON transcode helper.
    pub fn with_bridge_options(mut self, bridge_options: BridgeOptions) -> Self {
        self.bridge_options = Some(bridge_options);
        self
    }

    /// Attach a unified bridge customization object to the JSON transcode helper.
    pub fn with_bridge_customization(
        mut self,
        customization: Arc<dyn BridgeCustomization>,
    ) -> Self {
        self.bridge_options = Some(
            self.bridge_options
                .take()
                .unwrap_or_else(|| BridgeOptions::new(BridgeMode::BestEffort))
                .with_customization(customization),
        );
        self
    }

    /// Attach a partial bridge override to the JSON transcode helper.
    pub fn with_bridge_options_override(
        mut self,
        bridge_options_override: BridgeOptionsOverride,
    ) -> Self {
        self.bridge_options_override = Some(bridge_options_override);
        self
    }

    /// Override only the effective bridge mode used by the JSON transcode helper.
    pub fn with_bridge_mode_override(mut self, mode: BridgeMode) -> Self {
        let override_options = self
            .bridge_options_override
            .take()
            .unwrap_or_default()
            .with_mode(mode);
        self.bridge_options_override = Some(override_options);
        self
    }

    /// Attach a gateway bridge policy to the JSON transcode helper.
    pub fn with_policy(mut self, policy: GatewayBridgePolicy) -> Self {
        self.policy = Some(policy);
        self
    }
}

/// Convert a unified `ChatResponse` into a provider-native JSON response body (best-effort).
///
/// Note: This helper parses the encoded bytes back into `serde_json::Value` and is intended
/// mainly for debugging or JSON post-processing.
pub fn transcode_chat_response_to_json(
    response: &ChatResponse,
    target: TargetJsonFormat,
) -> serde_json::Value {
    let bytes =
        transcode_chat_response_to_json_bytes(response, target, &TranscodeJsonOptions::default());

    match bytes {
        Ok(payload) => serde_json::from_slice(&payload.bytes)
            .unwrap_or_else(|_| serde_json::json!({ "error": "failed to parse encoded json" })),
        Err(TranscodeJsonError::Rejected(report)) => {
            serde_json::json!({ "error": "bridge rejected", "report": report })
        }
        Err(e) => serde_json::json!({ "error": e.user_message() }),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format.
pub fn to_transcoded_json_response(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
) -> Response<Body> {
    let bytes = transcode_chat_response_to_json_bytes(&response, target, &opts);
    let policy = opts.policy.clone();

    match bytes {
        Ok(payload) => {
            let mut resp = Response::new(Body::from(payload.bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            if let Some(policy) = policy.as_ref() {
                apply_gateway_policy_headers(
                    resp.headers_mut(),
                    policy,
                    payload.target,
                    &payload.report,
                );
            }
            resp
        }
        Err(TranscodeJsonError::Rejected(report)) => Response::builder()
            .status(422)
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "error": "bridge rejected",
                    "report": report,
                }))
                .unwrap_or_else(|_| b"{\"error\":\"bridge rejected\"}".to_vec()),
            ))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
        Err(e) => Response::builder()
            .status(501)
            .header("content-type", "text/plain")
            .body(Body::from(
                if policy
                    .as_ref()
                    .map(|policy| policy.passthrough_runtime_errors)
                    .unwrap_or(true)
                {
                    e.user_message()
                } else {
                    "gateway bridge runtime error".to_string()
                },
            ))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format,
/// with a caller-provided post-processing hook.
pub fn to_transcoded_json_response_with_transform<F>(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
    transform: F,
) -> Response<Body>
where
    F: FnOnce(serde_json::Value) -> serde_json::Value + Send + Sync + 'static,
{
    let json = transcode_chat_response_to_json(&response, target);
    let json = transform(json);
    let body = if opts.pretty {
        serde_json::to_vec_pretty(&json)
    } else {
        serde_json::to_vec(&json)
    };

    match body {
        Ok(bytes) => {
            let mut resp = Response::new(Body::from(bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(_) => Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(Body::from("failed to serialize response"))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format,
/// with a caller-provided response transform hook (more efficient than JSON post-processing).
pub fn to_transcoded_json_response_with_response_transform<F>(
    mut response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
    transform: F,
) -> Response<Body>
where
    F: FnOnce(&mut ChatResponse) + Send + Sync + 'static,
{
    transform(&mut response);
    to_transcoded_json_response(response, target, opts)
}

#[derive(Debug)]
enum TranscodeJsonError {
    Runtime(LlmError),
    Rejected(siumai::experimental::bridge::BridgeReport),
}

struct TranscodedJsonPayload {
    bytes: Vec<u8>,
    report: BridgeReport,
    target: BridgeTarget,
}

impl TranscodeJsonError {
    fn user_message(&self) -> String {
        match self {
            Self::Runtime(error) => error.user_message().to_string(),
            Self::Rejected(_) => "bridge rejected".to_string(),
        }
    }
}

impl From<LlmError> for TranscodeJsonError {
    fn from(value: LlmError) -> Self {
        Self::Runtime(value)
    }
}

fn transcode_chat_response_to_json_bytes(
    response: &ChatResponse,
    target: TargetJsonFormat,
    opts: &TranscodeJsonOptions,
) -> Result<TranscodedJsonPayload, TranscodeJsonError> {
    let bridge_options = resolve_gateway_bridge_options(
        opts.policy.as_ref(),
        opts.bridge_options.clone(),
        opts.bridge_options_override.clone(),
    );
    let json_opts = JsonEncodeOptions {
        pretty: opts.pretty,
    };
    let bridge_target = bridge_target_for_json(target);
    let bridged = match target {
        TargetJsonFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                bridge_chat_response_to_openai_responses_json_bytes_with_options(
                    response,
                    None,
                    bridge_options,
                    json_opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                bridge_chat_response_to_openai_chat_completions_json_bytes_with_options(
                    response,
                    None,
                    bridge_options,
                    json_opts,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                bridge_chat_response_to_anthropic_messages_json_bytes_with_options(
                    response,
                    None,
                    bridge_options,
                    json_opts,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        TargetJsonFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                bridge_chat_response_to_gemini_generate_content_json_bytes_with_options(
                    response,
                    None,
                    bridge_options,
                    json_opts,
                )
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google feature is disabled".to_string(),
                ))
            }
        }
    }
    .map_err(TranscodeJsonError::Runtime)?;

    match bridged.into_parts() {
        (Some(bytes), report) => Ok(TranscodedJsonPayload {
            bytes,
            report,
            target: bridge_target,
        }),
        (None, report) => Err(TranscodeJsonError::Rejected(report)),
    }
}

fn bridge_target_for_json(target: TargetJsonFormat) -> BridgeTarget {
    match target {
        TargetJsonFormat::OpenAiResponses => BridgeTarget::OpenAiResponses,
        TargetJsonFormat::OpenAiChatCompletions => BridgeTarget::OpenAiChatCompletions,
        TargetJsonFormat::AnthropicMessages => BridgeTarget::AnthropicMessages,
        TargetJsonFormat::GeminiGenerateContent => BridgeTarget::GeminiGenerateContent,
    }
}

fn apply_gateway_policy_headers(
    headers: &mut axum::http::HeaderMap,
    policy: &GatewayBridgePolicy,
    target: BridgeTarget,
    report: &BridgeReport,
) {
    for entry in gateway_bridge_headers(policy, target, Some(report), report.mode) {
        if let Ok(value) = axum::http::HeaderValue::from_str(&entry.value) {
            headers.insert(entry.name, value);
        }
    }
}

#[cfg(test)]
mod json_transcode_tests {
    use super::*;

    use serde_json::json;
    use siumai::prelude::unified::{ContentPart, MessageContent};

    use crate::bridge::{
        ClosureBridgeCustomization, ClosurePrimitiveRemapper, response_bridge_hook,
    };

    #[test]
    #[cfg(feature = "openai")]
    fn openai_chat_completions_json_includes_tool_calls() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "get_weather", json!({"city":"GZ"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiChatCompletions);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    #[cfg(feature = "anthropic")]
    fn anthropic_messages_json_emits_tool_use_blocks() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("thinking"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::AnthropicMessages);
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["id"], "call_1");
        assert_eq!(v["content"][1]["name"], "search");
    }

    #[test]
    #[cfg(feature = "google")]
    fn gemini_generate_content_json_emits_function_call_parts() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::GeminiGenerateContent);
        assert_eq!(v["candidates"][0]["content"]["role"], "model");
        assert_eq!(
            v["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
            "search"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn openai_responses_json_includes_output_items() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiResponses);
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "call_1");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn openai_chat_completions_json_applies_bridge_primitive_remapper() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let payload = transcode_chat_response_to_json_bytes(
            &resp,
            TargetJsonFormat::OpenAiChatCompletions,
            &TranscodeJsonOptions::default().with_bridge_options(
                BridgeOptions::new(siumai::experimental::bridge::BridgeMode::BestEffort)
                    .with_route_label("tests.axum.json.remap")
                    .with_primitive_remapper(Arc::new(
                        ClosurePrimitiveRemapper::default()
                            .with_tool_name(|_, name| Some(format!("gw_{name}"))),
                    )),
            ),
        )
        .expect("json bytes");

        let v: serde_json::Value = serde_json::from_slice(&payload.bytes).expect("json value");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "gw_search"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn openai_responses_json_applies_response_bridge_hook() {
        let resp = ChatResponse::new(MessageContent::Text("original".to_string()));

        let payload = transcode_chat_response_to_json_bytes(
            &resp,
            TargetJsonFormat::OpenAiResponses,
            &TranscodeJsonOptions::default().with_bridge_options(
                BridgeOptions::new(siumai::experimental::bridge::BridgeMode::BestEffort)
                    .with_route_label("tests.axum.json.response-hook")
                    .with_response_hook(response_bridge_hook(|ctx, response, _report| {
                        assert_eq!(
                            ctx.target,
                            siumai::experimental::bridge::BridgeTarget::OpenAiResponses
                        );
                        response.content = MessageContent::Text("hooked".to_string());
                        Ok(())
                    })),
            ),
        )
        .expect("json bytes");

        let v: serde_json::Value = serde_json::from_slice(&payload.bytes).expect("json value");
        assert_eq!(v["output"][0]["content"][0]["text"], "hooked");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn openai_responses_json_applies_unified_bridge_customization() {
        let resp = ChatResponse::new(MessageContent::Text("original".to_string()));

        let payload = transcode_chat_response_to_json_bytes(
            &resp,
            TargetJsonFormat::OpenAiResponses,
            &TranscodeJsonOptions::default().with_bridge_customization(Arc::new(
                ClosureBridgeCustomization::default().with_response(|ctx, response, _report| {
                    assert_eq!(
                        ctx.target,
                        siumai::experimental::bridge::BridgeTarget::OpenAiResponses
                    );
                    response.content = MessageContent::Text("customized".to_string());
                    Ok(())
                }),
            )),
        )
        .expect("json bytes");

        let v: serde_json::Value = serde_json::from_slice(&payload.bytes).expect("json value");
        assert_eq!(v["output"][0]["content"][0]["text"], "customized");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn json_gateway_policy_emits_bridge_headers() {
        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));

        let response = to_transcoded_json_response(
            resp,
            TargetJsonFormat::OpenAiResponses,
            TranscodeJsonOptions::default().with_policy(
                GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
                    .with_bridge_headers(true)
                    .with_bridge_warning_headers(true),
            ),
        );

        assert_eq!(
            response.headers()["x-siumai-bridge-target"],
            "openai-responses"
        );
        assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");
        assert_eq!(response.headers()["x-siumai-bridge-decision"], "exact");
        assert_eq!(response.headers()["x-siumai-bridge-warnings"], "0");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn json_gateway_route_mode_override_updates_effective_headers() {
        let resp = ChatResponse::new(MessageContent::Text("ok".to_string()));

        let response = to_transcoded_json_response(
            resp,
            TargetJsonFormat::OpenAiResponses,
            TranscodeJsonOptions::default()
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort).with_bridge_headers(true),
                )
                .with_bridge_mode_override(BridgeMode::Strict),
        );

        assert_eq!(response.headers()["x-siumai-bridge-mode"], "strict");
    }
}
