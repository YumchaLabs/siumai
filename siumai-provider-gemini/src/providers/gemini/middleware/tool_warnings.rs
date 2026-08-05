//! Gemini tool warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, Warning};

use super::super::model_policy::{CapabilitySupport, model_capability_support};

#[derive(Debug, Default)]
pub struct GeminiToolWarningsMiddleware;

impl GeminiToolWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn warn_if_explicitly_unsupported(
        warnings: &mut Vec<Warning>,
        model_id: &str,
        capability: &str,
        tool_id: &str,
    ) {
        if model_capability_support(model_id, capability) == CapabilitySupport::Unsupported {
            warnings.push(Warning::unsupported_tool(
                tool_id,
                Some(format!(
                    "The '{tool_id}' tool is not supported by Gemini model '{model_id}'."
                )),
            ));
        }
    }

    fn compute_warnings(req: &ChatRequest) -> Vec<Warning> {
        let Some(tools) = req.tools.as_deref() else {
            return Vec::new();
        };
        if tools.is_empty() {
            return Vec::new();
        }

        let has_function_tools = tools.iter().any(|t| matches!(t, Tool::Function { .. }));
        let has_provider_tools = tools.iter().any(|t| matches!(t, Tool::ProviderDefined(_)));

        let mut warnings = Vec::new();

        if has_function_tools && has_provider_tools {
            warnings.push(Warning::unsupported_setting(
                "tools",
                Some("combination of function and provider-defined tools"),
            ));
        }

        if !has_provider_tools {
            return warnings;
        }

        // Vercel AI SDK alignment: `google.vertex_rag_store` is a Vertex-only tool. When used with
        // the Google Generative AI provider it may not be supported, so emit an informational warning.
        if tools.iter().any(|t| {
            matches!(
                t,
                Tool::ProviderDefined(provider_tool) if provider_tool.id == "google.vertex_rag_store"
            )
        }) {
            warnings.push(Warning::other(
                "The 'vertex_rag_store' tool is only supported with the Google Vertex provider and might not be supported or could behave unexpectedly with the current Google provider (gemini).",
            ));
        }

        let model_id = req.common_params.model.as_str();

        for tool in tools {
            let Tool::ProviderDefined(provider_tool) = tool else {
                continue;
            };

            match provider_tool.id.as_str() {
                "google.google_search" => Self::warn_if_explicitly_unsupported(
                    &mut warnings,
                    model_id,
                    "search_grounding",
                    "google.google_search",
                ),
                "google.enterprise_web_search" => {
                    Self::warn_if_explicitly_unsupported(
                        &mut warnings,
                        model_id,
                        "search_grounding",
                        "google.enterprise_web_search",
                    );
                }
                "google.url_context" => {
                    Self::warn_if_explicitly_unsupported(
                        &mut warnings,
                        model_id,
                        "url_context",
                        "google.url_context",
                    );
                }
                "google.code_execution" => {
                    Self::warn_if_explicitly_unsupported(
                        &mut warnings,
                        model_id,
                        "code_execution",
                        "google.code_execution",
                    );
                }
                "google.file_search" => {
                    Self::warn_if_explicitly_unsupported(
                        &mut warnings,
                        model_id,
                        "file_search",
                        "google.file_search",
                    );
                }
                "google.vertex_rag_store" => {}
                "google.google_maps" => {
                    Self::warn_if_explicitly_unsupported(
                        &mut warnings,
                        model_id,
                        "maps_grounding",
                        "google.google_maps",
                    );
                }
                _ => {
                    warnings.push(Warning::unsupported_tool(
                        provider_tool.id.clone(),
                        None::<String>,
                    ));
                }
            }
        }

        warnings
    }

    fn merge_warnings(mut resp: ChatResponse, additional: Vec<Warning>) -> ChatResponse {
        if additional.is_empty() {
            return resp;
        }

        match resp.warnings.as_mut() {
            Some(existing) => existing.extend(additional),
            None => resp.warnings = Some(additional),
        }
        resp
    }
}

impl LanguageModelMiddleware for GeminiToolWarningsMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        Ok(Self::merge_warnings(resp, Self::compute_warnings(req)))
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { response } => {
                let response = Self::merge_warnings(response, Self::compute_warnings(req));
                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::model_constants::gemini_3;
    use crate::types::{ChatMessage, MessageContent};

    fn dummy_resp() -> ChatResponse {
        ChatResponse::new(MessageContent::Text("ok".to_string()))
    }

    #[test]
    fn warns_on_mixed_function_and_provider_tools() {
        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                model: "gemini-2.5-flash".to_string(),
                ..Default::default()
            })
            .tools(vec![
                Tool::function("f".to_string(), "".to_string(), serde_json::json!({})),
                siumai_protocol_gemini::tool_catalog::google::google_search(),
            ])
            .build();

        let mw = GeminiToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        assert!(out.warnings.is_some());
        assert!(
            out.warnings
                .unwrap()
                .iter()
                .any(|w| matches!(w, Warning::Unsupported { feature, .. } if feature == "tools"))
        );
    }

    #[test]
    fn warns_on_explicitly_unsupported_url_context() {
        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                model: gemini_3::GEMINI_3_1_FLASH_IMAGE.to_string(),
                ..Default::default()
            })
            .tools(vec![
                siumai_protocol_gemini::tool_catalog::google::url_context(),
            ])
            .build();

        let mw = GeminiToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();
        assert!(warnings.iter().any(
            |w| matches!(w, Warning::Unsupported { feature, .. } if feature == "google.url_context")
        ));
    }

    #[test]
    fn unknown_models_do_not_receive_guessed_tool_warnings() {
        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                model: "gemini-4-future".to_string(),
                ..Default::default()
            })
            .tools(vec![
                siumai_protocol_gemini::tool_catalog::google::url_context(),
            ])
            .build();

        let mw = GeminiToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        assert!(out.warnings.unwrap_or_default().is_empty());
    }
}
