//! Gemini tool warning parity middleware (Vercel AI SDK aligned).

use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, Warning};

#[derive(Debug, Default)]
pub struct GeminiToolWarningsMiddleware;

impl GeminiToolWarningsMiddleware {
    pub const fn new() -> Self {
        Self
    }

    fn is_gemini_2_or_newer(model_id: &str) -> bool {
        let is_latest = matches!(
            model_id,
            "gemini-flash-latest" | "gemini-flash-lite-latest" | "gemini-pro-latest"
        );
        model_id.contains("gemini-2") || model_id.contains("gemini-3") || is_latest
    }

    fn supports_file_search(model_id: &str) -> bool {
        model_id.contains("gemini-2.5") || model_id.contains("gemini-3")
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
        let is_gemini_2_or_newer = Self::is_gemini_2_or_newer(model_id);
        let supports_file_search = Self::supports_file_search(model_id);

        for tool in tools {
            let Tool::ProviderDefined(provider_tool) = tool else {
                continue;
            };

            match provider_tool.id.as_str() {
                "google.google_search" => {}
                "google.enterprise_web_search" => {
                    if !is_gemini_2_or_newer {
                        warnings.push(Warning::unsupported_tool(
                            "google.enterprise_web_search",
                            Some("Enterprise Web Search requires Gemini 2.0 or newer."),
                        ));
                    }
                }
                "google.url_context" => {
                    if !is_gemini_2_or_newer {
                        warnings.push(Warning::unsupported_tool(
                            "google.url_context",
                            Some(
                                "The URL context tool is not supported with other Gemini models than Gemini 2.",
                            ),
                        ));
                    }
                }
                "google.code_execution" => {
                    if !is_gemini_2_or_newer {
                        warnings.push(Warning::unsupported_tool(
                            "google.code_execution",
                            Some(
                                "The code execution tools is not supported with other Gemini models than Gemini 2.",
                            ),
                        ));
                    }
                }
                "google.file_search" => {
                    if !supports_file_search {
                        warnings.push(Warning::unsupported_tool(
                            "google.file_search",
                            Some(
                                "The file search tool is only supported with Gemini 2.5 models and Gemini 3 models.",
                            ),
                        ));
                    }
                }
                "google.vertex_rag_store" => {
                    if !is_gemini_2_or_newer {
                        warnings.push(Warning::unsupported_tool(
                            "google.vertex_rag_store",
                            Some(
                                "The RAG store tool is not supported with other Gemini models than Gemini 2.",
                            ),
                        ));
                    }
                }
                "google.google_maps" => {
                    if !is_gemini_2_or_newer {
                        warnings.push(Warning::unsupported_tool(
                            "google.google_maps",
                            Some(
                                "The Google Maps grounding tool is not supported with Gemini models other than Gemini 2 or newer.",
                            ),
                        ));
                    }
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
                crate::tools::google::google_search(),
            ])
            .build();

        let mw = GeminiToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        assert!(out.warnings.is_some());
        assert!(out.warnings.unwrap().iter().any(
            |w| matches!(w, Warning::UnsupportedSetting { setting, .. } if setting == "tools")
        ));
    }

    #[test]
    fn warns_on_unsupported_url_context_on_gemini_1_5() {
        let req = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                model: "gemini-1.5-pro".to_string(),
                ..Default::default()
            })
            .tools(vec![crate::tools::google::url_context()])
            .build();

        let mw = GeminiToolWarningsMiddleware::new();
        let out = mw.post_generate(&req, dummy_resp()).unwrap();
        let warnings = out.warnings.unwrap_or_default();
        assert!(
            warnings.iter().any(|w| matches!(w, Warning::UnsupportedTool { tool_name, .. } if tool_name == "google.url_context"))
        );
    }
}
