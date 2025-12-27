#![cfg(feature = "google")]
//! Gemini provider-defined tools mapping integration test

use siumai::providers::gemini::convert::build_request_body;
use siumai::providers::gemini::types::GeminiConfig;
use siumai::types::{ChatMessage, Tool};

#[test]
fn gemini_build_request_includes_provider_defined_tools() {
    // Minimal config with a model id
    let cfg = GeminiConfig {
        // Use Gemini 2.x/2.5 behavior: google_search/url_context are supported as dedicated tools.
        model: "gemini-2.5-flash".to_string(),
        ..Default::default()
    };

    // One user message
    let messages = vec![ChatMessage::user("hi").build()];

    // Mix of function tool and Google provider-defined tools
    let tools = vec![
        Tool::function(
            "lookup".to_string(),
            "test function".to_string(),
            serde_json::json!({"type":"object"}),
        ),
        Tool::provider_defined("google.code_execution", "code_execution"),
        Tool::provider_defined("google.google_search", "google_search"),
        Tool::provider_defined("google.google_search_retrieval", "google_search_retrieval"),
        Tool::provider_defined("google.url_context", "url_context"),
    ];

    let req = build_request_body(&cfg, &messages, Some(&tools)).expect("build ok");
    let gemini_tools = req.tools.expect("tools present");

    // Expect a FunctionDeclarations block containing our function
    assert!(gemini_tools.iter().any(|t| match t {
        siumai::providers::gemini::types::GeminiTool::FunctionDeclarations {
            function_declarations,
        } => {
            function_declarations.iter().any(|f| f.name == "lookup")
        }
        _ => false,
    }));

    // Expect the Google provider-defined tools to be turned into Gemini variants
    assert!(gemini_tools.iter().any(|t| matches!(
        t,
        siumai::providers::gemini::types::GeminiTool::CodeExecution { .. }
    )));

    assert!(gemini_tools.iter().any(|t| matches!(
        t,
        siumai::providers::gemini::types::GeminiTool::GoogleSearch { .. }
    )));

    assert!(gemini_tools.iter().any(|t| matches!(
        t,
        siumai::providers::gemini::types::GeminiTool::GoogleSearchRetrieval { .. }
    )));

    assert!(gemini_tools.iter().any(|t| matches!(
        t,
        siumai::providers::gemini::types::GeminiTool::UrlContext { .. }
    )));
}

#[test]
fn gemini_1_5_maps_google_search_to_search_retrieval_and_omits_url_context() {
    // Gemini 1.5 uses googleSearchRetrieval for grounding (Vercel-aligned).
    let cfg = GeminiConfig {
        model: "gemini-1.5-pro".to_string(),
        ..Default::default()
    };

    let messages = vec![ChatMessage::user("hi").build()];
    let tools = vec![
        Tool::provider_defined("google.google_search", "google_search"),
        Tool::provider_defined("google.url_context", "url_context"),
    ];

    let req = build_request_body(&cfg, &messages, Some(&tools)).expect("build ok");
    let gemini_tools = req.tools.expect("tools present");

    assert!(
        gemini_tools.iter().any(|t| matches!(
            t,
            siumai::providers::gemini::types::GeminiTool::GoogleSearchRetrieval { .. }
        )),
        "Gemini 1.5 should map google_search to googleSearchRetrieval"
    );

    assert!(
        !gemini_tools.iter().any(|t| matches!(
            t,
            siumai::providers::gemini::types::GeminiTool::UrlContext { .. }
        )),
        "Gemini 1.5 should not include urlContext tool"
    );
}
