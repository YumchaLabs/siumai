#![cfg(feature = "google")]

use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_gemini_generate_content_json_value,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use siumai::experimental::standards::gemini::transformers::GeminiResponseTransformer;
use siumai::prelude::unified::{ContentPart, FinishReason, MessageContent};

fn count_reasoning_parts(content: &MessageContent) -> usize {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| matches!(part, ContentPart::Reasoning { .. }))
            .count(),
        _ => 0,
    }
}

#[test]
fn gemini_generate_content_response_bridge_roundtrip_preserves_projected_visible_output() {
    let raw = serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "internal thinking",
                        "thought": true,
                        "thoughtSignature": "sig_1"
                    },
                    {
                        "text": "visible answer"
                    },
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": { "city": "Tokyo" }
                        }
                    }
                ]
            },
            "finishReason": "STOP",
            "groundingMetadata": {
                "searchEntryPoint": { "renderedContent": "<div/>" },
                "groundingChunks": [
                    { "web": { "uri": "https://example.com/fact", "title": "Fact" } }
                ]
            },
            "urlContextMetadata": {
                "urlMetadata": [
                    {
                        "retrievedUrl": "https://example.com/fact",
                        "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                    }
                ]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "totalTokenCount": 18,
            "thoughtsTokenCount": 4
        },
        "modelVersion": "gemini-2.5-pro"
    });

    let tx = GeminiResponseTransformer {
        config: siumai::protocol::gemini::types::GeminiConfig::default(),
    };
    let response = tx
        .transform_chat_response(&raw)
        .expect("transform raw gemini response");

    assert_eq!(count_reasoning_parts(&response.content), 1);
    assert!(
        response
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("google"))
            .and_then(|meta| meta.get("sources"))
            .is_some(),
        "expected source metadata on initial Gemini transform"
    );

    let bridged = bridge_chat_response_to_gemini_generate_content_json_value(
        &response,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(
        !bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "content[0]"),
        "expected Gemini reasoning block to survive roundtrip bridge"
    );

    let bridged_json = bridged.value.expect("bridged json");
    assert_eq!(
        bridged_json["candidates"][0]["finishReason"],
        serde_json::json!("STOP")
    );
    assert_eq!(
        bridged_json["modelVersion"],
        serde_json::json!("gemini-2.5-pro")
    );
    assert_eq!(
        bridged_json["usageMetadata"]["thoughtsTokenCount"],
        serde_json::json!(4)
    );
    assert!(
        bridged_json["candidates"][0]["groundingMetadata"].is_object(),
        "expected grounding metadata to survive bridge"
    );
    assert!(
        bridged_json["candidates"][0]["urlContextMetadata"].is_object(),
        "expected url context metadata to survive bridge"
    );

    let roundtripped = tx
        .transform_chat_response(&bridged_json)
        .expect("transform bridged gemini response");

    assert_eq!(roundtripped.content_text(), Some("visible answer"));
    assert_eq!(count_reasoning_parts(&roundtripped.content), 1);
    assert_eq!(roundtripped.finish_reason, Some(FinishReason::ToolCalls));

    let calls = roundtripped.tool_calls();
    assert_eq!(calls.len(), 1);
    let info = calls[0].as_tool_call().expect("tool call info");
    assert_eq!(info.tool_name, "weather");
    assert_eq!(info.arguments, &serde_json::json!({ "city": "Tokyo" }));

    let usage = roundtripped.usage.as_ref().expect("usage");
    assert_eq!(usage.prompt_tokens, 11);
    assert_eq!(usage.completion_tokens, 7);
    assert_eq!(usage.total_tokens, 18);
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        Some(4)
    );

    let google_meta = roundtripped
        .provider_metadata
        .as_ref()
        .and_then(|meta| meta.get("google"))
        .expect("google provider metadata");
    assert!(google_meta.get("usageMetadata").is_some());
    assert!(google_meta.get("groundingMetadata").is_some());
    assert!(google_meta.get("urlContextMetadata").is_some());
    assert!(google_meta.get("sources").is_some());
}
