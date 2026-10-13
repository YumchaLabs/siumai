use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::annotations::{
    ContentAnnotationTarget, DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
    DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT, DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
    MessageAnnotationTarget, ProviderAnnotationBudget, ProviderAnnotationError,
    TypedProviderAnnotation,
};
use crate::error::Error as SiumaiError;
use crate::provider::{ModelId, ProtocolId, ProviderId, ProviderScope, ReplayDomain};
use crate::tool::{ToolCall, ToolOutcome, ToolResult, ToolSpec};
use crate::usage::Usage;
use bytes::Bytes;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheAnnotation {
    cache_control: String,
}

impl TypedProviderAnnotation for CacheAnnotation {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some("messages");
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageLabelAnnotation {
    label: String,
}

impl TypedProviderAnnotation for MessageLabelAnnotation {
    type Target = MessageAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some("messages");
}

fn provenance() -> ProviderProvenance {
    let scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_protocol(ProtocolId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::official(
            crate::provider::ReplayDomainId::new("openai-public-api").unwrap(),
        ));
    ProviderProvenance::from_scope(&scope, ModelId::new("gpt-future:preview").unwrap()).unwrap()
}

#[test]
fn provider_provenance_deserialization_cannot_bypass_replay_requirements() {
    let missing_protocol = json!({
        "provider": "openai",
        "platform": "public-api",
        "api_mode": "responses",
        "replay_domain": {
            "audience": {"Official": "public-api"}
        },
        "model": "gpt-5.6"
    });
    let missing_domain = json!({
        "provider": "openai",
        "platform": "public-api",
        "protocol": "openai-responses",
        "api_mode": "responses",
        "model": "gpt-5.6"
    });

    assert!(serde_json::from_value::<ProviderProvenance>(missing_protocol).is_err());
    assert!(serde_json::from_value::<ProviderProvenance>(missing_domain).is_err());
}

#[test]
fn opaque_items_round_trip_with_provenance() {
    let item = OpaqueProviderItem::builder(
        provenance(),
        "function_call",
        json!({
            "id": "item_1",
            "call_id": "call_1",
            "caller": {"type": "program", "caller_id": "program_1"}
        }),
    )
    .item_id("item_1")
    .relation(ProviderItemRelation::call("call_1").unwrap())
    .relation(ProviderItemRelation::caller("program_1").unwrap())
    .build()
    .unwrap();

    let encoded = serde_json::to_value(&item).unwrap();
    let decoded: OpaqueProviderItem = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, item);
    assert_eq!(decoded.provenance().protocol().as_str(), "responses");
    assert_eq!(decoded.item_id(), Some("item_1"));
    assert_eq!(decoded.relations()[0].kind(), "call");
    assert_eq!(decoded.relations()[0].target_id(), "call_1");
    assert_eq!(decoded.relations()[1].kind(), "caller");
    assert_eq!(decoded.relations()[1].target_id(), "program_1");
}

#[test]
fn opaque_item_debug_redacts_provider_state() {
    let item_id = "provider-item-secret-canary";
    let relation_target = "provider-relation-secret-canary";
    let relation = ProviderItemRelation::caller(relation_target).unwrap();
    let relation_debug = format!("{relation:?}");
    assert!(!relation_debug.contains(relation_target));
    assert!(relation_debug.contains("kind: \"caller\""));
    assert!(relation_debug.contains(&format!("target_id_bytes: {}", relation_target.len())));

    let builder = OpaqueProviderItem::builder(
        provenance(),
        "encrypted_reasoning",
        json!({"encrypted_content": "provider-secret-canary"}),
    )
    .item_id(item_id)
    .relation(relation);
    let builder_debug = format!("{builder:?}");
    assert!(!builder_debug.contains("provider-secret-canary"));
    assert!(!builder_debug.contains(item_id));
    assert!(!builder_debug.contains(relation_target));
    assert!(!builder_debug.contains("provenance"));
    assert!(!builder_debug.contains("data"));
    assert!(builder_debug.contains("kind: \"encrypted_reasoning\""));
    assert!(builder_debug.contains("has_item_id: true"));
    assert!(builder_debug.contains("relation_count: 1"));
    assert!(builder_debug.contains("maximum_bytes"));

    let item = builder.build().unwrap();
    let item_debug = format!("{item:?}");
    assert!(!item_debug.contains("provider-secret-canary"));
    assert!(!item_debug.contains(item_id));
    assert!(!item_debug.contains(relation_target));
    assert!(!item_debug.contains("provenance"));
    assert!(!item_debug.contains("data"));
    assert!(item_debug.contains("kind: \"encrypted_reasoning\""));
    assert!(item_debug.contains("has_item_id: true"));
    assert!(item_debug.contains("relation_count: 1"));
    assert!(item_debug.contains("encoded_json_bytes"));
}

#[test]
fn opaque_items_enforce_a_byte_limit() {
    let error = OpaqueProviderItem::with_limit(
        provenance(),
        "large",
        Value::String("too large".to_string()),
        2,
    )
    .unwrap_err();
    assert!(matches!(error, OpaqueProviderItemError::TooLarge { .. }));
}

#[test]
fn opaque_collection_budget_bounds_count_and_total_bytes() {
    let item = OpaqueProviderItem::new(
        provenance(),
        "reasoning",
        json!({"encrypted_content": "opaque"}),
    )
    .unwrap();
    let item_bytes = item.encoded_json_bytes();

    OpaqueProviderBudget::new(1, item_bytes, item_bytes)
        .validate([&item])
        .unwrap();
    assert!(matches!(
        OpaqueProviderBudget::new(1, item_bytes, item_bytes * 2).validate([&item, &item]),
        Err(OpaqueProviderItemError::TooManyItems { .. })
    ));
    assert!(matches!(
        OpaqueProviderBudget::new(2, item_bytes, item_bytes.saturating_sub(1)).validate([&item]),
        Err(OpaqueProviderItemError::CollectionTooLarge { .. })
    ));
}

#[test]
fn language_request_applies_opaque_budget_without_parsing_payloads() {
    let first = OpaqueProviderItem::new(provenance(), "program", json!({"wire": 1})).unwrap();
    let second =
        OpaqueProviderItem::new(provenance(), "program_output", json!({"wire": 2})).unwrap();
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::Assistant,
        [
            ContentPart::ProviderOpaque(first),
            ContentPart::ProviderOpaque(second),
        ],
    )]);

    assert!(matches!(
        request.validate_with_budget(LanguageRequestBudget::new(
            OpaqueProviderBudget::new(
                1,
                DEFAULT_OPAQUE_ITEM_LIMIT,
                DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
            ),
            ProviderAnnotationBudget::default(),
        )),
        Err(LanguageRequestError::OpaqueProviderItems(
            OpaqueProviderItemError::TooManyItems { .. }
        ))
    ));
}

#[test]
fn message_parts_are_ergonomic_and_preserve_annotations_through_serde() {
    let expected = CacheAnnotation {
        cache_control: "ephemeral".to_string(),
    };
    let part = MessagePart::text("hello")
        .with_provider_annotation(&expected)
        .unwrap();
    let message = Message::new(MessageRole::User, [part]);

    let decoded: Message = serde_json::from_value(serde_json::to_value(&message).unwrap()).unwrap();
    assert_eq!(decoded.role(), MessageRole::User);
    assert!(matches!(
        decoded.content()[0].content(),
        ContentPart::Text { text } if text == "hello"
    ));
    assert_eq!(
        decoded.content()[0]
            .annotations()
            .decode::<CacheAnnotation>()
            .unwrap(),
        Some(expected)
    );
}

#[test]
fn message_rebuild_paths_preserve_node_annotations() {
    let content_annotation = CacheAnnotation {
        cache_control: "ephemeral".to_string(),
    };
    let message_annotation = MessageLabelAnnotation {
        label: "history-entry".to_string(),
    };
    let mut part = MessagePart::text("before")
        .with_provider_annotation(&content_annotation)
        .unwrap();
    *part.content_mut() = ContentPart::Text {
        text: "after".to_string(),
    };
    let message = Message::new(MessageRole::User, [part])
        .with_provider_annotation(&message_annotation)
        .unwrap();

    let (role, content, annotations) = message.into_parts();
    let rebuilt = Message::from_parts(role, content, annotations);

    assert!(matches!(
        rebuilt.content()[0].content(),
        ContentPart::Text { text } if text == "after"
    ));
    assert_eq!(
        rebuilt.content()[0]
            .annotations()
            .decode::<CacheAnnotation>()
            .unwrap(),
        Some(content_annotation)
    );
    assert_eq!(
        rebuilt
            .annotations()
            .decode::<MessageLabelAnnotation>()
            .unwrap(),
        Some(message_annotation)
    );
}

#[test]
fn language_request_budget_counts_annotations_across_nodes() {
    let annotation = CacheAnnotation {
        cache_control: "ephemeral".to_string(),
    };
    let first = MessagePart::text("first")
        .with_provider_annotation(&annotation)
        .unwrap();
    let second = MessagePart::text("second")
        .with_provider_annotation(&annotation)
        .unwrap();
    let entry_bytes = first.annotations().encoded_bytes();
    let request = LanguageRequest::new(vec![Message::new(MessageRole::User, [first, second])]);
    let budget = LanguageRequestBudget::new(
        OpaqueProviderBudget::default(),
        ProviderAnnotationBudget::new(
            DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
            1,
            DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
            DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
        ),
    );

    assert!(matches!(
        request.validate_with_budget(budget),
        Err(LanguageRequestError::ProviderAnnotations(
            ProviderAnnotationError::TooManyEntries { .. }
        ))
    ));

    let total_budget = LanguageRequestBudget::new(
        OpaqueProviderBudget::default(),
        ProviderAnnotationBudget::new(
            DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
            2,
            DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
            entry_bytes.saturating_mul(2).saturating_sub(1),
        ),
    );
    assert!(matches!(
        request.validate_with_budget(total_budget),
        Err(LanguageRequestError::ProviderAnnotations(
            ProviderAnnotationError::CollectionTooLarge { .. }
        ))
    ));
}

#[test]
fn language_input_preserves_prompt_bytes_without_policy() {
    for text in ["", "  keep leading and trailing whitespace  "] {
        let request = LanguageInput::from(text).into_request();

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role(), MessageRole::User);
        assert!(matches!(
            request.messages[0].content()[0].content(),
            ContentPart::Text { text: actual } if actual == text
        ));
    }

    let owned = String::from("owned\0utf8-你好");
    let request = LanguageInput::from(owned.clone()).into_request();
    assert!(matches!(
        request.messages[0].content()[0].content(),
        ContentPart::Text { text } if text.as_bytes() == owned.as_bytes()
    ));
}

#[test]
fn language_input_preserves_messages_and_complete_requests() {
    let content_annotation = CacheAnnotation {
        cache_control: "ephemeral".to_string(),
    };
    let message_annotation = MessageLabelAnnotation {
        label: "policy".to_string(),
    };
    let annotated_message = Message::new(
        MessageRole::Developer,
        [MessagePart::text("preserve")
            .with_provider_annotation(&content_annotation)
            .unwrap()],
    )
    .with_provider_annotation(&message_annotation)
    .unwrap();

    assert_eq!(
        LanguageInput::from(annotated_message.clone())
            .into_request()
            .messages,
        vec![annotated_message.clone()]
    );

    let messages = vec![annotated_message.clone(), Message::user("question")];
    assert_eq!(
        LanguageInput::from(messages.clone())
            .into_request()
            .messages,
        messages
    );

    let replay_message = Message::new(
        MessageRole::Assistant,
        [ContentPart::ProviderOpaque(
            OpaqueProviderItem::new(provenance(), "response.output", json!({"id": "item-1"}))
                .unwrap(),
        )],
    );
    let request = LanguageRequest {
        messages: vec![annotated_message, replay_message],
        generation: GenerationConfig {
            max_output_tokens: Some(123),
            temperature: Some(0.25),
            top_p: Some(0.9),
            stop_sequences: vec!["stop".to_string()],
            seed: Some(7),
        },
        tools: vec![
            ToolSpec::new(
                "lookup",
                Some("Look up a value".to_string()),
                json!({"type": "object"}),
            )
            .unwrap(),
        ],
        tool_choice: Some(ToolChoice::Named {
            name: "lookup".to_string(),
        }),
        structured_output: Some(StructuredOutputSpec {
            name: "answer".to_string(),
            description: Some("Structured answer".to_string()),
            schema: json!({"type": "object"}),
            strict: true,
        }),
    };
    let expected = request.clone();

    assert_eq!(LanguageInput::from(request).into_request(), expected);
}

#[test]
fn language_response_has_one_completed_or_incomplete_termination() {
    let completed =
        LanguageResponse::completed(Vec::new(), LanguageCompletionReason::Stop, Usage::default())
            .unwrap();
    assert_eq!(
        completed.termination(),
        &LanguageTermination::Completed(LanguageCompletionReason::Stop)
    );

    let incomplete = LanguageResponse::incomplete(
        Vec::new(),
        LanguageIncompleteReason::MaxOutputTokens,
        Usage::default(),
    )
    .unwrap();
    let encoded = serde_json::to_value(&incomplete).unwrap();
    assert!(encoded.get("termination").is_some());
    assert!(encoded.get("status").is_none());
    assert!(encoded.get("finish_reason").is_none());

    let decoded: LanguageResponse = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        decoded.termination(),
        &LanguageTermination::Incomplete(LanguageIncompleteReason::MaxOutputTokens)
    );
    assert_eq!(
        LanguageResponse::completed(
            Vec::new(),
            LanguageCompletionReason::Other("  ".to_string()),
            Usage::default(),
        ),
        Err(LanguageResponseError::EmptyCompletionReason)
    );
    assert_eq!(
        LanguageResponse::incomplete(
            Vec::new(),
            LanguageIncompleteReason::Other(String::new()),
            Usage::default(),
        ),
        Err(LanguageResponseError::EmptyIncompleteReason)
    );
}

#[test]
fn language_response_text_projection_uses_only_canonical_text_parts() {
    let response = LanguageResponse::completed(
        vec![
            ContentPart::Text {
                text: String::new(),
            },
            ContentPart::Reasoning {
                text: "hidden reasoning".to_string(),
            },
            ContentPart::Text {
                text: "first".to_string(),
            },
            ContentPart::Media(MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Bytes(Bytes::from_static(b"image")),
                name: None,
            }),
            ContentPart::Citation(Citation {
                source_id: "source-1".to_string(),
                title: Some("source title".to_string()),
                url: None,
                start: None,
                end: None,
                provider: BTreeMap::new(),
            }),
            ContentPart::Refusal {
                reason: Some("refusal detail".to_string()),
            },
            ContentPart::ToolCall(
                ToolCall::local("call-1", "lookup", json!({"query": "value"})).unwrap(),
            ),
            ContentPart::ToolResult(ToolResult {
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                outcome: ToolOutcome::Success { value: json!(1) },
            }),
            ContentPart::ProviderOpaque(
                OpaqueProviderItem::new(provenance(), "response.output", json!({"opaque": true}))
                    .unwrap(),
            ),
            ContentPart::Text {
                text: "second".to_string(),
            },
        ],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .unwrap();

    assert_eq!(
        response.text_parts().collect::<Vec<_>>(),
        vec!["", "first", "second"]
    );
    assert_eq!(response.output_text(), Some("firstsecond".to_string()));

    let no_text = LanguageResponse::completed(
        vec![ContentPart::Reasoning {
            text: "reasoning only".to_string(),
        }],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .unwrap();
    assert_eq!(no_text.output_text(), None);

    let empty_text = LanguageResponse::completed(
        vec![ContentPart::Text {
            text: String::new(),
        }],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .unwrap();
    assert_eq!(empty_text.output_text(), Some(String::new()));
}

#[test]
fn partial_language_output_text_projection_matches_response_semantics() {
    let partial = PartialLanguageOutput::new(
        vec![
            PartialLanguageOutputPart::Reasoning {
                text: "hidden reasoning".to_string(),
            },
            PartialLanguageOutputPart::Text {
                text: String::new(),
            },
            PartialLanguageOutputPart::Refusal {
                reason: Some("refusal detail".to_string()),
            },
            PartialLanguageOutputPart::Text {
                text: "visible".to_string(),
            },
        ],
        Usage::default(),
    )
    .unwrap();

    assert_eq!(
        partial.text_parts().collect::<Vec<_>>(),
        vec!["", "visible"]
    );
    assert_eq!(partial.output_text(), Some("visible".to_string()));

    let no_text = PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Refusal { reason: None }],
        Usage::default(),
    )
    .unwrap();
    assert_eq!(no_text.output_text(), None);

    let empty_text = PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Text {
            text: String::new(),
        }],
        Usage::default(),
    )
    .unwrap();
    assert_eq!(empty_text.output_text(), Some(String::new()));
}

#[test]
fn partial_language_output_is_bounded_and_preserves_unknown_usage() {
    let part = PartialLanguageOutputPart::Text {
        text: "partial".to_string(),
    };
    let item_bytes = serde_json::to_vec(&part).unwrap().len();
    let usage = Usage::default()
        .with_input_tokens(0_u64)
        .with_provider_value("private_usage_detail", "tenant-secret");
    let partial = PartialLanguageOutput::new(vec![part.clone()], usage.clone()).unwrap();
    let encoded_bytes = partial.encoded_json_bytes();

    let decoded: PartialLanguageOutput =
        serde_json::from_value(serde_json::to_value(&partial).unwrap()).unwrap();
    assert_eq!(decoded.usage().input_tokens, crate::UsageValue::Known(0));
    assert_eq!(decoded.usage().output_tokens, crate::UsageValue::Unknown);
    assert!(decoded.usage().provider.is_empty());
    assert_eq!(decoded.encoded_json_bytes(), encoded_bytes);

    assert!(matches!(
        PartialLanguageOutput::with_budget(
            vec![part.clone()],
            usage.clone(),
            PartialLanguageOutputBudget::new(0, item_bytes, encoded_bytes),
        ),
        Err(PartialLanguageOutputError::TooManyItems { .. })
    ));
    assert!(matches!(
        PartialLanguageOutput::with_budget(
            vec![part.clone()],
            usage.clone(),
            PartialLanguageOutputBudget::new(1, item_bytes - 1, encoded_bytes),
        ),
        Err(PartialLanguageOutputError::ItemTooLarge { index: 0, .. })
    ));
    assert!(matches!(
        PartialLanguageOutput::with_budget(
            vec![part],
            usage,
            PartialLanguageOutputBudget::new(1, item_bytes, encoded_bytes - 1),
        ),
        Err(PartialLanguageOutputError::TooLarge { .. })
    ));
}

#[test]
fn partial_language_output_cannot_deserialize_executable_content() {
    let partial = PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Refusal {
            reason: Some("not available".to_string()),
        }],
        Usage::default(),
    )
    .unwrap();
    let mut encoded = serde_json::to_value(partial).unwrap();
    encoded["content"] = json!([{"ToolCall": {}}]);

    assert!(serde_json::from_value::<PartialLanguageOutput>(encoded).is_err());
}

#[test]
fn language_call_error_keeps_partial_output_out_of_default_diagnostics() {
    let partial = PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Text {
            text: "private generated text".to_string(),
        }],
        Usage::default(),
    )
    .unwrap();
    let error = LanguageCallError::new(
        SiumaiError::new(crate::ErrorKind::Provider, "generation failed"),
        Some(partial),
    );

    assert_eq!(error.error().kind(), crate::ErrorKind::Provider);
    assert!(error.partial().is_some());
    assert!(
        !format!("{:?}", error.partial().unwrap().content()[0]).contains("private generated text")
    );
    assert!(!format!("{:?}", error.partial().unwrap()).contains("private generated text"));
    assert!(!format!("{error:?}").contains("private generated text"));
    assert!(!error.to_string().contains("private generated text"));
}

#[test]
fn language_response_applies_an_explicit_opaque_collection_budget() {
    let response = LanguageResponse::completed(
        vec![
            ContentPart::ProviderOpaque(
                OpaqueProviderItem::new(provenance(), "program", json!({"wire": 1})).unwrap(),
            ),
            ContentPart::ProviderOpaque(
                OpaqueProviderItem::new(provenance(), "program_output", json!({"wire": 2}))
                    .unwrap(),
            ),
        ],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .unwrap();

    assert!(matches!(
        response.validate_with_opaque_budget(OpaqueProviderBudget::new(
            1,
            DEFAULT_OPAQUE_ITEM_LIMIT,
            DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
        )),
        Err(LanguageResponseError::OpaqueProviderItems(
            OpaqueProviderItemError::TooManyItems { .. }
        ))
    ));
}

#[test]
fn common_generation_controls_preserve_omission_and_validate_ranges() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::Developer, "rules")]);
    assert_eq!(request.generation.temperature, None);
    assert_eq!(request.tool_choice, None);

    let invalid = GenerationConfig {
        top_p: Some(1.5),
        ..GenerationConfig::default()
    };
    assert_eq!(invalid.validate(), Err(GenerationConfigError::InvalidTopP));
}

#[test]
fn language_request_validation_rejects_unknown_named_tool() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
        .with_tool_choice(ToolChoice::Named {
            name: "lookup".to_string(),
        });
    assert_eq!(
        request.validate(),
        Err(LanguageRequestError::UnknownToolChoice {
            name: "lookup".to_string()
        })
    );
}

#[test]
fn role_safe_message_constructors_follow_the_canonical_matrix() {
    assert_eq!(Message::system("rules").role(), MessageRole::System);
    assert_eq!(Message::developer("policy").role(), MessageRole::Developer);
    assert_eq!(Message::user("question").role(), MessageRole::User);
    assert_eq!(Message::assistant("answer").role(), MessageRole::Assistant);

    let media = MediaPart {
        media_type: "image/png".to_string(),
        data: MediaData::Bytes(Bytes::from_static(b"image")),
        name: None,
    };
    Message::user_parts([ContentPart::Media(media)]).expect("input media is valid user content");
    Message::assistant_parts([ContentPart::Reasoning {
        text: "bounded reasoning".to_string(),
    }])
    .expect("reasoning is valid assistant content");
    Message::tool_result(ToolResult {
        call_id: "call-1".to_string(),
        name: "lookup".to_string(),
        outcome: ToolOutcome::Success { value: json!(1) },
    })
    .validate()
    .expect("tool results are valid tool content");
}

#[test]
fn media_debug_redacts_bytes_and_urls() {
    let bytes_sentinel = b"media-bytes-debug-sentinel";
    let url_sentinel = "https://example.com/private/media-url-debug-sentinel";

    let bytes_debug = format!(
        "{:?}",
        MediaData::Bytes(Bytes::copy_from_slice(bytes_sentinel))
    );
    let url_debug = format!("{:?}", MediaData::Url(url_sentinel.to_string()));

    assert!(!bytes_debug.contains("media-bytes-debug-sentinel"));
    assert!(bytes_debug.contains(&bytes_sentinel.len().to_string()));
    assert!(!url_debug.contains(url_sentinel));
    assert!(url_debug.contains("<redacted>"));
}

#[test]
fn request_validation_rejects_invalid_role_content_before_encoding() {
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [ContentPart::ToolResult(ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!(1) },
        })],
    )]);

    assert!(matches!(
        request.validate(),
        Err(LanguageRequestError::InvalidMessage {
            message_index: 0,
            source: MessageValidationError::ContentNotAllowed {
                role: MessageRole::User,
                content_index: 0,
                content_kind: "tool result",
            },
        })
    ));

    let response_only = LanguageRequest::new(vec![Message::new(
        MessageRole::Assistant,
        [ContentPart::Refusal {
            reason: Some("declined".to_string()),
        }],
    )]);
    assert!(matches!(
        response_only.validate(),
        Err(LanguageRequestError::InvalidMessage { .. })
    ));
}

#[test]
fn assistant_history_projection_is_explicit_and_role_safe() {
    let response = LanguageResponse::completed(
        vec![
            ContentPart::Text {
                text: "answer".to_string(),
            },
            ContentPart::Citation(Citation {
                source_id: "source-1".to_string(),
                title: None,
                url: None,
                start: None,
                end: None,
                provider: BTreeMap::new(),
            }),
            ContentPart::Refusal {
                reason: Some("detail omitted".to_string()),
            },
            ContentPart::ProviderOpaque(
                OpaqueProviderItem::new(provenance(), "response.output", json!({"id": "item-1"}))
                    .unwrap(),
            ),
        ],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .unwrap();

    let projection = response.project_assistant_history();
    assert_eq!(
        projection
            .omissions()
            .iter()
            .map(|omission| omission.kind())
            .collect::<Vec<_>>(),
        vec![
            AssistantHistoryOmissionKind::Citation,
            AssistantHistoryOmissionKind::Refusal,
        ]
    );
    let message = projection.message().expect("portable history remains");
    assert_eq!(message.content().len(), 2);
    message.validate().expect("projected history is role safe");
}
