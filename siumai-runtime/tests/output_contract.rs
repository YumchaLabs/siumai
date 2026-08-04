use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use siumai_core::{
    CallOptions, Cancellation, ContentPart, FinishReason, LanguageIncompleteReason,
    LanguageRequest, LanguageResponse, LanguageResponseStatus, Message, MessageRole, ToolChoice,
    ToolSpec, Usage,
};
use siumai_runtime::{
    OutputDescriptor, OutputSchemaValidator, RepairPolicy, SchemaValidationError,
    StructuredOutputAttemptKind, StructuredOutputFailureKind,
};

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Person {
    name: String,
    age: u8,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct PersonWithCountry {
    name: String,
    age: u8,
    country: String,
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "required": ["name", "age"],
        "additionalProperties": false,
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 255}
        }
    })
}

fn person_validator() -> impl OutputSchemaValidator {
    |_schema: &Value, instance: &Value| {
        let object = instance
            .as_object()
            .ok_or_else(|| SchemaValidationError::new("expected an object"))?;
        if !object.get("name").is_some_and(Value::is_string) {
            return Err(SchemaValidationError::new("name must be a string"));
        }
        if !object
            .get("age")
            .and_then(Value::as_u64)
            .is_some_and(|age| age <= u8::MAX as u64)
        {
            return Err(SchemaValidationError::new("age must be an unsigned byte"));
        }
        Ok(())
    }
}

fn descriptor() -> OutputDescriptor<Person> {
    OutputDescriptor::new("person", person_schema(), person_validator()).unwrap()
}

fn text_response(text: impl Into<String>) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text { text: text.into() }],
        FinishReason::Stop,
        Usage::default(),
    )
    .unwrap()
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "Describe Ada")])
}

#[test]
fn valid_output_is_strictly_parsed_schema_validated_and_typed() {
    let descriptor = descriptor();
    let shaped = descriptor.shape_request(request());

    assert_eq!(shaped.structured_output.as_ref(), Some(descriptor.spec()));

    let result = descriptor
        .consume_response(text_response(r#"{"name":"Ada","age":36}"#))
        .unwrap();

    assert_eq!(
        result.value(),
        &Person {
            name: "Ada".to_string(),
            age: 36,
        }
    );
    assert_eq!(result.json(), &json!({"name": "Ada", "age": 36}));
    assert_eq!(result.attempt(), StructuredOutputAttemptKind::Initial);
    assert!(!result.was_repaired());
}

#[test]
fn refusal_is_non_repairable_even_if_it_contains_text() {
    let response = LanguageResponse::completed(
        vec![
            ContentPart::Refusal {
                reason: Some("unsafe request".to_string()),
            },
            ContentPart::Text {
                text: r#"{"name":"Ada","age":36}"#.to_string(),
            },
        ],
        FinishReason::Refusal,
        Usage::default(),
    )
    .unwrap();

    let error = descriptor().consume_response(response).unwrap_err();

    assert_eq!(error.kind(), StructuredOutputFailureKind::Refusal);
    assert!(!error.is_repair_eligible());
    assert!(error.response().is_some());
}

#[test]
fn content_filter_is_non_repairable() {
    let response = LanguageResponse::new(
        LanguageResponseStatus::Incomplete {
            reason: Some(LanguageIncompleteReason::ContentFilter),
        },
        Vec::new(),
        FinishReason::ContentFilter,
        Usage::default(),
    )
    .unwrap();

    let error = descriptor().consume_response(response).unwrap_err();

    assert_eq!(error.kind(), StructuredOutputFailureKind::ContentFilter);
    assert!(!error.is_repair_eligible());
}

#[test]
fn missing_output_is_non_repairable() {
    let response = LanguageResponse::completed(
        vec![ContentPart::Text {
            text: "   \n".to_string(),
        }],
        FinishReason::Stop,
        Usage::default(),
    )
    .unwrap();

    let error = descriptor().consume_response(response).unwrap_err();

    assert_eq!(error.kind(), StructuredOutputFailureKind::MissingOutput);
    assert!(!error.is_repair_eligible());
}

#[test]
fn provider_failure_is_non_repairable_and_retains_usage_response() {
    let response = LanguageResponse::new(
        LanguageResponseStatus::Failed,
        Vec::new(),
        FinishReason::Error,
        Usage::default(),
    )
    .unwrap();

    let error = descriptor().consume_response(response).unwrap_err();

    assert_eq!(error.kind(), StructuredOutputFailureKind::ProviderFailure);
    assert!(!error.is_repair_eligible());
    assert!(error.response().is_some());
}

#[test]
fn transport_failure_is_non_repairable() {
    let error = siumai_runtime::StructuredOutputError::from_transport(
        siumai_core::Error::new(siumai_core::ErrorKind::Transport, "connection failed"),
        StructuredOutputAttemptKind::Initial,
    );

    assert_eq!(error.kind(), StructuredOutputFailureKind::TransportFailure);
    assert!(!error.is_repair_eligible());
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn invalid_json_is_repairable_only_on_the_initial_attempt() {
    let descriptor = descriptor();
    let initial = descriptor
        .consume_response(text_response("```json\n{}\n```"))
        .unwrap_err();

    assert_eq!(initial.kind(), StructuredOutputFailureKind::InvalidJson);
    assert!(initial.is_repair_eligible());
    assert_eq!(initial.raw_output(), Some("```json\n{}\n```"));

    let repair = descriptor
        .consume_repair_response(text_response("{"))
        .unwrap_err();
    assert_eq!(repair.kind(), StructuredOutputFailureKind::InvalidJson);
    assert_eq!(repair.attempt(), StructuredOutputAttemptKind::Repair);
    assert!(!repair.is_repair_eligible());
}

#[test]
fn schema_mismatch_retains_parsed_json_and_is_repairable() {
    let error = descriptor()
        .consume_response(text_response(r#"{"name":"Ada","age":"old"}"#))
        .unwrap_err();

    assert_eq!(error.kind(), StructuredOutputFailureKind::SchemaMismatch);
    assert_eq!(
        error.parsed_value(),
        Some(&json!({"name": "Ada", "age": "old"}))
    );
    assert!(error.is_repair_eligible());
}

#[test]
fn schema_valid_but_typed_decode_mismatch_is_not_repairable() {
    let descriptor =
        OutputDescriptor::<PersonWithCountry>::new("person", person_schema(), person_validator())
            .unwrap();

    let error = descriptor
        .consume_response(text_response(r#"{"name":"Ada","age":36}"#))
        .unwrap_err();

    assert_eq!(
        error.kind(),
        StructuredOutputFailureKind::TypedDecodeMismatch
    );
    assert!(!error.is_repair_eligible());
}

#[test]
fn repair_is_disabled_by_default() {
    let descriptor = descriptor();
    let failure = descriptor
        .consume_response(text_response("not-json"))
        .unwrap_err();

    assert_eq!(descriptor.repair_policy(), RepairPolicy::Disabled);
    assert_eq!(descriptor.repair_policy().max_additional_steps(), 0);

    let returned = descriptor
        .plan_repair(request(), failure, &CallOptions::default())
        .unwrap_err();
    assert_eq!(returned.kind(), StructuredOutputFailureKind::InvalidJson);
}

#[test]
fn enabled_repair_is_tool_free_and_inherits_deadline_and_cancellation() {
    let descriptor = descriptor().with_repair_policy(RepairPolicy::OneAttempt);
    let failure = descriptor
        .consume_response(text_response("not-json"))
        .unwrap_err();
    let mut original = request();
    original.tools.push(
        ToolSpec::new(
            "lookup",
            Some("Look up a record".to_string()),
            json!({"type": "object"}),
        )
        .unwrap(),
    );
    original.tool_choice = Some(ToolChoice::Required);

    let deadline = Instant::now() + Duration::from_secs(30);
    let cancellation = Cancellation::new();
    let options = CallOptions::default()
        .with_deadline(deadline)
        .with_cancellation(cancellation.clone());

    let repair = descriptor.plan_repair(original, failure, &options).unwrap();

    assert_eq!(descriptor.repair_policy().max_additional_steps(), 1);
    assert!(repair.request().tools.is_empty());
    assert!(repair.request().tool_choice.is_none());
    assert_eq!(
        repair.request().structured_output.as_ref(),
        Some(descriptor.spec())
    );
    assert_eq!(repair.call_options().deadline(), Some(deadline));
    assert_eq!(
        repair.initial_failure().kind(),
        StructuredOutputFailureKind::InvalidJson
    );
    assert!(matches!(
        repair.request().messages.as_slice(),
        [
            Message {
                role: MessageRole::User,
                ..
            },
            Message {
                role: MessageRole::Assistant,
                ..
            },
            Message {
                role: MessageRole::Developer,
                ..
            }
        ]
    ));

    cancellation.cancel();
    assert!(repair.call_options().cancellation().is_cancelled());
}

#[test]
fn partial_output_is_explicitly_unvalidated() {
    let partial = descriptor().unvalidated_partial(json!({"name": "Ada"}));

    assert_eq!(partial.value(), &json!({"name": "Ada"}));
}
