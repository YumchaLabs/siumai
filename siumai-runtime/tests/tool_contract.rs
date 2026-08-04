use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Map, Value, json};
use siumai_core::{ExecutionOwner, ProviderId, ToolCall, ToolOutcome, ToolSpec};
use siumai_runtime::tool::{
    ApprovalPolicy, EffectCertainty, RecoveryPolicy, ToolArgumentError, ToolBinding,
    ToolBindingConfigError, ToolConcurrency, ToolEffect, ToolExecutionError, ToolSet,
    ToolSetBuildError, canonical_arguments_digest,
};

fn spec(name: &str, schema: Value) -> ToolSpec {
    ToolSpec::new(name, Some(format!("{name} tool")), schema).expect("valid tool spec")
}

fn successful_binding(name: &str, marker: &'static str) -> ToolBinding {
    successful_binding_with_revision(name, marker, "v1")
}

fn successful_binding_with_revision(
    name: &str,
    marker: &'static str,
    revision: &str,
) -> ToolBinding {
    ToolBinding::from_fn(
        spec(name, json!({ "type": "object" })),
        revision,
        |_| Ok(()),
        move |_| async move {
            Ok(ToolOutcome::Success {
                value: json!({ "binding": marker }),
            })
        },
    )
    .expect("valid binding revision")
}

fn local_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: format!("call_{name}"),
        name: name.to_string(),
        arguments,
        owner: ExecutionOwner::Local,
    }
}

#[test]
fn bindings_default_to_safe_side_effect_contracts() {
    let binding = successful_binding("charge", "default");

    assert_eq!(binding.effect(), ToolEffect::SideEffecting);
    assert_eq!(binding.concurrency(), ToolConcurrency::Sequential);
    assert_eq!(binding.approval_policy(), ApprovalPolicy::Required);
    assert_eq!(binding.recovery_policy(), RecoveryPolicy::NeverReplay);
}

#[test]
fn host_revision_is_required_and_changes_binding_identity() {
    let first = successful_binding_with_revision("charge", "first", "executor-v1");
    let second = successful_binding_with_revision("charge", "second", "executor-v2");

    assert_ne!(first.identity().fingerprint, second.identity().fingerprint);
    assert_eq!(first.revision(), "executor-v1");
    assert_eq!(second.revision(), "executor-v2");

    let invalid = ToolBinding::from_fn(
        spec("invalid_revision", json!({ "type": "object" })),
        "bad\nrevision",
        |_| Ok(()),
        |_| async { Ok(ToolOutcome::Success { value: Value::Null }) },
    )
    .expect_err("control characters must be rejected");
    assert_eq!(invalid, ToolBindingConfigError::InvalidRevision);

    let empty = ToolBinding::from_fn(
        spec("empty_revision", json!({ "type": "object" })),
        "",
        |_| Ok(()),
        |_| async { Ok(ToolOutcome::Success { value: Value::Null }) },
    )
    .expect_err("empty revisions must be rejected");
    assert_eq!(empty, ToolBindingConfigError::InvalidRevision);
}

#[test]
fn tool_set_rejects_duplicate_names_and_sorts_catalogs() {
    let mut builder = ToolSet::builder();
    builder
        .insert(successful_binding("zeta", "z"))
        .expect("first name is unique");
    builder
        .insert(successful_binding("alpha", "a"))
        .expect("second name is unique");
    let duplicate = builder
        .insert(successful_binding("alpha", "other"))
        .expect_err("duplicate name must be rejected");

    assert_eq!(
        duplicate,
        ToolSetBuildError::DuplicateName {
            name: "alpha".to_string()
        }
    );

    let tools = builder.build();
    assert_eq!(
        tools.specs().iter().map(ToolSpec::name).collect::<Vec<_>>(),
        vec!["alpha", "zeta"]
    );
}

#[test]
fn catalog_fingerprint_is_independent_of_insertion_and_json_object_order() {
    let mut schema_left = Map::new();
    schema_left.insert("type".to_string(), json!("object"));
    schema_left.insert(
        "properties".to_string(),
        json!({ "city": { "type": "string" }, "days": { "type": "integer" } }),
    );

    let mut schema_right = Map::new();
    schema_right.insert(
        "properties".to_string(),
        json!({ "days": { "type": "integer" }, "city": { "type": "string" } }),
    );
    schema_right.insert("type".to_string(), json!("object"));

    let left_weather = ToolBinding::from_fn(
        spec("weather", Value::Object(schema_left)),
        "v1",
        |_| Ok(()),
        |_| async { Ok(ToolOutcome::Success { value: Value::Null }) },
    )
    .expect("valid binding revision");
    let right_weather = ToolBinding::from_fn(
        spec("weather", Value::Object(schema_right)),
        "v1",
        |_| Ok(()),
        |_| async { Ok(ToolOutcome::Success { value: Value::Null }) },
    )
    .expect("valid binding revision");
    assert_eq!(
        left_weather.identity().fingerprint,
        right_weather.identity().fingerprint
    );

    let left = ToolSet::from_bindings([left_weather, successful_binding("search", "search")])
        .expect("unique names");
    let right = ToolSet::from_bindings([successful_binding("search", "search"), right_weather])
        .expect("unique names");

    assert_eq!(left.fingerprint(), right.fingerprint());
}

#[test]
fn approval_argument_digest_is_structural_and_order_independent() {
    let left = json!({ "city": "Paris", "units": { "speed": "kmh", "temp": "c" } });
    let right = json!({ "units": { "temp": "c", "speed": "kmh" }, "city": "Paris" });

    assert_eq!(
        canonical_arguments_digest(&left),
        canonical_arguments_digest(&right)
    );
    assert_ne!(
        canonical_arguments_digest(&left),
        canonical_arguments_digest(&json!({ "city": "London" }))
    );
}

#[test]
fn frozen_resolution_rejects_a_replaced_binding() {
    let original = successful_binding_with_revision("lookup", "original", "v1");
    let original_identity = original.identity().clone();
    let replacement = successful_binding_with_revision("lookup", "replacement", "v2");
    let tools = ToolSet::from_bindings([replacement]).expect("unique tool");

    let error = tools
        .resolve_frozen(local_call("lookup", json!({})), &original_identity)
        .expect_err("a changed executor identity must invalidate continuation state");

    assert!(matches!(
        error,
        ToolExecutionError::BindingIdentityMismatch { .. }
    ));
}

#[tokio::test]
async fn provider_owned_calls_never_resolve_to_colliding_local_bindings() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        spec("search", json!({ "type": "object" })),
        "v1",
        |_| Ok(()),
        move |_| {
            let observed = Arc::clone(&observed);
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success { value: Value::Null })
            }
        },
    )
    .expect("valid binding revision");
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let call = ToolCall {
        id: "provider_call".to_string(),
        name: "search".to_string(),
        arguments: json!({}),
        owner: ExecutionOwner::Provider {
            provider: ProviderId::new("openai").expect("valid provider ID"),
        },
    };

    let error = tools
        .resolve(call)
        .expect_err("provider-owned call must not resolve locally");
    assert!(matches!(
        error,
        ToolExecutionError::ProviderOwnedCall { .. }
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn argument_validation_is_always_applied_before_dispatch() {
    let validations = Arc::new(AtomicUsize::new(0));
    let executions = Arc::new(AtomicUsize::new(0));
    let validation_count = Arc::clone(&validations);
    let execution_count = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        spec("weather", json!({ "type": "object" })),
        "v1",
        move |arguments| {
            validation_count.fetch_add(1, Ordering::SeqCst);
            if arguments.get("city").and_then(Value::as_str).is_some() {
                Ok(())
            } else {
                Err(ToolArgumentError::new("city must be a string"))
            }
        },
        move |_| {
            let execution_count = Arc::clone(&execution_count);
            async move {
                execution_count.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success {
                    value: json!("sunny"),
                })
            }
        },
    )
    .expect("valid binding revision");
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");

    let invalid = tools
        .resolve(local_call("weather", json!({ "city": 7 })))
        .expect("local tool exists")
        .execute()
        .await
        .expect_err("invalid arguments must stop dispatch");
    assert!(matches!(
        invalid,
        ToolExecutionError::InvalidArguments { .. }
    ));
    assert_eq!(validations.load(Ordering::SeqCst), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);

    let request = tools
        .resolve(local_call("weather", json!({ "city": "Paris" })))
        .expect("local tool exists");
    request
        .validate()
        .expect("arguments validate before Prepared or approval");
    assert_eq!(validations.load(Ordering::SeqCst), 2);
    assert_eq!(executions.load(Ordering::SeqCst), 0);

    let result = request.execute().await.expect("valid request executes");
    assert!(matches!(result.outcome, ToolOutcome::Success { .. }));
    assert_eq!(validations.load(Ordering::SeqCst), 3);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn resolved_request_keeps_the_exact_frozen_binding() {
    let original = successful_binding("lookup", "original");
    let original_handle = original.clone();
    let first_set = ToolSet::from_bindings([original]).expect("unique tool");
    let request = first_set
        .resolve(local_call("lookup", json!({})))
        .expect("local tool exists");

    let replacement = successful_binding_with_revision("lookup", "replacement", "v2");
    let second_set = ToolSet::from_bindings([replacement]).expect("unique tool");
    assert!(!original_handle.same_instance(second_set.get("lookup").expect("replacement exists")));

    let result = request.execute().await.expect("frozen request executes");
    assert_eq!(
        result.outcome,
        ToolOutcome::Success {
            value: json!({ "binding": "original" })
        }
    );
}

#[tokio::test]
async fn indeterminate_executor_failure_remains_an_error() {
    let binding = ToolBinding::from_fn(
        spec("charge", json!({ "type": "object" })),
        "v1",
        |_| Ok(()),
        |request| async move {
            Err(ToolExecutionError::executor_failed(
                request.name(),
                "connection lost after dispatch",
                true,
                EffectCertainty::Indeterminate,
            ))
        },
    )
    .expect("valid binding revision");
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");

    let error = tools
        .resolve(local_call("charge", json!({})))
        .expect("local tool exists")
        .execute()
        .await
        .expect_err("uncertain effects must not become tool outcomes");

    assert_eq!(error.effect_certainty(), EffectCertainty::Indeterminate);
    assert!(matches!(
        error,
        ToolExecutionError::ExecutorFailed {
            certainty: EffectCertainty::Indeterminate,
            ..
        }
    ));
}

#[test]
fn concurrency_and_recovery_require_explicit_safe_declarations() {
    let parallel = ToolConcurrency::SafeParallel {
        max_in_flight: NonZeroUsize::new(4).expect("non-zero"),
    };
    let binding = successful_binding("lookup", "parallel")
        .with_effect(ToolEffect::ReadOnly)
        .with_concurrency(parallel)
        .with_approval_policy(ApprovalPolicy::NotRequired)
        .with_recovery_policy(RecoveryPolicy::ReplayWithStableIdempotencyKey);

    assert_eq!(binding.concurrency(), parallel);
    assert!(
        binding
            .recovery_policy()
            .permits_retry(EffectCertainty::Indeterminate, true)
    );
    assert!(
        !binding
            .recovery_policy()
            .permits_retry(EffectCertainty::Indeterminate, false)
    );
    assert!(
        !binding
            .recovery_policy()
            .permits_retry(EffectCertainty::Applied, true)
    );
}
