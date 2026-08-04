use std::sync::{Arc, Barrier};

use serde::{Deserialize, Serialize};
use siumai_core::{
    ExecutionOwner, Model, ModelDescriptor, ModelFamily, ModelId, ProviderId, RouteId,
    ToolBindingIdentity,
};

use super::*;
use crate::options::ModelTarget;

const NOW: u64 = 1_000;
const EXPIRY: u64 = 2_000;

#[derive(Debug)]
struct TestModel {
    descriptor: ModelDescriptor,
    route: RouteId,
}

impl Model for TestModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn route_id(&self) -> Option<&RouteId> {
        Some(&self.route)
    }
}

fn model_target(route: &str) -> ModelTarget {
    let model = TestModel {
        descriptor: ModelDescriptor::new(
            ProviderId::new("openai").expect("valid provider"),
            ModelId::new("gpt-test").expect("valid model"),
            ModelFamily::Language,
        ),
        route: RouteId::new(route).expect("valid route"),
    };
    ModelTarget::from_model(&model)
}

fn trust_context(tenant: &str, route: &str, arguments_digest: &str) -> TrustContext {
    trust_context_with_state(
        tenant,
        route,
        arguments_digest,
        "checkpoint-3",
        "sha256:binding-v4",
    )
}

fn trust_context_with_state(
    tenant: &str,
    route: &str,
    arguments_digest: &str,
    checkpoint: &str,
    binding_fingerprint: &str,
) -> TrustContext {
    TrustContext::builder()
        .issuer("siumai-host")
        .audience("payments-runtime")
        .subject("user-42")
        .tenant(tenant)
        .model_target(model_target(route))
        .run_lineage("run-7")
        .checkpoint(checkpoint)
        .execution_owner(ExecutionOwner::Local)
        .binding_identity(ToolBindingIdentity {
            name: "charge-card".to_owned(),
            fingerprint: binding_fingerprint.to_owned(),
        })
        .tool_call_id("call-9")
        .canonical_arguments_digest(arguments_digest)
        .catalog_fingerprint("sha256:catalog-v2")
        .policy_fingerprint("sha256:policy-v8")
        .build()
        .expect("valid trust context")
}

#[derive(Debug, Serialize, Deserialize)]
struct TestWireEnvelope {
    payload: Vec<u8>,
    signature: u64,
}

struct TestCrypto {
    key_id: String,
    secret: Vec<u8>,
}

impl TestCrypto {
    fn new() -> Self {
        Self {
            key_id: "test-key-1".to_owned(),
            secret: b"test-only-secret".to_vec(),
        }
    }
}

impl ApprovalSigner for TestCrypto {
    fn sign(&self, claims: &ApprovalClaims) -> Result<ApprovalEnvelope, ApprovalSigningError> {
        if claims.key_id() != self.key_id {
            return Err(ApprovalSigningError::Rejected);
        }
        let payload = serde_json::to_vec(claims).map_err(|_| ApprovalSigningError::Rejected)?;
        let signature = test_mac(&self.secret, &payload);
        let encoded = serde_json::to_vec(&TestWireEnvelope { payload, signature })
            .map_err(|_| ApprovalSigningError::Unavailable)?;
        ApprovalEnvelope::from_bytes(encoded).map_err(|_| ApprovalSigningError::Rejected)
    }
}

impl ApprovalVerifier for TestCrypto {
    fn verify(&self, envelope: &ApprovalEnvelope) -> Result<ApprovalClaims, ApprovalVerifierError> {
        let wire: TestWireEnvelope = serde_json::from_slice(envelope.as_bytes())
            .map_err(|_| ApprovalVerifierError::Rejected)?;
        if test_mac(&self.secret, &wire.payload) != wire.signature {
            return Err(ApprovalVerifierError::Rejected);
        }
        let claims: ApprovalClaims =
            serde_json::from_slice(&wire.payload).map_err(|_| ApprovalVerifierError::Rejected)?;
        if claims.key_id() != self.key_id {
            return Err(ApprovalVerifierError::Rejected);
        }
        Ok(claims)
    }
}

fn test_mac(secret: &[u8], payload: &[u8]) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in secret.iter().chain(payload) {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

fn signed_approval(crypto: &TestCrypto, context: &TrustContext) -> ApprovalEnvelope {
    let claims = ApprovalClaims::issue(context, EXPIRY, "nonce-unique-1", "test-key-1")
        .expect("valid claims");
    crypto.sign(&claims).expect("test signing succeeds")
}

#[test]
fn trust_context_debug_is_redacted() {
    let context = trust_context("secret-tenant", "production", "sha256:secret-arguments");
    let debug = format!("{context:?}");
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("secret-tenant"));
    assert!(!debug.contains("secret-arguments"));
}

#[test]
fn rejects_cross_tenant_and_cross_route_use_without_consuming() {
    let crypto = TestCrypto::new();
    let approved = trust_context("tenant-a", "production", "sha256:args-a");
    let envelope = signed_approval(&crypto, &approved);
    let store = InMemoryApprovalConsumeStore::default();

    let wrong_tenant = trust_context("tenant-b", "production", "sha256:args-a");
    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &wrong_tenant, NOW),
        Err(ApprovalVerificationError::ContextMismatch {
            field: ApprovalClaimField::Tenant,
        })
    ));

    let wrong_route = trust_context("tenant-a", "staging", "sha256:args-a");
    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &wrong_route, NOW),
        Err(ApprovalVerificationError::ContextMismatch {
            field: ApprovalClaimField::Route,
        })
    ));

    assert!(verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &approved, NOW).is_ok());
}

#[test]
fn rejects_changed_canonical_arguments() {
    let crypto = TestCrypto::new();
    let approved = trust_context("tenant-a", "production", "sha256:args-a");
    let changed = trust_context("tenant-a", "production", "sha256:args-b");
    let envelope = signed_approval(&crypto, &approved);
    let store = InMemoryApprovalConsumeStore::default();

    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &changed, NOW),
        Err(ApprovalVerificationError::ContextMismatch {
            field: ApprovalClaimField::CanonicalArgumentsDigest,
        })
    ));
}

#[test]
fn rejects_changed_checkpoint_and_frozen_binding() {
    let crypto = TestCrypto::new();
    let approved = trust_context("tenant-a", "production", "sha256:args-a");
    let envelope = signed_approval(&crypto, &approved);
    let store = InMemoryApprovalConsumeStore::default();

    let changed_checkpoint = trust_context_with_state(
        "tenant-a",
        "production",
        "sha256:args-a",
        "checkpoint-4",
        "sha256:binding-v4",
    );
    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &changed_checkpoint, NOW,),
        Err(ApprovalVerificationError::ContextMismatch {
            field: ApprovalClaimField::Checkpoint,
        })
    ));

    let changed_binding = trust_context_with_state(
        "tenant-a",
        "production",
        "sha256:args-a",
        "checkpoint-3",
        "sha256:binding-v5",
    );
    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &changed_binding, NOW),
        Err(ApprovalVerificationError::ContextMismatch {
            field: ApprovalClaimField::BindingIdentity,
        })
    ));

    assert!(verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &approved, NOW).is_ok());
}

#[test]
fn rejects_expired_approval_at_the_expiry_boundary() {
    let crypto = TestCrypto::new();
    let context = trust_context("tenant-a", "production", "sha256:args-a");
    let envelope = signed_approval(&crypto, &context);
    let store = InMemoryApprovalConsumeStore::default();

    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &envelope, &context, EXPIRY),
        Err(ApprovalVerificationError::Expired)
    ));
}

#[test]
fn maps_tampering_to_one_authenticity_failure() {
    let crypto = TestCrypto::new();
    let context = trust_context("tenant-a", "production", "sha256:args-a");
    let envelope = signed_approval(&crypto, &context);
    let mut wire: TestWireEnvelope =
        serde_json::from_slice(envelope.as_bytes()).expect("valid test wire");
    wire.payload[0] ^= 1;
    let tampered = ApprovalEnvelope::from_bytes(
        serde_json::to_vec(&wire).expect("tampered test envelope serializes"),
    )
    .expect("bounded envelope");
    let store = InMemoryApprovalConsumeStore::default();

    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &store, &tampered, &context, NOW),
        Err(ApprovalVerificationError::AuthenticityRejected)
    ));
}

#[test]
fn concurrent_replay_has_exactly_one_winner() {
    const CALLERS: usize = 16;

    let crypto = Arc::new(TestCrypto::new());
    let context = Arc::new(trust_context("tenant-a", "production", "sha256:args-a"));
    let envelope = Arc::new(signed_approval(&crypto, &context));
    let store = Arc::new(InMemoryApprovalConsumeStore::default());
    let barrier = Arc::new(Barrier::new(CALLERS));

    let results = std::thread::scope(|scope| {
        let handles = (0..CALLERS)
            .map(|_| {
                let crypto = Arc::clone(&crypto);
                let context = Arc::clone(&context);
                let envelope = Arc::clone(&envelope);
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    verify_and_consume_at_unix_ms(
                        crypto.as_ref(),
                        store.as_ref(),
                        &envelope,
                        &context,
                        NOW,
                    )
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("verification thread must not panic"))
            .collect::<Vec<_>>()
    });

    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(ApprovalVerificationError::AlreadyConsumed)))
            .count(),
        CALLERS - 1
    );
}

#[test]
fn consume_store_failure_is_fail_closed() {
    struct UnavailableStore;

    impl ApprovalConsumeStore for UnavailableStore {
        fn consume_once(&self, _key: &ApprovalConsumeKey) -> Result<(), ApprovalConsumeError> {
            Err(ApprovalConsumeError::Unavailable)
        }
    }

    let crypto = TestCrypto::new();
    let context = trust_context("tenant-a", "production", "sha256:args-a");
    let envelope = signed_approval(&crypto, &context);

    assert!(matches!(
        verify_and_consume_at_unix_ms(&crypto, &UnavailableStore, &envelope, &context, NOW),
        Err(ApprovalVerificationError::ConsumeStoreUnavailable)
    ));
}
