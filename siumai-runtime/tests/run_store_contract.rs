use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::to_string;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentPart, Error, ErrorKind, LanguageCallError, LanguageCompletionReason,
    LanguageModel, LanguageRequest, LanguageResponse, LanguageStream, LanguageStreamEvent, Message,
    MessageRole, Model, ModelDescriptor, ModelFamily, ModelId, ProviderId, StreamTerminal, Usage,
};
use siumai_runtime::DurableToolLoop;
use siumai_runtime::snapshot::{
    InMemoryRunStore, LineageId, RunId, RunLease, RunSnapshot, RunStore, RunStoreError,
    RunStoreFuture, SnapshotRevision, StoredRun,
};
use siumai_runtime::tool::ToolSet;

static NEXT_PROBE_STORE_ID: AtomicU64 = AtomicU64::new(1);

struct FixtureModel {
    descriptor: ModelDescriptor,
}

impl FixtureModel {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new("run-store-fixture-provider").expect("valid provider"),
                ModelId::new("run-store-fixture-model").expect("valid model"),
                ModelFamily::Language,
            ),
        })
    }
}

impl Model for FixtureModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for FixtureModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Internal,
            "run-store fixture uses the streaming path",
        )
        .into())
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let response = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "fixture response".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .expect("fixture response is valid");
        Ok(established_stream(
            options.cancellation().clone(),
            move |_| {
                futures::stream::iter([Ok(LanguageStreamEvent::Terminal(
                    StreamTerminal::Completed {
                        response: Box::new(response),
                    },
                ))])
            },
        ))
    }
}

async fn fixture_snapshot() -> RunSnapshot {
    let run_id = RunId::new("run-store-fixture-run").expect("valid run id");
    let loop_ = DurableToolLoop::new(
        FixtureModel::new(),
        ToolSet::default(),
        Arc::new(InMemoryRunStore::new()),
        siumai_runtime::snapshot::SnapshotFingerprint::new("options").unwrap(),
        siumai_runtime::snapshot::SnapshotFingerprint::new("approval").unwrap(),
    )
    .expect("fixture loop is valid");
    loop_
        .start(
            run_id,
            LineageId::new("run-store-fixture-lineage").expect("valid lineage"),
            LanguageRequest::new(vec![Message::text(MessageRole::User, "fixture prompt")]),
            CallOptions::default(),
        )
        .await
        .expect("fixture run completes")
        .into_snapshot()
}

#[derive(Clone)]
struct ProbeToken {
    store_id: u64,
    token: u64,
    run_id: RunId,
    secret: String,
}

struct ProbeLeaseRecord {
    token: u64,
    expires_at: Instant,
}

struct ProbeState {
    next_token: u64,
    lease: Option<ProbeLeaseRecord>,
    stored: Option<StoredRun>,
}

struct ProbeStore {
    store_id: u64,
    state: Mutex<ProbeState>,
}

impl ProbeStore {
    fn new() -> Self {
        Self {
            store_id: NEXT_PROBE_STORE_ID.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(ProbeState {
                next_token: 1,
                lease: None,
                stored: None,
            }),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, ProbeState>, RunStoreError> {
        self.state.lock().map_err(|_| RunStoreError::Unavailable)
    }

    fn validate_lease(
        &self,
        state: &mut ProbeState,
        lease: &RunLease,
    ) -> Result<(), RunStoreError> {
        let token = lease
            .store_token::<ProbeToken>()
            .ok_or(RunStoreError::ForeignLease)?;
        if token.store_id != self.store_id || token.run_id != *lease.run_id() {
            return Err(RunStoreError::ForeignLease);
        }
        let Some(record) = state.lease.as_ref() else {
            return Err(RunStoreError::LeaseLost);
        };
        if record.token != token.token {
            return Err(RunStoreError::LeaseLost);
        }
        if Instant::now() >= record.expires_at {
            state.lease = None;
            return Err(RunStoreError::LeaseExpired);
        }
        Ok(())
    }
}

impl RunStore for ProbeStore {
    fn acquire<'a>(&'a self, run_id: &'a RunId, ttl: Duration) -> RunStoreFuture<'a, RunLease> {
        Box::pin(async move {
            let now = Instant::now();
            let expires_at = now
                .checked_add(ttl)
                .filter(|_| !ttl.is_zero())
                .ok_or(RunStoreError::InvalidLeaseDuration)?;
            let mut state = self.lock()?;
            if state
                .lease
                .as_ref()
                .is_some_and(|lease| now < lease.expires_at)
            {
                return Err(RunStoreError::LeaseConflict {
                    run_id: run_id.clone(),
                });
            }
            state.lease = None;
            let token = state.next_token;
            state.next_token = state
                .next_token
                .checked_add(1)
                .ok_or(RunStoreError::LeaseTokenExhausted)?;
            state.lease = Some(ProbeLeaseRecord { token, expires_at });
            Ok(RunLease::from_store_token(
                run_id.clone(),
                expires_at,
                ProbeToken {
                    store_id: self.store_id,
                    token,
                    run_id: run_id.clone(),
                    secret: "private-token-secret".to_string(),
                },
            ))
        })
    }

    fn load<'a>(&'a self, lease: &'a RunLease) -> RunStoreFuture<'a, Option<StoredRun>> {
        Box::pin(async move {
            let mut state = self.lock()?;
            self.validate_lease(&mut state, lease)?;
            Ok(state.stored.clone())
        })
    }

    fn compare_and_swap<'a>(
        &'a self,
        lease: &'a RunLease,
        expected: SnapshotRevision,
        snapshot: RunSnapshot,
    ) -> RunStoreFuture<'a, SnapshotRevision> {
        Box::pin(async move {
            if snapshot.run_id() != lease.run_id() {
                return Err(RunStoreError::RunIdMismatch {
                    leased_run_id: lease.run_id().clone(),
                    snapshot_run_id: snapshot.run_id().clone(),
                });
            }
            let mut state = self.lock()?;
            self.validate_lease(&mut state, lease)?;
            let actual = state
                .stored
                .as_ref()
                .map_or(SnapshotRevision::EMPTY, StoredRun::revision);
            if actual != expected {
                return Err(RunStoreError::CasConflict { expected, actual });
            }
            let revision = SnapshotRevision::from_value(
                actual
                    .value()
                    .checked_add(1)
                    .ok_or(RunStoreError::RevisionExhausted)?,
            );
            state.stored = Some(StoredRun::new(revision, snapshot));
            Ok(revision)
        })
    }

    fn renew<'a>(&'a self, lease: &'a mut RunLease, ttl: Duration) -> RunStoreFuture<'a, ()> {
        Box::pin(async move {
            let expires_at = Instant::now()
                .checked_add(ttl)
                .filter(|_| !ttl.is_zero())
                .ok_or(RunStoreError::InvalidLeaseDuration)?;
            let mut state = self.lock()?;
            self.validate_lease(&mut state, lease)?;
            state
                .lease
                .as_mut()
                .ok_or(RunStoreError::LeaseLost)?
                .expires_at = expires_at;
            lease.set_expires_at(expires_at);
            Ok(())
        })
    }

    fn release<'a>(&'a self, lease: RunLease) -> RunStoreFuture<'a, ()> {
        Box::pin(async move {
            let mut state = self.lock()?;
            self.validate_lease(&mut state, &lease)?;
            state.lease = None;
            Ok(())
        })
    }
}

struct ForeignToken;

#[tokio::test]
async fn external_store_lease_contract_is_object_safe_and_private() {
    let snapshot = fixture_snapshot().await;
    let run_id = snapshot.run_id().clone();
    let store_impl = Arc::new(ProbeStore::new());
    let store: Arc<dyn RunStore> = store_impl.clone();
    let other_store: Arc<dyn RunStore> = Arc::new(ProbeStore::new());

    let mut lease = store
        .acquire(&run_id, Duration::from_secs(30))
        .await
        .expect("acquire succeeds through dyn RunStore");
    assert_eq!(
        lease
            .store_token::<ProbeToken>()
            .expect("issuing store token type is available")
            .secret,
        "private-token-secret"
    );
    assert!(store.load(&lease).await.unwrap().is_none());

    let revision = store
        .compare_and_swap(&lease, SnapshotRevision::EMPTY, snapshot.clone())
        .await
        .expect("initial CAS succeeds");
    assert_eq!(revision, SnapshotRevision::from_value(1));
    let loaded = store
        .load(&lease)
        .await
        .expect("load succeeds")
        .expect("snapshot is present");
    assert_eq!(loaded.revision(), revision);
    assert_eq!(loaded.snapshot().run_id(), &run_id);

    let before_renewal = lease.expires_at();
    store
        .renew(&mut lease, Duration::from_secs(60))
        .await
        .expect("renew succeeds");
    assert!(lease.expires_at() > before_renewal);

    let successor = store
        .compare_and_swap(&lease, revision, snapshot.clone())
        .await
        .expect("successor CAS succeeds");
    assert_eq!(successor, SnapshotRevision::from_value(2));

    assert_eq!(
        other_store.load(&lease).await.unwrap_err(),
        RunStoreError::ForeignLease
    );
    let foreign_type = RunLease::from_store_token(run_id.clone(), lease.expires_at(), ForeignToken);
    assert_eq!(
        store.load(&foreign_type).await.unwrap_err(),
        RunStoreError::ForeignLease
    );
    let private_token = lease
        .store_token::<ProbeToken>()
        .expect("the issuing store can inspect its private token")
        .clone();
    let foreign_run = RunLease::from_store_token(
        RunId::new("another-run").expect("valid run id"),
        lease.expires_at(),
        private_token,
    );
    assert_eq!(
        store.load(&foreign_run).await.unwrap_err(),
        RunStoreError::ForeignLease
    );
    assert_eq!(
        store.load(&lease).await.unwrap().unwrap().revision(),
        successor
    );

    let debug = format!("{lease:?}");
    assert!(!debug.contains("private-token-secret"));
    let encoded = to_string(&snapshot).expect("snapshot serializes");
    assert!(!encoded.contains("private-token-secret"));

    store
        .release(lease)
        .await
        .expect("release consumes the lease");
    let reacquired = store
        .acquire(&run_id, Duration::from_secs(30))
        .await
        .expect("release allows a new owner");
    store
        .release(reacquired)
        .await
        .expect("second release succeeds");
}
