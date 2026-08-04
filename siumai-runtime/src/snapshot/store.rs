use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{RunId, RunSnapshot, RunSnapshotError};

static NEXT_STORE_ID: AtomicU64 = AtomicU64::new(1);

/// Optimistic-concurrency version of one stored run.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SnapshotRevision(u64);

impl SnapshotRevision {
    /// Expected revision when creating a run that is not stored yet.
    pub const EMPTY: Self = Self(0);

    pub const fn value(self) -> u64 {
        self.0
    }

    fn next(self) -> Result<Self, RunStoreError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(RunStoreError::RevisionExhausted)
    }
}

impl fmt::Display for SnapshotRevision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A snapshot and the revision that must be supplied to the next CAS.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRun {
    revision: SnapshotRevision,
    snapshot: RunSnapshot,
}

impl StoredRun {
    pub fn revision(&self) -> SnapshotRevision {
        self.revision
    }

    pub fn snapshot(&self) -> &RunSnapshot {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> RunSnapshot {
        self.snapshot
    }
}

/// Exclusive, opaque authority to load or update one run.
///
/// Leases deliberately do not implement `Clone`. Dropping a live in-memory
/// lease releases it; expiration remains the fallback for crashed owners.
pub struct RunLease {
    run_id: RunId,
    expires_at: Instant,
    store_token: Box<dyn Any + Send + Sync>,
}

impl RunLease {
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn expires_at(&self) -> Instant {
        self.expires_at
    }

    /// Construct an opaque lease from a store-private token.
    ///
    /// This is an implementation seam for custom [`RunStore`] adapters. A
    /// private token type prevents callers outside that store from forging a
    /// lease that passes `store_token` downcasting and store-side validation.
    #[doc(hidden)]
    pub fn from_store_token<T>(run_id: RunId, expires_at: Instant, token: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            run_id,
            expires_at,
            store_token: Box::new(token),
        }
    }

    /// Recover a store-private token issued through [`Self::from_store_token`].
    #[doc(hidden)]
    pub fn store_token<T>(&self) -> Option<&T>
    where
        T: Any + Send + Sync,
    {
        self.store_token.downcast_ref()
    }

    /// Update the public expiration after a store has atomically renewed it.
    #[doc(hidden)]
    pub fn set_expires_at(&mut self, expires_at: Instant) {
        self.expires_at = expires_at;
    }
}

impl fmt::Debug for RunLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunLease")
            .field("run_id", &self.run_id)
            .field("store_token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Boxed future returned by an object-safe [`RunStore`].
pub type RunStoreFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, RunStoreError>> + Send + 'a>>;

/// Snapshot persistence hook with exclusive resume and optimistic updates.
pub trait RunStore: Send + Sync {
    fn acquire<'a>(&'a self, run_id: &'a RunId, ttl: Duration) -> RunStoreFuture<'a, RunLease>;

    fn load<'a>(&'a self, lease: &'a RunLease) -> RunStoreFuture<'a, Option<StoredRun>>;

    fn compare_and_swap<'a>(
        &'a self,
        lease: &'a RunLease,
        expected: SnapshotRevision,
        snapshot: RunSnapshot,
    ) -> RunStoreFuture<'a, SnapshotRevision>;

    fn renew<'a>(&'a self, lease: &'a mut RunLease, ttl: Duration) -> RunStoreFuture<'a, ()>;
}

/// Typed snapshot-store failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RunStoreError {
    #[error("lease duration must be greater than zero and representable")]
    InvalidLeaseDuration,
    #[error("run `{run_id}` already has an active lease")]
    LeaseConflict { run_id: RunId },
    #[error("the lease belongs to another store")]
    ForeignLease,
    #[error("the lease no longer owns this run")]
    LeaseLost,
    #[error("the lease has expired")]
    LeaseExpired,
    #[error("run-store lease tokens are exhausted")]
    LeaseTokenExhausted,
    #[error("snapshot revisions are exhausted")]
    RevisionExhausted,
    #[error("snapshot run `{snapshot_run_id}` does not match leased run `{leased_run_id}`")]
    RunIdMismatch {
        leased_run_id: RunId,
        snapshot_run_id: RunId,
    },
    #[error("CAS conflict: expected revision {expected}, actual revision {actual}")]
    CasConflict {
        expected: SnapshotRevision,
        actual: SnapshotRevision,
    },
    #[error("a terminal run cannot be updated")]
    RunAlreadyTerminal,
    #[error("successor snapshot changes immutable run identity or policy")]
    IncompatibleSuccessor,
    #[error("successor snapshot rewrites or removes execution-log history")]
    ExecutionLogRegression,
    #[error("run store is unavailable")]
    Unavailable,
    #[error(transparent)]
    InvalidSnapshot(#[from] RunSnapshotError),
}

/// Process-local reference implementation of lease and CAS semantics.
#[derive(Clone)]
pub struct InMemoryRunStore {
    inner: Arc<InMemoryRunStoreInner>,
}

impl Default for InMemoryRunStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryRunStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryRunStoreInner {
                store_id: NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed),
                state: Mutex::new(InMemoryState {
                    next_lease_token: 1,
                    leases: BTreeMap::new(),
                    runs: BTreeMap::new(),
                }),
            }),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, InMemoryState>, RunStoreError> {
        self.inner
            .state
            .lock()
            .map_err(|_| RunStoreError::Unavailable)
    }

    fn validate_lease(
        &self,
        state: &mut InMemoryState,
        lease: &RunLease,
    ) -> Result<(), RunStoreError> {
        let lease_token = lease
            .store_token::<InMemoryLeaseToken>()
            .ok_or(RunStoreError::ForeignLease)?;
        if lease_token.store_id != self.inner.store_id || &lease_token.run_id != lease.run_id() {
            return Err(RunStoreError::ForeignLease);
        }
        let now = Instant::now();
        let Some(record) = state.leases.get(lease.run_id()) else {
            return Err(RunStoreError::LeaseLost);
        };
        if record.token != lease_token.token {
            return Err(RunStoreError::LeaseLost);
        }
        if now >= record.expires_at {
            state.leases.remove(lease.run_id());
            return Err(RunStoreError::LeaseExpired);
        }
        Ok(())
    }
}

impl fmt::Debug for InMemoryRunStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counts = self
            .inner
            .state
            .lock()
            .ok()
            .map(|state| (state.runs.len(), state.leases.len()));
        formatter
            .debug_struct("InMemoryRunStore")
            .field("store_id", &self.inner.store_id)
            .field("runs", &counts.map(|counts| counts.0))
            .field("active_leases", &counts.map(|counts| counts.1))
            .finish()
    }
}

impl RunStore for InMemoryRunStore {
    fn acquire<'a>(&'a self, run_id: &'a RunId, ttl: Duration) -> RunStoreFuture<'a, RunLease> {
        Box::pin(async move {
            let now = Instant::now();
            let expires_at = now
                .checked_add(ttl)
                .filter(|_| !ttl.is_zero())
                .ok_or(RunStoreError::InvalidLeaseDuration)?;
            let mut state = self.lock()?;
            if let Some(record) = state.leases.get(run_id) {
                if now < record.expires_at {
                    return Err(RunStoreError::LeaseConflict {
                        run_id: run_id.clone(),
                    });
                }
                state.leases.remove(run_id);
            }

            let token = state.next_lease_token;
            state.next_lease_token = state
                .next_lease_token
                .checked_add(1)
                .ok_or(RunStoreError::LeaseTokenExhausted)?;
            state
                .leases
                .insert(run_id.clone(), LeaseRecord { token, expires_at });
            drop(state);

            Ok(RunLease::from_store_token(
                run_id.clone(),
                expires_at,
                InMemoryLeaseToken {
                    store_id: self.inner.store_id,
                    token,
                    run_id: run_id.clone(),
                    store: Arc::downgrade(&self.inner),
                },
            ))
        })
    }

    fn load<'a>(&'a self, lease: &'a RunLease) -> RunStoreFuture<'a, Option<StoredRun>> {
        Box::pin(async move {
            let mut state = self.lock()?;
            self.validate_lease(&mut state, lease)?;
            let stored = state.runs.get(lease.run_id()).cloned();
            drop(state);

            stored
                .map(|stored| {
                    Ok(StoredRun {
                        revision: stored.revision,
                        snapshot: stored
                            .snapshot
                            .recovered_for_resume(current_unix_millis())?,
                    })
                })
                .transpose()
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
                .runs
                .get(lease.run_id())
                .map_or(SnapshotRevision::EMPTY, |stored| stored.revision);
            if actual != expected {
                return Err(RunStoreError::CasConflict { expected, actual });
            }

            if let Some(current) = state.runs.get(lease.run_id()) {
                if current.snapshot.state().is_terminal() {
                    return Err(RunStoreError::RunAlreadyTerminal);
                }
                if !current.snapshot.is_compatible_successor(&snapshot) {
                    return Err(RunStoreError::IncompatibleSuccessor);
                }
                if !snapshot
                    .execution_log()
                    .has_prefix(current.snapshot.execution_log())
                {
                    return Err(RunStoreError::ExecutionLogRegression);
                }
            }

            let revision = actual.next()?;
            state
                .runs
                .insert(lease.run_id().clone(), StoredEntry { revision, snapshot });
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
            let record = state
                .leases
                .get_mut(lease.run_id())
                .ok_or(RunStoreError::LeaseLost)?;
            record.expires_at = expires_at;
            lease.set_expires_at(expires_at);
            Ok(())
        })
    }
}

struct InMemoryRunStoreInner {
    store_id: u64,
    state: Mutex<InMemoryState>,
}

struct InMemoryLeaseToken {
    store_id: u64,
    token: u64,
    run_id: RunId,
    store: Weak<InMemoryRunStoreInner>,
}

impl Drop for InMemoryLeaseToken {
    fn drop(&mut self) {
        let Some(store) = self.store.upgrade() else {
            return;
        };
        let Ok(mut state) = store.state.lock() else {
            return;
        };
        if state.leases.get(&self.run_id).map(|record| record.token) == Some(self.token) {
            state.leases.remove(&self.run_id);
        }
    }
}

struct InMemoryState {
    next_lease_token: u64,
    leases: BTreeMap<RunId, LeaseRecord>,
    runs: BTreeMap<RunId, StoredEntry>,
}

struct LeaseRecord {
    token: u64,
    expires_at: Instant,
}

#[derive(Clone)]
struct StoredEntry {
    revision: SnapshotRevision,
    snapshot: RunSnapshot,
}

fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}
