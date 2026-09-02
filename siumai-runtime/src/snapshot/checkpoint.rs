use std::io::{self, Write};
use std::time::Duration;

use siumai_core::LanguageRequest;
use thiserror::Error;

use super::{
    CheckpointId, LineageId, ResumePoint, RunId, RunLease, RunSnapshot, RunSnapshotError,
    RunSnapshotSuccessorError, RunStore, RunStoreError, SnapshotCheckpoint, SnapshotEngineVersion,
    SnapshotFingerprints, SnapshotRevision, StoredRun,
};
use crate::{BudgetError, BudgetKind, RunBudget, RunReport};

pub(crate) enum CheckpointIntent<'a> {
    Initial(InitialCheckpointIntent),
    Successor {
        previous: &'a RunSnapshot,
        checkpoint_id: CheckpointId,
        continuation: LanguageRequest,
        report: RunReport,
        resume_point: ResumePoint,
    },
}

pub(crate) struct InitialCheckpointIntent {
    pub(crate) engine_version: SnapshotEngineVersion,
    pub(crate) run_id: RunId,
    pub(crate) lineage_id: LineageId,
    pub(crate) checkpoint_id: CheckpointId,
    pub(crate) fingerprints: SnapshotFingerprints,
    pub(crate) continuation: LanguageRequest,
    pub(crate) report: RunReport,
    pub(crate) deadline_unix_ms: Option<u64>,
    pub(crate) resume_point: ResumePoint,
}

impl<'a> CheckpointIntent<'a> {
    pub(crate) fn initial(intent: InitialCheckpointIntent) -> Self {
        Self::Initial(intent)
    }

    pub(crate) fn successor(
        previous: &'a RunSnapshot,
        checkpoint_id: CheckpointId,
        continuation: LanguageRequest,
        report: RunReport,
        resume_point: ResumePoint,
    ) -> Self {
        Self::Successor {
            previous,
            checkpoint_id,
            continuation,
            report,
            resume_point,
        }
    }

    fn assemble(self) -> Result<(RunSnapshot, Option<&'a RunSnapshot>), RunSnapshotError> {
        match self {
            Self::Initial(InitialCheckpointIntent {
                engine_version,
                run_id,
                lineage_id,
                checkpoint_id,
                fingerprints,
                continuation,
                report,
                deadline_unix_ms,
                resume_point,
            }) => Ok((
                RunSnapshot::new(
                    SnapshotCheckpoint::new(
                        engine_version,
                        run_id,
                        lineage_id,
                        checkpoint_id,
                        None,
                    )?,
                    fingerprints,
                    continuation,
                    report,
                    deadline_unix_ms,
                    resume_point,
                )?,
                None,
            )),
            Self::Successor {
                previous,
                checkpoint_id,
                continuation,
                report,
                resume_point,
            } => Ok((
                RunSnapshot::new(
                    SnapshotCheckpoint::new(
                        previous.engine_version().clone(),
                        previous.run_id().clone(),
                        previous.lineage_id().clone(),
                        checkpoint_id,
                        Some(previous.checkpoint_id().clone()),
                    )?,
                    previous.fingerprints().clone(),
                    continuation,
                    report,
                    previous.deadline_unix_ms(),
                    resume_point,
                )?,
                Some(previous),
            )),
        }
    }
}

pub(crate) struct CheckpointWriter<'a> {
    store: &'a dyn RunStore,
    lease: &'a mut RunLease,
    lease_ttl: Duration,
    budget: &'a RunBudget,
}

impl<'a> CheckpointWriter<'a> {
    pub(crate) fn new(
        store: &'a dyn RunStore,
        lease: &'a mut RunLease,
        lease_ttl: Duration,
        budget: &'a RunBudget,
    ) -> Self {
        Self {
            store,
            lease,
            lease_ttl,
            budget,
        }
    }

    pub(crate) async fn commit(
        self,
        expected: SnapshotRevision,
        intent: CheckpointIntent<'_>,
    ) -> Result<StoredRun, CheckpointWriteError> {
        let (snapshot, previous) = intent.assemble()?;
        if let Some(previous) = previous {
            previous.validate_successor(&snapshot)?;
        }
        measure_compact_json(&snapshot, self.budget.max_snapshot_bytes())?;
        self.store.renew(self.lease, self.lease_ttl).await?;
        let revision = self
            .store
            .compare_and_swap(self.lease, expected, snapshot.clone())
            .await?;
        Ok(StoredRun::new(revision, snapshot))
    }
}

#[derive(Debug, Error)]
pub(crate) enum CheckpointWriteError {
    #[error(transparent)]
    Snapshot(#[from] RunSnapshotError),
    #[error(transparent)]
    InvalidSuccessor(#[from] RunSnapshotSuccessorError),
    #[error(transparent)]
    Budget(#[from] BudgetError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] RunStoreError),
}

fn measure_compact_json(
    snapshot: &RunSnapshot,
    maximum: usize,
) -> Result<usize, CheckpointWriteError> {
    let mut counter = BoundedByteCounter::new(maximum);
    match serde_json::to_writer(&mut counter, snapshot) {
        Ok(()) => Ok(counter.written),
        Err(_error) if counter.limit_exceeded => Err(snapshot_limit_error(maximum).into()),
        Err(error) => Err(error.into()),
    }
}

fn snapshot_limit_error(maximum: usize) -> BudgetError {
    let maximum = u64::try_from(maximum).unwrap_or(u64::MAX);
    BudgetError::Exceeded {
        kind: BudgetKind::SnapshotBytes,
        actual: maximum.saturating_add(1),
        maximum,
    }
}

struct BoundedByteCounter {
    written: usize,
    maximum: usize,
    limit_exceeded: bool,
}

impl BoundedByteCounter {
    fn new(maximum: usize) -> Self {
        Self {
            written: 0,
            maximum,
            limit_exceeded: false,
        }
    }
}

impl Write for BoundedByteCounter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(next) = self.written.checked_add(buffer.len()) else {
            self.limit_exceeded = true;
            return Err(io::Error::other("snapshot byte count exceeded its bound"));
        };
        if next > self.maximum {
            self.limit_exceeded = true;
            return Err(io::Error::other("snapshot byte count exceeded its bound"));
        }
        self.written = next;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
