use siumai_core::LanguageRequest;

use super::{
    CheckpointId, LineageId, ResumePoint, RunId, RunSnapshot, RunSnapshotError, SnapshotCheckpoint,
    SnapshotEngineVersion, SnapshotFingerprints,
};
use crate::RunReport;

pub(crate) struct InitialSnapshotParts {
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

pub(crate) fn assemble_initial_snapshot(
    parts: InitialSnapshotParts,
) -> Result<RunSnapshot, RunSnapshotError> {
    RunSnapshot::new(
        SnapshotCheckpoint::new(
            parts.engine_version,
            parts.run_id,
            parts.lineage_id,
            parts.checkpoint_id,
            None,
        )?,
        parts.fingerprints,
        parts.continuation,
        parts.report,
        parts.deadline_unix_ms,
        parts.resume_point,
    )
}

pub(crate) fn assemble_successor_snapshot(
    previous: &RunSnapshot,
    checkpoint_id: CheckpointId,
    continuation: LanguageRequest,
    report: RunReport,
    resume_point: ResumePoint,
) -> Result<RunSnapshot, RunSnapshotError> {
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
    )
}
