use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use siumai_core::LanguageRequest;

use super::{ResumePoint, RunSnapshot, RunSnapshotError, SnapshotCheckpoint, SnapshotFingerprints};
use crate::RunReport;

/// The only snapshot wire schema understood by this release.
///
/// This changes for incompatible serialized-shape revisions, not for runtime
/// execution or fingerprint interpretation changes.
pub const RUN_SNAPSHOT_SCHEMA_VERSION: u16 = 8;

#[derive(Deserialize)]
struct RunSnapshotEnvelope {
    snapshot_version: u16,
    #[serde(flatten)]
    payload: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunSnapshotPayloadWire {
    checkpoint: SnapshotCheckpoint,
    fingerprints: SnapshotFingerprints,
    continuation: LanguageRequest,
    report: RunReport,
    deadline_unix_ms: Option<u64>,
    resume_point: ResumePoint,
}

impl<'de> Deserialize<'de> for RunSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let envelope = RunSnapshotEnvelope::deserialize(deserializer)?;
        if envelope.snapshot_version != RUN_SNAPSHOT_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                RunSnapshotError::UnsupportedVersion {
                    found: envelope.snapshot_version,
                    supported: RUN_SNAPSHOT_SCHEMA_VERSION,
                },
            ));
        }
        let payload = Value::Object(envelope.payload.into_iter().collect());
        let wire = serde_json::from_value::<RunSnapshotPayloadWire>(payload)
            .map_err(serde::de::Error::custom)?;
        let snapshot = RunSnapshot {
            snapshot_version: envelope.snapshot_version,
            checkpoint: wire.checkpoint,
            fingerprints: wire.fingerprints,
            continuation: wire.continuation,
            report: wire.report,
            deadline_unix_ms: wire.deadline_unix_ms,
            resume_point: wire.resume_point,
        };
        snapshot.validate().map_err(serde::de::Error::custom)?;
        Ok(snapshot)
    }
}
