//! Explicitly unstable job and bidirectional session contracts.
//!
//! These APIs are intentionally separate from the six stable one-shot families.

mod session;

use std::pin::Pin;
use std::{fmt, str::FromStr};

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error as ThisError;

use crate::error::Error;
use crate::options::CallOptions;

pub use session::{
    ProviderSession, SessionCloseMetadata, SessionCloseOrigin, SessionCloseRequest, SessionFailure,
    SessionIncoming, SessionLineageId, SessionLineageIdError, SessionTerminal,
    SessionTransportKind,
};

const MAX_JOB_ID_BYTES: usize = 512;

/// A bounded provider job identifier that is safe to place in a relative request path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct JobId(String);

impl JobId {
    pub fn new(value: impl Into<String>) -> Result<Self, JobIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(JobIdError::Empty);
        }
        if value.len() > MAX_JOB_ID_BYTES {
            return Err(JobIdError::TooLong {
                maximum: MAX_JOB_ID_BYTES,
            });
        }
        if value.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(character, '/' | '\\' | '?' | '#')
        }) {
            return Err(JobIdError::InvalidCharacter);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for JobId {
    type Err = JobIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for JobId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum JobIdError {
    #[error("job identifier must not be empty")]
    Empty,
    #[error("job identifier exceeds {maximum} bytes")]
    TooLong { maximum: usize },
    #[error("job identifier contains a character that is unsafe in a request path")]
    InvalidCharacter,
}

/// A serializable snapshot of one provider-owned asynchronous media job.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaJob {
    pub id: JobId,
    pub status: JobStatus,
    pub state: Value,
}

impl MediaJob {
    pub fn new(id: JobId, status: JobStatus, state: Value) -> Self {
        Self { id, status, state }
    }

    pub fn id(&self) -> &JobId {
        &self.id
    }

    pub fn status(&self) -> &JobStatus {
        &self.status
    }

    pub fn state(&self) -> &Value {
        &self.state
    }

    pub fn into_parts(self) -> (JobId, JobStatus, Value) {
        (self.id, self.status, self.state)
    }
}

impl fmt::Debug for MediaJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaJob")
            .field("id", &self.id)
            .field("status", &self.status)
            .field("state", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed { message: String },
    Cancelled,
    Expired,
}

impl JobStatus {
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Cancelled | Self::Expired
        )
    }

    pub const fn is_successful(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Experimental dynamic video-job contract.
///
/// Provider packages should expose typed request and job wrappers as their primary API. This
/// value-based trait is the explicit dynamic seam for orchestration. Polling and cancellation
/// return complete snapshots so callers can persist progress without a hidden provider cache.
#[async_trait]
pub trait VideoJobModel: Send + Sync {
    async fn create(&self, request: Value, options: CallOptions) -> Result<MediaJob, Error>;
    async fn poll(&self, job: &MediaJob, options: CallOptions) -> Result<MediaJob, Error>;
    async fn cancel(&self, job: &MediaJob, options: CallOptions) -> Result<MediaJob, Error>;
    async fn materialize(&self, job: &MediaJob, options: CallOptions) -> Result<Vec<u8>, Error>;
}

#[derive(Debug)]
#[non_exhaustive]
pub enum TranscriptionStreamEvent {
    Delta { text: String },
    Final { text: String },
    Failed { error: Error },
    Cancelled,
}

pub type TranscriptionStream =
    Pin<Box<dyn Stream<Item = TranscriptionStreamEvent> + Send + 'static>>;

#[async_trait]
pub trait StreamingTranscriptionModel: Send + Sync {
    async fn stream(
        &self,
        audio: Pin<Box<dyn Stream<Item = Vec<u8>> + Send + 'static>>,
        options: CallOptions,
    ) -> Result<TranscriptionStream, Error>;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{JobId, JobStatus, MediaJob};

    #[test]
    fn job_ids_are_bounded_and_path_safe() {
        assert_eq!(JobId::new("task-42").unwrap().as_str(), "task-42");
        assert!(JobId::new("").is_err());
        assert!(JobId::new("task/42").is_err());
        assert!(JobId::new("task?secret=value").is_err());
        assert!(JobId::new("x".repeat(513)).is_err());
    }

    #[test]
    fn media_job_debug_redacts_provider_state() {
        let job = MediaJob::new(
            JobId::new("task-42").unwrap(),
            JobStatus::Completed,
            json!({"video_url": "https://example.com/video.mp4?secret=canary"}),
        );
        let debug = format!("{job:?}");
        assert!(!debug.contains("secret=canary"));
        assert!(job.status().is_terminal());
        assert!(job.status().is_successful());
    }
}
