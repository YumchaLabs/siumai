//! Explicitly unstable job and bidirectional session contracts.
//!
//! These APIs are intentionally separate from the six stable one-shot families.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;
use crate::options::CallOptions;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaJob {
    pub id: JobId,
    pub state: Value,
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

#[async_trait]
pub trait VideoJobModel: Send + Sync {
    async fn create(&self, request: Value, options: CallOptions) -> Result<MediaJob, Error>;
    async fn poll(&self, job: &MediaJob, options: CallOptions) -> Result<JobStatus, Error>;
    async fn materialize(&self, job: MediaJob, options: CallOptions) -> Result<Vec<u8>, Error>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RealtimeEvent {
    Json(Value),
    Binary(Vec<u8>),
    Interrupted {
        item_id: Option<String>,
    },
    Resumed {
        cursor: String,
    },
    Closed {
        code: Option<u16>,
        reason: Option<String>,
    },
}

#[async_trait]
pub trait RealtimeSession: Send {
    async fn send(&mut self, event: RealtimeEvent) -> Result<(), Error>;
    async fn next(&mut self) -> Result<Option<RealtimeEvent>, Error>;
    async fn close(&mut self) -> Result<(), Error>;
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

#[derive(Debug)]
#[non_exhaustive]
pub enum TranslationStreamEvent {
    SourceTranscript { text: String },
    TranslatedText { text: String },
    Audio(Vec<u8>),
    Failed { error: Error },
    Cancelled,
}

pub type TranslationStream = Pin<Box<dyn Stream<Item = TranslationStreamEvent> + Send + 'static>>;

#[async_trait]
pub trait StreamingTranslationModel: Send + Sync {
    async fn stream(
        &self,
        audio: Pin<Box<dyn Stream<Item = Vec<u8>> + Send + 'static>>,
        options: CallOptions,
    ) -> Result<TranslationStream, Error>;
}
