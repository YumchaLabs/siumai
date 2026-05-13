use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use siumai_core::speech::SpeechModel as FamilySpeechModel;
use siumai_core::transcription::TranscriptionModel as FamilyTranscriptionModel;
use siumai_core::video::VideoModel as FamilyVideoModel;

pub(super) type CacheEntry = TimedFamilyModelCacheEntry<dyn FamilyLanguageModel>;
pub(super) type CompletionCacheEntry = TimedFamilyModelCacheEntry<dyn FamilyCompletionModel>;
pub(super) type SpeechCacheEntry = TimedFamilyModelCacheEntry<dyn FamilySpeechModel>;
pub(super) type TranscriptionCacheEntry = TimedFamilyModelCacheEntry<dyn FamilyTranscriptionModel>;
pub(super) type VideoCacheEntry = TimedFamilyModelCacheEntry<dyn FamilyVideoModel>;

pub(super) struct TimedFamilyModelCacheEntry<T: ?Sized> {
    pub(super) model: Arc<T>,
    created_at: Instant,
}

impl<T: ?Sized> TimedFamilyModelCacheEntry<T> {
    pub(super) fn new(model: Arc<T>) -> Self {
        Self {
            model,
            created_at: Instant::now(),
        }
    }

    pub(super) fn is_expired(&self, ttl: Option<Duration>) -> bool {
        ttl.is_some_and(|ttl| self.created_at.elapsed() > ttl)
    }
}
