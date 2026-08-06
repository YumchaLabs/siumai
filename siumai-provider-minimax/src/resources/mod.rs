//! Provider-native MiniMax resource APIs.
//!
//! These operations have lifecycles distinct from language generation and are
//! intentionally exposed from the configured provider rather than projected
//! into provider-neutral model traits.

mod common;
mod files;
mod image;
mod music;
mod speech;
mod video;

pub(crate) use common::NativeRuntime;
pub use files::{
    MinimaxFile, MinimaxFileDeletePurpose, MinimaxFileDeleteResult, MinimaxFileId,
    MinimaxFileIdError, MinimaxFileList, MinimaxFileListPurpose, MinimaxFileUpload,
    MinimaxFileUploadPurpose, MinimaxFiles,
};
pub use image::{
    MinimaxImageAspectRatio, MinimaxImageDimensions, MinimaxImageGeneration, MinimaxImageMetadata,
    MinimaxImageRequest, MinimaxImageResponseFormat, MinimaxImageSize,
    MinimaxImageSubjectReference, MinimaxImages,
};
pub use music::{
    MinimaxMusic, MinimaxMusicAudioFormat, MinimaxMusicAudioSetting, MinimaxMusicBitrate,
    MinimaxMusicCoverSource, MinimaxMusicCoverSourceKind, MinimaxMusicGeneration,
    MinimaxMusicMetadata, MinimaxMusicOutput, MinimaxMusicOutputFormat, MinimaxMusicRequest,
    MinimaxMusicRequestKind, MinimaxMusicSampleRate, MinimaxMusicStatus,
};
pub use speech::{
    MinimaxAsyncSpeechInput, MinimaxAsyncSpeechRequest, MinimaxLanguageBoost,
    MinimaxPronunciationDictionary, MinimaxSoundEffect, MinimaxSpeech, MinimaxSpeechAudioFormat,
    MinimaxSpeechAudioSettings, MinimaxSpeechBitrate, MinimaxSpeechChannels,
    MinimaxSpeechDownloadUrl, MinimaxSpeechEmotion, MinimaxSpeechInfo, MinimaxSpeechOutput,
    MinimaxSpeechOutputFormat, MinimaxSpeechPitch, MinimaxSpeechResponse, MinimaxSpeechSampleRate,
    MinimaxSpeechSpeed, MinimaxSpeechSubmission, MinimaxSpeechSynthesisRequest, MinimaxSpeechTask,
    MinimaxSpeechTaskId, MinimaxSpeechTaskIdError, MinimaxSpeechTaskStatus, MinimaxSpeechTaskToken,
    MinimaxSpeechValueError, MinimaxSpeechVolume, MinimaxSubtitleGranularity,
    MinimaxVoiceEffectLevel, MinimaxVoiceEffects, MinimaxVoiceId, MinimaxVoiceSettings,
    SPEECH_API_VERIFIED_ON, SPEECH_ASYNC_API_SOURCE, SPEECH_ASYNC_QUERY_API_SOURCE,
    SPEECH_HTTP_API_SOURCE,
};
pub use video::{
    MinimaxVideo, MinimaxVideoCreateResult, MinimaxVideoDeleteAction, MinimaxVideoDeleteResult,
    MinimaxVideoInput, MinimaxVideoListQuery, MinimaxVideoMediaSource, MinimaxVideoRatio,
    MinimaxVideoRequest, MinimaxVideoResolution, MinimaxVideoTask, MinimaxVideoTaskContent,
    MinimaxVideoTaskFailure, MinimaxVideoTaskId, MinimaxVideoTaskList, MinimaxVideoTaskStatus,
    MinimaxVideoTaskType, MinimaxVideoUsage,
};
