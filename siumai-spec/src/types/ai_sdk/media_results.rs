use crate::types::Warning;
use serde::{Deserialize, Serialize};

use super::{
    GeneratedAudioFile, GeneratedFile, ImageModelProviderMetadata, ImageModelResponseMetadata,
    ImageModelUsage, ProviderMetadata, SpeechModelResponseMetadata,
    TranscriptionModelResponseMetadata, VideoModelProviderMetadata, VideoModelResponseMetadata,
};

/// Passive AI SDK-style result envelope for a `generateImage` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageResult {
    /// First generated image.
    pub image: GeneratedFile,
    /// Generated images.
    pub images: Vec<GeneratedFile>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<ImageModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ImageModelProviderMetadata,
    /// Combined image-model usage across all underlying provider calls.
    pub usage: ImageModelUsage,
}

impl GenerateImageResult {
    /// Create a result from a required first image and the full image list.
    pub fn new(image: GeneratedFile, mut images: Vec<GeneratedFile>) -> Self {
        if images.is_empty() {
            images.push(image.clone());
        }

        Self {
            image,
            images,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ImageModelProviderMetadata::default(),
            usage: ImageModelUsage::default(),
        }
    }

    /// Try to create a result from the full image list.
    pub fn from_images(images: Vec<GeneratedFile>) -> Option<Self> {
        let image = images.first()?.clone();
        Some(Self::new(image, images))
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<ImageModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ImageModelProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }

    /// Attach image-model usage.
    pub fn with_usage(mut self, usage: ImageModelUsage) -> Self {
        self.usage = usage;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_GenerateImageResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_GenerateImageResult = GenerateImageResult;

/// Passive AI SDK-style result envelope for an `experimental_generateVideo` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateVideoResult {
    /// First generated video.
    pub video: GeneratedFile,
    /// Generated videos.
    pub videos: Vec<GeneratedFile>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<VideoModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: VideoModelProviderMetadata,
}

impl GenerateVideoResult {
    /// Create a result from a required first video and the full video list.
    pub fn new(video: GeneratedFile, mut videos: Vec<GeneratedFile>) -> Self {
        if videos.is_empty() {
            videos.push(video.clone());
        }

        Self {
            video,
            videos,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: VideoModelProviderMetadata::default(),
        }
    }

    /// Try to create a result from the full video list.
    pub fn from_videos(videos: Vec<GeneratedFile>) -> Option<Self> {
        let video = videos.first()?.clone();
        Some(Self::new(video, videos))
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<VideoModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: VideoModelProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Passive AI SDK-style result envelope for a `generateSpeech` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechResult {
    /// Generated audio file.
    pub audio: GeneratedAudioFile,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<SpeechModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ProviderMetadata,
}

impl SpeechResult {
    /// Create a speech result from generated audio.
    pub fn new(audio: GeneratedAudioFile) -> Self {
        Self {
            audio,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ProviderMetadata::default(),
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<SpeechModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_SpeechResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_SpeechResult = SpeechResult;

/// Transcript segment used by AI SDK-style `TranscriptionResult`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionSegment {
    /// Segment text.
    pub text: String,
    /// Segment start time in seconds.
    pub start_second: f64,
    /// Segment end time in seconds.
    pub end_second: f64,
}

impl TranscriptionSegment {
    /// Create a transcript segment.
    pub fn new(text: impl Into<String>, start_second: f64, end_second: f64) -> Self {
        Self {
            text: text.into(),
            start_second,
            end_second,
        }
    }
}

/// Passive AI SDK-style result envelope for a `transcribe` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResult {
    /// Complete transcript text.
    pub text: String,
    /// Segment-level transcript timing information.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language, usually as an ISO-639-1 code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Total audio duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_in_seconds: Option<f64>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<TranscriptionModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ProviderMetadata,
}

impl TranscriptionResult {
    /// Create a transcription result from final text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            segments: Vec::new(),
            language: None,
            duration_in_seconds: None,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ProviderMetadata::default(),
        }
    }

    /// Attach transcript segments.
    pub fn with_segments(mut self, segments: Vec<TranscriptionSegment>) -> Self {
        self.segments = segments;
        self
    }

    /// Attach detected language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Attach total audio duration in seconds.
    pub fn with_duration_in_seconds(mut self, duration_in_seconds: f64) -> Self {
        self.duration_in_seconds = Some(duration_in_seconds);
        self
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<TranscriptionModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_TranscriptionResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_TranscriptionResult = TranscriptionResult;
