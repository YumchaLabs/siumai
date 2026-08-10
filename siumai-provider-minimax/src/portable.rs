use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageArtifact, ImageLimits, ImageModel,
    ImageRequest, ImageResponse, MediaData, Model, ModelDescriptor, ModelOperation,
    ResponseMetadata, SpeechLimits, SpeechModel, SpeechRequest, SpeechResponse, Usage,
};

use crate::resources::{
    MinimaxImageDimensions, MinimaxImageRequest, MinimaxImageResponseFormat, MinimaxImages,
    MinimaxSpeech, MinimaxSpeechAudioFormat, MinimaxSpeechAudioSettings, MinimaxSpeechOutput,
    MinimaxSpeechOutputFormat, MinimaxSpeechSpeed, MinimaxSpeechSynthesisRequest, MinimaxVoiceId,
    MinimaxVoiceSettings, NativeRuntime,
};

const MAX_IMAGE_OUTPUTS: u32 = 9;
const MAX_SPEECH_TEXT_CHARACTERS: usize = 9_999;

/// Portable image-generation adapter over MiniMax's provider-native Image API.
#[derive(Clone)]
pub struct MinimaxImageModel {
    images: MinimaxImages,
    descriptor: ModelDescriptor,
}

impl MinimaxImageModel {
    pub(crate) fn new(runtime: Arc<NativeRuntime>, descriptor: ModelDescriptor) -> Self {
        Self {
            images: MinimaxImages::new(runtime),
            descriptor,
        }
    }

    fn request(&self, request: &ImageRequest) -> Result<MinimaxImageRequest, Error> {
        if request.format().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "MiniMax portable image generation does not promise a selectable media encoding",
            ));
        }
        let count = u8::try_from(request.count()).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax portable image count exceeds the provider wire range",
            )
            .with_source(source)
        })?;
        let mut native = MinimaxImageRequest::new(self.model_id().as_str(), request.prompt())?
            .with_count(count)?
            .with_response_format(MinimaxImageResponseFormat::Base64);
        if let Some(size) = request.size() {
            let width = u16::try_from(size.width()).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "MiniMax portable image width exceeds the provider wire range",
                )
                .with_source(source)
            })?;
            let height = u16::try_from(size.height()).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "MiniMax portable image height exceeds the provider wire range",
                )
                .with_source(source)
            })?;
            native = native.with_dimensions(MinimaxImageDimensions::new(width, height)?)?;
        }
        Ok(native)
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::GenerateImage),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for MinimaxImageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for MinimaxImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for MinimaxImageModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(MAX_IMAGE_OUTPUTS),
        }
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        call: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let native = self
            .request(&request)
            .map_err(|error| self.contextualize(error))?;
        let generation = self
            .images
            .generate_with_options(native, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        let image_metadata = generation.metadata();
        let mut provider = BTreeMap::new();
        if let Some(value) = image_metadata.success_count() {
            provider.insert("success_count".to_string(), Value::from(value));
        }
        if let Some(value) = image_metadata.failed_count() {
            provider.insert("failed_count".to_string(), Value::from(value));
        }
        let response = ImageResponse {
            images: generation
                .base64_images()
                .iter()
                .map(|encoded| decode_image_artifact(encoded))
                .collect::<Result<Vec<_>, _>>()?,
            metadata: ResponseMetadata {
                response_id: generation.id().map(str::to_owned),
                request_id: None,
                model: Some(self.model_id().clone()),
            },
            usage: Usage::default(),
            warnings: Vec::new(),
            provider,
        };
        response
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(response)
    }
}

fn decode_image_artifact(encoded: &str) -> Result<ImageArtifact, Error> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax returned invalid base64 image data",
            )
            .with_source(source)
        })?;
    let media_type = image_media_type(&bytes).ok_or_else(|| {
        Error::new(
            ErrorKind::Protocol,
            "MiniMax returned an image with an unknown media type",
        )
    })?;
    Ok(ImageArtifact {
        media_type: media_type.to_string(),
        data: MediaData::Bytes(bytes.into()),
        revised_prompt: None,
    })
}

fn image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// Portable buffered speech adapter over MiniMax's synchronous Speech API.
#[derive(Clone)]
pub struct MinimaxSpeechModel {
    speech: MinimaxSpeech,
    descriptor: ModelDescriptor,
}

impl MinimaxSpeechModel {
    pub(crate) fn new(runtime: Arc<NativeRuntime>, descriptor: ModelDescriptor) -> Self {
        Self {
            speech: MinimaxSpeech::new(runtime),
            descriptor,
        }
    }

    fn request(
        &self,
        request: &SpeechRequest,
    ) -> Result<(MinimaxSpeechSynthesisRequest, MinimaxSpeechAudioFormat), Error> {
        if request.language().is_some() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "MiniMax portable speech does not map language selection to provider language boosting",
            ));
        }
        let voice = request.voice().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax portable speech requires an explicit voice",
            )
        })?;
        let voice_id = MinimaxVoiceId::new(voice).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "MiniMax speech voice is invalid")
                .with_source(source)
        })?;
        let mut voice = MinimaxVoiceSettings::new(voice_id);
        if let Some(speed) = request.speed() {
            voice = voice.with_speed(MinimaxSpeechSpeed::new(speed).map_err(|source| {
                Error::new(ErrorKind::InvalidInput, "MiniMax speech speed is invalid")
                    .with_source(source)
            })?);
        }
        let format = speech_format(request.format())?;
        let native =
            MinimaxSpeechSynthesisRequest::new(self.model_id().as_str(), request.text(), voice)?
                .with_audio(MinimaxSpeechAudioSettings::new(format))
                .with_output_format(MinimaxSpeechOutputFormat::Hex);
        Ok((native, format))
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::SynthesizeSpeech),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl fmt::Debug for MinimaxSpeechModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for MinimaxSpeechModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl SpeechModel for MinimaxSpeechModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: None,
            max_text_chars: Some(MAX_SPEECH_TEXT_CHARACTERS),
        }
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        call: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let (native, format) = self
            .request(&request)
            .map_err(|error| self.contextualize(error))?;
        let generated = self
            .speech
            .synthesize_with_options(native, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        let audio = match generated.output() {
            MinimaxSpeechOutput::Audio(audio) => audio.clone(),
            MinimaxSpeechOutput::Url(_) => {
                return Err(self.contextualize(Error::new(
                    ErrorKind::Protocol,
                    "MiniMax portable speech received a URL instead of buffered audio",
                )));
            }
        };
        let info = generated.info();
        let duration_seconds = info
            .and_then(|info| info.duration_millis())
            .map(|duration| duration as f64 / 1_000.0);
        let sample_rate_hz = info
            .and_then(|info| info.sample_rate_hertz())
            .map(u32::try_from)
            .transpose()
            .map_err(|source| {
                self.contextualize(
                    Error::new(
                        ErrorKind::Protocol,
                        "MiniMax speech sample rate exceeds the portable range",
                    )
                    .with_source(source),
                )
            })?;
        let mut usage = Usage::default();
        if let Some(characters) = info.and_then(|info| info.usage_characters()) {
            usage = usage.with_provider_value("characters", characters);
        }
        let response = SpeechResponse {
            media_type: speech_media_type(format).to_string(),
            audio: audio.into(),
            duration_seconds,
            sample_rate_hz,
            metadata: ResponseMetadata {
                response_id: None,
                request_id: generated.trace_id().map(str::to_owned),
                model: Some(self.model_id().clone()),
            },
            usage,
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response
            .validate()
            .map_err(|error| self.contextualize(error))?;
        Ok(response)
    }
}

fn speech_format(format: Option<&str>) -> Result<MinimaxSpeechAudioFormat, Error> {
    match format.map(str::to_ascii_lowercase).as_deref() {
        None | Some("mp3") | Some("audio/mpeg") => Ok(MinimaxSpeechAudioFormat::Mp3),
        Some("pcm") | Some("audio/pcm") => Ok(MinimaxSpeechAudioFormat::Pcm),
        Some("flac") | Some("audio/flac") => Ok(MinimaxSpeechAudioFormat::Flac),
        Some("wav") | Some("wave") | Some("audio/wav") | Some("audio/x-wav") => {
            Ok(MinimaxSpeechAudioFormat::Wav)
        }
        Some("pcmu") | Some("audio/pcmu") => Ok(MinimaxSpeechAudioFormat::PcmuRaw),
        Some("pcmu-wav") | Some("audio/pcmu-wav") => Ok(MinimaxSpeechAudioFormat::PcmuWav),
        Some("opus") | Some("audio/opus") | Some("audio/ogg") => Ok(MinimaxSpeechAudioFormat::Opus),
        Some(_) => Err(Error::new(
            ErrorKind::Unsupported,
            "MiniMax speech supports mp3, pcm, flac, wav, pcmu, pcmu-wav, or opus output",
        )),
    }
}

fn speech_media_type(format: MinimaxSpeechAudioFormat) -> &'static str {
    match format {
        MinimaxSpeechAudioFormat::Mp3 => "audio/mpeg",
        MinimaxSpeechAudioFormat::Pcm => "audio/pcm",
        MinimaxSpeechAudioFormat::Flac => "audio/flac",
        MinimaxSpeechAudioFormat::Wav => "audio/wav",
        MinimaxSpeechAudioFormat::PcmuRaw => "audio/pcmu",
        MinimaxSpeechAudioFormat::PcmuWav => "audio/wav",
        MinimaxSpeechAudioFormat::Opus => "audio/opus",
    }
}
