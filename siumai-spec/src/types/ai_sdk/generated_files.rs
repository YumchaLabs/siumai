use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// AI SDK-style generated file returned by text helper output parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneratedFile {
    /// File content as a base64 encoded string.
    pub base64: String,
    /// IANA media type of the file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
}

impl GeneratedFile {
    /// Create a generated file from base64 content.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            base64: base64.into(),
            media_type: media_type.into(),
        }
    }

    /// Create a generated file from bytes.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        Self::from_base64(STANDARD.encode(data.as_ref()), media_type)
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.base64.as_str()
    }

    /// Decode the file into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        STANDARD.decode(&self.base64)
    }
}

/// AI SDK `DefaultGeneratedFile` export. Rust keeps the same value carrier as `GeneratedFile`.
pub type DefaultGeneratedFile = GeneratedFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DefaultGeneratedFileWithTypeMarker {
    #[default]
    File,
}

impl Serialize for DefaultGeneratedFileWithTypeMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("file")
    }
}

impl<'de> Deserialize<'de> for DefaultGeneratedFileWithTypeMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "file" {
            Ok(Self::File)
        } else {
            Err(serde::de::Error::custom("expected file type marker"))
        }
    }
}

/// Passive AI SDK `DefaultGeneratedFileWithType` carrier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DefaultGeneratedFileWithType {
    #[serde(rename = "type", default)]
    marker: DefaultGeneratedFileWithTypeMarker,
    /// Generated file payload.
    #[serde(flatten)]
    pub file: GeneratedFile,
}

impl DefaultGeneratedFileWithType {
    /// Create a generated file with the AI SDK `type: "file"` discriminator.
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: DefaultGeneratedFileWithTypeMarker::File,
            file,
        }
    }

    /// Create a generated file from base64 content.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::new(GeneratedFile::from_base64(base64, media_type))
    }

    /// Create a generated file from bytes.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        Self::new(GeneratedFile::from_bytes(data, media_type))
    }

    /// Return the AI SDK output discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.file.base64()
    }

    /// Return the generated file media type.
    pub fn media_type(&self) -> &str {
        self.file.media_type.as_str()
    }

    /// Decode the generated file into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.file.uint8_array()
    }
}

/// Backwards-compatible AI SDK `Experimental_GeneratedImage` export.
#[allow(non_camel_case_types)]
pub type Experimental_GeneratedImage = GeneratedFile;

fn audio_format_from_media_type(media_type: &str) -> String {
    let normalized = media_type.trim().to_ascii_lowercase();
    if normalized == "audio/mpeg" {
        return "mp3".to_string();
    }

    normalized
        .split_once('/')
        .map(|(_, subtype)| subtype.to_string())
        .filter(|subtype| !subtype.is_empty())
        .unwrap_or_else(|| "mp3".to_string())
}

/// AI SDK-style generated audio file returned by `generateSpeech`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedAudioFile {
    /// File content and media type.
    #[serde(flatten)]
    pub file: GeneratedFile,
    /// Audio format such as `mp3` or `wav`.
    pub format: String,
}

impl GeneratedAudioFile {
    /// Create generated audio from a generated file and explicit format.
    pub fn new(file: GeneratedFile, format: impl Into<String>) -> Self {
        Self {
            file,
            format: format.into(),
        }
    }

    /// Create generated audio from base64 content, deriving the format from the media type.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        Self::new(
            GeneratedFile::from_base64(base64, media_type.as_str()),
            audio_format_from_media_type(&media_type),
        )
    }

    /// Create generated audio from bytes, deriving the format from the media type.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        Self::new(
            GeneratedFile::from_bytes(data, media_type.as_str()),
            audio_format_from_media_type(&media_type),
        )
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.file.base64()
    }

    /// Return the generated audio media type.
    pub fn media_type(&self) -> &str {
        self.file.media_type.as_str()
    }

    /// Decode the generated audio into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.file.uint8_array()
    }
}

/// AI SDK `DefaultGeneratedAudioFile` export. Rust keeps the same value carrier as `GeneratedAudioFile`.
pub type DefaultGeneratedAudioFile = GeneratedAudioFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DefaultGeneratedAudioFileWithTypeMarker {
    #[default]
    Audio,
}

impl Serialize for DefaultGeneratedAudioFileWithTypeMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("audio")
    }
}

impl<'de> Deserialize<'de> for DefaultGeneratedAudioFileWithTypeMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "audio" {
            Ok(Self::Audio)
        } else {
            Err(serde::de::Error::custom("expected audio type marker"))
        }
    }
}

/// Passive AI SDK `DefaultGeneratedAudioFileWithType` carrier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DefaultGeneratedAudioFileWithType {
    #[serde(rename = "type", default)]
    marker: DefaultGeneratedAudioFileWithTypeMarker,
    /// Generated audio payload.
    #[serde(flatten)]
    pub audio: GeneratedAudioFile,
}

impl DefaultGeneratedAudioFileWithType {
    /// Create generated audio with the AI SDK `type: "audio"` discriminator.
    pub fn new(audio: GeneratedAudioFile) -> Self {
        Self {
            marker: DefaultGeneratedAudioFileWithTypeMarker::Audio,
            audio,
        }
    }

    /// Create generated audio from base64 content, deriving the format from the media type.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::new(GeneratedAudioFile::from_base64(base64, media_type))
    }

    /// Create generated audio from bytes, deriving the format from the media type.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        Self::new(GeneratedAudioFile::from_bytes(data, media_type))
    }

    /// Return the AI SDK output discriminator.
    pub const fn r#type(&self) -> &'static str {
        "audio"
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.audio.base64()
    }

    /// Return the generated audio media type.
    pub fn media_type(&self) -> &str {
        self.audio.media_type()
    }

    /// Decode the generated audio into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.audio.uint8_array()
    }
}
