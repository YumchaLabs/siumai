//! ProviderCapabilities structure

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub audio: bool,
    pub vision: bool,
    pub tools: bool,
    pub embedding: bool,
    pub streaming: bool,
    pub file_management: bool,
    pub custom_features: HashMap<String, bool>,
}

impl ProviderCapabilities {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chat(mut self) -> Self {
        self.chat = true;
        self
    }
    pub fn with_audio(mut self) -> Self {
        self.audio = true;
        self
    }
    pub fn with_vision(mut self) -> Self {
        self.vision = true;
        self
    }
    pub fn with_tools(mut self) -> Self {
        self.tools = true;
        self
    }
    pub fn with_embedding(mut self) -> Self {
        self.embedding = true;
        self
    }
    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }
    pub fn with_file_management(mut self) -> Self {
        self.file_management = true;
        self
    }
    pub fn with_custom_feature(mut self, name: &str, enabled: bool) -> Self {
        self.custom_features.insert(name.to_string(), enabled);
        self
    }

    pub fn supports(&self, feature: &str) -> bool {
        match feature {
            "chat" => self.chat,
            "audio" => self.audio,
            "vision" => self.vision,
            "tools" => self.tools,
            "embedding" => self.embedding,
            "streaming" => self.streaming,
            "file_management" => self.file_management,
            _ => self.custom_features.get(feature).copied().unwrap_or(false),
        }
    }
}
