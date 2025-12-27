//! Ollama protocol parameters
//!
//! These parameters are used by request builders/transformers to populate the
//! Ollama request JSON. Provider builders may expose typed setters for these.

/// Ollama-specific parameters
#[derive(Debug, Clone, Default)]
pub struct OllamaParams {
    /// Keep model loaded in memory for this duration (default: 5m)
    pub keep_alive: Option<String>,
    /// Use raw mode (bypass templating)
    pub raw: Option<bool>,
    /// Format for structured outputs (json or schema)
    pub format: Option<String>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Enable/disable NUMA support
    pub numa: Option<bool>,
    /// Context window size
    pub num_ctx: Option<u32>,
    /// Batch size for processing
    pub num_batch: Option<u32>,
    /// Number of GPU layers to use
    pub num_gpu: Option<u32>,
    /// Main GPU to use
    pub main_gpu: Option<u32>,
    /// Use memory mapping
    pub use_mmap: Option<bool>,
    /// Number of threads to use
    pub num_thread: Option<u32>,
    /// Should the model think before responding (for thinking models)
    pub think: Option<bool>,
    /// Additional model options
    pub options: Option<std::collections::HashMap<String, serde_json::Value>>,
}
