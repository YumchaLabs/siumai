//! Google Provider-Defined Tools
//!
//! Factory helpers for creating Google/Gemini-specific provider-defined tools.
//! These tools are executed by Google's servers.
//!
//! Vercel AI SDK alignment:
//! - Use provider-defined tools for provider-hosted capabilities (search, file search, code execution).
//! - Prefer the stable tool IDs:
//!   - `google.google_search`
//!   - `google.file_search`
//!   - `google.code_execution`
//!   - `google.url_context`
//!   - `google.enterprise_web_search`
//!   - `google.google_maps`
//!   - `google.vertex_rag_store`

use crate::types::{ProviderDefinedTool, Tool};

/// Create a code execution tool.
///
/// This tool allows the model to execute Python code in a sandboxed environment.
pub fn code_execution() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.code_execution",
        "code_execution",
    ))
}

/// Google Search configuration builder.
#[derive(Debug, Clone, Default)]
pub struct GoogleSearchConfig {
    mode: Option<String>,
    dynamic_threshold: Option<f32>,
}

impl GoogleSearchConfig {
    /// Create a new Google Search configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set dynamic retrieval mode for Gemini 1.5 Google Search Retrieval (legacy).
    ///
    /// Examples:
    /// - `"MODE_DYNAMIC"`
    /// - `"MODE_UNSPECIFIED"`
    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = Some(mode.into());
        self
    }

    /// Set dynamic threshold for Gemini 1.5 Google Search Retrieval (legacy).
    pub fn with_dynamic_threshold(mut self, threshold: f32) -> Self {
        self.dynamic_threshold = Some(threshold);
        self
    }

    /// Build the Tool.
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(mode) = self.mode {
            args["mode"] = serde_json::json!(mode);
        }
        if let Some(th) = self.dynamic_threshold {
            args["dynamicThreshold"] = serde_json::json!(th);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("google.google_search", "google_search").with_args(args),
        )
    }
}

/// Create a Google Search tool with default settings.
pub fn google_search() -> GoogleSearchConfig {
    GoogleSearchConfig::new()
}

/// File Search configuration builder.
#[derive(Debug, Clone, Default)]
pub struct FileSearchConfig {
    file_search_store_names: Option<Vec<String>>,
    top_k: Option<u32>,
    metadata_filter: Option<String>,
}

impl FileSearchConfig {
    /// Create a new File Search configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the File Search store names (e.g., `fileSearchStores/my-store-123`).
    pub fn with_file_search_store_names(mut self, store_names: Vec<String>) -> Self {
        self.file_search_store_names = Some(store_names);
        self
    }

    /// Set top-K results to retrieve.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set metadata filter expression (provider-specific; AIP-160 string filter).
    pub fn with_metadata_filter(mut self, filter: impl Into<String>) -> Self {
        self.metadata_filter = Some(filter.into());
        self
    }

    /// Build the Tool.
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({});

        if let Some(names) = self.file_search_store_names {
            args["fileSearchStoreNames"] = serde_json::json!(names);
        }
        if let Some(top_k) = self.top_k {
            args["topK"] = serde_json::json!(top_k);
        }
        if let Some(filter) = self.metadata_filter {
            args["metadataFilter"] = serde_json::json!(filter);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("google.file_search", "file_search").with_args(args),
        )
    }
}

/// Create a File Search tool configuration.
pub fn file_search() -> FileSearchConfig {
    FileSearchConfig::new()
}

/// Create a URL context tool (Gemini 2.0+).
pub fn url_context() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.url_context",
        "url_context",
    ))
}

/// Create an Enterprise Web Search tool (Gemini 2.0+).
pub fn enterprise_web_search() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.enterprise_web_search",
        "enterprise_web_search",
    ))
}

/// Create a Google Maps grounding tool (Gemini 2.0+).
pub fn google_maps() -> Tool {
    Tool::ProviderDefined(ProviderDefinedTool::new(
        "google.google_maps",
        "google_maps",
    ))
}

/// Vertex RAG Store configuration builder.
///
/// Vercel AI SDK tool id: `google.vertex_rag_store`.
#[derive(Debug, Clone)]
pub struct VertexRagStoreConfig {
    rag_corpus: String,
    top_k: Option<u32>,
}

impl VertexRagStoreConfig {
    /// Create a new Vertex RAG Store configuration.
    pub fn new(rag_corpus: impl Into<String>) -> Self {
        Self {
            rag_corpus: rag_corpus.into(),
            top_k: None,
        }
    }

    /// Set top-K results to retrieve.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Build the Tool.
    pub fn build(self) -> Tool {
        let mut args = serde_json::json!({
            "ragCorpus": self.rag_corpus,
        });

        if let Some(top_k) = self.top_k {
            args["topK"] = serde_json::json!(top_k);
        }

        Tool::ProviderDefined(
            ProviderDefinedTool::new("google.vertex_rag_store", "vertex_rag_store").with_args(args),
        )
    }
}

/// Create a Vertex RAG Store tool configuration (Vertex Gemini only).
pub fn vertex_rag_store(rag_corpus: impl Into<String>) -> VertexRagStoreConfig {
    VertexRagStoreConfig::new(rag_corpus)
}
