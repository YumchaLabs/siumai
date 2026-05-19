//! Google/Gemini provider-defined tool catalog.
//!
//! Canonical Google/Gemini hosted-tool IDs, default names, and direct `Tool` constructors live here,
//! outside `siumai-spec`, because they are protocol/provider-owned facts.

pub mod google {
    use siumai_core::types::Tool;

    /// Mapping of provider tool ids to Google Gemini tool names (provider-native).
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (CODE_EXECUTION_ID, "code_execution"),
        (GOOGLE_SEARCH_ID, "google_search"),
        (GOOGLE_SEARCH_RETRIEVAL_ID, "google_search_retrieval"),
        (URL_CONTEXT_ID, "url_context"),
        (ENTERPRISE_WEB_SEARCH_ID, "enterprise_web_search"),
        (GOOGLE_MAPS_ID, "google_maps"),
        (VERTEX_RAG_STORE_ID, "vertex_rag_store"),
        (FILE_SEARCH_ID, "file_search"),
    ];

    pub const CODE_EXECUTION_ID: &str = "google.code_execution";
    pub const GOOGLE_SEARCH_ID: &str = "google.google_search";
    pub const GOOGLE_SEARCH_RETRIEVAL_ID: &str = "google.google_search_retrieval";
    pub const URL_CONTEXT_ID: &str = "google.url_context";
    pub const ENTERPRISE_WEB_SEARCH_ID: &str = "google.enterprise_web_search";
    pub const GOOGLE_MAPS_ID: &str = "google.google_maps";
    pub const VERTEX_RAG_STORE_ID: &str = "google.vertex_rag_store";
    pub const FILE_SEARCH_ID: &str = "google.file_search";

    pub fn code_execution() -> Tool {
        code_execution_named("code_execution")
    }

    pub fn code_execution_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_ID, name)
    }

    pub fn google_search() -> Tool {
        google_search_named("google_search")
    }

    pub fn google_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(GOOGLE_SEARCH_ID, name)
    }

    pub fn google_search_retrieval() -> Tool {
        google_search_retrieval_named("google_search_retrieval")
    }

    pub fn google_search_retrieval_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(GOOGLE_SEARCH_RETRIEVAL_ID, name)
    }

    pub fn url_context() -> Tool {
        url_context_named("url_context")
    }

    pub fn url_context_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(URL_CONTEXT_ID, name)
    }

    pub fn enterprise_web_search() -> Tool {
        enterprise_web_search_named("enterprise_web_search")
    }

    pub fn enterprise_web_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(ENTERPRISE_WEB_SEARCH_ID, name)
    }

    pub fn google_maps() -> Tool {
        google_maps_named("google_maps")
    }

    pub fn google_maps_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(GOOGLE_MAPS_ID, name)
    }

    pub fn vertex_rag_store(rag_corpus: impl Into<String>) -> Tool {
        vertex_rag_store_named(rag_corpus, "vertex_rag_store")
    }

    pub fn vertex_rag_store_named(rag_corpus: impl Into<String>, name: impl Into<String>) -> Tool {
        Tool::provider_defined(VERTEX_RAG_STORE_ID, name).with_args(serde_json::json!({
            "ragCorpus": rag_corpus.into(),
        }))
    }

    pub fn file_search(file_search_store_names: Vec<String>) -> Tool {
        file_search_named(file_search_store_names, "file_search")
    }

    pub fn file_search_named(
        file_search_store_names: Vec<String>,
        name: impl Into<String>,
    ) -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, name).with_args(serde_json::json!({
            "fileSearchStoreNames": file_search_store_names,
        }))
    }
    /// Create a provider-defined Google/Gemini tool by stable tool id when no required args are needed.
    pub fn provider_defined_tool(id: &str) -> Option<Tool> {
        match id {
            CODE_EXECUTION_ID => Some(code_execution()),
            GOOGLE_SEARCH_ID => Some(google_search()),
            GOOGLE_SEARCH_RETRIEVAL_ID => Some(google_search_retrieval()),
            URL_CONTEXT_ID => Some(url_context()),
            ENTERPRISE_WEB_SEARCH_ID => Some(enterprise_web_search()),
            GOOGLE_MAPS_ID => Some(google_maps()),
            _ => None,
        }
    }
}
