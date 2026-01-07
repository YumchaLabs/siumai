//! Provider-defined tool factories (Vercel-aligned).
//!
//! This module provides small helpers to construct `Tool::ProviderDefined` values with
//! stable IDs (`provider.tool_type`) and default names. The underlying serialized shape
//! matches the Vercel AI SDK `{ type: "provider", id, name, args }` convention.

use siumai_core::types::Tool;

/// OpenAI provider-defined tools.
pub mod openai {
    use super::Tool;

    pub const WEB_SEARCH_ID: &str = "openai.web_search";
    pub const WEB_SEARCH_PREVIEW_ID: &str = "openai.web_search_preview";
    pub const FILE_SEARCH_ID: &str = "openai.file_search";
    pub const CODE_INTERPRETER_ID: &str = "openai.code_interpreter";
    pub const IMAGE_GENERATION_ID: &str = "openai.image_generation";
    pub const COMPUTER_USE_ID: &str = "openai.computer_use";
    pub const MCP_ID: &str = "openai.mcp";
    pub const APPLY_PATCH_ID: &str = "openai.apply_patch";

    pub fn web_search() -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, "web_search")
    }

    pub fn web_search_preview() -> Tool {
        Tool::provider_defined(WEB_SEARCH_PREVIEW_ID, "web_search_preview")
    }

    pub fn file_search() -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, "file_search")
    }

    pub fn code_interpreter() -> Tool {
        Tool::provider_defined(CODE_INTERPRETER_ID, "code_interpreter")
    }

    pub fn image_generation() -> Tool {
        Tool::provider_defined(IMAGE_GENERATION_ID, "image_generation")
    }

    pub fn computer_use() -> Tool {
        Tool::provider_defined(COMPUTER_USE_ID, "computer_use")
    }

    pub fn mcp() -> Tool {
        Tool::provider_defined(MCP_ID, "mcp")
    }

    pub fn apply_patch() -> Tool {
        Tool::provider_defined(APPLY_PATCH_ID, "apply_patch")
    }
}

/// Anthropic provider-defined tools.
pub mod anthropic {
    use super::Tool;

    pub const WEB_SEARCH_20250305_ID: &str = "anthropic.web_search_20250305";
    pub const WEB_FETCH_20250910_ID: &str = "anthropic.web_fetch_20250910";
    pub const TOOL_SEARCH_REGEX_20251119_ID: &str = "anthropic.tool_search_regex_20251119";
    pub const TOOL_SEARCH_BM25_20251119_ID: &str = "anthropic.tool_search_bm25_20251119";
    pub const CODE_EXECUTION_20250522_ID: &str = "anthropic.code_execution_20250522";
    pub const CODE_EXECUTION_20250825_ID: &str = "anthropic.code_execution_20250825";
    pub const MEMORY_20250818_ID: &str = "anthropic.memory_20250818";

    pub fn web_search_20250305() -> Tool {
        Tool::provider_defined(WEB_SEARCH_20250305_ID, "web_search")
    }

    pub fn web_search() -> Tool {
        web_search_20250305()
    }

    pub fn web_fetch_20250910() -> Tool {
        Tool::provider_defined(WEB_FETCH_20250910_ID, "web_fetch")
    }

    pub fn tool_search_regex_20251119() -> Tool {
        Tool::provider_defined(TOOL_SEARCH_REGEX_20251119_ID, "tool_search")
    }

    pub fn tool_search_bm25_20251119() -> Tool {
        Tool::provider_defined(TOOL_SEARCH_BM25_20251119_ID, "tool_search")
    }

    pub fn code_execution_20250522() -> Tool {
        Tool::provider_defined(CODE_EXECUTION_20250522_ID, "code_execution")
    }

    pub fn code_execution_20250825() -> Tool {
        Tool::provider_defined(CODE_EXECUTION_20250825_ID, "code_execution")
    }

    pub fn memory_20250818() -> Tool {
        Tool::provider_defined(MEMORY_20250818_ID, "memory")
    }
}

/// Google (Gemini) provider-defined tools.
pub mod google {
    use super::Tool;

    pub const CODE_EXECUTION_ID: &str = "google.code_execution";
    pub const GOOGLE_SEARCH_ID: &str = "google.google_search";
    pub const GOOGLE_SEARCH_RETRIEVAL_ID: &str = "google.search_retrieval";
    pub const URL_CONTEXT_ID: &str = "google.url_context";
    pub const ENTERPRISE_WEB_SEARCH_ID: &str = "google.enterprise_web_search";
    pub const GOOGLE_MAPS_ID: &str = "google.google_maps";
    pub const VERTEX_RAG_STORE_ID: &str = "google.vertex_rag_store";
    pub const FILE_SEARCH_ID: &str = "google.file_search";

    pub fn code_execution() -> Tool {
        Tool::provider_defined(CODE_EXECUTION_ID, "code_execution")
    }

    pub fn google_search() -> Tool {
        Tool::provider_defined(GOOGLE_SEARCH_ID, "google_search")
    }

    pub fn search_retrieval() -> Tool {
        Tool::provider_defined(GOOGLE_SEARCH_RETRIEVAL_ID, "search_retrieval")
    }

    pub fn url_context() -> Tool {
        Tool::provider_defined(URL_CONTEXT_ID, "url_context")
    }

    pub fn enterprise_web_search() -> Tool {
        Tool::provider_defined(ENTERPRISE_WEB_SEARCH_ID, "enterprise_web_search")
    }

    pub fn google_maps() -> Tool {
        Tool::provider_defined(GOOGLE_MAPS_ID, "google_maps")
    }

    pub fn vertex_rag_store() -> Tool {
        Tool::provider_defined(VERTEX_RAG_STORE_ID, "vertex_rag_store")
    }

    pub fn file_search() -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, "file_search")
    }
}

/// xAI provider-defined tools (OpenAI-like family).
pub mod xai {
    use super::Tool;

    pub const WEB_SEARCH_ID: &str = "xai.web_search";
    pub const X_SEARCH_ID: &str = "xai.x_search";
    pub const CODE_EXECUTION_ID: &str = "xai.code_execution";

    pub fn web_search() -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, "web_search")
    }

    pub fn x_search() -> Tool {
        Tool::provider_defined(X_SEARCH_ID, "x_search")
    }

    pub fn code_execution() -> Tool {
        Tool::provider_defined(CODE_EXECUTION_ID, "code_execution")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::types::Tool as ToolEnum;

    #[test]
    fn openai_web_search_tool_has_expected_id_and_name() {
        let t = openai::web_search();
        let ToolEnum::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::WEB_SEARCH_ID);
        assert_eq!(pd.name, "web_search");
    }

    #[test]
    fn anthropic_web_fetch_tool_has_expected_id_and_name() {
        let t = anthropic::web_fetch_20250910();
        let ToolEnum::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, anthropic::WEB_FETCH_20250910_ID);
        assert_eq!(pd.name, "web_fetch");
    }

    #[test]
    fn google_google_search_tool_has_expected_id_and_name() {
        let t = google::google_search();
        let ToolEnum::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, google::GOOGLE_SEARCH_ID);
        assert_eq!(pd.name, "google_search");
    }

    #[test]
    fn xai_code_execution_tool_has_expected_id_and_name() {
        let t = xai::code_execution();
        let ToolEnum::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, xai::CODE_EXECUTION_ID);
        assert_eq!(pd.name, "code_execution");
    }
}
