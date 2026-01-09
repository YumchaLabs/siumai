//! Provider-defined tool factories (Vercel-aligned).
//!
//! This module provides small helpers to construct `Tool::ProviderDefined` values with
//! stable IDs (`provider.tool_type`) and default names. The underlying serialized shape
//! matches the Vercel AI SDK `{ type: "provider", id, name, args }` convention.

use crate::types::Tool;

/// Create a provider-defined tool with a Vercel-aligned default custom name by tool id.
///
/// This is useful when you only have a string id (e.g. config-driven), but still want to
/// reuse the same default custom tool names as the convenience constructors in this module.
pub fn provider_defined_tool(id: &str) -> Option<Tool> {
    match id {
        // OpenAI
        openai::WEB_SEARCH_ID => Some(openai::web_search()),
        openai::WEB_SEARCH_PREVIEW_ID => Some(openai::web_search_preview()),
        openai::FILE_SEARCH_ID => Some(openai::file_search()),
        openai::CODE_INTERPRETER_ID => Some(openai::code_interpreter()),
        openai::IMAGE_GENERATION_ID => Some(openai::image_generation()),
        openai::LOCAL_SHELL_ID => Some(openai::local_shell()),
        openai::SHELL_ID => Some(openai::shell()),
        openai::COMPUTER_USE_ID => Some(openai::computer_use()),
        openai::MCP_ID => Some(openai::mcp()),
        openai::APPLY_PATCH_ID => Some(openai::apply_patch()),

        // Anthropic
        anthropic::WEB_SEARCH_20250305_ID => Some(anthropic::web_search_20250305()),
        anthropic::WEB_FETCH_20250910_ID => Some(anthropic::web_fetch_20250910()),
        anthropic::COMPUTER_20250124_ID => Some(anthropic::computer_20250124()),
        anthropic::COMPUTER_20241022_ID => Some(anthropic::computer_20241022()),
        anthropic::TEXT_EDITOR_20250124_ID => Some(anthropic::text_editor_20250124()),
        anthropic::TEXT_EDITOR_20241022_ID => Some(anthropic::text_editor_20241022()),
        anthropic::TEXT_EDITOR_20250429_ID => Some(anthropic::text_editor_20250429()),
        anthropic::TEXT_EDITOR_20250728_ID => Some(anthropic::text_editor_20250728()),
        anthropic::BASH_20241022_ID => Some(anthropic::bash_20241022()),
        anthropic::BASH_20250124_ID => Some(anthropic::bash_20250124()),
        anthropic::TOOL_SEARCH_REGEX_20251119_ID => Some(anthropic::tool_search_regex_20251119()),
        anthropic::TOOL_SEARCH_BM25_20251119_ID => Some(anthropic::tool_search_bm25_20251119()),
        anthropic::CODE_EXECUTION_20250522_ID => Some(anthropic::code_execution_20250522()),
        anthropic::CODE_EXECUTION_20250825_ID => Some(anthropic::code_execution_20250825()),
        anthropic::MEMORY_20250818_ID => Some(anthropic::memory_20250818()),

        // Google (Gemini)
        google::CODE_EXECUTION_ID => Some(google::code_execution()),
        google::GOOGLE_SEARCH_ID => Some(google::google_search()),
        google::GOOGLE_SEARCH_RETRIEVAL_ID => Some(google::google_search_retrieval()),
        google::URL_CONTEXT_ID => Some(google::url_context()),
        google::ENTERPRISE_WEB_SEARCH_ID => Some(google::enterprise_web_search()),
        google::GOOGLE_MAPS_ID => Some(google::google_maps()),
        google::VERTEX_RAG_STORE_ID => Some(google::vertex_rag_store()),
        google::FILE_SEARCH_ID => Some(google::file_search()),

        // xAI
        xai::WEB_SEARCH_ID => Some(xai::web_search()),
        xai::X_SEARCH_ID => Some(xai::x_search()),
        xai::CODE_EXECUTION_ID => Some(xai::code_execution()),

        _ => None,
    }
}

/// OpenAI provider-defined tools.
pub mod openai {
    use super::Tool;

    /// Mapping of provider tool ids to OpenAI Responses tool names (provider-native).
    ///
    /// This is primarily used to map custom tool names back to provider tool names when
    /// serializing certain tool call / tool result message items (Vercel-aligned).
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (WEB_SEARCH_ID, "web_search"),
        (WEB_SEARCH_PREVIEW_ID, "web_search_preview"),
        (FILE_SEARCH_ID, "file_search"),
        (CODE_INTERPRETER_ID, "code_interpreter"),
        (IMAGE_GENERATION_ID, "image_generation"),
        (LOCAL_SHELL_ID, "local_shell"),
        (SHELL_ID, "shell"),
        (COMPUTER_USE_ID, "computer_use"),
        (MCP_ID, "mcp"),
        (APPLY_PATCH_ID, "apply_patch"),
    ];

    /// OpenAI Responses built-in tool types that we support for `tool_choice`.
    ///
    /// Note: `openai.computer_use` maps to the Responses API tool type `computer_use_preview`.
    pub const RESPONSES_BUILTIN_TOOL_TYPES: &[&str] = &[
        "code_interpreter",
        "file_search",
        "image_generation",
        "web_search_preview",
        "web_search",
        "mcp",
        "apply_patch",
        "computer_use_preview",
    ];

    /// Convert a stable provider tool type (from `openai.<tool_type>`) into a Responses API tool type.
    ///
    /// Returns `None` for unsupported tool types.
    pub fn responses_builtin_type_for_tool_type(tool_type: &str) -> Option<&'static str> {
        match tool_type {
            // Vercel alignment: `openai.computer_use` is sent as `computer_use_preview`.
            "computer_use" => Some("computer_use_preview"),
            "code_interpreter" => Some("code_interpreter"),
            "file_search" => Some("file_search"),
            "image_generation" => Some("image_generation"),
            "web_search_preview" => Some("web_search_preview"),
            "web_search" => Some("web_search"),
            "mcp" => Some("mcp"),
            "apply_patch" => Some("apply_patch"),
            _ => None,
        }
    }

    /// Convert a tool choice name into a Responses API built-in tool type.
    ///
    /// This supports both:
    /// - built-in type names (e.g. `web_search`)
    /// - a compatibility alias (`computer_use` -> `computer_use_preview`)
    pub fn responses_builtin_type_for_choice_name(name: &str) -> Option<&'static str> {
        match name {
            "computer_use" => Some("computer_use_preview"),
            "computer_use_preview" => Some("computer_use_preview"),
            "code_interpreter" => Some("code_interpreter"),
            "file_search" => Some("file_search"),
            "image_generation" => Some("image_generation"),
            "web_search_preview" => Some("web_search_preview"),
            "web_search" => Some("web_search"),
            "mcp" => Some("mcp"),
            "apply_patch" => Some("apply_patch"),
            _ => None,
        }
    }

    pub const WEB_SEARCH_ID: &str = "openai.web_search";
    pub const WEB_SEARCH_PREVIEW_ID: &str = "openai.web_search_preview";
    pub const FILE_SEARCH_ID: &str = "openai.file_search";
    pub const CODE_INTERPRETER_ID: &str = "openai.code_interpreter";
    pub const IMAGE_GENERATION_ID: &str = "openai.image_generation";
    pub const LOCAL_SHELL_ID: &str = "openai.local_shell";
    pub const SHELL_ID: &str = "openai.shell";
    pub const COMPUTER_USE_ID: &str = "openai.computer_use";
    pub const MCP_ID: &str = "openai.mcp";
    pub const APPLY_PATCH_ID: &str = "openai.apply_patch";

    pub fn web_search() -> Tool {
        // Vercel AI SDK default key: `webSearch`
        web_search_named("webSearch")
    }

    pub fn web_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, name)
    }

    pub fn web_search_preview() -> Tool {
        web_search_preview_named("web_search_preview")
    }

    pub fn web_search_preview_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_PREVIEW_ID, name)
    }

    pub fn file_search() -> Tool {
        // Vercel AI SDK default key: `fileSearch`
        file_search_named("fileSearch")
    }

    pub fn file_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, name)
    }

    pub fn code_interpreter() -> Tool {
        // Vercel fixtures commonly use `codeExecution` as the custom tool name.
        code_interpreter_named("codeExecution")
    }

    pub fn code_interpreter_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_INTERPRETER_ID, name)
    }

    pub fn image_generation() -> Tool {
        // Vercel fixtures commonly use `generateImage` as the custom tool name.
        image_generation_named("generateImage")
    }

    pub fn image_generation_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(IMAGE_GENERATION_ID, name)
    }

    pub fn local_shell() -> Tool {
        local_shell_named("shell")
    }

    pub fn local_shell_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(LOCAL_SHELL_ID, name)
    }

    pub fn shell() -> Tool {
        shell_named("shell")
    }

    pub fn shell_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(SHELL_ID, name)
    }

    pub fn computer_use() -> Tool {
        computer_use_named("computer_use")
    }

    pub fn computer_use_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(COMPUTER_USE_ID, name)
    }

    pub fn mcp() -> Tool {
        // Vercel fixtures commonly use `MCP` as the custom tool name.
        mcp_named("MCP")
    }

    pub fn mcp_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(MCP_ID, name)
    }

    pub fn apply_patch() -> Tool {
        apply_patch_named("apply_patch")
    }

    pub fn apply_patch_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(APPLY_PATCH_ID, name)
    }
}

/// Anthropic provider-defined tools.
pub mod anthropic {
    use super::Tool;

    /// Mapping of provider tool ids to Anthropic tool names (provider-native).
    ///
    /// Note: Anthropic "versioned tools" (e.g. `web_search_20250305`) still map to
    /// unversioned provider-native names (e.g. `web_search`) in request/response surfaces.
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (WEB_SEARCH_20250305_ID, "web_search"),
        (WEB_FETCH_20250910_ID, "web_fetch"),
        (COMPUTER_20250124_ID, "computer"),
        (COMPUTER_20241022_ID, "computer"),
        (TEXT_EDITOR_20250124_ID, "str_replace_editor"),
        (TEXT_EDITOR_20241022_ID, "str_replace_editor"),
        (TEXT_EDITOR_20250429_ID, "str_replace_based_edit_tool"),
        (TEXT_EDITOR_20250728_ID, "str_replace_based_edit_tool"),
        (BASH_20241022_ID, "bash"),
        (BASH_20250124_ID, "bash"),
        (TOOL_SEARCH_REGEX_20251119_ID, "tool_search"),
        (TOOL_SEARCH_BM25_20251119_ID, "tool_search"),
        (CODE_EXECUTION_20250522_ID, "code_execution"),
        (CODE_EXECUTION_20250825_ID, "code_execution"),
        (MEMORY_20250818_ID, "memory"),
    ];

    pub const WEB_SEARCH_20250305_ID: &str = "anthropic.web_search_20250305";
    pub const WEB_FETCH_20250910_ID: &str = "anthropic.web_fetch_20250910";
    pub const COMPUTER_20250124_ID: &str = "anthropic.computer_20250124";
    pub const COMPUTER_20241022_ID: &str = "anthropic.computer_20241022";
    pub const TEXT_EDITOR_20250124_ID: &str = "anthropic.text_editor_20250124";
    pub const TEXT_EDITOR_20241022_ID: &str = "anthropic.text_editor_20241022";
    pub const BASH_20241022_ID: &str = "anthropic.bash_20241022";
    pub const BASH_20250124_ID: &str = "anthropic.bash_20250124";
    pub const TEXT_EDITOR_20250429_ID: &str = "anthropic.text_editor_20250429";
    pub const TEXT_EDITOR_20250728_ID: &str = "anthropic.text_editor_20250728";
    pub const TOOL_SEARCH_REGEX_20251119_ID: &str = "anthropic.tool_search_regex_20251119";
    pub const TOOL_SEARCH_BM25_20251119_ID: &str = "anthropic.tool_search_bm25_20251119";
    pub const CODE_EXECUTION_20250522_ID: &str = "anthropic.code_execution_20250522";
    pub const CODE_EXECUTION_20250825_ID: &str = "anthropic.code_execution_20250825";
    pub const MEMORY_20250818_ID: &str = "anthropic.memory_20250818";

    pub fn web_search_20250305() -> Tool {
        web_search_20250305_named("web_search")
    }

    pub fn web_search_20250305_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_20250305_ID, name)
    }

    pub fn web_search() -> Tool {
        web_search_20250305()
    }

    pub fn web_fetch_20250910() -> Tool {
        web_fetch_20250910_named("web_fetch")
    }

    pub fn web_fetch_20250910_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_FETCH_20250910_ID, name)
    }

    pub fn computer_20250124() -> Tool {
        computer_20250124_named("computer")
    }

    pub fn computer_20250124_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(COMPUTER_20250124_ID, name)
    }

    pub fn computer_20241022() -> Tool {
        computer_20241022_named("computer")
    }

    pub fn computer_20241022_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(COMPUTER_20241022_ID, name)
    }

    pub fn text_editor_20250124() -> Tool {
        text_editor_20250124_named("str_replace_editor")
    }

    pub fn text_editor_20250124_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TEXT_EDITOR_20250124_ID, name)
    }

    pub fn text_editor_20241022() -> Tool {
        text_editor_20241022_named("str_replace_editor")
    }

    pub fn text_editor_20241022_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TEXT_EDITOR_20241022_ID, name)
    }

    pub fn text_editor_20250429() -> Tool {
        text_editor_20250429_named("str_replace_based_edit_tool")
    }

    pub fn text_editor_20250429_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TEXT_EDITOR_20250429_ID, name)
    }

    pub fn text_editor_20250728() -> Tool {
        text_editor_20250728_named("str_replace_based_edit_tool")
    }

    pub fn text_editor_20250728_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TEXT_EDITOR_20250728_ID, name)
    }

    pub fn bash_20241022() -> Tool {
        bash_20241022_named("bash")
    }

    pub fn bash_20241022_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(BASH_20241022_ID, name)
    }

    pub fn bash_20250124() -> Tool {
        bash_20250124_named("bash")
    }

    pub fn bash_20250124_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(BASH_20250124_ID, name)
    }

    pub fn tool_search_regex_20251119() -> Tool {
        tool_search_regex_20251119_named("tool_search")
    }

    pub fn tool_search_regex_20251119_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TOOL_SEARCH_REGEX_20251119_ID, name)
    }

    pub fn tool_search_bm25_20251119() -> Tool {
        tool_search_bm25_20251119_named("tool_search")
    }

    pub fn tool_search_bm25_20251119_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TOOL_SEARCH_BM25_20251119_ID, name)
    }

    pub fn code_execution_20250522() -> Tool {
        code_execution_20250522_named("code_execution")
    }

    pub fn code_execution_20250522_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_20250522_ID, name)
    }

    pub fn code_execution_20250825() -> Tool {
        code_execution_20250825_named("code_execution")
    }

    pub fn code_execution_20250825_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_20250825_ID, name)
    }

    pub fn memory_20250818() -> Tool {
        memory_20250818_named("memory")
    }

    pub fn memory_20250818_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(MEMORY_20250818_ID, name)
    }
}

/// Google (Gemini) provider-defined tools.
pub mod google {
    use super::Tool;

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

    pub fn vertex_rag_store() -> Tool {
        vertex_rag_store_named("vertex_rag_store")
    }

    pub fn vertex_rag_store_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(VERTEX_RAG_STORE_ID, name)
    }

    pub fn file_search() -> Tool {
        file_search_named("file_search")
    }

    pub fn file_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, name)
    }
}

/// xAI provider-defined tools (OpenAI-like family).
pub mod xai {
    use super::Tool;

    /// Mapping of provider tool ids to xAI tool names (provider-native).
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (WEB_SEARCH_ID, "web_search"),
        (X_SEARCH_ID, "x_search"),
        (CODE_EXECUTION_ID, "code_execution"),
    ];

    pub const WEB_SEARCH_ID: &str = "xai.web_search";
    pub const X_SEARCH_ID: &str = "xai.x_search";
    pub const CODE_EXECUTION_ID: &str = "xai.code_execution";

    pub fn web_search() -> Tool {
        web_search_named("web_search")
    }

    pub fn web_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, name)
    }

    pub fn x_search() -> Tool {
        x_search_named("x_search")
    }

    pub fn x_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(X_SEARCH_ID, name)
    }

    pub fn code_execution() -> Tool {
        code_execution_named("code_execution")
    }

    pub fn code_execution_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_ID, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_web_search_tool_has_expected_id_and_name() {
        let t = openai::web_search();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::WEB_SEARCH_ID);
        assert_eq!(pd.name, "webSearch");
    }

    #[test]
    fn openai_file_search_tool_has_expected_id_and_name() {
        let t = openai::file_search();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::FILE_SEARCH_ID);
        assert_eq!(pd.name, "fileSearch");
    }

    #[test]
    fn openai_code_interpreter_tool_has_expected_id_and_name() {
        let t = openai::code_interpreter();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::CODE_INTERPRETER_ID);
        assert_eq!(pd.name, "codeExecution");
    }

    #[test]
    fn openai_image_generation_tool_has_expected_id_and_name() {
        let t = openai::image_generation();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::IMAGE_GENERATION_ID);
        assert_eq!(pd.name, "generateImage");
    }

    #[test]
    fn openai_shell_tools_have_expected_ids_and_names() {
        for (tool, id) in [
            (openai::local_shell(), openai::LOCAL_SHELL_ID),
            (openai::shell(), openai::SHELL_ID),
        ] {
            let crate::types::Tool::ProviderDefined(pd) = tool else {
                panic!("expected provider-defined tool");
            };
            assert_eq!(pd.id, id);
            assert_eq!(pd.name, "shell");
        }
    }

    #[test]
    fn openai_mcp_tool_has_expected_id_and_name() {
        let t = openai::mcp();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::MCP_ID);
        assert_eq!(pd.name, "MCP");
    }

    #[test]
    fn anthropic_web_fetch_tool_has_expected_id_and_name() {
        let t = anthropic::web_fetch_20250910();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, anthropic::WEB_FETCH_20250910_ID);
        assert_eq!(pd.name, "web_fetch");
    }

    #[test]
    fn google_google_search_tool_has_expected_id_and_name() {
        let t = google::google_search();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, google::GOOGLE_SEARCH_ID);
        assert_eq!(pd.name, "google_search");
    }

    #[test]
    fn google_google_search_retrieval_tool_has_expected_id_and_name() {
        let t = google::google_search_retrieval();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, google::GOOGLE_SEARCH_RETRIEVAL_ID);
        assert_eq!(pd.name, "google_search_retrieval");
    }

    #[test]
    fn xai_code_execution_tool_has_expected_id_and_name() {
        let t = xai::code_execution();
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, xai::CODE_EXECUTION_ID);
        assert_eq!(pd.name, "code_execution");
    }

    #[test]
    fn provider_defined_tool_by_id_uses_vercel_aligned_default_name() {
        let t = provider_defined_tool(openai::WEB_SEARCH_ID).expect("known tool id");
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, openai::WEB_SEARCH_ID);
        assert_eq!(pd.name, "webSearch");
    }

    #[test]
    fn provider_defined_tool_by_id_returns_none_for_unknown_id() {
        assert!(provider_defined_tool("openai.unknown_tool").is_none());
    }
}
