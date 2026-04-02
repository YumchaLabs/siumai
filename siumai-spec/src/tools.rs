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
        // `google.vertex_rag_store` and `google.file_search` require args, so `provider_defined_id(...)`
        // cannot build a valid default `Tool` value.

        // xAI
        xai::WEB_SEARCH_ID => Some(xai::web_search()),
        xai::X_SEARCH_ID => Some(xai::x_search()),
        xai::CODE_EXECUTION_ID => Some(xai::code_execution()),
        xai::VIEW_IMAGE_ID => Some(xai::view_image()),
        xai::VIEW_X_VIDEO_ID => Some(xai::view_x_video()),

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
        (TOOL_SEARCH_REGEX_20251119_ID, "tool_search_tool_regex"),
        (TOOL_SEARCH_BM25_20251119_ID, "tool_search_tool_bm25"),
        (CODE_EXECUTION_20250522_ID, "code_execution"),
        (CODE_EXECUTION_20250825_ID, "code_execution"),
        (MEMORY_20250818_ID, "memory"),
    ];

    /// Anthropic server tool spec for provider-defined tool IDs.
    ///
    /// Anthropic tool calls in Messages API use:
    /// - `type`: versioned tool identifier (e.g. `web_search_20250305`)
    /// - `name`: unversioned provider-native name (e.g. `web_search`)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ServerToolSpec {
        pub id: &'static str,
        pub tool_type: &'static str,
        pub tool_name: &'static str,
    }

    pub const SERVER_TOOL_SPECS: &[ServerToolSpec] = &[
        ServerToolSpec {
            id: WEB_SEARCH_20250305_ID,
            tool_type: "web_search_20250305",
            tool_name: "web_search",
        },
        ServerToolSpec {
            id: WEB_FETCH_20250910_ID,
            tool_type: "web_fetch_20250910",
            tool_name: "web_fetch",
        },
        ServerToolSpec {
            id: COMPUTER_20241022_ID,
            tool_type: "computer_20241022",
            tool_name: "computer",
        },
        ServerToolSpec {
            id: COMPUTER_20250124_ID,
            tool_type: "computer_20250124",
            tool_name: "computer",
        },
        ServerToolSpec {
            id: TEXT_EDITOR_20241022_ID,
            tool_type: "text_editor_20241022",
            tool_name: "str_replace_editor",
        },
        ServerToolSpec {
            id: TEXT_EDITOR_20250124_ID,
            tool_type: "text_editor_20250124",
            tool_name: "str_replace_editor",
        },
        ServerToolSpec {
            id: TEXT_EDITOR_20250429_ID,
            tool_type: "text_editor_20250429",
            tool_name: "str_replace_based_edit_tool",
        },
        ServerToolSpec {
            id: TEXT_EDITOR_20250728_ID,
            tool_type: "text_editor_20250728",
            tool_name: "str_replace_based_edit_tool",
        },
        ServerToolSpec {
            id: BASH_20241022_ID,
            tool_type: "bash_20241022",
            tool_name: "bash",
        },
        ServerToolSpec {
            id: BASH_20250124_ID,
            tool_type: "bash_20250124",
            tool_name: "bash",
        },
        ServerToolSpec {
            id: TOOL_SEARCH_REGEX_20251119_ID,
            tool_type: "tool_search_tool_regex_20251119",
            tool_name: "tool_search_tool_regex",
        },
        ServerToolSpec {
            id: TOOL_SEARCH_BM25_20251119_ID,
            tool_type: "tool_search_tool_bm25_20251119",
            tool_name: "tool_search_tool_bm25",
        },
        ServerToolSpec {
            id: CODE_EXECUTION_20250522_ID,
            tool_type: "code_execution_20250522",
            tool_name: "code_execution",
        },
        ServerToolSpec {
            id: CODE_EXECUTION_20250825_ID,
            tool_type: "code_execution_20250825",
            tool_name: "code_execution",
        },
        ServerToolSpec {
            id: MEMORY_20250818_ID,
            tool_type: "memory_20250818",
            tool_name: "memory",
        },
    ];

    pub fn server_tool_spec(id: &str) -> Option<&'static ServerToolSpec> {
        SERVER_TOOL_SPECS.iter().find(|s| s.id == id)
    }

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
}

/// xAI provider-defined tools (OpenAI-like family).
pub mod xai {
    use super::Tool;
    use std::collections::BTreeMap;

    /// Mapping of provider tool ids to xAI tool names (provider-native).
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (WEB_SEARCH_ID, "web_search"),
        (X_SEARCH_ID, "x_search"),
        (CODE_EXECUTION_ID, "code_execution"),
        (VIEW_IMAGE_ID, "view_image"),
        (VIEW_X_VIDEO_ID, "view_x_video"),
        (FILE_SEARCH_ID, "file_search"),
        (MCP_ID, "mcp"),
    ];

    pub const WEB_SEARCH_ID: &str = "xai.web_search";
    pub const X_SEARCH_ID: &str = "xai.x_search";
    pub const CODE_EXECUTION_ID: &str = "xai.code_execution";
    pub const VIEW_IMAGE_ID: &str = "xai.view_image";
    pub const VIEW_X_VIDEO_ID: &str = "xai.view_x_video";
    pub const FILE_SEARCH_ID: &str = "xai.file_search";
    pub const MCP_ID: &str = "xai.mcp";

    fn args_value<T: serde::Serialize>(args: T) -> serde_json::Value {
        serde_json::to_value(args).expect("xAI tool args should serialize")
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct WebSearchArgs {
        #[serde(rename = "allowedDomains", skip_serializing_if = "Option::is_none")]
        pub allowed_domains: Option<Vec<String>>,
        #[serde(rename = "excludedDomains", skip_serializing_if = "Option::is_none")]
        pub excluded_domains: Option<Vec<String>>,
        #[serde(
            rename = "enableImageUnderstanding",
            skip_serializing_if = "Option::is_none"
        )]
        pub enable_image_understanding: Option<bool>,
    }

    impl WebSearchArgs {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_allowed_domains<T, I>(mut self, domains: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
            self
        }

        pub fn with_excluded_domains<T, I>(mut self, domains: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            self.excluded_domains = Some(domains.into_iter().map(Into::into).collect());
            self
        }

        pub fn with_enable_image_understanding(mut self, enabled: bool) -> Self {
            self.enable_image_understanding = Some(enabled);
            self
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct XSearchArgs {
        #[serde(rename = "allowedXHandles", skip_serializing_if = "Option::is_none")]
        pub allowed_x_handles: Option<Vec<String>>,
        #[serde(rename = "excludedXHandles", skip_serializing_if = "Option::is_none")]
        pub excluded_x_handles: Option<Vec<String>>,
        #[serde(rename = "fromDate", skip_serializing_if = "Option::is_none")]
        pub from_date: Option<String>,
        #[serde(rename = "toDate", skip_serializing_if = "Option::is_none")]
        pub to_date: Option<String>,
        #[serde(
            rename = "enableImageUnderstanding",
            skip_serializing_if = "Option::is_none"
        )]
        pub enable_image_understanding: Option<bool>,
        #[serde(
            rename = "enableVideoUnderstanding",
            skip_serializing_if = "Option::is_none"
        )]
        pub enable_video_understanding: Option<bool>,
    }

    impl XSearchArgs {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_allowed_x_handles<T, I>(mut self, handles: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            self.allowed_x_handles = Some(handles.into_iter().map(Into::into).collect());
            self
        }

        pub fn with_excluded_x_handles<T, I>(mut self, handles: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            self.excluded_x_handles = Some(handles.into_iter().map(Into::into).collect());
            self
        }

        pub fn with_from_date(mut self, date: impl Into<String>) -> Self {
            self.from_date = Some(date.into());
            self
        }

        pub fn with_to_date(mut self, date: impl Into<String>) -> Self {
            self.to_date = Some(date.into());
            self
        }

        pub fn with_enable_image_understanding(mut self, enabled: bool) -> Self {
            self.enable_image_understanding = Some(enabled);
            self
        }

        pub fn with_enable_video_understanding(mut self, enabled: bool) -> Self {
            self.enable_video_understanding = Some(enabled);
            self
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct FileSearchArgs {
        #[serde(rename = "vectorStoreIds")]
        pub vector_store_ids: Vec<String>,
        #[serde(rename = "maxNumResults", skip_serializing_if = "Option::is_none")]
        pub max_num_results: Option<u32>,
    }

    impl FileSearchArgs {
        pub fn new<T, I>(vector_store_ids: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            Self {
                vector_store_ids: vector_store_ids.into_iter().map(Into::into).collect(),
                max_num_results: None,
            }
        }

        pub fn with_max_num_results(mut self, max_num_results: u32) -> Self {
            self.max_num_results = Some(max_num_results);
            self
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct McpArgs {
        #[serde(rename = "serverUrl")]
        pub server_url: String,
        #[serde(rename = "serverLabel", skip_serializing_if = "Option::is_none")]
        pub server_label: Option<String>,
        #[serde(rename = "serverDescription", skip_serializing_if = "Option::is_none")]
        pub server_description: Option<String>,
        #[serde(rename = "allowedTools", skip_serializing_if = "Option::is_none")]
        pub allowed_tools: Option<Vec<String>>,
        #[serde(rename = "headers", skip_serializing_if = "Option::is_none")]
        pub headers: Option<BTreeMap<String, String>>,
        #[serde(rename = "authorization", skip_serializing_if = "Option::is_none")]
        pub authorization: Option<String>,
    }

    impl McpArgs {
        pub fn new(server_url: impl Into<String>) -> Self {
            Self {
                server_url: server_url.into(),
                server_label: None,
                server_description: None,
                allowed_tools: None,
                headers: None,
                authorization: None,
            }
        }

        pub fn with_server_label(mut self, server_label: impl Into<String>) -> Self {
            self.server_label = Some(server_label.into());
            self
        }

        pub fn with_server_description(mut self, server_description: impl Into<String>) -> Self {
            self.server_description = Some(server_description.into());
            self
        }

        pub fn with_allowed_tools<T, I>(mut self, allowed_tools: I) -> Self
        where
            T: Into<String>,
            I: IntoIterator<Item = T>,
        {
            self.allowed_tools = Some(allowed_tools.into_iter().map(Into::into).collect());
            self
        }

        pub fn with_headers<K, V, I>(mut self, headers: I) -> Self
        where
            K: Into<String>,
            V: Into<String>,
            I: IntoIterator<Item = (K, V)>,
        {
            self.headers = Some(
                headers
                    .into_iter()
                    .map(|(key, value)| (key.into(), value.into()))
                    .collect(),
            );
            self
        }

        pub fn with_authorization(mut self, authorization: impl Into<String>) -> Self {
            self.authorization = Some(authorization.into());
            self
        }
    }

    pub fn web_search() -> Tool {
        web_search_named("web_search")
    }

    pub fn web_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, name)
    }

    pub fn web_search_with(args: WebSearchArgs) -> Tool {
        web_search_named_with("web_search", args)
    }

    pub fn web_search_named_with(name: impl Into<String>, args: WebSearchArgs) -> Tool {
        Tool::provider_defined(WEB_SEARCH_ID, name).with_args(args_value(args))
    }

    pub fn x_search() -> Tool {
        x_search_named("x_search")
    }

    pub fn x_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(X_SEARCH_ID, name)
    }

    pub fn x_search_with(args: XSearchArgs) -> Tool {
        x_search_named_with("x_search", args)
    }

    pub fn x_search_named_with(name: impl Into<String>, args: XSearchArgs) -> Tool {
        Tool::provider_defined(X_SEARCH_ID, name).with_args(args_value(args))
    }

    pub fn code_execution() -> Tool {
        code_execution_named("code_execution")
    }

    pub fn code_execution_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_ID, name)
    }

    pub fn view_image() -> Tool {
        view_image_named("view_image")
    }

    pub fn view_image_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(VIEW_IMAGE_ID, name)
    }

    pub fn view_x_video() -> Tool {
        view_x_video_named("view_x_video")
    }

    pub fn view_x_video_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(VIEW_X_VIDEO_ID, name)
    }

    pub fn file_search(vector_store_ids: Vec<String>) -> Tool {
        file_search_named(vector_store_ids, "file_search")
    }

    pub fn file_search_named(vector_store_ids: Vec<String>, name: impl Into<String>) -> Tool {
        file_search_named_with(name, FileSearchArgs::new(vector_store_ids))
    }

    pub fn file_search_with(args: FileSearchArgs) -> Tool {
        file_search_named_with("file_search", args)
    }

    pub fn file_search_named_with(name: impl Into<String>, args: FileSearchArgs) -> Tool {
        Tool::provider_defined(FILE_SEARCH_ID, name).with_args(args_value(args))
    }

    pub fn mcp(server_url: impl Into<String>) -> Tool {
        mcp_named(server_url, "mcp")
    }

    pub fn mcp_named(server_url: impl Into<String>, name: impl Into<String>) -> Tool {
        mcp_named_with(name, McpArgs::new(server_url))
    }

    pub fn mcp_with(args: McpArgs) -> Tool {
        mcp_named_with("mcp", args)
    }

    pub fn mcp_named_with(name: impl Into<String>, args: McpArgs) -> Tool {
        Tool::provider_defined(MCP_ID, name).with_args(args_value(args))
    }

    pub fn mcp_server(server_url: impl Into<String>) -> Tool {
        mcp(server_url)
    }

    pub fn mcp_server_with(args: McpArgs) -> Tool {
        mcp_with(args)
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
    fn xai_view_tools_have_expected_ids_and_names() {
        for (tool, id, name) in [
            (xai::view_image(), xai::VIEW_IMAGE_ID, "view_image"),
            (xai::view_x_video(), xai::VIEW_X_VIDEO_ID, "view_x_video"),
        ] {
            let crate::types::Tool::ProviderDefined(pd) = tool else {
                panic!("expected provider-defined tool");
            };
            assert_eq!(pd.id, id);
            assert_eq!(pd.name, name);
        }
    }

    #[test]
    fn xai_file_search_tool_has_expected_id_name_and_args() {
        let t = xai::file_search(vec!["collection_1".to_string(), "collection_2".to_string()]);
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, xai::FILE_SEARCH_ID);
        assert_eq!(pd.name, "file_search");
        assert_eq!(
            pd.args,
            serde_json::json!({
                "vectorStoreIds": ["collection_1", "collection_2"]
            })
        );
    }

    #[test]
    fn xai_typed_tool_arg_builders_serialize_sdk_aligned_shapes() {
        let web_search = xai::web_search_with(
            xai::WebSearchArgs::new()
                .with_allowed_domains(["wikipedia.org"])
                .with_excluded_domains(["spam.com"])
                .with_enable_image_understanding(true),
        );
        let crate::types::Tool::ProviderDefined(web_pd) = web_search else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(web_pd.id, xai::WEB_SEARCH_ID);
        assert_eq!(
            web_pd.args,
            serde_json::json!({
                "allowedDomains": ["wikipedia.org"],
                "excludedDomains": ["spam.com"],
                "enableImageUnderstanding": true
            })
        );

        let x_search = xai::x_search_with(
            xai::XSearchArgs::new()
                .with_allowed_x_handles(["xai"])
                .with_excluded_x_handles(["spam_handle"])
                .with_from_date("2025-01-01")
                .with_to_date("2025-01-31")
                .with_enable_image_understanding(true)
                .with_enable_video_understanding(true),
        );
        let crate::types::Tool::ProviderDefined(x_pd) = x_search else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(x_pd.id, xai::X_SEARCH_ID);
        assert_eq!(
            x_pd.args,
            serde_json::json!({
                "allowedXHandles": ["xai"],
                "excludedXHandles": ["spam_handle"],
                "fromDate": "2025-01-01",
                "toDate": "2025-01-31",
                "enableImageUnderstanding": true,
                "enableVideoUnderstanding": true
            })
        );

        let file_search = xai::file_search_with(
            xai::FileSearchArgs::new(["collection_1"]).with_max_num_results(5),
        );
        let crate::types::Tool::ProviderDefined(file_pd) = file_search else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(file_pd.id, xai::FILE_SEARCH_ID);
        assert_eq!(
            file_pd.args,
            serde_json::json!({
                "vectorStoreIds": ["collection_1"],
                "maxNumResults": 5
            })
        );

        let mcp = xai::mcp_server_with(
            xai::McpArgs::new("https://example.com/mcp")
                .with_server_label("docs")
                .with_server_description("Docs MCP")
                .with_allowed_tools(["search_docs"])
                .with_headers([("X-Test", "1")])
                .with_authorization("Bearer token"),
        );
        let crate::types::Tool::ProviderDefined(mcp_pd) = mcp else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(mcp_pd.id, xai::MCP_ID);
        assert_eq!(
            mcp_pd.args,
            serde_json::json!({
                "serverUrl": "https://example.com/mcp",
                "serverLabel": "docs",
                "serverDescription": "Docs MCP",
                "allowedTools": ["search_docs"],
                "headers": { "X-Test": "1" },
                "authorization": "Bearer token"
            })
        );
    }

    #[test]
    fn xai_mcp_tool_has_expected_id_name_and_required_args() {
        let t = xai::mcp("https://example.com/mcp");
        let crate::types::Tool::ProviderDefined(pd) = t else {
            panic!("expected provider-defined tool");
        };
        assert_eq!(pd.id, xai::MCP_ID);
        assert_eq!(pd.name, "mcp");
        assert_eq!(
            pd.args,
            serde_json::json!({
                "serverUrl": "https://example.com/mcp"
            })
        );
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
