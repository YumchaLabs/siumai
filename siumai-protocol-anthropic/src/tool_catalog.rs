//! Anthropic provider-defined tool catalog.
//!
//! Canonical Anthropic server-tool IDs, default names, and direct `Tool` constructors live here,
//! outside `siumai-spec`, because they are protocol/provider-owned facts.

pub mod anthropic {
    use siumai_core::types::Tool;

    /// Mapping of provider tool ids to Anthropic tool names (provider-native).
    ///
    /// Note: Anthropic "versioned tools" (e.g. `web_search_20250305`) still map to
    /// unversioned provider-native names (e.g. `web_search`) in request/response surfaces.
    pub const PROVIDER_TOOL_NAMES: &[(&str, &str)] = &[
        (WEB_SEARCH_20250305_ID, "web_search"),
        (WEB_SEARCH_20260209_ID, "web_search"),
        (WEB_FETCH_20250910_ID, "web_fetch"),
        (WEB_FETCH_20260209_ID, "web_fetch"),
        (COMPUTER_20250124_ID, "computer"),
        (COMPUTER_20241022_ID, "computer"),
        (COMPUTER_20251124_ID, "computer"),
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
        (CODE_EXECUTION_20260120_ID, "code_execution"),
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
            id: WEB_SEARCH_20260209_ID,
            tool_type: "web_search_20260209",
            tool_name: "web_search",
        },
        ServerToolSpec {
            id: WEB_FETCH_20250910_ID,
            tool_type: "web_fetch_20250910",
            tool_name: "web_fetch",
        },
        ServerToolSpec {
            id: WEB_FETCH_20260209_ID,
            tool_type: "web_fetch_20260209",
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
            id: COMPUTER_20251124_ID,
            tool_type: "computer_20251124",
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
            id: CODE_EXECUTION_20260120_ID,
            tool_type: "code_execution_20260120",
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
    pub const WEB_SEARCH_20260209_ID: &str = "anthropic.web_search_20260209";
    pub const WEB_FETCH_20250910_ID: &str = "anthropic.web_fetch_20250910";
    pub const WEB_FETCH_20260209_ID: &str = "anthropic.web_fetch_20260209";
    pub const COMPUTER_20250124_ID: &str = "anthropic.computer_20250124";
    pub const COMPUTER_20241022_ID: &str = "anthropic.computer_20241022";
    pub const COMPUTER_20251124_ID: &str = "anthropic.computer_20251124";
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
    pub const CODE_EXECUTION_20260120_ID: &str = "anthropic.code_execution_20260120";
    pub const MEMORY_20250818_ID: &str = "anthropic.memory_20250818";

    pub fn web_search_20250305() -> Tool {
        web_search_20250305_named("web_search")
    }

    pub fn web_search_20250305_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_20250305_ID, name).with_supports_deferred_results(true)
    }

    pub fn web_search() -> Tool {
        web_search_20250305()
    }

    pub fn web_search_20260209() -> Tool {
        web_search_20260209_named("web_search")
    }

    pub fn web_search_20260209_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_SEARCH_20260209_ID, name).with_supports_deferred_results(true)
    }

    pub fn web_fetch_20250910() -> Tool {
        web_fetch_20250910_named("web_fetch")
    }

    pub fn web_fetch_20250910_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_FETCH_20250910_ID, name).with_supports_deferred_results(true)
    }

    pub fn web_fetch_20260209() -> Tool {
        web_fetch_20260209_named("web_fetch")
    }

    pub fn web_fetch_20260209_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(WEB_FETCH_20260209_ID, name).with_supports_deferred_results(true)
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

    pub fn computer_20251124() -> Tool {
        computer_20251124_named("computer")
    }

    pub fn computer_20251124_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(COMPUTER_20251124_ID, name)
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
            .with_supports_deferred_results(true)
    }

    pub fn tool_search_bm25_20251119() -> Tool {
        tool_search_bm25_20251119_named("tool_search")
    }

    pub fn tool_search_bm25_20251119_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TOOL_SEARCH_BM25_20251119_ID, name)
            .with_supports_deferred_results(true)
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
            .with_supports_deferred_results(true)
    }

    pub fn code_execution_20260120() -> Tool {
        code_execution_20260120_named("code_execution")
    }

    pub fn code_execution_20260120_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CODE_EXECUTION_20260120_ID, name)
            .with_supports_deferred_results(true)
    }

    pub fn memory_20250818() -> Tool {
        memory_20250818_named("memory")
    }

    pub fn memory_20250818_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(MEMORY_20250818_ID, name)
    }
    /// Create a provider-defined Anthropic server tool by stable tool id.
    pub fn provider_defined_tool(id: &str) -> Option<Tool> {
        match id {
            WEB_SEARCH_20250305_ID => Some(web_search_20250305()),
            WEB_SEARCH_20260209_ID => Some(web_search_20260209()),
            WEB_FETCH_20250910_ID => Some(web_fetch_20250910()),
            WEB_FETCH_20260209_ID => Some(web_fetch_20260209()),
            COMPUTER_20250124_ID => Some(computer_20250124()),
            COMPUTER_20241022_ID => Some(computer_20241022()),
            COMPUTER_20251124_ID => Some(computer_20251124()),
            TEXT_EDITOR_20250124_ID => Some(text_editor_20250124()),
            TEXT_EDITOR_20241022_ID => Some(text_editor_20241022()),
            TEXT_EDITOR_20250429_ID => Some(text_editor_20250429()),
            TEXT_EDITOR_20250728_ID => Some(text_editor_20250728()),
            BASH_20241022_ID => Some(bash_20241022()),
            BASH_20250124_ID => Some(bash_20250124()),
            TOOL_SEARCH_REGEX_20251119_ID => Some(tool_search_regex_20251119()),
            TOOL_SEARCH_BM25_20251119_ID => Some(tool_search_bm25_20251119()),
            CODE_EXECUTION_20250522_ID => Some(code_execution_20250522()),
            CODE_EXECUTION_20250825_ID => Some(code_execution_20250825()),
            CODE_EXECUTION_20260120_ID => Some(code_execution_20260120()),
            MEMORY_20250818_ID => Some(memory_20250818()),
            _ => None,
        }
    }
}
