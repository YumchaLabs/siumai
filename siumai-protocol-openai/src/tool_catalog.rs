//! OpenAI provider-defined tool catalog.
//!
//! Canonical OpenAI hosted-tool IDs, default names, and direct `Tool` constructors live here,
//! outside `siumai-spec`, because they are protocol/provider-owned facts.

pub mod openai {
    use siumai_core::types::Tool;

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
        (TOOL_SEARCH_ID, "tool_search"),
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
    pub const TOOL_SEARCH_ID: &str = "openai.tool_search";
    pub const CUSTOM_ID: &str = "openai.custom";

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

    pub fn tool_search() -> Tool {
        tool_search_named("toolSearch")
    }

    pub fn tool_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(TOOL_SEARCH_ID, name)
    }

    pub fn custom(name: impl Into<String>) -> Tool {
        Tool::provider_defined(CUSTOM_ID, name)
    }
    /// Create a provider-defined OpenAI tool by stable tool id.
    pub fn provider_defined_tool(id: &str) -> Option<Tool> {
        match id {
            WEB_SEARCH_ID => Some(web_search()),
            WEB_SEARCH_PREVIEW_ID => Some(web_search_preview()),
            FILE_SEARCH_ID => Some(file_search()),
            CODE_INTERPRETER_ID => Some(code_interpreter()),
            IMAGE_GENERATION_ID => Some(image_generation()),
            LOCAL_SHELL_ID => Some(local_shell()),
            SHELL_ID => Some(shell()),
            COMPUTER_USE_ID => Some(computer_use()),
            MCP_ID => Some(mcp()),
            APPLY_PATCH_ID => Some(apply_patch()),
            TOOL_SEARCH_ID => Some(tool_search()),
            CUSTOM_ID => Some(custom("custom")),
            _ => None,
        }
    }
}
