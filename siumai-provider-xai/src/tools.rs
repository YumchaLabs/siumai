//! xAI provider-defined tool catalog.
//!
//! Canonical xAI hosted-tool IDs, default names, and direct `Tool` constructors live in the
//! provider crate because they are provider-owned facts.

pub mod xai {
    use siumai_core::types::Tool;
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
    /// Create a provider-defined xAI tool by stable tool id when no required args are needed.
    pub fn provider_defined_tool(id: &str) -> Option<Tool> {
        match id {
            WEB_SEARCH_ID => Some(web_search()),
            X_SEARCH_ID => Some(x_search()),
            CODE_EXECUTION_ID => Some(code_execution()),
            VIEW_IMAGE_ID => Some(view_image()),
            VIEW_X_VIDEO_ID => Some(view_x_video()),
            _ => None,
        }
    }
}
