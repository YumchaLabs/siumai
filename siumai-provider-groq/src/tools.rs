//! Groq provider-defined tool catalog.
//!
//! Canonical Groq hosted-tool IDs, default names, and direct `Tool` constructors live in the
//! provider crate because they are provider-owned facts.

pub mod groq {
    use siumai_core::types::Tool;

    pub const BROWSER_SEARCH_ID: &str = "groq.browser_search";

    pub fn browser_search() -> Tool {
        browser_search_named("browser_search")
    }

    pub fn browser_search_named(name: impl Into<String>) -> Tool {
        Tool::provider_defined(BROWSER_SEARCH_ID, name)
    }
    /// Create a provider-defined Groq tool by stable tool id.
    pub fn provider_defined_tool(id: &str) -> Option<Tool> {
        match id {
            BROWSER_SEARCH_ID => Some(browser_search()),
            _ => None,
        }
    }
}
