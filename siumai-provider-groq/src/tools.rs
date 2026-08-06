//! Groq built-in tool helpers.

use crate::{GroqLanguageOptions, GroqResponsesOptions};

/// Create the typed provider option that enables Groq's built-in browser-search tool.
///
/// The tool is provider executed and therefore is not represented as a portable local
/// [`siumai_core::ToolSpec`].
pub fn browser_search() -> GroqLanguageOptions {
    GroqLanguageOptions::new().with_browser_search(true)
}

/// Enable Groq browser search on a Responses model.
pub fn responses_browser_search() -> GroqResponsesOptions {
    GroqResponsesOptions::new().with_browser_search(true)
}

/// Enable Groq code execution on a Responses model.
pub fn responses_code_execution() -> GroqResponsesOptions {
    GroqResponsesOptions::new().with_code_execution(true)
}

#[cfg(test)]
mod tests {
    use siumai_core::ProviderOptions;

    use super::*;

    #[test]
    fn browser_search_helper_builds_typed_options() {
        let options = ProviderOptions::typed(&browser_search()).unwrap();
        assert_eq!(options.namespace().as_str(), "groq");
        assert_eq!(options.value()["browser_search"], true);

        let options = ProviderOptions::typed(&responses_browser_search()).unwrap();
        assert_eq!(options.value()["browser_search"], true);
        let options = ProviderOptions::typed(&responses_code_execution()).unwrap();
        assert_eq!(options.value()["code_execution"], true);
    }
}
