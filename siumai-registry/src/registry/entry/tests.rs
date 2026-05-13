#[cfg(feature = "builtins")]
use super::*;

#[cfg(feature = "builtins")]
#[test]
fn create_registry_with_defaults_registers_native_factories() {
    let _g = reg_test_guard();
    let _reg = crate::registry::helpers::create_registry_with_defaults();

    // These checks validate that the default handle-level registry wiring
    // actually registers factory entries for the common native providers.
    // We deliberately stop at handle creation, so no API keys or network
    // access are required for this test.
    #[cfg(feature = "openai")]
    {
        assert!(_reg.language_model("openai:any-model").is_ok());
    }
    #[cfg(feature = "azure")]
    {
        assert!(_reg.language_model("azure:any-model").is_ok());
        assert!(_reg.language_model("azure-chat:any-model").is_ok());
    }
    #[cfg(feature = "google-vertex")]
    {
        assert!(_reg.language_model("anthropic-vertex:any-model").is_ok());
    }
    #[cfg(feature = "google")]
    {
        assert!(_reg.language_model("gemini:any-model").is_ok());
    }
    #[cfg(feature = "groq")]
    {
        assert!(_reg.language_model("groq:any-model").is_ok());
    }
    #[cfg(feature = "xai")]
    {
        assert!(_reg.language_model("xai:any-model").is_ok());
    }
    #[cfg(feature = "ollama")]
    {
        assert!(_reg.language_model("ollama:any-model").is_ok());
    }
    #[cfg(feature = "minimaxi")]
    {
        assert!(_reg.language_model("minimaxi:any-model").is_ok());
    }
}
