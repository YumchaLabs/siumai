use super::*;

#[test]
fn registry_options_default_keeps_registry_creation_defaults() {
    let opts = RegistryOptions::default();

    assert_eq!(opts.separator, ':');
    assert!(opts.language_model_middleware.is_empty());
    assert!(opts.http_interceptors.is_empty());
    assert!(opts.http_client.is_none());
    assert!(opts.http_transport.is_none());
    assert!(opts.http_config.is_none());
    assert!(opts.api_key.is_none());
    assert!(opts.base_url.is_none());
    assert!(opts.reasoning_enabled.is_none());
    assert!(opts.reasoning_budget.is_none());
    assert!(opts.provider_build_overrides.is_empty());
    assert!(opts.retry_options.is_none());
    assert!(opts.max_cache_entries.is_none());
    assert!(opts.client_ttl.is_none());
    assert!(opts.auto_middleware);
}

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
