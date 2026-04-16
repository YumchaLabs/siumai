#![cfg(feature = "openai-standard")]

use siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config;

#[test]
fn groq_provider_config_uses_canonical_api_key_env() {
    let config = get_provider_config("groq").expect("groq provider config");

    assert_eq!(config.id, "groq");
    assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
    assert_eq!(config.api_key_env.as_deref(), Some("GROQ_API_KEY"));
}
