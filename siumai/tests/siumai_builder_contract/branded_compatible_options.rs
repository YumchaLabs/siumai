use siumai::providers::groq::options::GroqLanguageOptions;
use siumai::{CallOptions, ProviderOptionError, Siumai};

use super::openai_compatible::custom_profile;

#[test]
fn branded_options_cannot_cross_into_a_compatible_profile() -> Result<(), Box<dyn std::error::Error>>
{
    let groq = Siumai::builder()
        .groq()
        .api_key("test-api-key")
        .build()?
        .language("future-groq-model")?;
    let compatible = Siumai::builder()
        .openai_compatible()
        .profile(custom_profile()?)
        .api_key("test-api-key")
        .build()?
        .language("future-compatible-model")?;
    let foreign = CallOptions::default()
        .with_provider_options_for(groq.model(), &GroqLanguageOptions::default())?;

    let error = match compatible.call("hello").with_options(foreign) {
        Ok(_) => panic!("Groq options must not target an explicit compatible profile"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ProviderOptionError::NamespaceMismatch { .. }
    ));

    Ok(())
}
