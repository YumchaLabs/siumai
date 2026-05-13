use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn language_model_normalizes_common_provider_aliases() {
    let _g = reg_test_guard();

    let mut providers = HashMap::new();
    providers.insert(
        "gemini".to_string(),
        Arc::new(TestProviderFactory::new("gemini")) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "vertex".to_string(),
        Arc::new(TestProviderFactory::new("vertex")) as Arc<dyn ProviderFactory>,
    );
    providers.insert(
        "anthropic-vertex".to_string(),
        Arc::new(TestProviderFactory::new("anthropic-vertex")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    assert!(reg.language_model("google:any-model").is_ok());
    assert!(reg.language_model("google-vertex:any-model").is_ok());
    assert!(
        reg.language_model("google-vertex-anthropic:any-model")
            .is_ok()
    );
}
