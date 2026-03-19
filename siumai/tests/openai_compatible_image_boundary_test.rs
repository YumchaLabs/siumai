#![cfg(feature = "openai")]

use siumai::prelude::unified::LlmError;
use siumai::registry::helpers::create_registry_with_defaults;

#[test]
fn compat_non_image_presets_reject_registry_image_handles() {
    let registry = create_registry_with_defaults();

    for model_id in [
        "fireworks:whisper-v3",
        "openrouter:openai/gpt-4o",
        "perplexity:sonar",
        "infini:text-embedding-3-small",
        "jina:jina-embeddings-v2-base-en",
        "voyageai:voyage-3",
    ] {
        match registry.image_model(model_id) {
            Ok(_) => panic!("{model_id} registry image handle should be unsupported"),
            Err(LlmError::UnsupportedOperation(_)) => {}
            Err(err) => panic!("{model_id} should fail with UnsupportedOperation, got {err:?}"),
        }
    }
}
