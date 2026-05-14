#[test]
fn core_utils_do_not_own_provider_model_aliases() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let utils_dir = manifest_dir.join("src").join("utils");

    assert!(
        !utils_dir.join("model_alias.rs").exists(),
        "provider-specific model aliases belong to registry/provider-owned code, not siumai-core"
    );

    let utils_mod = std::fs::read_to_string(utils_dir.join("mod.rs")).expect("read utils/mod.rs");
    assert!(
        !utils_mod.contains("model_alias"),
        "siumai-core::utils must not export a provider model alias module"
    );

    let builder_helpers = std::fs::read_to_string(utils_dir.join("builder_helpers.rs"))
        .expect("read utils/builder_helpers.rs");
    assert!(
        !builder_helpers.contains("normalize_model_id"),
        "siumai-core builder helpers must not expose provider-specific model normalization"
    );
}

#[test]
fn core_validator_does_not_own_provider_model_catalogs() {
    let validator = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("params")
            .join("validator.rs"),
    )
    .expect("read params/validator.rs");

    for forbidden in [
        "is_model_supported",
        "suggest_alternative_model",
        "deepseek",
        "openrouter",
        "siliconflow",
        "fireworks",
        "moonshotai",
        "llama-v3p1",
        "claude-",
        "gemini-",
        "gpt-4",
    ] {
        assert!(
            !validator.contains(forbidden),
            "siumai-core validator must not contain provider/model catalog fragment `{forbidden}`"
        );
    }
}
