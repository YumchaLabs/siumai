fn main() {
    // Check for conflicting feature combinations
    check_feature_conflicts();

    // Validate at least one provider is enabled
    ensure_provider_available();

    // Add compile-time information
    add_build_info();
}

/// Check for invalid feature combinations
fn check_feature_conflicts() {
    // Currently no conflicting features, but this is where we'd check
    // For example, if we had mutually exclusive features:
    // if cfg!(all(feature = "sync", feature = "async-only")) {
    //     panic!("Cannot enable both 'sync' and 'async-only' features");
    // }
}

/// Ensure at least one provider is available at compile time
fn ensure_provider_available() {
    let providers = [
        cfg!(feature = "openai"),
        cfg!(feature = "azure"),
        cfg!(feature = "anthropic"),
        cfg!(feature = "google"),
        cfg!(feature = "google-vertex"),
        cfg!(feature = "ollama"),
        cfg!(feature = "xai"),
        cfg!(feature = "groq"),
        cfg!(feature = "minimaxi"),
        cfg!(feature = "deepseek"),
    ];

    if !providers.iter().any(|&enabled| enabled) {
        panic!(
            "At least one provider feature must be enabled. Available features: openai, azure, anthropic, google, google-vertex, ollama, xai, groq, minimaxi, deepseek"
        );
    }
}

/// Add build-time information as environment variables
fn add_build_info() {
    // Count enabled providers
    let mut enabled_providers = Vec::new();

    if cfg!(feature = "openai") {
        enabled_providers.push("openai");
    }
    if cfg!(feature = "azure") {
        enabled_providers.push("azure");
    }
    if cfg!(feature = "anthropic") {
        enabled_providers.push("anthropic");
    }
    if cfg!(feature = "google") {
        enabled_providers.push("google");
    }
    if cfg!(feature = "google-vertex") {
        enabled_providers.push("google-vertex");
    }
    if cfg!(feature = "ollama") {
        enabled_providers.push("ollama");
    }
    if cfg!(feature = "xai") {
        enabled_providers.push("xai");
    }
    if cfg!(feature = "groq") {
        enabled_providers.push("groq");
    }
    if cfg!(feature = "minimaxi") {
        enabled_providers.push("minimaxi");
    }
    if cfg!(feature = "deepseek") {
        enabled_providers.push("deepseek");
    }

    // Set environment variables for runtime access
    println!(
        "cargo:rustc-env=SIUMAI_ENABLED_PROVIDERS={}",
        enabled_providers.join(",")
    );
    println!(
        "cargo:rustc-env=SIUMAI_PROVIDER_COUNT={}",
        enabled_providers.len()
    );

    // Print which providers are enabled for build logs
    if !enabled_providers.is_empty() {
        println!(
            "cargo:warning=Siumai compiled with providers: {}",
            enabled_providers.join(", ")
        );
    }
}
