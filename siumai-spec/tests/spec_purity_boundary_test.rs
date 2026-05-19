use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn collect_rust_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path.to_path_buf());
        }
        return;
    }

    for entry in fs::read_dir(path).expect("read source directory") {
        let entry = entry.expect("read source directory entry");
        collect_rust_files(&entry.path(), files);
    }
}

#[test]
fn spec_crate_does_not_depend_on_runtime_execution_crates() {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).expect("read Cargo.toml");

    for forbidden_dependency in [
        "tokio",
        "tokio-util",
        "futures",
        "reqwest",
        "hyper",
        "axum",
        "tower",
        "siumai-core",
        "siumai-registry",
        "siumai-bridge",
        "siumai-extras",
        "siumai-provider-",
        "siumai-protocol-",
    ] {
        assert!(
            !manifest.contains(forbidden_dependency),
            "siumai-spec must stay data-only and must not depend on `{forbidden_dependency}`"
        );
    }
}

#[test]
fn spec_source_does_not_define_runtime_handles_or_streams() {
    let mut files = Vec::new();
    collect_rust_files(&crate_root().join("src"), &mut files);
    files.sort();

    for file in files {
        let source = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));

        for forbidden in [
            "tokio_util",
            "tokio::",
            "CancellationToken",
            "WaitForCancellationFuture",
            "CancelHandle",
            "futures::",
            "reqwest::",
            "hyper::",
            "axum::",
            "tower::",
            "std::thread",
            "std::process",
            "Command::new",
            "Pin<Box<dyn Stream",
            "siumai_core::",
            "siumai_registry::",
            "siumai_bridge::",
            "siumai_extras::",
            "siumai_provider_",
            "siumai_protocol_",
            "ProviderFactory",
            "RegistryOptions",
            "ProviderBuildOverrides",
            "BuildContext",
            "create_provider_registry",
            "registry::global",
            "std::env",
            "CARGO_PKG_VERSION",
            "runtime_default",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} must not contain runtime-only spec boundary leak `{forbidden}`",
                file.display()
            );
        }
    }
}

#[test]
fn spec_docs_do_not_teach_facade_or_provider_runtime_construction() {
    let mut files = Vec::new();
    collect_rust_files(&crate_root().join("src"), &mut files);
    files.sort();

    for file in files {
        let source = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));

        for forbidden in [
            "Siumai::builder",
            "siumai::prelude",
            "provider_ext::",
            ".openai()",
            ".api_key(",
            "client.chat(",
            "client.chat().",
            "client.chat_request(",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} must keep siumai-spec docs/data examples free of facade or provider runtime construction `{forbidden}`",
                file.display()
            );
        }
    }
}

#[test]
fn spec_error_type_does_not_own_runtime_policy_helpers() {
    let error_source =
        fs::read_to_string(crate_root().join("src/error/types.rs")).expect("read error types");

    for forbidden in [
        "pub enum ErrorCategory",
        "pub fn is_retryable",
        "pub const fn is_auth_error",
        "pub const fn is_rate_limit_error",
        "pub const fn status_code",
        "pub fn category",
        "pub fn user_message",
        "pub fn recovery_suggestions",
        "pub const fn recommended_retry_delay",
        "pub const fn max_retry_attempts",
    ] {
        assert!(
            !error_source.contains(forbidden),
            "siumai-spec::LlmError must stay a passive error data type and must not own runtime policy helper `{forbidden}`"
        );
    }
}

#[test]
fn spec_provider_type_is_documented_as_compatibility_only() {
    let types_source =
        fs::read_to_string(crate_root().join("src/types.rs")).expect("read types.rs");
    let common_source =
        fs::read_to_string(crate_root().join("src/types/common.rs")).expect("read common.rs");

    assert!(
        types_source.contains("ProviderType` - Legacy compatibility provider classification enum"),
        "siumai-spec docs should not present ProviderType as the primary provider identity"
    );
    assert!(
        common_source.contains("Legacy compatibility provider classification."),
        "ProviderType should be documented as compatibility classification, not the extensibility seam"
    );
    assert!(
        common_source.contains("Provider ids are open strings"),
        "ProviderType docs should point new provider flows to open provider ids"
    );
}
