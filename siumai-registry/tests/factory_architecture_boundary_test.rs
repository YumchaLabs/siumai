use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn assert_source_order(source: &str, earlier: &str, later: &str, message: &str) {
    let earlier_pos = source.find(earlier).expect("earlier source marker present");
    let later_pos = source.find(later).expect("later source marker present");
    assert!(earlier_pos < later_pos, "{message}");
}

fn collect_markdown_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(path.to_path_buf());
        }
        return;
    }

    for entry in fs::read_dir(path).expect("read docs directory") {
        let entry = entry.expect("read docs directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_markdown_files(&path, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(path);
        }
    }
}

#[test]
fn no_builtins_custom_factory_example_is_family_first() {
    let root = crate_root();
    let example = fs::read_to_string(root.join("examples/no_builtins_custom_factory.rs"))
        .expect("read no_builtins_custom_factory example");
    let docs_readme =
        fs::read_to_string(root.join("../docs/README.md")).expect("read docs README index");
    let architecture_doc =
        fs::read_to_string(root.join("../docs/architecture/registry-without-builtins.md"))
            .expect("read registry-without-builtins architecture doc");
    let factory_doc =
        fs::read_to_string(root.join("src/registry/entry/factory.rs")).expect("read factory trait");

    assert!(
        example.contains("async fn language_model_text_with_ctx(")
            && example.contains("Arc<dyn LanguageModel>"),
        "custom ProviderFactory example should implement the native text-family method"
    );
    assert!(
        docs_readme.contains("docs/workstreams/fearless-architecture-convergence/"),
        "docs index should link to the active fearless architecture workstream"
    );
    assert!(
        architecture_doc.contains("language_model_text_with_ctx")
            && architecture_doc.contains("compat_*_client")
            && !architecture_doc.contains("Generic `LlmClient` methods"),
        "custom ProviderFactory docs should describe family-model construction as primary and keep generic clients as legacy compatibility only"
    );
    assert!(
        factory_doc.contains("The primary contract is to create family model objects")
            && factory_doc.contains("Compatibility alias for creating a generic `LlmClient`")
            && factory_doc.contains(
                "New registry execution should prefer `language_model_text_with_ctx(...)`"
            ),
        "ProviderFactory docs should keep the family-first contract explicit"
    );
    assert!(
        !factory_doc.contains("#[deprecated")
            && !factory_doc.contains("async fn language_model(")
            && !factory_doc.contains("async fn completion_model(")
            && !factory_doc.contains("async fn embedding_model(")
            && !factory_doc.contains("async fn image_model(")
            && !factory_doc.contains("async fn speech_model(")
            && !factory_doc.contains("async fn transcription_model(")
            && !factory_doc.contains("async fn video_model(")
            && !factory_doc.contains("async fn reranking_model("),
        "ProviderFactory legacy generic-client wrapper methods should be removed, not just deprecated"
    );
    assert_source_order(
        &factory_doc,
        "async fn language_model_text_with_ctx(",
        "async fn compat_language_client(",
        "language family methods should be listed before the compat language client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn completion_model_family_with_ctx(",
        "async fn compat_completion_client(",
        "completion family methods should be listed before the compat completion client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn embedding_model_family_with_ctx(",
        "async fn compat_embedding_client(",
        "embedding family methods should be listed before the compat embedding client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn image_model_family_with_ctx(",
        "async fn compat_image_client(",
        "image family methods should be listed before the compat image client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn speech_model_family_with_ctx(",
        "async fn compat_speech_client(",
        "speech family methods should be listed before the compat speech client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn transcription_model_family_with_ctx(",
        "async fn compat_transcription_client(",
        "transcription family methods should be listed before the compat transcription client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn video_model_family_with_ctx(",
        "async fn compat_video_client(",
        "video family methods should be listed before the compat video client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn reranking_model_family_with_ctx(",
        "async fn compat_reranking_client(",
        "reranking family methods should be listed before the compat reranking client entry point",
    );
}

#[test]
fn registry_root_does_not_mirror_broad_core_modules() {
    let root = crate_root();
    let lib_rs = fs::read_to_string(root.join("src/lib.rs")).expect("read siumai-registry lib.rs");
    let public_root_exports = lib_rs
        .split("// Internal aliases for registry implementation")
        .next()
        .expect("public root export section");

    for forbidden in [
        "custom_provider",
        "embedding",
        "hosted_tools",
        "image",
        "retry_api",
        "video",
    ] {
        assert!(
            !public_root_exports.contains(&format!("pub use siumai_core::{forbidden}"))
                && !public_root_exports.contains(&format!(" {forbidden},"))
                && !public_root_exports.contains(&format!("    {forbidden},"))
                && !public_root_exports.contains(&format!("{forbidden}, "))
                && !public_root_exports.contains(&format!("{forbidden}\n")),
            "siumai-registry root should not mirror broad siumai-core module `{forbidden}`; keep it internal or expose it through a documented experimental path"
        );
    }

    assert!(
        lib_rs.contains("pub use siumai_core::{LlmError, error, streaming, text, traits, types};"),
        "registry root should keep only the small custom-factory contract surface"
    );
    assert!(
        lib_rs.contains("pub mod experimental {"),
        "low-level core implementation modules should stay behind siumai_registry::experimental"
    );
}

#[test]
fn registry_family_handles_keep_llm_client_downcasts_isolated() {
    let handles_dir = crate_root()
        .join("src")
        .join("registry")
        .join("entry")
        .join("handles");

    for file in [
        "audio.rs",
        "completion.rs",
        "embedding.rs",
        "image.rs",
        "rerank.rs",
        "video.rs",
    ] {
        let source = fs::read_to_string(handles_dir.join(file)).expect("read handle source");
        let downcast_lines = source
            .lines()
            .filter(|line| line.contains(".as_") && line.contains("_capability("))
            .collect::<Vec<_>>();
        assert!(
            downcast_lines.is_empty(),
            "{file} should use family model factories directly; LlmClient capability downcasts belong only behind explicit compat_* methods for extension-only language paths: {downcast_lines:?}"
        );
    }

    let language_source =
        fs::read_to_string(handles_dir.join("language.rs")).expect("read language handle source");
    let allowed_extension_downcasts = [
        ".as_file_management_capability(",
        ".as_skills_capability(",
        ".as_music_generation_capability(",
    ];
    for line in language_source
        .lines()
        .filter(|line| line.contains(".as_") && line.contains("_capability("))
    {
        assert!(
            allowed_extension_downcasts
                .iter()
                .any(|allowed| line.contains(allowed)),
            "language.rs may only keep extension-only LlmClient downcasts until those surfaces become family models: {line}"
        );
    }
}

#[test]
fn production_factories_do_not_override_legacy_generic_language_method() {
    let factories_dir = crate_root().join("src").join("registry").join("factories");

    for entry in fs::read_dir(factories_dir).expect("read registry factories directory") {
        let path = entry.expect("read registry factory entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if matches!(file_name, "contract_tests.rs" | "test.rs" | "mod.rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("read registry factory source");
        assert!(
            !source.contains("async fn language_model("),
            "{file_name} should implement compat_language_client(...) instead of overriding the deprecated legacy generic language_model(...) method"
        );
        assert!(
            source.contains("async fn compat_language_client(")
                || source.contains("async fn compat_language_client_with_ctx("),
            "{file_name} should keep generic-client construction behind an explicit compat_* method"
        );
    }
}

#[test]
fn compatibility_audit_categorizes_public_deprecated_surfaces() {
    let audit = fs::read_to_string(
        crate_root()
            .join("../docs/workstreams/fearless-architecture-convergence/compatibility-audit.md"),
    )
    .expect("read compatibility audit");

    for surface in [
        "Siumai::builder()",
        "siumai::compat::{Siumai, SiumaiBuilder, builder::*}",
        "SiumaiBuilder::provider(...)",
        "SiumaiBuilder::vision(...)",
        "Siumai::vision_capability()",
        "VisionCapability",
        "VisionCapabilityProxy",
        "experimental_generate_image",
        "experimental_generate_speech",
        "experimental_transcribe",
        "experimental_generate_video",
        "create_google_generative_ai()",
        "AudioModelHandle::text_to_speech(...)",
        "execute_json_request_with_headers(...)",
        "siumai_core::utils::vertex",
        "Provider extension deprecated option/metadata aliases",
    ] {
        assert!(
            audit.contains(surface),
            "compatibility audit should categorize deprecated public surface `{surface}`"
        );
    }

    for category in ["keep, time-bounded", "remove", "move"] {
        assert!(
            audit.contains(category),
            "compatibility audit should include `{category}` decisions"
        );
    }
}

#[test]
fn public_docs_do_not_recommend_compatibility_surfaces_as_default() {
    let docs_root = crate_root().join("../docs");
    let mut files = Vec::new();

    for path in [
        docs_root.join("README.md"),
        docs_root.join("architecture"),
        docs_root.join("migration"),
    ] {
        collect_markdown_files(&path, &mut files);
    }
    files.sort();

    for file in files {
        let source = fs::read_to_string(&file).expect("read public docs source");
        for forbidden in [
            "Prefer `Siumai::builder()`",
            "Prefer the unified traits: `ChatCapability` / `LlmClient`",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} should not recommend compatibility surfaces as the default path",
                file.display()
            );
        }
    }
}

#[test]
fn public_docs_classify_generic_llm_client_factory_paths_as_migration_only() {
    let root = crate_root();
    let migration_doc =
        fs::read_to_string(root.join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read beta.7 migration guide");
    let public_surface_doc =
        fs::read_to_string(root.join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let registry_doc =
        fs::read_to_string(root.join("../docs/architecture/registry-without-builtins.md"))
            .expect("read registry without builtins doc");

    for (name, source) in [
        ("migration-0.11.0-beta.7.md", migration_doc.as_str()),
        ("public-surface.md", public_surface_doc.as_str()),
        ("registry-without-builtins.md", registry_doc.as_str()),
    ] {
        assert!(
            source.contains("family-first")
                || source.contains("family methods")
                || source.contains("family model"),
            "{name} should describe registry construction as family-first"
        );
        assert!(
            source.contains("compat_*_client"),
            "{name} should mention explicit compat_*_client paths when discussing generic clients"
        );
        assert!(
            source.contains("LlmClient"),
            "{name} should name generic LlmClient when classifying legacy generic-client paths"
        );
    }

    assert!(
        migration_doc.contains("Generic `LlmClient` compatibility paths")
            && migration_doc.contains("extension-only surfaces")
            && migration_doc.contains("language_model_text_with_ctx"),
        "the beta.7 migration guide should give downstream users a concrete replacement for generic LlmClient factory paths"
    );
}

#[test]
fn focused_public_facade_tests_use_registry_owned_builtin_factory_resolution() {
    let root = crate_root();

    for relative in [
        "../siumai/tests/openai_embedding_public_helper_request_parity_test.rs",
        "../siumai/tests/google_vertex_typed_metadata_boundary_test.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("read public facade test");
        assert!(
            source.contains("registry::builtin_provider_factory("),
            "{relative} should use registry-owned built-in factory resolution"
        );
        assert!(
            !source.contains("registry::factories::"),
            "{relative} should not instantiate concrete built-in factory structs through the facade"
        );
    }

    let public_path_source =
        fs::read_to_string(root.join("../siumai/tests/provider_public_path_parity_test.rs"))
            .expect("read provider public-path parity test");
    assert!(
        public_path_source.contains("registry::builtin_provider_factory(")
            && public_path_source.contains("registry::azure_provider_factory_with_options(")
            && public_path_source.contains("registry::openai_compatible_provider_factory("),
        "provider_public_path_parity_test.rs should route built-in, Azure option, and OpenAI-compatible registry setup through registry-owned helpers"
    );
    assert!(
        !public_path_source.contains("siumai::registry::factories::")
            && !public_path_source.contains("registry::factories::"),
        "provider_public_path_parity_test.rs should not instantiate concrete built-in factory structs through the facade"
    );

    let facade_tests_dir = root.join("../siumai/tests");
    for entry in fs::read_dir(&facade_tests_dir).expect("read siumai facade tests") {
        let entry = entry.expect("read siumai facade test entry");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read siumai facade test source");
        assert!(
            !source.contains("siumai::registry::factories::")
                && !source.contains("registry::factories::"),
            "{} should use registry-owned helpers instead of facade-visible concrete built-in factories",
            path.display()
        );
    }
}

#[test]
fn compatibility_builder_uses_registry_owned_default_model_resolution() {
    let root = crate_root();
    let build_source =
        fs::read_to_string(root.join("src/provider/build.rs")).expect("read provider build source");
    let metadata_source = fs::read_to_string(root.join("src/native_provider_metadata.rs"))
        .expect("read native provider metadata source");
    let registry_source =
        fs::read_to_string(root.join("src/registry/mod.rs")).expect("read registry source");

    assert!(
        build_source.contains("registry::helpers::builtin_provider_default_model("),
        "SiumaiBuilder compatibility construction should delegate default model selection to the registry helper"
    );

    for forbidden in [
        "siumai_provider_openai::providers::openai::model_constants",
        "siumai_provider_anthropic::providers::anthropic::model_constants",
        "siumai_provider_gemini::providers::gemini::model_constants",
        "siumai_provider_openai_compatible::providers::openai_compatible::default_models",
        "crate::utils::builder_helpers::get_effective_model",
        "llama3.2",
        "grok-beta",
        "MiniMax-M2",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
    ] {
        assert!(
            !build_source.contains(forbidden),
            "SiumaiBuilder build.rs should not encode provider default model source `{forbidden}` directly"
        );
    }

    assert!(
        metadata_source.contains("NativeProviderDefaultModelPolicy")
            && metadata_source.contains("default_model_policy:"),
        "native provider metadata should own native provider default-model policy"
    );
    assert!(
        registry_source.contains("meta.default_model_policy.default_model()"),
        "built-in provider catalog should reuse native provider default-model policy instead of hand-written per-provider patches"
    );
}

#[test]
fn focused_public_facade_tests_use_provider_build_override_shortcuts() {
    let root = crate_root();

    for relative in [
        "../siumai/tests/deepinfra_chat_stream_public_path_alignment_test.rs",
        "../siumai/tests/gemini_embedding_batch_helper_parity_test.rs",
        "../siumai/tests/google_vertex_typed_metadata_boundary_test.rs",
        "../siumai/tests/openai_embedding_public_helper_request_parity_test.rs",
        "../siumai/tests/vertex_embedding_batch_helper_parity_test.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("read public facade test");
        assert!(
            source.contains(".with_provider_api_key")
                || source.contains("ProviderBuildOverrides::api_key_base_url"),
            "{relative} should use registry-owned provider build override shortcuts"
        );
        assert!(
            !source.contains("provider_build_overrides.insert(")
                && !source.contains("provider_build_overrides:"),
            "{relative} should not hand-roll provider_build_overrides HashMap plumbing"
        );
    }
}

#[test]
fn registry_options_default_is_create_provider_registry_default_source() {
    let root = crate_root();
    let entry_source =
        fs::read_to_string(root.join("src/registry/entry.rs")).expect("read registry entry source");
    let helpers_source = fs::read_to_string(root.join("src/registry/helpers.rs"))
        .expect("read registry helpers source");

    assert!(
        entry_source.contains("impl Default for RegistryOptions")
            && entry_source.contains("opts.unwrap_or_default()"),
        "RegistryOptions::default should be the single default source for create_provider_registry"
    );
    assert!(
        !entry_source.contains("Defaults: no middlewares, no interceptors"),
        "create_provider_registry should not keep a second hand-written default tuple"
    );
    assert!(
        helpers_source.contains("create_provider_registry(HashMap::new(), None)")
            && helpers_source.contains("..Default::default()"),
        "registry helpers should reuse RegistryOptions::default instead of spelling every default field"
    );
}

fn module_source<'a>(source: &'a str, marker: &str, next_marker: &str) -> &'a str {
    let start = source.find(marker).expect("module marker present");
    let rest = &source[start..];
    if let Some(end) = rest.find(next_marker) {
        &rest[..end]
    } else {
        rest
    }
}

#[test]
fn migrated_public_path_modules_use_registry_builder_shortcuts() {
    let root = crate_root();
    let source =
        fs::read_to_string(root.join("../siumai/tests/provider_public_path_parity_test.rs"))
            .expect("read provider public-path parity test");

    for (module_name, module_source, shortcut_marker) in [
        (
            "azure_public_path",
            module_source(
                &source,
                "mod azure_public_path",
                "#[cfg(feature = \"google\")]",
            ),
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "deepseek_public_path",
            module_source(
                &source,
                "mod deepseek_public_path",
                "#[cfg(feature = \"openai\")]",
            ),
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "vertex_maas_public_path",
            module_source(
                &source,
                "mod vertex_maas_public_path",
                "#[cfg(feature = \"deepseek\")]",
            ),
            ".with_provider_base_url_http_config_fetch(",
        ),
        (
            "ollama_public_path",
            module_source(
                &source,
                "mod ollama_public_path",
                "#[cfg(feature = \"minimaxi\")]",
            ),
            ".with_provider_base_url_fetch(",
        ),
        (
            "xai_public_path",
            module_source(&source, "mod xai_public_path", "mod __end_marker"),
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "bedrock_public_path",
            module_source(
                &source,
                "mod bedrock_public_path",
                "#[cfg(feature = \"anthropic\")]",
            ),
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "anthropic_public_path",
            module_source(
                &source,
                "mod anthropic_public_path",
                "#[cfg(feature = \"google-vertex\")]",
            ),
            ".with_provider_api_key_base_url_fetch(",
        ),
    ] {
        assert!(
            module_source.contains("RegistryBuilder") && module_source.contains(shortcut_marker),
            "{module_name} should route provider override setup through RegistryBuilder shortcuts"
        );
        assert!(
            !module_source.contains("provider_build_overrides.insert(")
                && !module_source.contains("RegistryOptions {")
                && !module_source.contains("create_provider_registry("),
            "{module_name} should not hand-roll raw RegistryOptions provider override plumbing"
        );
    }
}
