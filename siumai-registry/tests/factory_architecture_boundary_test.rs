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
            && factory_doc.contains("Legacy generic-client entry point")
            && factory_doc.contains("Prefer `language_model_text_with_ctx(...)`"),
        "ProviderFactory docs should keep the family-first contract explicit"
    );
    assert!(
        factory_doc.contains("#[deprecated")
            && factory_doc.contains("Use compat_language_client(...)")
            && factory_doc.contains("Use compat_completion_client(...)")
            && factory_doc.contains("Use compat_embedding_client(...)")
            && factory_doc.contains("Use compat_image_client(...)")
            && factory_doc.contains("Use compat_speech_client(...)")
            && factory_doc.contains("Use compat_transcription_client(...)")
            && factory_doc.contains("Use compat_video_client(...)")
            && factory_doc.contains("Use compat_reranking_client(...)"),
        "ProviderFactory legacy generic-client methods should be explicitly deprecated in favor of compat_* aliases"
    );
    assert_source_order(
        &factory_doc,
        "async fn language_model_text_with_ctx(",
        "Legacy generic-client entry point",
        "language family methods should be listed before the generic language client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn completion_model_family_with_ctx(",
        "async fn completion_model(",
        "completion family methods should be listed before the generic completion client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn embedding_model_family_with_ctx(",
        "async fn embedding_model(",
        "embedding family methods should be listed before the generic embedding client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn image_model_family_with_ctx(",
        "async fn image_model(",
        "image family methods should be listed before the generic image client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn speech_model_family_with_ctx(",
        "async fn speech_model(",
        "speech family methods should be listed before the generic speech client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn transcription_model_family_with_ctx(",
        "async fn transcription_model(",
        "transcription family methods should be listed before the generic transcription client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn video_model_family_with_ctx(",
        "async fn video_model(",
        "video family methods should be listed before the generic video client entry point",
    );
    assert_source_order(
        &factory_doc,
        "async fn reranking_model_family_with_ctx(",
        "async fn reranking_model(",
        "reranking family methods should be listed before the generic reranking client entry point",
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
            "{file} should use family model factories directly; LlmClient capability downcasts belong in compat_client.rs or extension-only language paths: {downcast_lines:?}"
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
