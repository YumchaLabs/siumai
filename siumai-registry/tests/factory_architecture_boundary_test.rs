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

fn collect_rust_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path.to_path_buf());
        }
        return;
    }

    for entry in fs::read_dir(path).expect("read source directory") {
        let entry = entry.expect("read source directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn factory_source_path(file_name: &str) -> PathBuf {
    crate_root()
        .join("src")
        .join("registry")
        .join("factories")
        .join(file_name)
}

fn read_factory_source(file_name: &str) -> String {
    let path = factory_source_path(file_name);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

#[test]
fn registry_helpers_cross_builtin_provider_descriptor_seam() {
    let root = crate_root();
    let helpers =
        fs::read_to_string(root.join("src/registry/helpers.rs")).expect("read registry helpers");
    let registry_mod =
        fs::read_to_string(root.join("src/registry/mod.rs")).expect("read registry module");
    let descriptor = fs::read_to_string(root.join("src/registry/provider_descriptor.rs"))
        .expect("read provider descriptor module");

    assert!(
        registry_mod.contains("pub(crate) mod provider_descriptor;"),
        "registry should expose an internal provider descriptor seam"
    );
    assert!(
        descriptor.contains("pub(crate) fn builtin_provider_default_model(")
            && descriptor.contains("pub(crate) fn builtin_provider_factory(")
            && descriptor.contains("pub(crate) fn register_enabled_builtin_provider_factories("),
        "provider_descriptor should own default-model lookup, factory resolution, and enabled built-in registration"
    );

    for expected_call in [
        "provider_descriptor::builtin_provider_default_model(",
        "provider_descriptor::openai_compatible_provider_factory(",
        "provider_descriptor::builtin_provider_factory(",
        "provider_descriptor::register_enabled_builtin_provider_factories(",
    ] {
        assert!(
            helpers.contains(expected_call),
            "registry helpers should cross provider_descriptor for `{expected_call}`"
        );
    }

    for forbidden in [
        "BuiltinProviderId::",
        "OpenAIProviderFactory",
        "AnthropicProviderFactory",
        "GeminiProviderFactory",
        "insert_builtin_provider_factory(",
    ] {
        assert!(
            !helpers.contains(forbidden),
            "registry helpers should not own provider facts after the descriptor split: `{forbidden}`"
        );
    }

    let provider_catalog =
        fs::read_to_string(root.join("src/provider_catalog.rs")).expect("read provider catalog");
    assert!(
        provider_catalog.contains("struct ProviderCatalogDescriptor")
            && provider_catalog.contains("fn into_provider_info(")
            && provider_catalog.contains("descriptor.into_provider_info("),
        "provider catalog views should cross a named descriptor before producing ProviderInfo"
    );
    assert!(
        !provider_catalog.contains("ProviderInfoBody"),
        "provider catalog should not keep the old shallow ProviderInfoBody carrier"
    );
}

fn async_method_source<'a>(source: &'a str, file_name: &str, method_name: &str) -> &'a str {
    let marker = format!("    async fn {method_name}(");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing `{method_name}` in {file_name}"));
    let tail = &source[start..];
    let search_start = marker.len();
    let search_tail = &tail[search_start..];
    let end = search_tail
        .find("\n    async fn ")
        .or_else(|| search_tail.find("\n    fn provider_id("))
        .map(|index| search_start + index)
        .unwrap_or(tail.len());

    &tail[..end]
}

fn assert_factory_family_method_stays_native(
    file_name: &str,
    method_name: &str,
    method_source: &str,
    composite_client: &str,
) {
    let composite_construction = format!("Arc::new({composite_client} {{");
    let forbidden_compat_calls = [
        "compat_language_client_with_ctx(",
        "compat_completion_client_with_ctx(",
        "compat_embedding_client_with_ctx(",
        "compat_image_client_with_ctx(",
        "compat_speech_client_with_ctx(",
        "compat_transcription_client_with_ctx(",
        "compat_reranking_client_with_ctx(",
    ];

    assert!(
        !method_source.contains(&composite_construction),
        "{file_name}::{method_name} should construct a native family model, not `{composite_client}`"
    );
    for forbidden in forbidden_compat_calls {
        assert!(
            !method_source.contains(forbidden),
            "{file_name}::{method_name} should not route stable family construction through `{forbidden}`"
        );
    }

    let downcast_lines = method_source
        .lines()
        .filter(|line| line.contains(".as_") && line.contains("_capability("))
        .collect::<Vec<_>>();
    assert!(
        downcast_lines.is_empty(),
        "{file_name}::{method_name} should not depend on LlmClient capability downcasts: {downcast_lines:?}"
    );
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
fn hybrid_provider_composite_clients_are_compat_only_adapters() {
    let cases: &[(&str, &str, &[&str])] = &[
        (
            "deepinfra.rs",
            "DeepInfraCompatCompositeClient",
            &[
                "language_model_text_with_ctx",
                "completion_model_family_with_ctx",
                "embedding_model_family_with_ctx",
                "image_model_family_with_ctx",
            ],
        ),
        (
            "fireworks.rs",
            "FireworksCompatCompositeClient",
            &[
                "language_model_text_with_ctx",
                "completion_model_family_with_ctx",
                "embedding_model_family_with_ctx",
                "image_model_family_with_ctx",
                "transcription_model_family_with_ctx",
            ],
        ),
        (
            "togetherai.rs",
            "TogetherAiCompatCompositeClient",
            &[
                "language_model_text_with_ctx",
                "completion_model_family_with_ctx",
                "embedding_model_family_with_ctx",
                "image_model_family_with_ctx",
                "speech_model_family_with_ctx",
                "transcription_model_family_with_ctx",
                "reranking_model_family_with_ctx",
            ],
        ),
    ];

    for &(file_name, composite_client, family_methods) in cases {
        let source = read_factory_source(file_name);
        let composite_construction = format!("Arc::new({composite_client} {{");
        assert!(
            source.contains(&format!("struct {composite_client}")),
            "{file_name} should make its private composite client role explicit"
        );
        assert!(
            !source.contains("UnifiedClient"),
            "{file_name} should not name the compatibility composite client as the target unified architecture"
        );
        assert_eq!(
            source.matches(&composite_construction).count(),
            1,
            "{file_name} should construct `{composite_client}` in exactly one place"
        );

        let compat_language =
            async_method_source(&source, file_name, "compat_language_client_with_ctx");
        assert!(
            compat_language.contains(&composite_construction),
            "{file_name} should construct `{composite_client}` only in compat_language_client_with_ctx"
        );

        for &method_name in family_methods {
            let method = async_method_source(&source, file_name, method_name);
            assert_factory_family_method_stays_native(
                file_name,
                method_name,
                method,
                composite_client,
            );
        }
    }
}

#[test]
fn hybrid_provider_image_extras_use_native_extension_clients() {
    struct Case<'a> {
        file_name: &'a str,
        helper_call: &'a str,
    }

    let cases = [
        Case {
            file_name: "deepinfra.rs",
            helper_call: "build_image_client_arc(model_id, ctx).await?",
        },
        Case {
            file_name: "fireworks.rs",
            helper_call: "build_image_client_arc(model_id, ctx).await?",
        },
        Case {
            file_name: "togetherai.rs",
            helper_call: "build_image_client_arc(model_id, ctx)?",
        },
    ];

    for case in cases {
        let source = read_factory_source(case.file_name);
        let method = async_method_source(&source, case.file_name, "image_extras_with_ctx");
        assert!(
            method.contains("let client: Arc<dyn ImageExtras> =")
                && method.contains(case.helper_call),
            "{}::image_extras_with_ctx should return the native image extras client through the extension facet",
            case.file_name
        );

        for forbidden in [
            "compat_image_client_with_ctx(",
            "ClientBackedImageExtras",
            ".as_image_extras()",
            "CompatCompositeClient",
        ] {
            assert!(
                !method.contains(forbidden),
                "{}::image_extras_with_ctx should not fall back through generic-client adapter glue `{forbidden}`",
                case.file_name
            );
        }
    }
}

#[test]
fn provider_native_extension_hooks_bypass_generic_client_adapters() {
    struct Case<'a> {
        file_name: &'a str,
        method_name: &'a str,
        trait_name: &'a str,
        helper_call: &'a str,
        forbidden_adapter: &'a str,
        forbidden_downcast: &'a str,
    }

    let cases = [
        Case {
            file_name: "azure.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "openai.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "openai.rs",
            method_name: "skills_capability_with_ctx",
            trait_name: "SkillsCapability",
            helper_call: "build_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedSkillsCapability",
            forbidden_downcast: ".as_skills_capability()",
        },
        Case {
            file_name: "anthropic.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_text_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "anthropic.rs",
            method_name: "skills_capability_with_ctx",
            trait_name: "SkillsCapability",
            helper_call: "build_text_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedSkillsCapability",
            forbidden_downcast: ".as_skills_capability()",
        },
        Case {
            file_name: "gemini.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_text_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "xai.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_text_family_model_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "minimaxi.rs",
            method_name: "file_management_capability_with_ctx",
            trait_name: "FileManagementCapability",
            helper_call: "build_typed_client_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedFileManagementCapability",
            forbidden_downcast: ".as_file_management_capability()",
        },
        Case {
            file_name: "minimaxi.rs",
            method_name: "music_generation_capability_with_ctx",
            trait_name: "MusicGenerationCapability",
            helper_call: "build_typed_client_arc(model_id, ctx).await?",
            forbidden_adapter: "ClientBackedMusicGenerationCapability",
            forbidden_downcast: ".as_music_generation_capability()",
        },
    ];

    for case in cases {
        let source = read_factory_source(case.file_name);
        let method = async_method_source(&source, case.file_name, case.method_name);
        let trait_projection = format!("let client: Arc<dyn {}> =", case.trait_name);
        assert!(
            method.contains(&trait_projection) && method.contains(case.helper_call),
            "{}::{} should return the provider-native typed client as Arc<dyn {}>",
            case.file_name,
            case.method_name,
            case.trait_name
        );

        for forbidden in [
            "compat_language_client_with_ctx(",
            case.forbidden_adapter,
            case.forbidden_downcast,
        ] {
            assert!(
                !method.contains(forbidden),
                "{}::{} should not fall back through generic-client adapter glue `{forbidden}`",
                case.file_name,
                case.method_name
            );
        }
    }
}

#[test]
fn openai_compatible_factory_centralizes_checked_family_projection_glue() {
    let source = read_factory_source("openai_compatible.rs");

    assert_eq!(
        source
            .matches("async fn build_checked_text_family_model_arc(")
            .count(),
        1,
        "OpenAI-compatible factory should have one checked typed-client Arc projection helper"
    );
    assert_eq!(
        source.matches("Ok(Arc::new(client))").count(),
        1,
        "OpenAI-compatible factory should not duplicate Arc::new(client) projection in every family/compat method"
    );

    for (method_name, capability) in [
        ("compat_language_client_with_ctx", "chat"),
        ("language_model_text_with_ctx", "chat"),
        ("compat_embedding_client_with_ctx", "embedding"),
        ("embedding_model_family_with_ctx", "embedding"),
        ("compat_completion_client_with_ctx", "completion"),
        ("completion_model_family_with_ctx", "completion"),
        ("compat_image_client_with_ctx", "image_generation"),
        ("image_model_family_with_ctx", "image_generation"),
        ("compat_reranking_client_with_ctx", "rerank"),
        ("reranking_model_family_with_ctx", "rerank"),
        ("compat_speech_client_with_ctx", "speech"),
        ("speech_model_family_with_ctx", "speech"),
        ("compat_transcription_client_with_ctx", "transcription"),
        ("transcription_model_family_with_ctx", "transcription"),
    ] {
        let method = async_method_source(&source, "openai_compatible.rs", method_name);
        let helper_call = format!("build_checked_text_family_model_arc(\"{capability}\"");
        assert!(
            method.contains(&helper_call),
            "openai_compatible.rs::{method_name} should delegate checked capability + typed-client projection to `{helper_call}`"
        );
        assert!(
            !method.contains("ensure_capability(")
                && !method.contains("build_text_family_model_with_ctx(")
                && !method.contains("Arc::new(client)"),
            "openai_compatible.rs::{method_name} should not reintroduce local checked projection glue"
        );
    }
}

#[test]
fn openai_factory_centralizes_family_projection_glue() {
    let source = read_factory_source("openai.rs");

    assert_eq!(
        source.matches("async fn build_family_model_arc(").count(),
        1,
        "OpenAI factory should have one typed-client Arc projection helper"
    );
    assert_eq!(
        source.matches("Ok(Arc::new(client))").count(),
        1,
        "OpenAI factory should not duplicate Arc::new(client) projection in every family/compat method"
    );

    for method_name in [
        "compat_language_client_with_ctx",
        "language_model_text_with_ctx",
        "compat_completion_client_with_ctx",
        "completion_model_family_with_ctx",
        "compat_embedding_client_with_ctx",
        "embedding_model_family_with_ctx",
        "compat_image_client_with_ctx",
        "image_model_family_with_ctx",
        "compat_speech_client_with_ctx",
        "speech_model_family_with_ctx",
        "compat_transcription_client_with_ctx",
        "transcription_model_family_with_ctx",
    ] {
        let method = async_method_source(&source, "openai.rs", method_name);
        assert!(
            method.contains("build_family_model_arc(model_id, ctx)"),
            "openai.rs::{method_name} should delegate typed-client projection to build_family_model_arc"
        );
        assert!(
            !method.contains("build_family_model_with_ctx(")
                && !method.contains("Arc::new(client)"),
            "openai.rs::{method_name} should not reintroduce local projection glue"
        );
    }
}

#[test]
fn promoted_openai_compatible_vendor_factories_centralize_projection_glue() {
    struct Case<'a> {
        file_name: &'a str,
        image_client: &'a str,
        text_projection_count: usize,
        image_projection_count: usize,
        async_image_projection: bool,
        text_methods: &'a [&'a str],
        image_methods: &'a [&'a str],
        rerank_methods: &'a [&'a str],
    }

    let cases = [
        Case {
            file_name: "deepinfra.rs",
            image_client: "DeepInfraImageClient",
            text_projection_count: 1,
            image_projection_count: 1,
            async_image_projection: true,
            text_methods: &[
                "language_model_text_with_ctx",
                "compat_completion_client_with_ctx",
                "completion_model_family_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
            ],
            image_methods: &[
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
            ],
            rerank_methods: &[],
        },
        Case {
            file_name: "fireworks.rs",
            image_client: "FireworksImageClient",
            text_projection_count: 1,
            image_projection_count: 1,
            async_image_projection: true,
            text_methods: &[
                "language_model_text_with_ctx",
                "compat_completion_client_with_ctx",
                "completion_model_family_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
                "compat_transcription_client_with_ctx",
                "transcription_model_family_with_ctx",
            ],
            image_methods: &[
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
            ],
            rerank_methods: &[],
        },
        Case {
            file_name: "togetherai.rs",
            image_client: "TogetherAiImageClient",
            text_projection_count: 1,
            image_projection_count: 1,
            async_image_projection: false,
            text_methods: &[
                "language_model_text_with_ctx",
                "compat_completion_client_with_ctx",
                "completion_model_family_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
                "compat_speech_client_with_ctx",
                "speech_model_family_with_ctx",
                "compat_transcription_client_with_ctx",
                "transcription_model_family_with_ctx",
            ],
            image_methods: &[
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
            ],
            rerank_methods: &[
                "compat_reranking_client_with_ctx",
                "reranking_model_family_with_ctx",
            ],
        },
    ];

    for case in cases {
        let source = read_factory_source(case.file_name);
        let image_projection = format!("Ok(Arc::new({}::from_text_client(", case.image_client);

        assert_eq!(
            source.matches("async fn build_text_client_arc(").count(),
            case.text_projection_count,
            "{} should centralize text typed-client Arc projection in one helper",
            case.file_name
        );
        if case.async_image_projection {
            assert_eq!(
                source.matches("async fn build_image_client_arc(").count(),
                case.image_projection_count,
                "{} should centralize provider-owned image Arc projection in one async helper",
                case.file_name
            );
            assert_eq!(
                source.matches(&image_projection).count(),
                case.image_projection_count,
                "{} should not duplicate provider-owned image projection outside build_image_client_arc",
                case.file_name
            );
        } else {
            assert_eq!(
                source.matches("fn build_image_client_arc(").count(),
                case.image_projection_count,
                "{} should centralize provider-owned image Arc projection in one sync helper",
                case.file_name
            );
            assert_eq!(
                source
                    .matches("TogetherAiImageClient::from_config(")
                    .count()
                    + source
                        .matches("TogetherAiImageClient::with_http_client(")
                        .count(),
                2,
                "{} should construct provider-owned TogetherAI image clients directly through the provider crate",
                case.file_name
            );
            assert!(
                !source.contains("struct TogetherAiImageClient")
                    && !source.contains("impl ImageGenerationCapability for TogetherAiImageClient")
                    && !source.contains("impl ImageExtras for TogetherAiImageClient")
                    && !source.contains("build_generation_body(")
                    && !source.contains("build_edit_body(")
                    && !source.contains("execute_json_request("),
                "{} should not own TogetherAI image runtime or protocol mapping in the registry factory",
                case.file_name
            );
        }

        for &method_name in case.text_methods {
            let method = async_method_source(&source, case.file_name, method_name);
            assert!(
                method.contains("build_text_client_arc(model_id, ctx).await?"),
                "{}::{method_name} should delegate text typed-client projection to build_text_client_arc",
                case.file_name
            );
            assert!(
                !method.contains("build_text_client_with_ctx(")
                    && !method.contains("Arc::new(client)")
                    && !method.contains("from_text_client("),
                "{}::{method_name} should not reintroduce local text/image projection glue",
                case.file_name
            );
        }

        for &method_name in case.image_methods {
            let method = async_method_source(&source, case.file_name, method_name);
            let helper_call = if case.async_image_projection {
                "build_image_client_arc(model_id, ctx).await?"
            } else {
                "build_image_client_arc(model_id, ctx)?"
            };
            assert!(
                method.contains(helper_call),
                "{}::{method_name} should delegate provider-owned image projection to build_image_client_arc",
                case.file_name
            );
            assert!(
                !method.contains("build_text_client_with_ctx(")
                    && !method.contains("Arc::new(")
                    && !method.contains("from_text_client(")
                    && !method.contains("TogetherAiImageClient::from_config(")
                    && !method.contains("TogetherAiImageClient::with_http_client("),
                "{}::{method_name} should not reintroduce local image projection glue",
                case.file_name
            );
        }

        if !case.rerank_methods.is_empty() {
            assert_eq!(
                source.matches("fn build_rerank_client_arc(").count(),
                1,
                "{} should centralize provider-owned rerank Arc projection in one helper",
                case.file_name
            );
            for &method_name in case.rerank_methods {
                let method = async_method_source(&source, case.file_name, method_name);
                assert!(
                    method.contains("build_rerank_client_arc(model_id, ctx)?"),
                    "{}::{method_name} should delegate provider-owned rerank projection to build_rerank_client_arc",
                    case.file_name
                );
                assert!(
                    !method.contains("build_native_rerank_client_with_ctx(")
                        && !method.contains("Arc::new("),
                    "{}::{method_name} should not reintroduce local rerank projection glue",
                    case.file_name
                );
            }
        }
    }
}

#[test]
fn togetherai_provider_crate_owns_image_runtime() {
    let factory_source = read_factory_source("togetherai.rs");
    let provider_root = crate_root().join("../siumai-provider-togetherai/src/providers/togetherai");
    let provider_mod =
        fs::read_to_string(provider_root.join("mod.rs")).expect("read TogetherAI provider module");
    let provider_image =
        fs::read_to_string(provider_root.join("image.rs")).expect("read TogetherAI image runtime");
    let provider_facade =
        fs::read_to_string(crate_root().join("../siumai/src/provider_ext/togetherai.rs"))
            .expect("read TogetherAI facade provider extension");

    assert!(
        provider_mod.contains("mod image;")
            && provider_mod.contains("pub use image::TogetherAiImageClient;"),
        "TogetherAI provider crate should expose provider-owned image client from its image Module"
    );
    assert!(
        provider_image.contains("pub struct TogetherAiImageClient")
            && provider_image.contains("impl ImageGenerationCapability for TogetherAiImageClient")
            && provider_image.contains("impl ImageExtras for TogetherAiImageClient")
            && provider_image.contains("build_generation_body(")
            && provider_image.contains("build_edit_body(")
            && provider_image.contains("execute_json_request("),
        "TogetherAI provider crate should own image runtime, request mapping, response parsing, and HTTP execution"
    );
    assert!(
        provider_facade.contains("TogetherAiImageClient"),
        "facade provider_ext::togetherai should expose the provider-owned image client"
    );
    for forbidden in [
        "struct TogetherAiImageClient",
        "impl ImageGenerationCapability for TogetherAiImageClient",
        "impl ImageExtras for TogetherAiImageClient",
        "build_generation_body(",
        "build_edit_body(",
        "execute_json_request(",
        "struct TogetherAiImageResponse",
        "generated_image_from_together_item(",
    ] {
        assert!(
            !factory_source.contains(forbidden),
            "TogetherAI registry factory should not own image runtime detail `{forbidden}`"
        );
    }
    assert!(
        factory_source.contains("TogetherAiImageClient::from_config(")
            && factory_source.contains("TogetherAiImageClient::with_http_client(")
            && factory_source.contains("build_image_client_arc(model_id, ctx)?"),
        "TogetherAI registry factory should only build/project provider-owned image clients"
    );
}

#[test]
fn builtin_provider_factories_centralize_typed_client_arc_projection() {
    struct Case<'a> {
        file_name: &'a str,
        helper_decl: &'a str,
        helper_call: &'a str,
        forbidden_builder_call: &'a str,
        methods: &'a [&'a str],
    }

    let cases = [
        Case {
            file_name: "azure.rs",
            helper_decl: "async fn build_family_model_arc(",
            helper_call: "self.build_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_completion_client_with_ctx",
                "completion_model_family_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
                "compat_speech_client_with_ctx",
                "speech_model_family_with_ctx",
                "compat_transcription_client_with_ctx",
                "transcription_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "anthropic.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
            ],
        },
        Case {
            file_name: "gemini.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
                "compat_video_client_with_ctx",
                "video_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "xai.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
                "compat_speech_client_with_ctx",
                "speech_model_family_with_ctx",
                "compat_video_client_with_ctx",
                "video_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "groq.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_speech_client_with_ctx",
                "speech_model_family_with_ctx",
                "compat_transcription_client_with_ctx",
                "transcription_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "deepseek.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
            ],
        },
        Case {
            file_name: "bedrock.rs",
            helper_decl: "fn build_typed_client_arc(",
            helper_call: "build_typed_client_arc(model_id, ctx)?",
            forbidden_builder_call: "build_typed_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "compat_reranking_client_with_ctx",
                "compat_embedding_client_with_ctx",
                "compat_image_client_with_ctx",
                "language_model_text_with_ctx",
                "embedding_model_family_with_ctx",
                "image_model_family_with_ctx",
                "reranking_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "cohere.rs",
            helper_decl: "fn build_typed_client_arc(",
            helper_call: "build_typed_client_arc(model_id, ctx)?",
            forbidden_builder_call: "build_typed_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "compat_embedding_client_with_ctx",
                "compat_reranking_client_with_ctx",
                "language_model_text_with_ctx",
                "embedding_model_family_with_ctx",
                "reranking_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "google_vertex.rs",
            helper_decl: "async fn build_typed_client_arc(",
            helper_call: "self.build_typed_client_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_typed_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
                "compat_video_client_with_ctx",
                "video_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "minimaxi.rs",
            helper_decl: "async fn build_typed_client_arc(",
            helper_call: "self.build_typed_client_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_typed_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_image_client_with_ctx",
                "image_model_family_with_ctx",
                "compat_speech_client_with_ctx",
                "speech_model_family_with_ctx",
                "compat_video_client_with_ctx",
                "video_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "ollama.rs",
            helper_decl: "async fn build_text_family_model_arc(",
            helper_call: "self.build_text_family_model_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_family_model_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "anthropic_vertex.rs",
            helper_decl: "async fn build_typed_client_arc(",
            helper_call: "build_typed_client_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_typed_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
            ],
        },
        Case {
            file_name: "vertex_maas.rs",
            helper_decl: "async fn build_text_client_arc(",
            helper_call: "build_text_client_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
                "compat_completion_client_with_ctx",
                "completion_model_family_with_ctx",
                "compat_embedding_client_with_ctx",
                "embedding_model_family_with_ctx",
            ],
        },
        Case {
            file_name: "google_vertex_xai.rs",
            helper_decl: "async fn build_text_client_arc(",
            helper_call: "build_text_client_arc(model_id, ctx).await?",
            forbidden_builder_call: "build_text_client_with_ctx(",
            methods: &[
                "compat_language_client_with_ctx",
                "language_model_text_with_ctx",
            ],
        },
    ];

    for case in cases {
        let source = read_factory_source(case.file_name);
        assert_eq!(
            source.matches(case.helper_decl).count(),
            1,
            "{} should expose exactly one typed-client Arc projection helper `{}`",
            case.file_name,
            case.helper_decl
        );
        assert_eq!(
            source.matches("Ok(Arc::new(client))").count(),
            1,
            "{} should keep `Arc::new(client)` projection only in the helper",
            case.file_name
        );

        for &method_name in case.methods {
            let method = async_method_source(&source, case.file_name, method_name);
            assert!(
                method.contains(case.helper_call),
                "{}::{method_name} should delegate typed-client Arc projection to `{}`",
                case.file_name,
                case.helper_call
            );
            assert!(
                !method.contains(case.forbidden_builder_call)
                    && !method.contains("Arc::new(")
                    && !method.contains("self.compat_language_client_with_ctx("),
                "{}::{method_name} should not reintroduce local typed-client projection or compat-language delegation glue",
                case.file_name
            );
        }
    }
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
        "pub use siumai_core::client::LlmClient",
        "pub use siumai_core::{LlmError, client",
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
        lib_rs.contains("pub mod compat {")
            && lib_rs.contains("pub use siumai_core::compat::client::{ClientWrapper, LlmClient};"),
        "registry should expose generic client compatibility types through siumai_registry::compat::client"
    );
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
fn registry_generic_client_imports_are_compat_scoped() {
    let root = crate_root();
    let src = root.join("src");
    let mut rust_files = Vec::new();
    collect_rust_files(&src, &mut rust_files);

    let allowed_legacy_alias_files = ["src/lib.rs"];
    let mut violations = Vec::new();
    for file in rust_files {
        let relative = file
            .strip_prefix(&root)
            .expect("registry source under crate root")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(&file).expect("read registry source");
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .unwrap_or(source.as_str());

        for forbidden in [
            "use crate::client::LlmClient",
            "crate::client::LlmClient",
            "use siumai_core::client::LlmClient",
            "siumai_core::client::LlmClient",
        ] {
            if production_source.contains(forbidden)
                && !allowed_legacy_alias_files.contains(&relative.as_str())
            {
                violations.push(format!("{relative}: `{forbidden}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "registry production code should import generic LlmClient through explicit compat/client aliases:\n{}",
        violations.join("\n")
    );
}

#[test]
fn provider_compatibility_factory_is_method_style_only() {
    let root = crate_root();
    let src = root.join("src");
    let mut rust_files = Vec::new();
    collect_rust_files(&src, &mut rust_files);

    let allowed_production_files = ["src/registry/entry/factory.rs", "src/provider/build.rs"];
    let mut violations = Vec::new();
    for file in rust_files {
        let relative = file
            .strip_prefix(&root)
            .expect("registry source under crate root")
            .to_string_lossy()
            .replace('\\', "/");
        if relative.ends_with("_tests.rs") || relative.ends_with("contract_tests.rs") {
            continue;
        }

        let source = fs::read_to_string(&file).expect("read registry source");
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .unwrap_or(source.as_str());

        for marker in [
            "ProviderCompatibilityFactory",
            "compatibility_facet_from_provider_factory",
        ] {
            if production_source.contains(marker)
                && !allowed_production_files.contains(&relative.as_str())
            {
                violations.push(format!("{relative}: `{marker}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "ProviderCompatibilityFactory should be confined to the registry facet definition and historical SiumaiBuilder method-style compatibility construction:\n{}",
        violations.join("\n")
    );

    let builder_source = fs::read_to_string(root.join("src").join("provider").join("build.rs"))
        .expect("read provider build source");
    let builder_production = builder_source
        .split("\n#[cfg(test)]")
        .next()
        .expect("production provider build source");
    assert!(
        builder_production.contains("build_default_client_with_capabilities")
            && builder_production.contains("Arc<dyn LlmClient>")
            && builder_production.contains("compatibility_facet_from_provider_factory(factory)"),
        "SiumaiBuilder should be the only production consumer that adapts ProviderFactory into ProviderCompatibilityFactory for a generic LlmClient"
    );

    let adr = fs::read_to_string(root.join("../docs/adr/0007-llmclient-demotion-policy.md"))
        .expect("read ADR-0007");
    let migration_doc =
        fs::read_to_string(root.join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read beta.7 migration guide");
    let public_surface_doc =
        fs::read_to_string(root.join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    for (name, source) in [
        ("ADR-0007", adr.as_str()),
        ("migration-0.11.0-beta.7.md", migration_doc.as_str()),
        ("public-surface.md", public_surface_doc.as_str()),
    ] {
        assert!(
            source.contains("delete `ProviderCompatibilityFactory`")
                && source.contains("method-style")
                && source.contains("extension")
                && source.contains("family-native"),
            "{name} should state deletion gates for the generic-client compatibility factory"
        );
    }
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
fn registry_client_backed_family_model_adapters_are_removed() {
    let root = crate_root();
    let entry_dir = root.join("src").join("registry").join("entry");

    assert!(
        !entry_dir.join("compat_client.rs").exists(),
        "registry entry should not keep the removed ClientBacked*Model compatibility module file"
    );
    assert!(
        !entry_dir.join("compat_client").exists(),
        "registry entry should not keep a removed ClientBacked*Model compatibility module directory"
    );

    let mut rust_files = Vec::new();
    collect_rust_files(&entry_dir, &mut rust_files);
    for file in rust_files {
        let source = fs::read_to_string(&file).expect("read registry entry source");
        for forbidden in [
            "ClientBackedLanguageModel",
            "ClientBackedCompletionModel",
            "ClientBackedEmbeddingModel",
            "ClientBackedImageModel",
            "ClientBackedSpeechModel",
            "ClientBackedTranscriptionModel",
            "ClientBackedVideoModel",
            "ClientBackedRerankingModel",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} should not reintroduce removed registry family compatibility adapter `{forbidden}`",
                file.display()
            );
        }
    }

    let audit = fs::read_to_string(
        root.join("../docs/workstreams/fearless-architecture-convergence/compatibility-audit.md"),
    )
    .expect("read compatibility audit");
    assert!(
        audit.contains("Registry Compatibility Family Adapters - Removed"),
        "compatibility audit should record the removed ClientBacked*Model family adapter state"
    );
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
fn production_factories_do_not_call_legacy_broad_build_client_helpers() {
    let factories_dir = crate_root().join("src").join("registry").join("factories");
    let forbidden_helpers = [
        "build_openai_client",
        "build_openai_chat_completions_client",
        "build_openai_compatible_client",
        "build_anthropic_client",
        "build_gemini_client",
        "build_anthropic_vertex_client",
        "build_google_vertex_client",
        "build_ollama_client",
        "build_minimaxi_client",
    ];
    let mut violations = Vec::new();

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
        for helper in forbidden_helpers {
            let qualified_call = format!("crate::registry::factory::{helper}(");
            if source.contains(&qualified_call) {
                violations.push(format!("{file_name}: {qualified_call}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "production ProviderFactory implementations should construct provider-owned clients through private typed builders/family methods, not legacy broad registry::factory build helpers:\n{}",
        violations.join("\n")
    );
}

#[test]
fn production_factories_use_internal_typed_builders_not_legacy_factory_module() {
    let factories_dir = crate_root().join("src").join("registry").join("factories");
    let forbidden_legacy_factory_paths = [
        "crate::registry::factory::build_openai_compatible_typed_client(",
        "crate::registry::factory::build_gemini_typed_client(",
        "crate::registry::factory::build_anthropic_vertex_typed_client(",
        "crate::registry::factory::build_google_vertex_typed_client(",
        "crate::registry::factory::OpenAiChatApiMode",
    ];
    let mut violations = Vec::new();

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
        for forbidden in forbidden_legacy_factory_paths {
            if source.contains(forbidden) {
                violations.push(format!("{file_name}: {forbidden}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "production ProviderFactory implementations should use registry::typed_builders for typed provider construction, not the public legacy registry::factory module:\n{}",
        violations.join("\n")
    );

    let registry_mod = fs::read_to_string(crate_root().join("src").join("registry").join("mod.rs"))
        .expect("read registry module");
    assert!(
        registry_mod.contains("mod typed_builders;"),
        "registry should keep typed provider construction helpers in an internal typed_builders module"
    );

    let legacy_factory =
        fs::read_to_string(crate_root().join("src").join("registry").join("factory.rs"))
            .expect("read legacy registry factory module");
    assert!(
        legacy_factory.contains("pub use crate::registry::typed_builders::OpenAiChatApiMode;"),
        "registry::factory should re-export OpenAiChatApiMode as a compatibility wrapper, not own the enum"
    );
    for helper in [
        "build_openai_compatible_typed_client",
        "build_gemini_typed_client",
        "build_anthropic_vertex_typed_client",
        "build_google_vertex_typed_client",
    ] {
        assert!(
            legacy_factory.contains(&format!("pub async fn {helper}("))
                && legacy_factory.contains(&format!("crate::registry::typed_builders::{helper}(")),
            "registry::factory::{helper} should be a compatibility wrapper around registry::typed_builders::{helper}"
        );
    }
}

#[test]
fn legacy_registry_factory_build_helpers_are_deprecated_compatibility_shims() {
    let source = fs::read_to_string(crate_root().join("src").join("registry").join("factory.rs"))
        .expect("read legacy registry factory module");
    let broad_helpers = [
        "build_openai_client",
        "build_openai_chat_completions_client",
        "build_openai_compatible_client",
        "build_anthropic_client",
        "build_gemini_client",
        "build_anthropic_vertex_client",
        "build_google_vertex_client",
        "build_ollama_client",
        "build_minimaxi_client",
    ];

    assert!(
        source.contains("compatibility-only"),
        "registry::factory module docs should classify broad build helpers as compatibility-only"
    );

    for helper in broad_helpers {
        let marker = format!("pub async fn {helper}(");
        let Some(start) = source.find(&marker) else {
            continue;
        };
        let prefix_start = start.saturating_sub(256);
        let prefix = &source[prefix_start..start];
        assert!(
            prefix.contains("#[deprecated("),
            "legacy broad helper `{helper}` should be deprecated instead of advertised as a primary provider-construction path"
        );
    }
}

#[test]
fn all_providers_feature_includes_google_vertex_family() {
    let manifest =
        fs::read_to_string(crate_root().join("Cargo.toml")).expect("read siumai-registry manifest");
    let start = manifest
        .find("all-providers = [")
        .expect("all-providers feature should exist");
    let tail = &manifest[start..];
    let end = tail.find("\n]").expect("all-providers array should end");
    let all_providers = &tail[..end];

    assert!(
        all_providers.contains("\"google-vertex\""),
        "all-providers should include google-vertex because the OpenAI-compatible catalog contains Google Vertex xAI entries that require the google-vertex factory"
    );
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
        "SpeechModelHandle::text_to_speech(...)",
        "execute_json_request_with_headers(...)",
        "siumai_core::utils::vertex",
        "Provider extension deprecated option/metadata aliases",
    ] {
        assert!(
            audit.contains(surface),
            "compatibility audit should categorize deprecated public surface `{surface}`"
        );
    }

    for category in ["keep, time-bounded", "remove", "removed", "move"] {
        assert!(
            audit.contains(category),
            "compatibility audit should include `{category}` decisions"
        );
    }
}

#[test]
fn compatibility_audit_does_not_keep_removed_providerfactory_methods_alive() {
    let root = crate_root();
    let checked_docs = [
        (
            "compatibility-audit.md",
            root.join(
                "../docs/workstreams/fearless-architecture-convergence/compatibility-audit.md",
            ),
        ),
        (
            "milestones.md",
            root.join("../docs/workstreams/fearless-architecture-convergence/milestones.md"),
        ),
        (
            "todo.md",
            root.join("../docs/workstreams/fearless-architecture-convergence/todo.md"),
        ),
    ];

    for (name, path) in checked_docs {
        let source = fs::read_to_string(path).expect("read architecture convergence doc");
        for stale in [
            "remain deprecated",
            "remain as source-compatible wrappers",
            "explicit deprecated source-compatibility wrappers",
            "source-compatible wrappers for now",
            "old generic-client `*_model*` trait methods are deprecated",
        ] {
            assert!(
                !source.contains(stale),
                "{name} should not describe removed ProviderFactory generic-client methods as still-kept deprecated wrappers: {stale}"
            );
        }
    }

    let audit = fs::read_to_string(
        root.join("../docs/workstreams/fearless-architecture-convergence/compatibility-audit.md"),
    )
    .expect("read compatibility audit");
    assert!(
        audit.contains(
            "The old generic `*_model*` and `*_model_with_ctx` trait methods have been removed"
        ),
        "compatibility audit should record the current ProviderFactory contract, not a stale deprecated-wrapper state"
    );
}

#[test]
fn dedicated_vision_compatibility_surface_is_removed() {
    let root = crate_root();

    for relative in [
        "../siumai-core/src/client.rs",
        "../siumai-core/src/core/mod.rs",
        "../siumai-core/src/traits.rs",
        "../siumai-core/src/traits/vision.rs",
        "../siumai-registry/src/provider/mod.rs",
        "../siumai-registry/src/provider/proxies.rs",
        "../siumai-registry/src/provider/siumai.rs",
        "../siumai-registry/src/provider/siumai_builder.rs",
        "../siumai-registry/src/provider/siumai/llm_client.rs",
    ] {
        let path = root.join(relative);
        if !path.exists() {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read source");
        for forbidden in [
            "VisionCapability",
            "VisionCapabilityProxy",
            "as_vision_capability",
            "vision_capability",
            "with_vision",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative} should not expose the removed dedicated vision compatibility surface `{forbidden}`"
            );
        }
    }

    let spec_image = fs::read_to_string(root.join("../siumai-spec/src/types/image.rs"))
        .expect("read image types");
    for forbidden in [
        "pub type ImageGenRequest",
        "pub type ImageResponse",
        "pub type VisionRequest",
        "pub type VisionResponse",
    ] {
        assert!(
            !spec_image.contains(forbidden),
            "vision-only legacy alias `{forbidden}` should be removed from siumai-spec image types"
        );
    }

    let factories_dir = root.join("src").join("registry").join("factories");
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
            !source.contains("as_vision_capability"),
            "{file_name} should not forward the removed vision downcast"
        );
    }
}

#[test]
fn static_header_json_executor_compatibility_surface_is_removed() {
    let root = crate_root();

    for relative in [
        "../siumai-core/src/execution/executors/common.rs",
        "../siumai-core/src/execution/executors/http_request/json.rs",
        "../siumai-core/src/execution/executors/http_request/mod.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("read executor source");
        for forbidden in [
            "execute_json_request_with_headers",
            "StaticHeadersSpec",
            "static_headers",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative} should not expose the removed static-header JSON executor compatibility surface `{forbidden}`"
            );
        }
    }
}

#[test]
fn registry_speech_handle_inherent_text_to_speech_alias_is_removed() {
    let source = fs::read_to_string(crate_root().join("src/registry/entry/handles/audio.rs"))
        .expect("read audio handle source");

    for forbidden in [
        "pub async fn text_to_speech(",
        "Use the AudioCapability trait method directly",
    ] {
        assert!(
            !source.contains(forbidden),
            "SpeechModelHandle should not expose the removed inherent text_to_speech alias `{forbidden}`"
        );
    }
}

#[test]
fn siumai_builder_provider_type_alias_is_removed() {
    let source = fs::read_to_string(crate_root().join("src/provider/siumai_builder.rs"))
        .expect("read siumai builder source");

    for forbidden in [
        "pub fn provider(",
        "provider_type: ProviderType",
        "maps `ProviderType`",
    ] {
        assert!(
            !source.contains(forbidden),
            "SiumaiBuilder should not expose the removed ProviderType-based provider alias `{forbidden}`"
        );
    }
}

#[test]
fn provider_catalog_lookup_is_provider_id_first() {
    let source = fs::read_to_string(crate_root().join("src/provider_catalog.rs"))
        .expect("read provider catalog source");

    assert!(
        source.contains("pub provider_id: Cow<'static, str>"),
        "ProviderInfo should expose provider_id as the primary open provider identity"
    );
    for forbidden in [
        "let ptype = ProviderType::from_name",
        "match ptype",
        "get_provider_info(&ProviderType::from_name",
        "is_model_supported(&ProviderType::from_name",
    ] {
        assert!(
            !source.contains(forbidden),
            "provider catalog should not route primary lookup through closed ProviderType classification: `{forbidden}`"
        );
    }
    assert!(
        source.contains("CatalogProviderId::parse"),
        "provider catalog should route built-in metadata by registry-owned provider-id classification"
    );

    let lookup_start = source
        .find("pub fn get_provider_info_by_id(")
        .expect("provider-id lookup function should exist");
    let lookup_tail = &source[lookup_start..];
    let lookup_end = lookup_tail
        .find("/// Check if a model is supported by the provider")
        .expect("provider-id lookup function should precede model support helper");
    let lookup_source = &lookup_tail[..lookup_end];
    assert!(
        !lookup_source.contains("ProviderType::from_name"),
        "get_provider_info_by_id should resolve provider ids/aliases directly instead of classifying through ProviderType"
    );

    let model_lookup_start = source
        .find("pub fn is_model_supported_by_id(")
        .expect("provider-id model lookup function should exist");
    let model_lookup_tail = &source[model_lookup_start..];
    let model_lookup_end = model_lookup_tail
        .find("#[cfg(test)]")
        .unwrap_or(model_lookup_tail.len());
    let model_lookup_source = &model_lookup_tail[..model_lookup_end];
    assert!(
        !model_lookup_source.contains("ProviderType::from_name"),
        "is_model_supported_by_id should reuse provider-id lookup instead of classifying through ProviderType"
    );
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
    assert!(
        migration_doc.contains("siumai_registry::LlmClient")
            && migration_doc.contains("siumai_registry::compat::client::LlmClient")
            && public_surface_doc.contains("siumai_registry::compat::client::LlmClient"),
        "docs should spell out the registry-root LlmClient migration path"
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

    let public_path_source = provider_public_path_combined_source();
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
        "crate::provider_utils::builder_helpers::get_effective_model",
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
fn minimaxi_and_ollama_use_curated_models_not_legacy_model_constants() {
    let root = crate_root();

    for relative in [
        "../siumai-provider-minimaxi/src/providers/minimaxi/model_constants.rs",
        "../siumai-provider-ollama/src/providers/ollama/model_constants.rs",
    ] {
        assert!(
            !root.join(relative).exists(),
            "{relative} should be removed; MiniMaxi and Ollama should use provider-owned curated models.rs surfaces"
        );
    }

    for (relative, forbidden) in [
        (
            "../siumai-provider-minimaxi/src/providers/minimaxi/mod.rs",
            "pub mod model_constants",
        ),
        (
            "../siumai-provider-minimaxi/src/providers/minimaxi/models.rs",
            "model_constants",
        ),
        (
            "../siumai-provider-ollama/src/providers/ollama/mod.rs",
            "pub mod model_constants",
        ),
        (
            "../siumai-provider-ollama/src/providers/ollama/models.rs",
            "model_constants",
        ),
        (
            "../siumai/src/model_catalog.rs",
            "siumai_provider_minimaxi::providers::minimaxi::model_constants",
        ),
        (
            "../siumai/src/model_catalog.rs",
            "siumai_provider_ollama::providers::ollama::model_constants",
        ),
        (
            "src/native_provider_metadata.rs",
            "siumai_provider_ollama::providers::ollama::model_constants",
        ),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            !source.contains(forbidden),
            "{relative} should not depend on legacy MiniMaxi/Ollama model_constants surface `{forbidden}`"
        );
    }
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

fn provider_public_path_test_root() -> PathBuf {
    crate_root().join("../siumai/tests")
}

fn read_provider_public_path_module(module_name: &str) -> String {
    let path = provider_public_path_test_root()
        .join("provider_public_path_parity")
        .join(format!("{module_name}.rs"));
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn provider_public_path_module_manifest() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        (
            "openai_public_path",
            "openai",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "azure_public_path",
            "azure",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "gemini_public_path",
            "google",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "cohere_public_path",
            "cohere",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "togetherai_public_path",
            "togetherai",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "deepinfra_public_path",
            "deepinfra",
            "built_in_registry_builder(",
        ),
        (
            "vertex_maas_public_path",
            "google-vertex",
            ".with_provider_base_url_http_config_fetch(",
        ),
        (
            "google_vertex_xai_public_path",
            "google-vertex",
            ".with_provider_base_url_http_config_fetch(",
        ),
        (
            "deepseek_public_path",
            "deepseek",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "openai_compatible_audio_public_path",
            "openai",
            ".with_provider_api_key_fetch(",
        ),
        (
            "groq_public_path",
            "groq",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "ollama_public_path",
            "ollama",
            ".with_provider_base_url_fetch(",
        ),
        (
            "minimaxi_public_path",
            "minimaxi",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "bedrock_public_path",
            "bedrock",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "anthropic_public_path",
            "anthropic",
            ".with_provider_api_key_base_url_fetch(",
        ),
        (
            "vertex_public_path",
            "google-vertex",
            ".with_provider_base_url_http_config_fetch(",
        ),
        (
            "xai_public_path",
            "xai",
            ".with_provider_api_key_base_url_fetch(",
        ),
    ]
}

fn provider_public_path_combined_source() -> String {
    let mut source = fs::read_to_string(
        provider_public_path_test_root().join("provider_public_path_parity_test.rs"),
    )
    .expect("read provider public-path parity test root");
    for &(module_name, _, _) in provider_public_path_module_manifest() {
        source.push('\n');
        source.push_str(&read_provider_public_path_module(module_name));
    }
    source
}

#[test]
fn provider_public_path_parity_test_is_split_by_provider_module() {
    let root_source = fs::read_to_string(
        provider_public_path_test_root().join("provider_public_path_parity_test.rs"),
    )
    .expect("read provider public-path parity test root");

    for &(module_name, feature_name, _) in provider_public_path_module_manifest() {
        let path_marker = format!("#[path = \"provider_public_path_parity/{module_name}.rs\"]");
        let mod_marker = format!("mod {module_name};");
        let inline_marker = format!("mod {module_name} {{");
        let module_source = read_provider_public_path_module(module_name);

        assert!(
            root_source.contains(&format!("#[cfg(feature = \"{feature_name}\")]")),
            "provider_public_path_parity_test.rs should keep `{module_name}` feature-gated by `{feature_name}`"
        );
        assert!(
            root_source.contains(&path_marker) && root_source.contains(&mod_marker),
            "provider_public_path_parity_test.rs should load `{module_name}` from a provider-local module file"
        );
        assert!(
            !root_source.contains(&inline_marker),
            "provider_public_path_parity_test.rs should not keep oversized inline provider module `{module_name}`"
        );
        assert!(
            module_source.contains("use super::*;"),
            "{module_name}.rs should reuse the shared parity-test harness from the root module"
        );
    }
}

#[test]
fn migrated_public_path_modules_use_registry_builder_shortcuts() {
    for &(module_name, _, shortcut_marker) in provider_public_path_module_manifest() {
        let module_source = read_provider_public_path_module(module_name);
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
        assert!(
            !module_source.contains(".with_provider_build_overrides("),
            "{module_name} should use provider-level RegistryBuilder shortcuts instead of generic provider build overrides"
        );
    }
}
