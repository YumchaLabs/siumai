use std::fs;
use std::path::{Path, PathBuf};

fn handle_source_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("registry")
        .join("entry")
        .join("handles")
        .join(file_name)
}

fn registry_entry_source_path(relative_path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("registry")
        .join("entry")
        .join(relative_path)
}

fn read_handle_source(file_name: &str) -> String {
    let path = handle_source_path(file_name);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn read_registry_entry_source(relative_path: &str) -> String {
    let path = registry_entry_source_path(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn source_section<'a>(source: &'a str, label: &str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing start marker `{start}` in {label}"));
    let section_tail = &source[start_index..];
    let end_index = section_tail
        .find(end)
        .unwrap_or_else(|| panic!("missing end marker `{end}` in {label}"));

    &section_tail[..end_index]
}

fn source_section_to_end<'a>(source: &'a str, label: &str, start: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing start marker `{start}` in {label}"));

    &source[start_index..]
}

fn assert_no_compat_or_downcast_path(label: &str, source: &str) {
    for forbidden in [
        "compat_language_client_with_ctx",
        "compat_completion_client_with_ctx",
        "compat_embedding_client_with_ctx",
        "compat_image_client_with_ctx",
        "compat_speech_client_with_ctx",
        "compat_transcription_client_with_ctx",
        "compat_video_client_with_ctx",
        "compat_reranking_client_with_ctx",
    ] {
        let facet_method = format!("build_{forbidden}");
        assert!(
            !source.contains(forbidden) || source.contains(&facet_method),
            "{label} must use native family factory paths for primary execution, not `{forbidden}`"
        );
    }

    let downcast_lines = source
        .lines()
        .filter(|line| line.contains(".as_") && line.contains("_capability("))
        .collect::<Vec<_>>();
    assert!(
        downcast_lines.is_empty(),
        "{label} must use native family model paths for primary execution, not LlmClient capability downcasts: {downcast_lines:?}"
    );
}

fn count_occurrences(source: &str, needle: &str) -> usize {
    source.match_indices(needle).count()
}

#[test]
fn provider_factory_generic_client_paths_are_explicit_compatibility_aliases() {
    let source = read_registry_entry_source("factory.rs");
    let trait_source = source_section_to_end(
        &source,
        "ProviderFactory trait",
        "pub trait ProviderFactory",
    );

    for method in [
        "language_client",
        "language_client_with_ctx",
        "completion_client",
        "completion_client_with_ctx",
        "embedding_client",
        "embedding_client_with_ctx",
        "image_client",
        "image_client_with_ctx",
        "speech_client",
        "speech_client_with_ctx",
        "transcription_client",
        "transcription_client_with_ctx",
        "video_client",
        "video_client_with_ctx",
        "reranking_client",
        "reranking_client_with_ctx",
    ] {
        let forbidden = format!("async fn {method}(");
        assert!(
            !trait_source.contains(&forbidden),
            "ProviderFactory must not expose unprefixed generic LlmClient path `{method}`; use a `compat_*` alias or a native family method"
        );
    }

    for method in [
        "compat_language_client",
        "compat_language_client_with_ctx",
        "compat_completion_client",
        "compat_completion_client_with_ctx",
        "compat_embedding_client",
        "compat_embedding_client_with_ctx",
        "compat_image_client",
        "compat_image_client_with_ctx",
        "compat_speech_client",
        "compat_speech_client_with_ctx",
        "compat_transcription_client",
        "compat_transcription_client_with_ctx",
        "compat_video_client",
        "compat_video_client_with_ctx",
        "compat_reranking_client",
        "compat_reranking_client_with_ctx",
    ] {
        let expected = format!("async fn {method}(");
        assert!(
            trait_source.contains(&expected),
            "ProviderFactory compatibility path `{method}` should remain explicitly named while FCAB splits native family construction from legacy LlmClient construction"
        );
    }

    assert!(
        trait_source
            .contains("New registry execution should prefer `language_model_text_with_ctx(...)`."),
        "ProviderFactory docs should teach new registry execution to prefer native family model paths"
    );
    assert!(
        count_occurrences(
            trait_source,
            "Compatibility alias for creating a generic `LlmClient`",
        ) >= 8,
        "generic LlmClient methods should be documented as compatibility aliases"
    );
}

#[test]
fn provider_factory_facets_split_stable_compat_and_extension_execution() {
    let source = read_registry_entry_source("factory.rs");
    let family_trait = source_section(
        &source,
        "ProviderFamilyFactory trait",
        "pub trait ProviderFamilyFactory",
        "/// Compatibility factory facet",
    );
    let compat_trait = source_section(
        &source,
        "ProviderCompatibilityFactory trait",
        "pub trait ProviderCompatibilityFactory",
        "/// Extension-capability facet",
    );
    let extension_trait = source_section(
        &source,
        "ProviderExtensionFactory trait",
        "pub trait ProviderExtensionFactory",
        "/// Provider factory trait",
    );

    assert!(
        !family_trait.contains("LlmClient") && !family_trait.contains("compat_"),
        "ProviderFamilyFactory must stay free of legacy generic-client construction"
    );
    assert!(
        compat_trait.contains("LlmClient")
            && compat_trait.contains("build_compat_language_client_with_ctx"),
        "ProviderCompatibilityFactory should own the legacy generic-client build surface"
    );
    assert!(
        !extension_trait.contains("LlmClient")
            && extension_trait.contains("build_file_management_capability_with_ctx"),
        "ProviderExtensionFactory should expose extension capability objects without widening the family facet"
    );

    for required in [
        "build_language_model_text_with_ctx",
        "build_completion_model_family_with_ctx",
        "build_embedding_model_family_with_ctx",
        "build_image_model_family_with_ctx",
        "build_speech_model_family_with_ctx",
        "build_transcription_model_family_with_ctx",
        "build_video_model_family_with_ctx",
        "build_reranking_model_family_with_ctx",
    ] {
        assert!(
            family_trait.contains(required),
            "ProviderFamilyFactory should expose stable family build method `{required}`"
        );
    }
}

#[test]
fn stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution() {
    for (label, file_name) in [
        ("completion handle", "completion.rs"),
        ("embedding handle", "embedding.rs"),
        ("reranking handle", "rerank.rs"),
        ("video handle", "video.rs"),
    ] {
        let source = read_handle_source(file_name);
        assert_no_compat_or_downcast_path(label, &source);
    }

    let language_source = read_handle_source("language.rs");
    let chat_capability = source_section(
        &language_source,
        "language handle ChatCapability impl",
        "impl ChatCapability for LanguageModelHandle",
        "impl FileManagementCapability for LanguageModelHandle",
    );
    assert_no_compat_or_downcast_path("language handle ChatCapability impl", chat_capability);

    for (label, start, end) in [
        (
            "language handle FileManagementCapability impl",
            "impl FileManagementCapability for LanguageModelHandle",
            "impl SkillsCapability for LanguageModelHandle",
        ),
        (
            "language handle SkillsCapability impl",
            "impl SkillsCapability for LanguageModelHandle",
            "impl VideoGenerationCapability for LanguageModelHandle",
        ),
    ] {
        let section = source_section(&language_source, label, start, end);
        assert_no_compat_or_downcast_path(label, section);
    }

    let music_extension = source_section_to_end(
        &language_source,
        "language handle MusicGenerationCapability impl",
        "impl MusicGenerationCapability for LanguageModelHandle",
    );
    assert_no_compat_or_downcast_path(
        "language handle MusicGenerationCapability impl",
        music_extension,
    );

    let image_source = read_handle_source("image.rs");
    let image_generation_capability = source_section(
        &image_source,
        "image handle ImageGenerationCapability impl",
        "impl ImageGenerationCapability for ImageModelHandle",
        "impl ImageExtras for ImageModelHandle",
    );
    assert_no_compat_or_downcast_path(
        "image handle ImageGenerationCapability impl",
        image_generation_capability,
    );

    let audio_source = read_handle_source("audio.rs");
    let speech_primary = source_section(
        &audio_source,
        "speech handle text_to_speech primary method",
        "async fn text_to_speech(&self, request: TtsRequest)",
        "async fn text_to_speech_stream(&self, request: TtsRequest)",
    );
    assert_no_compat_or_downcast_path(
        "speech handle text_to_speech primary method",
        speech_primary,
    );

    let transcription_primary = source_section(
        &audio_source,
        "transcription handle speech_to_text primary method",
        "async fn speech_to_text(&self, request: SttRequest)",
        "async fn speech_to_text_stream(&self, request: SttRequest)",
    );
    assert_no_compat_or_downcast_path(
        "transcription handle speech_to_text primary method",
        transcription_primary,
    );
}

#[test]
fn remaining_registry_handle_compat_paths_are_extension_only() {
    for (label, file_name) in [
        ("completion handle", "completion.rs"),
        ("embedding handle", "embedding.rs"),
        ("reranking handle", "rerank.rs"),
        ("video handle", "video.rs"),
        ("language handle", "language.rs"),
    ] {
        let source = read_handle_source(file_name);
        assert_no_compat_or_downcast_path(label, &source);
    }

    let image_source = read_handle_source("image.rs");
    let image_generation_capability = source_section(
        &image_source,
        "image handle ImageGenerationCapability impl",
        "impl ImageGenerationCapability for ImageModelHandle",
        "impl ImageExtras for ImageModelHandle",
    );
    assert_no_compat_or_downcast_path(
        "image handle ImageGenerationCapability impl",
        image_generation_capability,
    );

    let image_extras = source_section(
        &image_source,
        "image handle ImageExtras impl",
        "impl ImageExtras for ImageModelHandle",
        "impl crate::traits::ModelMetadata for ImageModelHandle",
    );
    assert_eq!(
        count_occurrences(image_extras, "build_compat_image_client_with_ctx"),
        2,
        "image compat client access must stay isolated to image extras edit/variation paths"
    );
    assert_eq!(
        count_occurrences(&image_source, "build_compat_image_client_with_ctx"),
        2,
        "image handle must not grow new compat image client paths outside image extras"
    );

    let audio_source = read_handle_source("audio.rs");
    let speech_primary = source_section(
        &audio_source,
        "speech handle text_to_speech primary method",
        "async fn text_to_speech(&self, request: TtsRequest)",
        "async fn text_to_speech_stream(&self, request: TtsRequest)",
    );
    assert_no_compat_or_downcast_path(
        "speech handle text_to_speech primary method",
        speech_primary,
    );
    let transcription_primary = source_section(
        &audio_source,
        "transcription handle speech_to_text primary method",
        "async fn speech_to_text(&self, request: SttRequest)",
        "async fn speech_to_text_stream(&self, request: SttRequest)",
    );
    assert_no_compat_or_downcast_path(
        "transcription handle speech_to_text primary method",
        transcription_primary,
    );

    let speech_extension_helper = source_section(
        &audio_source,
        "speech handle extension helper",
        "async fn build_speech_client",
        "async fn get_or_create_speech_model",
    );
    assert_eq!(
        count_occurrences(
            speech_extension_helper,
            "build_compat_speech_client_with_ctx"
        ),
        1,
        "speech compat client access must stay isolated to speech extras paths"
    );
    assert_eq!(
        count_occurrences(&audio_source, "build_compat_speech_client_with_ctx"),
        1,
        "speech handle must not grow new compat speech client paths outside extras"
    );

    let transcription_extension_helper = source_section(
        &audio_source,
        "transcription handle extension helper",
        "async fn build_transcription_client",
        "async fn get_or_create_transcription_model",
    );
    assert_eq!(
        count_occurrences(
            transcription_extension_helper,
            "build_compat_transcription_client_with_ctx",
        ),
        1,
        "transcription compat client access must stay isolated to transcription extras paths"
    );
    assert_eq!(
        count_occurrences(&audio_source, "build_compat_transcription_client_with_ctx"),
        1,
        "transcription handle must not grow new compat transcription client paths outside extras"
    );
}
