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

fn read_handle_source(file_name: &str) -> String {
    let path = handle_source_path(file_name);
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
        assert!(
            !source.contains(forbidden),
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
        count_occurrences(image_extras, "compat_image_client_with_ctx"),
        2,
        "image compat client access must stay isolated to image extras edit/variation paths"
    );
    assert_eq!(
        count_occurrences(&image_source, "compat_image_client_with_ctx"),
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
        count_occurrences(speech_extension_helper, "compat_speech_client_with_ctx"),
        1,
        "speech compat client access must stay isolated to speech extras paths"
    );
    assert_eq!(
        count_occurrences(&audio_source, "compat_speech_client_with_ctx"),
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
            "compat_transcription_client_with_ctx",
        ),
        1,
        "transcription compat client access must stay isolated to transcription extras paths"
    );
    assert_eq!(
        count_occurrences(&audio_source, "compat_transcription_client_with_ctx"),
        1,
        "transcription handle must not grow new compat transcription client paths outside extras"
    );
}
