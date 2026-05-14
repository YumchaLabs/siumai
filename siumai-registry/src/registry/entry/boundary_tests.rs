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

fn assert_no_compat_client_path(label: &str, source: &str) {
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
        assert_no_compat_client_path(label, &source);
    }

    let language_source = read_handle_source("language.rs");
    let chat_capability = source_section(
        &language_source,
        "language handle ChatCapability impl",
        "impl ChatCapability for LanguageModelHandle",
        "impl FileManagementCapability for LanguageModelHandle",
    );
    assert_no_compat_client_path("language handle ChatCapability impl", chat_capability);

    let image_source = read_handle_source("image.rs");
    let image_generation_capability = source_section(
        &image_source,
        "image handle ImageGenerationCapability impl",
        "impl ImageGenerationCapability for ImageModelHandle",
        "impl ImageExtras for ImageModelHandle",
    );
    assert_no_compat_client_path(
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
    assert_no_compat_client_path(
        "speech handle text_to_speech primary method",
        speech_primary,
    );

    let transcription_primary = source_section(
        &audio_source,
        "transcription handle speech_to_text primary method",
        "async fn speech_to_text(&self, request: SttRequest)",
        "async fn speech_to_text_stream(&self, request: SttRequest)",
    );
    assert_no_compat_client_path(
        "transcription handle speech_to_text primary method",
        transcription_primary,
    );
}
