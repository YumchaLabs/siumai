use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn openai_protocol_is_not_imported_from_core() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let forbidden = "siumai_core::standards::openai";

    let mut offenders = Vec::new();
    for root in [
        "siumai/src",
        "siumai/tests",
        "siumai-registry/src",
        "siumai-provider-openai-compatible/src",
        "siumai-provider-openai/src",
        "siumai-provider-groq/src",
        "siumai-provider-deepseek/src",
        "siumai-protocol-openai/src",
        "siumai-extras/src",
        "siumai-extras/tests",
    ] {
        collect_forbidden_imports(&workspace.join(root), forbidden, &mut offenders);
    }

    assert!(
        offenders.is_empty(),
        "OpenAI protocol imports must go through siumai-protocol-openai:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn core_no_longer_owns_openai_protocol_files() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let core_openai_dir = workspace.join("siumai-core/src/standards/openai");
    let mut remaining_files = Vec::new();
    collect_rs_files(&core_openai_dir, &mut remaining_files);

    assert!(
        remaining_files.is_empty(),
        "siumai-core must not own OpenAI protocol files:\n{}",
        remaining_files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn provider_and_protocol_crates_do_not_publicly_mirror_core() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let mut offenders = Vec::new();

    for crate_name in [
        "siumai-protocol-openai",
        "siumai-protocol-anthropic",
        "siumai-protocol-gemini",
        "siumai-provider-openai",
        "siumai-provider-anthropic",
        "siumai-provider-gemini",
        "siumai-provider-azure",
        "siumai-provider-google-vertex",
        "siumai-provider-groq",
        "siumai-provider-xai",
        "siumai-provider-deepseek",
        "siumai-provider-minimaxi",
        "siumai-provider-ollama",
        "siumai-provider-cohere",
        "siumai-provider-togetherai",
        "siumai-provider-amazon-bedrock",
    ] {
        let lib_rs = workspace.join(crate_name).join("src/lib.rs");
        collect_forbidden_imports(&lib_rs, "pub use siumai_core::{", &mut offenders);
        collect_forbidden_imports(&lib_rs, "pub use siumai_core::builder::*;", &mut offenders);
    }

    assert!(
        offenders.is_empty(),
        "provider/protocol crates must not publicly mirror siumai-core modules:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn responses_feature_surface_uses_stable_parts_for_tool_stream_parts() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/responses_sse_feature_surface_test.rs"),
    )
    .expect("responses SSE feature surface test source");

    for forbidden in [
        r#"event_type: "openai:tool-call""#,
        r#"event_type: "openai:tool-result""#,
        r#""openai:tool-call""#,
        r#""openai:tool-result""#,
    ] {
        assert!(
            !source.contains(forbidden),
            "OpenAI Responses public feature surface tests must use stable ChatStreamEvent::Part \
             tool stream parts instead of provider custom events: found {forbidden}"
        );
    }

    for required in [
        "ChatStreamEvent::Part",
        "ChatStreamPart::ToolCall",
        "ChatStreamPart::ToolResult",
    ] {
        assert!(
            source.contains(required),
            "OpenAI Responses public feature surface tests should exercise stable stream parts: \
             missing {required}"
        );
    }
}

#[test]
fn openai_compatible_completion_streaming_conversion_is_protocol_owned() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let protocol_source = fs::read_to_string(
        workspace.join("siumai-protocol-openai/src/standards/openai/compat/completion.rs"),
    )
    .expect("protocol completion conversion source");
    let provider_completion_source = fs::read_to_string(workspace.join(
        "siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/mod.rs",
    ))
    .expect("OpenAI-compatible provider completion runtime source");
    let native_openai_source = fs::read_to_string(
        workspace.join("siumai-provider-openai/src/providers/openai/client/completion.rs"),
    )
    .expect("native OpenAI completion runtime source");
    let legacy_provider_streaming = workspace.join(
        "siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/streaming.rs",
    );

    for marker in [
        "pub struct CompletionSseConverter",
        "struct CompletionStreamState",
        "impl crate::streaming::SseEventConverter for CompletionSseConverter",
        "pub struct CompletionResponseConversion",
        "OpenAiCompatibleUsagePolicy::for_provider",
        "parse_provider_openai_finish_reason",
    ] {
        assert!(
            protocol_source.contains(marker),
            "OpenAI-compatible completion stream conversion should live in the protocol module: missing {marker}"
        );
    }

    assert!(
        provider_completion_source
            .contains("siumai_protocol_openai::standards::openai::compat::completion")
            && provider_completion_source.contains("CompletionSseConverter"),
        "OpenAI-compatible provider runtime should import the protocol-owned completion stream converter"
    );
    assert!(
        provider_completion_source.contains("CompletionResponseConversion::new")
            && provider_completion_source.contains(".build_response("),
        "OpenAI-compatible provider runtime should delegate response conversion to the protocol-owned completion module"
    );

    for forbidden in [
        "mod streaming;",
        "use streaming::CompletionSseConverter;",
        "struct CompletionStreamState",
        "struct CompletionSseConverter",
        "impl crate::streaming::SseEventConverter for CompletionSseConverter",
        "parse_provider_openai_finish_reason",
        "OpenAiCompatibleUsagePolicy::for_provider",
        "pub fn build_completion_response(",
    ] {
        assert!(
            !provider_completion_source.contains(forbidden),
            "OpenAI-compatible provider runtime must not mirror protocol conversion logic: found {forbidden}"
        );
    }

    assert!(
        !legacy_provider_streaming.exists(),
        "OpenAI-compatible provider must not keep a local completion/streaming.rs parser copy"
    );

    assert!(
        native_openai_source.contains("struct CompletionSseConverter"),
        "native OpenAI keeps its provider-specific completion converter until a native OpenAI seam task moves it"
    );
    assert!(
        !native_openai_source.contains("OpenAiCompatibleUsagePolicy::for_provider"),
        "native OpenAI completion runtime must not depend on OpenAI-compatible usage policy"
    );
}

#[test]
fn openai_audio_sse_wire_format_helpers_are_protocol_owned() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let protocol_audio_source =
        fs::read_to_string(workspace.join("siumai-protocol-openai/src/standards/openai/audio.rs"))
            .expect("protocol OpenAI audio source");
    let provider_sse_helpers_source = fs::read_to_string(
        workspace.join("siumai-provider-openai/src/providers/openai/client/sse_helpers.rs"),
    )
    .expect("provider OpenAI SSE helper source");
    let provider_transcription_source = fs::read_to_string(
        workspace
            .join("siumai-provider-openai/src/providers/openai/client/transcription_streaming.rs"),
    )
    .expect("provider OpenAI transcription streaming source");

    for marker in [
        "pub enum OpenAiTranscriptionStreamEvent",
        "pub type OpenAiTranscriptionStream",
        "pub fn openai_speech_audio_delta",
        "pub fn openai_speech_audio_done",
        "pub fn openai_transcript_text_delta",
        "pub fn openai_transcript_text_segment",
        "pub fn openai_transcript_text_done",
        "pub fn ensure_openai_sse_content_type",
    ] {
        assert!(
            protocol_audio_source.contains(marker),
            "OpenAI audio SSE wire-format helper should live in protocol audio module: missing {marker}"
        );
    }

    assert!(
        provider_sse_helpers_source.contains("pub(crate) use crate::standards::openai::audio::{")
            && provider_sse_helpers_source.lines().count() <= 12,
        "provider OpenAI SSE helper module should stay a thin protocol-owned helper re-export"
    );

    for forbidden in [
        "use base64::Engine",
        "Missing 'audio' field",
        "Missing 'delta' field",
        "Missing 'id' field",
        "Expected 'text/event-stream'",
        "pub enum OpenAiTranscriptionStreamEvent",
    ] {
        assert!(
            !provider_sse_helpers_source.contains(forbidden)
                && !provider_transcription_source.contains(forbidden),
            "provider OpenAI streaming modules must not own protocol SSE wire-format parsing: found {forbidden}"
        );
    }
}

fn collect_forbidden_imports(root: &Path, forbidden: &str, offenders: &mut Vec<String>) {
    let mut files = Vec::new();
    collect_rs_files(root, &mut files);

    for file in files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in content.lines().enumerate() {
            if line.contains(forbidden) {
                offenders.push(format!("{}:{}", file.display(), index + 1));
            }
        }
    }
}

fn collect_rs_files(root: &Path, files: &mut Vec<PathBuf>) {
    if root.is_file() {
        if root.extension().is_some_and(|ext| ext == "rs") {
            files.push(root.to_path_buf());
        }
        return;
    }

    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}
