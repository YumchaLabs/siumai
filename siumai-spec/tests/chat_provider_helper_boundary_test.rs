use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn workspace_root() -> PathBuf {
    crate_root()
        .parent()
        .expect("siumai-spec should live under workspace root")
        .to_path_buf()
}

fn chat_message_source() -> String {
    let path = crate_root().join("src/types/chat/message.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn workspace_rust_sources_under(workspace_root: &Path, relative_path: &str) -> Vec<PathBuf> {
    let mut pending = vec![workspace_root.join(relative_path)];
    let mut sources = Vec::new();

    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        {
            let entry = entry.expect("source directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }

    sources
}

fn normalized_workspace_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .expect("source path under workspace root")
        .iter()
        .map(|component| component.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[test]
fn chat_message_builder_does_not_expose_provider_specific_helpers() {
    let source = chat_message_source();

    for removed_helper in [
        "pub fn cache_control(",
        "pub fn cache_control_for_part(",
        "pub fn cache_control_for_parts",
        "pub fn anthropic_document_citations_for_part(",
        "pub fn anthropic_document_metadata_for_part(",
    ] {
        assert!(
            !source.contains(removed_helper),
            "siumai-spec::ChatMessageBuilder must not reintroduce removed Anthropic request helper `{removed_helper}`; use provider-owned AnthropicChatMessageExt instead"
        );
    }

    for forbidden_helper_prefix in [
        "pub fn openai_",
        "pub fn gemini_",
        "pub fn google_",
        "pub fn azure_",
        "pub fn bedrock_",
        "pub fn groq_",
        "pub fn xai_",
        "pub fn minimaxi_",
        "pub fn cohere_",
        "pub fn togetherai_",
        "pub fn ollama_",
        "pub fn deepseek_",
        "pub fn anthropic_",
    ] {
        assert!(
            !source.contains(forbidden_helper_prefix),
            "siumai-spec::ChatMessageBuilder must not grow new provider-specific request helper `{forbidden_helper_prefix}`; put new helpers in provider extension crates"
        );
    }
}

#[test]
fn chat_message_builder_source_stays_provider_agnostic() {
    let source = chat_message_source();
    let builder_start = source
        .find("impl ChatMessageBuilder {")
        .expect("ChatMessageBuilder impl start");
    let image_helper_start = source[builder_start..]
        .find("    /// Adds image content")
        .map(|offset| builder_start + offset)
        .expect("generic content helper section start");
    let builder_setup_source = &source[builder_start..image_helper_start];

    for forbidden in [
        "\"anthropic\"",
        "\"cacheControl\"",
        "\"citations\"",
        "\"title\"",
        "\"context\"",
        "metadata.custom.insert",
        "metadata.custom.get",
        "ChatResponse",
        "siumai_core::",
        "siumai_provider_",
        "siumai_protocol_",
        "tokio",
        "reqwest",
        "async fn",
        ".await",
    ] {
        assert!(
            !builder_setup_source.contains(forbidden),
            "siumai-spec::ChatMessageBuilder setup helpers must stay provider-agnostic and must not reintroduce `{forbidden}`"
        );
    }
}

#[test]
fn chat_message_production_source_does_not_embed_concrete_provider_names() {
    let source = chat_message_source();
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production ChatMessage source");

    for concrete_provider in [
        "\"anthropic\"",
        "\"openai\"",
        "\"gemini\"",
        "\"google\"",
        "\"azure\"",
        "\"bedrock\"",
        "\"groq\"",
        "\"xai\"",
        "\"minimaxi\"",
        "\"cohere\"",
        "\"togetherai\"",
        "\"ollama\"",
        "\"deepseek\"",
    ] {
        assert!(
            !production_source.contains(concrete_provider),
            "siumai-spec::ChatMessage production code must stay provider-agnostic; put `{concrete_provider}` helpers in provider extension crates"
        );
    }
}

#[test]
fn low_level_crates_do_not_consume_anthropic_chat_builder_compat_helpers() {
    let workspace_root = workspace_root();
    let audited_dirs = [
        "siumai-bridge/src",
        "siumai-protocol-anthropic/src",
        "siumai/src",
        "siumai/examples",
    ];
    let forbidden_calls = [
        ".cache_control(",
        ".cache_control_for_part(",
        ".cache_control_for_parts(",
        ".anthropic_document_citations_for_part(",
        ".anthropic_document_metadata_for_part(",
    ];

    let mut unexpected_hits = Vec::new();
    for audited_dir in audited_dirs {
        for source_path in workspace_rust_sources_under(&workspace_root, audited_dir) {
            let source = fs::read_to_string(&source_path).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", source_path.display())
            });
            for forbidden_call in forbidden_calls {
                if source.contains(forbidden_call) {
                    unexpected_hits.push(format!(
                        "{} contains `{}`",
                        normalized_workspace_path(&workspace_root, &source_path),
                        forbidden_call
                    ));
                }
            }
        }
    }

    assert!(
        unexpected_hits.is_empty(),
        "low-level bridge/protocol/facade/example code should use provider-owned Anthropic message extensions or explicit request-side provider options instead of consuming siumai-spec ChatMessageBuilder compatibility helpers: {unexpected_hits:?}"
    );
}
