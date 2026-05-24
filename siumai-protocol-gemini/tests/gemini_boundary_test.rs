use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("protocol crate should live under workspace root")
        .to_path_buf()
}

#[test]
fn gemini_typed_provider_metadata_is_protocol_owned() {
    let workspace = workspace_root();
    let protocol_metadata = fs::read_to_string(
        workspace.join("siumai-protocol-gemini/src/provider_metadata/gemini.rs"),
    )
    .expect("read protocol Gemini provider metadata source");
    let protocol_mod =
        fs::read_to_string(workspace.join("siumai-protocol-gemini/src/provider_metadata/mod.rs"))
            .expect("read protocol Gemini provider metadata mod");
    let protocol_lib = fs::read_to_string(workspace.join("siumai-protocol-gemini/src/lib.rs"))
        .expect("read protocol Gemini lib");
    let provider_metadata = fs::read_to_string(
        workspace.join("siumai-provider-gemini/src/provider_metadata/gemini.rs"),
    )
    .expect("read provider Gemini provider metadata source");

    for marker in [
        "pub struct GeminiMetadata",
        "pub type GoogleProviderMetadata",
        "pub struct GoogleInteractionsProviderMetadata",
        "pub trait GeminiChatResponseExt",
        "pub trait GeminiContentPartExt",
    ] {
        assert!(
            protocol_metadata.contains(marker),
            "Gemini typed provider metadata should live in the protocol crate: missing {marker}"
        );
    }

    assert!(
        protocol_mod.contains("pub mod gemini;")
            && protocol_lib.contains("pub mod provider_metadata;"),
        "siumai-protocol-gemini should expose the protocol-owned provider metadata module"
    );
    assert!(
        provider_metadata.contains("pub use siumai_protocol_gemini::provider_metadata::gemini::*;")
            && provider_metadata.lines().count() <= 3,
        "siumai-provider-gemini provider_metadata::gemini should stay a thin protocol re-export"
    );
    for forbidden in [
        "pub struct GeminiMetadata",
        "pub struct GoogleInteractionsProviderMetadata",
        "impl GeminiChatResponseExt",
        "impl GeminiContentPartExt",
    ] {
        assert!(
            !provider_metadata.contains(forbidden),
            "provider crate must not own Gemini typed provider metadata implementation: found {forbidden}"
        );
    }
}
