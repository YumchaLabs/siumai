use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("protocol crate should live under workspace root")
        .to_path_buf()
}

#[test]
fn anthropic_typed_provider_metadata_is_protocol_owned() {
    let workspace = workspace_root();
    let protocol_metadata = fs::read_to_string(
        workspace.join("siumai-protocol-anthropic/src/provider_metadata/anthropic.rs"),
    )
    .expect("read protocol Anthropic provider metadata source");
    let protocol_mod = fs::read_to_string(
        workspace.join("siumai-protocol-anthropic/src/provider_metadata/mod.rs"),
    )
    .expect("read protocol Anthropic provider metadata mod");
    let protocol_lib = fs::read_to_string(workspace.join("siumai-protocol-anthropic/src/lib.rs"))
        .expect("read protocol Anthropic lib");
    let provider_metadata = fs::read_to_string(
        workspace.join("siumai-provider-anthropic/src/provider_metadata/mod.rs"),
    )
    .expect("read provider Anthropic provider metadata module");

    for marker in [
        "pub struct AnthropicMetadata",
        "pub struct AnthropicMessageMetadata",
        "pub struct AnthropicContainerMetadata",
        "pub struct AnthropicMessageContainerMetadata",
        "pub trait AnthropicChatResponseExt",
        "pub trait AnthropicContentPartExt",
    ] {
        assert!(
            protocol_metadata.contains(marker),
            "Anthropic typed provider metadata should live in the protocol crate: missing {marker}"
        );
    }

    assert!(
        protocol_mod.contains("pub mod anthropic;")
            && protocol_lib.contains("pub mod provider_metadata;"),
        "siumai-protocol-anthropic should expose the protocol-owned provider metadata module"
    );
    assert!(
        provider_metadata
            .contains("pub use siumai_protocol_anthropic::provider_metadata::anthropic::*;")
            && provider_metadata.lines().count() <= 8,
        "siumai-provider-anthropic provider_metadata should stay a thin protocol re-export"
    );

    for forbidden in [
        "pub struct AnthropicMetadata",
        "pub struct AnthropicMessageMetadata",
        "impl AnthropicChatResponseExt",
        "impl AnthropicContentPartExt",
        "provider-owned",
    ] {
        assert!(
            !provider_metadata.contains(forbidden),
            "provider crate must not own Anthropic typed provider metadata implementation: found {forbidden}"
        );
    }
}
