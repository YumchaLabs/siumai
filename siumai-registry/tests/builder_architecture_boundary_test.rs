use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn siumai_builder_source() -> String {
    let path = crate_root().join("src/provider/siumai_builder.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

#[test]
fn builder_does_not_expose_noop_capability_flags() {
    let source = siumai_builder_source();

    for forbidden in [
        "capabilities: Vec<String>",
        "capabilities: Vec::new()",
        "pub fn with_capability<",
        "pub fn with_audio(self) -> Self",
        "pub fn with_embedding(self) -> Self",
        "pub fn with_image_generation(self) -> Self",
        "capabilities_count",
    ] {
        assert!(
            !source.contains(forbidden),
            "SiumaiBuilder should not expose write-only capability flag surface `{forbidden}`"
        );
    }
}
