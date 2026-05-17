use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read directory") {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

#[test]
fn runtime_bridge_code_imports_dedicated_bridge_crate() {
    let root = crate_root();
    let cargo_toml = fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    assert!(
        cargo_toml.contains("siumai-bridge"),
        "siumai-extras should depend on siumai-bridge directly"
    );

    let mut source_files = Vec::new();
    collect_rs_files(&root.join("src"), &mut source_files);

    let offenders: Vec<_> = source_files
        .iter()
        .filter_map(|path| {
            let source = fs::read_to_string(path).expect("read source file");
            source.contains("siumai::experimental::bridge").then(|| {
                path.strip_prefix(&root)
                    .unwrap_or(path)
                    .display()
                    .to_string()
            })
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "siumai-extras runtime code should import bridge APIs from siumai_bridge directly: {offenders:?}"
    );
}

#[test]
fn openai_responses_gateway_parts_adapter_uses_bridge_stream_seam_directly() {
    let root = crate_root();
    let source_path = root.join("src/server/axum/transcode_sse.rs");
    let source = fs::read_to_string(&source_path).expect("read transcode_sse.rs");

    assert!(
        source.contains("siumai_bridge::stream::OpenAiResponsesStreamPartsBridge"),
        "gateway OpenAI Responses stream adapter should use the dedicated bridge crate seam"
    );
    assert!(
        !source.contains("siumai::experimental::streaming::OpenAiResponsesStreamPartsBridge")
            && !source.contains(
                "siumai::experimental::streaming::{\n        OpenAiResponsesStreamPartsBridge"
            ),
        "gateway runtime code should not import bridge-owned adapters through the facade experimental streaming path"
    );
}

#[test]
fn runtime_bridge_code_does_not_use_removed_siumai_types_root() {
    let root = crate_root();
    let mut source_files = Vec::new();
    collect_rs_files(&root.join("src"), &mut source_files);

    let offenders: Vec<_> = source_files
        .iter()
        .filter_map(|path| {
            let source = fs::read_to_string(path).expect("read source file");
            source.contains("siumai::types::").then(|| {
                path.strip_prefix(&root)
                    .unwrap_or(path)
                    .display()
                    .to_string()
            })
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "siumai-extras runtime code should not import the removed `siumai::types::*` root path: {offenders:?}"
    );
}
